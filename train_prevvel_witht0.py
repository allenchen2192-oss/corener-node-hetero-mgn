# 與 train_prevvel.py 相同，但訓練資料包含 t=0（prev_vel=0, disp=0）
# 對應 conf/config_prevvel_no_nodetype_witht0.yaml

import math
import os
import random
import time

import hydra
from hydra.utils import to_absolute_path
import torch
from tqdm import tqdm

from omegaconf import DictConfig

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.utils.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch


def my_collate_fn(batch):
    return batch


def collate_fn_single_step(batch):
    graphs    = [seq[0]["graph"]              for seq in batch]
    mesh_efs  = [seq[0]["mesh_edge_features"] for seq in batch]
    world_efs = [seq[0]["world_edge_features"] for seq in batch]
    return (
        Batch.from_data_list(graphs),
        torch.cat(mesh_efs,  dim=0),
        torch.cat(world_efs, dim=0),
    )


def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    disp = pos[col] - pos[row]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryTimeStepDataset(torch.utils.data.Dataset):
    """In-memory dataset — includes t=0 (prev_vel=0, disp=0) in training."""

    def __init__(self, sample_dir, num_samples=None, max_nodes=None):
        self.data = []
        sample_files = sorted(
            [os.path.join(sample_dir, f)
             for f in os.listdir(sample_dir)
             if f.startswith("sample_") and f.endswith(".pt")]
        )
        print(f"Loading sample files into memory (max_nodes={max_nodes}, num_samples={num_samples}) ...")
        loaded = skipped = 0
        for f in sample_files:
            if num_samples is not None and loaded >= num_samples:
                break
            sample_data = torch.load(f, map_location="cpu", weights_only=False)
            if max_nodes is not None and sample_data[0]["graph"].num_nodes > max_nodes:
                skipped += 1
                continue
            # Include ALL timesteps — t=0 (prev_vel=0, disp=0) is included
            self.data.extend(sample_data)
            loaded += 1
        print(f"Loaded {loaded} samples / {len(self.data)} timesteps ({skipped} skipped, nodes>{max_nodes})")

    def __getitem__(self, idx):
        return [self.data[idx]]

    def __len__(self):
        return len(self.data)


class InMemorySequenceDataset(torch.utils.data.Dataset):
    """In-memory sequence dataset for rollout training."""

    def __init__(self, sample_dir, num_time_steps, seq_len, num_samples=None):
        sample_files = sorted(
            [os.path.join(sample_dir, f)
             for f in os.listdir(sample_dir)
             if f.startswith("sample_") and f.endswith(".pt")]
        )
        if num_samples is not None:
            sample_files = sample_files[:num_samples]

        num_steps    = num_time_steps - 1
        seq_len      = min(seq_len, num_steps)
        valid_starts = num_steps - seq_len + 1

        print(f"Loading {len(sample_files)} sample files (seq_len={seq_len}) ...")
        all_data = [torch.load(f, map_location="cpu", weights_only=False)
                    for f in sample_files]

        self.sequences = [(si, start)
                          for si in range(len(all_data))
                          for start in range(valid_starts)]
        self._data    = all_data
        self._seq_len = seq_len
        print(f"  {len(self.sequences)} sequences ready.")

    def __getitem__(self, idx):
        si, start = self.sequences[idx]
        return [self._data[si][start + k] for k in range(self._seq_len)]

    def __len__(self):
        return len(self.sequences)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp  = cfg.amp

        mlp_act = "silu" if cfg.recompute_activation else "relu"

        seq_len    = cfg.rollout_train_steps
        sample_dir = to_absolute_path(cfg.preprocess_output_dir)
        if seq_len > 1:
            dataset = InMemorySequenceDataset(
                sample_dir, cfg.num_training_time_steps, seq_len,
                num_samples=cfg.num_training_samples,
            )
        else:
            dataset = InMemoryTimeStepDataset(
                sample_dir,
                num_samples=cfg.num_training_samples,
                max_nodes=getattr(cfg, "max_nodes", None),
            )

        sampler = DistributedSampler(
            dataset, shuffle=True, drop_last=True,
            num_replicas=self.dist.world_size, rank=self.dist.rank,
        )
        n_workers = cfg.num_dataloader_workers
        collate   = collate_fn_single_step if seq_len == 1 else my_collate_fn
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=sampler,
            pin_memory=False, num_workers=n_workers,
            collate_fn=collate,
            persistent_workers=(n_workers > 0),
        )
        self.sampler = sampler

        self.model = HybridMeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            processor_size=cfg.processor_size,
            hidden_dim_processor=cfg.hidden_dim,
            hidden_dim_node_encoder=cfg.hidden_dim,
            hidden_dim_edge_encoder=cfg.hidden_dim,
            hidden_dim_node_decoder=cfg.hidden_dim,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        self.model = self.model.to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()

        self.criterion      = torch.nn.MSELoss()
        self.lambda_penalty = cfg.lambda_penalty
        self.w_vel          = cfg.w_vel
        self.w_stress       = cfg.w_stress
        self.use_node_type  = getattr(cfg, "use_node_type", True)
        self._train_step    = 0
        self._stress_every  = getattr(cfg, "stress_every", 5)

        self.optimizer = None
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            rank_zero_logger.warning("NVIDIA Apex not installed; falling back to Adam.")
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, foreach=True)
        rank_zero_logger.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

        if self.dist.rank == 0:
            self.writer = SummaryWriter(
                log_dir=to_absolute_path(cfg.tensorboard_log_dir)
            )

        self.rollout_train_steps = cfg.rollout_train_steps
        self.rollout_start_epoch = getattr(cfg, "rollout_start_epoch", 9999)
        self.rollout_prob        = getattr(cfg, "rollout_prob", 0.5)
        self.input_noise_std     = getattr(cfg, "input_noise_std", 0.0)

        self._load_statics(cfg)

    def _load_statics(self, cfg):
        data_dir = to_absolute_path(cfg.preprocess_output_dir)
        edge_stats = load_json(os.path.join(data_dir, "edge_stats.json"))
        self.e_mean_ref = edge_stats["edge_mean"].to(self.dist.device)[:4]
        self.e_std_ref  = edge_stats["edge_std"].to(self.dist.device)[:4]
        self.e_mean_wld = edge_stats["edge_mean"].to(self.dist.device)[4:]
        self.e_std_wld  = edge_stats["edge_std"].to(self.dist.device)[4:]

        node_stats  = load_json(os.path.join(data_dir, "node_stats.json"))
        self.v_mean = node_stats["velocity_mean"].to(self.dist.device)
        self.v_std  = node_stats["velocity_std"].to(self.dist.device)
        self.s_mean = node_stats["stress_mean"].to(self.dist.device)
        self.s_std  = node_stats["stress_std"].to(self.dist.device)
        self.d_mean = node_stats["disp_mean"].to(self.dist.device)
        self.d_std  = node_stats["disp_std"].to(self.dist.device)

        self.empty_world_ef = torch.zeros(
            0, cfg.num_edge_features, device=self.dist.device
        )

    def _vm_stress_raw(self, wp, ref, edge_index):
        M        = wp.shape[0]
        row, col = edge_index
        r_ij  = ref[col].float() - ref[row].float()
        u     = wp.float() - ref.float()
        du_ij = u[col] - u[row]
        w_ij  = 1.0 / (r_ij.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12))
        r_r   = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
        r_du  = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
        row_exp = row.view(-1, 1, 1).expand_as(r_r)
        XtX = torch.zeros(M, 3, 3, device=wp.device, dtype=torch.float32)
        XtY = torch.zeros(M, 3, 3, device=wp.device, dtype=torch.float32)
        XtX.scatter_add_(0, row_exp, r_r)
        XtY.scatter_add_(0, row_exp, r_du)
        XtX = XtX + 1e-6 * torch.eye(3, device=wp.device, dtype=torch.float32).unsqueeze(0)
        dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
        E_mod, nu = 70e9, 0.33
        lame1 = E_mod * nu / ((1+nu) * (1-2*nu))
        mu    = E_mod / (2*(1+nu))
        tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
        sx  = lame1 * tr + 2*mu * dU[:, 0, 0]
        sy  = lame1 * tr + 2*mu * dU[:, 1, 1]
        sz  = lame1 * tr + 2*mu * dU[:, 2, 2]
        txy = mu * (dU[:, 0, 1] + dU[:, 1, 0])
        txz = mu * (dU[:, 0, 2] + dU[:, 2, 0])
        tyz = mu * (dU[:, 1, 2] + dU[:, 2, 1])
        return torch.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                               + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)

    def compute_vm_stress_per_node(self, wp, ref, edge_index):
        vm = self._vm_stress_raw(wp, ref, edge_index)
        return (vm - self.s_mean.float()) / self.s_std.float()

    def _rebuild_mesh_ef(self, wp, mesh_ei, ref_feat_norm):
        wf = (compute_edge_features(wp, mesh_ei) - self.e_mean_wld) / self.e_std_wld
        return torch.cat([ref_feat_norm, wf], dim=1)

    def train(self, graph, mesh_edge_features, world_edge_features, epoch):
        mesh_edge_features  = mesh_edge_features.to(self.dist.device)
        world_edge_features = world_edge_features.to(self.dist.device)
        self.optimizer.zero_grad()
        self._train_step += 1
        loss, loss_vel, loss_stress = self.forward(
            graph, mesh_edge_features, world_edge_features,
        )
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_vel, loss_stress

    def forward(self, graph, mesh_edge_features, world_edge_features):
        if self.model.training and self.input_noise_std > 0:
            graph = graph.clone()
            graph.x = graph.x.clone()
            graph.x[:, :3] = (graph.x[:, :3]
                              + torch.randn_like(graph.x[:, :3]) * self.input_noise_std)

        x = graph.x.to(self.dist.device)
        disp_norm     = (x[:, :3] - self.d_mean) / self.d_std
        prev_vel_norm = (x[:, 3:6] - self.v_mean) / self.v_std
        if self.use_node_type:
            node_x = torch.cat([disp_norm, prev_vel_norm, x[:, 6:8]], dim=1)
        else:
            node_x = torch.cat([disp_norm, prev_vel_norm], dim=1)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_edge_features, world_edge_features, graph)
            loss_vel = self.criterion(pred, graph.y[:, :3])

        loss_stress = torch.zeros((), device=self.dist.device)

        if self.w_stress > 0 and (self._train_step % self._stress_every == 0):
            ref_all     = graph.mesh_pos.to(self.dist.device)
            disp_t      = graph.x[:, :3].float()
            vel_phys    = pred.float() * self.v_std + self.v_mean
            wp_pred     = ref_all + disp_t + vel_phys
            gt_vel_phys = (graph.y[:, :3].to(self.dist.device).float()
                           * self.v_std + self.v_mean)
            wp_gt       = ref_all + disp_t + gt_vel_phys
            edge_index  = graph.edge_index.to(self.dist.device)
            vm_pred   = self._vm_stress_raw(wp_pred, ref_all, edge_index)
            vm_target = self._vm_stress_raw(wp_gt,   ref_all, edge_index).detach()
            vm_pred_norm   = (vm_pred   - self.s_mean.float()) / self.s_std.float()
            vm_target_norm = (vm_target - self.s_mean.float()) / self.s_std.float()
            loss_stress    = self.criterion(vm_pred_norm, vm_target_norm)

        loss = self.w_vel * loss_vel.float() + self.w_stress * loss_stress.float()
        return loss, loss_vel, loss_stress

    def forward_rollout(self, sequence, epoch):
        p = self.rollout_prob if epoch >= self.rollout_start_epoch else 0.0

        first_graph   = sequence[0]["graph"]
        ref_pos       = first_graph.mesh_pos.to(self.dist.device)
        mesh_ei       = first_graph.edge_index.to(self.dist.device)
        ref_feat_norm = (compute_edge_features(ref_pos, mesh_ei) - self.e_mean_ref) / self.e_std_ref

        wp_current       = first_graph.world_pos.to(self.dist.device)
        prev_vel_current = first_graph.x[:, 3:6].to(self.dist.device)

        total_loss        = torch.tensor(0.0, device=self.dist.device)
        total_vel_loss    = torch.tensor(0.0, device=self.dist.device)
        total_stress_loss = torch.tensor(0.0, device=self.dist.device)

        for k, item in enumerate(sequence):
            graph    = item["graph"].clone().to(self.dist.device)
            world_ef = item["world_edge_features"].to(self.dist.device)

            use_student = (k > 0) and (random.random() < p)

            if use_student:
                input_wp = wp_current.detach()
                x_t      = input_wp - ref_pos
                node_x   = torch.cat([x_t, prev_vel_current.detach()], dim=1)
                mesh_ef  = self._rebuild_mesh_ef(input_wp, mesh_ei, ref_feat_norm)
            else:
                input_wp = graph.world_pos
                node_x   = graph.x
                mesh_ef  = item["mesh_edge_features"].to(self.dist.device)

            graph.x = node_x

            with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                pred     = self.model(node_x, mesh_ef, world_ef, graph)
                loss_vel = self.criterion(pred, graph.y[:, :3])

            loss_stress = torch.zeros((), device=self.dist.device)
            loss = self.w_vel * loss_vel.float()

            total_loss        = total_loss + loss
            total_vel_loss    = total_vel_loss + loss_vel.detach()
            total_stress_loss = total_stress_loss + loss_stress.detach()

            with torch.no_grad():
                vel_phys_detach = pred.float() * self.v_std + self.v_mean
            wp_current       = input_wp + vel_phys_detach.detach()
            prev_vel_current = vel_phys_detach.detach()

        n = len(sequence)
        return total_loss / n, total_vel_loss / n, total_stress_loss / n

    def train_rollout(self, sequence, epoch):
        self.optimizer.zero_grad()
        loss, loss_vel, loss_stress = self.forward_rollout(sequence, epoch)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_vel, loss_stress

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config_prevvel_no_nodetype_witht0")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Training started... [witht0: includes t=0 in training data]")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.sampler.set_epoch(epoch)
        start = time.time()

        progress_bar = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        step_count = epoch * len(trainer.dataloader)

        for batch in progress_bar:
            if cfg.rollout_train_steps > 1:
                trainer.optimizer.zero_grad()
                n = len(batch)
                total_loss = total_vel = total_stress = 0.0
                for seq in batch:
                    loss, loss_vel, loss_stress = trainer.forward_rollout(seq, epoch)
                    (loss / n).backward()
                    total_loss   += loss.item()
                    total_vel    += loss_vel.item()
                    total_stress += loss_stress.item()
                torch.nn.utils.clip_grad_norm_(
                    trainer.model.parameters(), max_norm=0.5
                )
                trainer.optimizer.step()
                trainer.scheduler.step()
                loss        = torch.tensor(total_loss  / n)
                loss_vel    = torch.tensor(total_vel   / n)
                loss_stress = torch.tensor(total_stress / n)
            else:
                batched_graph, batched_mesh_ef, batched_world_ef = batch
                batched_graph    = batched_graph.to(dist.device)
                batched_mesh_ef  = batched_mesh_ef.to(dist.device)
                batched_world_ef = batched_world_ef.to(dist.device)

                loss, loss_vel, loss_stress = trainer.train(
                    batched_graph, batched_mesh_ef, batched_world_ef, epoch,
                )
                del batched_graph, batched_mesh_ef, batched_world_ef

            if dist.rank == 0 and step_count % 10 == 0:
                trainer.writer.add_scalar("loss/step",        loss.item(),        step_count)
                trainer.writer.add_scalar("loss_vel/step",    loss_vel.item(),    step_count)
                trainer.writer.add_scalar("loss_stress/step", loss_stress.item(), step_count)

            step_count += 1
            progress_bar.set_postfix(loss=f"{loss.item():.3e}")

        torch.cuda.empty_cache()
        rank_zero_logger.info(
            f"epoch: {epoch + 1}, loss: {loss:10.3e}, "
            f"loss_vel: {loss_vel:10.3e}, loss_stress: {loss_stress:10.3e}, "
            f"time per epoch: {(time.time() - start):10.3e}"
        )
        if dist.rank == 0:
            trainer.writer.add_scalar("loss",        loss.detach().cpu().item(),        epoch)
            trainer.writer.add_scalar("loss_vel",    loss_vel.detach().cpu().item(),    epoch)
            trainer.writer.add_scalar("loss_stress", loss_stress.detach().cpu().item(), epoch)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            trainer.writer.add_scalar("learning_rate", current_lr, epoch)

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        torch.cuda.empty_cache()

    rank_zero_logger.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
