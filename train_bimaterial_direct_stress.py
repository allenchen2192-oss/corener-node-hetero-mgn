"""
Train MeshGraphNet on bi-material beam with direct Abaqus stress supervision.

Differences from train_bimaterial.py:
  - num_output_features = 4: model predicts [vel(3), stress_vm(1)]
  - stress supervised directly by Abaqus GT von Mises (graph.y[:, 3])
  - no GFDM stress computation during training
  - graph.y already 4D from preprocess_bimaterial.py
"""

import os
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
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch


def collate_fn(batch):
    graphs    = [seq[0]["graph"]               for seq in batch]
    mesh_efs  = [seq[0]["mesh_edge_features"]  for seq in batch]
    world_efs = [seq[0]["world_edge_features"] for seq in batch]
    return (
        Batch.from_data_list(graphs),
        torch.cat(mesh_efs,  dim=0),
        torch.cat(world_efs, dim=0),
    )


class BimaterialDataset(torch.utils.data.Dataset):
    def __init__(self, sample_dir, sample_indices):
        self.data = []
        all_files = sorted(
            [os.path.join(sample_dir, f)
             for f in os.listdir(sample_dir)
             if f.startswith("sample_") and f.endswith(".pt")]
        )
        sample_files = [all_files[i] for i in sample_indices]
        print(f"Loading {len(sample_files)} sample files into memory ...")
        for f in sample_files:
            sample_data = torch.load(f, map_location="cpu", weights_only=False)
            self.data.extend(
                item for item in sample_data
                if int(item["graph"].step_index.item()) > 0
            )
        print(f"  {len(self.data)} timesteps loaded.")

    def __getitem__(self, idx):
        return [self.data[idx]]

    def __len__(self):
        return len(self.data)


class BimaterialDirectStressTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp  = cfg.amp

        mlp_act = "silu" if cfg.recompute_activation else "relu"

        sample_dir = to_absolute_path(cfg.preprocess_output_dir)
        train_indices = list(range(cfg.train_start, cfg.train_end))
        dataset = BimaterialDataset(sample_dir, train_indices)

        sampler = DistributedSampler(
            dataset, shuffle=True, drop_last=True,
            num_replicas=self.dist.world_size, rank=self.dist.rank,
        )
        n_workers = cfg.num_dataloader_workers
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=sampler,
            pin_memory=False, num_workers=n_workers,
            collate_fn=collate_fn,
            persistent_workers=(n_workers > 0),
        )
        self.sampler = sampler

        self.model = HybridMeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,     # 4: vel(3) + stress_vm(1)
            processor_size=cfg.processor_size,
            hidden_dim_processor=cfg.hidden_dim,
            hidden_dim_node_encoder=cfg.hidden_dim,
            hidden_dim_edge_encoder=cfg.hidden_dim,
            hidden_dim_node_decoder=cfg.hidden_dim,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        ).to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()

        self.criterion       = torch.nn.MSELoss()
        self.w_vel           = cfg.w_vel
        self.w_stress_direct = getattr(cfg, "w_stress_direct", 1.0)

        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except (ImportError, AttributeError):
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, foreach=True
            )

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

        self.input_noise_std = getattr(cfg, "input_noise_std", 0.0)
        self._load_stats(cfg)

    def _load_stats(self, cfg):
        data_dir   = to_absolute_path(cfg.preprocess_output_dir)
        node_stats = load_json(
            os.path.join(data_dir, "node_stats_bimaterial.json")
        )
        self.v_mean = node_stats["velocity_mean"].to(self.dist.device)
        self.v_std  = node_stats["velocity_std"].to(self.dist.device)
        self.d_mean = node_stats["disp_mean"].to(self.dist.device)
        self.d_std  = node_stats["disp_std"].to(self.dist.device)

    def forward(self, graph, mesh_ef, world_ef):
        if self.model.training and self.input_noise_std > 0:
            graph = graph.clone()
            graph.x = graph.x.clone()
            graph.x[:, :3] = (graph.x[:, :3]
                              + torch.randn_like(graph.x[:, :3]) * self.input_noise_std)

        x = graph.x.to(self.dist.device)
        disp_norm     = (x[:, :3]  - self.d_mean) / self.d_std
        prev_vel_norm = (x[:, 3:6] - self.v_mean) / self.v_std
        node_type_oh  = x[:, 6:9]
        node_x = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_ef, world_ef, graph)

        gt = graph.y.to(self.dist.device).float()    # (N, 4): [vel_norm(3), stress_norm(1)]
        pred_f = pred.float()

        loss_vel    = self.criterion(pred_f[:, :3], gt[:, :3])
        loss_stress = self.criterion(pred_f[:, 3],  gt[:, 3])
        loss        = self.w_vel * loss_vel + self.w_stress_direct * loss_stress
        return loss, loss_vel, loss_stress

    def train_step(self, graph, mesh_ef, world_ef, epoch):
        mesh_ef  = mesh_ef.to(self.dist.device)
        world_ef = world_ef.to(self.dist.device)
        self.optimizer.zero_grad()
        loss, loss_vel, loss_stress = self.forward(graph, mesh_ef, world_ef)
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
        self.scheduler.step()
        return loss, loss_vel, loss_stress


@hydra.main(version_base="1.3", config_path="conf",
            config_name="config_bimaterial_direct_stress")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = BimaterialDirectStressTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info(
        "Training started... [bimaterial direct stress: vel(3) + stress_vm(1)]"
    )

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.sampler.set_epoch(epoch)
        start      = time.time()
        step_count = epoch * len(trainer.dataloader)

        progress_bar = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        for batch in progress_bar:
            batched_graph, batched_mesh_ef, batched_world_ef = batch
            batched_graph = batched_graph.to(dist.device)

            loss, loss_vel, loss_stress = trainer.train_step(
                batched_graph, batched_mesh_ef, batched_world_ef, epoch
            )
            del batched_graph, batched_mesh_ef, batched_world_ef

            if dist.rank == 0 and step_count % 10 == 0:
                trainer.writer.add_scalar("loss/step",        loss.item(),        step_count)
                trainer.writer.add_scalar("loss_vel/step",    loss_vel.item(),    step_count)
                trainer.writer.add_scalar("loss_stress/step", loss_stress.item(), step_count)

            step_count += 1
            progress_bar.set_postfix(
                loss=f"{loss.item():.3e}",
                vel=f"{loss_vel.item():.3e}",
                stress=f"{loss_stress.item():.3e}",
            )

        torch.cuda.empty_cache()
        rank_zero_logger.info(
            f"epoch: {epoch + 1}, loss: {loss:10.3e}, "
            f"loss_vel: {loss_vel:10.3e}, "
            f"loss_stress: {loss_stress:10.3e}, "
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
            logger.info(f"Saved checkpoint (epoch {epoch + 1})")
        torch.cuda.empty_cache()

    rank_zero_logger.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
