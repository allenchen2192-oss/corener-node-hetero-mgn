"""
train_bimaterial_direct.py
==========================
Minimal velocity-only model for bi-material beam.

Same architecture as v2 (HybridMeshGraphNet, 9D node, 9D edge, 3D output).
Loss: pure MSE on velocity — no stress loss, no strain loss, no extra terms.
Stress computed post-hoc via B-matrix (not in training loss).
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

import numpy as np


def collate_fn(batch):
    graphs    = [seq[0]["graph"]              for seq in batch]
    mesh_efs  = [seq[0]["mesh_edge_features"] for seq in batch]
    world_efs = [seq[0]["world_edge_features"] for seq in batch]
    return (
        Batch.from_data_list(graphs),
        torch.cat(mesh_efs,  dim=0),
        torch.cat(world_efs, dim=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (reuses existing preprocessed data)
# ─────────────────────────────────────────────────────────────────────────────

class BimaterialDataset(torch.utils.data.Dataset):
    def __init__(self, sample_dir, sample_indices):
        self.data = []
        all_files = sorted(
            [os.path.join(sample_dir, f)
             for f in os.listdir(sample_dir)
             if f.startswith("sample_") and f.endswith(".pt")]
        )
        sample_files = [all_files[i] for i in sample_indices]
        print(f"Loading {len(sample_files)} sample files ...")
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


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DirectDisplacementTrainer:
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
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=sampler,
            pin_memory=False, num_workers=0,
            collate_fn=collate_fn,
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
        self.criterion = torch.nn.MSELoss()

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
        node_stats = load_json(os.path.join(data_dir, "node_stats_bimaterial.json"))
        dev = self.dist.device
        self.v_mean = node_stats["velocity_mean"].to(dev)
        self.v_std  = node_stats["velocity_std"].to(dev)
        self.d_mean = node_stats["disp_mean"].to(dev)
        self.d_std  = node_stats["disp_std"].to(dev)

    def forward(self, graph, mesh_ef, world_ef):
        dev = self.dist.device
        x   = graph.x.to(dev)

        # Optional input noise on displacement
        if self.model.training and self.input_noise_std > 0:
            graph = graph.clone()
            graph.x = graph.x.clone().to(dev)
            graph.x[:, :3] += torch.randn_like(graph.x[:, :3]) * self.input_noise_std * self.d_std
            x = graph.x

        # Node features: [disp_norm(3), prev_vel_norm(3), node_type_oh(3)] = 9D
        node_x = torch.cat([
            (x[:, :3]  - self.d_mean) / self.d_std,
            (x[:, 3:6] - self.v_mean) / self.v_std,
            x[:, 6:9],
        ], dim=1)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_ef.to(dev), world_ef.to(dev), graph)

        # Target: normalized velocity (same as v2; no extra loss terms)
        gt_vel = graph.y[:, :3].to(dev).float()
        loss   = self.criterion(pred.float(), gt_vel)
        return loss

    def train_step(self, graph, mesh_ef, world_ef):
        self.optimizer.zero_grad()
        loss = self.forward(graph, mesh_ef, world_ef)
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
        return loss


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config_bimaterial_direct")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = DirectDisplacementTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Training started... [direct displacement rollout]")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.sampler.set_epoch(epoch)
        start = time.time()

        progress_bar = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        loss_sum, n_steps = 0.0, 0
        for batch in progress_bar:
            batched_graph, batched_mesh_ef, batched_world_ef = batch
            batched_graph = batched_graph.to(dist.device)

            loss = trainer.train_step(batched_graph, batched_mesh_ef, batched_world_ef)
            del batched_graph, batched_mesh_ef, batched_world_ef

            loss_sum += loss.item()
            n_steps  += 1
            progress_bar.set_postfix(loss=f"{loss.item():.3e}")

        elapsed   = time.time() - start
        mean_loss = loss_sum / max(n_steps, 1)

        if dist.rank == 0:
            trainer.writer.add_scalar("train/loss", mean_loss, epoch)
            rank_zero_logger.info(
                f"epoch: {epoch + 1}, loss: {mean_loss:.3e}, time: {elapsed:.1f}s"
            )
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )

    rank_zero_logger.info("Training complete.")


if __name__ == "__main__":
    main()
