# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 加 Δx_{t-1}（前一步速度）的訓練版本
# 對應 generate_synthetic_data_nonlinear_prevvel.py + conf/config_prevvel.yaml
#
# 與 train.py 的差異：
#   1. graph.x = [x_t(3), prev_vel(3)]  → num_input_features=6
#   2. forward(): input noise 只加在 x[:, :3]，不動 prev_vel
#   3. forward_rollout(): 在 rollout 過程中維護 prev_vel_current，
#      student forcing 時用預測速度重建 node features

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
from torch_geometric.nn import global_mean_pool



def my_collate_fn(batch):
    return batch


def collate_fn_single_step(batch):
    """Pre-batch single-step sequences in DataLoader worker to overlap CPU work with GPU."""
    graphs    = [seq[0]["graph"]              for seq in batch]
    mesh_efs  = [seq[0]["mesh_edge_features"] for seq in batch]
    world_efs = [seq[0]["world_edge_features"] for seq in batch]
    return (
        Batch.from_data_list(graphs),
        torch.cat(mesh_efs,  dim=0),
        torch.cat(world_efs, dim=0),
    )


def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index   # row=src, col=dst
    disp = pos[col] - pos[row]   # dst - src, matches preprocess_abaqus.py convention
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryTimeStepDataset(torch.utils.data.Dataset):
    """In-memory dataset — loads all timesteps at startup."""

    def __init__(self, sample_dir, num_samples=None, max_nodes=None, sample_indices=None):
        self.data = []
        all_files = sorted(
            [os.path.join(sample_dir, f)
             for f in os.listdir(sample_dir)
             if f.startswith("sample_") and f.endswith(".pt")]
        )
        if sample_indices is not None:
            sample_files = [all_files[i] for i in sample_indices]
        elif num_samples is not None:
            sample_files = all_files[:num_samples]
        else:
            sample_files = all_files
        print(f"Loading {len(sample_files)} sample files into memory (max_nodes={max_nodes}) ...")
        loaded = skipped = 0
        for f in sample_files:
            sample_data = torch.load(f, map_location="cpu", weights_only=False)
            if max_nodes is not None and sample_data[0]["graph"].num_nodes > max_nodes:
                skipped += 1
                continue
            # Merge world_ef into mesh_ef when both share the same edge_index
            # (beam dataset stores ref(E,4) + world(E,4) separately; model expects (E,8) + empty)
            for item in sample_data:
                wef = item["world_edge_features"]
                if wef.shape[0] > 0:
                    item["mesh_edge_features"] = torch.cat(
                        [item["mesh_edge_features"], wef], dim=1
                    )
                    item["world_edge_features"] = torch.zeros(
                        0, item["mesh_edge_features"].shape[1], dtype=torch.float32
                    )
            # skip t=0: prev_vel=0 and x_t=0, model can't infer A → only noise
            self.data.extend(item for item in sample_data
                             if int(item["graph"].step_index.item()) > 0)
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
            train_indices = None
            ranges = getattr(cfg, "train_sample_indices_ranges", None)
            if ranges:
                train_indices = []
                for start, end in ranges:
                    train_indices.extend(range(start, end))
            dataset = InMemoryTimeStepDataset(
                sample_dir,
                num_samples=cfg.num_training_samples,
                max_nodes=getattr(cfg, "max_nodes", None),
                sample_indices=train_indices,
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

        self.criterion             = torch.nn.MSELoss()
        self.lambda_penalty        = cfg.lambda_penalty
        self.w_vel                 = cfg.w_vel
        self.w_stress              = cfg.w_stress
        self.use_node_type         = getattr(cfg, "use_node_type", True)
        self.use_relative_vel_loss = getattr(cfg, "use_relative_vel_loss", False)
        self.use_per_sample_norm   = getattr(cfg, "use_per_sample_norm", False)
        self._train_step           = 0
        self._stress_every         = getattr(cfg, "stress_every", 5)

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
        # edge features: [d_mesh(3), n_mesh(1), d_world(3), n_world(1)]
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
        """Von Mises stress in Pa via GFDM with inverse-distance-squared weighting."""
        M        = wp.shape[0]
        row, col = edge_index
        r_ij  = ref[col].float() - ref[row].float()
        u     = wp.float() - ref.float()
        du_ij = u[col] - u[row]
        # GFDM weight: W_j = 1 / ||r_ij||^2, handles non-uniform mesh density
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

    # ── Standard one-step training ────────────────────────────────────────────
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
        # Input noise only on displacement part (x[:, :3]), not prev_vel
        if self.model.training and self.input_noise_std > 0:
            graph = graph.clone()
            graph.x = graph.x.clone()
            graph.x[:, :3] = (graph.x[:, :3]
                              + torch.randn_like(graph.x[:, :3]) * self.input_noise_std)

        # ── Node feature normalization ────────────────────────────────────────────
        x = graph.x.to(self.dist.device)

        if self.use_per_sample_norm:
            # Per-sample normalization: scale by each graph's own prev_vel magnitude
            # Exploits linear elasticity (u ∝ P): all samples have same shape after scaling
            prev_vel_mag = x[:, 3:6].norm(dim=1, keepdim=True).detach()  # (N, 1)
            per_graph_scale = global_mean_pool(prev_vel_mag, graph.batch) # (B, 1)
            node_scale = per_graph_scale[graph.batch].clamp(min=1e-6)     # (N, 1)

            disp_norm     = x[:, :3]   / node_scale   # ≈ step index (O(1))
            prev_vel_norm = x[:, 3:6]  / node_scale   # ≈ 1 for linear elastic
        else:
            node_scale    = None
            disp_norm     = (x[:, :3]   - self.d_mean) / self.d_std
            prev_vel_norm = (x[:, 3:6]  - self.v_mean) / self.v_std

        if self.use_node_type:
            node_x = torch.cat([disp_norm, prev_vel_norm, x[:, 6:8]], dim=1)  # (N, 8)
        else:
            node_x = torch.cat([disp_norm, prev_vel_norm], dim=1)              # (N, 6)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_edge_features, world_edge_features, graph)

        # ── Velocity loss ─────────────────────────────────────────────────────
        if self.use_per_sample_norm:
            # Target: denormalize from global space → physical → per-sample space
            gt_vel_phys = (graph.y[:, :3].to(self.dist.device).float()
                           * self.v_std + self.v_mean)
            gt_vel_norm = gt_vel_phys / node_scale
            loss_vel = self.criterion(pred.float(), gt_vel_norm)
        else:
            gt_vel = graph.y[:, :3].to(self.dist.device).float()
            if self.use_relative_vel_loss:
                gt_scale = gt_vel.norm(dim=1, keepdim=True).detach().clamp(min=0.1)
                loss_vel = ((pred.float() - gt_vel) / gt_scale).pow(2).mean()
            else:
                loss_vel = self.criterion(pred.float(), gt_vel)

        loss_stress = torch.zeros((), device=self.dist.device)

        if self.w_stress > 0 and (self._train_step % self._stress_every == 0):
            ref_all = graph.mesh_pos.to(self.dist.device)
            disp_t  = graph.x[:, :3].float()                          # wp[t] - ref
            # Denormalize pred vel back to physical space
            if self.use_per_sample_norm:
                vel_phys = pred.float() * node_scale
            else:
                vel_phys = pred.float() * self.v_std + self.v_mean   # predicted vel
            wp_pred     = ref_all + disp_t + vel_phys               # pred wp at t+1
            gt_vel_phys = (graph.y[:, :3].to(self.dist.device).float()
                           * self.v_std + self.v_mean)
            wp_gt       = ref_all + disp_t + gt_vel_phys            # GT wp at t+1
            edge_index  = graph.edge_index.to(self.dist.device)
            vm_pred   = self._vm_stress_raw(wp_pred, ref_all, edge_index)
            vm_target = self._vm_stress_raw(wp_gt,   ref_all, edge_index).detach()
            vm_pred_norm   = (vm_pred   - self.s_mean.float()) / self.s_std.float()
            vm_target_norm = (vm_target - self.s_mean.float()) / self.s_std.float()
            loss_stress    = self.criterion(vm_pred_norm, vm_target_norm)

        loss = self.w_vel * loss_vel.float() + self.w_stress * loss_stress.float()
        return loss, loss_vel, loss_stress

    # ── Rollout training (scheduled sampling) ─────────────────────────────────
    def train_rollout(self, sequence, epoch):
        self.optimizer.zero_grad()
        loss, loss_vel, loss_stress = self.forward_rollout(sequence, epoch)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_vel, loss_stress

    def forward_rollout(self, sequence, epoch):
        """
        Multi-step rollout with scheduled sampling.

        graph.x = [x_t (3), prev_vel (3)]
        Student forcing 時，用預測速度重建 node_x 的 prev_vel 部分。
        """
        p = self.rollout_prob if epoch >= self.rollout_start_epoch else 0.0

        first_graph   = sequence[0]["graph"]
        ref_pos       = first_graph.mesh_pos.to(self.dist.device)
        mesh_ei       = first_graph.edge_index.to(self.dist.device)
        ref_feat_norm = (compute_edge_features(ref_pos, mesh_ei) - self.e_mean_ref) / self.e_std_ref

        wp_current       = first_graph.world_pos.to(self.dist.device)
        # prev_vel_current: 從 dataset 取得第一步的 prev_vel（已在 graph.x[:, 3:6]）
        prev_vel_current = first_graph.x[:, 3:6].to(self.dist.device)  # (N, 3)

        total_loss        = torch.tensor(0.0, device=self.dist.device)
        total_vel_loss    = torch.tensor(0.0, device=self.dist.device)
        total_stress_loss = torch.tensor(0.0, device=self.dist.device)

        for k, item in enumerate(sequence):
            graph    = item["graph"].clone().to(self.dist.device)
            world_ef = item["world_edge_features"].to(self.dist.device)

            use_student = (k > 0) and (random.random() < p)

            if use_student:
                # 用模型上一步的預測位置和速度重建輸入
                input_wp = wp_current.detach()
                x_t      = input_wp - ref_pos
                node_x   = torch.cat([x_t, prev_vel_current.detach()], dim=1)  # (N, 6)
                mesh_ef  = self._rebuild_mesh_ef(input_wp, mesh_ei, ref_feat_norm)
            else:
                # Teacher forcing：直接用 dataset 的 graph.x（含正確 prev_vel）
                input_wp = graph.world_pos
                node_x   = graph.x                 # (N, 6), prev_vel 已在 dataset 中
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

            # update position and prev_vel for next step
            with torch.no_grad():
                vel_phys_detach = pred.float() * self.v_std + self.v_mean
            wp_current       = input_wp + vel_phys_detach.detach()
            prev_vel_current = vel_phys_detach.detach()   # (N, 3)

        n = len(sequence)
        return total_loss / n, total_vel_loss / n, total_stress_loss / n

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

@hydra.main(version_base="1.3", config_path="conf", config_name="config_prevvel")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Training started... [prevvel version: x_t + Δx_{t-1}]")

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
