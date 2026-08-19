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
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # comment out for multi-GPU A100
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
    # Return ALL sequences in the batch for proper graph batching
    return batch


def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """[Δpos(3), dist(1)] per directed edge → (E, 4)."""
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)




# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryTimeStepDataset(torch.utils.data.Dataset):
    """In-memory dataset — loads all timesteps at startup for fast iteration."""

    def __init__(self, sample_dir, num_samples=None):
        self.data = []
        sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("sample_") and f.endswith(".pt")
            ]
        )
        if num_samples is not None:
            sample_files = sample_files[:num_samples]
        print(f"Loading {len(sample_files)} sample files into memory ...")
        for sample_file in sample_files:
            sample_data = torch.load(
                sample_file, map_location="cpu", weights_only=False
            )
            self.data.extend(sample_data)
        print(f"Loaded {len(self.data)} timesteps into memory")

    def __getitem__(self, idx):
        return [self.data[idx]]   # wrap in list for uniform sequence interface

    def __len__(self):
        return len(self.data)


class LazyTimeStepDataset(torch.utils.data.Dataset):
    """Lazy single-step dataset."""

    def __init__(self, sample_dir, num_time_steps):
        self.sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("sample_") and f.endswith(".pt")
            ]
        )
        self.num_steps = num_time_steps - 1
        self.total_samples = len(self.sample_files) * self.num_steps
        print(
            f"Found {len(self.sample_files)} sample files, "
            f"{self.total_samples} samples in total."
        )

    def __getitem__(self, idx):
        file_idx = idx // self.num_steps
        idx_in_file = idx % self.num_steps
        sample_data = torch.load(
            self.sample_files[file_idx], map_location="cpu", weights_only=False
        )
        return [sample_data[idx_in_file]]   # wrap in list for uniform interface

    def __len__(self):
        return self.total_samples


class SequenceDataset(torch.utils.data.Dataset):
    """
    Returns sequences of consecutive time steps from the same sample.
    Used for rollout training (scheduled sampling).
    Each item is a list of seq_len consecutive dicts.
    """

    def __init__(self, sample_dir, num_time_steps, seq_len):
        self.sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("sample_") and f.endswith(".pt")
            ]
        )
        self.num_steps   = num_time_steps - 1
        self.seq_len     = min(seq_len, self.num_steps)
        self.valid_starts = self.num_steps - self.seq_len + 1
        self.total_samples = len(self.sample_files) * self.valid_starts
        print(
            f"Found {len(self.sample_files)} sample files, "
            f"seq_len={self.seq_len}, "
            f"{self.total_samples} sequences in total."
        )

    def __getitem__(self, idx):
        file_idx = idx // self.valid_starts
        start    = idx % self.valid_starts
        data = torch.load(
            self.sample_files[file_idx], map_location="cpu", weights_only=False
        )
        return [data[start + k] for k in range(self.seq_len)]

    def __len__(self):
        return self.total_samples


class InMemorySequenceDataset(torch.utils.data.Dataset):
    """In-memory sequence dataset for rollout training.

    Loads all training samples at startup and pre-builds all valid
    start-of-sequence indices so __getitem__ is a pure list lookup.
    """

    def __init__(self, sample_dir, num_time_steps, seq_len, num_samples=None):
        sample_files = sorted(
            [
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.startswith("sample_") and f.endswith(".pt")
            ]
        )
        if num_samples is not None:
            sample_files = sample_files[:num_samples]

        num_steps    = num_time_steps - 1
        seq_len      = min(seq_len, num_steps)
        valid_starts = num_steps - seq_len + 1

        print(f"Loading {len(sample_files)} sample files into memory (seq_len={seq_len}) ...")
        all_data = []
        for f in sample_files:
            all_data.append(
                torch.load(f, map_location="cpu", weights_only=False)
            )

        # Pre-build sequence list: each entry is a (sample_idx, start) tuple
        self.sequences = [
            (si, start)
            for si in range(len(all_data))
            for start in range(valid_starts)
        ]
        self._data     = all_data
        self._seq_len  = seq_len
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

        # ── Dataset ───────────────────────────────────────────────────────────
        seq_len    = cfg.rollout_train_steps
        sample_dir = to_absolute_path(cfg.preprocess_output_dir)
        if seq_len > 1:
            dataset = InMemorySequenceDataset(
                sample_dir,
                cfg.num_training_time_steps,
                seq_len,
                num_samples=cfg.num_training_samples,
            )
        else:
            dataset = InMemoryTimeStepDataset(
                sample_dir,
                num_samples=cfg.num_training_samples,
            )

        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            pin_memory=False,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=my_collate_fn,
        )
        self.sampler = sampler

        # ── Model ─────────────────────────────────────────────────────────────
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
        if cfg.jit:
            self.model = torch.compile(self.model).to(self.dist.device)
        else:
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

        # ── Loss / Optimizer / Scheduler ──────────────────────────────────────
        self.criterion      = torch.nn.MSELoss()
        self.lambda_penalty = cfg.lambda_penalty
        self.w_vel          = cfg.w_vel
        self.w_stress       = cfg.w_stress

        self.optimizer = None
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex not installed; falling back to Adam."
            )
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
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

        # ── Rollout training setup ────────────────────────────────────────────
        self.rollout_train_steps = cfg.rollout_train_steps
        self.rollout_start_epoch = getattr(cfg, "rollout_start_epoch", 9999)
        self.rollout_prob        = getattr(cfg, "rollout_prob", 0.5)
        self.input_noise_std     = getattr(cfg, "input_noise_std", 0.0)

        self._load_rollout_statics(cfg)

    def _load_rollout_statics(self, cfg):
        """Load normalization stats and precompute mesh-fixed quantities."""
        edge_stats   = load_json(to_absolute_path("edge_stats.json"))
        self.e_mean  = edge_stats["edge_mean"].to(self.dist.device)[:4]
        self.e_std   = edge_stats["edge_std"].to(self.dist.device)[:4]

        node_stats   = load_json(to_absolute_path("node_stats.json"))
        self.v_mean  = node_stats["velocity_mean"].to(self.dist.device)
        self.v_std   = node_stats["velocity_std"].to(self.dist.device)
        self.s_mean  = node_stats["stress_mean"].to(self.dist.device)
        self.s_std   = node_stats["stress_std"].to(self.dist.device)

        self.empty_world_ef = torch.zeros(
            0, cfg.num_edge_features, device=self.dist.device
        )

        # ── Precompute mesh-fixed stress quantities (done ONCE, reused every batch) ──
        # All samples share the same mesh topology, so XtX and r_ij are constant.
        sample_dir  = to_absolute_path(cfg.preprocess_output_dir)
        first_file  = sorted(
            f for f in os.listdir(sample_dir)
            if f.startswith("sample_") and f.endswith(".pt")
        )[0]
        first_step  = torch.load(
            os.path.join(sample_dir, first_file),
            map_location="cpu", weights_only=False,
        )[0]
        ref_pos_s   = first_step["graph"].mesh_pos.to(self.dist.device)   # (N, 3)
        E_mesh      = first_step["mesh_edge_features"].shape[0]
        mesh_ei_s   = first_step["graph"].edge_index[:, :E_mesh].to(self.dist.device)

        self.N_single = ref_pos_s.shape[0]
        self.E_single = E_mesh

        row_s, col_s  = mesh_ei_s
        r_ij_s        = ref_pos_s[col_s] - ref_pos_s[row_s]              # (E, 3)
        r_r_s         = r_ij_s.unsqueeze(-1) * r_ij_s.unsqueeze(-2)     # (E, 3, 3)
        row_exp_s     = row_s.view(-1, 1, 1).expand_as(r_r_s)
        XtX_s         = torch.zeros(self.N_single, 3, 3, device=self.dist.device)
        XtX_s.scatter_add_(0, row_exp_s, r_r_s)
        XtX_s         = XtX_s + 1e-6 * torch.eye(3, device=self.dist.device).unsqueeze(0)
        self.XtX_inv_single = torch.linalg.inv(XtX_s)                    # (N, 3, 3)
        self.r_ij_single    = r_ij_s                                      # (E, 3)

        # Precomputed material constants (avoid recomputing every call)
        E_mod, nu = 70e9, 0.33
        self.lame1 = float(E_mod * nu / ((1 + nu) * (1 - 2 * nu)))
        self.mu    = float(E_mod / (2 * (1 + nu)))

    def compute_vm_stress_per_node(self, wp: torch.Tensor,
                                    ref: torch.Tensor,
                                    edge_index: torch.Tensor) -> torch.Tensor:
        """
        Per-node von Mises stress via scatter gradient fit.

        Fast path  (homogeneous mesh): M is a multiple of N_single →
          uses precomputed XtX_inv and r_ij (one bmm, no solve).
        Fallback   (heterogeneous mesh): per-batch XtX construction + solve.
          Activated automatically when mesh sizes differ across samples.

        wp, ref    : (M, 3)
        edge_index : (2, E_total) with PyG-offset indices
        Returns    : (M,) normalized von Mises stress
        """
        M        = wp.shape[0]
        row, col = edge_index
        u        = wp.float() - ref.float()    # (M, 3) displacement
        du_ij    = u[col] - u[row]             # (E_total, 3)

        use_precomputed = (M % self.N_single == 0)

        if use_precomputed:
            B     = M // self.N_single
            r_ij  = self.r_ij_single.repeat(B, 1)             # (B*E, 3)
            r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)  # (B*E, 3, 3)
            row_exp = row.view(-1, 1, 1).expand_as(r_du)
            XtY   = torch.zeros(M, 3, 3, device=wp.device, dtype=torch.float32)
            XtY.scatter_add_(0, row_exp, r_du)
            XtX_inv = self.XtX_inv_single.repeat(B, 1, 1)     # (B*N, 3, 3)
            dU    = torch.bmm(XtX_inv, XtY).permute(0, 2, 1)  # (M, 3, 3)
        else:
            # Fallback: heterogeneous mesh — rebuild XtX from ref per batch
            r_ij  = ref[col].float() - ref[row].float()
            r_r   = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
            r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
            row_exp = row.view(-1, 1, 1).expand_as(r_r)
            XtX   = torch.zeros(M, 3, 3, device=wp.device, dtype=torch.float32)
            XtY   = torch.zeros(M, 3, 3, device=wp.device, dtype=torch.float32)
            XtX.scatter_add_(0, row_exp, r_r)
            XtY.scatter_add_(0, row_exp, r_du)
            reg   = 1e-6 * torch.eye(3, device=wp.device, dtype=torch.float32).unsqueeze(0)
            dU    = torch.linalg.solve(XtX + reg, XtY).permute(0, 2, 1)

        tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
        sx  = self.lame1 * tr + 2 * self.mu * dU[:, 0, 0]
        sy  = self.lame1 * tr + 2 * self.mu * dU[:, 1, 1]
        sz  = self.lame1 * tr + 2 * self.mu * dU[:, 2, 2]
        txy = self.mu * (dU[:, 0, 1] + dU[:, 1, 0])
        txz = self.mu * (dU[:, 0, 2] + dU[:, 2, 0])
        tyz = self.mu * (dU[:, 1, 2] + dU[:, 2, 1])
        vm  = torch.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                                 + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)
        return (vm - self.s_mean[0].float()) / self.s_std[0].float()

    # ── Node features: disp_vec only ─────────────────────────────────────────
    def _make_node_x(self, wp: torch.Tensor, ref_pos: torch.Tensor) -> torch.Tensor:
        return wp - ref_pos                                              # (N, 3)

    # ── Rebuild mesh edge features from predicted world_pos ──────────────────
    def _rebuild_mesh_ef(self, wp: torch.Tensor, mesh_ei: torch.Tensor,
                         ref_feat_norm: torch.Tensor) -> torch.Tensor:
        world_feat   = compute_edge_features(wp, mesh_ei)
        world_feat_n = (world_feat - self.e_mean) / self.e_std
        return torch.cat([ref_feat_norm, world_feat_n], dim=1)          # (E, 8)

    # ── Standard one-step training ───────────────────────────────────────────
    def train(self, graph, mesh_edge_features, world_edge_features, epoch):
        mesh_edge_features  = mesh_edge_features.to(self.dist.device)
        world_edge_features = world_edge_features.to(self.dist.device)
        self.optimizer.zero_grad()
        loss, loss_vel, loss_stress = self.forward(
            graph, mesh_edge_features, world_edge_features
        )
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_vel, loss_stress

    def forward(self, graph, mesh_edge_features, world_edge_features):
        x = graph.x
        if self.model.training and self.input_noise_std > 0:
            x = x + torch.randn_like(x) * self.input_noise_std   # no graph.clone()
        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred     = self.model(x, mesh_edge_features, world_edge_features, graph)
            loss_vel = self.criterion(pred, graph.y[:, :3])

        vel_phys    = pred.float() * self.v_std + self.v_mean
        wp_new      = graph.world_pos.float() + vel_phys
        stress_pred = self.compute_vm_stress_per_node(
            wp_new, graph.mesh_pos.float(), graph.edge_index
        ).unsqueeze(-1)                                                        # (total_N, 1)
        loss_stress = self.criterion(stress_pred, graph.y[:, 3:].float())
        penalty     = torch.relu(-0.3278 - stress_pred).pow(2).mean()
        loss = (self.w_vel * loss_vel.float()
                + self.w_stress * loss_stress
                + self.lambda_penalty * penalty)
        return loss, loss_vel, loss_stress

    # ── Rollout training (scheduled sampling) ────────────────────────────────
    def train_rollout(self, sequence, epoch):
        self.optimizer.zero_grad()
        loss, loss_vel, loss_stress = self.forward_rollout(sequence, epoch)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_vel, loss_stress

    def forward_rollout_batched(self, batch_sequences, epoch):
        """
        Batched rollout training with two execution paths:

        Fast path  (p == 0, pure teacher forcing):
          All B*K graphs have independent GT inputs → one forward pass.
          Identical speed to single-step teacher forcing.

        Sequential path (p > 0, student forcing active):
          Step k's input may depend on step k-1's output → K sequential
          forward passes, each on B graphs (same batch size as teacher forcing).
          Cost: K× slower than teacher forcing.

        Gradients accumulated internally; caller needs only:
          clip_grad_norm_ + optimizer.step() + scheduler.step()
        """
        p = self.rollout_prob if epoch >= self.rollout_start_epoch else 0.0
        K = len(batch_sequences[0])

        # ── Fast path: pure teacher forcing (no sequential dependency) ────────
        # Only use the first step of each sequence so TF speed == K=1 regardless
        # of rollout_train_steps.  SF phase uses the full K-step sequential path.
        if p == 0.0:
            graphs_all, mesh_efs_all = [], []
            for seq in batch_sequences:
                item = random.choice(seq)   # random step so all timesteps get coverage
                graphs_all.append(item["graph"])
                mesh_efs_all.append(item["mesh_edge_features"])

            batched_g   = Batch.from_data_list(graphs_all).to(self.dist.device)
            batched_mef = torch.cat(mesh_efs_all, dim=0).to(self.dist.device)

            if self.model.training and self.input_noise_std > 0:
                batched_g.x = batched_g.x + torch.randn_like(batched_g.x) * self.input_noise_std

            with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                pred     = self.model(batched_g.x, batched_mef, self.empty_world_ef, batched_g)
                loss_vel = self.criterion(pred, batched_g.y[:, :3])

            vel_phys    = pred.float() * self.v_std + self.v_mean
            wp_new      = batched_g.world_pos.float() + vel_phys
            stress_pred = self.compute_vm_stress_per_node(
                wp_new, batched_g.mesh_pos.float(), batched_g.edge_index
            ).unsqueeze(-1)
            loss_stress = self.criterion(stress_pred, batched_g.y[:, 3:].float())
            penalty     = torch.relu(-0.3278 - stress_pred).pow(2).mean()
            loss        = (self.w_vel * loss_vel.float()
                           + self.w_stress * loss_stress
                           + self.lambda_penalty * penalty)
            self.backward(loss)   # handles AMP scaler + optimizer.step internally
            return loss.detach(), loss_vel.detach(), loss_stress.detach()

        # ── Sequential path: student forcing active (K passes, each full batch) ─
        B = len(batch_sequences)
        item0   = batch_sequences[0][0]
        ref_pos = item0["graph"].mesh_pos.to(self.dist.device)
        mesh_ei = item0["graph"].edge_index.to(self.dist.device)
        N       = ref_pos.shape[0]
        ref_feat_n = (compute_edge_features(ref_pos, mesh_ei) - self.e_mean) / self.e_std

        wp_currents = torch.stack([
            seq[0]["graph"].world_pos.to(self.dist.device)
            for seq in batch_sequences
        ])  # (B, N, 3)

        total_loss_v = total_vel_v = total_stress_v = 0.0

        for k in range(K):
            student_mask = torch.tensor(
                [(k > 0) and (random.random() < p) for _ in range(B)],
                dtype=torch.bool, device=self.dist.device,
            )
            gt_wp    = torch.stack([seq[k]["graph"].world_pos.to(self.dist.device)
                                    for seq in batch_sequences])
            input_wp = torch.where(student_mask[:, None, None],
                                   wp_currents.detach(), gt_wp)  # (B, N, 3)

            graphs_k, mesh_efs_k = [], []
            for b in range(B):
                wp_b = input_wp[b]
                g    = batch_sequences[b][k]["graph"].clone().to(self.dist.device)
                g.x  = wp_b - ref_pos
                wf   = (compute_edge_features(wp_b, mesh_ei) - self.e_mean) / self.e_std
                graphs_k.append(g)
                mesh_efs_k.append(torch.cat([ref_feat_n, wf], dim=1))

            batched_g   = Batch.from_data_list(graphs_k)
            batched_mef = torch.cat(mesh_efs_k, dim=0)

            if self.model.training and self.input_noise_std > 0:
                batched_g.x = batched_g.x + torch.randn_like(batched_g.x) * self.input_noise_std

            with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                pred     = self.model(batched_g.x, batched_mef, self.empty_world_ef, batched_g)
                loss_vel = self.criterion(pred, batched_g.y[:, :3])

            vel_phys    = pred.float() * self.v_std + self.v_mean
            wp_flat     = input_wp.reshape(B * N, 3).float()
            wp_new      = wp_flat + vel_phys
            stress_pred = self.compute_vm_stress_per_node(
                wp_new, batched_g.mesh_pos.float(), batched_g.edge_index
            ).unsqueeze(-1)
            loss_stress = self.criterion(stress_pred, batched_g.y[:, 3:].float())
            penalty     = torch.relu(-0.3278 - stress_pred).pow(2).mean()
            loss        = (self.w_vel * loss_vel.float()
                           + self.w_stress * loss_stress
                           + self.lambda_penalty * penalty)

            (loss / K).backward()   # accumulate gradient across K steps
            total_loss_v   += loss.detach().item()
            total_vel_v    += loss_vel.detach().item()
            total_stress_v += loss_stress.detach().item()

            with torch.no_grad():
                wp_currents = (wp_flat + vel_phys.detach()).reshape(B, N, 3)

        return (
            torch.tensor(total_loss_v   / K),
            torch.tensor(total_vel_v    / K),
            torch.tensor(total_stress_v / K),
        )

    def forward_rollout(self, sequence, epoch):
        """
        Multi-step rollout with scheduled sampling.
        - epoch < rollout_start_epoch : pure teacher forcing
        - epoch >= rollout_start_epoch: each step (k>0) uses predicted world_pos
                                        with probability rollout_prob
        """
        p = self.rollout_prob if epoch >= self.rollout_start_epoch else 0.0

        # Extract per-sample geometry from the first timestep (constant within a sequence)
        first_graph   = sequence[0]["graph"]
        ref_pos       = first_graph.mesh_pos.to(self.dist.device)
        mesh_ei       = first_graph.edge_index.to(self.dist.device)
        ref_feat      = compute_edge_features(ref_pos, mesh_ei)
        ref_feat_norm = (ref_feat - self.e_mean) / self.e_std

        # Start from GT world_pos of the first step in the sequence
        wp_current = first_graph.world_pos.to(self.dist.device)

        total_loss        = torch.tensor(0.0, device=self.dist.device)
        total_vel_loss    = torch.tensor(0.0, device=self.dist.device)
        total_stress_loss = torch.tensor(0.0, device=self.dist.device)

        for k, item in enumerate(sequence):
            graph    = item["graph"].clone().to(self.dist.device)
            world_ef = item["world_edge_features"].to(self.dist.device)

            use_student = (k > 0) and (random.random() < p)

            if use_student:
                input_wp = wp_current.detach()
                node_x   = self._make_node_x(input_wp, ref_pos)
                mesh_ef  = self._rebuild_mesh_ef(input_wp, mesh_ei, ref_feat_norm)
            else:
                input_wp = graph.world_pos
                node_x   = graph.x                                        # already has disp_mag
                mesh_ef  = item["mesh_edge_features"].to(self.dist.device)

            # Override graph.x so the model sees the correct node features
            graph.x = node_x

            with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                pred     = self.model(node_x, mesh_ef, world_ef, graph)
                loss_vel = self.criterion(pred, graph.y[:, :3])

            # Per-node analytical stress (fp32)
            vel_phys    = pred.float() * self.v_std + self.v_mean
            wp_new      = input_wp.float() + vel_phys
            stress_pred = self.compute_vm_stress_per_node(
                wp_new, ref_pos, mesh_ei
            ).unsqueeze(-1)
            loss_stress = self.criterion(stress_pred, graph.y[:, 3:].float())
            penalty     = torch.relu(-0.3278 - stress_pred).pow(2).mean()
            loss        = (self.w_vel * loss_vel.float()
                           + self.w_stress * loss_stress
                           + self.lambda_penalty * penalty)

            total_loss        = total_loss + loss
            total_vel_loss    = total_vel_loss + loss_vel.detach()
            total_stress_loss = total_stress_loss + loss_stress.detach()

            # Update world_pos for next step using predicted velocity
            with torch.no_grad():
                vel_pred = pred.float() * self.v_std + self.v_mean
            wp_current = input_wp + vel_pred.detach()

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

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Training started...")

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
            # batch is now a list of sequences (my_collate_fn returns all items).
            if cfg.rollout_train_steps > 1:
                # forward_rollout_batched handles backward internally.
                # Fast path (p=0): also calls optimizer.step() via self.backward().
                # Sequential path (p>0): only accumulates gradients; step here.
                p_active = (trainer.rollout_prob
                            if epoch >= trainer.rollout_start_epoch else 0.0)
                trainer.optimizer.zero_grad()
                loss, loss_vel, loss_stress = trainer.forward_rollout_batched(batch, epoch)
                if p_active > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainer.model.parameters(), max_norm=0.5
                    )
                    trainer.optimizer.step()
                trainer.scheduler.step()
            else:
                # PyG graph batching: merge all sequences in the batch into one
                # batched graph for a single forward/backward pass.
                graphs   = [seq[0]["graph"]               for seq in batch]
                mesh_efs = [seq[0]["mesh_edge_features"]  for seq in batch]
                world_efs= [seq[0]["world_edge_features"] for seq in batch]

                batched_graph   = Batch.from_data_list(graphs).to(dist.device)
                batched_mesh_ef = torch.cat(mesh_efs,  dim=0).to(dist.device)
                batched_world_ef= torch.cat(world_efs, dim=0).to(dist.device)

                loss, loss_vel, loss_stress = trainer.train(
                    batched_graph, batched_mesh_ef, batched_world_ef, epoch
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
