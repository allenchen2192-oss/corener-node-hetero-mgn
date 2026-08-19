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

"""
infer_synthetic.py
==================
Autoregressive rollout inference for the synthetic linear block (100 samples).

策略:
  - λ(0) = sin(0) = 0，t=0 時所有 sample 位移皆為 0，無法區分不同 A 矩陣。
  - 以 ground truth t=1 的 world_pos 作為起始點（第一步位移已編碼 A 矩陣資訊）。
  - 模型自回歸預測 t=2 ~ t=N-1（共 N-2 步）的 world_pos 與 stress。
  - 對全部 100 個 sample 執行 rollout，印出逐步與最終誤差統計。
  - 對前 num_export_samples 個 sample 輸出 VTU 檔（ParaView 可視化）。

輸出:
  paraview_output_synthetic/
    sample_00000/
      step_001_pred.vtu   ← 預測（step 編號 = rollout 的第幾步預測，t=2 → step_001）
      step_001_exact.vtu
      ...
    sample_00001/
      ...
    error_per_step.csv    ← 全 sample 逐步位置誤差 & 應力誤差（mean ± std）

使用:
  python infer_synthetic.py
"""

import os

import hydra
import meshio
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """[relative_displacement(3), distance(1)] per directed edge → (E, 4)."""
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def build_hex_cells(n: int) -> np.ndarray:
    """VTK hexahedron connectivity for an n×n×n regular grid.

    Node ordering matches preprocess_cases_v3.py build_structured_grid:
      node_id(i, j, k) = k*n*n + j*n + i
      (outer loop k=z, middle j=y, inner i=x)
    """
    cells = []
    for ck in range(n - 1):
        for cj in range(n - 1):
            for ci in range(n - 1):
                def nd(di, dj, dk, ci=ci, cj=cj, ck=ck):
                    return (ck + dk) * n * n + (cj + dj) * n + (ci + di)
                cells.append([
                    nd(0,0,0), nd(1,0,0), nd(1,1,0), nd(0,1,0),
                    nd(0,0,1), nd(1,0,1), nd(1,1,1), nd(0,1,1),
                ])
    return np.array(cells, dtype=np.int64)


def load_gt_trajectory(sample_path: str, v_mean, v_std, s_mean, s_std):
    """
    Load ground truth world_pos and stress for all time steps from a sample .pt file.

    Returns:
      gt_world_pos : list of (N, 3) tensors, length = T   (t = 0 .. T-1)
      gt_stress    : list of (N, 1) tensors, length = T-1 (stress at t = 0 .. T-2)
    """
    sample_data = torch.load(sample_path, map_location="cpu", weights_only=False)

    gt_world_pos = []
    gt_stress    = []

    for step in sample_data:
        g = step["graph"]
        gt_world_pos.append(g.world_pos.clone())                        # world_pos at t
        stress = g.y[:, 3:] * s_std + s_mean                           # denorm stress at t
        gt_stress.append(stress)

    # Recover last world_pos: world_pos[T-1] = world_pos[T-2] + vel[T-2]
    last_vel = sample_data[-1]["graph"].y[:, :3] * v_std + v_mean      # (N, 3)
    gt_world_pos.append(gt_world_pos[-1] + last_vel)

    return gt_world_pos, gt_stress


# ─────────────────────────────────────────────────────────────────────────────
# Main inference class
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticRollout:
    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        self.cfg    = cfg
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # ── 1. Load trained model ─────────────────────────────────────────────
        self.model = HybridMeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            processor_size=cfg.processor_size,
            hidden_dim_processor=cfg.hidden_dim,
            hidden_dim_node_encoder=cfg.hidden_dim,
            hidden_dim_edge_encoder=cfg.hidden_dim,
            hidden_dim_node_decoder=cfg.hidden_dim,
            mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        ).to(self.device)
        self.model.eval()

        load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )
        logger.info("Model loaded.")

        # ── 2. Normalization stats ────────────────────────────────────────────
        node_stats = load_json("node_stats.json")
        self.v_mean = node_stats["velocity_mean"].to(self.device)
        self.v_std  = node_stats["velocity_std"].to(self.device)
        self.s_mean = node_stats["stress_mean"].to(self.device)
        self.s_std  = node_stats["stress_std"].to(self.device)

        # CPU copies for GT loading
        self.v_mean_cpu = self.v_mean.cpu()
        self.v_std_cpu  = self.v_std.cpu()
        self.s_mean_cpu = self.s_mean.cpu()
        self.s_std_cpu  = self.s_std.cpu()

        edge_stats = load_json("edge_stats.json")
        # edge_stats may be 8-dim (ref+world concatenated); our compute_edge_features
        # returns 4-dim, so use only the first 4 elements for normalization.
        self.e_mean = edge_stats["edge_mean"].to(self.device)[:4]
        self.e_std  = edge_stats["edge_std"].to(self.device)[:4]

        # ── 3. Static mesh topology (shared across all samples) ───────────────
        # Load from sample_00000 to get mesh structure
        first_path = os.path.join(
            to_absolute_path(cfg.preprocess_output_dir), "sample_00000.pt"
        )
        first_data = torch.load(first_path, map_location="cpu", weights_only=False)

        # graph.edge_index may contain mesh + world edges concatenated.
        # mesh_edge_features tells us exactly how many mesh edges there are.
        E_mesh = first_data[0]["mesh_edge_features"].shape[0]
        self.mesh_ei = first_data[0]["graph"].edge_index[:, :E_mesh].to(self.device)  # (2, E_mesh)

        # ref_pos = undeformed mesh positions stored during data generation
        self.ref_pos = first_data[0]["graph"].mesh_pos.to(self.device)    # (N, 3)
        ref_feat = compute_edge_features(self.ref_pos, self.mesh_ei)
        self.ref_feat_norm = (ref_feat - self.e_mean) / self.e_std        # (E, 4)

        self.empty_world_ef = torch.zeros(0, cfg.num_edge_features, device=self.device)

        # t=0 pair is excluded from .pt files (identical input across all samples).
        # Each file stores T-2 pairs (t=1..T-2). Rollout: seed=t1, predict t=2..T-1.
        self.rollout_steps = len(first_data)   # = T-2 = 19

        N = self.ref_pos.shape[0]
        n_axis = int(round(N ** (1 / 3)))
        self.hex_cells = build_hex_cells(n=n_axis)

        logger.info(
            f"seed=t1 (ground truth), rollout={self.rollout_steps} steps "
            f"(t=2..t={self.rollout_steps + 1})"
        )

        # Determine which samples to run
        data_dir = to_absolute_path(cfg.preprocess_output_dir)
        all_files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("sample_") and f.endswith(".pt")
        )

        indices = list(cfg.infer_sample_indices)
        if indices and indices[0] is not None:
            self.sample_files = [all_files[i] for i in indices if i < len(all_files)]
        else:
            self.sample_files = all_files   # run all

        logger.info(
            f"Inference on {len(self.sample_files)} sample(s): "
            + ", ".join(os.path.basename(f) for f in self.sample_files)
        )

    # ── Build graph for a given world_pos ────────────────────────────────────
    def _make_graph(self, wp: torch.Tensor):
        world_feat   = compute_edge_features(wp, self.mesh_ei)
        world_feat_n = (world_feat - self.e_mean) / self.e_std
        mesh_ef      = torch.cat([self.ref_feat_norm, world_feat_n], dim=1)  # (E, 8)

        node_x = wp - self.ref_pos                                             # (N, 3)

        graph = Data(
            x          = node_x,
            edge_index = self.mesh_ei,
            edge_attr  = mesh_ef,
            world_pos  = wp,
        )
        return graph, mesh_ef

    # ── Single-sample rollout ─────────────────────────────────────────────────
    @torch.inference_mode()
    def rollout_one(self, sample_path: str):
        """
        Run autoregressive rollout for one sample.

        Seed: ground truth world_pos at t=1.
        Predicts: world_pos and stress at t=2 .. T-1.

        Returns:
          pred_wp   : list of (N, 3) cpu tensors, length = rollout_steps   (t=2..T-1)
          pred_str  : list of (N, 1) cpu tensors, length = rollout_steps
          gt_wp     : list of (N, 3) cpu tensors, length = rollout_steps   (ground truth)
          gt_str    : list of (N, 1) cpu tensors, length = rollout_steps
        """
        gt_world_pos, gt_stress = load_gt_trajectory(
            sample_path,
            self.v_mean_cpu, self.v_std_cpu,
            self.s_mean_cpu, self.s_std_cpu,
        )
        # .pt file now starts from t=1 (t=0 pair excluded during data generation):
        #   gt_world_pos[k] = world_pos at t=k+1,  k=0..T-2   (len = T-1)
        #   gt_stress[k]    = stress    at t=k+1,  k=0..T-3   (len = T-2)

        # Seed: ground truth at t=1 → index 0 in new format
        current_wp = gt_world_pos[0].to(self.device)   # (N, 3)

        pred_wp  = []
        pred_str = []
        gt_wp    = []
        gt_str   = []

        for step in range(self.rollout_steps):
            # step=0: input=t1, predict vel[1→2]+stress[1], output wp at t=2
            # step=k: input=t(k+1), predict vel[(k+1)→(k+2)]+stress[k+1], output wp at t=k+2
            graph, mesh_ef = self._make_graph(current_wp)

            pred_norm   = self.model(graph.x, mesh_ef, self.empty_world_ef, graph)
            pred_vel    = pred_norm[:, :3] * self.v_std + self.v_mean   # (N, 3)
            pred_stress = pred_norm[:, 3:] * self.s_std + self.s_mean   # (N, 1)

            current_wp = current_wp + pred_vel

            pred_wp.append(current_wp.cpu())
            pred_str.append(pred_stress.cpu())

            # gt at t=k+2 → index k+1;  gt_stress at t=k+1 → index k
            gt_wp.append(gt_world_pos[step + 1])
            gt_str.append(gt_stress[step])

        return pred_wp, pred_str, gt_wp, gt_str

    # ── Run all samples ───────────────────────────────────────────────────────
    def run_all(
        self,
        output_dir: str      = "paraview_output_synthetic",
        num_export_samples: int = 5,
    ):
        """
        Rollout all samples, collect per-step errors, export VTU for the first
        num_export_samples samples.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Accumulators: shape (rollout_steps,) after stacking
        all_pos_err    = []   # list over samples of (rollout_steps,) arrays
        all_stress_err = []

        for s_idx, sample_path in enumerate(self.sample_files):
            pred_wp, pred_str, gt_wp, gt_str = self.rollout_one(sample_path)

            # Per-step error for this sample
            pos_err_steps    = []
            stress_err_steps = []
            for k in range(self.rollout_steps):
                pe = np.linalg.norm(
                    pred_wp[k].numpy() - gt_wp[k].numpy(), axis=1
                ).mean()
                se = np.abs(
                    pred_str[k].numpy() - gt_str[k].numpy()
                ).mean()
                pos_err_steps.append(pe)
                stress_err_steps.append(se)

            all_pos_err.append(pos_err_steps)
            all_stress_err.append(stress_err_steps)

            # VTU export for first num_export_samples
            if s_idx < num_export_samples:
                self._export_vtu(
                    sample_idx = s_idx,
                    pred_wp    = pred_wp,
                    pred_str   = pred_str,
                    gt_wp      = gt_wp,
                    gt_str     = gt_str,
                    output_dir = output_dir,
                )

            if s_idx % 10 == 0 or s_idx == len(self.sample_files) - 1:
                final_pe = pos_err_steps[-1]
                final_se = stress_err_steps[-1]
                self.logger.info(
                    f"  sample {s_idx+1:3d}/{len(self.sample_files)} | "
                    f"final pos err: {final_pe:.3e} | "
                    f"final stress err: {final_se:.3e}"
                )

        # ── Aggregate statistics ─────────────────────────────────────────────
        pos_arr    = np.array(all_pos_err)    # (S, rollout_steps)
        stress_arr = np.array(all_stress_err) # (S, rollout_steps)

        self.logger.info("\n=== Per-step error (mean ± std over all samples) ===")
        self.logger.info(f"{'t':>4}  {'pos_mean':>10}  {'pos_std':>10}  "
                         f"{'stress_mean':>12}  {'stress_std':>12}")
        csv_lines = ["t,pos_mean,pos_std,stress_mean,stress_std"]
        for k in range(self.rollout_steps):
            t_val = k + 2
            pm    = pos_arr[:, k].mean()
            ps    = pos_arr[:, k].std()
            sm    = stress_arr[:, k].mean()
            ss    = stress_arr[:, k].std()
            self.logger.info(
                f"  t={t_val:2d}  {pm:10.3e}  {ps:10.3e}  {sm:12.3e}  {ss:12.3e}"
            )
            csv_lines.append(f"{t_val},{pm:.6e},{ps:.6e},{sm:.6e},{ss:.6e}")

        # Save CSV
        csv_path = os.path.join(output_dir, "error_per_step.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_lines))
        self.logger.info(f"\nError statistics saved to {csv_path}")

        # Final step summary
        self.logger.info(
            f"\n=== Final step (t={self.rollout_steps + 1}) summary over {len(self.sample_files)} samples ===\n"
            f"  Position error — mean: {pos_arr[:, -1].mean():.3e}, "
            f"max: {pos_arr[:, -1].max():.3e}\n"
            f"  Stress   error — mean: {stress_arr[:, -1].mean():.3e}, "
            f"max: {stress_arr[:, -1].max():.3e}"
        )

    # ── VTU export for one sample ─────────────────────────────────────────────
    def _export_vtu(
        self,
        sample_idx: int,
        pred_wp: list,
        pred_str: list,
        gt_wp: list,
        gt_str: list,
        output_dir: str,
    ):
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        cells = [("hexahedron", self.hex_cells)]

        for k in range(self.rollout_steps):
            # step_000 = t=2, step_001 = t=3, ...
            pred_pts  = pred_wp[k].numpy()        # (N, 3)
            exact_pts = gt_wp[k].numpy()
            pred_s    = pred_str[k].numpy().squeeze(1)   # (N,)
            exact_s   = gt_str[k].numpy().squeeze(1)

            ref_pos_np = self.ref_pos.cpu().numpy()
            pred_disp  = np.linalg.norm(pred_pts  - ref_pos_np, axis=1)
            exact_disp = np.linalg.norm(exact_pts - ref_pos_np, axis=1)
            pos_err    = np.linalg.norm(pred_pts - exact_pts, axis=1)
            str_err    = np.abs(pred_s - exact_s)

            meshio.Mesh(
                points=pred_pts, cells=cells,
                point_data={
                    "Displacement_Mag": pred_disp,
                    "Stress"          : pred_s,
                    "Position_Error"  : pos_err,
                    "Stress_Error"    : str_err,
                },
            ).write(os.path.join(sample_dir, f"step_{k:03d}_pred.vtu"))

            meshio.Mesh(
                points=exact_pts, cells=cells,
                point_data={
                    "Displacement_Mag": exact_disp,
                    "Stress"          : exact_s,
                },
            ).write(os.path.join(sample_dir, f"step_{k:03d}_exact.vtu"))


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("infer_synthetic")
    logger.file_logging()
    logger.info("Synthetic rollout inference started.")

    rollout = SyntheticRollout(cfg, logger)
    rollout.run_all(
        output_dir         = "paraview_output_synthetic",
        num_export_samples = len(rollout.sample_files),   # export VTU for all samples
    )

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
