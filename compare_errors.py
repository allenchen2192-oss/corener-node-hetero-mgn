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
compare_errors.py
=================
比較三組資料的推論誤差百分比（依圖示公式）：

  pos error(t)    = sum_l ||p_hat_l(t) - p_l(t)||_2 / sum_l ||p_l(t)||_2 * 100%
  stress error(t) = sum_l |s_hat_l(t) - s_l(t)|    / sum_l |s_l(t)|     * 100%

三組資料：
  - Training data : 04_preprocessed_pt,     samples 0–9
  - Test data     : 04_preprocessed_pt,     samples 1000–1009
  - OOD data      : 04_preprocessed_pt_ood, samples 0–9

輸出:
  error_comparison.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Config (mirrors conf/config.yaml)
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PATH          = "./checkpoints_h64_analytical_wstress01"
CKPT_EPOCH         = None   # None = latest saved checkpoint
NUM_INPUT_FEATURES = 3
NUM_EDGE_FEATURES  = 8
NUM_OUTPUT_FEATURES= 3      # model outputs velocity only (stress is analytical)
PROCESSOR_SIZE     = 4
HIDDEN_DIM         = 64

DATASETS = [
    ("Training (in-dist)",  "./04_preprocessed_pt",     list(range(0,   10))),
    ("Test (in-dist)",      "./04_preprocessed_pt",     list(range(2000,2010))),
    ("OOD (out-of-dist)",   "./04_preprocessed_pt_ood", list(range(0,   10))),
]

COLORS = ["tab:blue", "tab:green", "tab:red"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (same as infer_synthetic.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def load_gt_trajectory(sample_path, v_mean, v_std, s_mean, s_std):
    sample_data = torch.load(sample_path, map_location="cpu", weights_only=False)
    gt_world_pos, gt_stress = [], []
    for step in sample_data:
        g = step["graph"]
        gt_world_pos.append(g.world_pos.clone())
        gt_stress.append(g.y[:, 3:] * s_std + s_mean)
    last_vel = sample_data[-1]["graph"].y[:, :3] * v_std + v_mean
    gt_world_pos.append(gt_world_pos[-1] + last_vel)
    return gt_world_pos, gt_stress


def compute_vm_stress_per_node_np(world_pos_np: np.ndarray, ref_pos_np: np.ndarray,
                                   edge_index_np: np.ndarray,
                                   E_mod: float = 70e9, nu: float = 0.33,
                                   reg: float = 1e-6) -> np.ndarray:
    """
    Per-node von Mises stress via local neighborhood gradient fit.
    For each node i, builds normal equations from mesh neighbors and solves a
    3x3 system — no global assumption, captures stress concentrations.
    Returns (N,) per-node von Mises stress.
    """
    N   = ref_pos_np.shape[0]
    u   = world_pos_np - ref_pos_np                       # (N, 3)
    row, col = edge_index_np                              # (E,) each

    r_ij  = ref_pos_np[col] - ref_pos_np[row]            # (E, 3)
    du_ij = u[col] - u[row]                              # (E, 3)

    # Outer products per edge
    r_r  = r_ij[:, :, None] * r_ij[:, None, :]           # (E, 3, 3)  r⊗r
    r_du = r_ij[:, :, None] * du_ij[:, None, :]          # (E, 3, 3)  r⊗du

    # Scatter-accumulate normal equations per node
    XtX = np.zeros((N, 3, 3), dtype=np.float64)
    XtY = np.zeros((N, 3, 3), dtype=np.float64)
    np.add.at(XtX, row, r_r)
    np.add.at(XtY, row, r_du)

    # Tikhonov regularization (stabilizes boundary nodes with few neighbors)
    XtX += reg * np.eye(3, dtype=np.float64)[None]

    # Solve N independent 3×3 systems; dU[n,k,l] = ∂u_k/∂x_l
    dU = np.linalg.solve(XtX, XtY).transpose(0, 2, 1)   # (N, 3, 3)

    lame1 = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E_mod / (2 * (1 + nu))
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = lame1 * tr + 2 * mu * dU[:, 0, 0]
    sy  = lame1 * tr + 2 * mu * dU[:, 1, 1]
    sz  = lame1 * tr + 2 * mu * dU[:, 2, 2]
    txy = mu * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = mu * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = mu * (dU[:, 1, 2] + dU[:, 2, 1])
    vm  = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                          + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)
    return vm.astype(np.float32)                         # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Load model + stats once
# ─────────────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

model = HybridMeshGraphNet(
    NUM_INPUT_FEATURES, NUM_EDGE_FEATURES, NUM_OUTPUT_FEATURES,
    processor_size=PROCESSOR_SIZE,
    hidden_dim_processor=HIDDEN_DIM,
    hidden_dim_node_encoder=HIDDEN_DIM,
    hidden_dim_edge_encoder=HIDDEN_DIM,
    hidden_dim_node_decoder=HIDDEN_DIM,
    mlp_activation_fn="relu",
    do_concat_trick=False,
    num_processor_checkpoint_segments=0,
    recompute_activation=False,
).to(device)
model.eval()
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)
print("[model] loaded")

node_stats = load_json("node_stats.json")
v_mean = node_stats["velocity_mean"].to(device)
v_std  = node_stats["velocity_std"].to(device)
s_mean = node_stats["stress_mean"].to(device)
s_std  = node_stats["stress_std"].to(device)
v_mean_cpu = v_mean.cpu(); v_std_cpu = v_std.cpu()
s_mean_cpu = s_mean.cpu(); s_std_cpu = s_std.cpu()

edge_stats = load_json("edge_stats.json")
e_mean = edge_stats["edge_mean"].to(device)[:4]
e_std  = edge_stats["edge_std"].to(device)[:4]

# Static mesh topology from first training sample
first = torch.load("./04_preprocessed_pt/sample_00000.pt", map_location="cpu", weights_only=False)
E_mesh  = first[0]["mesh_edge_features"].shape[0]
mesh_ei = first[0]["graph"].edge_index[:, :E_mesh].to(device)
ref_pos = first[0]["graph"].mesh_pos.to(device)
ref_feat_norm = (compute_edge_features(ref_pos, mesh_ei) - e_mean) / e_std
empty_world_ef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)
rollout_steps = len(first)   # = T-2 = 19

print(f"[mesh] N={ref_pos.shape[0]}, rollout_steps={rollout_steps}")


# ─────────────────────────────────────────────────────────────────────────────
# Rollout helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_graph(wp):
    wf = (compute_edge_features(wp, mesh_ei) - e_mean) / e_std
    ef = torch.cat([ref_feat_norm, wf], dim=1)
    return Data(x=wp - ref_pos, edge_index=mesh_ei, edge_attr=ef, world_pos=wp), ef


@torch.inference_mode()
def rollout_one(sample_path):
    gt_wp, _ = load_gt_trajectory(
        sample_path, v_mean_cpu, v_std_cpu, s_mean_cpu, s_std_cpu
    )
    cur = gt_wp[0].to(device)
    pred_wp_list, gt_wp_list = [], []
    for step in range(rollout_steps):
        g, ef = make_graph(cur)
        out = model(g.x, ef, empty_world_ef, g)
        vel = out[:, :3] * v_std + v_mean
        cur = cur + vel
        pred_wp_list.append(cur.cpu())
        gt_wp_list.append(gt_wp[step + 1])
    return pred_wp_list, gt_wp_list


ref_pos_np  = ref_pos.cpu().numpy()       # (N, 3) undeformed mesh positions
mesh_ei_np  = mesh_ei.cpu().numpy()       # (2, E) mesh edge connectivity


def relative_errors(pred_wp_list, gt_wp_list):
    """
    Returns pos_pct, stress_pct: arrays of shape (rollout_steps,)

    pos_pct(t)    = sum_l ||u_hat_l - u_l||_2 / sum_l ||u_l||_2 * 100
    stress_pct(t) = |sigma_vm_hat - sigma_vm| / sigma_vm * 100
      where sigma_vm is computed analytically from the fitted displacement gradient.
    """
    pos_pct, stress_pct = [], []
    for k in range(rollout_steps):
        p = pred_wp_list[k].numpy()   # (N, 3)
        g = gt_wp_list[k].numpy()

        # displacement error
        num_pos = np.linalg.norm(p - g, axis=1).sum()
        den_pos = np.linalg.norm(g - ref_pos_np, axis=1).sum()
        pos_pct.append(num_pos / den_pos * 100.0)

        # per-node stress from local displacement gradient fit
        ps = compute_vm_stress_per_node_np(p, ref_pos_np, mesh_ei_np)   # (N,)
        gs = compute_vm_stress_per_node_np(g, ref_pos_np, mesh_ei_np)   # (N,)
        num_s = np.abs(ps - gs).sum()
        den_s = np.abs(gs).sum() + 1e-30
        stress_pct.append(num_s / den_s * 100.0)

    return np.array(pos_pct), np.array(stress_pct)


# ─────────────────────────────────────────────────────────────────────────────
# Run inference for all three datasets
# ─────────────────────────────────────────────────────────────────────────────

results = {}   # label -> (pos_mean, pos_std, stress_mean, stress_std)

for label, data_dir, indices in DATASETS:
    all_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("sample_") and f.endswith(".pt")
    )
    files = [all_files[i] for i in indices if i < len(all_files)]
    print(f"\n[{label}] {len(files)} samples")

    pos_list, stress_list = [], []
    for sample_path in files:
        pw, gw = rollout_one(sample_path)
        pp, sp = relative_errors(pw, gw)
        pos_list.append(pp)
        stress_list.append(sp)

    pos_arr    = np.stack(pos_list,    axis=0)   # (S, T)
    stress_arr = np.stack(stress_list, axis=0)

    results[label] = (
        pos_arr.mean(axis=0), pos_arr.std(axis=0),
        stress_arr.mean(axis=0), stress_arr.std(axis=0),
    )
    print(f"  final pos err:    {pos_arr[:,-1].mean():.2f}% ± {pos_arr[:,-1].std():.2f}%")
    print(f"  final stress err: {stress_arr[:,-1].mean():.2f}% ± {stress_arr[:,-1].std():.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

t_axis = np.arange(2, rollout_steps + 2)   # t=2..20

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Rollout Error Comparison (mean ± std over 10 samples)", fontsize=13)

for ax, metric, ylabel in zip(
    axes,
    ["pos", "stress"],
    ["Displacement Error (%)", "Stress Error (%)"]
):
    for (label, _dd, _ii), color in zip(DATASETS, COLORS):
        pm, ps, sm, ss = results[label]
        arr_m = pm if metric == "pos" else sm
        arr_s = ps if metric == "pos" else ss
        ax.plot(t_axis, arr_m, color=color, label=label, linewidth=2)
        ax.fill_between(t_axis, arr_m - arr_s, arr_m + arr_s, color=color, alpha=0.15)

    ax.set_xlabel("Time step t", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_axis[0], t_axis[-1])

plt.tight_layout()
out_path = "error_comparison_noise_analytical_stress_wstress01.png"
plt.savefig(out_path, dpi=150)
print(f"\n[saved] {out_path}")
