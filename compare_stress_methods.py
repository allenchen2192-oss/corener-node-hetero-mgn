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
compare_stress_methods.py
=========================
在同一座標下比較兩種 Stress Error 的計算方式：
  1. Model stress    : 模型直接輸出的應力（stress head output）
  2. Gradient stress : 從預測位移梯度推導的 von Mises 應力

三組資料：Training (in-dist), Test (in-dist), OOD (out-of-dist)
實線  = Gradient-derived stress error
虛線  = Model-predicted stress error
陰影  = ± 1 std（同色系）

輸出：stress_method_comparison.png
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
# Config
# ─────────────────────────────────────────────────────────────────────────────

CKPT_PATH           = "./checkpoints_h64_1k_old"
NUM_INPUT_FEATURES  = 3
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 4
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 64

DATASETS = [
    ("Training (in-dist)",  "./04_preprocessed_pt",     list(range(0,   10))),
    ("Test (in-dist)",      "./04_preprocessed_pt",     list(range(1000, 1010))),
    ("OOD (out-of-dist)",   "./04_preprocessed_pt_ood", list(range(0,   10))),
]

COLORS = ["tab:blue", "tab:green", "tab:red"]


# ─────────────────────────────────────────────────────────────────────────────
# Load model + stats
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
load_checkpoint(CKPT_PATH, models=model, device=device)
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

first = torch.load("./04_preprocessed_pt/sample_00000.pt", map_location="cpu", weights_only=False)
E_mesh  = first[0]["mesh_edge_features"].shape[0]
mesh_ei = first[0]["graph"].edge_index[:, :E_mesh].to(device)
ref_pos = first[0]["graph"].mesh_pos.to(device)

ref_pos_np     = ref_pos.cpu().numpy()
rollout_steps  = len(first)   # T-2 = 19
empty_world_ef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)

print(f"[mesh] N={ref_pos.shape[0]}, rollout_steps={rollout_steps}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def make_graph(wp):
    ref_feat_norm = (compute_edge_features(ref_pos, mesh_ei) - e_mean) / e_std
    wf = (compute_edge_features(wp, mesh_ei) - e_mean) / e_std
    ef = torch.cat([ref_feat_norm, wf], dim=1)
    return Data(x=wp - ref_pos, edge_index=mesh_ei, edge_attr=ef, world_pos=wp), ef


def load_gt_trajectory(sample_path):
    sample_data = torch.load(sample_path, map_location="cpu", weights_only=False)
    gt_world_pos, gt_stress = [], []
    for step in sample_data:
        g = step["graph"]
        gt_world_pos.append(g.world_pos.clone())
        gt_stress.append(g.y[:, 3:] * s_std_cpu + s_mean_cpu)
    last_vel = sample_data[-1]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
    gt_world_pos.append(gt_world_pos[-1] + last_vel)
    return gt_world_pos, gt_stress


def vm_stress_from_disp(world_pos_np, E_mod=70e9, nu=0.33):
    """Fit linear displacement field → von Mises stress scalar (uniform over mesh)."""
    u = world_pos_np - ref_pos_np
    N = ref_pos_np.shape[0]
    X = np.hstack([np.ones((N, 1)), ref_pos_np])
    A_fit = np.linalg.lstsq(X, u, rcond=None)[0].T   # (3, 4)
    dU = A_fit[:, 1:]                                  # (3, 3)

    lame1 = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E_mod / (2 * (1 + nu))

    eps_xx = dU[0, 0]; eps_yy = dU[1, 1]; eps_zz = dU[2, 2]
    eps_xy = 0.5 * (dU[0, 1] + dU[1, 0])
    eps_xz = 0.5 * (dU[0, 2] + dU[2, 0])
    eps_yz = 0.5 * (dU[1, 2] + dU[2, 1])

    tr_eps = eps_xx + eps_yy + eps_zz
    sig_xx = lame1 * tr_eps + 2 * mu * eps_xx
    sig_yy = lame1 * tr_eps + 2 * mu * eps_yy
    sig_zz = lame1 * tr_eps + 2 * mu * eps_zz
    sig_xy = 2 * mu * eps_xy
    sig_xz = 2 * mu * eps_xz
    sig_yz = 2 * mu * eps_yz

    return float(np.sqrt(0.5 * (
        (sig_xx - sig_yy)**2 + (sig_yy - sig_zz)**2 + (sig_zz - sig_xx)**2
        + 6 * (sig_xy**2 + sig_xz**2 + sig_yz**2)
    )))


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def rollout_one(sample_path):
    gt_wp, gt_str = load_gt_trajectory(sample_path)
    cur = gt_wp[0].to(device)

    pred_wp_list   = []   # predicted world positions
    model_str_list = []   # model stress head output (denormalized)
    gt_wp_list     = []
    gt_str_list    = []

    for step in range(rollout_steps):
        g, ef = make_graph(cur)
        out   = model(g.x, ef, empty_world_ef, g)
        vel   = out[:, :3] * v_std + v_mean
        stress_model = (out[:, 3:] * s_std + s_mean).cpu()
        cur   = cur + vel

        pred_wp_list.append(cur.cpu())
        model_str_list.append(stress_model)
        gt_wp_list.append(gt_wp[step + 1])
        gt_str_list.append(gt_str[step])

    return pred_wp_list, model_str_list, gt_wp_list, gt_str_list


# ─────────────────────────────────────────────────────────────────────────────
# Error computation
# ─────────────────────────────────────────────────────────────────────────────

def stress_errors(pred_wp_list, model_str_list, gt_wp_list, gt_str_list):
    """
    Returns two arrays of shape (rollout_steps,):
      model_pct    : stress error from model's stress head
      gradient_pct : stress error from displacement gradient
    """
    model_pct, gradient_pct = [], []

    for k in range(rollout_steps):
        p  = pred_wp_list[k].numpy()
        g  = gt_wp_list[k].numpy()
        ps = model_str_list[k].numpy().squeeze(1)
        gs = gt_str_list[k].numpy().squeeze(1)

        # Method 1: model stress head
        num_m = np.abs(ps - gs).sum()
        den_m = np.abs(gs).sum() + 1e-30
        model_pct.append(num_m / den_m * 100.0)

        # Method 2: displacement gradient → von Mises
        vm_pred = vm_stress_from_disp(p)
        vm_gt   = vm_stress_from_disp(g)
        gradient_pct.append(abs(vm_pred - vm_gt) / (abs(vm_gt) + 1e-30) * 100.0)

    return np.array(model_pct), np.array(gradient_pct)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

results = {}   # label -> (model_mean, model_std, grad_mean, grad_std)

for label, data_dir, indices in DATASETS:
    all_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("sample_") and f.endswith(".pt")
    )
    files = [all_files[i] for i in indices if i < len(all_files)]
    print(f"\n[{label}] {len(files)} samples")

    model_list, grad_list = [], []
    for sample_path in files:
        pw, ms, gw, gs = rollout_one(sample_path)
        mp, gp = stress_errors(pw, ms, gw, gs)
        model_list.append(mp)
        grad_list.append(gp)

    model_arr = np.stack(model_list, axis=0)
    grad_arr  = np.stack(grad_list,  axis=0)

    results[label] = (
        model_arr.mean(axis=0), model_arr.std(axis=0),
        grad_arr.mean(axis=0),  grad_arr.std(axis=0),
    )
    print(f"  final model stress err:    {model_arr[:,-1].mean():.2f}% ± {model_arr[:,-1].std():.2f}%")
    print(f"  final gradient stress err: {grad_arr[:,-1].mean():.2f}%  ± {grad_arr[:,-1].std():.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

t_axis = np.arange(2, rollout_steps + 2)

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Stress Error Comparison: Model Output vs. Gradient-Derived", fontsize=12)

for (label, _dd, _ii), color in zip(DATASETS, COLORS):
    mm, ms, gm, gs = results[label]

    # gradient-derived: solid line
    ax.plot(t_axis, gm, color=color, linestyle="-",  linewidth=2,
            label=f"{label} — gradient")

    # model stress head: dashed line
    ax.plot(t_axis, mm, color=color, linestyle="--", linewidth=2,
            label=f"{label} — model output")

ax.set_xlabel("Time step t", fontsize=11)
ax.set_ylabel("Stress Error (%)", fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(t_axis[0], t_axis[-1])
ax.set_ylim(0, 100)

plt.tight_layout()
out_path = "stress_method_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"\n[saved] {out_path}")
