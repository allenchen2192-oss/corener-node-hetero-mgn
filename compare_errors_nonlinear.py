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
compare_errors_nonlinear.py
============================
Rollout evaluation for the nonlinear synthetic dataset.
Plots displacement error and stress error for training vs test samples.

Outputs:
  error_nonlinear_train_vs_test.png  — two-panel figure (disp + stress)
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


# ── Config ────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 6    # x_t(3) + prev_vel(3)
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3    # velocity only (stress supervised analytically)
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 64

DATA_DIR  = "./04_preprocessed_pt_nonlinear_prevvel"
CKPT_PATH  = "./checkpoints_nonlinear_prevvel"
CKPT_EPOCH = None   # set to None to load the latest epoch

TRAIN_INDICES = list(range(0, 50))
TEST_INDICES  = list(range(2000, 2100))


# ── Physics helpers ───────────────────────────────────────────────────────────

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


def compute_edge_features(pos, edge_index):
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def precompute_xtx_inv(ref_pos, mesh_ei):
    """Precompute XtX^{-1} for scatter stress — identical to train.py formula."""
    N = ref_pos.shape[0]
    row, col = mesh_ei
    r_ij = ref_pos[col] - ref_pos[row]                        # (E, 3)
    r_r  = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)           # (E, 3, 3)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    XtX = torch.zeros(N, 3, 3, device=ref_pos.device)
    XtX.scatter_add_(0, row_exp, r_r)
    XtX = XtX + 1e-6 * torch.eye(3, device=ref_pos.device).unsqueeze(0)
    return torch.linalg.inv(XtX), r_ij, row, col


def compute_vm_stress(wp, ref, XtX_inv, r_ij, row, col):
    """Per-node von Mises stress via scatter fit (same as train.py)."""
    u    = wp - ref
    du_ij = u[col] - u[row]                                   # (E, 3)
    r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)         # (E, 3, 3)
    row_exp = row.view(-1, 1, 1).expand_as(r_du)
    XtY = torch.zeros(wp.shape[0], 3, 3, device=wp.device)
    XtY.scatter_add_(0, row_exp, r_du)
    dU = torch.bmm(XtX_inv, XtY).permute(0, 2, 1)            # (N, 3, 3)
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy  = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz  = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                              + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)


# ── Global setup ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean     = node_stats["velocity_mean"].to(device)
v_std      = node_stats["velocity_std"].to(device)
s_mean_val = node_stats["stress_mean"].item()
s_std_val  = node_stats["stress_std"].item()
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean = edge_stats["edge_mean"].to(device)[:4]
e_std  = edge_stats["edge_std"].to(device)[:4]

# Load mesh topology from first sample
first    = torch.load(f"{DATA_DIR}/sample_00000.pt", map_location="cpu", weights_only=False)
E_mesh   = first[0]["mesh_edge_features"].shape[0]
mesh_ei  = first[0]["graph"].edge_index[:, :E_mesh].to(device)
ref_pos  = first[0]["graph"].mesh_pos.to(device)
ref_pos_np = ref_pos.cpu().numpy()

ref_feat_n    = (compute_edge_features(ref_pos, mesh_ei) - e_mean) / e_std
empty_wef     = torch.zeros(0, NUM_EDGE_FEATURES, device=device)
rollout_steps = len(first)
t_axis        = np.arange(1, rollout_steps + 1)

XtX_inv, r_ij, row_ei, col_ei = precompute_xtx_inv(ref_pos, mesh_ei)

print(f"[mesh] N={ref_pos.shape[0]}, rollout_steps={rollout_steps}")


def make_graph(wp, prev_vel):
    wf = (compute_edge_features(wp, mesh_ei) - e_mean) / e_std
    ef = torch.cat([ref_feat_n, wf], dim=1)
    x  = torch.cat([wp - ref_pos, prev_vel], dim=1)   # (N, 6)
    return Data(x=x, edge_index=mesh_ei, edge_attr=ef), ef


# ── Rollout ───────────────────────────────────────────────────────────────────

@torch.inference_mode()
def rollout_one(model, sample_path):
    data = torch.load(sample_path, map_location="cpu", weights_only=False)

    # Reconstruct GT world positions: t0 is the world_pos of the first stored step
    gt_wp = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp.append(gt_wp[-1] + vel_gt)

    # GT stress per timestep (denormalized)
    gt_stress = [
        (step["graph"].y[:, 3] * s_std_val + s_mean_val).numpy()
        for step in data
    ]   # list of (N,) arrays

    cur      = gt_wp[0].to(device)
    prev_vel = data[0]["graph"].x[:, 3:].to(device)   # initial prev_vel from dataset
    disp_errs   = []
    stress_errs = []

    for k in range(rollout_steps):
        g, ef = make_graph(cur, prev_vel)
        out      = model(g.x, ef, empty_wef, g)
        vel      = out[:, :3] * v_std + v_mean
        prev_vel = vel                             # update for next step
        cur      = cur + vel

        pred_np = cur.cpu().numpy()
        gt_np   = gt_wp[k + 1].numpy()

        # Displacement error (% of GT displacement magnitude)
        num = np.linalg.norm(pred_np - gt_np, axis=1).sum()
        den = np.linalg.norm(gt_np - ref_pos_np, axis=1).sum() + 1e-30
        disp_errs.append(num / den * 100.0)

        # Stress error: scatter-fit stress on predicted pos vs GT scatter-fit stress
        pred_stress = compute_vm_stress(
            cur, ref_pos, XtX_inv, r_ij, row_ei, col_ei
        ).cpu().numpy()
        gt_s = gt_stress[k]
        stress_errs.append(
            np.abs(pred_stress - gt_s).mean() / (gt_s.mean() + 1e-30) * 100.0
        )

    return np.array(disp_errs), np.array(stress_errs)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    m = HybridMeshGraphNet(
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
    m.eval()
    return m


def get_sample_files(indices):
    all_files = sorted(
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith("sample_") and f.endswith(".pt")
    )
    return [all_files[i] for i in indices if i < len(all_files)]


# ── Main ──────────────────────────────────────────────────────────────────────

epoch_tag = f"_ep{CKPT_EPOCH}" if CKPT_EPOCH is not None else "_latest"
print(f"\nLoading checkpoint: {CKPT_PATH}  epoch={CKPT_EPOCH}")
model = build_model()
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)

results = {}
for split, indices in [("train", TRAIN_INDICES), ("test", TEST_INDICES)]:
    files = get_sample_files(indices)
    print(f"\n[{split}] {len(files)} samples")
    disp_list, stress_list = [], []
    for f in files:
        d, s = rollout_one(model, f)
        disp_list.append(d)
        stress_list.append(s)
        print(f"  {os.path.basename(f)}: "
              f"final disp={d[-1]:.2f}%  stress={s[-1]:.2f}%")
    results[split] = {
        "disp":   np.stack(disp_list,   axis=0),   # (10, T)
        "stress": np.stack(stress_list, axis=0),   # (10, T)
    }
    print(f"  → mean final disp={results[split]['disp'][:, -1].mean():.2f}%  "
          f"stress={results[split]['stress'][:, -1].mean():.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"train": "tab:blue", "test": "tab:green"}

for split in ["train", "test"]:
    d  = results[split]["disp"]
    s  = results[split]["stress"]
    c  = colors[split]
    lbl = split.capitalize()

    axes[0].plot(t_axis, d.mean(axis=0), color=c, linewidth=2, label=f"{lbl} mean")
    axes[0].fill_between(t_axis, d.min(axis=0), d.max(axis=0), color=c, alpha=0.15,
                         label=f"{lbl} min–max")

    axes[1].plot(t_axis, s.mean(axis=0), color=c, linewidth=2, label=f"{lbl} mean")
    axes[1].fill_between(t_axis, s.min(axis=0), s.max(axis=0), color=c, alpha=0.15,
                         label=f"{lbl} min–max")

for ax, title, ylabel in [
    (axes[0], "Displacement Error — Train vs Test", "Displacement Error (%)"),
    (axes[1], "Stress Error — Train vs Test",       "Stress Error (% of mean GT)"),
]:
    ax.set_xlabel("Rollout step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = f"error_nonlinear_prevvel_train_vs_test{epoch_tag}.png"
plt.savefig(out_path, dpi=150)
print(f"\n[saved] {out_path}")
