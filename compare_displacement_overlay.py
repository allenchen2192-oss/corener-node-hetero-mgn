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
compare_displacement_overlay.py
================================
Overlays displacement error curves (mean only, no std) from multiple
experiments onto a single plot for easy comparison.
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
# Experiment configs  (adjust paths/epochs here)
# ─────────────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 3
NUM_EDGE_FEATURES   = 8
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 64

DATA_DIR = "./04_preprocessed_pt"
OOD_DIR  = "./04_preprocessed_pt_ood"

EXPERIMENTS = [
    {
        "label":            "1000 data / 1000 epoch",
        "ckpt_path":        "./checkpoints_h64_1k_1000data_5000epoch",
        "ckpt_epoch":       1000,
        "num_out_features": 4,
        "train_idx":        list(range(0,    10)),
        "test_idx":         list(range(2000, 2010)),
        "ood_idx":          list(range(0,    10)),
        "linestyle":        "solid",
        "marker":           "o",
    },
    {
        "label":            "1000 data / 5000 epoch",
        "ckpt_path":        "./checkpoints_h64_1k_1000data_5000epoch",
        "ckpt_epoch":       5000,
        "num_out_features": 4,
        "train_idx":        list(range(0,    10)),
        "test_idx":         list(range(2000, 2010)),
        "ood_idx":          list(range(0,    10)),
        "linestyle":        "solid",
        "marker":           "s",
    },
    {
        "label":            "2000 data / 1000 epoch",
        "ckpt_path":        "./checkpoints_h64_direct_2k_5000epoch",
        "ckpt_epoch":       1000,
        "num_out_features": 4,
        "train_idx":        list(range(0,    10)),
        "test_idx":         list(range(2000, 2010)),
        "ood_idx":          list(range(0,    10)),
        "linestyle":        "dashed",
        "marker":           "^",
    },
    {
        "label":            "2000 data / 5000 epoch",
        "ckpt_path":        "./checkpoints_h64_direct_2k_5000epoch",
        "ckpt_epoch":       5000,
        "num_out_features": 4,
        "train_idx":        list(range(0,    10)),
        "test_idx":         list(range(2000, 2010)),
        "ood_idx":          list(range(0,    10)),
        "linestyle":        "dashed",
        "marker":           "D",
    },
]

# Color per curve type (consistent with original compare_errors.py)
COLORS = {"train": "tab:blue", "test": "tab:green", "ood": "tab:red"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

# Load stats and mesh topology once (same across all experiments)
node_stats  = load_json("node_stats.json")
v_mean      = node_stats["velocity_mean"].to(device)
v_std       = node_stats["velocity_std"].to(device)
v_mean_cpu  = v_mean.cpu()
v_std_cpu   = v_std.cpu()

edge_stats  = load_json("edge_stats.json")
e_mean      = edge_stats["edge_mean"].to(device)[:4]
e_std       = edge_stats["edge_std"].to(device)[:4]

first       = torch.load(f"{DATA_DIR}/sample_00000.pt", map_location="cpu", weights_only=False)
E_mesh      = first[0]["mesh_edge_features"].shape[0]
mesh_ei     = first[0]["graph"].edge_index[:, :E_mesh].to(device)
ref_pos     = first[0]["graph"].mesh_pos.to(device)
ref_pos_np  = ref_pos.cpu().numpy()
ref_feat_n  = (compute_edge_features(ref_pos, mesh_ei) - e_mean) / e_std
empty_wef   = torch.zeros(0, NUM_EDGE_FEATURES, device=device)
rollout_steps = len(first)
t_axis      = np.arange(2, rollout_steps + 2)

print(f"[mesh] N={ref_pos.shape[0]}, rollout_steps={rollout_steps}")


def make_graph(wp):
    wf = (compute_edge_features(wp, mesh_ei) - e_mean) / e_std
    ef = torch.cat([ref_feat_n, wf], dim=1)
    return Data(x=wp - ref_pos, edge_index=mesh_ei, edge_attr=ef, world_pos=wp), ef


@torch.inference_mode()
def rollout_one(model, sample_path):
    data = torch.load(sample_path, map_location="cpu", weights_only=False)
    gt_wp = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp.append(gt_wp[-1] + vel)

    cur = gt_wp[0].to(device)
    pos_pct = []
    for k in range(rollout_steps):
        g, ef = make_graph(cur)
        out   = model(g.x, ef, empty_wef, g)
        vel   = out[:, :3] * v_std + v_mean
        cur   = cur + vel
        pred_np = cur.cpu().numpy()
        gt_np   = gt_wp[k + 1].numpy()
        num = np.linalg.norm(pred_np - gt_np, axis=1).sum()
        den = np.linalg.norm(gt_np - ref_pos_np, axis=1).sum()
        pos_pct.append(num / den * 100.0)

    return np.array(pos_pct)


def build_model(num_output_features):
    m = HybridMeshGraphNet(
        NUM_INPUT_FEATURES, NUM_EDGE_FEATURES, num_output_features,
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


def get_files(data_dir, indices):
    all_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("sample_") and f.endswith(".pt")
    )
    return [all_files[i] for i in indices if i < len(all_files)]


# ─────────────────────────────────────────────────────────────────────────────
# Run all experiments
# ─────────────────────────────────────────────────────────────────────────────

results = {}   # exp_label -> {train, test, ood} -> mean pos_pct array

for exp in EXPERIMENTS:
    print(f"\n[{exp['label']}] loading checkpoint epoch={exp['ckpt_epoch']} ...")
    model = build_model(exp["num_out_features"])
    load_checkpoint(exp["ckpt_path"], models=model, device=device, epoch=exp["ckpt_epoch"])

    exp_result = {}
    for split, indices, data_dir in [
        ("train", exp["train_idx"], DATA_DIR),
        ("test",  exp["test_idx"],  DATA_DIR),
        ("ood",   exp["ood_idx"],   OOD_DIR),
    ]:
        files = get_files(data_dir, indices)
        print(f"  {split}: {len(files)} samples")
        arr = np.stack([rollout_one(model, f) for f in files], axis=0)
        exp_result[split] = arr.mean(axis=0)
        print(f"    final pos err: {arr[:, -1].mean():.2f}%")

    results[exp["label"]] = exp_result
    del model
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

for exp in EXPERIMENTS:
    label = exp["label"]
    ls    = exp["linestyle"]
    mk    = exp["marker"]
    for split, split_label in [("train", "Training"), ("test", "Test"), ("ood", "OOD")]:
        color = COLORS[split]
        mean  = results[label][split]
        ax.plot(
            t_axis, mean,
            color=color, linestyle=ls, linewidth=2,
            marker=mk, markersize=5, markevery=3,
            label=f"{split_label} — {label}",
        )

ax.set_xlabel("Time step t", fontsize=12)
ax.set_ylabel("Displacement Error (%)", fontsize=12)
ax.set_title("Displacement Error Comparison — 1k vs 2k data × 1k vs 5k epoch (mean over 10 samples)", fontsize=11)
ax.set_xlim(t_axis[0], t_axis[-1])
ax.grid(True, alpha=0.3)

# Split legend into two columns for readability
ax.legend(fontsize=8, ncol=2, loc="upper left")

plt.tight_layout()
out_path = "displacement_overlay_1k2k_data_1k5k_epoch.png"
plt.savefig(out_path, dpi=150)
print(f"\n[saved] {out_path}")
