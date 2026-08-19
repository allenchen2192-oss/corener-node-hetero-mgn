"""
compare_errors_rollout.py
=========================
Rollout error vs timestep comparison plot (mean ± std over 100 samples).

Produces two subplots:
  - Displacement Error (%) vs time step
  - Stress Error (%) vs time step

For Train (in-dist), Test (in-dist), OOD (out-of-dist).

Usage:
  python compare_errors_rollout.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 8
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128

DATA_DIR   = "./04_preprocessed_pt_abaqus_prevvel"
CKPT_PATH  = "./checkpoints_abaqus_prevvel_1000data_1000nodes_h128_gfdm_stress_nodetype_dispnorm"
CKPT_EPOCH = None   # None = latest

NUM_TRAIN_SAMPLES = 100
NUM_TEST_SAMPLES  = 100
MAX_NODES_TRAIN   = 1000

OUTPUT_PLOT = "./rollout_error_comparison.png"


# ── Material constants ────────────────────────────────────────────────────────

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    row, col = edge_index
    d = pos[col] - pos[row]
    n = torch.norm(d, dim=1, keepdim=True)
    return torch.cat([d, n], dim=1)


def compute_vm_stress(wp, ref, mesh_ei):
    """Von Mises stress (Pa) via GFDM scatter linear fit."""
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    u     = wp.float() - ref.float()
    du_ij = u[col] - u[row]
    r_r   = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    N = wp.shape[0]
    XtX = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtY = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtX.scatter_add_(0, row_exp, r_r)
    XtY.scatter_add_(0, row_exp, r_du)
    XtX += 1e-6 * torch.eye(3, device=wp.device).unsqueeze(0)
    dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy  = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz  = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                              + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)


# ── Setup ─────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean = torch.tensor(node_stats["velocity_mean"], device=device)
v_std  = torch.tensor(node_stats["velocity_std"],  device=device)
d_mean = torch.tensor(node_stats["disp_mean"],     device=device)
d_std  = torch.tensor(node_stats["disp_std"],      device=device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref = torch.tensor(edge_stats["edge_mean"], device=device)[:4]
e_std_ref  = torch.tensor(edge_stats["edge_std"],  device=device)[:4]
e_mean_wld = torch.tensor(edge_stats["edge_mean"], device=device)[4:]
e_std_wld  = torch.tensor(edge_stats["edge_std"],  device=device)[4:]

empty_wef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)

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

ep_tag = f" epoch={CKPT_EPOCH}" if CKPT_EPOCH else " (latest)"
print(f"Loading checkpoint: {CKPT_PATH}{ep_tag}\n")
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)


# ── Rollout function ──────────────────────────────────────────────────────────

def run_sample(data_dir, fname):
    """
    Returns:
        disp_rel_per_t  : (T-1,) displacement error % per timestep (t=2..T)
        stress_rel_per_t: (T-1,) stress error % per timestep
    """
    data = torch.load(os.path.join(data_dir, fname),
                      map_location="cpu", weights_only=False)
    T = len(data)
    N = data[0]["graph"].num_nodes

    ref_pos      = data[0]["graph"].mesh_pos.to(device)
    mesh_ei      = data[0]["graph"].edge_index.to(device)
    node_type_oh = data[0]["graph"].x[:, 6:8].to(device)

    ref_feat_n = (compute_edge_features(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref

    # Build GT world-pos sequence
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    # GT max displacement magnitude at final step (for normalizing displacement error)
    gt_disp_mag_final = torch.norm(gt_wp_list[-1] - data[0]["graph"].mesh_pos, dim=1).max().item()
    if gt_disp_mag_final < 1e-10:
        return None

    # Rollout seeded at t=1
    gt_vel_0  = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur       = gt_wp_list[1].to(device)
    prev_vel  = gt_vel_0
    pred_wp_list = [gt_wp_list[0], gt_wp_list[1]]

    with torch.inference_mode():
        for _ in range(T - 1):
            world_feat    = (compute_edge_features(cur, mesh_ei) - e_mean_wld) / e_std_wld
            mesh_ef       = torch.cat([ref_feat_n, world_feat], dim=1)
            disp_norm     = (cur - ref_pos - d_mean) / d_std
            prev_vel_norm = (prev_vel - v_mean) / v_std
            node_x        = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)
            g             = Data(x=node_x, edge_index=mesh_ei, num_nodes=N)
            out           = model(node_x, mesh_ef, empty_wef, g)
            vel           = out[:, :3] * v_std + v_mean
            prev_vel      = vel
            cur           = cur + vel
            pred_wp_list.append(cur.cpu())

    # Per-timestep errors (t=2..T, i.e. t=2..20 for 20-step training)
    disp_rel_per_t   = []
    stress_rel_per_t = []

    with torch.inference_mode():
        for t in range(2, T + 1):
            # Displacement error
            err = torch.norm(pred_wp_list[t] - gt_wp_list[t], dim=1).mean().item()
            disp_rel_per_t.append(err / gt_disp_mag_final * 100.0)

            # Stress error
            vm_pred = compute_vm_stress(pred_wp_list[t].to(device), ref_pos, mesh_ei).cpu()
            vm_gt   = compute_vm_stress(gt_wp_list[t].to(device),   ref_pos, mesh_ei).cpu()
            gt_mean = vm_gt.mean().item()
            if gt_mean > 1e-10:
                stress_rel_per_t.append((vm_pred - vm_gt).abs().mean().item() / gt_mean * 100.0)
            else:
                stress_rel_per_t.append(float("nan"))

    # Prepend t=1 with error=0 (GT seed)
    disp_rel_per_t   = [0.0] + disp_rel_per_t
    stress_rel_per_t = [0.0] + stress_rel_per_t
    return np.array(disp_rel_per_t), np.array(stress_rel_per_t)


# ── Build sample index lists ──────────────────────────────────────────────────

all_files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith("sample_") and f.endswith(".pt"))

train_indices = []
for i in range(2000):
    if len(train_indices) >= NUM_TRAIN_SAMPLES:
        break
    if i >= len(all_files):
        break
    d0 = torch.load(os.path.join(DATA_DIR, all_files[i]),
                    map_location="cpu", weights_only=False)
    if d0[0]["graph"].num_nodes <= MAX_NODES_TRAIN:
        train_indices.append(i)

test_indices = list(range(2000, min(2000 + NUM_TEST_SAMPLES, len(all_files))))

print(f"Train: {len(train_indices)} | Test: {len(test_indices)}\n")


# ── Collect per-timestep errors ───────────────────────────────────────────────

def collect(data_dir, fnames, label):
    disp_list, stress_list = [], []
    for i, fname in enumerate(fnames):
        result = run_sample(data_dir, fname)
        if result is None:
            continue
        d, s = result
        disp_list.append(d)
        stress_list.append(s)
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(fnames)} done")
    return np.array(disp_list), np.array(stress_list)   # (N, T-1)


print("=== Train ===")
train_d, train_s = collect(DATA_DIR, [all_files[i] for i in train_indices], "train")
print("=== Test ===")
test_d,  test_s  = collect(DATA_DIR, [all_files[i] for i in test_indices],  "test")


# ── Plot ──────────────────────────────────────────────────────────────────────

T_steps = train_d.shape[1]
ts = np.arange(1, 1 + T_steps)   # [1, 2, ..., 20]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Rollout Error Comparison (mean ± std over {len(train_d)} samples)", fontsize=13)

colors = {"train": "tab:blue", "test": "tab:green"}
labels = {"train": "Training (in-dist)", "test": "Test (in-dist)"}

datasets = [("train", train_d, train_s), ("test", test_d, test_s)]

for ax, (ylabel, idx) in zip(axes, [("Displacement Error (%)", 0), ("Stress Error (%)", 1)]):
    for key, d_arr, s_arr in datasets:
        arr = d_arr if idx == 0 else s_arr
        mu  = np.nanmean(arr, axis=0)
        sd  = np.nanstd(arr,  axis=0)
        c   = colors[key]
        ax.plot(ts, mu, color=c, label=labels[key], linewidth=2)
        ax.fill_between(ts, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.set_xlabel("Time step t", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12)
    ax.set_xlim(2, 19)
    ax.set_xticks(range(2, 20))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT_PLOT}")
