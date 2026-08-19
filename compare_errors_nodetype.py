"""
compare_errors_nodetype.py
===========================
Rollout with train_prevvel.py model (8 features) and report displacement
+ stress error metrics for training and test samples.

Output columns:
  sample        : sample index
  split         : train / test
  mean_err_m    : mean node displacement error over all timesteps (metres)
  final_err_m   : mean node displacement error at last timestep (metres)
  max_err_m     : max node displacement error at last timestep (metres)
  rel_disp_pct  : final_err / max_GT_disp_mag × 100 (%)
  stress_mae_Pa : mean |vm_pred - vm_GT| at last timestep (Pa)
  rel_stress_pct: stress_mae / mean(vm_GT) × 100 (%)

Usage:
  python compare_errors_nodetype.py
"""

import os
import numpy as np
import torch
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

DATA_DIR  = "./04_preprocessed_pt_abaqus_prevvel"
CKPT_PATH = "./checkpoints_abaqus_prevvel_1000data_1000nodes_h128_gfdm_stress_nodetype_dispnorm"
CKPT_EPOCH = None  # None = latest

NUM_TRAIN_SAMPLES = 100   # max training samples to evaluate (nodes < 1000 only)
NUM_TEST_SAMPLES  = 100   # max test samples to evaluate
MAX_NODES_TRAIN   = 1000  # match training filter


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    row, col = edge_index
    d = pos[col] - pos[row]
    n = torch.norm(d, dim=1, keepdim=True)
    return torch.cat([d, n], dim=1)


E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))

def compute_vm_stress(wp, ref, mesh_ei):
    """Von Mises stress (Pa) via scatter linear fit (GFDM W=I)."""
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
v_mean     = node_stats["velocity_mean"].to(device)
v_std      = node_stats["velocity_std"].to(device)
d_mean     = node_stats["disp_mean"].to(device)
d_std      = node_stats["disp_std"].to(device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref = edge_stats["edge_mean"].to(device)[:4]
e_std_ref  = edge_stats["edge_std"].to(device)[:4]
e_mean_wld = edge_stats["edge_mean"].to(device)[4:]
e_std_wld  = edge_stats["edge_std"].to(device)[4:]

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

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)


# ── Rollout & error computation ───────────────────────────────────────────────

def run_sample(idx, split):
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range")
        return None

    fname = all_files[idx]
    data  = torch.load(os.path.join(DATA_DIR, fname),
                       map_location="cpu", weights_only=False)
    T = len(data)
    N = data[0]["graph"].num_nodes

    ref_pos      = data[0]["graph"].mesh_pos.to(device)
    mesh_ei      = data[0]["graph"].edge_index.to(device)
    node_type_oh = data[0]["graph"].x[:, 6:8].to(device)  # (N, 2), constant

    ref_feat_n = (compute_edge_features(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref

    # GT world-pos sequence
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    # Rollout (seed at t=1 with GT)
    gt_vel_0 = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur       = gt_wp_list[1].to(device)
    prev_vel  = gt_vel_0
    pred_wp_list = [gt_wp_list[0], gt_wp_list[1]]

    with torch.inference_mode():
        for _ in range(T - 1):
            world_feat = (compute_edge_features(cur, mesh_ei) - e_mean_wld) / e_std_wld
            mesh_ef    = torch.cat([ref_feat_n, world_feat], dim=1)
            disp_norm     = (cur - ref_pos - d_mean) / d_std
            prev_vel_norm = (prev_vel - v_mean) / v_std
            node_x        = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)
            g          = Data(x=node_x, edge_index=mesh_ei, num_nodes=N)
            out        = model(node_x, mesh_ef, empty_wef, g)
            vel        = out[:, :3] * v_std + v_mean
            prev_vel   = vel
            cur        = cur + vel
            pred_wp_list.append(cur.cpu())

    # Error per timestep (skip seeded t=0,1)
    step_errors = []
    for t in range(2, T + 1):
        err = torch.norm(pred_wp_list[t] - gt_wp_list[t], dim=1).mean().item()
        step_errors.append(err)

    mean_err  = float(np.mean(step_errors))
    final_err = step_errors[-1]
    max_err   = float(torch.norm(
        pred_wp_list[-1] - gt_wp_list[-1], dim=1
    ).max().item())

    # GT max displacement magnitude (for relative displacement error)
    gt_disp_mag = torch.norm(gt_wp_list[-1] - ref_pos.cpu(), dim=1).max().item()
    rel_disp_pct = (final_err / gt_disp_mag * 100) if gt_disp_mag > 1e-10 else float("nan")

    # Stress error at final frame (GFDM on pred vs GT positions)
    with torch.inference_mode():
        vm_pred = compute_vm_stress(pred_wp_list[-1].to(device), ref_pos, mesh_ei).cpu()
        vm_gt   = compute_vm_stress(gt_wp_list[-1].to(device),   ref_pos, mesh_ei).cpu()
    stress_mae   = (vm_pred - vm_gt).abs().mean().item()
    gt_stress_mean = vm_gt.mean().item()
    rel_stress_pct = (stress_mae / gt_stress_mean * 100) if gt_stress_mean > 1e-10 else float("nan")

    return {
        "sample":         idx,
        "split":          split,
        "N_nodes":        N,
        "mean_err_m":     mean_err,
        "final_err_m":    final_err,
        "max_err_m":      max_err,
        "rel_disp_pct":   rel_disp_pct,
        "stress_mae_Pa":  stress_mae,
        "rel_stress_pct": rel_stress_pct,
    }


# ── Build index lists dynamically ─────────────────────────────────────────────

# Training: indices 0–1999, filter nodes < MAX_NODES_TRAIN, take first NUM_TRAIN_SAMPLES
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

# Test: indices 2000–end, take first NUM_TEST_SAMPLES
test_indices = list(range(2000, min(2000 + NUM_TEST_SAMPLES, len(all_files))))

print(f"Evaluating {len(train_indices)} train samples, {len(test_indices)} test samples\n")


# ── Run all samples ───────────────────────────────────────────────────────────

results = []
for idx in train_indices:
    r = run_sample(idx, "train")
    if r:
        results.append(r)
        print(f"[train] sample_{idx:05d}  N={r['N_nodes']:4d}  "
              f"disp_mean={r['mean_err_m']*1e3:.3f}mm  "
              f"disp_final={r['final_err_m']*1e3:.3f}mm  "
              f"disp_rel={r['rel_disp_pct']:.2f}%  "
              f"stress_mae={r['stress_mae_Pa']/1e6:.2f}MPa  "
              f"stress_rel={r['rel_stress_pct']:.2f}%")

print()
for idx in test_indices:
    r = run_sample(idx, "test")
    if r:
        results.append(r)
        print(f"[test]  sample_{idx:05d}  N={r['N_nodes']:4d}  "
              f"disp_mean={r['mean_err_m']*1e3:.3f}mm  "
              f"disp_final={r['final_err_m']*1e3:.3f}mm  "
              f"disp_rel={r['rel_disp_pct']:.2f}%  "
              f"stress_mae={r['stress_mae_Pa']/1e6:.2f}MPa  "
              f"stress_rel={r['rel_stress_pct']:.2f}%")

# ── Summary ───────────────────────────────────────────────────────────────────

train_res = [r for r in results if r["split"] == "train"]
test_res  = [r for r in results if r["split"] == "test"]

print("\n" + "="*60)
print("Summary")
print("="*60)
if train_res:
    print(f"Train  disp_mean_err : {np.mean([r['mean_err_m']     for r in train_res])*1e3:.3f} mm")
    print(f"Train  disp_final_err: {np.mean([r['final_err_m']    for r in train_res])*1e3:.3f} mm")
    print(f"Train  disp_rel_err  : {np.mean([r['rel_disp_pct']   for r in train_res]):.2f}%")
    print(f"Train  stress_mae    : {np.mean([r['stress_mae_Pa']  for r in train_res])/1e6:.2f} MPa")
    print(f"Train  stress_rel_err: {np.mean([r['rel_stress_pct'] for r in train_res]):.2f}%")
if test_res:
    print(f"Test   disp_mean_err : {np.mean([r['mean_err_m']     for r in test_res])*1e3:.3f} mm")
    print(f"Test   disp_final_err: {np.mean([r['final_err_m']    for r in test_res])*1e3:.3f} mm")
    print(f"Test   disp_rel_err  : {np.mean([r['rel_disp_pct']   for r in test_res]):.2f}%")
    print(f"Test   stress_mae    : {np.mean([r['stress_mae_Pa']  for r in test_res])/1e6:.2f} MPa")
    print(f"Test   stress_rel_err: {np.mean([r['rel_stress_pct'] for r in test_res]):.2f}%")
print("="*60)
