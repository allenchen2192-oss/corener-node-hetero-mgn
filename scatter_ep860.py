"""
scatter_ep860.py
================
Scatter plot: prediction vs GT for all nodes x all rollout frames (2-20).
Metrics: R² and MAE in physical units.

Train: S0001-S0010 / Test: S0201-S0210
Variables: displacement magnitude (mm), von Mises stress (MPa), PEEQ (solder only)
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
from export_abaqus_paraview import build_surface_cells_from_elems
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR    = "./02_abaqus_npz_m0035"
PT_DIR     = "./04_preprocessed_pt_m0035"
CKPT_PATH  = "./checkpoints_m0035_h256"
NODE_STATS = "./04_preprocessed_pt_m0035/node_stats_m0035.json"
EDGE_STATS = "./edge_stats_m0035.json"
_parser = argparse.ArgumentParser()
_parser.add_argument("--epoch", type=int, default=860)
_args, _ = _parser.parse_known_args()
CKPT_EPOCH = _args.epoch
OUTPUT_PNG = f"./scatter_ep{CKPT_EPOCH}.png"

TRAIN_IDS = [f"S{i:04d}" for i in range(1, 11)]    # S0001-S0010
TEST_IDS  = [f"S{i:04d}" for i in range(201, 211)]  # S0201-S0210

NUM_INPUT  = 17
NUM_EDGE   = 9
NUM_OUTPUT = 10
PROC_SIZE  = 15
HIDDEN_DIM = 256

E_REF       = 130000.0
MAT_E_FIXED = np.array([130000.0, 130000.0, 47000.0, np.nan], dtype=np.float32)
C3D8_EDGES  = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def von_mises(s):
    """s: (..., 6) [xx,yy,zz,xy,xz,yz] → von Mises scalar"""
    xx, yy, zz = s[..., 0], s[..., 1], s[..., 2]
    xy, xz, yz = s[..., 3], s[..., 4], s[..., 5]
    return np.sqrt(0.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2
                          + 6*(xy**2 + xz**2 + yz**2)))


def build_multigraph(elem_conn, elem_mat, mat_E_sample):
    seen = {}
    rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E_sample[mat_idx])
        for (a, b) in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst, int(mat_idx))
                if key not in seen:
                    seen[key] = True
                    rows.append(src); cols.append(dst); E_vals.append(E_elem)
    return (np.array([rows, cols], dtype=np.int64),
            np.array(E_vals, dtype=np.float32))


def compute_edge_feats(mesh_pos, world_pos_t, edge_index, edge_E):
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    E_col   = (edge_E / E_REF)[:, None]
    return np.concatenate([d_mesh, n_mesh, d_world, n_world, E_col], axis=1).astype(np.float32)


def load_stats():
    with open(NODE_STATS) as f: ns = json.load(f)
    with open(EDGE_STATS) as f: es = json.load(f)
    return {
        "disp_mean": np.array(ns["disp_mean"],          dtype=np.float32),
        "disp_std":  np.array(ns["disp_std"],           dtype=np.float32),
        "vel_mean":  np.array(ns["vel_mean"],            dtype=np.float32),
        "vel_std":   np.array(ns["vel_std"],             dtype=np.float32),
        "s_mean_Si": np.array(ns["stress_mean_Si"],      dtype=np.float32),
        "s_std_Si":  np.array(ns["stress_std_Si"],       dtype=np.float32),
        "s_mean_So": np.array(ns["stress_mean_Solder"],  dtype=np.float32),
        "s_std_So":  np.array(ns["stress_std_Solder"],   dtype=np.float32),
        "s_mean_UF": np.array(ns["stress_mean_UF"],      dtype=np.float32),
        "s_std_UF":  np.array(ns["stress_std_UF"],       dtype=np.float32),
        "peeq_mean": float(ns["peeq_mean"]),
        "peeq_std":  float(ns["peeq_std"]),
        "edge_mean": np.array(es["edge_mean"],           dtype=np.float32),
        "edge_std":  np.array(es["edge_std"],            dtype=np.float32),
    }


def denorm(pred_np, node_mat, st):
    N = pred_np.shape[0]
    si  = (node_mat == 0) | (node_mat == 1)
    sol =  node_mat == 2
    uf  =  node_mat == 3
    vel_phys = pred_np[:, :3] * st["vel_std"] + st["vel_mean"]
    s = np.zeros((N, 6), dtype=np.float32)
    s[si]  = pred_np[si,  3:9] * st["s_std_Si"] + st["s_mean_Si"]
    s[sol] = pred_np[sol, 3:9] * st["s_std_So"] + st["s_mean_So"]
    s[uf]  = pred_np[uf,  3:9] * st["s_std_UF"] + st["s_mean_UF"]
    p = np.zeros(N, dtype=np.float32)
    p[sol] = pred_np[sol, 9] * st["peeq_std"] + st["peeq_mean"]
    return vel_phys, s, p


# ── Rollout for one sample ────────────────────────────────────────────────────

@torch.no_grad()
def rollout_sample(sid, model, st, device):
    """Returns dict with arrays (disp_mag, vm_stress, peeq_sol) for pred and gt,
    across all rollout frames (frames 2-20), all nodes."""
    npz_path = os.path.join(NPZ_DIR, f"{sid}.npz")
    pt_path  = os.path.join(PT_DIR,  f"{sid}.pt")
    if not os.path.exists(npz_path) or not os.path.exists(pt_path):
        print(f"  MISSING {sid}, skipping"); return None

    d         = np.load(npz_path, allow_pickle=True)
    mesh_pos  = d["mesh_pos"].astype(np.float32)
    world_pos = d["world_pos"].astype(np.float32)   # (21, N, 3)
    stress_gt = d["stress"].astype(np.float32)       # (21, N, 6)
    peeq_gt   = d["peeq"].astype(np.float32)         # (21, N)
    node_mat  = d["node_mat"].astype(np.int32)
    elem_conn = d["elem_conn"].astype(np.int32)
    elem_mat  = d["elem_mat"].astype(np.int32)
    E_uf      = float(d["E_uf_MPa"])

    N = mesh_pos.shape[0]
    T = world_pos.shape[0]
    si  = (node_mat == 0) | (node_mat == 1)
    sol =  node_mat == 2
    uf  =  node_mat == 3

    mat_E_s = MAT_E_FIXED.copy(); mat_E_s[3] = E_uf
    edge_index, edge_E = build_multigraph(elem_conn, elem_mat.astype(np.int64), mat_E_s)
    ei_tensor = torch.from_numpy(edge_index)

    step_data  = torch.load(pt_path, map_location="cpu", weights_only=False)
    const_feat = step_data[0]["graph"].x[:, 13:].numpy()

    x1      = step_data[1]["graph"].x.numpy()
    wp_curr = step_data[1]["graph"].world_pos.numpy().astype(np.float32)
    disp_n  = x1[:, 0:3].copy()
    pv_n    = x1[:, 3:6].copy()
    s_n     = x1[:, 6:12].copy()
    peeq_n  = x1[:, 12:13].copy()

    # Collect frames 2-20 (19 rollout steps)
    pred_disp_mag = []
    gt_disp_mag   = []
    pred_vm_si, gt_vm_si   = [], []
    pred_vm_so, gt_vm_so   = [], []
    pred_vm_uf, gt_vm_uf   = [], []
    pred_peeq_sol = []
    gt_peeq_sol   = []

    for t in range(T - 2):
        ef_raw  = compute_edge_feats(mesh_pos, wp_curr, edge_index, edge_E)
        ef_norm = ((ef_raw - st["edge_mean"]) / st["edge_std"]).astype(np.float32)
        node_x  = np.concatenate([disp_n, pv_n, s_n, peeq_n, const_feat], axis=1).astype(np.float32)

        graph = Data(x=torch.from_numpy(node_x), edge_index=ei_tensor, num_nodes=N).to(device)
        mef   = torch.from_numpy(ef_norm).to(device)
        wef   = torch.zeros(0, NUM_EDGE, device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = model(graph.x.float(), mef, wef, graph)
        pred_np = pred.float().cpu().numpy()

        vel_phys, s_phys, p_phys = denorm(pred_np, node_mat, st)
        wp_curr = wp_curr + vel_phys

        disp_n  = ((wp_curr - mesh_pos) - st["disp_mean"]) / st["disp_std"]
        pv_n    = pred_np[:, :3].copy()
        s_n     = pred_np[:, 3:9].copy()
        peeq_n  = pred_np[:, 9:10].copy()

        frame_idx = t + 2
        gt_wp     = world_pos[frame_idx]
        gt_s      = stress_gt[frame_idx]   # (N, 6) physical MPa

        # Displacement magnitude (m → mm), all nodes
        pred_disp_mag.append(np.linalg.norm((wp_curr - mesh_pos) * 1e3, axis=1))
        gt_disp_mag.append(  np.linalg.norm((gt_wp   - mesh_pos) * 1e3, axis=1))

        # Per-material Von Mises stress (MPa)
        pred_vm_si.append(von_mises(s_phys[si]));   gt_vm_si.append(von_mises(gt_s[si]))
        pred_vm_so.append(von_mises(s_phys[sol]));  gt_vm_so.append(von_mises(gt_s[sol]))
        pred_vm_uf.append(von_mises(s_phys[uf]));   gt_vm_uf.append(von_mises(gt_s[uf]))

        # PEEQ (solder only)
        pred_peeq_sol.append(p_phys[sol])
        gt_peeq_sol.append(  peeq_gt[frame_idx][sol])

    return {
        "pred_disp":   np.concatenate(pred_disp_mag),
        "gt_disp":     np.concatenate(gt_disp_mag),
        "pred_vm_si":  np.concatenate(pred_vm_si),
        "gt_vm_si":    np.concatenate(gt_vm_si),
        "pred_vm_so":  np.concatenate(pred_vm_so),
        "gt_vm_so":    np.concatenate(gt_vm_so),
        "pred_vm_uf":  np.concatenate(pred_vm_uf),
        "gt_vm_uf":    np.concatenate(gt_vm_uf),
        "pred_peeq":   np.concatenate(pred_peeq_sol),
        "gt_peeq":     np.concatenate(gt_peeq_sol),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def r2_score(pred, gt):
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mae(pred, gt):
    return np.mean(np.abs(pred - gt))


# ── Plot one panel ────────────────────────────────────────────────────────────

def plot_panel(ax, pred, gt, title, unit, color, max_pts=50000):
    # Subsample for visual clarity
    n = len(pred)
    if n > max_pts:
        idx = np.random.choice(n, max_pts, replace=False)
        pred, gt = pred[idx], gt[idx]

    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())

    ax.scatter(gt, pred, s=1, alpha=0.3, color=color, rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.0, label="y = x")

    r2  = r2_score(pred, gt)
    mae_val = mae(pred, gt)

    ax.set_xlabel(f"GT ({unit})", fontsize=10)
    ax.set_ylabel(f"Pred ({unit})", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.05, 0.93, f"R² = {r2:.4f}\nMAE = {mae_val:.4e} {unit}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_aspect("equal", adjustable="datalim")


# ── Main ──────────────────────────────────────────────────────────────────────

def collect(sample_ids, model, st, device, label):
    keys = ["pred_disp", "gt_disp",
            "pred_vm_si", "gt_vm_si",
            "pred_vm_so", "gt_vm_so",
            "pred_vm_uf", "gt_vm_uf",
            "pred_peeq",  "gt_peeq"]
    buckets = {k: [] for k in keys}
    for sid in sample_ids:
        print(f"  [{label}] {sid} ...", flush=True)
        res = rollout_sample(sid, model, st, device)
        if res is None: continue
        for k in keys:
            buckets[k].append(res[k])
    return {k: np.concatenate(v) for k, v in buckets.items() if v}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    st = load_stats()

    model = HybridMeshGraphNet(
        NUM_INPUT, NUM_EDGE, NUM_OUTPUT,
        processor_size=PROC_SIZE,
        hidden_dim_processor=HIDDEN_DIM,
        hidden_dim_node_encoder=HIDDEN_DIM,
        hidden_dim_edge_encoder=HIDDEN_DIM,
        hidden_dim_node_decoder=HIDDEN_DIM,
    ).to(device)
    ep = load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)
    model.eval()
    print(f"Loaded epoch {ep}\n")

    train_data = collect(TRAIN_IDS, model, st, device, "train")
    test_data  = collect(TEST_IDS,  model, st, device, "test")

    # ── Figure: 2 rows (train/test) × 5 cols (disp, vm_Si, vm_Solder, vm_UF, PEEQ)
    specs = [
        ("disp",  "Displacement",   "mm",  "#2176AE"),
        ("vm_si", "VM Stress — Si", "MPa", "#C0392B"),
        ("vm_so", "VM Stress — Solder", "MPa", "#E84855"),
        ("vm_uf", "VM Stress — UF", "MPa", "#8E44AD"),
        ("peeq",  "PEEQ (Solder)",  "-",   "#F4A261"),
    ]
    row_labels = ["Train (S0001–S0010)", "Test (S0201–S0210)"]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(f"H256 Epoch {CKPT_EPOCH} — Per-Material Prediction vs GT (rollout frames 2–20)",
                 fontsize=13, fontweight="bold")

    np.random.seed(0)
    for col, (key, title, unit, color) in enumerate(specs):
        for row, (data, rlabel) in enumerate([(train_data, "Train"), (test_data, "Test")]):
            plot_panel(axes[row, col], data[f"pred_{key}"], data[f"gt_{key}"],
                       f"{rlabel} — {title}", unit, color)
        axes[row, 0].set_ylabel(f"{row_labels[row]}\nPred", fontsize=9)

    for row, rlabel in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{rlabel}\nPred", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUTPUT_PNG}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'':22s} {'Train R²':>10} {'Train MAE':>12} {'Test R²':>10} {'Test MAE':>12}")
    print("-" * 72)
    for key, title, unit, _ in specs:
        tr_r2  = r2_score(train_data[f"pred_{key}"], train_data[f"gt_{key}"])
        tr_mae = mae(     train_data[f"pred_{key}"], train_data[f"gt_{key}"])
        te_r2  = r2_score(test_data[f"pred_{key}"],  test_data[f"gt_{key}"])
        te_mae = mae(     test_data[f"pred_{key}"],  test_data[f"gt_{key}"])
        print(f"{title:22s} {tr_r2:>10.4f} {tr_mae:>12.4e} {te_r2:>10.4f} {te_mae:>12.4e}  ({unit})")


if __name__ == "__main__":
    main()