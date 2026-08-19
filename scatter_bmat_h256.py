"""
scatter_bmat_h256.py
====================
Post-process h256 rollout with B-matrix stress for Si/UF nodes.

- Displacement : h256 rollout (autoregressive)
- Si/UF stress : B-matrix constitutive law from predicted displacement (exact for linear elastic)
- Solder stress: h256 direct prediction output
- PEEQ         : h256 direct prediction output

Output: 2×5 per-material scatter plot (same format as scatter_ep860.py --epoch 940)
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

_parser = argparse.ArgumentParser()
_parser.add_argument("--epoch", type=int, default=860)
_args, _ = _parser.parse_known_args()
CKPT_EPOCH = _args.epoch

NPZ_DIR    = "./02_abaqus_npz_m0035"
PT_DIR     = "./04_preprocessed_pt_m0035"
TOPO_DIR   = "./04_preprocessed_pt_m0035_constloss"   # has {sid}_topo.pt
CKPT_PATH  = "./checkpoints_m0035_h256"
NODE_STATS = "./04_preprocessed_pt_m0035/node_stats_m0035.json"
EDGE_STATS = "./edge_stats_m0035.json"
OUTPUT_PNG = f"./scatter_bmat_siuf_ep{CKPT_EPOCH}.png"

_rng      = np.random.default_rng(seed=42)
TRAIN_IDS = [f"S{i:04d}" for i in sorted(_rng.choice(range(1,   201), size=10, replace=False))]
TEST_IDS  = [f"S{i:04d}" for i in sorted(_rng.choice(range(201, 221), size=10, replace=False))]

NUM_INPUT  = 17
NUM_EDGE   = 9
NUM_OUTPUT = 10
PROC_SIZE  = 15
HIDDEN_DIM = 256

E_REF       = 130000.0
MAT_E_FIXED = np.array([130000.0, 130000.0, 47000.0, np.nan], dtype=np.float32)
C3D8_EDGES  = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
               (0,4),(1,5),(2,6),(3,7)]

# Temperature: Frame 0 = 250°C (stress-free reference), Frame 20 = 20°C, linear ramp
T_REF   = 250.0   # °C, stress-free state
T_FINAL = 20.0    # °C
N_FRAMES = 21     # frames 0..20

def dT_at_frame(t):
    """Cumulative temperature change from reference at frame t."""
    return (T_FINAL - T_REF) / (N_FRAMES - 1) * t   # -11.5 * t


# ── Helpers ───────────────────────────────────────────────────────────────────

def von_mises(s):
    xx, yy, zz = s[..., 0], s[..., 1], s[..., 2]
    xy, xz, yz = s[..., 3], s[..., 4], s[..., 5]
    return np.sqrt(np.maximum(
        0.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2 + 6*(xy**2+xz**2+yz**2)),
        0.0))


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
    d_mesh  = mesh_pos[dst] - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    E_col   = (edge_E / E_REF)[:, None]
    return np.concatenate([d_mesh, n_mesh, d_world, n_world, E_col], axis=1).astype(np.float32)


def load_stats():
    with open(NODE_STATS) as f: ns = json.load(f)
    with open(EDGE_STATS) as f: es = json.load(f)
    return {
        "disp_mean": np.array(ns["disp_mean"],         dtype=np.float32),
        "disp_std":  np.array(ns["disp_std"],          dtype=np.float32),
        "vel_mean":  np.array(ns["vel_mean"],           dtype=np.float32),
        "vel_std":   np.array(ns["vel_std"],            dtype=np.float32),
        "s_mean_Si": np.array(ns["stress_mean_Si"],     dtype=np.float32),
        "s_std_Si":  np.array(ns["stress_std_Si"],      dtype=np.float32),
        "s_mean_So": np.array(ns["stress_mean_Solder"], dtype=np.float32),
        "s_std_So":  np.array(ns["stress_std_Solder"],  dtype=np.float32),
        "s_mean_UF": np.array(ns["stress_mean_UF"],     dtype=np.float32),
        "s_std_UF":  np.array(ns["stress_std_UF"],      dtype=np.float32),
        "peeq_mean": float(ns["peeq_mean"]),
        "peeq_std":  float(ns["peeq_std"]),
        "edge_mean": np.array(es["edge_mean"],          dtype=np.float32),
        "edge_std":  np.array(es["edge_std"],           dtype=np.float32),
    }


# ── B-matrix stress (numpy) ───────────────────────────────────────────────────

def bmat_stress_np(u_next, topo, node_mat, dT):
    """
    u_next  : (N, 3) displacement from reference (mesh_pos), metres
    topo    : dict with elem_conn(E,8), elem_mat(E,), elem_B(E,3,8),
              elem_vol(E,), elem_E(E,), elem_nu(E,), elem_CTE(E,)
    node_mat: (N,) int
    dT      : scalar, temperature change from stress-free reference (°C)
    Returns : sig (N, 6) physical stress in MPa  (zeros for Solder nodes)
    """
    N   = u_next.shape[0]
    ec  = topo["elem_conn"].numpy() if hasattr(topo["elem_conn"], "numpy") else np.asarray(topo["elem_conn"])
    em  = topo["elem_mat"].numpy()  if hasattr(topo["elem_mat"],  "numpy") else np.asarray(topo["elem_mat"])
    B   = topo["elem_B"].numpy()    if hasattr(topo["elem_B"],    "numpy") else np.asarray(topo["elem_B"])
    vol = topo["elem_vol"].numpy()  if hasattr(topo["elem_vol"],  "numpy") else np.asarray(topo["elem_vol"])
    E_e = topo["elem_E"].numpy()    if hasattr(topo["elem_E"],    "numpy") else np.asarray(topo["elem_E"])
    nu_e= topo["elem_nu"].numpy()   if hasattr(topo["elem_nu"],   "numpy") else np.asarray(topo["elem_nu"])
    cte = topo["elem_CTE"].numpy()  if hasattr(topo["elem_CTE"],  "numpy") else np.asarray(topo["elem_CTE"])

    # Elastic elements only (skip Solder mat=2)
    el = em != 2
    if not el.any():
        return np.zeros((N, 6), dtype=np.float32)

    ec_el  = ec[el];  B_el  = B[el].astype(np.float64)
    vol_el = vol[el]; E_el  = E_e[el].astype(np.float64)
    nu_el  = nu_e[el].astype(np.float64)
    cte_el = cte[el].astype(np.float64)
    em_el  = em[el]

    # Element displacements (N,3) → (E_el,8,3)
    u_elem = u_next[ec_el].astype(np.float64)   # (E_el,8,3)
    grad   = np.matmul(B_el, u_elem)             # (E_el,3,3)

    # Mechanical strain (subtract isotropic thermal)
    eps_th = cte_el * dT                         # (E_el,)
    e11 = grad[:, 0, 0] - eps_th
    e22 = grad[:, 1, 1] - eps_th
    e33 = grad[:, 2, 2] - eps_th
    g12 = grad[:, 1, 0] + grad[:, 0, 1]
    g13 = grad[:, 2, 0] + grad[:, 0, 2]
    g23 = grad[:, 2, 1] + grad[:, 1, 2]

    lam = E_el * nu_el / ((1 + nu_el) * (1 - 2*nu_el))
    mu  = E_el / (2 * (1 + nu_el))
    tr  = e11 + e22 + e33

    sig_el = np.stack([
        lam*tr + 2*mu*e11,
        lam*tr + 2*mu*e22,
        lam*tr + 2*mu*e33,
        mu*g12, mu*g13, mu*g23,
    ], axis=1).astype(np.float32)   # (E_el, 6) MPa

    # Volume-weighted scatter to nodes (only match material)
    node_mat_el = node_mat[ec_el]                        # (E_el, 8)
    matched     = (node_mat_el == em_el[:, None]).astype(np.float32)  # (E_el, 8)

    ec_flat  = ec_el.reshape(-1)                         # (E_el*8,)
    vol_rep  = (vol_el[:, None] * matched).reshape(-1)
    vw_rep   = (vol_el[:, None, None] * sig_el[:, None, :] * matched[:, :, None]).reshape(-1, 6)

    sig_num = np.zeros((N, 6), dtype=np.float32)
    vol_sum = np.zeros(N,      dtype=np.float32)
    np.add.at(sig_num, ec_flat, vw_rep)
    np.add.at(vol_sum, ec_flat, vol_rep)

    sig_out = np.zeros((N, 6), dtype=np.float32)
    has_w = vol_sum > 0
    sig_out[has_w] = sig_num[has_w] / vol_sum[has_w, None]
    return sig_out


# ── Rollout for one sample ────────────────────────────────────────────────────

@torch.no_grad()
def rollout_sample(sid, model, st, device):
    npz_path  = os.path.join(NPZ_DIR,  f"{sid}.npz")
    pt_path   = os.path.join(PT_DIR,   f"{sid}.pt")
    topo_path = os.path.join(TOPO_DIR, f"{sid}_topo.pt")
    for p in (npz_path, pt_path, topo_path):
        if not os.path.exists(p):
            print(f"  MISSING {p}"); return None

    d         = np.load(npz_path, allow_pickle=True)
    mesh_pos  = d["mesh_pos"].astype(np.float32)
    world_pos = d["world_pos"].astype(np.float32)
    stress_gt = d["stress"].astype(np.float32)
    peeq_gt   = d["peeq"].astype(np.float32)
    node_mat  = d["node_mat"].astype(np.int32)
    elem_conn = d["elem_conn"].astype(np.int32)
    elem_mat  = d["elem_mat"].astype(np.int32)
    E_uf      = float(d["E_uf_MPa"])

    N  = mesh_pos.shape[0]
    T  = world_pos.shape[0]
    si  = (node_mat == 0) | (node_mat == 1)
    sol =  node_mat == 2
    uf  =  node_mat == 3

    mat_E_s = MAT_E_FIXED.copy(); mat_E_s[3] = E_uf
    edge_index, edge_E = build_multigraph(elem_conn, elem_mat.astype(np.int64), mat_E_s)
    ei_tensor = torch.from_numpy(edge_index)

    topo      = torch.load(topo_path, map_location="cpu", weights_only=False)
    step_data = torch.load(pt_path,   map_location="cpu", weights_only=False)
    const_feat = step_data[0]["graph"].x[:, 13:].numpy()

    x1      = step_data[1]["graph"].x.numpy()
    wp_curr = step_data[1]["graph"].world_pos.numpy().astype(np.float32)
    disp_n  = x1[:, 0:3].copy()
    pv_n    = x1[:, 3:6].copy()
    s_n     = x1[:, 6:12].copy()
    peeq_n  = x1[:, 12:13].copy()

    pred_disp_mag = []; gt_disp_mag = []
    pred_vm_si = []; gt_vm_si = []
    pred_vm_so = []; gt_vm_so = []
    pred_vm_uf = []; gt_vm_uf = []
    pred_peeq  = []; gt_peeq  = []

    for _t in range(T - 2):   # predicting frames 2..20
        ef_raw  = compute_edge_feats(mesh_pos, wp_curr, edge_index, edge_E)
        ef_norm = ((ef_raw - st["edge_mean"]) / st["edge_std"]).astype(np.float32)
        node_x  = np.concatenate([disp_n, pv_n, s_n, peeq_n, const_feat], axis=1).astype(np.float32)

        graph = Data(x=torch.from_numpy(node_x), edge_index=ei_tensor, num_nodes=N).to(device)
        mef   = torch.from_numpy(ef_norm).to(device)
        wef   = torch.zeros(0, NUM_EDGE, device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = model(graph.x.float(), mef, wef, graph)
        pred_np = pred.float().cpu().numpy()   # (N, 10) normalized

        # Velocity → update position
        vel_phys = pred_np[:, :3] * st["vel_std"] + st["vel_mean"]
        wp_curr  = wp_curr + vel_phys

        # Update normalized input for next step
        disp_n = ((wp_curr - mesh_pos) - st["disp_mean"]) / st["disp_std"]
        pv_n   = pred_np[:, :3].copy()
        s_n    = pred_np[:, 3:9].copy()
        peeq_n = pred_np[:, 9:10].copy()

        frame_idx = _t + 2
        gt_wp     = world_pos[frame_idx]
        gt_s      = stress_gt[frame_idx]   # (N,6) MPa

        # ── Displacement magnitude (m → mm) ──────────────────────────────────
        pred_disp_mag.append(np.linalg.norm((wp_curr - mesh_pos) * 1e3, axis=1))
        gt_disp_mag.append(  np.linalg.norm((gt_wp   - mesh_pos) * 1e3, axis=1))

        # ── Stress: UF via B-matrix, Si/Solder via direct model output ───────
        # Si: B-matrix amplifies tiny displacement errors (thin die, small CTE)
        #     → direct model prediction is more stable for Si
        # UF: larger displacement magnitude → B-matrix works well
        u_pred   = wp_curr - mesh_pos           # (N,3) displacement in metres
        dT       = dT_at_frame(frame_idx)
        sig_bmat = bmat_stress_np(u_pred, topo, node_mat, dT)  # (N,6) MPa

        s_phys = np.zeros((N, 6), dtype=np.float32)
        s_phys[si]  = sig_bmat[si]                                            # B-matrix
        s_phys[uf]  = sig_bmat[uf]                                            # B-matrix
        s_phys[sol] = pred_np[sol, 3:9] * st["s_std_So"] + st["s_mean_So"]  # direct

        # PEEQ solder (direct)
        peeq_phys = np.zeros(N, dtype=np.float32)
        peeq_phys[sol] = pred_np[sol, 9] * st["peeq_std"] + st["peeq_mean"]

        # Per-material Von Mises
        pred_vm_si.append(von_mises(s_phys[si]));  gt_vm_si.append(von_mises(gt_s[si]))
        pred_vm_so.append(von_mises(s_phys[sol])); gt_vm_so.append(von_mises(gt_s[sol]))
        pred_vm_uf.append(von_mises(s_phys[uf]));  gt_vm_uf.append(von_mises(gt_s[uf]))
        pred_peeq.append(peeq_phys[sol]);          gt_peeq.append(peeq_gt[frame_idx][sol])

    return {
        "pred_disp":  np.concatenate(pred_disp_mag),
        "gt_disp":    np.concatenate(gt_disp_mag),
        "pred_vm_si": np.concatenate(pred_vm_si),
        "gt_vm_si":   np.concatenate(gt_vm_si),
        "pred_vm_so": np.concatenate(pred_vm_so),
        "gt_vm_so":   np.concatenate(gt_vm_so),
        "pred_vm_uf": np.concatenate(pred_vm_uf),
        "gt_vm_uf":   np.concatenate(gt_vm_uf),
        "pred_peeq":  np.concatenate(pred_peeq),
        "gt_peeq":    np.concatenate(gt_peeq),
    }


# ── Metrics / plot ────────────────────────────────────────────────────────────

def r2(pred, gt):
    ss_res = np.sum((gt - pred)**2)
    ss_tot = np.sum((gt - gt.mean())**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def plot_panel(ax, pred, gt, title, unit, color, max_pts=50000):
    n = len(pred)
    if n > max_pts:
        idx = np.random.choice(n, max_pts, replace=False)
        pred, gt = pred[idx], gt[idx]
    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())
    ax.scatter(gt, pred, s=1, alpha=0.3, color=color, rasterized=True)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.0)
    r2v = r2(pred, gt); maev = mae(pred, gt)
    ax.set_xlabel(f"GT ({unit})", fontsize=9)
    ax.set_ylabel(f"Pred ({unit})", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.text(0.05, 0.93, f"R² = {r2v:.4f}\nMAE = {maev:.3e} {unit}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_aspect("equal", adjustable="datalim")


# ── Main ──────────────────────────────────────────────────────────────────────

def collect(ids, model, st, device, label):
    keys = ["pred_disp","gt_disp",
            "pred_vm_si","gt_vm_si","pred_vm_so","gt_vm_so","pred_vm_uf","gt_vm_uf",
            "pred_peeq","gt_peeq"]
    buckets = {k: [] for k in keys}
    for sid in ids:
        print(f"  [{label}] {sid} ...", flush=True)
        res = rollout_sample(sid, model, st, device)
        if res is None: continue
        for k in keys: buckets[k].append(res[k])
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

    np.random.seed(0)
    train_data = collect(TRAIN_IDS, model, st, device, "train")
    test_data  = collect(TEST_IDS,  model, st, device, "test")

    specs = [
        ("disp",  "Displacement",        "mm",  "#2176AE"),
        ("vm_si", "VM Stress — Si",      "MPa", "#C0392B"),
        ("vm_so", "VM Stress — Solder",  "MPa", "#E84855"),
        ("vm_uf", "VM Stress — UF",      "MPa", "#8E44AD"),
        ("peeq",  "PEEQ (Solder)",       "-",   "#F4A261"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        f"H256 Epoch {CKPT_EPOCH} — Si/UF: B-matrix | Solder: direct pred (rollout frames 2–20)",
        fontsize=12, fontweight="bold")

    row_labels = ["Train (S0001–S0010)", "Test (S0201–S0210)"]
    for col, (key, title, unit, color) in enumerate(specs):
        for row, (data, rlabel) in enumerate([(train_data, "Train"), (test_data, "Test")]):
            plot_panel(axes[row, col], data[f"pred_{key}"], data[f"gt_{key}"],
                       f"{rlabel} — {title}", unit, color)
    for row, rlabel in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{rlabel}\nPred", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUTPUT_PNG}")

    print(f"\n{'':22s} {'Train R²':>10} {'Train MAE':>12} {'Test R²':>10} {'Test MAE':>12}")
    print("-" * 72)
    for key, title, unit, _ in specs:
        tr2  = r2( train_data[f"pred_{key}"], train_data[f"gt_{key}"])
        tmae = mae(train_data[f"pred_{key}"], train_data[f"gt_{key}"])
        ter2 = r2( test_data[f"pred_{key}"],  test_data[f"gt_{key}"])
        tema = mae(test_data[f"pred_{key}"],  test_data[f"gt_{key}"])
        print(f"{title:22s} {tr2:>10.4f} {tmae:>12.4e} {ter2:>10.4f} {tema:>12.4e}  ({unit})")


if __name__ == "__main__":
    main()