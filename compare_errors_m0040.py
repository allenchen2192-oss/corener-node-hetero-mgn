"""
compare_errors_m0040.py
========================
Load latest M0040 checkpoint, rollout on train + test samples,
plot 2x5 Pred vs GT scatter:
  Displacement | VM Stress Si | VM Stress Solder | VM Stress UF | PEEQ (Solder)

Rollout: seeded at t=1 from GT, frames 2-20 collected.

Usage:
  python compare_errors_m0040.py              # default 10 train + 10 test
  python compare_errors_m0040.py 20 20        # 20 train + 20 test samples
"""

import os
import sys
import json
import glob
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData

from physicsnemo.utils import load_checkpoint
from train_m0040 import HeteroMGN

# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR   = "./02_abaqus_npz_m0040"
STATS_DIR = "./04_preprocessed_hetero_m0040"
CKPT_PATH = "./checkpoints_m0040"
OUT_PNG   = "./compare_errors_m0040.png"

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_TEST  = int(sys.argv[2]) if len(sys.argv) > 2 else 10

rng = np.random.default_rng(42)
TRAIN_IDS = sorted(rng.choice([f"S{i:04d}" for i in range(1,   201)], N_TRAIN, replace=False).tolist())
TEST_IDS  = sorted(rng.choice([f"S{i:04d}" for i in range(201, 221)], N_TEST,  replace=False).tolist())

MODEL_CFG = SimpleNamespace(
    hidden_dim=256, processor_size=10,
    num_node_features=11, num_edge_features=8,
    num_elem_features=8,  num_node_outputs=3,
    num_elem_outputs=2,
)

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_singlegraph(elem_conn):
    seen = set(); rows, cols = [], []
    for c0 in elem_conn:
        for a, b in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u,v),(v,u)):
                if (src,dst) not in seen:
                    seen.add((src,dst)); rows.append(src); cols.append(dst)
    return np.array([rows, cols], dtype=np.int64)


def build_b02_edges(elem_conn):
    M, npe   = elem_conn.shape
    elem_ids = np.repeat(np.arange(M, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    return np.stack([node_ids, elem_ids]), np.stack([elem_ids, node_ids])


def compute_edge_feats(mesh_pos, world_pos_t, edge_index):
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


def node_type_onehot(node_type, n_classes=5):
    oh = np.zeros((len(node_type), n_classes), dtype=np.float32)
    for i, nt in enumerate(node_type):
        if 0 <= int(nt) < n_classes:
            oh[i, int(nt)] = 1.0
    return oh


def mat_onehot(elem_mat):
    oh = np.zeros((len(elem_mat), 3), dtype=np.float32)
    oh[:, 0] = ((elem_mat == 0) | (elem_mat == 1)).astype(np.float32)
    oh[:, 1] = (elem_mat == 2).astype(np.float32)
    oh[:, 2] = (elem_mat == 3).astype(np.float32)
    return oh


def r2_mae(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - gt.mean()) ** 2)
    r2  = 1 - ss_res / (ss_tot + 1e-30)
    mae = np.mean(np.abs(gt - pred))
    return r2, mae


# ── Load stats ────────────────────────────────────────────────────────────────

with open(os.path.join(STATS_DIR, "stats_m0040.json")) as f:
    stats = json.load(f)

disp_mean  = np.array(stats["disp_mean"],  dtype=np.float32)
disp_std   = np.array(stats["disp_std"],   dtype=np.float32)
vel_mean   = np.array(stats["vel_mean"],   dtype=np.float32)
vel_std    = np.array(stats["vel_std"],    dtype=np.float32)
edge_mean  = np.array(stats["edge_mean"],  dtype=np.float32)
edge_std   = np.array(stats["edge_std"],   dtype=np.float32)
CTE_mean   = float(stats["CTE_mean"]);  CTE_std  = float(stats["CTE_std"])
nu_mean    = float(stats["nu_mean"]);   nu_std   = float(stats["nu_std"])
E_MIN      = float(stats["E_min"]);     E_MAX    = float(stats["E_max"])
peeq_mean_sl = float(stats["peeq_mean_solder"])
peeq_std_sl  = float(stats["peeq_std_solder"])

VM_MEAN = {0: stats["vm_mean_Si"], 1: stats["vm_mean_Si"],
           2: stats["vm_mean_Solder"], 3: stats["vm_mean_UF"]}
VM_STD  = {0: stats["vm_std_Si"],  1: stats["vm_std_Si"],
           2: stats["vm_std_Solder"],  3: stats["vm_std_UF"]}

# ── Detect epoch from checkpoint files ────────────────────────────────────────

ckpt_files = glob.glob(os.path.join(CKPT_PATH, "HeteroMGN.0.*.pt"))
if ckpt_files:
    latest_epoch = max(int(os.path.basename(f).split(".")[2]) for f in ckpt_files)
else:
    latest_epoch = 0
print(f"Latest checkpoint epoch: {latest_epoch}")

# ── Load model ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = HeteroMGN(MODEL_CFG).to(device)
model.eval()
load_checkpoint(CKPT_PATH, models=model, device=device)

# ── Rollout & collect ─────────────────────────────────────────────────────────

def rollout_sample(sid):
    """
    Returns dict of gt/pred arrays collected from rollout frames 2-20:
      disp_gt, disp_pred   : (n_frames * N,) displacement magnitude [mm]
      vm_gt_Si, vm_pred_Si : (n_frames * M_si,) element stress [MPa]
      vm_gt_Sl, vm_pred_Sl : solder
      vm_gt_UF, vm_pred_UF : UF
      peeq_gt, peeq_pred   : (n_frames * M_solder,)
    """
    npz_path = os.path.join(NPZ_DIR, f"{sid}.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path)

    mesh_pos       = d["mesh_pos"]
    world_pos      = d["world_pos"]       # (21, N, 3)
    velocity       = d["velocity"]        # (20, N, 3)
    elem_vm_stress = d["elem_vm_stress"]  # (21, M)
    elem_peeq      = d["elem_peeq"]       # (21, M)
    elem_conn      = d["elem_conn"]
    elem_mat       = d["elem_mat"]
    elem_E         = d["elem_E"]
    elem_CTE       = d["elem_CTE"]
    elem_nu        = d["elem_nu"]
    node_type      = d["node_type"]

    T = velocity.shape[0]   # 20
    N = mesh_pos.shape[0]
    M = elem_mat.shape[0]
    solder_mask = elem_mat == 2
    si_mask     = (elem_mat == 0) | (elem_mat == 1)
    uf_mask     = elem_mat == 3

    # Static graph
    edge_index = build_singlegraph(elem_conn)
    n2e_np, e2n_np = build_b02_edges(elem_conn)
    edge_index_t = torch.from_numpy(edge_index).to(device)
    n2e_t        = torch.from_numpy(n2e_np).to(device)
    e2n_t        = torch.from_numpy(e2n_np).to(device)

    mat_oh  = mat_onehot(elem_mat)
    E_norm  = ((elem_E   - E_MIN)    / (E_MAX - E_MIN)).astype(np.float32)[:, None]
    CTE_n   = ((elem_CTE - CTE_mean) / CTE_std).astype(np.float32)[:, None]
    nu_n    = ((elem_nu  - nu_mean)  / nu_std).astype(np.float32)[:, None]
    elem_static = np.concatenate([mat_oh, E_norm, CTE_n, nu_n], axis=1)

    vm_mean_e = np.array([VM_MEAN[int(m)] for m in elem_mat], dtype=np.float32)
    vm_std_e  = np.array([VM_STD[int(m)]  for m in elem_mat], dtype=np.float32)
    nt_oh     = node_type_onehot(node_type, n_classes=5)

    # Seed at t=1
    cur_pos       = world_pos[1].astype(np.float32)
    prev_vel_phys = velocity[0].astype(np.float32)
    vm_cur_phys   = elem_vm_stress[1].astype(np.float32)
    peeq_cur_phys = elem_peeq[1].astype(np.float32)

    out = {k: [] for k in
           ["disp_gt","disp_pred",
            "vm_gt_Si","vm_pred_Si",
            "vm_gt_Sl","vm_pred_Sl",
            "vm_gt_UF","vm_pred_UF",
            "peeq_gt","peeq_pred"]}

    for fi in range(1, T + 1):   # fi = 1..20
        # Collect from frames 2..20 (skip seed fi=1)
        if fi >= 2:
            gt_pos  = world_pos[fi].astype(np.float32)
            gt_disp = np.linalg.norm(gt_pos  - mesh_pos, axis=1)
            pd_disp = np.linalg.norm(cur_pos - mesh_pos, axis=1)
            out["disp_gt"].append(gt_disp);    out["disp_pred"].append(pd_disp)

            gt_vm  = elem_vm_stress[fi].astype(np.float32)
            out["vm_gt_Si"].append(gt_vm[si_mask]);   out["vm_pred_Si"].append(vm_cur_phys[si_mask])
            out["vm_gt_Sl"].append(gt_vm[solder_mask]);out["vm_pred_Sl"].append(vm_cur_phys[solder_mask])
            out["vm_gt_UF"].append(gt_vm[uf_mask]);   out["vm_pred_UF"].append(vm_cur_phys[uf_mask])

            gt_peeq = elem_peeq[fi].astype(np.float32)
            out["peeq_gt"].append(gt_peeq[solder_mask])
            out["peeq_pred"].append(peeq_cur_phys[solder_mask])

        # Rollout fi → fi+1
        if fi < T:
            with torch.inference_mode():
                disp_n = (cur_pos - mesh_pos - disp_mean) / disp_std
                prev_n = (prev_vel_phys - vel_mean) / vel_std
                node_x = np.concatenate([disp_n, prev_n, nt_oh], axis=1)

                vm_n  = ((vm_cur_phys - vm_mean_e) / vm_std_e)[:, None]
                pq_n  = np.zeros(M, dtype=np.float32)
                pq_n[solder_mask] = (peeq_cur_phys[solder_mask] - peeq_mean_sl) / peeq_std_sl
                elem_x = np.concatenate([elem_static, vm_n, pq_n[:, None]], axis=1)

                ef = ((compute_edge_feats(mesh_pos, cur_pos, edge_index)
                       - edge_mean) / edge_std).astype(np.float32)

                hd = HeteroData()
                hd["node"].x    = torch.from_numpy(node_x).to(device)
                hd["element"].x = torch.from_numpy(elem_x).to(device)
                hd["node",    "mesh", "node"   ].edge_index = edge_index_t
                hd["node",    "mesh", "node"   ].edge_attr  = torch.from_numpy(ef).to(device)
                hd["node",    "in",   "element"].edge_index = n2e_t
                hd["element", "has",  "node"   ].edge_index = e2n_t

                vel_pred, elem_pred = model(hd)

            prev_vel_phys = (vel_pred.cpu().numpy() * vel_std + vel_mean).astype(np.float32)
            cur_pos       = cur_pos + prev_vel_phys
            vm_cur_phys   = (elem_pred[:, 0].cpu().numpy() * vm_std_e + vm_mean_e).astype(np.float32)
            peeq_cur_phys = np.zeros(M, dtype=np.float32)
            peeq_cur_phys[solder_mask] = (
                elem_pred[solder_mask, 1].cpu().numpy() * peeq_std_sl + peeq_mean_sl
            )
            peeq_cur_phys = np.clip(peeq_cur_phys, 0.0, None)

    return {k: np.concatenate(v) for k, v in out.items()}


def collect(sample_ids, label):
    arrays = {k: [] for k in
              ["disp_gt","disp_pred",
               "vm_gt_Si","vm_pred_Si",
               "vm_gt_Sl","vm_pred_Sl",
               "vm_gt_UF","vm_pred_UF",
               "peeq_gt","peeq_pred"]}
    for sid in sample_ids:
        print(f"  [{label}] {sid} ...", end=" ", flush=True)
        res = rollout_sample(sid)
        if res is None:
            print("MISSING"); continue
        for k in arrays:
            arrays[k].append(res[k])
        print("done")
    return {k: np.concatenate(v) for k, v in arrays.items() if v}


print(f"\nCollecting {N_TRAIN} train samples ...")
train_data = collect(TRAIN_IDS, "train")
print(f"Collecting {N_TEST} test samples ...")
test_data  = collect(TEST_IDS,  "test")

# ── Plot ──────────────────────────────────────────────────────────────────────

COLS = [
    ("disp_gt",  "disp_pred",  "Displacement",     "mm",  "blue"),
    ("vm_gt_Si", "vm_pred_Si", "VM Stress — Si",   "MPa", "#c0392b"),
    ("vm_gt_Sl", "vm_pred_Sl", "VM Stress — Solder","MPa","#922b21"),
    ("vm_gt_UF", "vm_pred_UF", "VM Stress — UF",   "MPa", "#7d3c98"),
    ("peeq_gt",  "peeq_pred",  "PEEQ (Solder)",    "(-)", "#e67e22"),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle(
    f"H256 Epoch {latest_epoch} — Per-Material Prediction vs GT (rollout frames 2-20)",
    fontsize=13, fontweight="bold"
)

ROW_DATA  = [train_data, test_data]
ROW_LABEL = [
    f"Train ({TRAIN_IDS[0]}-{TRAIN_IDS[-1]})",
    f"Test  ({TEST_IDS[0]}-{TEST_IDS[-1]})",
]

for row, (data, rlabel) in enumerate(zip(ROW_DATA, ROW_LABEL)):
    for col, (gt_key, pd_key, title, unit, color) in enumerate(COLS):
        ax  = axes[row, col]
        gt  = data[gt_key]
        pred = data[pd_key]

        # Sub-sample for speed (max 80k points)
        if len(gt) > 80000:
            idx  = np.random.choice(len(gt), 80000, replace=False)
            gt   = gt[idx]; pred = pred[idx]

        ax.scatter(gt, pred, s=1, alpha=0.25, color=color, linewidths=0)

        lo = min(gt.min(), pred.min())
        hi = max(gt.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        r2, mae = r2_mae(gt, pred)
        ax.text(0.04, 0.96,
                f"$R^2$={r2:.4f}\nMAE={mae:.4e} {unit}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        if row == 0:
            ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"GT ({unit})", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{rlabel}\nPred ({unit})", fontsize=8)
        else:
            ax.set_ylabel(f"Pred ({unit})", fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
