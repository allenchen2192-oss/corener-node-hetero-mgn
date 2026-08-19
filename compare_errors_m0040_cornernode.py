"""
compare_errors_m0040_cornernode.py
===================================
Load latest corner-node checkpoint, rollout on train + test samples,
plot 2x5 Pred vs GT scatter:
  Displacement | VM Stress Si | VM Stress Solder | VM Stress UF | PEEQ (Solder)

vm_stress split by "pure" node masks:
  pure_si     : nodes adjacent ONLY to Si elements
  solder_nodes: solder_node_mask (256 interface nodes)
  pure_uf     : nodes adjacent ONLY to UF elements

Rollout: seeded at t=1 from GT, frames 2-20 collected.

Usage:
  python compare_errors_m0040_cornernode.py           # 10 train + 10 test
  python compare_errors_m0040_cornernode.py 20 20     # 20 + 20
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
from torch_geometric.data import Data

from physicsnemo.utils import load_checkpoint
from train_m0040_cornernode import CornerNodeMGN

# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR   = "./02_abaqus_npz_m0040"
STATS_DIR = "./04_preprocessed_cornernode_m0040"
CKPT_PATH = "./checkpoints_m0040_cornernode"
OUT_PNG   = "./compare_errors_m0040_cornernode.png"

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_TEST  = int(sys.argv[2]) if len(sys.argv) > 2 else 10

rng = np.random.default_rng(42)
TRAIN_IDS = sorted(rng.choice([f"S{i:04d}" for i in range(1,   201)], N_TRAIN, replace=False).tolist())
TEST_IDS  = sorted(rng.choice([f"S{i:04d}" for i in range(201, 221)], N_TEST,  replace=False).tolist())

MODEL_CFG = SimpleNamespace(
    hidden_dim=256, processor_size=10,
    num_node_features=13, num_edge_features=14, num_outputs=5,
)

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_multigraph(elem_conn, elem_mat):
    seen = {}; rows, cols, eidx, mids = [], [], [], []
    for ei, conn in enumerate(elem_conn):
        mid = int(elem_mat[ei])
        for a, b in C3D8_EDGES:
            u, v = int(conn[a]), int(conn[b])
            for s, t in ((u,v),(v,u)):
                key = (s, t, mid)
                if key not in seen:
                    seen[key] = len(rows)
                    rows.append(s); cols.append(t)
                    eidx.append(ei); mids.append(mid)
    return (np.array([rows, cols], dtype=np.int64),
            np.array(eidx, dtype=np.int64),
            np.array(mids, dtype=np.int64))


def build_mat_ef(elem_idx, mat_ids, elem_E, elem_CTE, elem_nu,
                 CTE_mean, CTE_std, nu_mean, nu_std, E_MIN, E_MAX):
    mat_oh = np.zeros((len(mat_ids), 3), dtype=np.float32)
    mat_oh[:, 0] = ((mat_ids==0)|(mat_ids==1)).astype(np.float32)
    mat_oh[:, 1] = (mat_ids==2).astype(np.float32)
    mat_oh[:, 2] = (mat_ids==3).astype(np.float32)
    E_n   = ((elem_E[elem_idx]   - E_MIN)    / (E_MAX - E_MIN))[:, None].astype(np.float32)
    CTE_n = ((elem_CTE[elem_idx] - CTE_mean) / CTE_std)[:, None].astype(np.float32)
    nu_n  = ((elem_nu[elem_idx]  - nu_mean)  / nu_std)[:, None].astype(np.float32)
    return np.concatenate([mat_oh, E_n, CTE_n, nu_n], axis=1)


def compute_geom_feats(mesh_pos, world_pos_t, edge_index):
    src, dst = edge_index[0], edge_index[1]
    dm = mesh_pos[dst] - mesh_pos[src];  nm = np.linalg.norm(dm, axis=1, keepdims=True)
    dw = world_pos_t[dst] - world_pos_t[src]; nw = np.linalg.norm(dw, axis=1, keepdims=True)
    return np.concatenate([dm, nm, dw, nw], axis=1).astype(np.float32)


def node_type_onehot(node_type, n_classes=5):
    oh = np.zeros((len(node_type), n_classes), dtype=np.float32)
    for i, nt in enumerate(node_type):
        if 0 <= int(nt) < n_classes:
            oh[i, int(nt)] = 1.0
    return oh


def elem_to_node_mean(elem_vals, elem_conn, N):
    node_sum = np.zeros(N, dtype=np.float64)
    node_cnt = np.zeros(N, dtype=np.int64)
    np.add.at(node_sum, elem_conn.ravel(), np.repeat(elem_vals, 8))
    np.add.at(node_cnt, elem_conn.ravel(), np.ones(elem_conn.size, dtype=np.int64))
    return (node_sum / np.maximum(node_cnt, 1)).astype(np.float32)


def solder_to_node(elem_vals, elem_conn, solder_mask, N):
    """Average solder-element scalars onto their nodes only. Non-solder nodes = 0."""
    node_sum = np.zeros(N, dtype=np.float64)
    node_cnt = np.zeros(N, dtype=np.int64)
    sol_conn = elem_conn[solder_mask]
    np.add.at(node_sum, sol_conn.ravel(), np.repeat(elem_vals[solder_mask], 8))
    np.add.at(node_cnt, sol_conn.ravel(), np.ones(sol_conn.size, dtype=np.int64))
    out = np.zeros(N, dtype=np.float32)
    has = node_cnt > 0
    out[has] = (node_sum[has] / node_cnt[has]).astype(np.float32)
    return out


def r2_mae(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    ss_res = np.sum((gt - pred)**2)
    ss_tot = np.sum((gt - gt.mean())**2)
    return 1 - ss_res / (ss_tot + 1e-30), np.mean(np.abs(gt - pred))


# ── Load stats ────────────────────────────────────────────────────────────────

with open(os.path.join(STATS_DIR, "stats_cornernode_m0040.json")) as f:
    stats = json.load(f)

disp_mean    = np.array(stats["disp_mean"],  dtype=np.float32)
disp_std     = np.array(stats["disp_std"],   dtype=np.float32)
vel_mean     = np.array(stats["vel_mean"],   dtype=np.float32)
vel_std      = np.array(stats["vel_std"],    dtype=np.float32)
geom_mean    = np.array(stats["geom_mean"],  dtype=np.float32)
geom_std     = np.array(stats["geom_std"],   dtype=np.float32)
vm_node_mean = float(stats["vm_node_mean"])
vm_node_std  = float(stats["vm_node_std"])
peeq_mean_sl = float(stats["peeq_mean_solder"])
peeq_std_sl  = float(stats["peeq_std_solder"])
CTE_mean     = float(stats["CTE_mean"]);  CTE_std = float(stats["CTE_std"])
nu_mean      = float(stats["nu_mean"]);   nu_std  = float(stats["nu_std"])
E_MIN        = float(stats["E_min"]);     E_MAX   = float(stats["E_max"])

# ── Load model ────────────────────────────────────────────────────────────────

ckpt_files = glob.glob(os.path.join(CKPT_PATH, "CornerNodeMGN.0.*.pt"))
latest_epoch = max((int(os.path.basename(f).split(".")[2]) for f in ckpt_files), default=0)
print(f"Latest checkpoint epoch: {latest_epoch}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = CornerNodeMGN(MODEL_CFG).to(device)
model.eval()
load_checkpoint(CKPT_PATH, models=model, device=device)

# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout_sample(sid):
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

    solder_mask = (elem_mat == 2)
    si_mask     = (elem_mat == 0) | (elem_mat == 1)
    uf_mask     = (elem_mat == 3)

    # Static multigraph topology
    edge_index, elem_idx, mat_ids = build_multigraph(elem_conn, elem_mat)
    mat_ef = build_mat_ef(elem_idx, mat_ids, elem_E, elem_CTE, elem_nu,
                          CTE_mean, CTE_std, nu_mean, nu_std, E_MIN, E_MAX)
    edge_index_t = torch.from_numpy(edge_index).to(device)

    # Pure-material node masks
    si_node_set     = set(elem_conn[si_mask].ravel())
    solder_node_set = set(elem_conn[solder_mask].ravel())
    uf_node_set     = set(elem_conn[uf_mask].ravel())

    pure_si  = np.array(sorted(si_node_set - uf_node_set - solder_node_set), dtype=np.int64)
    pure_uf  = np.array(sorted(uf_node_set - si_node_set - solder_node_set), dtype=np.int64)
    solder_node_mask = np.zeros(N, dtype=bool)
    solder_node_mask[np.array(sorted(solder_node_set))] = True

    nt_oh = node_type_onehot(node_type)

    # Seed at t=1
    cur_pos        = world_pos[1].astype(np.float32)
    prev_vel_phys  = velocity[0].astype(np.float32)
    vm_node_phys   = elem_to_node_mean(elem_vm_stress[1].astype(np.float32), elem_conn, N)
    peeq_node_phys = solder_to_node(elem_peeq[1].astype(np.float32), elem_conn, solder_mask, N)

    out = {k: [] for k in [
        "disp_gt","disp_pred",
        "vm_gt_Si","vm_pred_Si",
        "vm_gt_Sl","vm_pred_Sl",
        "vm_gt_UF","vm_pred_UF",
        "peeq_gt","peeq_pred",
    ]}

    for fi in range(1, T + 1):   # fi = 1..20
        # Collect frames 2..20 (skip seed fi=1)
        if fi >= 2:
            gt_pos = world_pos[fi].astype(np.float32)
            out["disp_gt"].append(np.linalg.norm(gt_pos - mesh_pos, axis=1))
            out["disp_pred"].append(np.linalg.norm(cur_pos - mesh_pos, axis=1))

            gt_vm_node = elem_to_node_mean(elem_vm_stress[fi].astype(np.float32), elem_conn, N)
            if len(pure_si) > 0:
                out["vm_gt_Si"].append(gt_vm_node[pure_si])
                out["vm_pred_Si"].append(vm_node_phys[pure_si])
            if len(pure_uf) > 0:
                out["vm_gt_UF"].append(gt_vm_node[pure_uf])
                out["vm_pred_UF"].append(vm_node_phys[pure_uf])
            out["vm_gt_Sl"].append(gt_vm_node[solder_node_mask])
            out["vm_pred_Sl"].append(vm_node_phys[solder_node_mask])

            gt_peeq_node = solder_to_node(elem_peeq[fi].astype(np.float32), elem_conn, solder_mask, N)
            out["peeq_gt"].append(gt_peeq_node[solder_node_mask])
            out["peeq_pred"].append(peeq_node_phys[solder_node_mask])

        # Rollout fi → fi+1
        if fi < T:
            with torch.inference_mode():
                disp_n = (cur_pos - mesh_pos - disp_mean) / disp_std
                prev_n = (prev_vel_phys - vel_mean) / vel_std
                vm_n   = ((vm_node_phys - vm_node_mean) / vm_node_std)[:, None]
                pq_n   = np.zeros(N, dtype=np.float32)
                pq_n[solder_node_mask] = (
                    (peeq_node_phys[solder_node_mask] - peeq_mean_sl) / peeq_std_sl
                )
                node_x = np.concatenate([disp_n, prev_n, nt_oh, vm_n, pq_n[:, None]], axis=1)

                geom_n = ((compute_geom_feats(mesh_pos, cur_pos, edge_index)
                           - geom_mean) / geom_std).astype(np.float32)
                edge_attr = np.concatenate([geom_n, mat_ef], axis=1)

                data = Data(
                    x          = torch.from_numpy(node_x).to(device),
                    edge_index = edge_index_t,
                    edge_attr  = torch.from_numpy(edge_attr).to(device),
                )
                pred = model(data)   # (N, 5)

            pred_np = pred.cpu().numpy()
            prev_vel_phys = (pred_np[:, 0:3] * vel_std + vel_mean).astype(np.float32)
            cur_pos       = cur_pos + prev_vel_phys
            vm_node_phys  = (pred_np[:, 3] * vm_node_std + vm_node_mean).astype(np.float32)
            peeq_node_phys = np.zeros(N, dtype=np.float32)
            peeq_node_phys[solder_node_mask] = np.clip(
                pred_np[solder_node_mask, 4] * peeq_std_sl + peeq_mean_sl, 0.0, None
            )

    return {k: np.concatenate(v) for k, v in out.items() if v}


def collect(sample_ids, label):
    keys = ["disp_gt","disp_pred",
            "vm_gt_Si","vm_pred_Si",
            "vm_gt_Sl","vm_pred_Sl",
            "vm_gt_UF","vm_pred_UF",
            "peeq_gt","peeq_pred"]
    arrays = {k: [] for k in keys}
    for sid in sample_ids:
        print(f"  [{label}] {sid} ...", end=" ", flush=True)
        res = rollout_sample(sid)
        if res is None:
            print("MISSING"); continue
        for k in keys:
            if k in res:
                arrays[k].append(res[k])
        print("done")
    return {k: np.concatenate(v) for k, v in arrays.items() if v}


print(f"\nCollecting {N_TRAIN} train samples ...")
train_data = collect(TRAIN_IDS, "train")
print(f"Collecting {N_TEST} test samples ...")
test_data  = collect(TEST_IDS,  "test")

# ── Plot ──────────────────────────────────────────────────────────────────────

COLS = [
    ("disp_gt",  "disp_pred",  "Displacement",      "mm",  "blue"),
    ("vm_gt_Si", "vm_pred_Si", "VM Stress — Si",    "MPa", "#c0392b"),
    ("vm_gt_Sl", "vm_pred_Sl", "VM Stress — Solder","MPa", "#922b21"),
    ("vm_gt_UF", "vm_pred_UF", "VM Stress — UF",   "MPa", "#7d3c98"),
    ("peeq_gt",  "peeq_pred",  "PEEQ (Solder)",     "(-)", "#e67e22"),
]

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle(
    f"Corner-Node MGN  H256 Epoch {latest_epoch} — Pred vs GT (rollout frames 2-20)",
    fontsize=13, fontweight="bold",
)

ROW_DATA  = [train_data, test_data]
ROW_LABEL = [
    f"Train ({TRAIN_IDS[0]}-{TRAIN_IDS[-1]})",
    f"Test  ({TEST_IDS[0]}-{TEST_IDS[-1]})",
]

for row, (data, rlabel) in enumerate(zip(ROW_DATA, ROW_LABEL)):
    for col, (gt_key, pd_key, title, unit, color) in enumerate(COLS):
        ax = axes[row, col]
        if gt_key not in data or pd_key not in data:
            ax.set_title(f"{title}\n(no data)"); continue
        gt   = data[gt_key]
        pred = data[pd_key]

        if len(gt) > 80000:
            idx  = np.random.default_rng(0).choice(len(gt), 80000, replace=False)
            gt   = gt[idx]; pred = pred[idx]

        ax.scatter(gt, pred, s=1, alpha=0.25, color=color, linewidths=0)
        lo = min(gt.min(), pred.min())
        hi = max(gt.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        r2, mae = r2_mae(gt, pred)
        ax.text(0.04, 0.96,
                f"$R^2$={r2:.4f}\nMAE={mae:.4e} {unit}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        if row == 0:
            ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"GT ({unit})", fontsize=8)
        ax.set_ylabel(f"{'Pred' if col else rlabel + chr(10) + 'Pred'} ({unit})", fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
