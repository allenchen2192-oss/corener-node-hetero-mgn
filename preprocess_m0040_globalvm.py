"""
preprocess_m0040_globalvm.py
=============================
Same as preprocess_m0040.py but uses GLOBAL z-score for vm_stress
(single mean/std across all materials: Si + Solder + UF combined).

Purpose: ablation study to isolate the effect of per-material normalization
vs global normalization, keeping HeteroMGN architecture identical.

Output: ./04_preprocessed_hetero_m0040_globalvm/
Stats:  stats_m0040_globalvm.json  (vm_mean_global / vm_std_global)

Usage:
    python preprocess_m0040_globalvm.py           # skip existing .pt
    python preprocess_m0040_globalvm.py --force   # overwrite all
"""

import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

INPUT_DIR  = "./02_abaqus_npz_m0040"
OUTPUT_DIR = "./04_preprocessed_hetero_m0040_globalvm"

E_MIN  = 3000.0
E_MAX  = 130000.0
EPS    = 1e-8
FORCE  = "--force" in sys.argv

C3D8_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

TRAIN_IDS = [f"S{i:04d}" for i in range(1,   201)]
TEST_IDS  = [f"S{i:04d}" for i in range(201, 221)]
ALL_IDS   = TRAIN_IDS + TEST_IDS


# ── Helpers (identical to preprocess_m0040.py) ────────────────────────────────

def load_npz(sid):
    path = os.path.join(INPUT_DIR, f"{sid}.npz")
    return np.load(path) if os.path.exists(path) else None


def build_singlegraph(elem_conn):
    seen = set(); rows, cols = [], []
    for c0 in elem_conn:
        for a, b in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                if (src, dst) not in seen:
                    seen.add((src, dst))
                    rows.append(src); cols.append(dst)
    return np.array([rows, cols], dtype=np.int64)


def build_b02_edges(elem_conn):
    M, npe   = elem_conn.shape
    elem_ids = np.repeat(np.arange(M, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    n2e = np.stack([node_ids, elem_ids])
    e2n = np.stack([elem_ids, node_ids])
    return n2e, e2n


def compute_edge_feats(mesh_pos, world_pos_t, edge_index):
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world],
                          axis=1).astype(np.float32)


def node_type_onehot(node_type, n_classes=5):
    N  = len(node_type)
    oh = np.zeros((N, n_classes), dtype=np.float32)
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


def robust_stats(arr, axis=0):
    mean = arr.mean(axis=axis).astype(np.float32)
    std  = arr.std(axis=axis).astype(np.float32)
    std  = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


# ── Build static graph topology ───────────────────────────────────────────────

_d0        = load_npz(TRAIN_IDS[0])
_edge_index = build_singlegraph(_d0["elem_conn"])
_n2e, _e2n  = build_b02_edges(_d0["elem_conn"])
print(f"Mesh: N={_d0['mesh_pos'].shape[0]} nodes  "
      f"M={_d0['elem_conn'].shape[0]} elems  "
      f"n2n={_edge_index.shape[1]} edges  "
      f"B02={_n2e.shape[1]} edges")


# ── Pass 1: normalization stats from training set ─────────────────────────────

print(f"\nPass 1: collecting stats from {len(TRAIN_IDS)} training samples ...")

all_disp    = []
all_vel     = []
all_vm      = []   # ALL materials combined → global vm stats
all_peeq_sl = []
all_CTE     = []
all_nu      = []
all_ef      = []

for sid in tqdm(TRAIN_IDS):
    d = load_npz(sid)
    if d is None:
        continue

    mesh_pos       = d["mesh_pos"]
    world_pos      = d["world_pos"]
    velocity       = d["velocity"]
    elem_vm_stress = d["elem_vm_stress"]
    elem_peeq      = d["elem_peeq"]
    elem_mat       = d["elem_mat"]
    elem_CTE       = d["elem_CTE"]
    elem_nu        = d["elem_nu"]

    T = velocity.shape[0]
    solder_mask = elem_mat == 2

    for t in range(1, T):
        disp_t = (world_pos[t] - mesh_pos).astype(np.float32)
        all_disp.append(disp_t)
        all_vel.append(velocity[t - 1].astype(np.float32))
        all_ef.append(compute_edge_feats(mesh_pos, world_pos[t], _edge_index))

    # Global vm_stress: collect ALL elements across ALL frames
    for fi in range(1, T + 1):
        all_vm.append(elem_vm_stress[fi].astype(np.float32))
        all_peeq_sl.append(elem_peeq[fi, solder_mask].astype(np.float32))

    all_CTE.append(elem_CTE.astype(np.float32))
    all_nu.append(elem_nu.astype(np.float32))

disp_mean, disp_std = robust_stats(np.concatenate(all_disp, axis=0))
vel_mean,  vel_std  = robust_stats(np.concatenate(all_vel,  axis=0))
edge_mean, edge_std = robust_stats(np.concatenate(all_ef,   axis=0))

vm_arr      = np.concatenate(all_vm,      axis=0)
peeq_sl_arr = np.concatenate(all_peeq_sl, axis=0)
CTE_arr     = np.concatenate(all_CTE,     axis=0)
nu_arr      = np.concatenate(all_nu,      axis=0)

vm_mean_global   = float(vm_arr.mean())
vm_std_global    = max(float(vm_arr.std()), EPS)
peeq_mean_sl     = float(peeq_sl_arr.mean())
peeq_std_sl      = max(float(peeq_sl_arr.std()), EPS)
CTE_mean         = float(CTE_arr.mean());  CTE_std = max(float(CTE_arr.std()), EPS)
nu_mean          = float(nu_arr.mean());   nu_std  = max(float(nu_arr.std()),  EPS)

stats = {
    "disp_mean":        disp_mean.tolist(),
    "disp_std":         disp_std.tolist(),
    "vel_mean":         vel_mean.tolist(),
    "vel_std":          vel_std.tolist(),
    "vm_mean_global":   vm_mean_global,    # single global value
    "vm_std_global":    vm_std_global,     # single global value
    "peeq_mean_solder": peeq_mean_sl,
    "peeq_std_solder":  peeq_std_sl,
    "CTE_mean":         CTE_mean,
    "CTE_std":          CTE_std,
    "nu_mean":          nu_mean,
    "nu_std":           nu_std,
    "edge_mean":        edge_mean.tolist(),
    "edge_std":         edge_std.tolist(),
    "E_min":            E_MIN,
    "E_max":            E_MAX,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
stats_path = os.path.join(OUTPUT_DIR, "stats_m0040_globalvm.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Stats saved -> {stats_path}")
print(f"  vm_stress global: mean={vm_mean_global:.2f}  std={vm_std_global:.2f} MPa")
print(f"  peeq solder:      mean={peeq_mean_sl:.4e}  std={peeq_std_sl:.4e}")
print(f"  CTE:              mean={CTE_mean:.3e}  std={CTE_std:.3e}")
print(f"  nu:               mean={nu_mean:.4f}    std={nu_std:.4f}")


# ── Pass 2: build and save HeteroData .pt files ───────────────────────────────

print(f"\nPass 2: building HeteroData .pt files ...")

edge_index_t = torch.from_numpy(_edge_index)
n2e_t        = torch.from_numpy(_n2e)
e2n_t        = torch.from_numpy(_e2n)

for sid in tqdm(ALL_IDS):
    pt_path = os.path.join(OUTPUT_DIR, f"{sid}.pt")
    if os.path.exists(pt_path) and not FORCE:
        continue

    d = load_npz(sid)
    if d is None:
        print(f"  MISSING: {sid}.npz")
        continue

    mesh_pos       = d["mesh_pos"]
    world_pos      = d["world_pos"]
    velocity       = d["velocity"]
    elem_vm_stress = d["elem_vm_stress"]
    elem_peeq      = d["elem_peeq"]
    elem_mat       = d["elem_mat"]
    elem_E         = d["elem_E"]
    elem_CTE       = d["elem_CTE"]
    elem_nu        = d["elem_nu"]
    node_type      = d["node_type"]

    T = velocity.shape[0]
    M = elem_mat.shape[0]
    solder_mask = elem_mat == 2

    mat_oh  = mat_onehot(elem_mat)
    E_norm  = ((elem_E   - E_MIN)    / (E_MAX - E_MIN)).astype(np.float32)[:, None]
    CTE_n   = ((elem_CTE - CTE_mean) / CTE_std).astype(np.float32)[:, None]
    nu_n    = ((elem_nu  - nu_mean)  / nu_std).astype(np.float32)[:, None]
    nt_oh   = node_type_onehot(node_type, n_classes=5)

    steps = []

    for t in range(1, T):

        # Corner node features (11D) — unchanged
        disp_t  = (world_pos[t] - mesh_pos).astype(np.float32)
        prev_v  = velocity[t - 1].astype(np.float32)
        disp_n  = (disp_t - disp_mean) / disp_std
        prev_n  = (prev_v - vel_mean)  / vel_std
        node_x  = np.concatenate([disp_n, prev_n, nt_oh], axis=1)   # (N, 11)

        # Corner node targets (3D) — unchanged
        vel_n   = (velocity[t].astype(np.float32) - vel_mean) / vel_std

        # Element node features (8D) — global vm normalization
        vm_n    = ((elem_vm_stress[t].astype(np.float32) - vm_mean_global)
                   / vm_std_global)[:, None]                          # (M, 1)
        pq_n    = np.zeros(M, dtype=np.float32)
        pq_n[solder_mask] = (
            (elem_peeq[t, solder_mask].astype(np.float32) - peeq_mean_sl) / peeq_std_sl
        )
        elem_x  = np.concatenate([mat_oh, E_norm, CTE_n, nu_n, vm_n, pq_n[:, None]],
                                  axis=1)   # (M, 8)

        # Element node targets (2D) — global vm normalization
        vm_n1   = ((elem_vm_stress[t+1].astype(np.float32) - vm_mean_global)
                   / vm_std_global)[:, None]
        pq_n1   = np.zeros(M, dtype=np.float32)
        pq_n1[solder_mask] = (
            (elem_peeq[t+1, solder_mask].astype(np.float32) - peeq_mean_sl) / peeq_std_sl
        )
        elem_y  = np.concatenate([vm_n1, pq_n1[:, None]], axis=1)   # (M, 2)

        # n2n edge features (8D) — unchanged
        ef_raw  = compute_edge_feats(mesh_pos, world_pos[t], _edge_index)
        ef_norm = ((ef_raw - edge_mean) / edge_std).astype(np.float32)

        hd = HeteroData()
        hd["node"].x    = torch.from_numpy(node_x)
        hd["node"].y    = torch.from_numpy(vel_n)
        hd["element"].x = torch.from_numpy(elem_x)
        hd["element"].y = torch.from_numpy(elem_y)
        hd["node",    "mesh", "node"   ].edge_index = edge_index_t
        hd["node",    "mesh", "node"   ].edge_attr  = torch.from_numpy(ef_norm)
        hd["node",    "in",   "element"].edge_index = n2e_t
        hd["element", "has",  "node"   ].edge_index = e2n_t

        hd.mesh_pos               = torch.from_numpy(mesh_pos.astype(np.float32))
        hd.world_pos              = torch.from_numpy(world_pos[t].astype(np.float32))
        hd.node_type              = torch.from_numpy(node_type.astype(np.int64))
        hd.elem_mat               = torch.from_numpy(elem_mat.astype(np.int64))
        hd["element"].solder_mask = torch.from_numpy(solder_mask)
        hd.step_index             = torch.tensor([t], dtype=torch.long)

        steps.append(hd)

    torch.save(steps, pt_path)

print(f"\nDone. {len(ALL_IDS)} samples -> {OUTPUT_DIR}/")
print(f"  Train: {TRAIN_IDS[0]}..{TRAIN_IDS[-1]}  ({len(TRAIN_IDS)} samples)")
print(f"  Test:  {TEST_IDS[0]}..{TEST_IDS[-1]}   ({len(TEST_IDS)} samples)")
print(f"  Steps per sample: {T-1}  (t=1..{T-1})")
print(f"  Node features: 11D | Element features: 8D | Edge features: 8D")
