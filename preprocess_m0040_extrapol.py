"""
preprocess_m0040_extrapol.py
============================
Extrapolation-split preprocessing for M0040 HeteroMGN.

Train/Test split (extrapolation):
  Test  (20): extreme UF CTE and UF E samples
    - UF CTE low:  S0001, S0011, S0021, S0031, S0041
    - UF CTE high: S0010, S0020, S0030, S0040, S0050
    - UF E low:    S0002, S0003, S0004, S0005, S0006
    - UF E high:   S0091, S0092, S0093, S0094, S0095
  Train (200): all remaining samples

Key difference from preprocess_m0040.py:
  - Stats (mean/std/E_min/E_max) computed ONLY from the 200 training samples
  - E_MIN/E_MAX are derived from training data (not hardcoded) so that extreme
    test samples have E_norm outside [0,1] -> true extrapolation in input space
  - All 220 samples are normalized and saved (using train-set stats)

Output: ./04_preprocessed_hetero_m0040_extrapol/

Usage:
    python preprocess_m0040_extrapol.py           # skip existing .pt
    python preprocess_m0040_extrapol.py --force   # overwrite all
"""

import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

INPUT_DIR  = "./02_abaqus_npz_m0040"
OUTPUT_DIR = "./04_preprocessed_hetero_m0040_extrapol"

EPS   = 1e-8
FORCE = "--force" in sys.argv

# C3D8 structural edge pairs (12 undirected per element)
C3D8_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# ── Extrapolation split ───────────────────────────────────────────────────────

# All 4 extreme boundaries (union, 36 unique samples):
#   E_min   (UF E=3000):    S0001-S0010
#   E_max   (UF E=12000):   S0091-S0100
#   CTE_min (UF CTE=2.5e-5): S0001,S0011,S0021,S0031,S0041,S0051,S0061,S0071,S0081,S0091
#   CTE_max (UF CTE=3.4e-5): S0010,S0020,S0030,S0040,S0050,S0060,S0070,S0080,S0090,S0100
TEST_INDICES = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,           # E_min block
    11, 20, 21, 30, 31, 40, 41, 50,            # CTE extremes (mid-E)
    51, 60, 61, 70, 71, 80, 81, 90,            # CTE extremes (mid-E)
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100,  # E_max block
}

TEST_IDS  = sorted([f"S{i:04d}" for i in TEST_INDICES])
TRAIN_IDS = sorted([f"S{i:04d}" for i in range(1, 221) if i not in TEST_INDICES])
ALL_IDS   = sorted([f"S{i:04d}" for i in range(1, 221)])

print(f"Train: {len(TRAIN_IDS)} samples")
print(f"Test:  {len(TEST_IDS)} samples (extreme UF CTE/E)")
print(f"Test IDs: {TEST_IDS}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_npz(sid):
    path = os.path.join(INPUT_DIR, f"{sid}.npz")
    return np.load(path) if os.path.exists(path) else None


def build_singlegraph(elem_conn):
    seen = set()
    rows, cols = [], []
    for c0 in elem_conn:
        for a, b in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                if (src, dst) not in seen:
                    seen.add((src, dst))
                    rows.append(src)
                    cols.append(dst)
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
    M  = len(elem_mat)
    oh = np.zeros((M, 3), dtype=np.float32)
    oh[:, 0] = ((elem_mat == 0) | (elem_mat == 1)).astype(np.float32)
    oh[:, 1] = (elem_mat == 2).astype(np.float32)
    oh[:, 2] = (elem_mat == 3).astype(np.float32)
    return oh


def robust_stats(arr, axis=0):
    mean = arr.mean(axis=axis).astype(np.float32)
    std  = arr.std(axis=axis).astype(np.float32)
    std  = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


# ── Static graph topology from first training sample ─────────────────────────

_d0         = load_npz(TRAIN_IDS[0])
_edge_index  = build_singlegraph(_d0["elem_conn"])
_n2e, _e2n  = build_b02_edges(_d0["elem_conn"])
print(f"\nMesh: N={_d0['mesh_pos'].shape[0]} nodes  "
      f"M={_d0['elem_conn'].shape[0]} elems  "
      f"n2n={_edge_index.shape[1]} edges  "
      f"B02={_n2e.shape[1]} edges")


# ── Pass 1: normalization stats from TRAINING set only ───────────────────────

print(f"\nPass 1: collecting stats from {len(TRAIN_IDS)} training samples ...")

all_disp    = []
all_vel     = []
all_vm_Si   = []
all_vm_Sold = []
all_vm_UF   = []
all_peeq_sl = []
all_CTE     = []
all_nu      = []
all_ef      = []
all_E       = []      # all elements, for global min/max (used by Si/Solder)
all_E_UF    = []      # UF elements only, for UF-specific min-max
all_CTE_UF  = []      # UF elements only, for UF-specific min-max

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
    elem_E         = d["elem_E"]
    elem_CTE       = d["elem_CTE"]
    elem_nu        = d["elem_nu"]

    T = velocity.shape[0]
    si_mask     = (elem_mat == 0) | (elem_mat == 1)
    solder_mask = elem_mat == 2
    uf_mask     = elem_mat == 3

    for t in range(1, T):
        disp_t = (world_pos[t] - mesh_pos).astype(np.float32)
        all_disp.append(disp_t)
        all_vel.append(velocity[t - 1].astype(np.float32))
        all_ef.append(compute_edge_feats(mesh_pos, world_pos[t], _edge_index))

    for fi in range(1, T + 1):
        all_vm_Si.append(elem_vm_stress[fi, si_mask].astype(np.float32))
        all_vm_Sold.append(elem_vm_stress[fi, solder_mask].astype(np.float32))
        all_vm_UF.append(elem_vm_stress[fi, uf_mask].astype(np.float32))
        all_peeq_sl.append(elem_peeq[fi, solder_mask].astype(np.float32))

    all_CTE.append(elem_CTE.astype(np.float32))
    all_nu.append(elem_nu.astype(np.float32))
    all_E.append(elem_E.astype(np.float32))
    all_E_UF.append(elem_E[uf_mask].astype(np.float32))
    all_CTE_UF.append(elem_CTE[uf_mask].astype(np.float32))

# Global E range (all elements) — used for Si/Solder normalization
E_arr = np.concatenate(all_E)
E_MIN = float(E_arr.min())
E_MAX = float(E_arr.max())

# UF-only E range from training — used for UF normalization
UF_E_arr  = np.concatenate(all_E_UF)
UF_E_MIN  = float(UF_E_arr.min())
UF_E_MAX  = float(UF_E_arr.max())

# UF-only CTE range from training
UF_CTE_arr = np.concatenate(all_CTE_UF)
UF_CTE_MIN = float(UF_CTE_arr.min())
UF_CTE_MAX = float(UF_CTE_arr.max())

print(f"  Global E range:  E_min={E_MIN:.1f}  E_max={E_MAX:.1f} MPa")
print(f"  UF E range (train):   {UF_E_MIN:.1f} ~ {UF_E_MAX:.1f} MPa")
print(f"    test UF E=3000  -> E_norm = {(3000-UF_E_MIN)/(UF_E_MAX-UF_E_MIN):.4f}  (< 0 = extrapolation)")
print(f"    test UF E=12000 -> E_norm = {(12000-UF_E_MIN)/(UF_E_MAX-UF_E_MIN):.4f}  (> 1 = extrapolation)")
print(f"  UF CTE range (train): {UF_CTE_MIN:.4e} ~ {UF_CTE_MAX:.4e}")
print(f"    test CTE_min=2.5e-5 -> CTE_norm = {(2.5e-5-UF_CTE_MIN)/(UF_CTE_MAX-UF_CTE_MIN):.4f}  (< 0 = extrapolation)")
print(f"    test CTE_max=3.4e-5 -> CTE_norm = {(3.4e-5-UF_CTE_MIN)/(UF_CTE_MAX-UF_CTE_MIN):.4f}  (> 1 = extrapolation)")

disp_mean, disp_std = robust_stats(np.concatenate(all_disp, axis=0))
vel_mean,  vel_std  = robust_stats(np.concatenate(all_vel,  axis=0))
edge_mean, edge_std = robust_stats(np.concatenate(all_ef,   axis=0))

vm_Si_arr   = np.concatenate(all_vm_Si,   axis=0)
vm_Sold_arr = np.concatenate(all_vm_Sold, axis=0)
vm_UF_arr   = np.concatenate(all_vm_UF,  axis=0)
peeq_sl_arr = np.concatenate(all_peeq_sl, axis=0)
CTE_arr     = np.concatenate(all_CTE, axis=0)
nu_arr      = np.concatenate(all_nu,  axis=0)

vm_mean_Si,     vm_std_Si     = float(vm_Si_arr.mean()),   max(float(vm_Si_arr.std()),   EPS)
vm_mean_Solder, vm_std_Solder = float(vm_Sold_arr.mean()), max(float(vm_Sold_arr.std()), EPS)
vm_mean_UF,     vm_std_UF     = float(vm_UF_arr.mean()),   max(float(vm_UF_arr.std()),   EPS)
peeq_mean_sl,   peeq_std_sl   = float(peeq_sl_arr.mean()), max(float(peeq_sl_arr.std()), EPS)
CTE_mean,       CTE_std       = float(CTE_arr.mean()),     max(float(CTE_arr.std()),     EPS)
nu_mean,        nu_std        = float(nu_arr.mean()),      max(float(nu_arr.std()),      EPS)

stats = {
    "disp_mean":        disp_mean.tolist(),
    "disp_std":         disp_std.tolist(),
    "vel_mean":         vel_mean.tolist(),
    "vel_std":          vel_std.tolist(),
    "vm_mean_Si":       vm_mean_Si,
    "vm_std_Si":        vm_std_Si,
    "vm_mean_Solder":   vm_mean_Solder,
    "vm_std_Solder":    vm_std_Solder,
    "vm_mean_UF":       vm_mean_UF,
    "vm_std_UF":        vm_std_UF,
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
    "UF_E_min":         UF_E_MIN,
    "UF_E_max":         UF_E_MAX,
    "UF_CTE_min":       UF_CTE_MIN,
    "UF_CTE_max":       UF_CTE_MAX,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
stats_path = os.path.join(OUTPUT_DIR, "stats_m0040_extrapol.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Stats saved -> {stats_path}")
print(f"  vm_stress Si:     mean={vm_mean_Si:.2f}  std={vm_std_Si:.2f} MPa")
print(f"  vm_stress Solder: mean={vm_mean_Solder:.2f}  std={vm_std_Solder:.2f} MPa")
print(f"  vm_stress UF:     mean={vm_mean_UF:.2f}  std={vm_std_UF:.2f} MPa")
print(f"  peeq solder:      mean={peeq_mean_sl:.4e}  std={peeq_std_sl:.4e}")
print(f"  CTE:              mean={CTE_mean:.3e}  std={CTE_std:.3e}")
print(f"  nu:               mean={nu_mean:.4f}    std={nu_std:.4f}")


# ── Pass 2: build and save HeteroData .pt files for ALL 220 samples ───────────

print(f"\nPass 2: building HeteroData .pt files for all {len(ALL_IDS)} samples ...")

VM_MEAN = {0: vm_mean_Si, 1: vm_mean_Si, 2: vm_mean_Solder, 3: vm_mean_UF}
VM_STD  = {0: vm_std_Si,  1: vm_std_Si,  2: vm_std_Solder,  3: vm_std_UF}

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

    mat_oh   = mat_onehot(elem_mat)
    uf_e_mask = (elem_mat == 3)

    # E: global min-max for Si/Solder (fixed values); UF-specific min-max for UF
    E_norm = ((elem_E - E_MIN) / (E_MAX - E_MIN)).astype(np.float32)
    E_norm[uf_e_mask] = ((elem_E[uf_e_mask] - UF_E_MIN) / (UF_E_MAX - UF_E_MIN)).astype(np.float32)
    E_norm = E_norm[:, None]

    # CTE: global z-score for Si/Solder (fixed values); UF-specific min-max for UF
    CTE_n = ((elem_CTE - CTE_mean) / CTE_std).astype(np.float32)
    CTE_n[uf_e_mask] = ((elem_CTE[uf_e_mask] - UF_CTE_MIN) / (UF_CTE_MAX - UF_CTE_MIN)).astype(np.float32)
    CTE_n = CTE_n[:, None]

    nu_n    = ((elem_nu  - nu_mean)  / nu_std).astype(np.float32)[:, None]

    vm_mean_e = np.array([VM_MEAN[int(m)] for m in elem_mat], dtype=np.float32)
    vm_std_e  = np.array([VM_STD[int(m)]  for m in elem_mat], dtype=np.float32)

    nt_oh = node_type_onehot(node_type, n_classes=5)

    steps = []

    for t in range(1, T):
        disp_t = (world_pos[t] - mesh_pos).astype(np.float32)
        prev_v = velocity[t - 1].astype(np.float32)
        disp_n = (disp_t - disp_mean) / disp_std
        prev_n = (prev_v - vel_mean)  / vel_std
        node_x = np.concatenate([disp_n, prev_n, nt_oh], axis=1)

        vel_n  = (velocity[t].astype(np.float32) - vel_mean) / vel_std

        vm_n   = ((elem_vm_stress[t].astype(np.float32) - vm_mean_e) / vm_std_e)[:, None]
        pq_n   = np.zeros(M, dtype=np.float32)
        pq_n[solder_mask] = (
            (elem_peeq[t, solder_mask].astype(np.float32) - peeq_mean_sl) / peeq_std_sl
        )
        elem_x = np.concatenate([mat_oh, E_norm, CTE_n, nu_n, vm_n, pq_n[:, None]], axis=1)

        vm_n1  = ((elem_vm_stress[t+1].astype(np.float32) - vm_mean_e) / vm_std_e)[:, None]
        pq_n1  = np.zeros(M, dtype=np.float32)
        pq_n1[solder_mask] = (
            (elem_peeq[t+1, solder_mask].astype(np.float32) - peeq_mean_sl) / peeq_std_sl
        )
        elem_y = np.concatenate([vm_n1, pq_n1[:, None]], axis=1)

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
print(f"  Train: {len(TRAIN_IDS)} samples  Test: {len(TEST_IDS)} samples")
print(f"  Steps per sample: {T-1}  (t=1..{T-1})")
