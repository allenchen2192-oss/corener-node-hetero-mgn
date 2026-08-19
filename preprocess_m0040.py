"""
Preprocess M0040 NPZ -> HeteroData .pt for HeteroMGN training.

Architecture: Bipartite Heterogeneous GNN
  Corner nodes ('node'):    displacement/velocity features
  Element nodes ('element'): material/stress/PEEQ features
  n2n edges ('mesh'):       C3D8 single graph, 8D features
  B02 edges ('in','has'):   node<->element, no features

Corner node features (11D):
  disp(3) + prev_vel(3) + node_type_oh(5)
  node_type: 0=free, 1=XSYMM, 2=YSYMM, 3=XSYMM+YSYMM, 4=fully fixed

Element node features (8D):
  mat_oh(3) + E_norm(1) + CTE_norm(1) + nu_norm(1) + vm_stress_t(1) + peeq_t(1)
  mat_oh: [is_Si, is_Solder, is_UF]  (Si_bottom + Si_SoC merged)

Targets:
  Corner:  velocity       (3D)  z-score
  Element: [vm_stress, peeq] (2D)

Normalization:
  disp, vel     : z-score
  vm_stress     : per-material z-score (Si / Solder / UF)
  peeq          : solder-only z-score for solder elements; 0.0 for non-solder
  CTE, nu       : z-score
  E             : min-max  (3000 -> 130000 MPa)
  n2n edge (8D) : z-score

Split: S0001-S0200 train | S0201-S0220 test
Steps: skip t=0 (all zeros), use t=1..19 (19 steps per sample)

Usage:
    python preprocess_m0040.py           # skip existing .pt
    python preprocess_m0040.py --force   # overwrite all
"""

import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

INPUT_DIR  = "./02_abaqus_npz_m0040"
OUTPUT_DIR = "./04_preprocessed_hetero_m0040"

E_MIN  = 3000.0     # min UF Young's modulus [MPa]
E_MAX  = 130000.0   # Si Young's modulus [MPa]
EPS    = 1e-8
FORCE  = "--force" in sys.argv

# C3D8 structural edge pairs (12 undirected per element)
C3D8_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

TRAIN_IDS = [f"S{i:04d}" for i in range(1,   201)]   # S0001-S0200
TEST_IDS  = [f"S{i:04d}" for i in range(201, 221)]   # S0201-S0220
ALL_IDS   = TRAIN_IDS + TEST_IDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_npz(sid):
    path = os.path.join(INPUT_DIR, f"{sid}.npz")
    return np.load(path) if os.path.exists(path) else None


def build_singlegraph(elem_conn):
    """Directed single graph from C3D8 connectivity (deduplicated by (src,dst) pair)."""
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
    return np.array([rows, cols], dtype=np.int64)   # (2, E_edges)


def build_b02_edges(elem_conn):
    """B02 incidence edges: node->element (n2e) and element->node (e2n)."""
    M, npe   = elem_conn.shape
    elem_ids = np.repeat(np.arange(M, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    n2e = np.stack([node_ids, elem_ids])   # (2, M*8)
    e2n = np.stack([elem_ids, node_ids])   # (2, M*8)
    return n2e, e2n


def compute_edge_feats(mesh_pos, world_pos_t, edge_index):
    """8D raw edge features: d_mesh(3), n_mesh(1), d_world(3), n_world(1)."""
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world],
                          axis=1).astype(np.float32)   # (E_edges, 8)


def node_type_onehot(node_type, n_classes=5):
    """5-class one-hot for BC type: 0=free,1=XSYMM,2=YSYMM,3=corner,4=fixed."""
    N  = len(node_type)
    oh = np.zeros((N, n_classes), dtype=np.float32)
    for i, nt in enumerate(node_type):
        if 0 <= int(nt) < n_classes:
            oh[i, int(nt)] = 1.0
    return oh


def mat_onehot(elem_mat):
    """3-class one-hot: [is_Si, is_Solder, is_UF]. Si_bottom + Si_SoC merged."""
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


# ── Build static graph topology from first sample (same mesh for all 220) ─────

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
all_vm_Si   = []
all_vm_Sold = []
all_vm_UF   = []
all_peeq_sl = []   # solder elements only
all_CTE     = []
all_nu      = []
all_ef      = []

for sid in tqdm(TRAIN_IDS):
    d = load_npz(sid)
    if d is None:
        continue

    mesh_pos       = d["mesh_pos"]           # (N, 3)
    world_pos      = d["world_pos"]          # (21, N, 3)
    velocity       = d["velocity"]           # (20, N, 3)
    elem_vm_stress = d["elem_vm_stress"]     # (21, M)
    elem_peeq      = d["elem_peeq"]          # (21, M)
    elem_mat       = d["elem_mat"]           # (M,)
    elem_CTE       = d["elem_CTE"]           # (M,)
    elem_nu        = d["elem_nu"]            # (M,)

    T = velocity.shape[0]   # 20
    si_mask     = (elem_mat == 0) | (elem_mat == 1)
    solder_mask = elem_mat == 2
    uf_mask     = elem_mat == 3

    # Input frame stats: t=1..T-1 (skip t=0, all zeros)
    for t in range(1, T):
        disp_t = (world_pos[t] - mesh_pos).astype(np.float32)
        all_disp.append(disp_t)
        all_vel.append(velocity[t - 1].astype(np.float32))   # prev_vel = velocity[t-1]
        all_ef.append(compute_edge_feats(mesh_pos, world_pos[t], _edge_index))

    # vm_stress and peeq: frames 1..T (inputs 1..T-1 plus target frame T)
    for fi in range(1, T + 1):
        all_vm_Si.append(elem_vm_stress[fi, si_mask].astype(np.float32))
        all_vm_Sold.append(elem_vm_stress[fi, solder_mask].astype(np.float32))
        all_vm_UF.append(elem_vm_stress[fi, uf_mask].astype(np.float32))
        all_peeq_sl.append(elem_peeq[fi, solder_mask].astype(np.float32))

    # Static per-element properties (vary across samples for UF)
    all_CTE.append(elem_CTE.astype(np.float32))
    all_nu.append(elem_nu.astype(np.float32))

# Compute stats
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
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
stats_path = os.path.join(OUTPUT_DIR, "stats_m0040.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Stats saved -> {stats_path}")
print(f"  vm_stress Si:     mean={vm_mean_Si:.2f}  std={vm_std_Si:.2f} MPa")
print(f"  vm_stress Solder: mean={vm_mean_Solder:.2f}  std={vm_std_Solder:.2f} MPa")
print(f"  vm_stress UF:     mean={vm_mean_UF:.2f}  std={vm_std_UF:.2f} MPa")
print(f"  peeq solder:      mean={peeq_mean_sl:.4e}  std={peeq_std_sl:.4e}")
print(f"  CTE:              mean={CTE_mean:.3e}  std={CTE_std:.3e}")
print(f"  nu:               mean={nu_mean:.4f}    std={nu_std:.4f}")
print(f"  disp_std: {disp_std}")
print(f"  vel_std:  {vel_std}")


# ── Pass 2: build and save HeteroData .pt files ───────────────────────────────

print(f"\nPass 2: building HeteroData .pt files ...")

# Per-material vm_stress normalization lookup
VM_MEAN = {0: vm_mean_Si, 1: vm_mean_Si, 2: vm_mean_Solder, 3: vm_mean_UF}
VM_STD  = {0: vm_std_Si,  1: vm_std_Si,  2: vm_std_Solder,  3: vm_std_UF}

# Static graph topology tensors (shared across all samples)
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

    mesh_pos       = d["mesh_pos"]           # (N, 3)
    world_pos      = d["world_pos"]          # (21, N, 3)
    velocity       = d["velocity"]           # (20, N, 3)
    elem_vm_stress = d["elem_vm_stress"]     # (21, M)
    elem_peeq      = d["elem_peeq"]          # (21, M)
    elem_mat       = d["elem_mat"]           # (M,)
    elem_E         = d["elem_E"]             # (M,)
    elem_CTE       = d["elem_CTE"]           # (M,)
    elem_nu        = d["elem_nu"]            # (M,)
    node_type      = d["node_type"]          # (N,)

    T = velocity.shape[0]   # 20
    M = elem_mat.shape[0]

    solder_mask = elem_mat == 2

    # ── Static element features (same for all timesteps within a sample) ──────
    mat_oh  = mat_onehot(elem_mat)                                                # (M, 3)
    E_norm  = ((elem_E   - E_MIN)    / (E_MAX - E_MIN)).astype(np.float32)[:, None]  # (M, 1)
    CTE_n   = ((elem_CTE - CTE_mean) / CTE_std).astype(np.float32)[:, None]     # (M, 1)
    nu_n    = ((elem_nu  - nu_mean)  / nu_std).astype(np.float32)[:, None]      # (M, 1)

    # Per-element normalization arrays (vectorised lookup)
    vm_mean_e = np.array([VM_MEAN[int(m)] for m in elem_mat], dtype=np.float32)  # (M,)
    vm_std_e  = np.array([VM_STD[int(m)]  for m in elem_mat], dtype=np.float32)  # (M,)

    # ── Static corner node features ───────────────────────────────────────────
    nt_oh = node_type_onehot(node_type, n_classes=5)   # (N, 5)

    steps = []

    for t in range(1, T):   # t = 1 .. 19  (skip t=0)

        # Corner node features (11D)
        disp_t  = (world_pos[t] - mesh_pos).astype(np.float32)
        prev_v  = velocity[t - 1].astype(np.float32)
        disp_n  = (disp_t - disp_mean) / disp_std
        prev_n  = (prev_v - vel_mean)  / vel_std
        node_x  = np.concatenate([disp_n, prev_n, nt_oh], axis=1)   # (N, 11)

        # Corner node targets (3D)
        vel_n   = (velocity[t].astype(np.float32) - vel_mean) / vel_std  # (N, 3)

        # Element node features (8D)
        vm_n    = ((elem_vm_stress[t].astype(np.float32) - vm_mean_e) / vm_std_e)[:, None]  # (M, 1)
        pq_n    = np.zeros(M, dtype=np.float32)
        pq_n[solder_mask] = (
            (elem_peeq[t, solder_mask].astype(np.float32) - peeq_mean_sl) / peeq_std_sl
        )
        elem_x  = np.concatenate([mat_oh, E_norm, CTE_n, nu_n, vm_n, pq_n[:, None]],
                                  axis=1)   # (M, 8)

        # Element node targets (2D)
        vm_n1   = ((elem_vm_stress[t+1].astype(np.float32) - vm_mean_e) / vm_std_e)[:, None]
        pq_n1   = np.zeros(M, dtype=np.float32)
        pq_n1[solder_mask] = (
            (elem_peeq[t+1, solder_mask].astype(np.float32) - peeq_mean_sl) / peeq_std_sl
        )
        elem_y  = np.concatenate([vm_n1, pq_n1[:, None]], axis=1)   # (M, 2)

        # n2n edge features (8D)
        ef_raw  = compute_edge_feats(mesh_pos, world_pos[t], _edge_index)
        ef_norm = ((ef_raw - edge_mean) / edge_std).astype(np.float32)

        # Build HeteroData
        hd = HeteroData()

        hd["node"].x    = torch.from_numpy(node_x)    # (N, 11)
        hd["node"].y    = torch.from_numpy(vel_n)     # (N, 3)

        hd["element"].x = torch.from_numpy(elem_x)   # (M, 8)
        hd["element"].y = torch.from_numpy(elem_y)   # (M, 2)

        hd["node",    "mesh", "node"   ].edge_index = edge_index_t
        hd["node",    "mesh", "node"   ].edge_attr  = torch.from_numpy(ef_norm)
        hd["node",    "in",   "element"].edge_index = n2e_t
        hd["element", "has",  "node"   ].edge_index = e2n_t

        # Metadata for rollout and validation
        hd.mesh_pos                = torch.from_numpy(mesh_pos.astype(np.float32))
        hd.world_pos               = torch.from_numpy(world_pos[t].astype(np.float32))
        hd.node_type               = torch.from_numpy(node_type.astype(np.int64))
        hd.elem_mat                = torch.from_numpy(elem_mat.astype(np.int64))
        hd["element"].solder_mask  = torch.from_numpy(solder_mask)   # element-level: PyG concat correctly when batching
        hd.step_index              = torch.tensor([t], dtype=torch.long)

        steps.append(hd)

    torch.save(steps, pt_path)

print(f"\nDone. {len(ALL_IDS)} samples -> {OUTPUT_DIR}/")
print(f"  Train: {TRAIN_IDS[0]}..{TRAIN_IDS[-1]}  ({len(TRAIN_IDS)} samples)")
print(f"  Test:  {TEST_IDS[0]}..{TEST_IDS[-1]}   ({len(TEST_IDS)} samples)")
print(f"  Steps per sample: {T-1}  (t=1..{T-1})")
print(f"  Node features: 11D | Element features: 8D | Edge features: 8D")
