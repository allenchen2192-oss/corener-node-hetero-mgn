"""
Preprocess M0040 NPZ -> corner-node-only multigraph .pt for MeshGraphNet training.

Architecture: Single-type corner node multigraph (no element nodes)
  Node features (13D):
    disp(3) + prev_vel(3) + node_type_oh(5) + prev_vm_stress_node(1) + prev_peeq_node(1)
    prev_vm_stress_node : elem_to_node average over ALL adjacent elements, global z-score
    prev_peeq_node      : average over SOLDER-adjacent elements only; 0 for non-solder nodes
  Edge features (14D):
    geom(8): d_mesh(3) + n_mesh(1) + d_world(3) + n_world(1)
    mat(6) : mat_type_oh(3) + E_norm(1) + CTE_norm(1) + nu_norm(1)
  Multigraph: key=(src, dst, mat_id) -- interface edges appear with different material features

Targets (5D per node):
  vel(3) + vm_stress_node(1) + peeq_node(1)
  peeq loss masked to solder_node_mask in training

Normalization:
  disp, vel         : z-score
  vm_stress_node    : global z-score (all nodes / timesteps / train samples)
  peeq_node         : solder-only z-score (same definition as HeteroMGN)
  E                 : min-max  [3000, 130000] MPa
  CTE, nu           : z-score
  edge geom (8D)    : z-score
  mat_type_oh       : no normalization

Split: S0001-S0200 train | S0201-S0220 test
Steps: skip t=0 (all zeros), use t=1..19 (input) / t=2..20 (target)
"""

import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

INPUT_DIR  = "./02_abaqus_npz_m0040"
OUTPUT_DIR = "./04_preprocessed_cornernode_m0040"

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_npz(sid):
    path = os.path.join(INPUT_DIR, f"{sid}.npz")
    return np.load(path) if os.path.exists(path) else None


def elem_to_node_mean(elem_vals, elem_conn, N):
    """Average element scalar values onto nodes (all adjacent elements)."""
    node_sum = np.zeros(N, dtype=np.float64)
    node_cnt = np.zeros(N, dtype=np.int64)
    np.add.at(node_sum, elem_conn.ravel(), np.repeat(elem_vals, 8))
    np.add.at(node_cnt, elem_conn.ravel(), np.ones(elem_conn.size, dtype=np.int64))
    return (node_sum / np.maximum(node_cnt, 1)).astype(np.float32)


def solder_to_node(elem_vals, elem_conn, solder_mask, N):
    """Average scalar from solder elements only onto their corner nodes. Non-solder nodes get 0."""
    node_sum = np.zeros(N, dtype=np.float64)
    node_cnt = np.zeros(N, dtype=np.int64)
    sol_conn = elem_conn[solder_mask]
    np.add.at(node_sum, sol_conn.ravel(), np.repeat(elem_vals[solder_mask], 8))
    np.add.at(node_cnt, sol_conn.ravel(), np.ones(sol_conn.size, dtype=np.int64))
    out = np.zeros(N, dtype=np.float32)
    has = node_cnt > 0
    out[has] = (node_sum[has] / node_cnt[has]).astype(np.float32)
    return out


def build_multigraph_topology(elem_conn, elem_mat):
    """
    Build directed multigraph edges keyed by (src, dst, mat_id).
    Returns:
      edge_index : (2, E_mg)  int64
      elem_idx   : (E_mg,)    int64  -- which element generated each edge
      mat_ids    : (E_mg,)    int64  -- mat_id of that element (0=Si_bot,1=Si_SoC,2=Solder,3=UF)
    """
    seen = {}
    rows, cols, elem_idx_list, mat_id_list = [], [], [], []
    for ei, conn in enumerate(elem_conn):
        mat_id = int(elem_mat[ei])
        for a, b in C3D8_EDGES:
            u, v = int(conn[a]), int(conn[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst, mat_id)
                if key not in seen:
                    seen[key] = len(rows)
                    rows.append(src)
                    cols.append(dst)
                    elem_idx_list.append(ei)
                    mat_id_list.append(mat_id)
    edge_index = np.array([rows, cols], dtype=np.int64)
    elem_idx   = np.array(elem_idx_list, dtype=np.int64)
    mat_ids    = np.array(mat_id_list,   dtype=np.int64)
    return edge_index, elem_idx, mat_ids


def build_mat_edge_feats(elem_idx, mat_ids, elem_E, elem_CTE, elem_nu,
                          CTE_mean, CTE_std, nu_mean, nu_std):
    """Per-sample material edge features (6D): mat_oh(3) + E_norm(1) + CTE_n(1) + nu_n(1)."""
    mat_oh = np.zeros((len(mat_ids), 3), dtype=np.float32)
    mat_oh[:, 0] = ((mat_ids == 0) | (mat_ids == 1)).astype(np.float32)
    mat_oh[:, 1] = (mat_ids == 2).astype(np.float32)
    mat_oh[:, 2] = (mat_ids == 3).astype(np.float32)

    E_arr   = elem_E[elem_idx].astype(np.float32)
    CTE_arr = elem_CTE[elem_idx].astype(np.float32)
    nu_arr  = elem_nu[elem_idx].astype(np.float32)

    E_norm = ((E_arr   - E_MIN)    / (E_MAX - E_MIN))[:, None]
    CTE_n  = ((CTE_arr - CTE_mean) / CTE_std)[:, None]
    nu_n   = ((nu_arr  - nu_mean)  / nu_std)[:, None]

    return np.concatenate([mat_oh, E_norm, CTE_n, nu_n], axis=1).astype(np.float32)


def compute_geom_feats(mesh_pos, world_pos_t, edge_index):
    """8D geometric edge features: d_mesh(3), n_mesh(1), d_world(3), n_world(1)."""
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


def node_type_onehot(node_type, n_classes=5):
    N  = len(node_type)
    oh = np.zeros((N, n_classes), dtype=np.float32)
    for i, nt in enumerate(node_type):
        if 0 <= int(nt) < n_classes:
            oh[i, int(nt)] = 1.0
    return oh


def robust_stats(arr):
    mean = arr.mean(axis=0).astype(np.float32)
    std  = arr.std(axis=0).astype(np.float32)
    std  = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


# ── Build static topology from first sample (mesh shared across all 220) ──────

_d0       = load_npz(TRAIN_IDS[0])
_elem_conn = _d0["elem_conn"]
_elem_mat  = _d0["elem_mat"]
N = int(_d0["mesh_pos"].shape[0])
M = int(_elem_conn.shape[0])

_edge_index, _elem_idx, _mat_ids = build_multigraph_topology(_elem_conn, _elem_mat)

# solder_node_mask: nodes adjacent to at least one solder element (static)
_solder_mask_elem = (_elem_mat == 2)
_solder_node_mask = np.zeros(N, dtype=bool)
_solder_node_mask[_elem_conn[_solder_mask_elem].ravel()] = True

print(f"Mesh: N={N} nodes  M={M} elems  multigraph edges={_edge_index.shape[1]}")
print(f"Solder elements: {_solder_mask_elem.sum()}  solder-adjacent nodes: {_solder_node_mask.sum()}")


# ── Pass 1: normalization stats from training set ─────────────────────────────

print(f"\nPass 1: collecting stats from {len(TRAIN_IDS)} training samples ...")

all_disp      = []
all_vel       = []
all_vm_node   = []   # node-level vm_stress, global
all_peeq_sl   = []   # solder-element-level peeq (same as HeteroMGN)
all_CTE       = []
all_nu        = []
all_geom      = []

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

    T = velocity.shape[0]   # 20
    solder_mask = (elem_mat == 2)

    for t in range(1, T):
        disp_t = (world_pos[t] - mesh_pos).astype(np.float32)
        all_disp.append(disp_t)
        all_vel.append(velocity[t - 1].astype(np.float32))
        all_geom.append(compute_geom_feats(mesh_pos, world_pos[t], _edge_index))

    # vm_stress_node and peeq: frames 1..T (inputs 1..T-1 plus target frame T)
    for fi in range(1, T + 1):
        vm_node = elem_to_node_mean(elem_vm_stress[fi].astype(np.float32), elem_conn=_elem_conn, N=N)
        all_vm_node.append(vm_node)
        all_peeq_sl.append(elem_peeq[fi, solder_mask].astype(np.float32))

    all_CTE.append(elem_CTE.astype(np.float32))
    all_nu.append(elem_nu.astype(np.float32))

disp_mean, disp_std     = robust_stats(np.concatenate(all_disp, axis=0))
vel_mean,  vel_std      = robust_stats(np.concatenate(all_vel,  axis=0))
geom_mean, geom_std     = robust_stats(np.concatenate(all_geom, axis=0))

vm_node_arr = np.concatenate(all_vm_node, axis=0)   # (samples*frames*N,)  -- 1D per node
vm_node_mean = float(vm_node_arr.mean())
vm_node_std  = max(float(vm_node_arr.std()), EPS)

peeq_sl_arr = np.concatenate(all_peeq_sl, axis=0)
peeq_mean_sl = float(peeq_sl_arr.mean())
peeq_std_sl  = max(float(peeq_sl_arr.std()), EPS)

CTE_arr   = np.concatenate(all_CTE, axis=0)
nu_arr    = np.concatenate(all_nu,  axis=0)
CTE_mean  = float(CTE_arr.mean());  CTE_std  = max(float(CTE_arr.std()),  EPS)
nu_mean   = float(nu_arr.mean());   nu_std   = max(float(nu_arr.std()),   EPS)

stats = {
    "disp_mean":        disp_mean.tolist(),
    "disp_std":         disp_std.tolist(),
    "vel_mean":         vel_mean.tolist(),
    "vel_std":          vel_std.tolist(),
    "vm_node_mean":     vm_node_mean,
    "vm_node_std":      vm_node_std,
    "peeq_mean_solder": peeq_mean_sl,
    "peeq_std_solder":  peeq_std_sl,
    "CTE_mean":         CTE_mean,
    "CTE_std":          CTE_std,
    "nu_mean":          nu_mean,
    "nu_std":           nu_std,
    "geom_mean":        geom_mean.tolist(),
    "geom_std":         geom_std.tolist(),
    "E_min":            E_MIN,
    "E_max":            E_MAX,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
stats_path = os.path.join(OUTPUT_DIR, "stats_cornernode_m0040.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Stats saved -> {stats_path}")
print(f"  vm_stress_node: mean={vm_node_mean:.2f}  std={vm_node_std:.2f} MPa")
print(f"  peeq solder:    mean={peeq_mean_sl:.4e}  std={peeq_std_sl:.4e}")
print(f"  CTE: mean={CTE_mean:.3e}  std={CTE_std:.3e}")
print(f"  nu:  mean={nu_mean:.4f}   std={nu_std:.4f}")


# ── Pass 2: build and save Data .pt files ─────────────────────────────────────

print(f"\nPass 2: building Data .pt files ...")

edge_index_t      = torch.from_numpy(_edge_index)
solder_node_mask_t = torch.from_numpy(_solder_node_mask)

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

    T = velocity.shape[0]   # 20
    solder_mask = (elem_mat == 2)

    nt_oh = node_type_onehot(node_type, n_classes=5)   # (N, 5)

    # Per-sample static material edge features (6D) -- E/CTE/nu vary per sample for UF
    mat_ef = build_mat_edge_feats(
        _elem_idx, _mat_ids, elem_E, elem_CTE, elem_nu,
        CTE_mean, CTE_std, nu_mean, nu_std,
    )   # (E_mg, 6)

    steps = []

    for t in range(1, T):   # t = 1 .. 19

        # ── Node features (13D) ───────────────────────────────────────────────
        disp_t  = (world_pos[t] - mesh_pos).astype(np.float32)
        prev_v  = velocity[t - 1].astype(np.float32)
        disp_n  = (disp_t - disp_mean) / disp_std
        prev_n  = (prev_v - vel_mean)  / vel_std

        vm_t    = elem_to_node_mean(elem_vm_stress[t].astype(np.float32), _elem_conn, N)
        vm_n    = ((vm_t - vm_node_mean) / vm_node_std)[:, None]               # (N, 1)

        pq_t    = solder_to_node(elem_peeq[t].astype(np.float32), _elem_conn, solder_mask, N)
        pq_n    = np.zeros(N, dtype=np.float32)
        pq_n[_solder_node_mask] = (
            (pq_t[_solder_node_mask] - peeq_mean_sl) / peeq_std_sl
        )

        node_x = np.concatenate([disp_n, prev_n, nt_oh, vm_n, pq_n[:, None]], axis=1)  # (N, 13)

        # ── Node targets (5D) ─────────────────────────────────────────────────
        vel_n   = (velocity[t].astype(np.float32) - vel_mean) / vel_std        # (N, 3)

        vm_t1   = elem_to_node_mean(elem_vm_stress[t + 1].astype(np.float32), _elem_conn, N)
        vm_n1   = ((vm_t1 - vm_node_mean) / vm_node_std)[:, None]

        pq_t1   = solder_to_node(elem_peeq[t + 1].astype(np.float32), _elem_conn, solder_mask, N)
        pq_n1   = np.zeros(N, dtype=np.float32)
        pq_n1[_solder_node_mask] = (
            (pq_t1[_solder_node_mask] - peeq_mean_sl) / peeq_std_sl
        )

        node_y = np.concatenate([vel_n, vm_n1, pq_n1[:, None]], axis=1)       # (N, 5)

        # ── Edge features (14D = geom8 + mat6) ───────────────────────────────
        geom_raw  = compute_geom_feats(mesh_pos, world_pos[t], _edge_index)
        geom_norm = ((geom_raw - geom_mean) / geom_std).astype(np.float32)
        edge_attr = np.concatenate([geom_norm, mat_ef], axis=1)                # (E_mg, 14)

        # ── Build Data ────────────────────────────────────────────────────────
        data = Data()
        data.x          = torch.from_numpy(node_x)        # (N, 13)
        data.y          = torch.from_numpy(node_y)        # (N, 5)
        data.edge_index = edge_index_t
        data.edge_attr  = torch.from_numpy(edge_attr)     # (E_mg, 14)

        data.solder_node_mask = solder_node_mask_t        # (N,) bool, static
        data.mesh_pos         = torch.from_numpy(mesh_pos.astype(np.float32))
        data.world_pos        = torch.from_numpy(world_pos[t].astype(np.float32))
        data.node_type        = torch.from_numpy(node_type.astype(np.int64))
        data.elem_mat         = torch.from_numpy(elem_mat.astype(np.int64))
        data.elem_conn        = torch.from_numpy(_elem_conn.astype(np.int64))
        data.step_index       = torch.tensor([t], dtype=torch.long)

        steps.append(data)

    torch.save(steps, pt_path)

print(f"\nDone. {len(ALL_IDS)} samples -> {OUTPUT_DIR}/")
print(f"  Train: {TRAIN_IDS[0]}..{TRAIN_IDS[-1]}  ({len(TRAIN_IDS)} samples)")
print(f"  Test:  {TEST_IDS[0]}..{TEST_IDS[-1]}   ({len(TEST_IDS)} samples)")
print(f"  Steps per sample: {T-1}  (t=1..{T-1})")
print(f"  Node features: 13D | Edge features: 14D | Multigraph edges: {_edge_index.shape[1]}")
