"""
Preprocess M0035 IC package NPZ → sample_*.pt for train_m0035.py.

Node features (17D):
  [disp(3), prev_vel(3), stress(6), peeq(1), mat_oh(3), CTE_norm(1)]
  - stress: per-material z-score (separate mean/std for Si, Solder, UF)
  - peeq:   z-score from solder nodes only (Si/UF always zero)
  - mat_oh: [is_Si, is_Solder, is_UF]  (merges Si_bottom + Si_SoC)
  - CTE_norm: per-node, varies per sample for underfill (E omitted: redundant with edge E)

Edge features (9D, Multigraph):
  [d_mesh(3), n_mesh(1), d_world(3), n_world(1), E_elem/E_ref(1)]
  Interface nodes receive multiple directed edges with different E values.

Targets (10D):
  [vel(3), stress_next(6), peeq_next(1)]
  vel        = world_pos[t+1] - world_pos[t]   (displacement increment)
  stress_next = Abaqus stress at frame t+1
  peeq_next   = Abaqus PEEQ at frame t+1

Usage:
    python preprocess_m0035.py           # skip existing .pt
    python preprocess_m0035.py --force   # overwrite existing .pt

Input:  ./02_abaqus_npz_m0035/S00xx.npz          (220 files)
Output: ./04_preprocessed_pt_m0035/
        - S0001.pt .. S0220.pt
        - node_stats_m0035.json
        - edge_stats_m0035.json
"""

import csv
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

INPUT_DIR  = "./02_abaqus_npz_m0035"
OUTPUT_DIR = "./04_preprocessed_pt_m0035"
CSV_PATH   = "./0035/full_validation.csv"

NODE_DIM   = 17   # disp(3)+prev_vel(3)+stress(6)+peeq(1)+mat_oh(3)+CTE_norm(1)
EDGE_DIM   = 9    # d_mesh(3)+n_mesh(1)+d_world(3)+n_world(1)+E_norm(1)
TARGET_DIM = 10   # vel(3)+stress_next(6)+peeq_next(1)

# Fixed material E [MPa] for non-UF materials (index matches NPZ elem_mat)
#   0=Si_bottom  1=Si_SoC  2=Solder  3=Underfill (per-sample)
MAT_E_FIXED = np.array([130000.0, 130000.0, 47000.0, 0.0], dtype=np.float64)
MAT_CTE     = np.array([3.1e-6,   3.1e-6,   22.5e-6, 0.0], dtype=np.float64)
E_REF       = 130000.0   # reference E for edge feature normalisation (Si)

EPS   = 1e-8
FORCE = "--force" in sys.argv

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


# ── Read CSV split info ────────────────────────────────────────────────────────
train_ids, test_ids = [], []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        sid = row["sample_id"]
        if row["split"] == "Train":
            train_ids.append(sid)
        else:
            test_ids.append(sid)

all_ids = train_ids + test_ids
print(f"Train: {len(train_ids)}  Test: {len(test_ids)}  Total: {len(all_ids)}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_multigraph(elem_conn, elem_mat, mat_E_sample):
    """
    Build directed multigraph from element connectivity.
    Interface nodes get multiple directed edges (one per adjacent material).
    mat_E_sample: (4,) array with sample-specific underfill E at index 3.
    Returns edge_index (2, E_mg), edge_E (E_mg,).
    """
    seen = {}   # (src, dst, mat_idx) -> E
    rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E_sample[mat_idx])
        for (a, b) in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst, int(mat_idx))
                if key not in seen:
                    seen[key] = True
                    rows.append(src)
                    cols.append(dst)
                    E_vals.append(E_elem)
    return (np.array([rows, cols], dtype=np.int64),
            np.array(E_vals, dtype=np.float32))


def compute_edge_feats(mesh_pos, world_pos_t, edge_index, edge_E):
    """9D raw edge features."""
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    E_col   = (edge_E / E_REF)[:, None]
    return np.concatenate([d_mesh, n_mesh, d_world, n_world, E_col],
                          axis=1).astype(np.float32)


def mat_onehot(node_mat):
    """3D one-hot: [is_Si, is_Solder, is_UF]. Merges Si_bottom + Si_SoC."""
    N  = node_mat.shape[0]
    oh = np.zeros((N, 3), dtype=np.float32)
    oh[:, 0] = ((node_mat == 0) | (node_mat == 1)).astype(np.float32)  # Si
    oh[:, 1] = (node_mat == 2).astype(np.float32)                       # Solder
    oh[:, 2] = (node_mat == 3).astype(np.float32)                       # UF
    return oh


def robust_stats(arr, axis=0):
    mean = arr.mean(axis=axis).astype(np.float32)
    std  = arr.std(axis=axis).astype(np.float32)
    std  = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


def load_npz(sid):
    path = os.path.join(INPUT_DIR, f"{sid}.npz")
    return np.load(path) if os.path.exists(path) else None


# ── Pass 1: normalization stats from training set ──────────────────────────────
print(f"\nPass 1: collecting stats from {len(train_ids)} training samples ...")

all_disp, all_vel, all_peeq = [], [], []
all_stress_Si, all_stress_Solder, all_stress_UF = [], [], []
all_CTE = []
all_ef  = []

for sid in tqdm(train_ids):
    d = load_npz(sid)
    if d is None:
        continue

    mesh_pos  = d["mesh_pos"]      # (N, 3)
    world_pos = d["world_pos"]     # (T, N, 3)  T=21
    velocity  = d["velocity"]      # (T-1, N, 3)
    stress    = d["stress"]        # (T, N, 6)
    peeq      = d["peeq"]         # (T, N)
    node_mat  = d["node_mat"]
    node_E    = d["node_E"]        # (N,) MPa
    node_CTE  = d["node_CTE"]     # (N,)
    elem_conn = d["elem_conn"]
    elem_mat  = d["elem_mat"]
    E_uf      = float(d["E_uf_MPa"])
    CTE_uf    = float(d["CTE_uf"])

    mat_E_s = MAT_E_FIXED.copy()
    mat_E_s[3] = E_uf

    edge_index, edge_E = build_multigraph(elem_conn, elem_mat, mat_E_s)
    si_mask     = (node_mat == 0) | (node_mat == 1)
    solder_mask = node_mat == 2
    uf_mask     = node_mat == 3

    T = velocity.shape[0]  # 20
    for t in range(T):
        disp_t = (world_pos[t] - mesh_pos).astype(np.float32)
        all_disp.append(disp_t)
        all_vel.append(velocity[t].astype(np.float32))
        all_stress_Si.append(stress[t][si_mask].astype(np.float32))
        all_stress_Solder.append(stress[t][solder_mask].astype(np.float32))
        all_stress_UF.append(stress[t][uf_mask].astype(np.float32))
        all_peeq.append(peeq[t][solder_mask].astype(np.float32)[:, None])
        all_ef.append(compute_edge_feats(mesh_pos, world_pos[t], edge_index, edge_E))

    # Include final frame (t=T) for stress/peeq stats — it appears as target at t=T-1
    all_stress_Si.append(stress[T][si_mask].astype(np.float32))
    all_stress_Solder.append(stress[T][solder_mask].astype(np.float32))
    all_stress_UF.append(stress[T][uf_mask].astype(np.float32))
    all_peeq.append(peeq[T][solder_mask].astype(np.float32)[:, None])

    all_CTE.append(node_CTE)

all_disp   = np.concatenate(all_disp,          axis=0)
all_vel    = np.concatenate(all_vel,           axis=0)
all_peeq   = np.concatenate(all_peeq,          axis=0)
all_CTE_arr= np.concatenate(all_CTE,           axis=0)
all_ef_arr = np.concatenate(all_ef,            axis=0)
stress_Si_arr     = np.concatenate(all_stress_Si,     axis=0)
stress_Solder_arr = np.concatenate(all_stress_Solder, axis=0)
stress_UF_arr     = np.concatenate(all_stress_UF,     axis=0)

disp_mean,   disp_std   = robust_stats(all_disp)
vel_mean,    vel_std    = robust_stats(all_vel)
peeq_mean,   peeq_std  = robust_stats(all_peeq)
CTE_mean,    CTE_std    = float(all_CTE_arr.mean()), float(all_CTE_arr.std())
edge_mean,   edge_std   = robust_stats(all_ef_arr)

stress_mean_Si,     stress_std_Si     = robust_stats(stress_Si_arr)
stress_mean_Solder, stress_std_Solder = robust_stats(stress_Solder_arr)
stress_mean_UF,     stress_std_UF     = robust_stats(stress_UF_arr)

if CTE_std < EPS: CTE_std = 1.0

node_stats = {
    "disp_mean":          disp_mean.tolist(),
    "disp_std":           disp_std.tolist(),
    "vel_mean":           vel_mean.tolist(),
    "vel_std":            vel_std.tolist(),
    "stress_mean_Si":     stress_mean_Si.tolist(),
    "stress_std_Si":      stress_std_Si.tolist(),
    "stress_mean_Solder": stress_mean_Solder.tolist(),
    "stress_std_Solder":  stress_std_Solder.tolist(),
    "stress_mean_UF":     stress_mean_UF.tolist(),
    "stress_std_UF":      stress_std_UF.tolist(),
    "peeq_mean":          float(peeq_mean[0]),
    "peeq_std":           float(peeq_std[0]),
    "CTE_mean":           CTE_mean,
    "CTE_std":            CTE_std,
}
edge_stats_out = {
    "edge_mean": edge_mean.tolist(),
    "edge_std":  edge_std.tolist(),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "node_stats_m0035.json"), "w") as f:
    json.dump(node_stats, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "edge_stats_m0035.json"), "w") as f:
    json.dump(edge_stats_out, f, indent=2)
with open("node_stats_m0035.json", "w") as f:
    json.dump(node_stats, f, indent=2)
with open("edge_stats_m0035.json", "w") as f:
    json.dump(edge_stats_out, f, indent=2)

print("Stats saved.")
print(f"  disp_std:   {disp_std}")
print(f"  vel_std:    {vel_std}")
print(f"  stress_std_Si:     {stress_std_Si}")
print(f"  stress_std_Solder: {stress_std_Solder}")
print(f"  stress_std_UF:     {stress_std_UF}")
print(f"  peeq_std:   {float(peeq_std[0]):.4e}  (solder nodes only)")
print(f"  CTE_mean={CTE_mean:.3e}  CTE_std={CTE_std:.3e}")


# ── Pass 2: build .pt files ────────────────────────────────────────────────────
print(f"\nPass 2: building .pt files ...")

disp_mean   = np.array(node_stats["disp_mean"], dtype=np.float32)
disp_std    = np.array(node_stats["disp_std"],  dtype=np.float32)
vel_mean    = np.array(node_stats["vel_mean"],  dtype=np.float32)
vel_std     = np.array(node_stats["vel_std"],   dtype=np.float32)
peeq_mean_v = float(node_stats["peeq_mean"])
peeq_std_v  = float(node_stats["peeq_std"])
CTE_mean_v  = float(node_stats["CTE_mean"])
CTE_std_v   = float(node_stats["CTE_std"])
edge_mean   = np.array(edge_stats_out["edge_mean"], dtype=np.float32)
edge_std    = np.array(edge_stats_out["edge_std"],  dtype=np.float32)

# Per-material stress stats (shape (6,) each)
stress_mean_by_mat = {
    "Si":     np.array(node_stats["stress_mean_Si"],     dtype=np.float32),
    "Solder": np.array(node_stats["stress_mean_Solder"], dtype=np.float32),
    "UF":     np.array(node_stats["stress_mean_UF"],     dtype=np.float32),
}
stress_std_by_mat = {
    "Si":     np.array(node_stats["stress_std_Si"],     dtype=np.float32),
    "Solder": np.array(node_stats["stress_std_Solder"], dtype=np.float32),
    "UF":     np.array(node_stats["stress_std_UF"],     dtype=np.float32),
}

for sid in tqdm(all_ids):
    pt_path = os.path.join(OUTPUT_DIR, f"{sid}.pt")
    if os.path.exists(pt_path) and not FORCE:
        continue

    d = load_npz(sid)
    if d is None:
        print(f"  MISSING: {sid}.npz")
        continue

    mesh_pos  = d["mesh_pos"]
    world_pos = d["world_pos"]     # (T, N, 3)
    velocity  = d["velocity"]      # (T-1, N, 3)
    stress    = d["stress"]        # (T, N, 6)
    peeq      = d["peeq"]         # (T, N)
    node_mat  = d["node_mat"]
    node_E    = d["node_E"]
    node_nu   = d["node_nu"]
    node_CTE  = d["node_CTE"]
    elem_conn = d["elem_conn"]
    elem_mat  = d["elem_mat"]
    mesh_ei   = d["mesh_edge_index"].astype(np.int64)
    E_uf      = float(d["E_uf_MPa"])
    CTE_uf    = float(d["CTE_uf"])

    N = mesh_pos.shape[0]
    T = velocity.shape[0]   # 20

    mat_E_s = MAT_E_FIXED.copy()
    mat_E_s[3] = E_uf

    edge_index, edge_E = build_multigraph(elem_conn, elem_mat, mat_E_s)

    # Constant per-node features (don't change across timesteps)
    mat_oh      = mat_onehot(node_mat)                                          # (N, 3)
    CTE_n       = ((node_CTE - CTE_mean_v) / CTE_std_v).astype(np.float32)    # (N,)
    si_mask     = (node_mat == 0) | (node_mat == 1)
    solder_mask = node_mat == 2
    uf_mask     = node_mat == 3

    # Pre-compute per-node stress mean/std arrays — set once, reused across all t
    stress_mean_node = np.empty((N, 6), dtype=np.float32)
    stress_std_node  = np.empty((N, 6), dtype=np.float32)
    stress_mean_node[si_mask]     = stress_mean_by_mat["Si"]
    stress_mean_node[solder_mask] = stress_mean_by_mat["Solder"]
    stress_mean_node[uf_mask]     = stress_mean_by_mat["UF"]
    stress_std_node[si_mask]      = stress_std_by_mat["Si"]
    stress_std_node[solder_mask]  = stress_std_by_mat["Solder"]
    stress_std_node[uf_mask]      = stress_std_by_mat["UF"]

    # Temperature change per frame (uniform field, scalar per frame)
    # Frame 0: T=250 (ref), Frame 20: T=20.  Linear ramp.
    # delta_T[t] = T_current[t] - T_ref = temperature[t, any_node] - 250
    # Store as scalar (uniform across nodes per frame from Abaqus static analysis)
    delta_T = (d["temperature"][:, 0] - 250.0).astype(np.float32)  # (T,)

    steps = []

    for t in range(T):
        # ── Input node features ────────────────────────────────────────────────
        disp_t    = (world_pos[t] - mesh_pos).astype(np.float32)        # (N, 3)
        prev_vel  = (velocity[t-1].astype(np.float32)
                     if t > 0 else np.zeros_like(velocity[0]))           # (N, 3)
        stress_t  = stress[t].astype(np.float32)                         # (N, 6)
        peeq_t    = peeq[t].astype(np.float32)[:, None]                  # (N, 1)

        # Normalize
        disp_n    = (disp_t   - disp_mean)   / disp_std                  # (N, 3)
        vel_n     = (prev_vel - vel_mean)     / vel_std                   # (N, 3)
        stress_n  = (stress_t - stress_mean_node) / stress_std_node        # (N, 6)
        peeq_n    = (peeq_t   - peeq_mean_v) / peeq_std_v               # (N, 1)

        node_x = np.concatenate(
            [disp_n, vel_n, stress_n, peeq_n,
             mat_oh, CTE_n[:, None]],
            axis=1).astype(np.float32)    # (N, 17)

        # ── Target ────────────────────────────────────────────────────────────
        vel_t1     = velocity[t].astype(np.float32)          # (N, 3)
        stress_t1  = stress[t+1].astype(np.float32)          # (N, 6)
        peeq_t1    = peeq[t+1].astype(np.float32)[:, None]   # (N, 1)

        vel_n_tgt     = (vel_t1    - vel_mean)    / vel_std
        stress_n_tgt  = (stress_t1 - stress_mean_node) / stress_std_node
        peeq_n_tgt    = (peeq_t1   - peeq_mean_v) / peeq_std_v

        y_norm = np.concatenate([vel_n_tgt, stress_n_tgt, peeq_n_tgt],
                                 axis=1).astype(np.float32)   # (N, 10)

        # ── Edge features ─────────────────────────────────────────────────────
        ef_raw  = compute_edge_feats(mesh_pos, world_pos[t], edge_index, edge_E)
        ef_norm = ((ef_raw - edge_mean) / edge_std).astype(np.float32)

        graph = Data(
            x          = torch.from_numpy(node_x),
            y          = torch.from_numpy(y_norm),
            edge_index = torch.from_numpy(edge_index),
            num_nodes  = N,
        )
        graph.mesh_pos        = torch.from_numpy(mesh_pos.astype(np.float32))
        graph.world_pos       = torch.from_numpy(world_pos[t].astype(np.float32))
        graph.node_mat        = torch.from_numpy(node_mat.astype(np.int64))
        graph.node_E          = torch.from_numpy(node_E.astype(np.float32))
        graph.node_nu         = torch.from_numpy(node_nu.astype(np.float32))
        graph.node_CTE        = torch.from_numpy(node_CTE.astype(np.float32))
        graph.mesh_edge_index = torch.from_numpy(mesh_ei)
        graph.step_index      = torch.tensor([t], dtype=torch.long)
        graph.delta_T         = torch.tensor([float(delta_T[t])],   dtype=torch.float32)
        graph.delta_T_next    = torch.tensor([float(delta_T[t+1])], dtype=torch.float32)
        graph.E_uf_MPa        = torch.tensor(E_uf, dtype=torch.float32)

        steps.append({
            "graph":               graph,
            "mesh_edge_features":  torch.from_numpy(ef_norm),
            "world_edge_features": torch.zeros(0, EDGE_DIM, dtype=torch.float32),
        })

    torch.save(steps, pt_path)

print(f"\nDone. {len(all_ids)} samples → {OUTPUT_DIR}/")
print(f"Node features: {NODE_DIM}D   Edge features: {EDGE_DIM}D   Targets: {TARGET_DIM}D")