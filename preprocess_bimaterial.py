"""
Preprocess bi-material beam NPZ → sample_*.pt for train_bimaterial.py.

Key difference from preprocess_abaqus.py:
  - Builds MULTIGRAPH from elem_conn + elem_mat (not a deduplicated edge_set).
    Interface nodes (J=2 layer) share edges with both Mat1 and Mat2 elements
    → those edges appear TWICE with different E values.
  - Edge features: 9D = [d_mesh(3), n_mesh(1), d_world(3), n_world(1), E_edge/E1(1)]
  - Node features: 9D = [disp_vec(3), prev_vel(3), node_type_oh(3)]
      node_type_oh: free=[1,0,0], clamped=[0,1,0], loaded=[0,0,1]

Usage:
    python preprocess_bimaterial.py           # skip existing .pt
    python preprocess_bimaterial.py --force   # overwrite existing .pt

Input:  ./02_abaqus_npz_bimaterial/sample_XXXXX.npz  (1100 files)
Output: ./04_preprocessed_pt_bimaterial/
        - sample_XXXXX.pt
        - node_stats_bimaterial.json
        - edge_stats_bimaterial.json

Train indices: 0-999   Test indices: 1000-1099
"""

import csv
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from fem_stress import compute_nodal_vm_stress, _D_batch, _gauss_c3d8, _gauss_stress_at

_GFDM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR   = "./02_abaqus_npz_bimaterial"
OUTPUT_DIR  = "./04_preprocessed_pt_bimaterial"
NUM_SAMPLES = 1100
NUM_TRAIN   = 1000      # indices 0-999
NODE_DIM    = 10        # disp(3) + prev_vel(3) + node_type_oh(3) + p_norm(1)
EDGE_DIM    = 9         # d_mesh(3) + n_mesh(1) + d_world(3) + n_world(1) + E_norm(1)
EPS         = 1e-8
FORCE       = "--force" in sys.argv

# C3D8: 12 undirected edges per element (local node index pairs)
C3D8_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


# ── Multigraph builder ─────────────────────────────────────────────────────────

def build_multigraph(elem_conn, elem_mat, mat_E):
    """
    Build directed Multigraph edges from element connectivity.

    Within-material edges are deduplicated (multiple elements sharing the same
    edge in the same material give ONE directed edge).
    Cross-material (interface) edges give TWO directed edges — one from each
    material — with different E values.

    Returns:
        edge_index : (2, E_mg) int64
        edge_E     : (E_mg,)  float32 — E of the element that generated each edge
    """
    seen = set()
    rows, cols, E_vals = [], [], []

    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E[mat_idx])
        for (a, b) in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst)
                if key not in seen:
                    seen.add(key)
                    rows.append(src)
                    cols.append(dst)
                    E_vals.append(E_elem)

    edge_index = np.array([rows, cols], dtype=np.int64)  # (2, E_mg)
    edge_E     = np.array(E_vals, dtype=np.float32)      # (E_mg,)
    return edge_index, edge_E


def compute_edge_feats(mesh_pos, world_pos_t, edge_index, edge_E, E_ref):
    """9D raw edge features: [d_mesh(3), n_mesh(1), d_world(3), n_world(1), E/E_ref(1)]."""
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]       - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst]    - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    E_col   = (edge_E / E_ref)[:, None]
    return np.concatenate([d_mesh, n_mesh, d_world, n_world, E_col],
                          axis=1).astype(np.float32)   # (E_mg, 9)


def node_type_onehot(node_type_arr):
    """Encode node_type (0=free, 1=clamped, 2=loaded) as 3D one-hot."""
    N = node_type_arr.shape[0]
    oh = np.zeros((N, 3), dtype=np.float32)
    for i, nt in enumerate(node_type_arr):
        oh[i, int(nt)] = 1.0
    return oh   # (N, 3)


def compute_gfdm_vm_stress_batch(mesh_pos, world_pos_batch, mesh_ei, node_E, nu=0.33):
    """
    Compute GFDM von Mises stress for multiple timesteps using PyTorch.

    Args:
        mesh_pos       : (N, 3)     numpy - reference positions
        world_pos_batch: (T, N, 3)  numpy - deformed positions at T timesteps
        mesh_ei        : (2, E)     numpy int64 - deduplicated edge index
        node_E         : (N,)       numpy float32 - per-node Young's modulus
        nu             : Poisson's ratio

    Returns:
        (T, N) numpy float32 von Mises stress
    """
    dev = _GFDM_DEVICE
    ref  = torch.from_numpy(mesh_pos.astype(np.float32)).to(dev)          # (N, 3)
    wp   = torch.from_numpy(world_pos_batch.astype(np.float32)).to(dev)   # (T, N, 3)
    row  = torch.from_numpy(mesh_ei[0]).long().to(dev)
    col  = torch.from_numpy(mesh_ei[1]).long().to(dev)
    E_n  = torch.from_numpy(node_E.astype(np.float32)).to(dev)

    T, N, E_cnt = wp.shape[0], wp.shape[1], row.shape[0]

    r_ij = ref[col] - ref[row]                                            # (E, 3)
    r2   = r_ij.pow(2).sum(-1, keepdim=True).clamp(min=1e-12)            # (E, 1)
    w    = 1.0 / r2                                                        # (E, 1)

    # XtX — geometry only, same for all T
    r_r      = w.unsqueeze(-1) * r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2) # (E, 3, 3)
    row_e    = row.view(-1, 1, 1).expand_as(r_r)
    XtX      = torch.zeros(N, 3, 3, device=dev)
    XtX.scatter_add_(0, row_e, r_r)
    XtX      = XtX + 1e-6 * torch.eye(3, device=dev).unsqueeze(0)        # (N, 3, 3)

    # Batched XtY across all T timesteps
    u_batch  = wp - ref.unsqueeze(0)                                      # (T, N, 3)
    du_batch = u_batch[:, col] - u_batch[:, row]                          # (T, E, 3)

    w_b  = w.unsqueeze(0).unsqueeze(-1)                                   # (1, E, 1, 1)
    r_b  = r_ij.unsqueeze(0).unsqueeze(-1)                                # (1, E, 3, 1)
    du_b = du_batch.unsqueeze(-2)                                          # (T, E, 1, 3)
    r_du = w_b * r_b * du_b                                               # (T, E, 3, 3)

    # Scatter (T, E, 3, 3) → (T, N, 3, 3)
    r_du_flat = r_du.reshape(T * E_cnt, 3, 3)
    row_rep   = (row.unsqueeze(0).expand(T, -1)
                 + torch.arange(T, device=dev).unsqueeze(1) * N).reshape(-1)
    row_exp_b = row_rep.view(-1, 1, 1).expand_as(r_du_flat)
    XtY_flat  = torch.zeros(T * N, 3, 3, device=dev)
    XtY_flat.scatter_add_(0, row_exp_b, r_du_flat)
    XtY = XtY_flat.reshape(T, N, 3, 3)                                    # (T, N, 3, 3)

    # Solve: dU = XtX^{-1} @ XtY, then transpose last two dims
    XtX_exp = XtX.unsqueeze(0).expand(T, -1, -1, -1)                     # (T, N, 3, 3)
    dU = torch.linalg.solve(XtX_exp, XtY).permute(0, 1, 3, 2)            # (T, N, 3, 3)

    lam = E_n * nu / ((1 + nu) * (1 - 2 * nu))   # (N,)
    mu  = E_n / (2 * (1 + nu))

    tr  = dU[:, :, 0, 0] + dU[:, :, 1, 1] + dU[:, :, 2, 2]
    sx  = lam * tr + 2 * mu * dU[:, :, 0, 0]
    sy  = lam * tr + 2 * mu * dU[:, :, 1, 1]
    sz  = lam * tr + 2 * mu * dU[:, :, 2, 2]
    txy = mu * (dU[:, :, 0, 1] + dU[:, :, 1, 0])
    txz = mu * (dU[:, :, 0, 2] + dU[:, :, 2, 0])
    tyz = mu * (dU[:, :, 1, 2] + dU[:, :, 2, 1])

    vm = torch.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                            + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)
    return vm.cpu().numpy().astype(np.float32)   # (T, N)


def compute_bmat_vm_stress_batch(mesh_pos, world_pos_batch, elem_conn, elem_E, elem_nu):
    """
    B-matrix von Mises stress for T timesteps (Gauss-to-node extrapolation).

    mesh_pos:        (N, 3)    numpy float32 - reference positions
    world_pos_batch: (T, N, 3) numpy float32 - deformed positions
    elem_conn:       (E, 8)    numpy int64
    elem_E:          (E,)      numpy float64 - per-element Young's modulus
    elem_nu:         (E,)      numpy float64 - per-element Poisson's ratio
    Returns: (T, N) numpy float32
    """
    dev = _GFDM_DEVICE
    ref_t  = torch.from_numpy(mesh_pos.astype(np.float32)).to(dev)
    ec_t   = torch.from_numpy(elem_conn.astype(np.int64)).to(dev)
    eE_t   = torch.from_numpy(elem_E.astype(np.float64)).to(dev)
    enu_t  = torch.from_numpy(elem_nu.astype(np.float64)).to(dev)
    T      = world_pos_batch.shape[0]
    result = np.empty((T, mesh_pos.shape[0]), dtype=np.float32)
    for t in range(T):
        disp_t = torch.from_numpy(
            (world_pos_batch[t] - mesh_pos).astype(np.float32)
        ).to(dev)
        vm = compute_nodal_vm_stress(
            disp_t, ref_t, ec_t, npe=8, elem_E=eE_t, elem_nu=enu_t
        )
        result[t] = vm.cpu().numpy()
    return result   # (T, N)


def compute_elem_vm_stress_batch(mesh_pos, world_pos_batch, elem_conn, elem_E, elem_nu):
    """
    Per-element von Mises stress for T timesteps via C3D8 Gauss-point averaging.

    mesh_pos:        (N, 3)    numpy float32
    world_pos_batch: (T, N, 3) numpy float32
    elem_conn:       (E, 8)    numpy int64
    elem_E:          (E,)      numpy float64
    elem_nu:         (E,)      numpy float64
    Returns: (T, E_elem) numpy float32 — element-level (not scattered to nodes)
    """
    dev   = _GFDM_DEVICE
    ref_t = torch.from_numpy(mesh_pos.astype(np.float32)).to(dev)
    ec_t  = torch.from_numpy(elem_conn.astype(np.int64)).to(dev)
    eE_t  = torch.from_numpy(elem_E.astype(np.float64)).to(dev)
    enu_t = torch.from_numpy(elem_nu.astype(np.float64)).to(dev)
    D     = _D_batch(eE_t, enu_t)                          # (E_elem, 6, 6) — fixed per sample
    xe    = ref_t[ec_t].to(torch.float64)                  # (E_elem, 8, 3) — fixed geometry

    T      = world_pos_batch.shape[0]
    E_elem = elem_conn.shape[0]
    result = np.empty((T, E_elem), dtype=np.float32)

    for t in range(T):
        disp_t = torch.from_numpy(
            (world_pos_batch[t] - mesh_pos).astype(np.float32)
        ).to(dev)
        ue = disp_t[ec_t].to(torch.float64)                # (E_elem, 8, 3)
        sig_sum = torch.zeros(E_elem, 6, dtype=torch.float64, device=dev)
        for dNdxi_cpu, _ in _gauss_c3d8():
            sig_sum += _gauss_stress_at(xe, ue, dNdxi_cpu, D)
        sig_avg = sig_sum / 8.0
        s  = sig_avg
        vm = torch.sqrt(0.5 * (
            (s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2 +
            6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
        ) + 1e-30).float()
        result[t] = vm.cpu().numpy()
    return result  # (T, E_elem)


def robust_stats(arr):
    mean = arr.mean(axis=0).astype(np.float32)
    std  = arr.std(axis=0).astype(np.float32)
    std  = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


# ── Pass 1: normalization stats from training set ──────────────────────────────

print(f"Pass 1: collecting stats from training set (0-{NUM_TRAIN-1}) ...")

# Load per-sample metadata (P, E2) from CSV generated by generate_bimaterial_beam.py
metadata_path = os.path.join(INPUT_DIR, "sample_metadata.csv")
sample_P = {}  # sample_id -> P in Newtons
with open(metadata_path, "r") as f:
    for row in csv.DictReader(f):
        sample_P[int(row["sample_id"])] = float(row["P_N"])

all_y      = []
all_edge   = []
all_disp   = []
all_P      = []  # collect P values from training set for normalization
all_bmat   = []  # B-matrix node-level stress for stats
all_elem   = []  # B-matrix element-level stress for stats (used by hetero training)

for sid in tqdm(range(NUM_TRAIN)):
    path = os.path.join(INPUT_DIR, f"sample_{sid:05d}.npz")
    if not os.path.exists(path):
        continue
    d = np.load(path)

    mesh_pos        = d["mesh_pos"]          # (N, 3)
    world_pos       = d["world_pos"]         # (F, N, 3)
    velocity        = d["velocity"]          # (F-1, N, 3)
    elem_conn       = d["elem_conn"]
    elem_mat        = d["elem_mat"]
    mat_E           = d["mat_E"]
    mat_nu          = d["mat_nu"]
    E_ref           = float(mat_E[0])
    mesh_edge_index = d["mesh_edge_index"].astype(np.int64)

    edge_index, edge_E = build_multigraph(elem_conn, elem_mat, mat_E)

    # Per-node E for GFDM
    mat_count = np.zeros((mesh_pos.shape[0], 2), dtype=np.int32)
    for c0, m in zip(elem_conn, elem_mat):
        for ni in c0:
            mat_count[ni, int(m)] += 1
    node_mat_p1  = np.argmax(mat_count, axis=1)
    node_E_p1    = np.where(node_mat_p1 == 0,
                            float(mat_E[0]), float(mat_E[1])).astype(np.float32)

    # GFDM stress for t=1..T (world_pos[1:])
    gfdm_stress = compute_gfdm_vm_stress_batch(
        mesh_pos, world_pos[1:], mesh_edge_index, node_E_p1
    )  # (T, N)

    # B-matrix stress for t=1..T
    elem_E  = mat_E[elem_mat].astype(np.float64)
    elem_nu = mat_nu[elem_mat].astype(np.float64)
    bmat_stress = compute_bmat_vm_stress_batch(
        mesh_pos, world_pos[1:], elem_conn, elem_E, elem_nu
    )  # (T, N) node-level

    # Element-level stress for stats (from world_pos[1:] = t+1 positions, consistent with targets)
    elem_stress_p1 = compute_elem_vm_stress_batch(
        mesh_pos, world_pos[1:], elem_conn, elem_E, elem_nu
    )  # (T, E_elem)

    P_sid = sample_P.get(sid, 0.0)
    for t in range(velocity.shape[0]):
        all_y.append(
            np.concatenate([velocity[t], gfdm_stress[t][:, None]],
                           axis=1).astype(np.float32))
        all_edge.append(
            compute_edge_feats(mesh_pos, world_pos[t], edge_index, edge_E, E_ref))
        all_disp.append((world_pos[t] - mesh_pos).astype(np.float32))
        all_bmat.append(bmat_stress[t])
        all_elem.append(elem_stress_p1[t])
    all_P.append(P_sid)

y_all      = np.concatenate(all_y,    axis=0)
edge_all   = np.concatenate(all_edge, axis=0)
disp_all   = np.concatenate(all_disp, axis=0)
bmat_all   = np.concatenate(all_bmat, axis=0)
P_arr      = np.array(all_P, dtype=np.float32)

y_mean,    y_std    = robust_stats(y_all)
edge_mean, edge_std = robust_stats(edge_all)
disp_mean, disp_std = robust_stats(disp_all)
bmat_mean  = float(bmat_all.mean())
bmat_std   = float(bmat_all.std())
if bmat_std < EPS:
    bmat_std = 1.0

# Log-normalize P: P is log-uniformly distributed, so log(P) ~ Uniform → symmetric N(0,1)
logP_arr  = np.log(P_arr.clip(min=1.0))  # clip guards against P=0 edge cases
P_log_mean = float(logP_arr.mean())
P_log_std  = float(logP_arr.std())
if P_log_std < 1e-6:
    P_log_std = 1.0

node_stats = {
    "velocity_mean":    y_mean[:3].tolist(),
    "velocity_std":     y_std[:3].tolist(),
    "stress_mean":      float(y_mean[3]),
    "stress_std":       float(y_std[3]),
    "bmat_stress_mean": bmat_mean,
    "bmat_stress_std":  bmat_std,
    "disp_mean":        disp_mean.tolist(),
    "disp_std":         disp_std.tolist(),
    "P_log_mean":       P_log_mean,
    "P_log_std":        P_log_std,
}
edge_stats_out = {
    "edge_mean": edge_mean.tolist(),
    "edge_std":  edge_std.tolist(),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for fpath in [
    os.path.join(OUTPUT_DIR, "node_stats_bimaterial.json"),
    "node_stats_bimaterial.json",
]:
    with open(fpath, "w") as f:
        json.dump(node_stats, f, indent=2)
for fpath in [
    os.path.join(OUTPUT_DIR, "edge_stats_bimaterial.json"),
    "edge_stats_bimaterial.json",
]:
    with open(fpath, "w") as f:
        json.dump(edge_stats_out, f, indent=2)

print("Stats saved.")
print(f"  edge[8] (E_norm): mean={edge_mean[8]:.4f}  std={edge_std[8]:.4f}")
print(f"  disp_std: {disp_std}")


# ── Pass 2: build .pt files ────────────────────────────────────────────────────

print(f"\nPass 2: building sample_*.pt ...")

for sid in tqdm(range(NUM_SAMPLES)):
    path    = os.path.join(INPUT_DIR,  f"sample_{sid:05d}.npz")
    pt_path = os.path.join(OUTPUT_DIR, f"sample_{sid:05d}.pt")

    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        continue
    if os.path.exists(pt_path) and not FORCE:
        continue

    d = np.load(path)
    mesh_pos   = d["mesh_pos"]
    world_pos  = d["world_pos"]
    velocity   = d["velocity"]
    stress_vm  = d["stress_vm"]
    node_type  = d["node_type"]
    fixed_mask = d["fixed_mask"]
    elem_conn  = d["elem_conn"]
    elem_mat   = d["elem_mat"]
    mat_E      = d["mat_E"]
    E_ref      = float(mat_E[0])

    edge_index, edge_E = build_multigraph(elem_conn, elem_mat, mat_E)

    # Deduplicated mesh edges + per-node E
    mesh_edge_index = d["mesh_edge_index"].astype(np.int64)  # (2, E_dedup)

    # Per-node E: majority-vote material assignment from adjacent elements
    mat_count = np.zeros((d["mesh_pos"].shape[0], 2), dtype=np.int32)
    for c0, m in zip(elem_conn, elem_mat):
        for ni in c0:
            mat_count[ni, int(m)] += 1
    node_mat = np.argmax(mat_count, axis=1)   # (N,) 0=Mat1 1=Mat2
    node_E_arr = np.where(node_mat == 0, float(mat_E[0]), float(mat_E[1])).astype(np.float32)

    # 3D one-hot: free=[1,0,0], clamped=[0,1,0], loaded=[0,0,1]
    node_type_oh = node_type_onehot(node_type)   # (N, 3)

    N = mesh_pos.shape[0]
    num_steps = velocity.shape[0]

    # Log-normalized force magnitude (scalar, broadcast to all nodes)
    P_sid   = sample_P.get(sid, 0.0)
    p_norm  = float((np.log(max(P_sid, 1.0)) - P_log_mean) / P_log_std)
    p_col   = np.full((N, 1), p_norm, dtype=np.float32)

    # GFDM stress for all timesteps at once (world_pos[1..T])
    gfdm_stress_p2 = compute_gfdm_vm_stress_batch(
        mesh_pos, world_pos[1:], mesh_edge_index, node_E_arr
    )  # (T, N)

    # B-matrix stress for all timesteps
    mat_nu      = d["mat_nu"]
    elem_E_p2   = mat_E[elem_mat].astype(np.float64)
    elem_nu_p2  = mat_nu[elem_mat].astype(np.float64)
    bmat_stress_p2 = compute_bmat_vm_stress_batch(
        mesh_pos, world_pos[1:], elem_conn, elem_E_p2, elem_nu_p2
    )  # (T, N)

    steps = []

    for t in range(num_steps):
        x_t = world_pos[t].astype(np.float32)

        disp_vec = (x_t - mesh_pos).astype(np.float32)
        prev_vel = (velocity[t - 1].astype(np.float32)
                    if t > 0 else np.zeros_like(velocity[0]))
        # Node features: [disp(3), prev_vel(3), node_type_oh(3), p_norm(1)] = 10D
        node_x = np.concatenate([disp_vec, prev_vel, node_type_oh, p_col],
                                  axis=1).astype(np.float32)

        # Target: [vel(3), gfdm_stress(1)] — normalized with global training stats
        y_raw  = np.concatenate([velocity[t], gfdm_stress_p2[t][:, None]],
                                 axis=1).astype(np.float32)
        y_norm = ((y_raw - y_mean[None, :]) / y_std[None, :]).astype(np.float32)

        # Edge features: 9D, normalized
        ef_raw  = compute_edge_feats(mesh_pos, x_t, edge_index, edge_E, E_ref)
        ef_norm = ((ef_raw - edge_mean[None, :]) / edge_std[None, :]).astype(np.float32)

        graph = Data(
            x          = torch.from_numpy(node_x),
            y          = torch.from_numpy(y_norm),
            edge_index = torch.from_numpy(edge_index),
            num_nodes  = N,
        )
        graph.mesh_pos        = torch.from_numpy(mesh_pos.astype(np.float32))
        graph.world_pos       = torch.from_numpy(x_t)
        graph.node_type       = torch.from_numpy(node_type.astype(np.int64))
        graph.fixed_mask      = torch.from_numpy(fixed_mask.astype(bool))
        graph.step_index      = torch.tensor([t], dtype=torch.long)
        graph.mat_E           = torch.from_numpy(mat_E.astype(np.float32))
        graph.mesh_edge_index = torch.from_numpy(mesh_edge_index)  # deduplicated
        graph.node_E          = torch.from_numpy(node_E_arr)       # per-node E (Pa)
        graph.bmat_stress     = torch.from_numpy(bmat_stress_p2[t])  # (N,) B-matrix GT stress

        steps.append({
            "graph":               graph,
            "mesh_edge_features":  torch.from_numpy(ef_norm),
            "world_edge_features": torch.zeros(0, EDGE_DIM, dtype=torch.float32),
        })

    torch.save(steps, pt_path)

print(f"\nDone. {NUM_SAMPLES} samples → {OUTPUT_DIR}/")
print(f"Train: sample_00000 .. sample_00999")
print(f"Test:  sample_01000 .. sample_01099")
print(f"Node features: {NODE_DIM}D   Edge features: {EDGE_DIM}D (Multigraph)")
