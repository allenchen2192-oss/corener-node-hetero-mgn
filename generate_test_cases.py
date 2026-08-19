"""
generate_test_cases.py
======================
Generate 10 unseen test cases and save to 04_preprocessed_pt/ as
sample_00100.pt ~ sample_00109.pt.

Key differences from training data:
  - seed=123  (training used seed=42 → no overlap)
  - Normalization stats loaded from existing node_stats.json / edge_stats.json
    (NOT recomputed) so the scale matches the trained model exactly.
  - Output format identical to convert_abaqus_dataset.py:
      graph.x = disp_vec  (N, 3)
      world_edge_features = empty (0, 8)
      edge_index = mesh edges only
      t=0 step skipped
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

# ── Config ────────────────────────────────────────────────────────────────────
TEST_SEED        = 123
NUM_TEST_CASES   = 10
START_IDX        = 100        # first output: sample_00100.pt

H              = 1.0
NUM_FRAMES     = 21
N_NODES_AXIS   = 4
E              = 70e9
NU             = 0.33
COEFF_MIN      = -0.02
COEFF_MAX      =  0.02

SCRIPT_DIR = Path(__file__).resolve().parent
DST_DIR    = SCRIPT_DIR / "04_preprocessed_pt"


# ── Physics (from preprocess_cases_v3.py) ─────────────────────────────────────

def quarter_sine(num_frames: int) -> np.ndarray:
    n = np.arange(num_frames, dtype=np.float64)
    return np.sin(0.5 * math.pi * n / (num_frames - 1))


def build_structured_grid(n: int, H: float) -> np.ndarray:
    coords = []
    for k, z in enumerate(np.linspace(0.0, H, n)):
        for j, y in enumerate(np.linspace(0.0, H, n)):
            for i, x in enumerate(np.linspace(0.0, H, n)):
                coords.append([x, y, z])
    return np.asarray(coords, dtype=np.float32)


def build_mesh_edge_index(n: int) -> np.ndarray:
    def nid(i, j, k):
        return k * n * n + j * n + i
    edges = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                s = nid(i, j, k)
                for di, dj, dk in [(1,0,0),(0,1,0),(0,0,1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if ni < n and nj < n and nk < n:
                        t = nid(ni, nj, nk)
                        edges.extend([[s, t], [t, s]])
    return np.asarray(edges, dtype=np.int64).T


def displacement_field(mesh_pos: np.ndarray, lam: float, coeff: np.ndarray, H: float) -> np.ndarray:
    basis = np.concatenate([np.ones((len(mesh_pos), 1), dtype=np.float32), mesh_pos], axis=1)  # (N,4)
    return (lam * (basis @ coeff.T) / H).astype(np.float32)


def edge_features(edge_index: np.ndarray, mesh_pos: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]  - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos[dst] - world_pos[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


def compute_stress_vm(mesh_pos: np.ndarray, lam: float, coeff: np.ndarray, H: float,
                      E: float, nu: float) -> np.ndarray:
    mu       = E / (2.0 * (1.0 + nu))
    lam_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    grad_const = (lam / H) * coeff[:, 1:4]          # (3,3)
    grad = np.broadcast_to(grad_const[None], (len(mesh_pos), 3, 3)).copy()
    strain = 0.5 * (grad + grad.transpose(0, 2, 1))
    tr = np.trace(strain, axis1=1, axis2=2)[:, None, None]
    eye = np.eye(3, dtype=np.float64)[None]
    sigma = lam_lame * tr * eye + 2.0 * mu * strain
    sxx, syy, szz = sigma[:,0,0], sigma[:,1,1], sigma[:,2,2]
    sxy, syz, szx = sigma[:,0,1], sigma[:,1,2], sigma[:,2,0]
    vm = np.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2) + 3*(sxy**2+syz**2+szx**2))
    return vm.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Load training normalization stats
    with open(DST_DIR / "node_stats.json") as f:
        ns = json.load(f)
    with open(DST_DIR / "edge_stats.json") as f:
        es = json.load(f)

    v_mean = np.array(ns["velocity_mean"] + [ns["stress_mean"]], dtype=np.float32)  # (4,)
    v_std  = np.array(ns["velocity_std"]  + [ns["stress_std"]],  dtype=np.float32)
    e_mean = np.array(es["edge_mean"], dtype=np.float32)   # (8,)
    e_std  = np.array(es["edge_std"],  dtype=np.float32)

    # Generate test parameters
    rng    = np.random.default_rng(TEST_SEED)
    coeffs = rng.uniform(COEFF_MIN, COEFF_MAX, size=(NUM_TEST_CASES, 12))

    # Static mesh geometry
    mesh_pos    = build_structured_grid(N_NODES_AXIS, H)
    mesh_ei_np  = build_mesh_edge_index(N_NODES_AXIS)
    mesh_ei     = torch.from_numpy(mesh_ei_np)
    lambdas     = quarter_sine(NUM_FRAMES)

    for idx in range(NUM_TEST_CASES):
        sample_idx = START_IDX + idx
        coeff = coeffs[idx].reshape(3, 4).astype(np.float32)

        # Compute world_pos sequence  (T, N, 3)
        world_pos_seq = np.stack(
            [mesh_pos + displacement_field(mesh_pos, lam, coeff, H) for lam in lambdas],
            axis=0,
        )
        velocity = world_pos_seq[1:] - world_pos_seq[:-1]   # (T-1, N, 3)

        # Compute stress VM sequence  (T, N)
        stress_vm_seq = np.stack(
            [compute_stress_vm(mesh_pos, lam, coeff, H, E, NU) for lam in lambdas],
            axis=0,
        )

        # Build steps, skip t=0 (λ=0 → all zeros, no sample-specific information)
        converted = []
        for t in range(1, NUM_FRAMES - 1):   # t = 1 .. 19  → 19 steps
            x_t = world_pos_seq[t]            # world_pos at time t

            # Target: velocity to next step + stress at next frame
            y_raw = np.concatenate(
                [velocity[t], stress_vm_seq[t + 1][:, None]], axis=1
            ).astype(np.float32)              # (N, 4)
            y_norm = (y_raw - v_mean) / v_std

            # Mesh edge features (normalized, 8-dim)
            ef_raw  = edge_features(mesh_ei_np, mesh_pos, x_t)
            ef_norm = (ef_raw - e_mean) / e_std

            # graph.x = disp_vec only (no one-hot)
            disp_vec = (x_t - mesh_pos).astype(np.float32)

            graph = Data(
                x          = torch.from_numpy(disp_vec),
                y          = torch.from_numpy(y_norm.astype(np.float32)),
                edge_index = mesh_ei,
                num_nodes  = mesh_pos.shape[0],
            )
            graph.mesh_pos  = torch.from_numpy(mesh_pos)
            graph.world_pos = torch.from_numpy(x_t)
            graph.node_type = torch.zeros(mesh_pos.shape[0], dtype=torch.long)

            converted.append({
                "graph"               : graph,
                "mesh_edge_features"  : torch.from_numpy(ef_norm.astype(np.float32)),
                "world_edge_features" : torch.zeros(0, 8),
            })

        out_path = DST_DIR / f"sample_{sample_idx:05d}.pt"
        torch.save(converted, out_path)
        print(f"  sample_{sample_idx:05d}.pt  steps={len(converted)}  x.shape={converted[0]['graph'].x.shape}")

    print(f"\nGenerated {NUM_TEST_CASES} test cases  (seed={TEST_SEED})")
    print(f"Saved to: {DST_DIR}")
    print(f"Sample indices: {START_IDX} ~ {START_IDX + NUM_TEST_CASES - 1}")
    print(f"\nTo run inference on test cases, set in conf/config.yaml:")
    indices = list(range(START_IDX, START_IDX + NUM_TEST_CASES))
    print(f"  infer_sample_indices: {indices}")


if __name__ == "__main__":
    main()
