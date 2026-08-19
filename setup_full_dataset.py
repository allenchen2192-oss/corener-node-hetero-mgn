"""
setup_full_dataset.py
=====================
Generate 500 training + 10 test samples and save to 04_preprocessed_pt/.

  Training : seed=42,  sample_00000 ~ sample_00499  (first 100 match old data)
  Test     : seed=123, sample_00500 ~ sample_00509

Two-pass approach:
  Pass 1 – compute raw velocities / stresses for all 500 training samples
            → derive normalization stats (node_stats.json, edge_stats.json)
  Pass 2 – normalize and save .pt for all 510 samples

Also copies stats to outputs/ so infer_synthetic.py finds them after hydra chdir.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_SEED       = 42
TEST_SEED        = 123
NUM_TRAIN        = 500
NUM_TEST         = 10
TEST_START_IDX   = NUM_TRAIN          # sample_00500 ~ sample_00509

H              = 1.0
NUM_FRAMES     = 21
N_NODES_AXIS   = 4
E              = 70e9
NU             = 0.33
COEFF_MIN      = -0.02
COEFF_MAX      =  0.02

SCRIPT_DIR = Path(__file__).resolve().parent
DST_DIR    = SCRIPT_DIR / "04_preprocessed_pt"
OUT_DIR    = SCRIPT_DIR / "outputs"   # hydra chdir target; inference reads stats here


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
                    ni2, nj2, nk2 = i+di, j+dj, k+dk
                    if ni2 < n and nj2 < n and nk2 < n:
                        t = nid(ni2, nj2, nk2)
                        edges.extend([[s, t], [t, s]])
    return np.asarray(edges, dtype=np.int64).T


def displacement_field(mesh_pos: np.ndarray, lam: float, coeff: np.ndarray, H: float) -> np.ndarray:
    basis = np.concatenate([np.ones((len(mesh_pos), 1), dtype=np.float32), mesh_pos], axis=1)
    return (lam * (basis @ coeff.T) / H).astype(np.float32)


def compute_stress_vm(mesh_pos: np.ndarray, lam: float, coeff: np.ndarray,
                      H: float, E: float, nu: float) -> np.ndarray:
    mu       = E / (2.0 * (1.0 + nu))
    lam_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    grad_const = (lam / H) * coeff[:, 1:4]
    grad = np.broadcast_to(grad_const[None], (len(mesh_pos), 3, 3)).copy()
    strain = 0.5 * (grad + grad.transpose(0, 2, 1))
    tr = np.trace(strain, axis1=1, axis2=2)[:, None, None]
    sigma = lam_lame * tr * np.eye(3)[None] + 2.0 * mu * strain
    sxx, syy, szz = sigma[:,0,0], sigma[:,1,1], sigma[:,2,2]
    sxy, syz, szx = sigma[:,0,1], sigma[:,1,2], sigma[:,2,0]
    return np.sqrt(
        0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2) + 3*(sxy**2+syz**2+szx**2)
    ).astype(np.float32)


def edge_features(edge_index: np.ndarray, mesh_pos: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]  - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos[dst] - world_pos[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


def simulate_one(mesh_pos: np.ndarray, coeff: np.ndarray, lambdas: np.ndarray):
    """Return (world_pos_seq, velocity, stress_vm_seq) for one case."""
    world_pos_seq = np.stack(
        [mesh_pos + displacement_field(mesh_pos, lam, coeff, H) for lam in lambdas], axis=0
    )   # (T, N, 3)
    velocity = world_pos_seq[1:] - world_pos_seq[:-1]   # (T-1, N, 3)
    stress_vm_seq = np.stack(
        [compute_stress_vm(mesh_pos, lam, coeff, H, E, NU) for lam in lambdas], axis=0
    )   # (T, N)
    return world_pos_seq, velocity, stress_vm_seq


def build_steps(
    world_pos_seq: np.ndarray,
    velocity: np.ndarray,
    stress_vm_seq: np.ndarray,
    mesh_pos: np.ndarray,
    mesh_ei_np: np.ndarray,
    mesh_ei: torch.Tensor,
    v_mean: np.ndarray,
    v_std: np.ndarray,
    e_mean: np.ndarray,
    e_std: np.ndarray,
) -> list:
    """Build normalized timestep list, skipping t=0."""
    steps = []
    for t in range(1, NUM_FRAMES - 1):   # t = 1..19  (skip t=0)
        x_t  = world_pos_seq[t]
        y_raw = np.concatenate(
            [velocity[t], stress_vm_seq[t+1][:, None]], axis=1
        ).astype(np.float32)
        y_norm   = (y_raw - v_mean) / v_std

        ef_raw  = edge_features(mesh_ei_np, mesh_pos, x_t)
        ef_norm = (ef_raw - e_mean) / e_std

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

        steps.append({
            "graph"              : graph,
            "mesh_edge_features" : torch.from_numpy(ef_norm.astype(np.float32)),
            "world_edge_features": torch.zeros(0, 8),
        })
    return steps


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_pos   = build_structured_grid(N_NODES_AXIS, H)
    mesh_ei_np = build_mesh_edge_index(N_NODES_AXIS)
    mesh_ei    = torch.from_numpy(mesh_ei_np)
    lambdas    = quarter_sine(NUM_FRAMES)

    # ── Generate coefficients ─────────────────────────────────────────────────
    train_coeffs = np.random.default_rng(TRAIN_SEED).uniform(
        COEFF_MIN, COEFF_MAX, size=(NUM_TRAIN, 12)
    ).reshape(NUM_TRAIN, 3, 4).astype(np.float32)

    test_coeffs = np.random.default_rng(TEST_SEED).uniform(
        COEFF_MIN, COEFF_MAX, size=(NUM_TEST, 12)
    ).reshape(NUM_TEST, 3, 4).astype(np.float32)

    # ── Pass 1: simulate training cases, collect raw stats ───────────────────
    print(f"Pass 1 / 2 — simulating {NUM_TRAIN} training cases to compute stats ...")
    train_cache   = []
    all_y_raw     = []
    all_edge_raw  = []

    for i, coeff in enumerate(train_coeffs):
        wp_seq, vel, stress_seq = simulate_one(mesh_pos, coeff, lambdas)
        train_cache.append((wp_seq, vel, stress_seq))

        for t in range(1, NUM_FRAMES - 1):
            x_t   = wp_seq[t]
            y_raw = np.concatenate([vel[t], stress_seq[t+1][:, None]], axis=1)
            all_y_raw.append(y_raw)
            ef    = edge_features(mesh_ei_np, mesh_pos, x_t)
            all_edge_raw.append(ef)

        if (i + 1) % 50 == 0:
            print(f"  simulated {i+1}/{NUM_TRAIN}")

    y_all    = np.concatenate(all_y_raw,    axis=0)   # (N_total, 4)
    edge_all = np.concatenate(all_edge_raw, axis=0)   # (E_total, 8)

    v_mean = y_all.mean(axis=0).astype(np.float32)
    v_std  = y_all.std(axis=0).astype(np.float32)
    v_std  = np.where(v_std < 1e-8, 1.0, v_std).astype(np.float32)

    e_mean = edge_all.mean(axis=0).astype(np.float32)
    e_std  = edge_all.std(axis=0).astype(np.float32)
    e_std  = np.where(e_std < 1e-8, 1.0, e_std).astype(np.float32)

    node_stats = {
        "velocity_mean": v_mean[:3].tolist(),
        "velocity_std" : v_std[:3].tolist(),
        "stress_mean"  : float(v_mean[3]),
        "stress_std"   : float(v_std[3]),
    }
    edge_stats = {
        "edge_mean": e_mean.tolist(),
        "edge_std" : e_std.tolist(),
    }

    for target_dir in [DST_DIR, OUT_DIR]:
        (target_dir / "node_stats.json").write_text(
            json.dumps(node_stats, indent=2), encoding="utf-8"
        )
        (target_dir / "edge_stats.json").write_text(
            json.dumps(edge_stats, indent=2), encoding="utf-8"
        )

    # Also copy to script dir (fallback)
    for fname in ["node_stats.json", "edge_stats.json"]:
        shutil.copy2(DST_DIR / fname, SCRIPT_DIR / fname)

    print(f"  velocity_mean={v_mean[:3]}, velocity_std={v_std[:3]}")
    print(f"  stress_mean={v_mean[3]:.3e}, stress_std={v_std[3]:.3e}")
    print(f"Stats saved to {DST_DIR} and {OUT_DIR}")

    # ── Pass 2: normalize and save training .pt ───────────────────────────────
    print(f"\nPass 2 / 2 — saving {NUM_TRAIN} training samples ...")
    for i, (wp_seq, vel, stress_seq) in enumerate(train_cache):
        steps = build_steps(
            wp_seq, vel, stress_seq, mesh_pos, mesh_ei_np, mesh_ei,
            v_mean, v_std, e_mean, e_std,
        )
        torch.save(steps, DST_DIR / f"sample_{i:05d}.pt")
        if (i + 1) % 100 == 0:
            print(f"  saved {i+1}/{NUM_TRAIN}  x.shape={steps[0]['graph'].x.shape}")

    # ── Save test .pt ─────────────────────────────────────────────────────────
    print(f"\nGenerating {NUM_TEST} test cases (seed={TEST_SEED}) ...")
    for j, coeff in enumerate(test_coeffs):
        sample_idx = TEST_START_IDX + j
        wp_seq, vel, stress_seq = simulate_one(mesh_pos, coeff, lambdas)
        steps = build_steps(
            wp_seq, vel, stress_seq, mesh_pos, mesh_ei_np, mesh_ei,
            v_mean, v_std, e_mean, e_std,
        )
        torch.save(steps, DST_DIR / f"sample_{sample_idx:05d}.pt")
        print(f"  sample_{sample_idx:05d}.pt  steps={len(steps)}")

    total = NUM_TRAIN + NUM_TEST
    print(f"\nDone. {total} files in {DST_DIR}")
    print(f"  Training : sample_00000 ~ sample_{NUM_TRAIN-1:05d}")
    print(f"  Test     : sample_{TEST_START_IDX:05d} ~ sample_{TEST_START_IDX+NUM_TEST-1:05d}")
    print(f"\nNext steps:")
    print(f"  1. Set in conf/config.yaml: num_training_samples: {NUM_TRAIN}")
    print(f"  2. rm -rf checkpoints_synthetic/")
    print(f"  3. python train.py")
    print(f"  4. Set infer_sample_indices: {list(range(TEST_START_IDX, TEST_START_IDX+NUM_TEST))}")
    print(f"  5. python infer_synthetic.py")


if __name__ == "__main__":
    main()
