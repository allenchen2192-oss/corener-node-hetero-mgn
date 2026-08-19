"""
Convert Abaqus NPZ files to sample_*.pt format for train.py.

Usage:
    python preprocess_abaqus.py           # skip existing .pt
    python preprocess_abaqus.py --force   # overwrite existing .pt

Pipeline:
    02_abaqus_npz/sample_XXXXX.npz
        -> 04_preprocessed_pt_abaqus/sample_XXXXX.pt
        -> 04_preprocessed_pt_abaqus/node_stats.json
        -> 04_preprocessed_pt_abaqus/edge_stats.json

Split: samples 0-1999 = train, samples 2000-2099 = test
Node features  : disp_vec (N, 3)  = world_pos - mesh_pos
Target y       : [velocity(3), stress_vm(1)] normalized
Edge features  : [d_mesh(3), n_mesh(1), d_world(3), n_world(1)] = 8D, normalized
World edges    : empty (0, 8) — mesh_fallback disabled for cleaner separation
"""

import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

INPUT_DIR  = "./02_abaqus_npz"
OUTPUT_DIR = "./04_preprocessed_pt_abaqus_prevvel"
NUM_SAMPLES    = 2100
NUM_TRAIN      = 2000   # 0 .. 1999
EPS            = 1e-8
EDGE_DIM       = 8
FORCE          = "--force" in sys.argv


# ─── helpers ──────────────────────────────────────────────────────────────────

def edge_features(mesh_pos, world_pos, edge_index):
    """8D edge features: [d_mesh(3), n_mesh(1), d_world(3), n_world(1)]."""
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]  - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos[dst] - world_pos[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


def robust_stats(arr):
    mean = arr.mean(axis=0).astype(np.float32)
    std  = arr.std(axis=0).astype(np.float32)
    std  = np.where(std < EPS, 1.0, std).astype(np.float32)
    return mean, std


# ─── pass 1: collect raw stats from training set ──────────────────────────────

print("Pass 1: collecting normalization stats from training set (0-{}) ...".format(NUM_TRAIN - 1))

all_y    = []
all_edge = []
all_disp = []

for sid in tqdm(range(NUM_TRAIN)):
    npz_path = os.path.join(INPUT_DIR, "sample_{:05d}.npz".format(sid))
    if not os.path.exists(npz_path):
        continue
    d = np.load(npz_path)
    mesh_pos        = d["mesh_pos"]       # (N, 3)
    world_pos       = d["world_pos"]      # (F, N, 3)
    velocity        = d["velocity"]       # (F-1, N, 3)
    stress_vm       = d["stress_vm"]      # (F, N)
    mesh_edge_index = d["mesh_edge_index"]  # (2, E)

    num_steps = velocity.shape[0]
    for t in range(num_steps):
        y_raw = np.concatenate([
            velocity[t],                       # (N, 3)
            stress_vm[t + 1][:, None],         # (N, 1)
        ], axis=1).astype(np.float32)          # (N, 4)
        all_y.append(y_raw)

        ef = edge_features(mesh_pos, world_pos[t], mesh_edge_index)
        all_edge.append(ef)

        disp_vec = (world_pos[t] - mesh_pos).astype(np.float32)  # (N, 3)
        all_disp.append(disp_vec)

y_all    = np.concatenate(all_y,    axis=0)
edge_all = np.concatenate(all_edge, axis=0)
disp_all = np.concatenate(all_disp, axis=0)
y_mean, y_std       = robust_stats(y_all)
edge_mean, edge_std = robust_stats(edge_all)
disp_mean, disp_std = robust_stats(disp_all)

node_stats = {
    "velocity_mean": y_mean[:3].tolist(),
    "velocity_std":  y_std[:3].tolist(),
    "stress_mean":   float(y_mean[3]),
    "stress_std":    float(y_std[3]),
    "disp_mean":     disp_mean.tolist(),
    "disp_std":      disp_std.tolist(),
}
edge_stats_dict = {
    "edge_mean": edge_mean.tolist(),
    "edge_std":  edge_std.tolist(),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "node_stats.json"), "w") as f:
    json.dump(node_stats, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "edge_stats.json"), "w") as f:
    json.dump(edge_stats_dict, f, indent=2)

# Also copy to working dir so train.py can find them via load_json("edge_stats.json")
with open("node_stats.json", "w") as f:
    json.dump(node_stats, f, indent=2)
with open("edge_stats.json", "w") as f:
    json.dump(edge_stats_dict, f, indent=2)

print("node_stats.json and edge_stats.json written.")


# ─── pass 2: build .pt sample files ───────────────────────────────────────────

print("\nPass 2: building sample_*.pt ...")

for sid in tqdm(range(NUM_SAMPLES)):
    npz_path = os.path.join(INPUT_DIR,  "sample_{:05d}.npz".format(sid))
    pt_path  = os.path.join(OUTPUT_DIR, "sample_{:05d}.pt".format(sid))

    if not os.path.exists(npz_path):
        print("  MISSING: {}".format(npz_path))
        continue
    if os.path.exists(pt_path) and not FORCE:
        continue  # resume-safe

    d = np.load(npz_path)
    mesh_pos        = d["mesh_pos"]
    world_pos       = d["world_pos"]
    velocity        = d["velocity"]
    stress_vm       = d["stress_vm"]
    mesh_edge_index = d["mesh_edge_index"]
    node_type       = d["node_type"]
    fixed_mask      = d["fixed_mask"]
    elem_conn        = d["elem_conn"] if "elem_conn" in d else None
    n_nodes_per_elem = int(d["n_nodes_per_elem"]) if "n_nodes_per_elem" in d else 0

    # Derive corner(0) vs mid-side(1) node type from element connectivity.
    # C3D8 (npe=8): all corner; C3D10 (npe=10): first 4 per elem; C3D20 (npe=20): first 8 per elem.
    N = mesh_pos.shape[0]
    node_type_derived = np.zeros(N, dtype=np.float32)   # default: all corner
    if elem_conn is not None and n_nodes_per_elem in (10, 20):
        n_corner = 4 if n_nodes_per_elem == 10 else 8
        node_type_derived[:] = 1.0                                      # all mid-side initially
        corner_ids = np.unique(elem_conn[:, :n_corner])
        node_type_derived[corner_ids] = 0.0                             # mark corners
    # one-hot: corner→[1,0], mid-side→[0,1]
    node_type_oh = np.stack([1.0 - node_type_derived,
                              node_type_derived], axis=1).astype(np.float32)  # (N, 2)

    num_steps = velocity.shape[0]
    steps = []

    for t in range(num_steps):
        x_t = world_pos[t].astype(np.float32)   # (N, 3)

        # Node features: [disp_vec(3), prev_vel(3), node_type_oh(2)]  → (N, 8)
        disp_vec = (x_t - mesh_pos).astype(np.float32)          # (N, 3)
        prev_vel = velocity[t - 1] if t > 0 else np.zeros_like(velocity[0])  # (N, 3)
        node_x   = np.concatenate([disp_vec, prev_vel, node_type_oh],
                                   axis=1).astype(np.float32)   # (N, 8)

        # Target: normalized velocity + normalized stress_vm
        y_raw = np.concatenate([
            velocity[t],
            stress_vm[t + 1][:, None],
        ], axis=1).astype(np.float32)
        y_norm = ((y_raw - y_mean[None, :]) / y_std[None, :]).astype(np.float32)

        # Edge features (8D, normalized)
        ef_raw  = edge_features(mesh_pos, x_t, mesh_edge_index)
        ef_norm = ((ef_raw - edge_mean[None, :]) / edge_std[None, :]).astype(np.float32)

        # Mesh edges only; world edge features = empty
        mesh_ef_t = torch.from_numpy(ef_norm)                 # (E, 8)
        world_ef_t = torch.zeros(0, EDGE_DIM, dtype=torch.float32)  # (0, 8)

        graph = Data(
            x          = torch.from_numpy(node_x),
            y          = torch.from_numpy(y_norm),
            edge_index = torch.from_numpy(mesh_edge_index.astype(np.int64)),
            num_nodes  = int(mesh_pos.shape[0]),
        )
        graph.mesh_pos   = torch.from_numpy(mesh_pos.astype(np.float32))
        graph.world_pos  = torch.from_numpy(x_t)
        graph.node_type  = torch.from_numpy(node_type.astype(np.int64))
        graph.fixed_mask = torch.from_numpy(fixed_mask.astype(bool))
        graph.step_index = torch.tensor([t], dtype=torch.int64)

        step = {
            "graph":               graph,
            "mesh_edge_features":  mesh_ef_t,
            "world_edge_features": world_ef_t,
            "n_nodes_per_elem":    n_nodes_per_elem,
        }
        if elem_conn is not None:
            step["elem_conn"] = torch.from_numpy(elem_conn.astype(np.int64))
        steps.append(step)

    torch.save(steps, pt_path)

print("\nDone. {} samples written to {}".format(NUM_SAMPLES, OUTPUT_DIR))
print("Train: sample_00000 .. sample_{:05d}".format(NUM_TRAIN - 1))
print("Test:  sample_{:05d} .. sample_{:05d}".format(NUM_TRAIN, NUM_SAMPLES - 1))
