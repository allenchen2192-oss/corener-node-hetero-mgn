"""
preprocess_abaqus_rect.py
==========================
Preprocess rectangular prism (長方體) NPZ files into sample_*.pt format.

Differences from preprocess_abaqus.py:
  - INPUT_DIR  : 02_abaqus_npz_rect  (rect OOD samples)
  - OUTPUT_DIR : 04_preprocessed_pt_abaqus_rect
  - NUM_SAMPLES: 10
  - Normalization stats: loaded from existing cube training data
    (04_preprocessed_pt_abaqus_prevvel/) so that the model sees
    the same normalization it was trained with.
  - No pass 1 (stats computation skipped).

Usage:
    python preprocess_abaqus_rect.py
    python preprocess_abaqus_rect.py --force   # overwrite existing .pt
"""

import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

INPUT_DIR   = "./02_abaqus_npz_rect"
OUTPUT_DIR  = "./04_preprocessed_pt_abaqus_rect"
STATS_DIR   = "./04_preprocessed_pt_abaqus_prevvel"   # reuse cube training stats
NUM_SAMPLES = 10
EPS         = 1e-8
EDGE_DIM    = 8
FORCE       = "--force" in sys.argv


# ─── helpers ──────────────────────────────────────────────────────────────────

def edge_features(mesh_pos, world_pos, edge_index):
    """8D edge features: [d_mesh(3), n_mesh(1), d_world(3), n_world(1)]."""
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]  - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos[dst] - world_pos[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


# ─── load normalization stats from cube training data ─────────────────────────

print(f"Loading normalization stats from: {STATS_DIR}")

with open(os.path.join(STATS_DIR, "node_stats.json")) as f:
    node_stats = json.load(f)
with open(os.path.join(STATS_DIR, "edge_stats.json")) as f:
    edge_stats_dict = json.load(f)

y_mean = np.array(
    node_stats["velocity_mean"] + [node_stats["stress_mean"][0]
                                    if isinstance(node_stats["stress_mean"], list)
                                    else node_stats["stress_mean"]],
    dtype=np.float32,
)
y_std = np.array(
    node_stats["velocity_std"] + [node_stats["stress_std"][0]
                                   if isinstance(node_stats["stress_std"], list)
                                   else node_stats["stress_std"]],
    dtype=np.float32,
)
edge_mean = np.array(edge_stats_dict["edge_mean"], dtype=np.float32)
edge_std  = np.array(edge_stats_dict["edge_std"],  dtype=np.float32)

print(f"  velocity_mean : {y_mean[:3]}")
print(f"  velocity_std  : {y_std[:3]}")
print(f"  stress_mean   : {y_mean[3]:.3e}")
print(f"  stress_std    : {y_std[3]:.3e}")


# ─── build .pt sample files ───────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy stats to OUTPUT_DIR so inference scripts can find them
import shutil
for fname in ["node_stats.json", "edge_stats.json"]:
    shutil.copy2(os.path.join(STATS_DIR, fname), os.path.join(OUTPUT_DIR, fname))
print(f"Copied stats → {OUTPUT_DIR}")

print(f"\nBuilding sample_*.pt for {NUM_SAMPLES} rect samples ...")

for sid in tqdm(range(NUM_SAMPLES)):
    npz_path = os.path.join(INPUT_DIR,  f"sample_{sid:05d}.npz")
    pt_path  = os.path.join(OUTPUT_DIR, f"sample_{sid:05d}.pt")

    if not os.path.exists(npz_path):
        print(f"  MISSING: {npz_path}")
        continue
    if os.path.exists(pt_path) and not FORCE:
        continue

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

    num_steps = velocity.shape[0]
    steps = []

    for t in range(num_steps):
        x_t = world_pos[t].astype(np.float32)

        disp_vec = (x_t - mesh_pos).astype(np.float32)
        prev_vel = velocity[t - 1] if t > 0 else np.zeros_like(velocity[0])
        node_x   = np.concatenate([disp_vec, prev_vel], axis=1).astype(np.float32)

        y_raw = np.concatenate([
            velocity[t],
            stress_vm[t + 1][:, None],
        ], axis=1).astype(np.float32)
        y_norm = ((y_raw - y_mean[None, :]) / y_std[None, :]).astype(np.float32)

        ef_raw  = edge_features(mesh_pos, x_t, mesh_edge_index)
        ef_norm = ((ef_raw - edge_mean[None, :]) / edge_std[None, :]).astype(np.float32)

        mesh_ef_t  = torch.from_numpy(ef_norm)
        world_ef_t = torch.zeros(0, EDGE_DIM, dtype=torch.float32)

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

print(f"\nDone. {NUM_SAMPLES} samples → {OUTPUT_DIR}")
