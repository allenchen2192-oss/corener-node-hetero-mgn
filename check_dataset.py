"""
check_dataset.py
================
Diagnose training data: check shapes, value ranges, normalization, and stats.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

PT_DIR   = Path("./04_preprocessed_pt")
STATS_DIR = PT_DIR

def check_stats():
    print("=" * 60)
    print("node_stats.json / edge_stats.json")
    print("=" * 60)
    for fname in ["node_stats.json", "edge_stats.json"]:
        for d in [PT_DIR, Path("./outputs"), Path(".")]:
            p = d / fname
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                print(f"\n[{p}]")
                for k, v in data.items():
                    print(f"  {k}: {v}")
            else:
                print(f"\n[{p}]  ← NOT FOUND")

def check_samples(n=3):
    print("\n" + "=" * 60)
    print(f"Checking first {n} sample .pt files")
    print("=" * 60)

    files = sorted(PT_DIR.glob("sample_*.pt"))
    print(f"Total .pt files: {len(files)}")

    for fpath in files[:n]:
        print(f"\n--- {fpath.name} ---")
        sample = torch.load(fpath, map_location="cpu", weights_only=False)
        print(f"  num steps (after skip t=0): {len(sample)}")

        item  = sample[0]
        g     = item["graph"]
        mef   = item["mesh_edge_features"]
        wef   = item["world_edge_features"]

        print(f"  graph.x shape     : {g.x.shape}   (should be [64, 3])")
        print(f"  graph.y shape     : {g.y.shape}   (should be [64, 4])")
        print(f"  edge_index shape  : {g.edge_index.shape}")
        print(f"  mesh_ef shape     : {mef.shape}   (should be [E_mesh, 8])")
        print(f"  world_ef shape    : {wef.shape}   (should be [0, 8])")
        print(f"  graph.x  min/max  : {g.x.min():.4f} / {g.x.max():.4f}")
        print(f"  graph.y  min/max  : {g.y.min():.4f} / {g.y.max():.4f}")
        print(f"  mesh_ef  min/max  : {mef.min():.4f} / {mef.max():.4f}")

        # Check y is normalized (should be ~N(0,1) range)
        y_mean = g.y.mean(0)
        y_std  = g.y.std(0)
        print(f"  graph.y  mean     : {y_mean.tolist()}")
        print(f"  graph.y  std      : {y_std.tolist()}")

        # Check world_pos and mesh_pos
        if hasattr(g, 'world_pos'):
            print(f"  world_pos shape   : {g.world_pos.shape}")
            print(f"  mesh_pos shape    : {g.mesh_pos.shape}")
            disp = g.world_pos - g.mesh_pos
            print(f"  disp_vec(=x) diff : {(disp - g.x).abs().max():.2e}  (should be ~0)")

def check_normalization():
    print("\n" + "=" * 60)
    print("Checking normalization across all training samples")
    print("=" * 60)

    with open(PT_DIR / "node_stats.json") as f:
        ns = json.load(f)
    v_mean = torch.tensor(ns["velocity_mean"] + [ns["stress_mean"]])
    v_std  = torch.tensor(ns["velocity_std"]  + [ns["stress_std"]])

    files = sorted(PT_DIR.glob("sample_*.pt"))[:500]  # training only

    all_y = []
    for fpath in files:
        sample = torch.load(fpath, map_location="cpu", weights_only=False)
        for item in sample:
            all_y.append(item["graph"].y)

    all_y = torch.cat(all_y, dim=0)   # (N_total, 4)
    print(f"  Total y vectors: {all_y.shape}")
    print(f"  y mean (should be ~0): {all_y.mean(0).tolist()}")
    print(f"  y std  (should be ~1): {all_y.std(0).tolist()}")

    # Check if first sample graph.x matches disp_vec
    files = sorted(PT_DIR.glob("sample_*.pt"))
    print(f"\n  Spot-check: graph.x == world_pos - mesh_pos for sample_00000:")
    s = torch.load(files[0], map_location="cpu", weights_only=False)
    g = s[0]["graph"]
    diff = (g.x - (g.world_pos - g.mesh_pos)).abs().max()
    print(f"  max diff: {diff:.2e}  (should be ~0)")

def check_edge_consistency():
    print("\n" + "=" * 60)
    print("Checking edge_index vs mesh_edge_features consistency")
    print("=" * 60)
    files = sorted(PT_DIR.glob("sample_*.pt"))
    s = torch.load(files[0], map_location="cpu", weights_only=False)
    for t, item in enumerate(s[:3]):
        g   = item["graph"]
        mef = item["mesh_edge_features"]
        wef = item["world_edge_features"]
        E_total = g.edge_index.shape[1]
        E_mesh  = mef.shape[0]
        E_world = wef.shape[0]
        ok = (E_total == E_mesh + E_world)
        print(f"  step t={t+1}: edge_index cols={E_total}, mesh_ef={E_mesh}, world_ef={E_world}, "
              f"consistent={'OK' if ok else 'MISMATCH!'}")

if __name__ == "__main__":
    check_stats()
    check_samples(n=3)
    check_normalization()
    check_edge_consistency()
    print("\nDiagnosis complete.")
