"""
convert_abaqus_dataset.py
=========================
Convert Abaqus_version3 preprocessed .pt files to the format expected by train.py.

Changes made:
  1. graph.x: world_pos (N,3) → [one_hot(node_type, 3) | disp_vec (N,3)]  shape (N,6)
  2. Skip t=0 step (λ(0)=0 → disp_mag=0 for all samples → identical inputs, wasted capacity)
  3. world_edge_features → empty (0, 8): mesh_fallback duplicates mesh edges, giving the
     model redundant info through a separate MLP head. Empty is cleaner.
  4. Copy edge_stats.json and node_stats.json as-is (already computed)

Usage:
    python convert_abaqus_dataset.py
"""

import os
import shutil

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR = "./Abaqus_version3/train_dataset/preprocess/cases_v3/04_preprocessed_pt"
DST_DIR = "./04_preprocessed_pt"

os.makedirs(DST_DIR, exist_ok=True)

# ── Copy normalization stats ───────────────────────────────────────────────────
for fname in ["node_stats.json", "edge_stats.json"]:
    src = os.path.join(SRC_DIR, fname)
    dst = os.path.join(DST_DIR, fname)
    shutil.copy2(src, dst)
    print(f"Copied {fname}")

# Also copy to current dir (train.py reads from cwd via load_json)
for fname in ["node_stats.json", "edge_stats.json"]:
    shutil.copy2(os.path.join(SRC_DIR, fname), fname)
    print(f"Copied {fname} → ./{fname}")

# ── Convert each sample ────────────────────────────────────────────────────────
sample_files = sorted(
    f for f in os.listdir(SRC_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"\nFound {len(sample_files)} sample files in {SRC_DIR}")

for fname in sample_files:
    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, fname)

    sample = torch.load(src_path, map_location="cpu", weights_only=False)

    # sample[0] is t=0 (λ=0, all displacements=0, identical across samples → skip)
    converted = []
    for item in sample[1:]:   # skip t=0
        graph = item["graph"]

        # disp_vec = full displacement vector (world_pos - mesh_pos), preserves direction
        # one-hot node type omitted: all nodes are type 0 → constant, no information
        disp_vec = graph.world_pos - graph.mesh_pos   # (N, 3)

        graph.x = disp_vec   # (N, 3)

        # Clear world edges: mesh_fallback = same edges as mesh → redundant, replace with empty.
        # Also trim graph.edge_index to mesh-only so edge count matches:
        #   graph.edge_index cols must equal mesh_ef rows + world_ef rows
        E_mesh = item["mesh_edge_features"].shape[0]
        graph.edge_index = graph.edge_index[:, :E_mesh]
        item["world_edge_features"] = torch.zeros(0, 8)

        converted.append(item)

    torch.save(converted, dst_path)
    print(f"  {fname}  steps: {len(sample)} → {len(converted)}  "
          f"graph.x: {converted[0]['graph'].x.shape}")   # expect (N, 3)

print(f"\nConverted {len(sample_files)} samples → {DST_DIR}")
print("graph.x shape:", converted[0]["graph"].x.shape)   # expect (N, 3)
print("\nNext steps:")
print("  1. rm -rf checkpoints_synthetic/   (or set ckpt_path to a new dir)")
print("  2. python train.py")
print("  3. python infer_synthetic.py")
