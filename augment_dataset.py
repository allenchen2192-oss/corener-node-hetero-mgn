"""
Reset preprocessed PT files to original design:

  graph.x shape: [N, 3]
    [:, 0:3]  one-hot node type only (original MeshGraphNet design)

Usage:
    python augment_dataset.py
"""

import os
import torch

PT_DIR = "04_preprocessed_pt"


def reset_pt_file(sample_idx):
    pt_path = os.path.join(PT_DIR, f"sample_{sample_idx:05d}.pt")
    if not os.path.exists(pt_path):
        print(f"  SKIP (no PT): {pt_path}")
        return

    sample = torch.load(pt_path, map_location="cpu", weights_only=False)

    if sample[0]["graph"].x.shape[1] == 3:
        print(f"  already 3-feature, skipping: {pt_path}")
        return

    for t, item in enumerate(sample):
        graph = item["graph"]
        # Keep only the first 3 columns (one-hot node type)
        graph.x = graph.x[:, :3].contiguous()

    torch.save(sample, pt_path)
    print(f"  reset: {pt_path}  x.shape={sample[0]['graph'].x.shape}")


def main():
    print("Resetting PT files to 3-feature graph.x (one-hot node type only) ...")
    for sample_idx in range(30):
        reset_pt_file(sample_idx)
    print("\nDone. Set num_input_features: 3 in conf/config.yaml")


if __name__ == "__main__":
    main()
