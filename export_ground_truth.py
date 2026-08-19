"""
Export ground truth VTU files directly from 03_graph_npz without running the model.

Usage (run from the deforming_plate directory):
    python export_ground_truth.py --sample_idx 0
    python export_ground_truth.py --sample_idx 29 --output_dir gt_output_sample29
"""

import argparse
import os
import glob
import numpy as np


def export_ground_truth(graph_npz_path: str, output_dir: str):
    import meshio

    print(f"Loading {graph_npz_path} ...")
    data = np.load(graph_npz_path, allow_pickle=True)

    # ── mesh topology ─────────────────────────────────────────────────────────
    # merged_cells is C3D10 (10-node quadratic tet); take first 4 corner nodes
    # to get standard linear tetrahedra recognised by meshio/ParaView
    cells_linear = data["merged_cells"][:, :4]
    cell_block = [("tetra", cells_linear)]

    # ── field data ────────────────────────────────────────────────────────────
    world_pos  = data["world_pos"]   # (T, N, 3)  deformed positions
    mesh_pos   = data["mesh_pos"]    # (N, 3)     reference (undeformed) positions
    mises      = data["mises"]       # (T, N)     von Mises stress
    node_type  = data["node_type"]   # (N,)       0=beam, 1=beam_fixed, 2=rod
    T = world_pos.shape[0]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting {T} frames to {output_dir} ...")

    for t in range(T):
        pts = world_pos[t].astype(np.float32)
        disp_mag = np.linalg.norm(pts - mesh_pos.astype(np.float32), axis=1)

        meshio.Mesh(
            points=pts,
            cells=cell_block,
            point_data={
                "Mises_Stress":      mises[t].astype(np.float32),
                "Displacement_Mag":  disp_mag,
                "Node_Type":         node_type.astype(np.float32),
            },
        ).write(os.path.join(output_dir, f"step_{t:03d}.vtu"))

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx",    type=int, default=0,
                        help="Sample index (0-based); maps to graph_case_{idx+1:02d}_*.npz")
    parser.add_argument("--graph_npz_dir", default="03_graph_npz",
                        help="Directory containing graph_case_XX_*.npz files")
    parser.add_argument("--output_dir",    default=None,
                        help="Output directory (default: gt_output_sample{idx})")
    args = parser.parse_args()

    case_num = args.sample_idx + 1
    pattern  = os.path.join(args.graph_npz_dir, f"graph_case_{case_num:02d}_*.npz")
    matches  = glob.glob(pattern)
    if not matches:
        raise RuntimeError(f"No npz found matching: {pattern}")
    graph_npz_path = matches[0]

    output_dir = args.output_dir or f"gt_output_sample{args.sample_idx:02d}"

    export_ground_truth(graph_npz_path, output_dir)


if __name__ == "__main__":
    main()
