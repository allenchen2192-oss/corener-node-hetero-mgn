"""
Single-sample ROLLOUT inference script with ParaView export.
Loads a preprocessed .pt file (e.g. sample_00029.pt) and runs the model
autoregressively: the predicted world_pos at step t is fed back as input
to step t+1.  Boundary conditions (object / clamped nodes) are enforced
using the ground-truth positions from graph.y, exactly as in inference.py.

Tetrahedral cell connectivity is loaded from graph_case_XX_*.npz files in
03_graph_npz/ (merged_cells, C3D10 → linear tet using first 4 corner nodes).

Usage (run from the deforming_plate directory):
    python infer_single_sample.py \
        --sample         04_preprocessed_pt/sample_00029.pt \
        --sample_idx     29 \
        --ckpt           checkpoints_3Dbeam \
        --stats          outputs/node_stats.json \
        --edge_stats     outputs/edge_stats.json \
        --graph_npz_dir  03_graph_npz \
        --output_dir     infer_output_sample29
"""

import argparse
import os
import torch
import numpy as np

import glob

from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint
from physicsnemo.datapipes.gnn.utils import load_json
from helpers import add_world_edges


def denormalize(x, mean, std):
    return x * std + mean


def get_masks_from_node_features(graph_x):
    """
    graph.x[:,0:3] is one-hot encoded with 3 classes:
        col 0 -> node_type 0 (free / moving nodes)
        col 1 -> node_type 1 (object boundary)
        col 2 -> node_type 3 (clamped boundary)
    Only the first 3 columns are used (graph.x may have extra feature dims).
    Returns boolean masks of shape (num_nodes, 1).
    """
    node_class   = graph_x[:, :3].argmax(dim=-1)
    moving_mask  = (node_class == 0).unsqueeze(-1)
    object_mask  = (node_class == 1).unsqueeze(-1)
    clamped_mask = (node_class == 2).unsqueeze(-1)
    return moving_mask, object_mask, clamped_mask




def export_to_paraview(pred_positions, exact_positions, pred_stresses,
                       exact_stresses, cells, mesh_pos, output_dir):
    import meshio

    os.makedirs(output_dir, exist_ok=True)
    num_steps = len(pred_positions)

    # Use full tetrahedra (avoids winding-order artifacts from surface extraction)
    cell_block = [("tetra", cells)]

    print(f"Exporting {num_steps} steps to {output_dir} ...")
    for t in range(num_steps):
        pred_pts  = pred_positions[t]
        exact_pts = exact_positions[t]

        # Displacement relative to the undeformed reference mesh
        pred_disp_mag  = np.linalg.norm(pred_pts  - mesh_pos, axis=1)
        exact_disp_mag = np.linalg.norm(exact_pts - mesh_pos, axis=1)
        error          = np.linalg.norm(pred_pts  - exact_pts, axis=1)

        # predicted VTU
        meshio.Mesh(
            points=pred_pts,
            cells=cell_block,
            point_data={
                "Stress":           pred_stresses[t],
                "Displacement_Mag": pred_disp_mag,
                "Position_Error":   error,
            },
        ).write(os.path.join(output_dir, f"step_{t:03d}_pred.vtu"))

        # ground-truth VTU
        meshio.Mesh(
            points=exact_pts,
            cells=cell_block,
            point_data={
                "Stress":           exact_stresses[t],
                "Displacement_Mag": exact_disp_mag,
            },
        ).write(os.path.join(output_dir, f"step_{t:03d}_exact.vtu"))

    print("ParaView export complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",      default="04_preprocessed_pt/sample_00029.pt")
    parser.add_argument("--sample_idx",  type=int, default=29,
                        help="Index of the sample (filename number, 0-based)")
    parser.add_argument("--ckpt",        default="checkpoints_3Dbeam")
    parser.add_argument("--ckpt_epoch",  type=int, default=None,
                        help="Load specific epoch checkpoint, e.g. 400 loads HybridMeshGraphNet.0.400.mdlus")
    parser.add_argument("--stats",       default="04_preprocessed_pt/node_stats.json")
    parser.add_argument("--edge_stats",  default="04_preprocessed_pt/edge_stats.json")
    parser.add_argument("--graph_npz_dir", default="03_graph_npz",
                        help="Directory containing graph_case_XX_*.npz files")
    parser.add_argument("--num_input_features",  type=int, default=3)
    parser.add_argument("--num_edge_features",   type=int, default=8)
    parser.add_argument("--num_output_features", type=int, default=4)
    parser.add_argument("--output_dir",  default="infer_output_sample29")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    model = HybridMeshGraphNet(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)
    if args.ckpt_epoch is not None:
        load_checkpoint(args.ckpt, models=model, epoch=args.ckpt_epoch, device=device)
        print(f"Model loaded from epoch {args.ckpt_epoch}")
    else:
        load_checkpoint(args.ckpt, models=model, device=device)
        print(f"Model loaded from: {args.ckpt} (latest checkpoint)")
    model.eval()

    # ── load stats ────────────────────────────────────────────────────────────
    node_stats  = load_json(args.stats)
    vel_mean    = node_stats["velocity_mean"].to(device)
    vel_std     = node_stats["velocity_std"].to(device)
    stress_mean = node_stats["stress_mean"].to(device)
    stress_std  = node_stats["stress_std"].to(device)

    # ── load cell connectivity from graph npz ────────────────────────────────
    # sample_idx is 0-based; graph npz files are named case_01, case_02, ...
    case_num = args.sample_idx + 1
    npz_pattern = os.path.join(args.graph_npz_dir, f"graph_case_{case_num:02d}_*.npz")
    npz_matches = glob.glob(npz_pattern)
    if not npz_matches:
        raise RuntimeError(f"No graph npz found for pattern: {npz_pattern}")
    graph_npz_path = npz_matches[0]
    print(f"Loading cell connectivity from {graph_npz_path} ...")
    graph_data = np.load(graph_npz_path, allow_pickle=True)
    # merged_cells is C3D10 (10-node quadratic tet); take first 4 corner nodes
    # for standard linear tetrahedra used by meshio/ParaView
    cells = graph_data["merged_cells"][:, :4]
    print(f"Cells loaded: {cells.shape}, max_idx={cells.max()}")

    # ── load preprocessed sample ──────────────────────────────────────────────
    sample_data = torch.load(args.sample, map_location=device)
    num_steps   = len(sample_data)
    print(f"Sample: {args.sample}  |  time steps: {num_steps}")

    pred_positions  = []
    exact_positions = []
    pred_stresses   = []
    exact_stresses  = []
    rollout_world_pos = None

    # Frame 0: undeformed reference configuration, stress = 0 for both
    init_pos = sample_data[0]["graph"].mesh_pos.cpu().numpy()[:, :3]
    N_nodes  = init_pos.shape[0]
    pred_positions.append(init_pos.copy())
    exact_positions.append(init_pos.copy())
    pred_stresses.append(np.zeros((N_nodes, 1), dtype=np.float32))
    exact_stresses.append(np.zeros((N_nodes, 1), dtype=np.float32))

    # The graphs stored in .pt already had add_world_edges applied (edge_attr = 8 dims).
    # We need to restore the mesh-only state (4-dim edge_attr) before calling
    # add_world_edges each step.  The first 4 columns of the stored mesh_edge_features
    # are the original normalized reference-config edge attributes (constant for a
    # non-adaptive mesh).
    first_step    = sample_data[0]
    N_mesh        = first_step["mesh_edge_features"].shape[0]
    # mesh-only edge_index (world edges were appended after the first N_mesh edges)
    mesh_edge_index   = first_step["graph"].edge_index[:, :N_mesh].to(device)
    # original 4-dim edge_attr from reference mesh positions
    mesh_edge_attr_4d = first_step["mesh_edge_features"][:, :4].to(device)

    with torch.inference_mode():
        for t, step in enumerate(sample_data):
            graph = step["graph"].to(device)

            moving_mask, object_mask, clamped_mask = get_masks_from_node_features(graph.x)

            # ground-truth next position (used for boundary enforcement + exact output)
            exact_vel          = denormalize(graph.y[:, 0:3], vel_mean, vel_std)
            exact_next_world_pos = graph.world_pos[:, 0:3] + exact_vel

            # rollout: feed previous prediction as new world_pos
            if rollout_world_pos is not None:
                graph.world_pos = rollout_world_pos

            # restore mesh-only graph state before calling add_world_edges,
            # otherwise edge_attr dims keep growing each call
            graph.edge_index = mesh_edge_index.clone()
            graph.edge_attr   = mesh_edge_attr_4d.clone()

            # recompute world edges based on (possibly updated) world_pos
            graph, mesh_edge_features, world_edge_features = add_world_edges(
                graph, edge_stats_path=args.edge_stats
            )

            # graph.x: one-hot node type only (3 features) — original MeshGraphNet design
            graph.x = graph.x[:, :3]

            # forward pass
            pred = model(graph.x, mesh_edge_features, world_edge_features, graph)

            # denormalize predicted velocity
            pred_vel = denormalize(pred[:, 0:3], vel_mean, vel_std)

            # zero velocity on boundary nodes
            moving_mask_3d = moving_mask.expand_as(pred_vel)
            pred_vel = torch.where(moving_mask_3d, pred_vel, torch.zeros_like(pred_vel))

            # integrate
            pred_world_pos = graph.world_pos[:, 0:3] + pred_vel

            # enforce boundary conditions with ground-truth positions
            pred_world_pos = torch.where(object_mask.expand_as(pred_world_pos),
                                         exact_next_world_pos, pred_world_pos)
            pred_world_pos = torch.where(clamped_mask.expand_as(pred_world_pos),
                                         exact_next_world_pos, pred_world_pos)

            # stress (von Mises is physically non-negative, clamp to 0)
            # boundary nodes (rod/clamped) are rigid → zero stress
            if pred.shape[1] > 3:
                pred_stress  = denormalize(pred[:, 3:],      stress_mean, stress_std).clamp(min=0)
                exact_stress = denormalize(graph.y[:, 3:],   stress_mean, stress_std).clamp(min=0)
                # Only zero out stress on rod (rigid body actuator) nodes.
                # Clamped (fixed-end) nodes carry the maximum bending stress — do NOT zero them.
                pred_stress  = torch.where(object_mask, torch.zeros_like(pred_stress),  pred_stress)
                exact_stress = torch.where(object_mask, torch.zeros_like(exact_stress), exact_stress)
            else:
                pred_stress  = torch.zeros((pred.shape[0], 1), device=device)
                exact_stress = torch.zeros((pred.shape[0], 1), device=device)

            rollout_world_pos = pred_world_pos.clone()

            pred_positions.append(pred_world_pos.cpu().numpy())
            exact_positions.append(exact_next_world_pos.cpu().numpy())
            pred_stresses.append(pred_stress.cpu().numpy())
            exact_stresses.append(exact_stress.cpu().numpy())

            if t % 10 == 0 or t == num_steps - 1:
                disp_mag = torch.linalg.norm(pred_vel, dim=-1)
                print(f"  step {t:3d} | disp_mag  mean={disp_mag.mean().item():.4e}  "
                      f"max={disp_mag.max().item():.4e}")

    # ── save npy ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "pred_positions.npy"),  np.stack(pred_positions))
    np.save(os.path.join(args.output_dir, "pred_stresses.npy"),   np.stack(pred_stresses))
    np.save(os.path.join(args.output_dir, "exact_positions.npy"), np.stack(exact_positions))
    np.save(os.path.join(args.output_dir, "exact_stresses.npy"),  np.stack(exact_stresses))
    print(f"Saved .npy files to {args.output_dir}/")

    # ── ParaView VTU export ───────────────────────────────────────────────────
    mesh_pos = graph_data["mesh_pos"].astype(np.float32)
    export_to_paraview(
        pred_positions, exact_positions,
        pred_stresses, exact_stresses,
        cells, mesh_pos,
        output_dir=os.path.join(args.output_dir, "paraview"),
    )


if __name__ == "__main__":
    main()
