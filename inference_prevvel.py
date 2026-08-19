"""
inference_prevvel.py
====================
Rollout inference for the prevvel model (HybridMeshGraphNet, 6-D input).

Rollout seed: step t=1 from each test sample
  - disp_t    = world_pos[1] - mesh_pos   (first actual displaced state)
  - prev_vel  = velocity[0]               (ground truth first velocity, stored in graph.x[:,3:])

Then predicts steps 2 → T (up to num_test_time_steps steps).

Usage:
    python inference_prevvel.py
    python inference_prevvel.py ckpt_path=./checkpoints_abaqus_prevvel_1000data_1000nodes
"""

import os

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch_geometric.data import Data
from tqdm import tqdm

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger


# ─── edge feature convention: dst - src (matches preprocess_abaqus.py) ────────

def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    disp = pos[col] - pos[row]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


# ─── VTU export (optional, requires meshio) ───────────────────────────────────

def _export_vtu(output_dir, sample_id, pred_list, gt_list, mesh_pos, elem_conn, npe):
    try:
        import meshio
    except ImportError:
        return

    cell_type = {8: "hexahedron", 20: "hexahedron20", 10: "tetra10"}.get(npe)
    if cell_type is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    for k, (pred_wp, gt_wp) in enumerate(zip(pred_list, gt_list)):
        pred_disp = np.linalg.norm(pred_wp - mesh_pos, axis=1)
        gt_disp   = np.linalg.norm(gt_wp   - mesh_pos, axis=1)
        err       = np.linalg.norm(pred_wp - gt_wp, axis=1)

        cells = [(cell_type, elem_conn)]
        meshio.Mesh(
            points=pred_wp, cells=cells,
            point_data={"disp_mag": pred_disp, "pos_error": err},
        ).write(os.path.join(output_dir, f"sample_{sample_id:05d}_step_{k:03d}_pred.vtu"))
        meshio.Mesh(
            points=gt_wp, cells=cells,
            point_data={"disp_mag": gt_disp},
        ).write(os.path.join(output_dir, f"sample_{sample_id:05d}_step_{k:03d}_gt.vtu"))


# ─── rollout helper ───────────────────────────────────────────────────────────

def run_rollout(sample_indices, data_dir, model, device,
                e_mean_ref, e_std_ref, e_mean_wld, e_std_wld,
                v_mean, v_std, num_rollout_steps, num_edge_features,
                output_dir, logger):
    """Run rollout for a list of sample indices. Returns list of error arrays."""
    vtu_dir   = os.path.join(output_dir, "vtu")
    all_errors = []

    for sid in tqdm(sample_indices, desc=f"rollout ({output_dir.split('/')[-1]})"):
        pt_path = os.path.join(data_dir, f"sample_{sid:05d}.pt")
        if not os.path.exists(pt_path):
            logger.warning(f"missing: {pt_path}")
            continue

        steps = torch.load(pt_path, map_location="cpu", weights_only=False)
        if len(steps) < 2:
            logger.warning(f"too few steps in sample_{sid:05d}.pt")
            continue

        # ── seed: step t=1 (first actual displaced state) ──────────────────
        seed_graph = steps[1]["graph"]
        ref_pos    = seed_graph.mesh_pos.to(device)
        mesh_ei    = seed_graph.edge_index.to(device)

        ref_feat_norm = (
            compute_edge_features(ref_pos, mesh_ei) - e_mean_ref
        ) / e_std_ref

        wp_current       = seed_graph.world_pos.to(device)
        prev_vel_current = seed_graph.x[:, 3:].to(device)    # prev_vel = vel[0]

        gt_world_pos = [s["graph"].world_pos for s in steps]
        elem_conn    = steps[0].get("elem_conn", None)
        npe          = steps[0].get("n_nodes_per_elem", 0)
        num_rollout  = min(num_rollout_steps, len(steps) - 2)

        pred_list, gt_list, errors = [], [], []

        with torch.no_grad():
            for k in range(num_rollout):
                x_t    = wp_current - ref_pos
                node_x = torch.cat([x_t, prev_vel_current], dim=1)

                wf      = (compute_edge_features(wp_current, mesh_ei) - e_mean_wld) / e_std_wld
                mesh_ef = torch.cat([ref_feat_norm, wf], dim=1)
                world_ef = torch.zeros(0, num_edge_features, device=device)

                graph_in = Data(x=node_x, edge_index=mesh_ei,
                                num_nodes=ref_pos.shape[0])
                graph_in.mesh_pos  = ref_pos
                graph_in.world_pos = wp_current

                pred     = model(node_x, mesh_ef, world_ef, graph_in)
                vel_phys = pred.float() * v_std + v_mean
                wp_next  = wp_current + vel_phys

                gt_idx = k + 2
                if gt_idx < len(gt_world_pos):
                    gt_wp = gt_world_pos[gt_idx].to(device)
                    errors.append((wp_next - gt_wp).norm(dim=1).mean().item())
                    gt_list.append(gt_wp.cpu().numpy())
                else:
                    gt_list.append(np.zeros_like(wp_next.cpu().numpy()))

                pred_list.append(wp_next.cpu().numpy())
                wp_current       = wp_next
                prev_vel_current = vel_phys

        all_errors.append(np.array(errors))

        os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(output_dir, f"rollout_{sid:05d}.npz"),
            pred_positions = np.stack(pred_list, axis=0),
            gt_positions   = np.stack(gt_list,   axis=0),
            mesh_pos       = ref_pos.cpu().numpy(),
            errors         = np.array(errors),
        )

        if elem_conn is not None:
            _export_vtu(vtu_dir, sid, pred_list, gt_list,
                        ref_pos.cpu().numpy(), elem_conn.numpy(), npe)

    return all_errors


def _print_summary(label, all_errors, logger):
    if not all_errors:
        return
    max_len = max(len(e) for e in all_errors)
    padded  = [np.concatenate([e, np.full(max_len - len(e), np.nan)])
               for e in all_errors]
    err_mat  = np.stack(padded, axis=0)
    mean_err = np.nanmean(err_mat, axis=0)
    logger.info(f"── {label} ──")
    for k, e in enumerate(mean_err):
        logger.info(f"  step {k+1:3d}: {e:.4e}")
    logger.info(f"  overall mean: {np.nanmean(err_mat):.4e}")


# ─── main ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config_prevvel")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("inference_prevvel")
    logger.file_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = HybridMeshGraphNet(
        cfg.num_input_features,
        cfg.num_edge_features,
        cfg.num_output_features,
        processor_size=cfg.processor_size,
        hidden_dim_processor=cfg.hidden_dim,
        hidden_dim_node_encoder=cfg.hidden_dim,
        hidden_dim_edge_encoder=cfg.hidden_dim,
        hidden_dim_node_decoder=cfg.hidden_dim,
        mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
        do_concat_trick=cfg.do_concat_trick,
        num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
    ).to(device)
    model.eval()

    load_checkpoint(to_absolute_path(cfg.ckpt_path), models=model, device=device)
    logger.info("checkpoint loaded")

    # ── normalization stats ────────────────────────────────────────────────────
    data_dir   = to_absolute_path(cfg.preprocess_output_dir)
    edge_stats = load_json(os.path.join(data_dir, "edge_stats.json"))
    e_mean_ref = edge_stats["edge_mean"].to(device)[:4]
    e_std_ref  = edge_stats["edge_std"].to(device)[:4]
    e_mean_wld = edge_stats["edge_mean"].to(device)[4:]
    e_std_wld  = edge_stats["edge_std"].to(device)[4:]

    node_stats = load_json(os.path.join(data_dir, "node_stats.json"))
    v_mean = node_stats["velocity_mean"].to(device)
    v_std  = node_stats["velocity_std"].to(device)

    rollout_kwargs = dict(
        data_dir        = data_dir,
        model           = model,
        device          = device,
        e_mean_ref      = e_mean_ref,
        e_std_ref       = e_std_ref,
        e_mean_wld      = e_mean_wld,
        e_std_wld       = e_std_wld,
        v_mean          = v_mean,
        v_std           = v_std,
        num_rollout_steps = cfg.num_test_time_steps,
        num_edge_features = cfg.num_edge_features,
        logger          = logger,
    )

    base_dir = to_absolute_path(getattr(cfg, "infer_output_dir", "./infer_output_prevvel"))

    # ── train samples ─────────────────────────────────────────────────────────
    train_indices = list(getattr(cfg, "infer_train_sample_indices", [0, 1, 2]))
    train_errors  = run_rollout(
        train_indices,
        output_dir=os.path.join(base_dir, "train"),
        **rollout_kwargs,
    )

    # ── test samples ──────────────────────────────────────────────────────────
    test_indices = list(getattr(cfg, "infer_sample_indices", [2000, 2001, 2002]))
    test_errors  = run_rollout(
        test_indices,
        output_dir=os.path.join(base_dir, "test"),
        **rollout_kwargs,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("===== rollout summary (mean node position error) =====")
    _print_summary(f"train  {train_indices}", train_errors, logger)
    _print_summary(f"test   {test_indices}",  test_errors,  logger)
    logger.info(f"results saved to {base_dir}/train  and  {base_dir}/test")


if __name__ == "__main__":
    main()
