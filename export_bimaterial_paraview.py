"""
export_bimaterial_paraview.py
==============================
Rollout bi-material Multigraph model and export GT + Pred to VTU/PVD for ParaView.

Output:
  export_bimaterial_paraview/
    sample_XXXXX/
      gt/   t_000.vtu ... sample_XXXXX_gt.pvd
      pred/ t_000.vtu ... sample_XXXXX_pred.pvd

PointData fields:
  displacement   (vector, 3D, m)
  disp_mag       (scalar, m)
  von_mises      (scalar, Pa)  — GT: from Abaqus; Pred: GFDM with per-node E
  disp_error     (scalar, m)   — pred only
  material_id    (scalar)      — 0=Mat1, 1=Mat2 (per node, for interface viz)

Usage:
  python export_bimaterial_paraview.py
"""

import csv
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from export_abaqus_paraview import (
    build_surface_cells_from_elems,
    write_vtu,
    write_pvd,
)
from preprocess_bimaterial import build_multigraph


# ── Config ─────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 10
NUM_EDGE_FEATURES   = 9
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128

DATA_DIR  = "./04_preprocessed_pt_bimaterial"
NPZ_DIR   = "./02_abaqus_npz_bimaterial"
CKPT_PATH = "./checkpoints_bimaterial_v3"
OUT_DIR   = "./export_bimaterial_paraview_v3"

TRAIN_INDICES  = list(range(10))           # sample_00000 .. sample_00009
TEST_INDICES   = list(range(1000, 1010))   # sample_01000 .. sample_01009
SAMPLE_INDICES = TRAIN_INDICES + TEST_INDICES


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_rollout_ef(mesh_pos, world_pos_t, edge_index, edge_E, E_ref,
                     e_mean, e_std):
    """Recompute normalized 9D edge features from current world_pos."""
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]       - mesh_pos[src]
    n_mesh  = torch.norm(d_mesh,  dim=1, keepdim=True)
    d_world = world_pos_t[dst]    - world_pos_t[src]
    n_world = torch.norm(d_world, dim=1, keepdim=True)
    E_col   = (edge_E / E_ref).unsqueeze(1)
    ef_raw  = torch.cat([d_mesh, n_mesh, d_world, n_world, E_col], dim=1)
    return (ef_raw - e_mean) / e_std


def node_material_id(elem_conn, elem_mat, N):
    """Per-node material: majority vote from adjacent elements."""
    mat_count = np.zeros((N, 2), dtype=np.int32)
    for c0, m in zip(elem_conn, elem_mat):
        for ni in c0:
            mat_count[ni, int(m)] += 1
    return np.argmax(mat_count, axis=1).astype(np.float32)


def compute_vm_stress_bimaterial(wp, ref, mesh_ei, node_E, nu=0.33):
    """GFDM von Mises stress with per-node Young's modulus.

    Uses deduplicated mesh_ei (NOT Multigraph) to avoid double-counting
    at interface nodes.

    Args:
        wp      : (N, 3) current world positions
        ref     : (N, 3) reference positions
        mesh_ei : (2, E) deduplicated directed edge index
        node_E  : (N,)  per-node Young's modulus in Pa
        nu      : Poisson's ratio (same for both materials)
    Returns:
        (N,) von Mises stress in Pa
    """
    N = wp.shape[0]
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    u     = wp.float() - ref.float()
    du_ij = u[col] - u[row]

    w_ij  = 1.0 / (r_ij.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12))
    r_r   = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)

    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    XtX = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtY = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtX.scatter_add_(0, row_exp, r_r)
    XtY.scatter_add_(0, row_exp, r_du)
    XtX = XtX + 1e-6 * torch.eye(3, device=wp.device).unsqueeze(0)
    dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)  # (N, 3, 3)

    E   = node_E.float()   # (N,)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))

    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = lam * tr + 2 * mu * dU[:, 0, 0]
    sy  = lam * tr + 2 * mu * dU[:, 1, 1]
    sz  = lam * tr + 2 * mu * dU[:, 2, 2]
    txy = mu * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = mu * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = mu * (dU[:, 1, 2] + dU[:, 2, 1])

    return torch.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                              + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)


# ── Load stats ─────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats_bimaterial.json"))
v_mean = node_stats["velocity_mean"].to(device)
v_std  = node_stats["velocity_std"].to(device)
d_mean = node_stats["disp_mean"].to(device)
d_std  = node_stats["disp_std"].to(device)
P_log_mean = float(node_stats["P_log_mean"])
P_log_std  = float(node_stats["P_log_std"])
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats_bimaterial.json"))
e_mean = edge_stats["edge_mean"].to(device)
e_std  = edge_stats["edge_std"].to(device)

empty_wef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)


# ── Load model ─────────────────────────────────────────────────────────────────

model = HybridMeshGraphNet(
    NUM_INPUT_FEATURES, NUM_EDGE_FEATURES, NUM_OUTPUT_FEATURES,
    processor_size=PROCESSOR_SIZE,
    hidden_dim_processor=HIDDEN_DIM,
    hidden_dim_node_encoder=HIDDEN_DIM,
    hidden_dim_edge_encoder=HIDDEN_DIM,
    hidden_dim_node_decoder=HIDDEN_DIM,
    mlp_activation_fn="relu",
    do_concat_trick=False,
    num_processor_checkpoint_segments=0,
    recompute_activation=False,
).to(device)
model.eval()

print(f"Loading checkpoint: {CKPT_PATH}")
load_checkpoint(CKPT_PATH, models=model, device=device)


# ── File list ──────────────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"Found {len(all_files)} .pt files in {DATA_DIR}")

# Load per-sample P values from metadata CSV
sample_P = {}
metadata_path = os.path.join(NPZ_DIR, "sample_metadata.csv")
with open(metadata_path, "r") as f:
    for row in csv.DictReader(f):
        sample_P[int(row["sample_id"])] = float(row["P_N"])


# ── Export loop ────────────────────────────────────────────────────────────────

for idx in SAMPLE_INDICES:
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range")
        continue

    fname     = all_files[idx]
    sample_id = fname.replace(".pt", "")
    split     = "train" if idx < 1000 else "test"
    fpath     = os.path.join(DATA_DIR, fname)

    gt_dir   = os.path.join(OUT_DIR, sample_id, "gt")
    pred_dir = os.path.join(OUT_DIR, sample_id, "pred")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    data = torch.load(fpath, map_location="cpu", weights_only=False)
    T    = len(data)
    N    = data[0]["graph"].num_nodes
    print(f"\n[{split}] {sample_id}  N={N}  T={T}")

    # ── NPZ: topology + material + GT stress ───────────────────────────────────
    npz          = np.load(os.path.join(NPZ_DIR, sample_id + ".npz"))
    elem_conn    = npz["elem_conn"]
    elem_mat     = npz["elem_mat"]
    mat_E        = npz["mat_E"]
    E1           = float(mat_E[0])
    E2           = float(mat_E[1])
    stress_vm_np = npz["stress_vm"]   # (F, N) Pa
    print(f"  E1={E1/1e9:.0f} GPa  E2={E2/1e9:.1f} GPa")

    # ── Multigraph edges (for model inference) ────────────────────────────────
    edge_index_np, edge_E_np = build_multigraph(elem_conn, elem_mat, mat_E)
    edge_index = torch.from_numpy(edge_index_np).to(device)
    edge_E     = torch.from_numpy(edge_E_np).to(device)

    # ── Reference mesh & surface cells ────────────────────────────────────────
    ref_pos = data[0]["graph"].mesh_pos.to(device)
    ref_np  = ref_pos.cpu().numpy()

    connectivity, offsets, types, _ = build_surface_cells_from_elems(
        ref_np, elem_conn, 8, debug_label=sample_id
    )
    mat_id_np = node_material_id(elem_conn, elem_mat, N)

    # ── Deduplicated mesh edges + per-node E (for GFDM stress) ───────────────
    mesh_ei = torch.from_numpy(npz["mesh_edge_index"].astype(np.int64)).to(device)
    mat_id_torch = torch.from_numpy(mat_id_np).to(device)
    node_E = torch.where(mat_id_torch == 0,
                         torch.full_like(mat_id_torch, E1),
                         torch.full_like(mat_id_torch, E2))

    # Static node type one-hot (3D, from first timestep x[:,6:9])
    node_type_oh = data[0]["graph"].x[:, 6:9].to(device)

    # Clamped mask: node_type_oh[:, 1] == 1  →  enforce vel=0 in rollout
    clamped_mask = (node_type_oh[:, 1] == 1)

    # Normalized force (broadcast to all nodes)
    p_norm_val = (np.log(max(sample_P.get(idx, 1.0), 1.0)) - P_log_mean) / P_log_std
    p_col = torch.full((N, 1), p_norm_val, dtype=torch.float32, device=device)

    # ── GT world-pos reconstruction ────────────────────────────────────────────
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    # ── Autoregressive rollout (seed from GT at t=1) ───────────────────────────
    gt_vel_0 = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur      = gt_wp_list[1].to(device)
    prev_vel = gt_vel_0
    pred_wp_list = [gt_wp_list[0].numpy(), gt_wp_list[1].numpy()]

    with torch.inference_mode():
        for _ in range(T - 1):
            ef = build_rollout_ef(ref_pos, cur, edge_index, edge_E, E1,
                                  e_mean, e_std)
            disp_norm     = (cur - ref_pos - d_mean) / d_std
            prev_vel_norm = (prev_vel - v_mean) / v_std
            node_x        = torch.cat([disp_norm, prev_vel_norm, node_type_oh, p_col],
                                       dim=1)
            g   = Data(x=node_x, edge_index=edge_index, num_nodes=N)
            out = model(node_x, ef, empty_wef, g)

            vel = out[:, :3] * v_std + v_mean
            vel[clamped_mask] = 0.0  # enforce clamped BC: fixed nodes never move
            prev_vel = vel
            cur      = cur + vel
            pred_wp_list.append(cur.cpu().numpy())

    # ── Write VTU + PVD ───────────────────────────────────────────────────────
    gt_pvd, pred_pvd = [], []

    for t_idx in range(T):
        vtu_name = f"t_{t_idx:03d}.vtu"

        gt_pos  = gt_wp_list[t_idx + 1].numpy().astype(np.float32)
        gt_disp = gt_pos - ref_np
        gt_vm   = stress_vm_np[t_idx + 1].astype(np.float32)
        with torch.inference_mode():
            gt_vm_gfdm = compute_vm_stress_bimaterial(
                torch.from_numpy(gt_pos).to(device),
                ref_pos, mesh_ei, node_E,
            ).cpu().numpy().astype(np.float32)

        write_vtu(
            os.path.join(gt_dir, vtu_name),
            gt_pos, connectivity, offsets, types,
            {
                "displacement":    gt_disp,
                "disp_mag":        np.linalg.norm(gt_disp, axis=1),
                "von_mises":       gt_vm,
                "von_mises_gfdm":  gt_vm_gfdm,
                "material_id":     mat_id_np,
            },
        )
        gt_pvd.append((t_idx, vtu_name))

        pred_pos  = pred_wp_list[t_idx + 1].astype(np.float32)
        pred_disp = pred_pos - ref_np
        disp_err  = np.linalg.norm(pred_pos - gt_pos, axis=1).astype(np.float32)
        with torch.inference_mode():
            pred_vm = compute_vm_stress_bimaterial(
                torch.from_numpy(pred_pos).to(device),
                ref_pos, mesh_ei, node_E,
            ).cpu().numpy().astype(np.float32)

        write_vtu(
            os.path.join(pred_dir, vtu_name),
            pred_pos, connectivity, offsets, types,
            {
                "displacement": pred_disp,
                "disp_mag":     np.linalg.norm(pred_disp, axis=1),
                "von_mises":    pred_vm,
                "disp_error":   disp_err,
                "material_id":  mat_id_np,
            },
        )
        pred_pvd.append((t_idx, vtu_name))

    write_pvd(os.path.join(gt_dir,   f"{sample_id}_gt.pvd"),   gt_pvd)
    write_pvd(os.path.join(pred_dir, f"{sample_id}_pred.pvd"), pred_pvd)

    print(f"  rollout disp_error (mm):")
    for t_idx in range(0, T, max(1, T // 10)):
        err = np.linalg.norm(
            pred_wp_list[t_idx + 1] - gt_wp_list[t_idx + 1].numpy(), axis=1
        ).mean() * 1000
        print(f"    t={t_idx:3d}: {err:.4f} mm")
    final_err = np.linalg.norm(
        pred_wp_list[-1] - gt_wp_list[-1].numpy(), axis=1
    ).mean()
    print(f"  final mean node error: {final_err*1000:.4f} mm")

print(f"\n完成！結果在 {OUT_DIR}/<sample_id>/gt/*.pvd  +  pred/*.pvd")
