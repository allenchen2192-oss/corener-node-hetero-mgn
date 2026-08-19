"""
export_bimaterial_paraview_v4.py
=================================
Rollout v4 bi-material model (9D, no P feature) and export GT + Pred to VTU/PVD.

v4 ablation: same as v3 except use_p_feature=False (num_input=9).

Output:
  export_bimaterial_paraview_v4/
    sample_XXXXX/
      gt/   t_000.vtu ... sample_XXXXX_gt.pvd
      pred/ t_000.vtu ... sample_XXXXX_pred.pvd
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

NUM_INPUT_FEATURES  = 9    # v4: no P feature
NUM_EDGE_FEATURES   = 9
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128

DATA_DIR  = "./04_preprocessed_pt_bimaterial"
NPZ_DIR   = "./02_abaqus_npz_bimaterial"
CKPT_PATH = "./checkpoints_bimaterial_v4"
OUT_DIR   = "./export_bimaterial_paraview_v4"

TRAIN_INDICES  = list(range(10))
TEST_INDICES   = list(range(1000, 1010))
SAMPLE_INDICES = TRAIN_INDICES + TEST_INDICES


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_rollout_ef(mesh_pos, world_pos_t, edge_index, edge_E, E_ref,
                     e_mean, e_std):
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = torch.norm(d_mesh,  dim=1, keepdim=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = torch.norm(d_world, dim=1, keepdim=True)
    E_col   = (edge_E / E_ref).unsqueeze(1)
    ef_raw  = torch.cat([d_mesh, n_mesh, d_world, n_world, E_col], dim=1)
    return (ef_raw - e_mean) / e_std


def node_material_id(elem_conn, elem_mat, N):
    mat_count = np.zeros((N, 2), dtype=np.int32)
    for c0, m in zip(elem_conn, elem_mat):
        for ni in c0:
            mat_count[ni, int(m)] += 1
    return np.argmax(mat_count, axis=1).astype(np.float32)


def compute_vm_stress_bimaterial(wp, ref, mesh_ei, node_E, nu=0.33):
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
    dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    E   = node_E.float()
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
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats_bimaterial.json"))
e_mean = edge_stats["edge_mean"].to(device)
e_std  = edge_stats["edge_std"].to(device)

empty_wef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)

# Load per-sample P values (for display only, not used in model input)
sample_P = {}
metadata_path = os.path.join(NPZ_DIR, "sample_metadata.csv")
with open(metadata_path, "r") as f:
    for row in csv.DictReader(f):
        sample_P[int(row["sample_id"])] = float(row["P_N"])


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
    P_kN = sample_P.get(idx, 0.0) / 1e3
    print(f"\n[{split}] {sample_id}  N={N}  T={T}  P={P_kN:.0f}kN")

    npz          = np.load(os.path.join(NPZ_DIR, sample_id + ".npz"))
    elem_conn    = npz["elem_conn"]
    elem_mat     = npz["elem_mat"]
    mat_E        = npz["mat_E"]
    E1           = float(mat_E[0])
    E2           = float(mat_E[1])
    stress_vm_np = npz["stress_vm"]
    print(f"  E1={E1/1e9:.0f} GPa  E2={E2/1e9:.1f} GPa")

    edge_index_np, edge_E_np = build_multigraph(elem_conn, elem_mat, mat_E)
    edge_index = torch.from_numpy(edge_index_np).to(device)
    edge_E     = torch.from_numpy(edge_E_np).to(device)

    ref_pos = data[0]["graph"].mesh_pos.to(device)
    ref_np  = ref_pos.cpu().numpy()

    connectivity, offsets, types, _ = build_surface_cells_from_elems(
        ref_np, elem_conn, 8, debug_label=sample_id
    )
    mat_id_np = node_material_id(elem_conn, elem_mat, N)

    mesh_ei = torch.from_numpy(npz["mesh_edge_index"].astype(np.int64)).to(device)
    mat_id_torch = torch.from_numpy(mat_id_np).to(device)
    node_E = torch.where(mat_id_torch == 0,
                         torch.full_like(mat_id_torch, E1),
                         torch.full_like(mat_id_torch, E2))

    node_type_oh = data[0]["graph"].x[:, 6:9].to(device)
    clamped_mask = (node_type_oh[:, 1] == 1)

    # GT world-pos reconstruction
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    # Autoregressive rollout (seed from GT at t=1), 9D node features (no P)
    cur      = gt_wp_list[1].to(device)
    prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    pred_wp_list = [gt_wp_list[0].numpy(), gt_wp_list[1].numpy()]

    with torch.inference_mode():
        for _ in range(T - 1):
            ef = build_rollout_ef(ref_pos, cur, edge_index, edge_E, E1,
                                  e_mean, e_std)
            disp_norm     = (cur - ref_pos - d_mean) / d_std
            prev_vel_norm = (prev_vel - v_mean) / v_std
            node_x        = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)
            g   = Data(x=node_x, edge_index=edge_index, num_nodes=N)
            out = model(node_x, ef, empty_wef, g)

            vel = out[:, :3] * v_std + v_mean
            vel[clamped_mask] = 0.0
            prev_vel = vel
            cur      = cur + vel
            pred_wp_list.append(cur.cpu().numpy())

    # Write VTU + PVD
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
                "displacement":   gt_disp,
                "disp_mag":       np.linalg.norm(gt_disp, axis=1),
                "von_mises":      gt_vm,
                "von_mises_gfdm": gt_vm_gfdm,
                "material_id":    mat_id_np,
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

    for t_idx in range(0, T, max(1, T // 5)):
        err = np.linalg.norm(
            pred_wp_list[t_idx + 1] - gt_wp_list[t_idx + 1].numpy(), axis=1
        ).mean() * 1000
        print(f"    t={t_idx:3d}: disp_err={err:.4f} mm")
    final_err = np.linalg.norm(
        pred_wp_list[-1] - gt_wp_list[-1].numpy(), axis=1
    ).mean()
    print(f"  final mean node error: {final_err*1000:.4f} mm")

print(f"\n完成！結果在 {OUT_DIR}/<sample_id>/gt/*.pvd  +  pred/*.pvd")