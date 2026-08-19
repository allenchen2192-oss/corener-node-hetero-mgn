"""
export_hetero_paraview.py
=========================
Rollout HeteroMGN (Case 5) and export GT + Pred to VTU/PVD for ParaView.

Element-level stress is extrapolated to nodes by averaging over connected
elements (standard FEM nodal recovery).

Output:
  export_hetero_paraview/
    sample_XXXXX/
      gt/   t_000.vtu ... sample_XXXXX_gt.pvd
      pred/ t_000.vtu ... sample_XXXXX_pred.pvd

PointData fields:
  displacement      (vector, 3D, m)
  disp_mag          (scalar, m)
  von_mises_abaqus  (scalar, Pa)   GT only — from Abaqus
  von_mises_bmat    (scalar, Pa)   GT: B-matrix element stress extrapolated to nodes
  von_mises_pred    (scalar, Pa)   Pred: model element stress extrapolated to nodes
  disp_error        (scalar, m)    Pred only
  stress_error      (scalar, Pa)   Pred only (|pred_elem - gt_elem| extrapolated to nodes)
  material_id       (scalar)       0=Mat1 1=Mat2

Usage:
  python export_hetero_paraview.py
"""

import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import HeteroData

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.utils import load_checkpoint

sys.path.insert(0, os.path.dirname(__file__))
from export_abaqus_paraview import (
    build_surface_cells_from_elems,
    write_vtu,
    write_pvd,
)
from train_bimaterial_hetero import HeteroMGN, compute_elem_vm_stress


_C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

def build_singlegraph(elem_conn, elem_mat, mat_E):
    seen = set()
    rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E[mat_idx])
        for (a, b) in _C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst)
                if key not in seen:
                    seen.add(key)
                    rows.append(src);  cols.append(dst);  E_vals.append(E_elem)
    ei = np.array([rows, cols], dtype=np.int64)
    eE = np.array(E_vals, dtype=np.float32)
    return ei, eE

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR  = "./04_preprocessed_pt_bimaterial"
NPZ_DIR   = "./02_abaqus_npz_bimaterial"
CKPT_PATH = "./checkpoints_bimaterial_hetero"
OUT_DIR   = "./export_hetero_paraview"

SAMPLE_INDICES = list(range(3)) + list(range(1000, 1003))

ELEM_E_MIN = 50.0e9
ELEM_E_MAX = 200.0e9

# ── Helpers ─────────────────────────────────────────────────────────────────────

def build_b02_edges(elem_conn):
    n_elem, npe = elem_conn.shape
    elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    n2e = torch.from_numpy(np.stack([node_ids, elem_ids]))
    e2n = torch.from_numpy(np.stack([elem_ids, node_ids]))
    return n2e, e2n


def build_ef(mesh_pos, world_pos, edge_index, edge_E, E_ref, e_mean, e_std):
    src, dst = edge_index
    d_mesh  = mesh_pos[dst]  - mesh_pos[src]
    d_world = world_pos[dst] - world_pos[src]
    ef = torch.cat([
        d_mesh,  d_mesh.norm(dim=1, keepdim=True),
        d_world, d_world.norm(dim=1, keepdim=True),
        (edge_E / E_ref).unsqueeze(1),
    ], dim=1)
    return (ef - e_mean) / e_std


def elem_to_node(elem_val, elem_conn, N):
    """Average element scalar values onto nodes."""
    val_sum = np.zeros(N, dtype=np.float64)
    cnt     = np.zeros(N, dtype=np.int32)
    np.add.at(val_sum, elem_conn.ravel(),
               np.repeat(elem_val.astype(np.float64), elem_conn.shape[1]))
    np.add.at(cnt, elem_conn.ravel(), 1)
    return (val_sum / np.maximum(cnt, 1)).astype(np.float32)


def node_material_id(elem_conn, elem_mat, N):
    mat_count = np.zeros((N, 2), dtype=np.int32)
    for c0, m in zip(elem_conn, elem_mat):
        for ni in c0:
            mat_count[ni, int(m)] += 1
    return np.argmax(mat_count, axis=1).astype(np.float32)


# ── Device & stats ──────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats_bimaterial.json"))
v_mean = node_stats["velocity_mean"].to(device)
v_std  = node_stats["velocity_std"].to(device)
d_mean = node_stats["disp_mean"].to(device)
d_std  = node_stats["disp_std"].to(device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

elem_s_mean = float(node_stats.get("bmat_stress_mean", 0.0))
elem_s_std  = max(float(node_stats.get("bmat_stress_std", 1.0)), 1e-6)

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats_bimaterial.json"))
e_mean = edge_stats["edge_mean"].to(device)
e_std  = edge_stats["edge_std"].to(device)

# ── Load model ──────────────────────────────────────────────────────────────────

cfg = SimpleNamespace(
    hidden_dim=128, processor_size=4,
    num_node_features=9, num_edge_features=8,
    num_elem_features=3, num_node_outputs=3,
)
model = HeteroMGN(cfg).to(device)
model.eval()
print(f"Loading checkpoint: {CKPT_PATH}")
load_checkpoint(CKPT_PATH, models=model, device=device)

# ── File list & metadata ────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"Found {len(all_files)} .pt files")

sample_P = {}
with open(os.path.join(NPZ_DIR, "sample_metadata.csv")) as f:
    for row in csv.DictReader(f):
        sample_P[int(row["sample_id"])] = float(row["P_N"])

# ── Export loop ─────────────────────────────────────────────────────────────────

for idx in SAMPLE_INDICES:
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range"); continue

    fname     = all_files[idx]
    sample_id = fname.replace(".pt", "")
    split     = "train" if idx < 1000 else "test"

    gt_dir   = os.path.join(OUT_DIR, sample_id, "gt")
    pred_dir = os.path.join(OUT_DIR, sample_id, "pred")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    data = torch.load(os.path.join(DATA_DIR, fname),
                      map_location="cpu", weights_only=False)
    T = len(data)
    N = data[0]["graph"].num_nodes
    print(f"\n[{split}] {sample_id}  N={N}  T={T}")

    # ── NPZ topology & GT stress ───────────────────────────────────────────────
    npz          = np.load(os.path.join(NPZ_DIR, sample_id + ".npz"))
    elem_conn    = npz["elem_conn"].astype(np.int64)
    elem_mat     = npz["elem_mat"].astype(np.int64)
    mat_E        = npz["mat_E"].astype(np.float64)
    mat_nu       = npz["mat_nu"].astype(np.float64)
    E1           = float(mat_E[0])
    stress_vm_np = npz["stress_vm"]   # (F, N) node-level from Abaqus

    elem_E_arr = mat_E[elem_mat]
    elem_nu_arr = mat_nu[elem_mat]

    # ── Graph topology ─────────────────────────────────────────────────────────
    ei_np, eE_np = build_singlegraph(elem_conn, elem_mat, mat_E)
    edge_index = torch.from_numpy(ei_np).to(device)
    edge_E     = torch.from_numpy(eE_np).to(device)

    n2e, e2n = build_b02_edges(elem_conn)
    n2e = n2e.to(device); e2n = e2n.to(device)

    # ── Element static features ────────────────────────────────────────────────
    mat_type = torch.from_numpy(elem_mat.astype(np.float32)).unsqueeze(1).to(device)
    E_norm_e = torch.from_numpy(
        ((elem_E_arr - ELEM_E_MIN) / (ELEM_E_MAX - ELEM_E_MIN)).astype(np.float32)
    ).unsqueeze(1).to(device)
    elem_base = torch.cat([mat_type, E_norm_e], dim=1)   # (E_elem, 2)

    elem_E_t    = torch.from_numpy(elem_E_arr).to(device)
    elem_nu_t   = torch.from_numpy(elem_nu_arr).to(device)
    elem_conn_t = torch.from_numpy(elem_conn).to(device)

    # ── Reference & surface cells ──────────────────────────────────────────────
    ref_pos = data[0]["graph"].mesh_pos.to(device)
    ref_np  = ref_pos.cpu().numpy()
    node_type = data[0]["graph"].x[:, 6:9].to(device)

    connectivity, offsets, types, _ = build_surface_cells_from_elems(
        ref_np, elem_conn, 8, debug_label=sample_id
    )
    mat_id_np = node_material_id(elem_conn, elem_mat, N)

    # ── GT world-pos reconstruction ────────────────────────────────────────────
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        gt_wp_list.append(gt_wp_list[-1] + step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)

    # ── Autoregressive rollout ─────────────────────────────────────────────────
    cur      = gt_wp_list[1].to(device)
    prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    pred_wp_list = [gt_wp_list[0].numpy(), gt_wp_list[1].numpy()]

    # seed prev_stress from GT at t=1
    with torch.inference_mode():
        disp_t1 = (gt_wp_list[1].to(device) - ref_pos).float()
        gt_stress_t1 = compute_elem_vm_stress(
            disp_t1, ref_pos, elem_conn_t, elem_E_t, elem_nu_t
        )
    prev_stress_norm = ((gt_stress_t1 - elem_s_mean) / elem_s_std).unsqueeze(1)

    pred_elem_stress_list = []   # (T-1,) each entry is (E_elem,) Pa

    with torch.inference_mode():
        for step_i in range(T - 1):
            ea = build_ef(ref_pos, cur, edge_index, edge_E, E1, e_mean, e_std)

            disp_n = (cur - ref_pos - d_mean) / d_std
            vel_n  = (prev_vel - v_mean) / v_std
            node_x = torch.cat([disp_n, vel_n, node_type], dim=1)

            elem_x = torch.cat([elem_base, prev_stress_norm], dim=1)

            hd = HeteroData()
            hd["node"].x    = node_x
            hd["element"].x = elem_x
            hd["node",    "mesh", "node"   ].edge_index = edge_index
            hd["node",    "mesh", "node"   ].edge_attr  = ea
            hd["node",    "in",   "element"].edge_index = n2e
            hd["element", "has",  "node"   ].edge_index = e2n

            vel_pred, stress_pred = model(hd)
            prev_stress_norm = stress_pred.detach()

            vel = vel_pred * v_std + v_mean
            prev_vel = vel
            cur = cur + vel
            pred_wp_list.append(cur.cpu().numpy())

            pred_s_pa = (stress_pred.squeeze(1) * elem_s_std + elem_s_mean).cpu().numpy()
            pred_elem_stress_list.append(pred_s_pa)

            # diagnostic: first step sanity
            if step_i == 0:
                gt_s1 = compute_elem_vm_stress(
                    (gt_wp_list[2].to(device) - ref_pos).float(),
                    ref_pos, elem_conn_t, elem_E_t, elem_nu_t,
                ).cpu().numpy()
                err1 = np.abs(pred_s_pa - gt_s1).mean() / max(gt_s1.mean(), 1e-10) * 100
                print(f"    [step1] gt_stress_mean={gt_s1.mean():.2e} Pa  "
                      f"pred_stress_mean={pred_s_pa.mean():.2e} Pa  err={err1:.2f}%")

    # ── Write VTU/PVD ─────────────────────────────────────────────────────────
    gt_pvd, pred_pvd = [], []

    for t_idx in range(T):
        vtu_name = f"t_{t_idx:03d}.vtu"

        gt_pos  = gt_wp_list[t_idx + 1].numpy().astype(np.float32)
        gt_disp = gt_pos - ref_np

        # GT stresses
        gt_vm_abaqus = stress_vm_np[t_idx + 1].astype(np.float32)   # node-level from Abaqus
        with torch.inference_mode():
            gt_elem_vm = compute_elem_vm_stress(
                torch.from_numpy(gt_disp).to(device),
                ref_pos, elem_conn_t, elem_E_t, elem_nu_t,
            ).cpu().numpy()
        gt_vm_bmat_node = elem_to_node(gt_elem_vm, elem_conn, N)

        write_vtu(
            os.path.join(gt_dir, vtu_name),
            gt_pos, connectivity, offsets, types,
            {
                "displacement":     gt_disp,
                "disp_mag":         np.linalg.norm(gt_disp, axis=1),
                "von_mises_abaqus": gt_vm_abaqus,
                "von_mises_bmat":   gt_vm_bmat_node,
                "material_id":      mat_id_np,
            },
        )
        gt_pvd.append((t_idx, vtu_name))

        pred_pos  = pred_wp_list[t_idx + 1].astype(np.float32)
        pred_disp = pred_pos - ref_np
        disp_err  = np.linalg.norm(pred_pos - gt_pos, axis=1).astype(np.float32)

        # Predicted element stress (t=1 seeded from GT, t>1 from model)
        if t_idx > 0:
            pred_elem_vm = pred_elem_stress_list[t_idx - 1]
        else:
            pred_elem_vm = gt_elem_vm  # t=1: seeded from GT

        pred_vm_node  = elem_to_node(pred_elem_vm,  elem_conn, N)
        stress_err_node = elem_to_node(
            np.abs(pred_elem_vm - gt_elem_vm), elem_conn, N
        )

        write_vtu(
            os.path.join(pred_dir, vtu_name),
            pred_pos, connectivity, offsets, types,
            {
                "displacement":    pred_disp,
                "disp_mag":        np.linalg.norm(pred_disp, axis=1),
                "von_mises_pred":  pred_vm_node,
                "von_mises_bmat":  gt_vm_bmat_node,
                "disp_error":      disp_err,
                "stress_error":    stress_err_node,
                "material_id":     mat_id_np,
            },
        )
        pred_pvd.append((t_idx, vtu_name))

    write_pvd(os.path.join(gt_dir,   f"{sample_id}_gt.pvd"),   gt_pvd)
    write_pvd(os.path.join(pred_dir, f"{sample_id}_pred.pvd"), pred_pvd)

    final_disp_err = np.linalg.norm(
        pred_wp_list[-1] - gt_wp_list[-1].numpy(), axis=1
    ).mean() * 1000

    pred_last = pred_elem_stress_list[-1]
    with torch.inference_mode():
        gt_last = compute_elem_vm_stress(
            torch.from_numpy((gt_wp_list[-1].numpy() - ref_np)).to(device),
            ref_pos, elem_conn_t, elem_E_t, elem_nu_t,
        ).cpu().numpy()
    final_stress_err = np.abs(pred_last - gt_last).mean() / max(gt_last.mean(), 1e-10) * 100
    print(f"  final disp_err={final_disp_err:.3f} mm  stress_err={final_stress_err:.2f}%")
    print(f"    gt_last stress: mean={gt_last.mean():.2f} Pa  pred_last: mean={pred_last.mean():.2f} Pa")

print(f"\n完成！結果在 {OUT_DIR}/<sample_id>/gt/*.pvd  +  pred/*.pvd")
