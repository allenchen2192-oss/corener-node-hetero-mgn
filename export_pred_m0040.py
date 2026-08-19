"""
export_pred_m0040.py
====================
Rollout HeteroMGN (M0040) and export GT + Pred to VTU/PVD for ParaView.

Seeded at t=1 from GT (position, element stress, PEEQ), then autoregressive
rollout for t=2..20 (19 model steps).

Output per sample:
  export_pred_m0040/S####/
    gt/
      t_001.vtu ... t_020.vtu   S####_gt.pvd          full mesh
      Si/     t_001.vtu ...     S####_Si_gt.pvd        Si only
      Solder/ t_001.vtu ...     S####_Solder_gt.pvd    Solder only
      UF/     t_001.vtu ...     S####_UF_gt.pvd        UF only
    pred/
      t_001.vtu ... t_020.vtu   S####_pred.pvd         full mesh
      Si/     t_001.vtu ...     S####_Si_pred.pvd
      Solder/ t_001.vtu ...     S####_Solder_pred.pvd
      UF/     t_001.vtu ...     S####_UF_pred.pvd

Full-mesh PointData: displacement(3D), disp_mag, velocity(3D),
                     vm_stress_node, peeq_node  [pred: + disp/vm/peeq error fields]
Full-mesh CellData:  vm_stress, peeq, material_id  [pred: + vm_error_pct (%), peeq_error]

Per-material:
  PointData: displacement, disp_mag, vm_stress_node, peeq_node  [pred: + error nodes]
  CellData:  vm_stress, peeq  [pred: + vm_error_pct (%), peeq_error]

Usage:
  python export_pred_m0040.py                  # default: S0201-S0205
  python export_pred_m0040.py S0001 S0201      # specific samples
"""

import os
import sys
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import HeteroData

from physicsnemo.utils import load_checkpoint
from train_m0040 import HeteroMGN

# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR   = "./02_abaqus_npz_m0040"
STATS_DIR = "./04_preprocessed_hetero_m0040"
CKPT_PATH = "./checkpoints_m0040"
OUT_DIR   = "./export_pred_m0040"

DEFAULT_SAMPLES = [f"S{i:04d}" for i in range(201, 206)]
SAMPLES = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SAMPLES

MODEL_CFG = SimpleNamespace(
    hidden_dim=256, processor_size=10,
    num_node_features=11, num_edge_features=8,
    num_elem_features=8,  num_node_outputs=3,
    num_elem_outputs=2,
)

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# material name -> list of elem_mat integer codes
MATERIALS = {
    "Si":     [0, 1],
    "Solder": [2],
    "UF":     [3],
}


# ── VTU / PVD writers ─────────────────────────────────────────────────────────

def _fmt(arr, fmt="%.6e"):
    return " ".join(fmt % v for v in np.asarray(arr).ravel())


def write_vtu(filename, points, hex_conn, point_data=None, cell_data=None):
    N, M = len(points), len(hex_conn)
    offsets    = np.arange(8, 8*(M+1), 8, dtype=np.int32)
    cell_types = np.full(M, 12, dtype=np.uint8)

    def da(name, arr, nc):
        return [f'<DataArray type="Float32" Name="{name}" '
                f'NumberOfComponents="{nc}" format="ascii">',
                _fmt(arr), "</DataArray>"]

    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
             "<UnstructuredGrid>",
             f'<Piece NumberOfPoints="{N}" NumberOfCells="{M}">',
             "<PointData>"]
    for name, arr in (point_data or {}).items():
        arr = np.asarray(arr, dtype=np.float32)
        lines.extend(da(name, arr, arr.shape[1] if arr.ndim == 2 else 1))
    lines.append("</PointData>")
    lines.append("<CellData>")
    for name, arr in (cell_data or {}).items():
        arr = np.asarray(arr, dtype=np.float32)
        lines.extend(da(name, arr, arr.shape[1] if arr.ndim == 2 else 1))
    lines += ["</CellData>", "<Points>",
              '<DataArray type="Float32" NumberOfComponents="3" format="ascii">',
              _fmt(points.astype(np.float32)), "</DataArray></Points>",
              "<Cells>",
              '<DataArray type="Int32" Name="connectivity" format="ascii">',
              _fmt(hex_conn.astype(np.int32), fmt="%d"), "</DataArray>",
              '<DataArray type="Int32" Name="offsets" format="ascii">',
              _fmt(offsets, fmt="%d"), "</DataArray>",
              '<DataArray type="UInt8" Name="types" format="ascii">',
              _fmt(cell_types, fmt="%d"),
              "</DataArray></Cells></Piece></UnstructuredGrid></VTKFile>"]
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def write_pvd(filename, entries):
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             "<Collection>"]
    for t, vtu in entries:
        lines.append(f'<DataSet timestep="{float(t):.4f}" group="" part="0" file="{vtu}"/>')
    lines += ["</Collection>", "</VTKFile>"]
    with open(filename, "w") as f:
        f.write("\n".join(lines))


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_singlegraph(elem_conn):
    seen = set(); rows, cols = [], []
    for c0 in elem_conn:
        for a, b in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u,v),(v,u)):
                if (src,dst) not in seen:
                    seen.add((src,dst)); rows.append(src); cols.append(dst)
    return np.array([rows, cols], dtype=np.int64)


def build_b02_edges(elem_conn):
    M, npe   = elem_conn.shape
    elem_ids = np.repeat(np.arange(M, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    return np.stack([node_ids, elem_ids]), np.stack([elem_ids, node_ids])


def compute_edge_feats(mesh_pos, world_pos_t, edge_index):
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


def node_type_onehot(node_type, n_classes=5):
    oh = np.zeros((len(node_type), n_classes), dtype=np.float32)
    for i, nt in enumerate(node_type):
        if 0 <= int(nt) < n_classes:
            oh[i, int(nt)] = 1.0
    return oh


def mat_onehot(elem_mat):
    oh = np.zeros((len(elem_mat), 3), dtype=np.float32)
    oh[:, 0] = ((elem_mat == 0) | (elem_mat == 1)).astype(np.float32)
    oh[:, 1] = (elem_mat == 2).astype(np.float32)
    oh[:, 2] = (elem_mat == 3).astype(np.float32)
    return oh


def elem_to_node(elem_val, elem_conn, N):
    val_sum = np.zeros(N, dtype=np.float64)
    cnt     = np.zeros(N, dtype=np.int64)
    np.add.at(val_sum, elem_conn.ravel(),
               np.repeat(elem_val.astype(np.float64), elem_conn.shape[1]))
    np.add.at(cnt, elem_conn.ravel(), 1)
    return (val_sum / np.maximum(cnt, 1)).astype(np.float32)


def build_submesh(mesh_pos, elem_conn, elem_mat, mat_ids):
    """
    Extract submesh for elements whose material code is in mat_ids.

    Returns
    -------
    unique_nodes  : (N_sub,) original node indices kept in submesh
    sub_conn      : (M_sub, 8) re-indexed connectivity (0-based into unique_nodes)
    elem_mask     : (M,) bool mask selecting elements
    """
    elem_mask    = np.isin(elem_mat, mat_ids)
    sub_conn_raw = elem_conn[elem_mask]
    unique_nodes = np.unique(sub_conn_raw)
    old2new      = np.full(mesh_pos.shape[0], -1, dtype=np.int64)
    old2new[unique_nodes] = np.arange(len(unique_nodes), dtype=np.int64)
    sub_conn     = old2new[sub_conn_raw]
    return unique_nodes, sub_conn, elem_mask


# ── Load stats ────────────────────────────────────────────────────────────────

with open(os.path.join(STATS_DIR, "stats_m0040.json")) as f:
    stats = json.load(f)

disp_mean  = np.array(stats["disp_mean"],  dtype=np.float32)
disp_std   = np.array(stats["disp_std"],   dtype=np.float32)
vel_mean   = np.array(stats["vel_mean"],   dtype=np.float32)
vel_std    = np.array(stats["vel_std"],    dtype=np.float32)
edge_mean  = np.array(stats["edge_mean"],  dtype=np.float32)
edge_std   = np.array(stats["edge_std"],   dtype=np.float32)
CTE_mean   = float(stats["CTE_mean"]);  CTE_std  = float(stats["CTE_std"])
nu_mean    = float(stats["nu_mean"]);   nu_std   = float(stats["nu_std"])
E_MIN      = float(stats["E_min"]);     E_MAX    = float(stats["E_max"])
peeq_mean_sl = float(stats["peeq_mean_solder"])
peeq_std_sl  = float(stats["peeq_std_solder"])

VM_MEAN = {0: stats["vm_mean_Si"], 1: stats["vm_mean_Si"],
           2: stats["vm_mean_Solder"], 3: stats["vm_mean_UF"]}
VM_STD  = {0: stats["vm_std_Si"],  1: stats["vm_std_Si"],
           2: stats["vm_std_Solder"],  3: stats["vm_std_UF"]}

# ── Load model ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = HeteroMGN(MODEL_CFG).to(device)
model.eval()
load_checkpoint(CKPT_PATH, models=model, device=device)
print(f"Checkpoint loaded from {CKPT_PATH}")

# ── Export loop ───────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

for sid in SAMPLES:
    npz_path = os.path.join(NPZ_DIR, f"{sid}.npz")
    if not os.path.exists(npz_path):
        print(f"[skip] {sid}.npz not found"); continue

    d = np.load(npz_path)
    mesh_pos       = d["mesh_pos"]           # (N, 3)
    world_pos      = d["world_pos"]          # (21, N, 3)
    velocity       = d["velocity"]           # (20, N, 3)
    elem_vm_stress = d["elem_vm_stress"]     # (21, M)
    elem_peeq      = d["elem_peeq"]          # (21, M)
    elem_conn      = d["elem_conn"]          # (M, 8)
    elem_mat       = d["elem_mat"]           # (M,)
    elem_E         = d["elem_E"]
    elem_CTE       = d["elem_CTE"]
    elem_nu        = d["elem_nu"]
    node_type      = d["node_type"]          # (N,)

    T           = velocity.shape[0]   # 20
    N           = mesh_pos.shape[0]
    M           = elem_mat.shape[0]
    solder_mask = elem_mat == 2

    gt_dir   = os.path.join(OUT_DIR, sid, "gt")
    pred_dir = os.path.join(OUT_DIR, sid, "pred")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"\n{sid}  N={N}  M={M}")

    # ── Pre-compute submesh per material (constant across time) ───────────────
    submeshes = {}
    for mat_name, mat_ids in MATERIALS.items():
        gt_sub_dir   = os.path.join(gt_dir,   mat_name)
        pred_sub_dir = os.path.join(pred_dir, mat_name)
        os.makedirs(gt_sub_dir,   exist_ok=True)
        os.makedirs(pred_sub_dir, exist_ok=True)
        uniq, sub_conn, emask = build_submesh(mesh_pos, elem_conn, elem_mat, mat_ids)
        submeshes[mat_name] = {
            "gt_dir":    gt_sub_dir,
            "pred_dir":  pred_sub_dir,
            "uniq":      uniq,
            "sub_conn":  sub_conn,
            "emask":     emask,
            "N_sub":     len(uniq),
            "gt_pvd":    [],
            "pred_pvd":  [],
        }
        print(f"  {mat_name}: N_sub={len(uniq)}  M_sub={emask.sum()}")

    # ── Static graph topology & features ─────────────────────────────────────
    edge_index  = build_singlegraph(elem_conn)
    n2e_np, e2n_np = build_b02_edges(elem_conn)
    edge_index_t = torch.from_numpy(edge_index).to(device)
    n2e_t        = torch.from_numpy(n2e_np).to(device)
    e2n_t        = torch.from_numpy(e2n_np).to(device)

    mat_oh  = mat_onehot(elem_mat)
    E_norm  = ((elem_E   - E_MIN)    / (E_MAX - E_MIN)).astype(np.float32)[:, None]
    CTE_n   = ((elem_CTE - CTE_mean) / CTE_std).astype(np.float32)[:, None]
    nu_n    = ((elem_nu  - nu_mean)  / nu_std).astype(np.float32)[:, None]
    elem_static = np.concatenate([mat_oh, E_norm, CTE_n, nu_n], axis=1)   # (M, 6)

    vm_mean_e = np.array([VM_MEAN[int(m)] for m in elem_mat], dtype=np.float32)
    vm_std_e  = np.array([VM_STD[int(m)]  for m in elem_mat], dtype=np.float32)

    nt_oh = node_type_onehot(node_type, n_classes=5)

    # ── Seed: t=1 from GT ────────────────────────────────────────────────────
    cur_pos       = world_pos[1].astype(np.float32)
    prev_vel_phys = velocity[0].astype(np.float32)
    vm_cur_phys   = elem_vm_stress[1].astype(np.float32)
    peeq_cur_phys = elem_peeq[1].astype(np.float32)

    gt_pvd, pred_pvd = [], []

    rollout_start = time.perf_counter()
    _t_infer = 0.0
    _t_io    = 0.0
    for fi in range(1, T + 1):   # fi = 1 .. 20

        # ── GT ───────────────────────────────────────────────────────────────
        gt_pos  = world_pos[fi].astype(np.float32)
        gt_disp = gt_pos - mesh_pos
        gt_vel  = velocity[fi - 1].astype(np.float32)
        gt_vm   = elem_vm_stress[fi].astype(np.float32)
        gt_peeq = elem_peeq[fi].astype(np.float32)

        gt_vm_node   = elem_to_node(gt_vm,   elem_conn, N)
        gt_peeq_node = elem_to_node(gt_peeq, elem_conn, N)

        vtu_name = f"t_{fi:03d}.vtu"

        # Full-mesh GT
        _t0_io = time.perf_counter()
        write_vtu(
            os.path.join(gt_dir, vtu_name), gt_pos, elem_conn,
            point_data={
                "displacement":   gt_disp,
                "disp_mag":       np.linalg.norm(gt_disp, axis=1).astype(np.float32),
                "velocity":       gt_vel,
                "vel_mag":        np.linalg.norm(gt_vel,  axis=1).astype(np.float32),
                "vm_stress_node": gt_vm_node,
                "peeq_node":      gt_peeq_node,
            },
            cell_data={
                "vm_stress":   gt_vm,
                "peeq":        gt_peeq,
                "material_id": elem_mat.astype(np.float32),
            },
        )
        gt_pvd.append((fi, vtu_name))

        # Per-material GT submesh
        for mat_name, sm in submeshes.items():
            uniq     = sm["uniq"]
            sub_conn = sm["sub_conn"]
            emask    = sm["emask"]
            N_sub    = sm["N_sub"]

            sub_pos  = gt_pos[uniq]
            sub_disp = gt_disp[uniq]
            sub_vm   = gt_vm[emask]
            sub_peeq = gt_peeq[emask]

            write_vtu(
                os.path.join(sm["gt_dir"], vtu_name), sub_pos, sub_conn,
                point_data={
                    "displacement":   sub_disp,
                    "disp_mag":       np.linalg.norm(sub_disp, axis=1).astype(np.float32),
                    "vm_stress_node": elem_to_node(sub_vm,   sub_conn, N_sub),
                    "peeq_node":      elem_to_node(sub_peeq, sub_conn, N_sub),
                },
                cell_data={
                    "vm_stress": sub_vm,
                    "peeq":      sub_peeq,
                },
            )
            sm["gt_pvd"].append((fi, vtu_name))

        # ── Pred ─────────────────────────────────────────────────────────────
        pred_disp = cur_pos - mesh_pos
        disp_err  = np.linalg.norm(cur_pos - gt_pos, axis=1).astype(np.float32)
        vm_err    = (np.abs(vm_cur_phys - gt_vm) / np.maximum(gt_vm, 1e-10) * 100).astype(np.float32)
        peeq_err  = np.abs(peeq_cur_phys - gt_peeq).astype(np.float32)

        pred_vm_node      = elem_to_node(vm_cur_phys,   elem_conn, N)
        pred_peeq_node    = elem_to_node(peeq_cur_phys, elem_conn, N)
        vm_err_node       = elem_to_node(vm_err,   elem_conn, N)
        peeq_err_node     = elem_to_node(peeq_err, elem_conn, N)

        # Full-mesh Pred
        write_vtu(
            os.path.join(pred_dir, vtu_name), cur_pos, elem_conn,
            point_data={
                "displacement":    pred_disp,
                "disp_mag":        np.linalg.norm(pred_disp, axis=1).astype(np.float32),
                "velocity":        prev_vel_phys,
                "vel_mag":         np.linalg.norm(prev_vel_phys, axis=1).astype(np.float32),
                "vm_stress_node":  pred_vm_node,
                "peeq_node":       pred_peeq_node,
                "disp_error":      disp_err,
                "vm_error_pct_node":  vm_err_node,
                "peeq_error_node":    peeq_err_node,
            },
            cell_data={
                "vm_stress":      vm_cur_phys,
                "peeq":           peeq_cur_phys,
                "vm_error_pct":   vm_err,
                "peeq_error":  peeq_err,
                "material_id": elem_mat.astype(np.float32),
            },
        )
        pred_pvd.append((fi, vtu_name))

        # Per-material Pred submesh
        for mat_name, sm in submeshes.items():
            uniq     = sm["uniq"]
            sub_conn = sm["sub_conn"]
            emask    = sm["emask"]
            N_sub    = sm["N_sub"]

            sub_pred_pos  = cur_pos[uniq]
            sub_pred_disp = pred_disp[uniq]
            sub_vm_pred   = vm_cur_phys[emask]
            sub_peeq_pred = peeq_cur_phys[emask]
            sub_vm_err    = vm_err[emask]
            sub_peeq_err  = peeq_err[emask]
            sub_disp_err  = np.linalg.norm(cur_pos[uniq] - gt_pos[uniq], axis=1).astype(np.float32)

            write_vtu(
                os.path.join(sm["pred_dir"], vtu_name), sub_pred_pos, sub_conn,
                point_data={
                    "displacement":    sub_pred_disp,
                    "disp_mag":        np.linalg.norm(sub_pred_disp, axis=1).astype(np.float32),
                    "disp_error":      sub_disp_err,
                    "vm_stress_node":  elem_to_node(sub_vm_pred,  sub_conn, N_sub),
                    "peeq_node":       elem_to_node(sub_peeq_pred, sub_conn, N_sub),
                    "vm_error_pct_node":  elem_to_node(sub_vm_err,   sub_conn, N_sub),
                    "peeq_error_node":    elem_to_node(sub_peeq_err, sub_conn, N_sub),
                },
                cell_data={
                    "vm_stress":     sub_vm_pred,
                    "peeq":          sub_peeq_pred,
                    "vm_error_pct":  sub_vm_err,
                    "peeq_error": sub_peeq_err,
                },
            )
            sm["pred_pvd"].append((fi, vtu_name))
        _t_io += time.perf_counter() - _t0_io

        # ── Model rollout: predict fi → fi+1 ─────────────────────────────────
        if fi < T:
            with torch.inference_mode():
                disp_n = (cur_pos - mesh_pos - disp_mean) / disp_std
                prev_n = (prev_vel_phys       - vel_mean)  / vel_std
                node_x = np.concatenate([disp_n, prev_n, nt_oh], axis=1)  # (N, 11)

                vm_n  = ((vm_cur_phys - vm_mean_e) / vm_std_e)[:, None]
                pq_n  = np.zeros(M, dtype=np.float32)
                pq_n[solder_mask] = (
                    (peeq_cur_phys[solder_mask] - peeq_mean_sl) / peeq_std_sl
                )
                elem_x = np.concatenate([elem_static, vm_n, pq_n[:, None]], axis=1)  # (M, 8)

                ef_norm = ((compute_edge_feats(mesh_pos, cur_pos, edge_index)
                            - edge_mean) / edge_std).astype(np.float32)

                hd = HeteroData()
                hd["node"].x    = torch.from_numpy(node_x).to(device)
                hd["element"].x = torch.from_numpy(elem_x).to(device)
                hd["node",    "mesh", "node"   ].edge_index = edge_index_t
                hd["node",    "mesh", "node"   ].edge_attr  = torch.from_numpy(ef_norm).to(device)
                hd["node",    "in",   "element"].edge_index = n2e_t
                hd["element", "has",  "node"   ].edge_index = e2n_t

                _t0 = time.perf_counter()
                vel_pred, elem_pred = model(hd)
                if device == "cuda":
                    torch.cuda.synchronize()
                _t_infer += time.perf_counter() - _t0

            prev_vel_phys = (vel_pred.cpu().numpy() * vel_std + vel_mean).astype(np.float32)
            cur_pos       = cur_pos + prev_vel_phys

            vm_cur_phys   = (elem_pred[:, 0].cpu().numpy() * vm_std_e + vm_mean_e).astype(np.float32)
            peeq_cur_phys = np.zeros(M, dtype=np.float32)
            peeq_cur_phys[solder_mask] = (
                elem_pred[solder_mask, 1].cpu().numpy() * peeq_std_sl + peeq_mean_sl
            )
            peeq_cur_phys = np.clip(peeq_cur_phys, 0.0, None)

    # ── Write PVD files ───────────────────────────────────────────────────────
    write_pvd(os.path.join(gt_dir,   f"{sid}_gt.pvd"),   gt_pvd)
    write_pvd(os.path.join(pred_dir, f"{sid}_pred.pvd"), pred_pvd)
    for mat_name, sm in submeshes.items():
        write_pvd(os.path.join(sm["gt_dir"],   f"{sid}_{mat_name}_gt.pvd"),   sm["gt_pvd"])
        write_pvd(os.path.join(sm["pred_dir"], f"{sid}_{mat_name}_pred.pvd"), sm["pred_pvd"])

    rollout_time = time.perf_counter() - rollout_start
    _t_other = rollout_time - _t_infer - _t_io
    print(f"  rollout time : {rollout_time:.3f} s  ({rollout_time/T*1000:.1f} ms/step, {T} steps)")
    print(f"    model infer: {_t_infer:.3f} s  ({_t_infer/(T-1)*1000:.1f} ms/step, {T-1} steps)")
    print(f"    file I/O   : {_t_io:.3f} s  ({_t_io/T*1000:.1f} ms/step)")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_gt_pos  = world_pos[T].astype(np.float32)
    final_disp_um = np.linalg.norm(cur_pos - final_gt_pos, axis=1).mean() * 1e3
    final_vm_gt   = elem_vm_stress[T].astype(np.float32)
    final_vm_err  = (np.abs(vm_cur_phys - final_vm_gt).mean()
                     / max(final_vm_gt.mean(), 1e-10) * 100)
    peeq_gt_sl    = elem_peeq[T][solder_mask]
    peeq_pr_sl    = peeq_cur_phys[solder_mask]
    final_peeq_err = (np.abs(peeq_pr_sl - peeq_gt_sl).mean()
                      / max(peeq_gt_sl.mean(), 1e-10) * 100)
    print(f"  final disp_err={final_disp_um:.3f} um  "
          f"vm_err={final_vm_err:.1f}%  peeq_err={final_peeq_err:.1f}%")

print(f"\nDone -> {OUT_DIR}/")
print("Open gt/<sid>_gt.pvd or gt/Si/<sid>_Si_gt.pvd (and pred/) in ParaView.")
