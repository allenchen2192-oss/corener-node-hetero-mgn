"""
export_abaqus_paraview_prevvel_ds.py
=====================================
Run rollout with the prevvel + direct stress model and export GT + predicted
results to VTU / PVD format for side-by-side comparison in ParaView.

Model: checkpoints_abaqus_prevvel_direct_stress_bs256
  Input  : disp_vec(3) + prev_vel(3) = 6D
  Output : velocity(3) + stress_vm(1) = 4D  ← model predicts stress directly

Output structure:
  export_abaqus_paraview_prevvel_ds/
    sample_XXXXX/
      gt/    t_000.vtu ...  sample_XXXXX_gt.pvd
      pred/  t_000.vtu ...  sample_XXXXX_pred.pvd

PointData fields:
  GT:
    displacement        (vector, m)
    disp_mag            (scalar, m)
    von_mises           (scalar, Pa)  — from Abaqus NPZ labels
    von_mises_computed  (scalar, Pa)  — scatter linear fit (for comparison)

  Pred:
    displacement        (vector, m)
    disp_mag            (scalar, m)
    von_mises_direct    (scalar, Pa)  — model direct prediction (denormalized)
    von_mises_computed  (scalar, Pa)  — scatter linear fit from predicted positions
    disp_error          (scalar, m)   — ||pred_pos - gt_pos||
    stress_error        (scalar, Pa)  — |von_mises_direct - gt_von_mises|
      (stress_error = 0 at t_idx=0 because that is the GT seed step)

Usage:
  python export_abaqus_paraview_prevvel_ds.py
"""

import os
import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 6    # disp_vec(3) + prev_vel(3)
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 4    # velocity(3) + stress_vm(1)
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 64

DATA_DIR   = "./04_preprocessed_pt_abaqus_prevvel"
CKPT_PATH  = "./checkpoints_abaqus_prevvel_direct_stress_bs128"
CKPT_EPOCH = None          # None = latest; set e.g. 500 for a specific epoch
OUTPUT_DIR = "./export_abaqus_paraview_prevvel_ds_bs128"

# Indices into the sorted .pt file list (test set = files sample_02000 onward)
SAMPLE_INDICES = [0, 1, 2, 2000, 2001, 2002, 2003, 2004, 2005]

NPZ_DIR = "./02_abaqus_npz"  # for elem_conn + GT stress_vm


# ── VTU / PVD helpers ─────────────────────────────────────────────────────────

def build_surface_cells(points_np, tol=1e-4):
    """Build surface cells for each axis-aligned face via structured quad or Delaunay fallback."""
    mins   = points_np.min(axis=0)
    maxs   = points_np.max(axis=0)
    ranges = np.where((maxs - mins) > 0, maxs - mins, 1.0)
    p_norm = (points_np - mins) / ranges

    face_specs = [
        (0, 0.0, [1, 2], True),
        (0, 1.0, [1, 2], False),
        (1, 0.0, [0, 2], False),
        (1, 1.0, [0, 2], True),
        (2, 0.0, [0, 1], True),
        (2, 1.0, [0, 1], False),
    ]

    conn_list, off_list, type_list = [], [], []
    cur_off = 0

    for axis, val, proj, flip in face_specs:
        mask = np.abs(p_norm[:, axis] - val) < tol
        idx  = np.where(mask)[0]
        if len(idx) < 3:
            continue
        pts_2d = p_norm[idx][:, proj]
        u0 = np.sort(np.unique(np.round(pts_2d[:, 0], 8)))
        u1 = np.sort(np.unique(np.round(pts_2d[:, 1], 8)))
        n0, n1 = len(u0), len(u1)
        if n0 * n1 == len(idx):
            p0_idx = np.searchsorted(u0, np.round(pts_2d[:, 0], 8))
            p1_idx = np.searchsorted(u1, np.round(pts_2d[:, 1], 8))
            grid = np.full((n0, n1), -1, dtype=np.int32)
            for k in range(len(idx)):
                grid[p0_idx[k], p1_idx[k]] = idx[k]
            for i in range(n0 - 1):
                for j in range(n1 - 1):
                    a, b, c, e = grid[i,j], grid[i+1,j], grid[i+1,j+1], grid[i,j+1]
                    if a >= 0 and b >= 0 and c >= 0 and e >= 0:
                        quad = [a, e, c, b] if flip else [a, b, c, e]
                        conn_list.extend(quad)
                        cur_off += 4
                        off_list.append(cur_off)
                        type_list.append(9)
        else:
            tri = Delaunay(pts_2d)
            for t in tri.simplices:
                conn_list.extend(idx[t].tolist())
                cur_off += 3
                off_list.append(cur_off)
                type_list.append(5)

    return (np.array(conn_list, dtype=np.int32),
            np.array(off_list,  dtype=np.int32),
            np.array(type_list, dtype=np.uint8))


_FACE_LOCAL_IDX = {
    8:  [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
    10: [[0,1,2],[0,3,1],[1,3,2],[0,2,3]],
    20: [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
}
_FACE_VTK_TYPE = {8: 9, 10: 5, 20: 9}
_FACE_CORNER_N = {8: 8, 10: 4, 20: 8}


def build_surface_cells_from_elems(points_np, elem_conn_np, n_nodes_per_elem, debug_label=""):
    """Build VTK surface cells from Abaqus element connectivity."""
    face_defs = _FACE_LOCAL_IDX.get(n_nodes_per_elem)
    vtk_type  = _FACE_VTK_TYPE.get(n_nodes_per_elem)
    corner_n  = _FACE_CORNER_N.get(n_nodes_per_elem)

    if face_defs is None:
        print(f"  [{debug_label}] unknown npe={n_nodes_per_elem}, Delaunay fallback")
        conn, offs, types = build_surface_cells(points_np)
        return conn, offs, types, np.zeros((points_np.shape[0], 3), dtype=np.float32)

    conn_corners = elem_conn_np[:, :corner_n]
    mesh_center  = points_np.mean(axis=0)

    face_map = {}
    for e_idx in range(len(conn_corners)):
        elem = conn_corners[e_idx]
        for face_local in face_defs:
            g   = elem[np.array(face_local)]
            key = tuple(sorted(g.tolist()))
            face_map[key] = None if key in face_map else g.copy()

    surface_faces = [v for v in face_map.values() if v is not None]
    if not surface_faces:
        print(f"  [{debug_label}] no boundary faces detected!")
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.uint8), np.zeros((points_np.shape[0], 3), dtype=np.float32))

    N = points_np.shape[0]
    point_normals = np.zeros((N, 3), dtype=np.float64)
    conn_list, off_list, type_list = [], [], []
    cur_off = 0

    for face_nodes in surface_faces:
        pts      = points_np[face_nodes]
        face_ctr = pts.mean(axis=0)
        normal   = np.cross(pts[1] - pts[0], pts[2] - pts[0]).astype(np.float64)
        nlen     = np.linalg.norm(normal)
        if nlen < 1e-30:
            continue
        normal /= nlen
        if np.dot(normal, face_ctr - mesh_center) < 0:
            face_nodes = face_nodes[::-1].copy()
            normal     = -normal
        conn_list.extend(face_nodes.tolist())
        cur_off += len(face_nodes)
        off_list.append(cur_off)
        type_list.append(vtk_type)
        for ni in face_nodes:
            point_normals[ni] += normal

    norms = np.linalg.norm(point_normals, axis=1, keepdims=True)
    point_normals /= np.where(norms < 1e-12, 1.0, norms)

    print(f"  [{debug_label}] npe={n_nodes_per_elem}, "
          f"{len(elem_conn_np)} elems → {len(surface_faces)} surface faces")

    return (np.array(conn_list, dtype=np.int32),
            np.array(off_list,  dtype=np.int32),
            np.array(type_list, dtype=np.uint8),
            point_normals.astype(np.float32))


def _arr_str(arr):
    return " ".join(f"{v:.6g}" for v in np.asarray(arr, dtype=np.float32).ravel())


def write_vtu(path, points, connectivity, offsets, types, point_data):
    N, n_cells = points.shape[0], len(types)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{N}" NumberOfCells="{n_cells}">',
        '      <Points>',
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
        f'          {_arr_str(points)}',
        '        </DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int32" Name="connectivity" format="ascii">',
        f'          {_arr_str(connectivity)}',
        '        </DataArray>',
        '        <DataArray type="Int32" Name="offsets" format="ascii">',
        f'          {_arr_str(offsets)}',
        '        </DataArray>',
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        f'          {_arr_str(types)}',
        '        </DataArray>',
        '      </Cells>',
        '      <PointData>',
    ]
    for name, arr in point_data.items():
        arr = np.asarray(arr, dtype=np.float32)
        nc  = arr.shape[1] if arr.ndim > 1 else None
        if nc:
            lines.append(f'        <DataArray type="Float32" Name="{name}" '
                         f'NumberOfComponents="{nc}" format="ascii">')
        else:
            lines.append(f'        <DataArray type="Float32" Name="{name}" format="ascii">')
        lines.append(f'          {_arr_str(arr)}')
        lines.append('        </DataArray>')
    lines += ['      </PointData>', '    </Piece>', '  </UnstructuredGrid>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_pvd(path, vtu_entries):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t_val, vtu_rel in vtu_entries:
        lines.append(f'    <DataSet timestep="{t_val:.4f}" group="" part="0" file="{vtu_rel}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Physics helpers ───────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    """4D: [disp_vec(3), dist(1)], convention dst-src to match preprocess_abaqus.py."""
    row, col = edge_index
    d = pos[col] - pos[row]
    n = torch.norm(d, dim=1, keepdim=True)
    return torch.cat([d, n], dim=1)


E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


def compute_vm_stress(wp, ref, mesh_ei):
    """Scatter linear fit von Mises stress (Pa), returns (N,) tensor."""
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    u     = wp.float() - ref.float()
    du_ij = u[col] - u[row]
    r_r   = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    N = wp.shape[0]
    XtX = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtY = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtX.scatter_add_(0, row_exp, r_r)
    XtY.scatter_add_(0, row_exp, r_du)
    XtX = XtX + 1e-6 * torch.eye(3, device=wp.device, dtype=torch.float32).unsqueeze(0)
    dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy  = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz  = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                              + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)


# ── Load normalization stats ──────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean     = node_stats["velocity_mean"].to(device)
v_std      = node_stats["velocity_std"].to(device)
s_mean_val = float(node_stats["stress_mean"])
s_std_val  = float(node_stats["stress_std"])
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats  = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref = edge_stats["edge_mean"].to(device)[:4]
e_std_ref  = edge_stats["edge_std"].to(device)[:4]
e_mean_wld = edge_stats["edge_mean"].to(device)[4:]
e_std_wld  = edge_stats["edge_std"].to(device)[4:]

empty_wef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)


# ── Load model ────────────────────────────────────────────────────────────────

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

ep_tag = f" epoch={CKPT_EPOCH}" if CKPT_EPOCH else " (latest)"
print(f"Loading checkpoint from {CKPT_PATH}{ep_tag}")
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)


# ── Main export loop ──────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"Found {len(all_files)} sample files in {DATA_DIR}")

for idx in SAMPLE_INDICES:
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range ({len(all_files)} files)")
        continue

    fname     = all_files[idx]
    sample_id = fname.replace(".pt", "")
    fpath     = os.path.join(DATA_DIR, fname)

    gt_dir   = os.path.join(OUTPUT_DIR, sample_id, "gt")
    pred_dir = os.path.join(OUTPUT_DIR, sample_id, "pred")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    data = torch.load(fpath, map_location="cpu", weights_only=False)
    T    = len(data)
    N    = data[0]["graph"].num_nodes
    print(f"\n[{sample_id}] N={N}, T={T} → {os.path.join(OUTPUT_DIR, sample_id)}")

    ref_pos = data[0]["graph"].mesh_pos.to(device)
    mesh_ei = data[0]["graph"].edge_index.to(device)
    ref_np  = ref_pos.cpu().numpy()

    # Surface topology from NPZ elem_conn
    npz_d            = np.load(os.path.join(NPZ_DIR, sample_id + ".npz"))
    elem_conn_np     = npz_d["elem_conn"]
    n_nodes_per_elem = int(npz_d["n_nodes_per_elem"])
    connectivity, offsets, types, ref_normals = build_surface_cells_from_elems(
        ref_np, elem_conn_np, n_nodes_per_elem, debug_label=sample_id
    )

    ref_feat_n = (compute_edge_features(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref

    # ── GT sequences ─────────────────────────────────────────────────────────
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    stress_vm_np   = npz_d["stress_vm"]                         # (F, N) Pa
    gt_stress_list = [stress_vm_np[t + 1].astype(np.float32)
                      for t in range(T)]

    # ── Autoregressive rollout ────────────────────────────────────────────────
    # Seed at t=1 with GT position and prev_vel = GT vel at t=0
    gt_vel_0 = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur      = gt_wp_list[1].to(device)
    prev_vel = gt_vel_0

    pred_wp_list     = [gt_wp_list[0].numpy(), gt_wp_list[1].numpy()]  # t=0,1: GT seed
    pred_stress_list = [gt_stress_list[0]]   # t_idx=0 seed: use GT stress (no model pred yet)

    with torch.inference_mode():
        for k in range(T - 1):   # predict t=2..T
            world_feat = (compute_edge_features(cur, mesh_ei) - e_mean_wld) / e_std_wld
            mesh_ef    = torch.cat([ref_feat_n, world_feat], dim=1)

            node_x = torch.cat([cur - ref_pos, prev_vel], dim=1)
            g      = Data(x=node_x, edge_index=mesh_ei, num_nodes=N)
            out    = model(node_x, mesh_ef, empty_wef, g)

            vel      = out[:, :3] * v_std + v_mean
            # Direct stress prediction: denormalize from normalized output
            stress_pred = (out[:, 3] * s_std_val + s_mean_val).cpu().numpy().astype(np.float32)

            prev_vel = vel
            cur      = cur + vel
            pred_wp_list.append(cur.cpu().numpy())
            pred_stress_list.append(stress_pred)

    # ── Write VTU + PVD ───────────────────────────────────────────────────────
    gt_pvd, pred_pvd = [], []

    for t_idx in range(T):
        vtu_name = f"t_{t_idx:03d}.vtu"

        # Ground truth
        gt_pos    = gt_wp_list[t_idx + 1].numpy().astype(np.float32)
        gt_disp   = gt_pos - ref_np
        gt_vm     = gt_stress_list[t_idx]
        gt_vm_computed = compute_vm_stress(
            torch.from_numpy(gt_pos).to(device), ref_pos, mesh_ei,
        ).cpu().numpy().astype(np.float32)

        write_vtu(
            os.path.join(gt_dir, vtu_name),
            gt_pos, connectivity, offsets, types,
            {
                "displacement":       gt_disp,
                "disp_mag":           np.linalg.norm(gt_disp, axis=1),
                "von_mises":          gt_vm,
                "von_mises_computed": gt_vm_computed,
            },
        )
        gt_pvd.append((t_idx, vtu_name))

        # Prediction
        pred_pos  = pred_wp_list[t_idx + 1].astype(np.float32)
        pred_disp = pred_pos - ref_np
        # Direct model stress (denormalized)
        pred_vm_direct = pred_stress_list[t_idx].astype(np.float32)
        # Analytical stress from predicted positions
        pred_vm_computed = compute_vm_stress(
            torch.from_numpy(pred_pos).to(device), ref_pos, mesh_ei,
        ).cpu().numpy().astype(np.float32)
        disp_err    = np.linalg.norm(pred_pos - gt_pos, axis=1).astype(np.float32)
        stress_err  = np.abs(pred_vm_direct - gt_vm).astype(np.float32)

        write_vtu(
            os.path.join(pred_dir, vtu_name),
            pred_pos, connectivity, offsets, types,
            {
                "displacement":       pred_disp,
                "disp_mag":           np.linalg.norm(pred_disp, axis=1),
                "von_mises_direct":   pred_vm_direct,
                "von_mises_computed": pred_vm_computed,
                "disp_error":         disp_err,
                "stress_error":       stress_err,
            },
        )
        pred_pvd.append((t_idx, vtu_name))

    write_pvd(os.path.join(gt_dir,   f"{sample_id}_gt.pvd"),   gt_pvd)
    write_pvd(os.path.join(pred_dir, f"{sample_id}_pred.pvd"), pred_pvd)

    final_err = np.linalg.norm(
        pred_wp_list[-1] - gt_wp_list[-1].numpy(), axis=1
    ).mean()
    final_stress_err = np.abs(pred_stress_list[-1] - gt_stress_list[-1]).mean()
    print(f"  final mean disp   error: {final_err:.6f} m")
    print(f"  final mean stress error: {final_stress_err:.3e} Pa")
    print(f"  GT   PVD → {gt_dir}/{sample_id}_gt.pvd")
    print(f"  Pred PVD → {pred_dir}/{sample_id}_pred.pvd")

print("\n完成！用 ParaView 開各 sample 下 gt/ 和 pred/ 裡的 .pvd 檔。")
