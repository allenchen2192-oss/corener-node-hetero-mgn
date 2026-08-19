"""
export_gfdm_stress_validation.py
=================================
Export GT Abaqus stress vs GFDM-computed stress to VTU/PVD for ParaView.

Goal: Validate that GFDM correctly approximates stress from displacement field.
      No model rollout — uses ground-truth positions from Abaqus at each step.

Output structure:
  export_gfdm_stress_validation/
    sample_XXXXX/
      t_000.vtu  ...  sample_XXXXX.pvd

Fields exported per timestep:
  stress_abaqus  : Von Mises stress from Abaqus (MPa)
  stress_gfdm    : Von Mises stress via GFDM    (MPa)
  stress_error   : |stress_gfdm - stress_abaqus| (MPa)
  displacement   : displacement magnitude (mm)

Usage:
  python export_gfdm_stress_validation.py
"""

import os
import numpy as np
import torch
from scipy.spatial import Delaunay

# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR    = "./02_abaqus_npz"
OUTPUT_DIR = "./export_gfdm_stress_validation"

SAMPLE_INDICES = [0, 1, 2, 3, 4]   # first 5 training cases

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


# ── GFDM stress ───────────────────────────────────────────────────────────────

def compute_gfdm_stress(world_pos_np, mesh_pos_np, edge_index_np):
    """Von Mises stress (Pa) via GFDM inverse-distance-squared weighting."""
    wp  = torch.from_numpy(world_pos_np).float()
    ref = torch.from_numpy(mesh_pos_np).float()
    ei  = torch.from_numpy(edge_index_np).long()
    row, col = ei[0], ei[1]

    r_ij  = ref[col] - ref[row]
    u     = wp - ref
    du_ij = u[col] - u[row]
    w_ij  = 1.0 / (r_ij.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12))

    r_r  = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)

    N = wp.shape[0]
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    XtX = torch.zeros(N, 3, 3)
    XtY = torch.zeros(N, 3, 3)
    XtX.scatter_add_(0, row_exp, r_r)
    XtY.scatter_add_(0, row_exp, r_du)
    XtX += 1e-6 * torch.eye(3).unsqueeze(0)

    dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy  = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz  = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    vm  = torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                             + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)
    return vm.numpy()


# ── VTU / PVD helpers (same as export_abaqus_paraview_nodetype.py) ────────────

_FACE_LOCAL_IDX = {
    8:  [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
    10: [[0,1,2],[0,3,1],[1,3,2],[0,2,3]],
    20: [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
}
_FACE_VTK_TYPE = {8: 9, 10: 5, 20: 9}
_FACE_CORNER_N = {8: 8, 10: 4, 20: 8}


def build_surface_cells_from_elems(points_np, elem_conn_np, n_nodes_per_elem):
    face_defs = _FACE_LOCAL_IDX.get(n_nodes_per_elem)
    vtk_type  = _FACE_VTK_TYPE.get(n_nodes_per_elem)
    corner_n  = _FACE_CORNER_N.get(n_nodes_per_elem)

    if face_defs is None:
        tri = Delaunay(points_np[:, :2])
        conn = tri.simplices.ravel()
        offs = np.arange(3, 3 * len(tri.simplices) + 1, 3, dtype=np.int32)
        types = np.full(len(tri.simplices), 5, dtype=np.uint8)
        return conn.astype(np.int32), offs, types

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
    conn_list, off_list, type_list = [], [], []
    cur_off = 0
    for face_nodes in surface_faces:
        center = points_np[face_nodes].mean(axis=0)
        if len(face_nodes) == 3:
            v1 = points_np[face_nodes[1]] - points_np[face_nodes[0]]
            v2 = points_np[face_nodes[2]] - points_np[face_nodes[0]]
            normal = np.cross(v1, v2)
        else:
            v1 = points_np[face_nodes[2]] - points_np[face_nodes[0]]
            v2 = points_np[face_nodes[3]] - points_np[face_nodes[1]]
            normal = np.cross(v1, v2)
        if np.dot(normal, center - mesh_center) < 0:
            face_nodes = face_nodes[::-1].copy()
        conn_list.extend(face_nodes.tolist())
        cur_off += len(face_nodes)
        off_list.append(cur_off)
        type_list.append(vtk_type)

    print(f"  npe={n_nodes_per_elem}, {len(elem_conn_np)} elems → {len(surface_faces)} surface faces")
    return (np.array(conn_list, dtype=np.int32),
            np.array(off_list,  dtype=np.int32),
            np.array(type_list, dtype=np.uint8))


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


# ── Main ──────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

for sid in SAMPLE_INDICES:
    npz_path = os.path.join(NPZ_DIR, f"sample_{sid:05d}.npz")
    if not os.path.exists(npz_path):
        print(f"[skip] {npz_path} not found")
        continue

    print(f"\n=== sample_{sid:05d} ===")
    d = np.load(npz_path)
    mesh_pos        = d["mesh_pos"]         # (N, 3)
    world_pos       = d["world_pos"]        # (F, N, 3)
    stress_vm       = d["stress_vm"]        # (F, N)  in Pa
    mesh_edge_index = d["mesh_edge_index"]  # (2, E)
    elem_conn        = d["elem_conn"] if "elem_conn" in d else None
    n_nodes_per_elem = int(d["n_nodes_per_elem"]) if "n_nodes_per_elem" in d else 0

    N, F = mesh_pos.shape[0], world_pos.shape[0]
    print(f"  N={N} nodes, F={F} frames, npe={n_nodes_per_elem}")

    # Build surface mesh (constant topology)
    if elem_conn is not None and n_nodes_per_elem in _FACE_LOCAL_IDX:
        conn, offs, types = build_surface_cells_from_elems(
            mesh_pos, elem_conn, n_nodes_per_elem)
    else:
        tri = Delaunay(mesh_pos[:, :2])
        conn  = tri.simplices.ravel().astype(np.int32)
        offs  = np.arange(3, 3 * len(tri.simplices) + 1, 3, dtype=np.int32)
        types = np.full(len(tri.simplices), 5, dtype=np.uint8)

    sample_dir = os.path.join(OUTPUT_DIR, f"sample_{sid:05d}")
    os.makedirs(sample_dir, exist_ok=True)

    vtu_entries = []

    for t in range(F):
        wp_t        = world_pos[t]           # (N, 3)
        stress_gt_t = stress_vm[t]           # (N,) in Pa

        # GFDM stress at GT positions
        stress_gfdm_t = compute_gfdm_stress(wp_t, mesh_pos, mesh_edge_index)

        # Convert to MPa
        s_gt   = stress_gt_t   / 1e6
        s_gfdm = stress_gfdm_t / 1e6
        s_err  = np.abs(s_gfdm - s_gt)

        # Displacement magnitude in mm
        disp_mag = np.linalg.norm(wp_t - mesh_pos, axis=1) * 1e3

        point_data = {
            "stress_abaqus_MPa": s_gt,
            "stress_gfdm_MPa":   s_gfdm,
            "stress_error_MPa":  s_err,
            "displacement_mm":   disp_mag,
        }

        vtu_name = f"t_{t:03d}.vtu"
        vtu_path = os.path.join(sample_dir, vtu_name)
        write_vtu(vtu_path, wp_t.astype(np.float32), conn, offs, types, point_data)
        vtu_entries.append((float(t), vtu_name))

        if t % 5 == 0:
            # Print quick error stats
            rel = s_err.mean() / (s_gt.mean() + 1e-10) * 100
            print(f"  t={t:2d}  stress_gt={s_gt.mean():.2f} MPa  "
                  f"stress_gfdm={s_gfdm.mean():.2f} MPa  err={rel:.2f}%")

    pvd_path = os.path.join(sample_dir, f"sample_{sid:05d}.pvd")
    write_pvd(pvd_path, vtu_entries)
    print(f"  Saved: {pvd_path}")

print("\nDone. Open sample_XXXXX/sample_XXXXX.pvd in ParaView.")
