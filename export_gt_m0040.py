"""
export_gt_m0040.py
==================
Export M0040 ground truth NPZ -> VTU/PVD for ParaView.

Output per sample:
  export_gt_m0040/S####/
    t_000.vtu ... t_020.vtu   S####.pvd          full mesh
    Si/     t_000.vtu ...     S####_Si.pvd        Si elements only
    Solder/ t_000.vtu ...     S####_Solder.pvd    Solder elements only
    UF/     t_000.vtu ...     S####_UF.pvd        UF elements only

Full mesh PointData : displacement(3D), disp_mag, velocity(3D), vel_mag,
                      node_type, vm_stress_node, peeq_node
Full mesh CellData  : vm_stress [MPa], peeq, material_id

Per-material submesh:
  PointData : displacement, disp_mag, vm_stress_node, peeq_node
  CellData  : vm_stress, peeq

Usage:
  python export_gt_m0040.py                      # default samples
  python export_gt_m0040.py S0001 S0010 S0205    # specific samples
"""

import os
import sys

import numpy as np

NPZ_DIR = "./02_abaqus_npz_m0040"
OUT_DIR = "./export_gt_m0040"

DEFAULT_SAMPLES = (
    [f"S{i:04d}" for i in range(1, 4)] +
    [f"S{i:04d}" for i in range(201, 204)]
)
SAMPLES = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SAMPLES

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
    lines.append("</PointData><CellData>")
    for name, arr in (cell_data or {}).items():
        arr = np.asarray(arr, dtype=np.float32)
        lines.extend(da(name, arr, arr.shape[1] if arr.ndim == 2 else 1))
    lines += ["</CellData><Points>",
              '<DataArray type="Float32" NumberOfComponents="3" format="ascii">',
              _fmt(points.astype(np.float32)), "</DataArray></Points><Cells>",
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

def elem_to_node(elem_val, elem_conn, N):
    """Average element scalars onto nodes."""
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
    sub_conn_raw = elem_conn[elem_mask]                         # (M_sub, 8)
    unique_nodes = np.unique(sub_conn_raw)                      # sorted original IDs
    old2new      = np.full(mesh_pos.shape[0], -1, dtype=np.int64)
    old2new[unique_nodes] = np.arange(len(unique_nodes), dtype=np.int64)
    sub_conn     = old2new[sub_conn_raw]                        # re-indexed
    return unique_nodes, sub_conn, elem_mask


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
    node_type      = d["node_type"]          # (N,)

    T = world_pos.shape[0]   # 21
    N = mesh_pos.shape[0]
    M = elem_conn.shape[0]

    out_dir = os.path.join(OUT_DIR, sid)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{sid}  N={N}  M={M}  T={T}")

    # ── Pre-compute submesh per material (constant across time) ───────────────
    submeshes = {}
    for mat_name, mat_ids in MATERIALS.items():
        sub_dir = os.path.join(out_dir, mat_name)
        os.makedirs(sub_dir, exist_ok=True)
        uniq, sub_conn, emask = build_submesh(mesh_pos, elem_conn, elem_mat, mat_ids)
        submeshes[mat_name] = {
            "dir":       sub_dir,
            "uniq":      uniq,          # (N_sub,) node indices
            "sub_conn":  sub_conn,      # (M_sub, 8) re-indexed
            "emask":     emask,         # (M,) bool
            "N_sub":     len(uniq),
            "pvd":       [],
        }
        print(f"  {mat_name}: N_sub={len(uniq)}  M_sub={emask.sum()}")

    pvd_entries = []

    for fi in range(T):   # fi = 0 .. 20
        pos  = world_pos[fi].astype(np.float32)
        disp = (pos - mesh_pos).astype(np.float32)
        vel  = (velocity[fi - 1].astype(np.float32)
                if fi > 0 else np.zeros((N, 3), dtype=np.float32))

        vm   = elem_vm_stress[fi].astype(np.float32)
        peeq = elem_peeq[fi].astype(np.float32)

        vm_node   = elem_to_node(vm,   elem_conn, N)
        peeq_node = elem_to_node(peeq, elem_conn, N)

        # ── Full mesh VTU ─────────────────────────────────────────────────────
        vtu_name = f"t_{fi:03d}.vtu"
        write_vtu(
            os.path.join(out_dir, vtu_name), pos, elem_conn,
            point_data={
                "displacement":   disp,
                "disp_mag":       np.linalg.norm(disp, axis=1).astype(np.float32),
                "velocity":       vel,
                "vel_mag":        np.linalg.norm(vel, axis=1).astype(np.float32),
                "node_type":      node_type.astype(np.float32),
                "vm_stress_node": vm_node,
                "peeq_node":      peeq_node,
            },
            cell_data={
                "vm_stress":   vm,
                "peeq":        peeq,
                "material_id": elem_mat.astype(np.float32),
            },
        )
        pvd_entries.append((fi, vtu_name))

        # ── Per-material submesh VTU ──────────────────────────────────────────
        for mat_name, sm in submeshes.items():
            uniq     = sm["uniq"]
            sub_conn = sm["sub_conn"]
            emask    = sm["emask"]
            N_sub    = sm["N_sub"]

            sub_pos  = pos[uniq]
            sub_disp = disp[uniq]
            sub_vm   = vm[emask]
            sub_peeq = peeq[emask]

            write_vtu(
                os.path.join(sm["dir"], vtu_name), sub_pos, sub_conn,
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
            sm["pvd"].append((fi, vtu_name))

    # ── Write PVD files ───────────────────────────────────────────────────────
    write_pvd(os.path.join(out_dir, f"{sid}.pvd"), pvd_entries)
    for mat_name, sm in submeshes.items():
        write_pvd(os.path.join(sm["dir"], f"{sid}_{mat_name}.pvd"), sm["pvd"])

    print(f"  -> {T} frames | full + Si/Solder/UF submesh PVDs written")

print(f"\nDone. Open export_gt_m0040/<sid>/<sid>.pvd (or <mat>/<sid>_<mat>.pvd) in ParaView.")
