"""
export_gt_m0035.py
==================
Export M0035 ground-truth (Abaqus) results to VTU/PVD for ParaView inspection.

Output structure:
  export_gt_m0035/
    S0001/
      surface/  t_000.vtu ... S0001_surface.pvd   ← full assembly surface (overview)
      solder/   t_000.vtu ... S0001_solder.pvd    ← only solder hex cells (volume)
      si/       t_000.vtu ... S0001_si.pvd        ← only Si hex cells (volume)
      uf/       t_000.vtu ... S0001_uf.pvd        ← only UF hex cells (volume)

PointData fields (all outputs):
  displacement  (vector 3D, mm)
  disp_mag      (scalar, mm)
  von_mises     (scalar, MPa)
  stress_xx/yy/zz/xy/xz/yz  (scalar, MPa)
  PEEQ          (scalar)
  temperature   (scalar, °C)

Usage:
  python export_gt_m0035.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from export_abaqus_paraview import (
    build_surface_cells_from_elems,
    write_vtu,
    write_pvd,
)


# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR    = "./02_abaqus_npz_m0035"
OUTPUT_DIR = "./export_gt_m0035"

SAMPLE_IDS = [
    "S0001", "S0002", "S0003",    # train
    "S0201", "S0202", "S0203",    # test
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def von_mises(stress):
    """stress: (N, 6) [S11,S22,S33,S12,S13,S23] → (N,) von Mises."""
    s11, s22, s33 = stress[:, 0], stress[:, 1], stress[:, 2]
    s12, s13, s23 = stress[:, 3], stress[:, 4], stress[:, 5]
    return np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
                          + 6 * (s12**2 + s13**2 + s23**2)))


def build_hex_cells(elem_conn_subset):
    """Build VTK_HEXAHEDRON (type 12) cells from a subset of element connectivity."""
    n = len(elem_conn_subset)
    conn    = elem_conn_subset.ravel().astype(np.int32)       # flat 8*n node indices
    offsets = np.arange(8, (n + 1) * 8, 8, dtype=np.int32)  # [8, 16, ..., 8n]
    types   = np.full(n, 12, dtype=np.uint8)                 # VTK_HEXAHEDRON = 12
    return conn, offsets, types


def point_data_for_frame(world_pos_t, mesh_pos, stress_t, peeq_t, temperature_t):
    disp     = (world_pos_t - mesh_pos).astype(np.float32)
    disp_mag = np.linalg.norm(disp, axis=1)
    vm       = von_mises(stress_t)
    return {
        "displacement": disp,
        "disp_mag":     disp_mag,
        "von_mises":    vm,
        "stress_xx":    stress_t[:, 0],
        "stress_yy":    stress_t[:, 1],
        "stress_zz":    stress_t[:, 2],
        "stress_xy":    stress_t[:, 3],
        "stress_xz":    stress_t[:, 4],
        "stress_yz":    stress_t[:, 5],
        "PEEQ":         peeq_t,
        "temperature":  temperature_t,
    }


def export_frames(out_dir, pvd_name, points_seq, conn, offsets, types, pdata_seq, time_vals):
    """Write one VTU per frame + PVD index."""
    os.makedirs(out_dir, exist_ok=True)
    vtu_names = []
    for t, (pts, pdata) in enumerate(zip(points_seq, pdata_seq)):
        fname = f"t_{t:03d}.vtu"
        write_vtu(os.path.join(out_dir, fname), pts, conn, offsets, types, pdata)
        vtu_names.append(fname)
    pvd_entries = list(zip(time_vals, vtu_names))
    write_pvd(os.path.join(out_dir, pvd_name), pvd_entries)


# ── Main export ───────────────────────────────────────────────────────────────

def export_sample(sid):
    npz_path = os.path.join(NPZ_DIR, f"{sid}.npz")
    if not os.path.exists(npz_path):
        print(f"  MISSING: {npz_path}")
        return

    d = np.load(npz_path, allow_pickle=True)

    mesh_pos    = d["mesh_pos"].astype(np.float32)      # (N, 3)
    world_pos   = d["world_pos"].astype(np.float32)     # (21, N, 3)
    stress      = d["stress"].astype(np.float32)        # (21, N, 6)
    peeq        = d["peeq"].astype(np.float32)          # (21, N)
    temperature = d["temperature"].astype(np.float32)   # (21, N)
    node_mat    = d["node_mat"].astype(np.int32)        # (N,)
    elem_conn   = d["elem_conn"].astype(np.int32)       # (n_elem, 8)
    elem_mat    = d["elem_mat"].astype(np.int32)        # (n_elem,)

    T    = world_pos.shape[0]   # 21 frames
    npe  = elem_conn.shape[1]   # 8

    base_dir  = os.path.join(OUTPUT_DIR, sid)
    time_vals = list(range(T))   # 0, 1, ..., 20  (ParaView plays low→high)

    # Per-frame data (shared across all outputs)
    points_seq = [world_pos[t] for t in range(T)]
    pdata_seq  = [
        point_data_for_frame(world_pos[t], mesh_pos, stress[t], peeq[t], temperature[t])
        for t in range(T)
    ]

    # ── 1. Full assembly surface (overview) ───────────────────────────────────
    surf_conn, surf_offs, surf_types, _ = build_surface_cells_from_elems(
        mesh_pos, elem_conn, npe, debug_label=f"{sid}/surface"
    )
    # Add material_id only to surface output (for region verification)
    surf_pdata = [{**pd, "material_id": node_mat.astype(np.float32)} for pd in pdata_seq]
    export_frames(
        os.path.join(base_dir, "surface"), f"{sid}_surface.pvd",
        points_seq, surf_conn, surf_offs, surf_types, surf_pdata, time_vals,
    )

    # ── 2. Per-material volume hex exports ────────────────────────────────────
    mat_groups = {
        "solder": (elem_mat == 2),
        "si":     (elem_mat == 0) | (elem_mat == 1),
        "uf":     (elem_mat == 3),
    }
    for name, mask in mat_groups.items():
        elems = elem_conn[mask]
        if len(elems) == 0:
            print(f"    [{sid}/{name}] no elements, skipping")
            continue
        conn, offsets, types = build_hex_cells(elems)
        export_frames(
            os.path.join(base_dir, name), f"{sid}_{name}.pvd",
            points_seq, conn, offsets, types, pdata_seq, time_vals,
        )
        print(f"    [{sid}/{name}] {len(elems)} elements")

    print(f"  {sid}: done → {base_dir}/")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sid in SAMPLE_IDS:
        print(f"Exporting {sid} ...")
        export_sample(sid)
    print("Done.")


if __name__ == "__main__":
    main()