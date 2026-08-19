"""
export_gfdm_compare_m0035.py
=============================
Compare GFDM stress (computed from GT displacement) vs Abaqus FEM stress.
Uses per-material volume hex cells so each material can be inspected separately.

Output:
  export_gfdm_compare_m0035/
    S0001/
      si/      t_000.vtu ... S0001_si.pvd
      uf/      t_000.vtu ... S0001_uf.pvd
      solder/  t_000.vtu ... S0001_solder.pvd  ← GFDM invalid here (elastoplastic)

PointData fields (all outputs):
  vm_fem      (MPa): Abaqus von Mises
  vm_gfdm     (MPa): GFDM von Mises from GT displacement
  vm_error    (MPa): |vm_gfdm - vm_fem|
  s33_fem     (MPa): Abaqus S33
  s33_gfdm    (MPa): GFDM S33
  s33_error   (MPa): |s33_gfdm - s33_fem|

GFDM is physically valid for Si and UF (linear elastic).
Solder error will be large (plasticity ignored) — shown for reference.

Usage:
  python export_gfdm_compare_m0035.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from export_abaqus_paraview import build_surface_cells_from_elems, write_vtu, write_pvd
from export_gt_m0035 import build_hex_cells, von_mises


# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR    = "./02_abaqus_npz_m0035"
OUTPUT_DIR = "./export_gfdm_compare_m0035"

SAMPLE_IDS = ["S0001", "S0201"]   # one train + one test


# ── GFDM stress (pure numpy) ──────────────────────────────────────────────────

def gfdm_stress_np(mesh_pos, world_pos_t, mesh_ei, node_E, node_nu, node_CTE, delta_T):
    """
    Full 6-component stress tensor via GFDM with isotropic thermal correction.
    σ = C:(ε_total − α·ΔT·I),  ε_total = sym(∇u) from weighted least squares.

    Returns (N, 6): [S11, S22, S33, S12, S13, S23] in MPa.
    """
    N     = mesh_pos.shape[0]
    row   = mesh_ei[0].astype(np.int64)
    col   = mesh_ei[1].astype(np.int64)

    r_ij  = (mesh_pos[col] - mesh_pos[row]).astype(np.float64)       # (E, 3)
    u     = (world_pos_t   - mesh_pos    ).astype(np.float64)        # (N, 3)
    du_ij = u[col] - u[row]                                          # (E, 3)

    w_ij  = 1.0 / (np.sum(r_ij**2, axis=1, keepdims=True) + 1e-12)  # (E, 1)
    wr_ij = w_ij * r_ij                                              # (E, 3)

    r_r   = wr_ij[:, :, None] * r_ij[:, None, :]    # (E, 3, 3)
    r_du  = wr_ij[:, :, None] * du_ij[:, None, :]   # (E, 3, 3)

    XtX   = np.zeros((N, 3, 3), dtype=np.float64)
    XtY   = np.zeros((N, 3, 3), dtype=np.float64)
    np.add.at(XtX, row, r_r)
    np.add.at(XtY, row, r_du)
    XtX  += 1e-6 * np.eye(3)[None]

    # dU[i, j, k] = ∂u_j/∂x_k  (displacement gradient)
    dU    = np.linalg.solve(XtX, XtY).transpose(0, 2, 1)  # (N, 3, 3)

    E_    = node_E.astype(np.float64)
    nu_   = node_nu.astype(np.float64)
    alp   = node_CTE.astype(np.float64)
    dT    = float(delta_T)

    lam   = E_ * nu_ / ((1 + nu_) * (1 - 2 * nu_))
    mu_   = E_ / (2 * (1 + nu_))

    # Mechanical strains = total strains − isotropic thermal strains
    eps_th  = alp * dT                        # (N,)
    eps11   = dU[:, 0, 0] - eps_th
    eps22   = dU[:, 1, 1] - eps_th
    eps33   = dU[:, 2, 2] - eps_th
    tr_mech = eps11 + eps22 + eps33

    s11 = lam * tr_mech + 2 * mu_ * eps11
    s22 = lam * tr_mech + 2 * mu_ * eps22
    s33 = lam * tr_mech + 2 * mu_ * eps33
    s12 = mu_ * (dU[:, 0, 1] + dU[:, 1, 0])  # shear: no thermal correction
    s13 = mu_ * (dU[:, 0, 2] + dU[:, 2, 0])
    s23 = mu_ * (dU[:, 1, 2] + dU[:, 2, 1])

    return np.stack([s11, s22, s33, s12, s13, s23], axis=1).astype(np.float32)


# ── Main export ───────────────────────────────────────────────────────────────

def export_sample(sid):
    npz_path = os.path.join(NPZ_DIR, f"{sid}.npz")
    if not os.path.exists(npz_path):
        print(f"  MISSING: {npz_path}")
        return

    d = np.load(npz_path, allow_pickle=True)

    mesh_pos    = d["mesh_pos"].astype(np.float32)      # (N, 3)
    world_pos   = d["world_pos"].astype(np.float32)     # (21, N, 3)
    stress_fem  = d["stress"].astype(np.float32)        # (21, N, 6)
    temperature = d["temperature"].astype(np.float32)   # (21, N)
    node_mat    = d["node_mat"].astype(np.int32)        # (N,)
    node_E      = d["node_E"].astype(np.float32)        # (N,)  MPa
    node_nu     = d["node_nu"].astype(np.float32)       # (N,)
    node_CTE    = d["node_CTE"].astype(np.float32)      # (N,)
    elem_conn   = d["elem_conn"].astype(np.int32)       # (n_elem, 8)
    elem_mat    = d["elem_mat"].astype(np.int32)        # (n_elem,)
    mesh_ei     = d["mesh_edge_index"].astype(np.int64) # (2, E)

    T    = world_pos.shape[0]   # 21 frames
    T_ref = 250.0               # reference temperature (°C)

    base_dir  = os.path.join(OUTPUT_DIR, sid)
    time_vals = list(range(T))

    # Pre-build per-material hex cell topology (constant across frames)
    mat_groups = {
        "si":     (elem_mat == 0) | (elem_mat == 1),
        "uf":     (elem_mat == 3),
        "solder": (elem_mat == 2),
    }

    print(f"\n  {sid}: computing GFDM stress for {T} frames ...")

    # Collect all frames for each material output
    per_mat_frames = {name: [] for name in mat_groups}

    for t in range(T):
        pts_t      = world_pos[t]                   # (N, 3) deformed positions
        delta_T_t  = float(temperature[t, 0]) - T_ref  # scalar ΔT at frame t

        # GFDM stress from GT displacement
        stress_g = gfdm_stress_np(
            mesh_pos, pts_t, mesh_ei, node_E, node_nu, node_CTE, delta_T_t
        )  # (N, 6)

        sf = stress_fem[t]   # (N, 6)

        vm_fem  = von_mises(sf)
        vm_gfdm = von_mises(stress_g)

        pdata = {
            "vm_fem":    vm_fem.astype(np.float32),
            "vm_gfdm":   vm_gfdm.astype(np.float32),
            "vm_error":  np.abs(vm_gfdm - vm_fem).astype(np.float32),
            "s33_fem":   sf[:, 2],
            "s33_gfdm":  stress_g[:, 2],
            "s33_error": np.abs(stress_g[:, 2] - sf[:, 2]).astype(np.float32),
        }

        for name in mat_groups:
            per_mat_frames[name].append((pts_t, pdata))

        # Print relative error at key frames (t=5, 10, 20)
        if t in (5, 10, 20):
            si_mask = (node_mat == 0) | (node_mat == 1)
            uf_mask = node_mat == 3
            sol_mask = node_mat == 2
            for label, mask in [("Si", si_mask), ("UF", uf_mask), ("Solder", sol_mask)]:
                fem_rms  = np.sqrt(np.mean(vm_fem[mask]**2))
                err_rms  = np.sqrt(np.mean((vm_gfdm[mask] - vm_fem[mask])**2))
                rel      = err_rms / (fem_rms + 1e-8) * 100
                print(f"    t={t:2d}  {label:6s}: RMSE={err_rms:.2f} MPa  "
                      f"FEM_RMS={fem_rms:.2f} MPa  rel={rel:.1f}%")

    # ── Surface output (full assembly, for spatial overview) ─────────────────
    surf_conn, surf_offs, surf_types, _ = build_surface_cells_from_elems(
        mesh_pos, elem_conn, elem_conn.shape[1], debug_label=f"{sid}/surface"
    )
    surf_dir = os.path.join(base_dir, "surface")
    os.makedirs(surf_dir, exist_ok=True)
    surf_vtu_names = []
    for t, (pts_t, pdata) in enumerate(per_mat_frames["si"]):  # pdata is same for all mats
        # Add material_id for region identification on surface
        pdata_surf = {**pdata, "material_id": node_mat.astype(np.float32)}
        fname = f"t_{t:03d}.vtu"
        write_vtu(os.path.join(surf_dir, fname),
                  pts_t, surf_conn, surf_offs, surf_types, pdata_surf)
        surf_vtu_names.append(fname)
    write_pvd(os.path.join(surf_dir, f"{sid}_surface.pvd"),
              list(zip(time_vals, surf_vtu_names)))
    print(f"    [surface] → {surf_dir}/")

    # ── Per-material volume hex outputs ───────────────────────────────────────
    for name, mask in mat_groups.items():
        elems = elem_conn[mask]
        if len(elems) == 0:
            continue
        conn, offsets, types = build_hex_cells(elems)
        out_dir = os.path.join(base_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        vtu_names = []
        for t, (pts_t, pdata) in enumerate(per_mat_frames[name]):
            fname = f"t_{t:03d}.vtu"
            write_vtu(os.path.join(out_dir, fname), pts_t, conn, offsets, types, pdata)
            vtu_names.append(fname)

        pvd_entries = list(zip(time_vals, vtu_names))
        write_pvd(os.path.join(out_dir, f"{sid}_{name}.pvd"), pvd_entries)
        print(f"    [{name}] {len(elems)} elements → {out_dir}/")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sid in SAMPLE_IDS:
        print(f"Exporting {sid} ...")
        export_sample(sid)
    print("\nDone.")


if __name__ == "__main__":
    main()