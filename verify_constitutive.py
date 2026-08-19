"""
verify_constitutive.py
======================
Verify that the B-matrix constitutive stress matches Abaqus GT nodal stress
for elastic (Si / UF) nodes.

For NLGEOM=NO with temperature-independent elastic properties and uniform
temperature field (all confirmed from S0001_M0035.inp), the formula:

    σ_const_node_i = Σ_{e ∋ i, elem_mat==node_mat, elastic} V_e · C:(B_e·u - α·ΔT·m)
                     ─────────────────────────────────────────────────────────────────
                                  Σ V_e

must equal the GT nodal stress exactly (to float32 precision).

Usage:
    python verify_constitutive.py [workdir]
"""

import os
import sys
import numpy as np

if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

NPZ_DIR = "./02_abaqus_npz_m0035"
SAMPLES = ["S0001", "S0002", "S0003"]
FRAMES  = [1, 5, 10, 15, 20]

XI   = np.array([-1, 1, 1,-1,-1, 1, 1,-1], dtype=np.float64)
ETA  = np.array([-1,-1, 1, 1,-1,-1, 1, 1], dtype=np.float64)
ZETA = np.array([-1,-1,-1,-1, 1, 1, 1, 1], dtype=np.float64)
PARAM = np.stack([XI, ETA, ZETA], axis=0)   # (3, 8)
THERMAL_M = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)


def elastic_stress(eps_mech, E, nu):
    """Voigt stress from mechanical strain for isotropic linear elastic material."""
    lam  = E * nu / ((1 + nu) * (1 - 2*nu))
    mu   = E / (2 * (1 + nu))
    ev   = eps_mech[0] + eps_mech[1] + eps_mech[2]   # volumetric strain
    s    = np.empty(6, dtype=np.float64)
    s[0] = lam*ev + 2*mu*eps_mech[0]
    s[1] = lam*ev + 2*mu*eps_mech[1]
    s[2] = lam*ev + 2*mu*eps_mech[2]
    s[3] = mu * eps_mech[3]
    s[4] = mu * eps_mech[4]
    s[5] = mu * eps_mech[5]
    return s


def compute_const_stress_nodes(mesh_pos, disp, delta_T,
                                elem_conn, elem_mat,
                                node_mat, node_E, node_nu, node_CTE):
    """
    Node-level constitutive stress for elastic (Si/UF) elements.
    Uses material-separated, volume-weighted averaging — identical to
    preprocess_m0035.py / extract_odb_m0035.py.

    Returns:
        sigma_const : (N, 6) float64  — zero for non-elastic nodes
        has_weight  : (N,) bool       — True where constitutive stress is valid
    """
    N = mesh_pos.shape[0]
    sigma_sum  = np.zeros((N, 6), dtype=np.float64)
    weight_sum = np.zeros(N,      dtype=np.float64)

    for e, mat in enumerate(elem_mat):
        if mat == 2:          # Solder: elastoplastic, skip
            continue

        conn = elem_conn[e]
        xyz  = mesh_pos[conn].astype(np.float64)    # (8, 3) reference coords

        # Jacobian at center IP (ξ=η=ζ=0), volume
        J   = (PARAM / 8.0) @ xyz                   # (3, 3)
        V_e = 8.0 * abs(np.linalg.det(J))

        # Physical shape-function gradients: dN (3, 8)
        # dN/dX = J^{-1} @ dN/dxi  (NOT J^{-T})
        dN = np.linalg.solve(J, PARAM / 8.0)        # (3, 8)

        # Displacement at 8 nodes (8, 3)
        u_e = disp[conn]                             # (8, 3)

        # Strain (engineering Voigt: γ not ε for shear)
        eps = np.array([
            dN[0] @ u_e[:, 0],
            dN[1] @ u_e[:, 1],
            dN[2] @ u_e[:, 2],
            dN[1] @ u_e[:, 0] + dN[0] @ u_e[:, 1],   # γ12
            dN[2] @ u_e[:, 0] + dN[0] @ u_e[:, 2],   # γ13
            dN[2] @ u_e[:, 1] + dN[1] @ u_e[:, 2],   # γ23
        ])

        # Thermal strain: all 8 nodes of same element share same CTE (single-mat elem)
        alpha_e = float(node_CTE[conn[0]])
        eps_th  = alpha_e * delta_T * THERMAL_M
        eps_mech = eps - eps_th

        # Constitutive stress via Hooke's law
        E_e  = float(node_E[conn[0]])
        nu_e = float(node_nu[conn[0]])
        sig_e = elastic_stress(eps_mech, E_e, nu_e)

        # Scatter to nodes: material-separated (only same-mat nodes receive contribution)
        for i in conn:
            if node_mat[i] == mat:
                sigma_sum[i]  += V_e * sig_e
                weight_sum[i] += V_e

    has_weight = weight_sum > 0
    sigma_const = np.zeros((N, 6), dtype=np.float64)
    sigma_const[has_weight] = sigma_sum[has_weight] / weight_sum[has_weight, None]
    return sigma_const, has_weight


def analyze(sid):
    path = os.path.join(NPZ_DIR, f"{sid}.npz")
    d = np.load(path, allow_pickle=True)

    mesh_pos    = d["mesh_pos"].astype(np.float64)
    world_pos   = d["world_pos"]
    stress_gt   = d["stress"]                          # (T, N, 6) float32
    elem_conn   = d["elem_conn"].astype(np.int32)
    elem_mat    = d["elem_mat"].astype(np.int32)
    node_mat    = d["node_mat"].astype(np.int32)
    node_E      = d["node_E"].astype(np.float64)
    node_nu     = d["node_nu"].astype(np.float64)
    node_CTE    = d["node_CTE"].astype(np.float64)
    temperature = d["temperature"]                     # (T, N)

    N = mesh_pos.shape[0]
    print(f"\n{'='*60}")
    print(f"Sample {sid}  N={N}  M={elem_conn.shape[0]}")

    mat_names = {0:"Si_bot", 1:"Si_SoC", 2:"Solder", 3:"UF"}

    for fi in FRAMES:
        disp_t    = (world_pos[fi] - mesh_pos).astype(np.float64)
        delta_T   = float(temperature[fi, 0]) - 250.0
        s_gt      = stress_gt[fi].astype(np.float64)   # (N, 6)

        sigma_c, has_w = compute_const_stress_nodes(
            mesh_pos, disp_t, delta_T, elem_conn, elem_mat,
            node_mat, node_E, node_nu, node_CTE
        )

        # Absolute and relative errors at elastic nodes
        err     = np.abs(sigma_c[has_w] - s_gt[has_w])        # (K, 6)
        gt_mag  = np.abs(s_gt[has_w])                          # (K, 6)
        rel_err = err / (gt_mag + 1e-10)

        component_names = ["S11","S22","S33","S12","S13","S23"]

        print(f"\n  Frame {fi:2d}  ΔT={delta_T:.1f}°C   elastic nodes={has_w.sum()}")
        print(f"  {'Comp':4s}  {'abs_mean':>10s}  {'abs_max':>10s}  "
              f"{'rel_mean':>10s}  {'rel_max':>10s}")
        for c, name in enumerate(component_names):
            print(f"  {name:4s}  {err[:,c].mean():10.3e}  {err[:,c].max():10.3e}  "
                  f"{rel_err[:,c].mean():10.3e}  {rel_err[:,c].max():10.3e}")

        # Per-material breakdown (for nodes with elastic weight)
        print(f"  -- per material (all components combined) --")
        for mat_id in [0, 1, 3]:
            mat_mask = (node_mat[has_w] == mat_id)
            if not mat_mask.any():
                continue
            e_m   = err[mat_mask]
            gt_m  = gt_mag[mat_mask]
            rel_m = e_m / (gt_m + 1e-10)
            print(f"  [{mat_names[mat_id]:7s}]  "
                  f"abs_mean={e_m.mean():.3e}  abs_max={e_m.max():.3e}  "
                  f"rel_mean={rel_m.mean():.3e}  rel_max={rel_m.max():.3e}")


if __name__ == "__main__":
    for sid in SAMPLES:
        analyze(sid)
    print("\nDone.")