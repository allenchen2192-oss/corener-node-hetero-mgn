"""
verify_constitutive_v2.py
=========================
Debug version: adds single-element diagnostic + fixes material property lookup.

Key fix: use elem_mat directly for material properties instead of node_CTE[conn[0]],
which can be wrong at Si-UF interface elements (due to any-adjacency priority rule).

Usage:
    python verify_constitutive_v2.py [workdir]
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

# Hardcoded material properties (same as extract_odb_m0035.py)
# mat 0=Si_bottom, 1=Si_SoC, 2=Solder, 3=UF
SI_E   = 130000.0  # MPa
SI_NU  = 0.28
SI_CTE = 3.1e-6    # /°C


def elastic_stress(eps_mech, E, nu):
    """Voigt stress from mechanical strain for isotropic linear elastic material."""
    lam  = E * nu / ((1 + nu) * (1 - 2*nu))
    mu   = E / (2 * (1 + nu))
    ev   = eps_mech[0] + eps_mech[1] + eps_mech[2]
    s    = np.empty(6, dtype=np.float64)
    s[0] = lam*ev + 2*mu*eps_mech[0]
    s[1] = lam*ev + 2*mu*eps_mech[1]
    s[2] = lam*ev + 2*mu*eps_mech[2]
    s[3] = mu * eps_mech[3]
    s[4] = mu * eps_mech[4]
    s[5] = mu * eps_mech[5]
    return s


def get_elem_props(mat, conn, node_E, node_nu, node_CTE, node_mat):
    """
    Get material properties for element with given mat index.

    FIX: For Si elements (mat 0,1), use hardcoded Si props.
    For UF (mat 3), find a node of the correct material in the element
    (avoids using conn[0] which may be a different-material node).

    Returns (E, nu, alpha) for the element.
    """
    if mat == 0 or mat == 1:
        return SI_E, SI_NU, SI_CTE
    elif mat == 3:
        # UF: find a UF node in this element
        for i in conn:
            if node_mat[i] == 3:
                return float(node_E[i]), float(node_nu[i]), float(node_CTE[i])
        # Fallback: use conn[0] (shouldn't happen if mesh is valid)
        return float(node_E[conn[0]]), float(node_nu[conn[0]]), float(node_CTE[conn[0]])
    else:
        return float(node_E[conn[0]]), float(node_nu[conn[0]]), float(node_CTE[conn[0]])


def compute_const_stress_nodes(mesh_pos, disp, delta_T,
                                elem_conn, elem_mat,
                                node_mat, node_E, node_nu, node_CTE):
    N = mesh_pos.shape[0]
    sigma_sum  = np.zeros((N, 6), dtype=np.float64)
    weight_sum = np.zeros(N,      dtype=np.float64)

    for e, mat in enumerate(elem_mat):
        if mat == 2:  # Solder: elastoplastic
            continue

        conn = elem_conn[e]
        xyz  = mesh_pos[conn].astype(np.float64)

        J   = (PARAM / 8.0) @ xyz
        V_e = 8.0 * abs(np.linalg.det(J))
        dN  = np.linalg.solve(J, PARAM / 8.0)   # J^{-1} @ dNdxi (NOT J^T)

        u_e = disp[conn]

        eps = np.array([
            dN[0] @ u_e[:, 0],
            dN[1] @ u_e[:, 1],
            dN[2] @ u_e[:, 2],
            dN[1] @ u_e[:, 0] + dN[0] @ u_e[:, 1],
            dN[2] @ u_e[:, 0] + dN[0] @ u_e[:, 2],
            dN[2] @ u_e[:, 1] + dN[1] @ u_e[:, 2],
        ])

        # FIXED: use element material directly, not conn[0]
        E_e, nu_e, alpha_e = get_elem_props(mat, conn, node_E, node_nu, node_CTE, node_mat)

        eps_th   = alpha_e * delta_T * THERMAL_M
        eps_mech = eps - eps_th
        sig_e    = elastic_stress(eps_mech, E_e, nu_e)

        for i in conn:
            if node_mat[i] == mat:
                sigma_sum[i]  += V_e * sig_e
                weight_sum[i] += V_e

    has_weight = weight_sum > 0
    sigma_const = np.zeros((N, 6), dtype=np.float64)
    sigma_const[has_weight] = sigma_sum[has_weight] / weight_sum[has_weight, None]
    return sigma_const, has_weight


def diagnose_element(e_idx, mesh_pos, disp, delta_T,
                     elem_conn, elem_mat, node_mat, node_E, node_nu, node_CTE,
                     stress_gt):
    """Print step-by-step computation for one element at one frame."""
    mat  = int(elem_mat[e_idx])
    conn = elem_conn[e_idx]
    xyz  = mesh_pos[conn].astype(np.float64)

    J    = (PARAM / 8.0) @ xyz
    detJ = np.linalg.det(J)
    V_e  = 8.0 * abs(detJ)
    dN   = np.linalg.solve(J, PARAM / 8.0)   # J^{-1} @ dNdxi (NOT J^T)
    u_e  = disp[conn]

    # Print node-level coords and displacements for manual verification
    print(f"  Node coords (mesh_pos) and displacements for elem {e_idx}:")
    print(f"  {'Nidx':>6}  {'x':>10}  {'y':>10}  {'z':>10}  {'ux':>12}  {'uy':>12}  {'uz':>12}")
    for k, ni in enumerate(conn):
        x, y, z = xyz[k]
        ux, uy, uz = u_e[k]
        print(f"  {ni:6d}  {x:10.4f}  {y:10.4f}  {z:10.4f}  {ux:12.6e}  {uy:12.6e}  {uz:12.6e}")
    print(f"  Jacobian J:\n{J}")
    print(f"  det(J) = {detJ:.6e}   V_e = {V_e:.6e} mm^3")

    eps = np.array([
        dN[0] @ u_e[:, 0],
        dN[1] @ u_e[:, 1],
        dN[2] @ u_e[:, 2],
        dN[1] @ u_e[:, 0] + dN[0] @ u_e[:, 1],
        dN[2] @ u_e[:, 0] + dN[0] @ u_e[:, 2],
        dN[2] @ u_e[:, 1] + dN[1] @ u_e[:, 2],
    ])

    E_fix,  nu_fix,  a_fix  = get_elem_props(mat, conn, node_E, node_nu, node_CTE, node_mat)
    E_old = float(node_E[conn[0]])
    a_old = float(node_CTE[conn[0]])

    eps_th_fix = a_fix * delta_T * THERMAL_M
    eps_th_old = a_old * delta_T * THERMAL_M

    sig_fix = elastic_stress(eps - eps_th_fix, E_fix,  nu_fix)
    sig_old = elastic_stress(eps - eps_th_old, E_old, float(node_nu[conn[0]]))

    # GT stress at the contributing nodes (mat-matched)
    valid_nodes = [i for i in conn if node_mat[i] == mat]
    gt_vals = stress_gt[valid_nodes]   # (k, 6)

    labels = ["S11","S22","S33","S12","S13","S23"]

    print(f"\n  === Element {e_idx}  elem_mat={mat}  V_e={V_e:.4f} mm³ ===")
    print(f"  conn node_mat: {[int(node_mat[i]) for i in conn]}")
    print(f"  Valid (mat-matched) nodes: {valid_nodes} ({len(valid_nodes)}/8)")
    print(f"  conn[0] node_mat: {int(node_mat[conn[0]])}  (old code picks this)")
    print(f"  OLD  E={E_old:.0f} MPa  alpha={a_old:.3e}")
    print(f"  FIXED E={E_fix:.0f} MPa  alpha={a_fix:.3e}")
    print(f"  delta_T = {delta_T:.2f} °C")
    print(f"  eps_total: {eps}")
    print(f"  eps_th_fix: {eps_th_fix}")
    print(f"  eps_th_old: {eps_th_old}")
    print(f"  eps_mech_fix: {eps - eps_th_fix}")
    print(f"  eps_mech_old: {eps - eps_th_old}")
    print(f"  sigma_fix: {sig_fix}")
    print(f"  sigma_old: {sig_old}")
    print(f"  GT stress at valid nodes (mean): {gt_vals.mean(axis=0)}")
    for c, lbl in enumerate(labels):
        gt_c  = gt_vals[:, c].mean()
        err_f = abs(sig_fix[c] - gt_c)
        err_o = abs(sig_old[c] - gt_c)
        print(f"    {lbl}: fix={sig_fix[c]:+9.3f}  old={sig_old[c]:+9.3f}  GT={gt_c:+9.3f}  err_fix={err_f:.3f}  err_old={err_o:.3f}")


def analyze(sid):
    path = os.path.join(NPZ_DIR, f"{sid}.npz")
    d = np.load(path, allow_pickle=True)

    mesh_pos    = d["mesh_pos"].astype(np.float64)
    world_pos   = d["world_pos"]
    stress_gt   = d["stress"]
    elem_conn   = d["elem_conn"].astype(np.int32)
    elem_mat    = d["elem_mat"].astype(np.int32)
    node_mat    = d["node_mat"].astype(np.int32)
    node_E      = d["node_E"].astype(np.float64)
    node_nu     = d["node_nu"].astype(np.float64)
    node_CTE    = d["node_CTE"].astype(np.float64)
    temperature = d["temperature"]

    N = mesh_pos.shape[0]
    M = elem_conn.shape[0]
    print(f"\n{'='*60}")
    print(f"Sample {sid}  N={N}  M={M}")

    # Count how many interface elements have conn[0] with wrong material
    wrong_prop_count = {0:0, 1:0, 3:0}
    total_count = {0:0, 1:0, 3:0}
    for e, mat in enumerate(elem_mat):
        if mat == 2:
            continue
        conn = elem_conn[e]
        total_count[mat] = total_count.get(mat, 0) + 1
        if int(node_mat[conn[0]]) != mat:
            wrong_prop_count[mat] = wrong_prop_count.get(mat, 0) + 1

    print("\n  Elements with conn[0] pointing to WRONG material (old bug):")
    for mat in [0, 1, 3]:
        n_wrong = wrong_prop_count.get(mat, 0)
        n_total = total_count.get(mat, 0)
        print(f"    mat={mat}: {n_wrong}/{n_total} elements ({100*n_wrong/max(n_total,1):.1f}%)")

    # Find a pure-Si interior element for diagnosis (all 8 nodes == mat)
    diag_elem_si = -1
    for e, mat in enumerate(elem_mat):
        if mat == 1:  # Si_SoC
            conn = elem_conn[e]
            if all(int(node_mat[i]) == 1 for i in conn):
                diag_elem_si = e
                break

    # Find an interface Si element where conn[0] has wrong mat
    diag_elem_iface = -1
    for e, mat in enumerate(elem_mat):
        if mat == 1:
            conn = elem_conn[e]
            if int(node_mat[conn[0]]) != 1:
                diag_elem_iface = e
                break

    # Run frame 10 diagnosis
    fi = 10
    disp_t  = (world_pos[fi] - mesh_pos).astype(np.float64)
    delta_T = float(temperature[fi, 0]) - 250.0
    s_gt    = stress_gt[fi].astype(np.float64)

    # Global GT stress statistics at frame 10
    print(f"\n  Global GT stress stats at frame {fi} (ΔT={delta_T:.1f}°C):")
    mat_names2 = {0:"Si_bot", 1:"Si_SoC", 2:"Solder", 3:"UF"}
    for mid in range(4):
        mask = node_mat == mid
        if not mask.any(): continue
        sv = s_gt[mask]
        print(f"    mat={mid} {mat_names2[mid]:7s}: S11 mean={sv[:,0].mean():+8.2f}  "
              f"min={sv[:,0].min():+8.2f}  max={sv[:,0].max():+8.2f}  "
              f"S22 mean={sv[:,1].mean():+8.2f}  S33 mean={sv[:,2].mean():+8.2f}")

    if diag_elem_si >= 0:
        print(f"\n  -- Diagnosis: interior Si_SoC element (all nodes == Si_SoC) --")
        conn_si = elem_conn[diag_elem_si]
        print(f"  Per-node GT S11 at element {diag_elem_si} nodes:")
        for ni in conn_si:
            print(f"    node {ni}: GT S11={s_gt[ni,0]:+10.4f}  S22={s_gt[ni,1]:+10.4f}  S33={s_gt[ni,2]:+10.4f}")
        diagnose_element(diag_elem_si, mesh_pos, disp_t, delta_T,
                         elem_conn, elem_mat, node_mat, node_E, node_nu, node_CTE, s_gt)

    if diag_elem_iface >= 0:
        print(f"\n  -- Diagnosis: interface Si_SoC element (conn[0] != Si_SoC) --")
        diagnose_element(diag_elem_iface, mesh_pos, disp_t, delta_T,
                         elem_conn, elem_mat, node_mat, node_E, node_nu, node_CTE, s_gt)

    mat_names = {0:"Si_bot", 1:"Si_SoC", 2:"Solder", 3:"UF"}

    print(f"\n  --- FIXED version error stats ---")
    for fi in FRAMES:
        disp_t  = (world_pos[fi] - mesh_pos).astype(np.float64)
        delta_T = float(temperature[fi, 0]) - 250.0
        s_gt    = stress_gt[fi].astype(np.float64)

        sigma_c, has_w = compute_const_stress_nodes(
            mesh_pos, disp_t, delta_T, elem_conn, elem_mat,
            node_mat, node_E, node_nu, node_CTE
        )

        err     = np.abs(sigma_c[has_w] - s_gt[has_w])
        gt_mag  = np.abs(s_gt[has_w])
        rel_err = err / (gt_mag + 1e-10)

        print(f"\n  Frame {fi:2d}  ΔT={delta_T:.1f}°C   elastic nodes={has_w.sum()}")
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