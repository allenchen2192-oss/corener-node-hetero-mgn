"""
verify_equilibrium_m0035.py
============================
Compute the nodal equilibrium residual for Abaqus GT stress data.

For a static problem with no external mechanical forces, every interior node
must satisfy:  R_i = Σ_e (σ_e · ∇N_{e,i}) V_e = 0

where the sum is over all elements containing node i.

If GT residual is small (relative to stress × volume scale), then using
equilibrium as an additional loss term is valid and will give a clean signal.

Usage:
  python verify_equilibrium_m0035.py
"""

import os
import sys
import numpy as np

# Allow running from any working directory by passing a workdir argument
if len(sys.argv) > 1:
    os.chdir(sys.argv[1])

NPZ_DIR = "./02_abaqus_npz_m0035"
SAMPLES = ["S0001", "S0002", "S0003"]

# C3D8R node isoparametric coordinates (same as extract_odb_m0035.py)
XI   = np.array([-1, 1, 1,-1,-1, 1, 1,-1], dtype=np.float64)
ETA  = np.array([-1,-1, 1, 1,-1,-1, 1, 1], dtype=np.float64)
ZETA = np.array([-1,-1,-1,-1, 1, 1, 1, 1], dtype=np.float64)
PARAM_COORDS = np.stack([XI, ETA, ZETA], axis=0)  # (3, 8)


def compute_residual(mesh_pos, stress_frame, elem_conn):
    """
    Compute nodal equilibrium residual R_i (N, 3) for one frame.

    R_i = Σ_{e ∋ i} V_e · σ_e · ∇N_{e,i}

    σ_e = average of 8 nodal stresses (Voigt: S11,S22,S33,S12,S13,S23)
    ∇N_{e,i} = physical gradient of shape function N_i at element center

    For a correct stress field: R_i ≈ 0 at all interior non-fixed nodes.
    Vectorized over all elements simultaneously.
    """
    N = mesh_pos.shape[0]
    M = elem_conn.shape[0]
    R = np.zeros((N, 3), dtype=np.float64)

    # All element node positions: (M, 8, 3)
    xyz_all = mesh_pos[elem_conn].astype(np.float64)

    # Jacobian for all elements: (M, 3, 3)
    # J[e, p, q] = Σ_a (param_a^p / 8) * xyz[e,a,q]
    J_all = (PARAM_COORDS / 8.0) @ xyz_all   # (3,8) @ (M,8,3) via einsum below
    # Above doesn't broadcast; use einsum:
    J_all = np.einsum("pa,eaq->epq", PARAM_COORDS / 8.0, xyz_all)  # (M, 3, 3)

    # Element volumes: V_e = 8 * |det(J)|
    det_J = np.linalg.det(J_all)             # (M,)
    V_all = 8.0 * np.abs(det_J)             # (M,)

    # Element stress (average of 8 nodal stresses): (M, 6)
    s_all = stress_frame[elem_conn].mean(axis=1).astype(np.float64)  # (M, 8, 6) → (M, 6)

    # Stress tensors: (M, 3, 3)
    sigma_all = np.stack([
        np.stack([s_all[:,0], s_all[:,3], s_all[:,4]], axis=1),
        np.stack([s_all[:,3], s_all[:,1], s_all[:,5]], axis=1),
        np.stack([s_all[:,4], s_all[:,5], s_all[:,2]], axis=1),
    ], axis=1)   # (M, 3, 3)

    # Isoparametric gradients of all 8 shape functions: (3, 8) rows=param-dir cols=node
    param_grads = PARAM_COORDS / 8.0        # (3, 8)

    # Physical gradients: ∇N_{e,a} = J_e^{-1} · param_grad_a
    # (NOT J^{-T}: correct FEM formula is dN/dX = J^{-1} @ dN/dxi)
    # np.linalg.solve(A (M,3,3), b (M,3,8)) solves Ax=b
    rhs = np.tile(param_grads[None], (M, 1, 1))     # (M, 3, 8)
    phys_grads = np.linalg.solve(J_all, rhs)        # (M, 3, 8): J^{-1} @ dNdxi

    # Force on each of 8 nodes from each element: (M, 8, 3)
    # f_{e,a} = V_e * σ_e · ∇N_{e,a}
    # sigma_all (M,3,3) @ phys_grads (M,3,8) → (M,3,8), then transpose → (M,8,3)
    f_all = V_all[:, None, None] * np.einsum("eij,ejn->ein", sigma_all, phys_grads).transpose(0,2,1)
    # f_all shape: (M, 8, 3)

    # Scatter-add to global residual
    np.add.at(R, elem_conn, f_all)  # R[conn[e,a]] += f_all[e,a]

    return R


def analyze_sample(sid):
    path = os.path.join(NPZ_DIR, f"{sid}.npz")
    d = np.load(path, allow_pickle=True)

    mesh_pos  = d["mesh_pos"].astype(np.float32)    # (N, 3)
    stress    = d["stress"].astype(np.float32)      # (T, N, 6)
    elem_conn = d["elem_conn"].astype(np.int32)     # (M, 8)
    node_mat  = d["node_mat"].astype(np.int32)      # (N,)

    N, T = mesh_pos.shape[0], stress.shape[0]
    M    = elem_conn.shape[0]

    # Fixed node (only 1 in M0035)
    fixed_node = 281

    # Reference force scale: typical stress × typical element volume
    # Use frame 10 (middle of cooling) for representative values
    ref_frame = min(10, T - 1)
    vm_ref    = np.sqrt(0.5 * (
        (stress[ref_frame,:,0]-stress[ref_frame,:,1])**2 +
        (stress[ref_frame,:,1]-stress[ref_frame,:,2])**2 +
        (stress[ref_frame,:,2]-stress[ref_frame,:,0])**2 +
        6*(stress[ref_frame,:,3]**2 + stress[ref_frame,:,4]**2 + stress[ref_frame,:,5]**2)
    )).mean()

    # Estimate average element volume from total mesh volume
    # (rough: just for normalisation)
    xyz_range = mesh_pos.max(0) - mesh_pos.min(0)
    V_total   = xyz_range[0] * xyz_range[1] * xyz_range[2]
    V_avg     = V_total / M
    force_ref = vm_ref * V_avg   # MPa × m³  (or whatever length unit mesh_pos is in)

    print(f"\n{'='*60}")
    print(f"Sample: {sid}   N={N}  M={M}")
    print(f"  von Mises (frame 10, mean): {vm_ref:.2f} [stress units]")
    print(f"  Avg element volume:         {V_avg:.4e} [length³]")
    print(f"  Force reference scale:      {force_ref:.4e}")

    # Compute residual at selected frames
    for frame_idx in [1, 5, 10, 15, 20]:
        if frame_idx >= T:
            continue
        R = compute_residual(mesh_pos, stress[frame_idx], elem_conn)

        # Exclude fixed node
        mask = np.ones(N, dtype=bool)
        mask[fixed_node] = False

        R_free = R[mask]                              # (N-1, 3)
        R_mag  = np.linalg.norm(R_free, axis=1)      # (N-1,) per-node residual magnitude

        rel = R_mag / force_ref   # relative to force scale

        print(f"\n  Frame {frame_idx:2d}:")
        print(f"    |R| mean={R_mag.mean():.3e}  max={R_mag.max():.3e}  "
              f"  relative: mean={rel.mean():.4f}  max={rel.max():.4f}")

        # Per-material breakdown
        free_node_mat = node_mat[mask]   # (N-1,) material id for each free node
        for mat_idx, mat_name in [(0,"Si_bot"), (1,"Si_SoC"), (2,"Solder"), (3,"UF")]:
            mat_mask = (free_node_mat == mat_idx)
            if mat_mask.any():
                rm = R_mag[mat_mask]
                print(f"    [{mat_name}]  mean={rm.mean():.3e}  max={rm.max():.3e}  "
                      f"rel_mean={rm.mean()/force_ref:.4f}")


if __name__ == "__main__":
    for sid in SAMPLES:
        analyze_sample(sid)
    print("\nDone.")