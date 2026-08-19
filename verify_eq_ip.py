"""
verify_eq_ip.py  –  Run with Abaqus Python:
    abaqus python verify_eq_ip.py <odb_path>

Checks whether Abaqus integration-point stresses satisfy static equilibrium.
For C3D8R (1 IP per element at center), the FEM guarantee is:

    R_i = Σ_{e ∋ i}  V_e · σ_e^IP · ∇N_{e,i}  =  0   (interior nodes)

Unlike the earlier nodal-averaging check (verify_equilibrium_m0035.py),
this reads the raw IP stress directly – no extrapolation, no averaging –
so R_i should be near machine precision for a converged solution.

If confirmed, it validates storing elem_stress (T,M,6) as part of the NPZ
and using it for an equilibrium physics-informed loss (direction A).
"""

import sys
import numpy as np
from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT

odb_path = sys.argv[1]

# ── isoparametric coords for C3D8R nodes (same as extract_odb_m0035.py) ──────
XI   = np.array([-1, 1, 1,-1,-1, 1, 1,-1], dtype=np.float64)
ETA  = np.array([-1,-1, 1, 1,-1,-1, 1, 1], dtype=np.float64)
ZETA = np.array([-1,-1,-1,-1, 1, 1, 1, 1], dtype=np.float64)
PARAM = np.stack([XI, ETA, ZETA], axis=0)   # (3, 8)


def compute_residual_ip(mesh_pos, elem_ip_stress, elem_conn):
    """
    mesh_pos       : (N, 3)  reference node positions
    elem_ip_stress : (M, 6)  integration-point stress per element [S11..S23]
    elem_conn      : (M, 8)  0-indexed node connectivity

    Returns R (N, 3): nodal equilibrium residual.
    """
    N = mesh_pos.shape[0]
    M = elem_conn.shape[0]
    R = np.zeros((N, 3), dtype=np.float64)

    # All node coords: (M, 8, 3)
    xyz = mesh_pos[elem_conn].astype(np.float64)

    # Jacobians at center: (M, 3, 3)
    J_all = np.einsum("pa,eaq->epq", PARAM / 8.0, xyz)

    # Volumes: V_e = 8 * |det J|
    V_all = 8.0 * np.abs(np.linalg.det(J_all))       # (M,)

    # Stress tensors from Voigt [S11,S22,S33,S12,S13,S23]: (M, 3, 3)
    s = elem_ip_stress.astype(np.float64)              # (M, 6)
    sigma = np.stack([
        np.stack([s[:,0], s[:,3], s[:,4]], axis=1),
        np.stack([s[:,3], s[:,1], s[:,5]], axis=1),
        np.stack([s[:,4], s[:,5], s[:,2]], axis=1),
    ], axis=1)                                         # (M, 3, 3)

    # Physical shape-fn gradients at each of the 8 nodes: solve J^T x = param_grad
    Jt     = J_all.transpose(0, 2, 1)                 # (M, 3, 3)
    rhs    = np.tile((PARAM / 8.0)[None], (M, 1, 1))   # (M, 3, 8)  [param grads]
    dN     = np.linalg.solve(Jt, rhs)                 # (M, 3, 8)  [∇N_{e,a}]

    # f_{e,a} = V_e * σ_e · ∇N_{e,a}: (M, 8, 3)
    f = V_all[:,None,None] * np.einsum("eij,ejn->ein", sigma, dN).transpose(0,2,1)

    # Scatter-add to global residual (handles repeated indices correctly)
    np.add.at(R, elem_conn, f)
    return R


# ── Open ODB ─────────────────────────────────────────────────────────────────
print("Opening: {}".format(odb_path))
odb      = openOdb(odb_path, readOnly=True)
instance = list(odb.rootAssembly.instances.values())[0]

# Node index mapping
labels       = sorted([n.label for n in instance.nodes])
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
N = len(labels)

# Reference positions
mesh_pos = np.zeros((N, 3), dtype=np.float64)
for node in instance.nodes:
    mesh_pos[label_to_idx[node.label]] = node.coordinates

# Element connectivity (0-indexed, sorted)
elem_labels_sorted = sorted([e.label for e in instance.elements])
elem_lbl_to_eidx   = {lbl: i for i, lbl in enumerate(elem_labels_sorted)}
M = len(elem_labels_sorted)

elem_conn = np.zeros((M, 8), dtype=np.int32)
for elem in instance.elements:
    ei = elem_lbl_to_eidx[elem.label]
    for a, lbl in enumerate(elem.connectivity):
        elem_conn[ei, a] = label_to_idx[lbl]

print("  N={} nodes  M={} elements".format(N, M))

# Fixed nodes: zero-displacement magnitude across all frames → identify BCs
# (we exclude them from the residual check)
step   = list(odb.steps.values())[0]
frames = step.frames
T      = len(frames)
print("  T={} frames".format(T))

disp_max = np.zeros(N, dtype=np.float64)
for frame in frames:
    if 'U' not in frame.fieldOutputs:
        continue
    for v in frame.fieldOutputs['U'].getSubset(region=instance).values:
        idx = label_to_idx[v.nodeLabel]
        disp_max[idx] = max(disp_max[idx], float(np.linalg.norm(v.data)))

fixed_mask = disp_max < 1e-10   # (N,) True = fixed/constrained node
n_fixed    = int(fixed_mask.sum())
print("  Fixed nodes detected: {}".format(n_fixed))

# ── Compute residual at selected frames ───────────────────────────────────────
CHECK_FRAMES = [1, 5, 10, 15, T-1]

print("\n{:=^60}".format("  IP-stress equilibrium residual  "))

for fi in sorted(set(CHECK_FRAMES)):
    if fi >= T:
        continue
    frame = frames[fi]

    # Extract IP stress: one value per C3D8R element
    if 'S' not in frame.fieldOutputs:
        print("Frame {:2d}: no S field, skip".format(fi))
        continue

    fip         = frame.fieldOutputs['S'].getSubset(region=instance,
                                                     position=INTEGRATION_POINT)
    elem_stress = np.zeros((M, 6), dtype=np.float64)
    for v in fip.values:
        ei = elem_lbl_to_eidx.get(v.elementLabel, -1)
        if ei >= 0:
            elem_stress[ei] = np.array(v.data, dtype=np.float64)

    # Reference: mean von Mises stress × average element volume
    vm  = np.sqrt(0.5 * (
        (elem_stress[:,0]-elem_stress[:,1])**2 +
        (elem_stress[:,1]-elem_stress[:,2])**2 +
        (elem_stress[:,2]-elem_stress[:,0])**2 +
        6*(elem_stress[:,3]**2 + elem_stress[:,4]**2 + elem_stress[:,5]**2)
    ))                                             # (M,)

    # element volumes
    xyz_all = mesh_pos[elem_conn]
    J_all   = np.einsum("pa,eaq->epq", PARAM/8.0, xyz_all)
    V_all   = 8.0 * np.abs(np.linalg.det(J_all))  # (M,)
    V_avg   = V_all.mean()
    vm_mean = vm.mean()
    force_ref = vm_mean * V_avg

    # Equilibrium residual (all nodes)
    R = compute_residual_ip(mesh_pos, elem_stress, elem_conn)

    # Fixed node: its residual ≈ reaction force (the one non-zero external force)
    R_bc_mag = float(np.linalg.norm(R[fixed_mask].sum(axis=0)))

    # Free nodes
    R_free = R[~fixed_mask]                        # (N-n_fixed, 3)
    R_mag  = np.linalg.norm(R_free, axis=1)        # per-node residual magnitude

    # --- Characteristic element force: expected magnitude of one node's contribution
    # For node i connected to k elements, |f_elem| ~ vm * V_e * |grad_N| ~ vm * V_e / h
    # where h ~ V_e^(1/3).  This is the scale BEFORE inter-element cancellation.
    h_avg      = V_avg ** (1.0 / 3.0)
    f_char     = vm_mean * V_avg / h_avg   # ≈ vm * h^2  (force scale per element per node)

    # Estimate total element-force magnitude at each node (before cancellation)
    # A node with k elements contributes ~k * f_char to the "gross" internal force
    # The residual R_i is what REMAINS after these k forces (nearly) cancel.
    # We compute the actual gross force:
    R_gross    = np.zeros((len(labels), 3), dtype=np.float64)
    # accumulate |absolute| contribution of each element to each of its nodes
    xyz_all    = mesh_pos[elem_conn].astype(np.float64)
    J_all      = np.einsum("pa,eaq->epq", PARAM / 8.0, xyz_all)
    V_all      = 8.0 * np.abs(np.linalg.det(J_all))
    s_e        = elem_stress.astype(np.float64)
    sig_all    = np.stack([
        np.stack([s_e[:,0], s_e[:,3], s_e[:,4]], axis=1),
        np.stack([s_e[:,3], s_e[:,1], s_e[:,5]], axis=1),
        np.stack([s_e[:,4], s_e[:,5], s_e[:,2]], axis=1)], axis=1)
    Jt         = J_all.transpose(0, 2, 1)
    rhs_g      = np.tile((PARAM / 8.0)[None], (len(elem_conn), 1, 1))
    dN_all     = np.linalg.solve(Jt, rhs_g)
    f_all      = V_all[:,None,None] * np.einsum("eij,ejn->ein", sig_all, dN_all).transpose(0,2,1)
    np.add.at(R_gross, elem_conn, np.abs(f_all))   # sum of abs contributions
    R_gross_mag = np.linalg.norm(R_gross[~fixed_mask], axis=1)   # (N-1,)

    # Key metric: how much of the gross force survives as residual (non-cancelled)?
    cancel_ratio = R_mag / (R_gross_mag + 1e-30)   # 0 = perfect cancellation, 1 = no cancellation

    rel_elem = R_mag / force_ref if force_ref > 0 else R_mag

    print("\nFrame {:2d}  (frameValue={:.3f})".format(fi, frame.frameValue))
    print("  vm_mean={:.4e}  V_avg={:.4e}  h_avg={:.4e}  f_char={:.4e}".format(
          vm_mean, V_avg, h_avg, f_char))
    print("  |R_gross| (sum |elem contrib|)  mean={:.4e}  max={:.4e}".format(
          R_gross_mag.mean(), R_gross_mag.max()))
    print("  |R_net|   (actual residual)      mean={:.4e}  max={:.4e}".format(
          R_mag.mean(), R_mag.max()))
    print("  Cancellation ratio |R_net|/|R_gross| mean={:.5f}  max={:.5f}".format(
          cancel_ratio.mean(), cancel_ratio.max()))
    print("  Rel vs vm*V_avg   mean={:.4f}  max={:.4f}".format(
          rel_elem.mean(), rel_elem.max()))
    print("  Reaction |R_BC|   = {:.4e}  (global sum of free-node residuals)".format(
          R_bc_mag))

odb.close()
print("\nDone.")