"""
Extract Abaqus ODB -> NPZ for IC package M0035.
Run via Abaqus Python:
    abaqus python extract_odb_m0035.py <odb_path> <npz_path> <E_uf_MPa> <CTE_uf_per_C>

Nodal recovery rule for integration-point fields (S, PE, PEEQ, LE):
  1. C3D8R has exactly 1 integration point (center). Abaqus ELEMENT_NODAL
     extrapolates this single IP value identically to all 8 element corner nodes.
  2. Any-adjacency material assignment (node_mat):
       Solder/Microbump (mat 2) > Underfill (mat 3) > Si (mat 0, 1)
     A node is assigned to the highest-priority material it is adjacent to,
     regardless of how many elements of other materials also share it.
     This protects microbump nodes that would otherwise be absorbed by majority vote.
  3. Material-separated averaging: only elements whose material == node_mat[n]
     contribute to node n.  Cross-interface averaging is suppressed.
  4. Volume-weighted: weight each element's contribution by its undeformed volume
       V_e = 8 * det J(0,0,0)
     where J is the Jacobian at the reduced-integration point.  det J > 0 is
     verified for every element.  Same-mesh samples share the same elem_vols array.

This rule is fully deterministic, mesh-size independent, and applied identically
to all 220 samples (and any future meshes of M0035).

NPZ fields:
    mesh_pos        (N, 3)        reference node positions
    world_pos       (T, N, 3)     deformed positions, T frames incl. frame 0
    velocity        (T-1, N, 3)   displacement increments between frames
    stress          (T, N, 6)     [S11,S22,S33,S12,S13,S23]  (frame 0 = 0)
    pe              (T, N, 6)     plastic strain tensor        (frame 0 = 0)
    peeq            (T, N)        equiv. plastic strain scalar (frame 0 = 0)
    le              (T, N, 6)     log strain tensor            (frame 0 = 0)
    temperature     (T, N)        temperature [deg C]
    mesh_edge_index (2, E)        directed edges (0-indexed)
    elem_conn       (M, 8)        C3D8R connectivity (0-indexed)
    elem_mat        (M,)          0=bottom_si 1=soc_si 2=solder 3=underfill
    node_mat        (N,)          majority-vote material index
    node_E          (N,)          Young's modulus [MPa]
    node_nu         (N,)          Poisson's ratio
    node_CTE        (N,)          CTE [1/deg C]
    node_is_plastic (N,)          bool, True for solder nodes
    E_uf_MPa        ()            underfill E for this sample [MPa]
    CTE_uf          ()            underfill CTE for this sample [1/deg C]
"""

import sys
import os
import numpy as np
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL

# ── Args ──────────────────────────────────────────────────────────────────────
odb_path = sys.argv[1]
npz_path = sys.argv[2]
E_uf_MPa = float(sys.argv[3])
CTE_uf   = float(sys.argv[4])

# ── Material definitions ───────────────────────────────────────────────────────
# mat_idx: 0=BOTTOM_SI  1=SOC_SI  2=SOLDER  3=UNDERFILL
MAT_NAME_TO_IDX = {
    'MAT_BOTTOM_DIE_SI':    0,
    'MAT_SOC_SI':           1,
    'MAT_MICROBUMP_SOLDER': 2,
    'MAT_UF_E03_CTE25':     3,
}
MAT_E   = np.array([130000.0, 130000.0, 47000.0, E_uf_MPa], dtype=np.float32)
MAT_NU  = np.array([0.28,     0.28,     0.36,    0.33    ], dtype=np.float32)
MAT_CTE = np.array([3.1e-6,   3.1e-6,   22.5e-6, CTE_uf  ], dtype=np.float32)
MAT_IS_PLASTIC = np.array([False, False, True, False])

# C3D8/C3D8R local edge pairs (12 unique edges per hex)
C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# C3D8R isoparametric node coords (xi, eta, zeta) in [-1,1]^3
_XI   = np.array([-1, 1, 1,-1,-1, 1, 1,-1], dtype=np.float64)
_ETA  = np.array([-1,-1, 1, 1,-1,-1, 1, 1], dtype=np.float64)
_ZETA = np.array([-1,-1,-1,-1, 1, 1, 1, 1], dtype=np.float64)


# ── Geometry helpers ───────────────────────────────────────────────────────────
def hex_volume_1pt(p, elem_label=None):
    """
    Undeformed volume of a C3D8R element using the single Gauss-point (center)
    Jacobian.  p: (8, 3) float64 node coordinates in Abaqus C3D8 order.
    Raises ValueError if det J <= 0 (inverted / degenerate element).
    Returns scalar volume (float64).
    """
    dNdxi   = _XI   / 8.0
    dNdeta  = _ETA  / 8.0
    dNdzeta = _ZETA / 8.0
    J = np.array([dNdxi @ p, dNdeta @ p, dNdzeta @ p])  # (3, 3)
    det_J = np.linalg.det(J)
    if det_J <= 0:
        raise ValueError("Non-positive Jacobian det={} for element {}".format(
            det_J, elem_label))
    return 8.0 * det_J


def compute_elem_volumes(mesh_pos, elem_conns, elem_labels):
    """Return (M,) float64 array of undeformed element volumes. Verifies det J > 0."""
    M = len(elem_conns)
    vols = np.zeros(M, dtype=np.float64)
    for i, conn in enumerate(elem_conns):
        vols[i] = hex_volume_1pt(mesh_pos[conn].astype(np.float64),
                                  elem_label=elem_labels[i])
    return vols


# ── Nodal recovery ─────────────────────────────────────────────────────────────
def _recover_nodal(field_en_values, N, label_to_idx,
                   node_mat, elem_label_to_mat, elem_label_to_eidx, elem_vols,
                   n_comp):
    """
    Core routine: volume-weighted, material-separated nodal averaging.
    field_en_values : iterable of FieldValue objects at ELEMENT_NODAL position.
    n_comp          : number of components (6 for tensors, 1 for scalars).
    Returns (N, n_comp) float32.
    """
    vals    = np.zeros((N, n_comp), dtype=np.float64)
    weights = np.zeros(N,           dtype=np.float64)

    for v in field_en_values:
        nidx     = label_to_idx[v.nodeLabel]
        elem_m   = elem_label_to_mat.get(v.elementLabel, -1)
        if elem_m != node_mat[nidx]:       # skip cross-material contributions
            continue
        w = elem_vols[elem_label_to_eidx[v.elementLabel]]
        d = v.data
        if n_comp == 1:
            vals[nidx, 0] += w * (float(d) if not hasattr(d, '__len__') else float(d[0]))
        else:
            vals[nidx]    += w * np.array(d, dtype=np.float64)
        weights[nidx] += w

    result = np.zeros((N, n_comp), dtype=np.float32)
    mask = weights > 0
    result[mask] = (vals[mask] / weights[mask, np.newaxis]).astype(np.float32)
    return result


def ip_to_nodal_tensor(frame, key, instance, N, label_to_idx,
                        node_mat, elem_label_to_mat, elem_label_to_eidx, elem_vols):
    if key not in frame.fieldOutputs:
        return np.zeros((N, 6), dtype=np.float32)
    fen = frame.fieldOutputs[key].getSubset(region=instance, position=ELEMENT_NODAL)
    return _recover_nodal(fen.values, N, label_to_idx,
                          node_mat, elem_label_to_mat, elem_label_to_eidx, elem_vols, 6)


def ip_to_nodal_scalar(frame, key, instance, N, label_to_idx,
                        node_mat, elem_label_to_mat, elem_label_to_eidx, elem_vols):
    if key not in frame.fieldOutputs:
        return np.zeros(N, dtype=np.float32)
    fen = frame.fieldOutputs[key].getSubset(region=instance, position=ELEMENT_NODAL)
    return _recover_nodal(fen.values, N, label_to_idx,
                          node_mat, elem_label_to_mat, elem_label_to_eidx, elem_vols, 1)[:, 0]


# ── Open ODB ──────────────────────────────────────────────────────────────────
print("Opening: {}".format(odb_path))
odb      = openOdb(odb_path, readOnly=True)
instance = list(odb.rootAssembly.instances.values())[0]

# Node label -> 0-indexed (sorted for determinism across all samples)
labels       = sorted([n.label for n in instance.nodes])
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
N = len(labels)
print("  Nodes: {}".format(N))

# Reference positions
mesh_pos = np.zeros((N, 3), dtype=np.float32)
for node in instance.nodes:
    mesh_pos[label_to_idx[node.label]] = node.coordinates

# ── Element material assignment ────────────────────────────────────────────────
elem_label_to_mat = {}
try:
    for sa in instance.sectionAssignments:
        sec_name = sa.sectionName
        try:
            mat_name = odb.sections[sec_name].material
        except Exception:
            mat_name = sec_name
        mat_idx = MAT_NAME_TO_IDX.get(mat_name, -1)
        if mat_idx < 0:
            for k, v in MAT_NAME_TO_IDX.items():
                if k.lower() in mat_name.lower() or mat_name.lower() in k.lower():
                    mat_idx = v
                    break
        for elem in sa.region.elements:
            elem_label_to_mat[elem.label] = mat_idx
except Exception as e:
    print("  Warning: section assignment read failed: {}".format(e))

# ── Build elem_conn, elem_mat, graph edges ─────────────────────────────────────
elem_labels_sorted  = sorted([e.label for e in instance.elements])
elem_label_to_eidx  = {lbl: i for i, lbl in enumerate(elem_labels_sorted)}
M = len(elem_labels_sorted)

elem_conns    = [None] * M
elem_mats_out = np.zeros(M, dtype=np.int32)
edge_set      = set()

for elem in instance.elements:
    ei           = elem_label_to_eidx[elem.label]
    conn_labels  = list(elem.connectivity)
    conn_0idx    = [label_to_idx[lbl] for lbl in conn_labels]
    elem_conns[ei]    = conn_0idx
    elem_mats_out[ei] = elem_label_to_mat.get(elem.label, 0)
    for (a, b) in C3D8_EDGES:
        u, v_node = conn_0idx[a], conn_0idx[b]
        edge_set.add((u, v_node))
        edge_set.add((v_node, u))

elem_conn_arr = np.array(elem_conns, dtype=np.int32)   # (M, 8)
edges         = np.array(sorted(edge_set), dtype=np.int64).T  # (2, E)
print("  Elements: {}  Edges: {}".format(M, edges.shape[1]))

# Node material: any-adjacency rule with fixed priority
#   Solder/Microbump (mat 2) > Underfill (mat 3) > Si (mat 0, 1)
# Protects microbump nodes that have no internal nodes (1-element bumps).
# Deterministic: depends only on adjacency, not on element reading order.
MAT_PRIORITY = np.array([2, 3, 0, 1], dtype=np.int32)  # index=mat_idx, value=priority (0=highest)

node_mat_votes = np.zeros((N, 4), dtype=np.int32)
for ei in range(M):
    mat_i = int(elem_mats_out[ei])
    if 0 <= mat_i < 4:
        for nidx in elem_conns[ei]:
            node_mat_votes[nidx, mat_i] += 1

adjacent = node_mat_votes > 0                                     # (N, 4) bool
if not adjacent.any(axis=1).all():
    raise ValueError("Found nodes not connected to any element")
pri = np.where(adjacent, MAT_PRIORITY[np.newaxis, :], 999)       # 999 = not adjacent
node_mat        = np.argmin(pri, axis=1).astype(np.int32)
node_E          = MAT_E[node_mat]
node_nu         = MAT_NU[node_mat]
node_CTE        = MAT_CTE[node_mat]
node_is_plastic = MAT_IS_PLASTIC[node_mat]

# Undeformed element volumes for volume-weighted averaging
# Same mesh for all 220 samples -> computed once from reference coordinates
print("  Computing element volumes ...")
elem_vols = compute_elem_volumes(mesh_pos, elem_conns, elem_labels_sorted)

# ── Rebuild elem_label_to_mat using 0-indexed key ─────────────────────────────
# _recover_nodal uses v.elementLabel (Abaqus label, 1-indexed)
# elem_label_to_mat already uses Abaqus labels, elem_label_to_eidx likewise

# ── Extract frame data ─────────────────────────────────────────────────────────
step   = list(odb.steps.values())[0]
frames = step.frames
T      = len(frames)
print("  Step: {}  Frames: {}".format(step.name, T))

# Shared kwargs for all IP->nodal calls
kw = dict(
    instance          = instance,
    N                 = N,
    label_to_idx      = label_to_idx,
    node_mat          = node_mat,
    elem_label_to_mat = elem_label_to_mat,
    elem_label_to_eidx= elem_label_to_eidx,
    elem_vols         = elem_vols,
)

world_pos   = np.zeros((T, N, 3), dtype=np.float32)
stress      = np.zeros((T, N, 6), dtype=np.float32)
pe          = np.zeros((T, N, 6), dtype=np.float32)
peeq_arr    = np.zeros((T, N),    dtype=np.float32)
le          = np.zeros((T, N, 6), dtype=np.float32)
temperature = np.zeros((T, N),    dtype=np.float32)

for fi, frame in enumerate(frames):
    print("  Frame {}/{}  t={:.4f}".format(fi, T - 1, frame.frameValue))

    # Displacement (nodal, no averaging needed)
    world_pos[fi] = mesh_pos.copy()
    if 'U' in frame.fieldOutputs:
        for v in frame.fieldOutputs['U'].getSubset(region=instance).values:
            idx = label_to_idx[v.nodeLabel]
            world_pos[fi, idx] = mesh_pos[idx] + np.array(v.data, dtype=np.float32)

    # Temperature (nodal)
    for nt_key in ('NT11', 'NT'):
        if nt_key in frame.fieldOutputs:
            for v in frame.fieldOutputs[nt_key].getSubset(region=instance).values:
                idx = label_to_idx[v.nodeLabel]
                d   = v.data
                temperature[fi, idx] = float(d) if not hasattr(d, '__len__') else float(d[0])
            break

    # Integration-point fields: material-separated, volume-weighted nodal averaging
    stress[fi]   = ip_to_nodal_tensor(frame, 'S',    **kw)
    pe[fi]       = ip_to_nodal_tensor(frame, 'PE',   **kw)
    peeq_arr[fi] = ip_to_nodal_scalar(frame, 'PEEQ', **kw)
    le[fi]       = ip_to_nodal_tensor(frame, 'LE',   **kw)

velocity = np.diff(world_pos, axis=0).astype(np.float32)  # (T-1, N, 3)

# ── Validation (before odb.close — last frame still accessible) ───────────────
MAT_NAMES = ['Si_bottom', 'Si_SoC', 'Solder', 'Underfill']
print("  --- Validation ---")
for m in range(4):
    n_elem = int((elem_mats_out == m).sum())
    n_node = int((node_mat == m).sum())
    print("  mat {:d} {:12s}: {:4d} elems  {:4d} nodes".format(
        m, MAT_NAMES[m], n_elem, n_node))

# Check: every Solder element must have at least one Solder node
solder_elem_idx = np.where(elem_mats_out == 2)[0]
orphan = sum(1 for ei in solder_elem_idx
             if not any(node_mat[nidx] == 2 for nidx in elem_conns[ei]))
if orphan > 0:
    print("  WARNING: {} Solder element(s) have ZERO Solder nodes!".format(orphan))
else:
    print("  OK: all Solder elements have at least one Solder node")

# Check: weight denominator > 0 for all nodes (spot-check on last frame S)
last_frame = frames[-1]
if 'S' in last_frame.fieldOutputs:
    fen   = last_frame.fieldOutputs['S'].getSubset(region=instance, position=ELEMENT_NODAL)
    w_sum = np.zeros(N, dtype=np.float64)
    for v in fen.values:
        nidx   = label_to_idx[v.nodeLabel]
        elem_m = elem_label_to_mat.get(v.elementLabel, -1)
        if elem_m == node_mat[nidx]:
            w_sum[nidx] += elem_vols[elem_label_to_eidx[v.elementLabel]]
    zero_w = int((w_sum == 0).sum())
    if zero_w > 0:
        print("  WARNING: {} nodes have zero weight denominator for S!".format(zero_w))
    else:
        print("  OK: all nodes have positive weight denominator")

odb.close()

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(npz_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

np.savez(
    npz_path,
    mesh_pos        = mesh_pos,
    world_pos       = world_pos,
    velocity        = velocity,
    stress          = stress,
    pe              = pe,
    peeq            = peeq_arr,
    le              = le,
    temperature     = temperature,
    mesh_edge_index = edges,
    elem_conn       = elem_conn_arr,
    elem_mat        = elem_mats_out,
    node_mat        = node_mat,
    node_E          = node_E,
    node_nu         = node_nu,
    node_CTE        = node_CTE,
    node_is_plastic = node_is_plastic,
    E_uf_MPa        = np.float32(E_uf_MPa),
    CTE_uf          = np.float32(CTE_uf),
)

print("Saved: {}".format(npz_path))
print("  world_pos {} | stress {} | pe {} | peeq {} | le {}".format(
    world_pos.shape, stress.shape, pe.shape, peeq_arr.shape, le.shape))