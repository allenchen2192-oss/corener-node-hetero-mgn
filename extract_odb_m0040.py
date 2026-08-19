"""
Extract Abaqus ODB -> NPZ for IC package M0040 (HeteroMGN element-node format).
Run via Abaqus Python:
    abaqus python extract_odb_m0040.py <odb_path> <inp_path> <npz_path>

Architecture: HeteroMGN with element nodes.
C3D8 (full integration, 8 Gauss points): S and PEEQ are averaged over all
integration points per element to produce element-level scalars.

NPZ fields:
    mesh_pos:       (N, 3)      reference node positions (at T_ref=250 degC)
    world_pos:      (T, N, 3)   deformed positions; T=21 frames incl. frame 0
    velocity:       (T-1, N, 3) displacement increments between frames
    node_type:      (N,)        BC encoding:
                                  0 = free interior
                                  1 = XSYMM (U1=0)
                                  2 = YSYMM (U2=0)
                                  3 = XSYMM + YSYMM (U1=U2=0)
                                  4 = XSYMM + YSYMM + U3=0 (origin, fully fixed)
    elem_conn:      (M, 8)      element connectivity (0-indexed node indices)
    elem_mat:       (M,)        0=Si_bottom 1=Si_SoC 2=Solder 3=UF
    elem_E:         (M,)        Young's modulus [MPa]
    elem_nu:        (M,)        Poisson's ratio
    elem_CTE:       (M,)        CTE [1/degC]
    elem_vm_stress: (T, M)      element-level von Mises stress [MPa]
    elem_peeq:      (T, M)      element-level equiv. plastic strain (0 for Si/UF)
"""

import sys
import os
import re
import numpy as np
from odbAccess import openOdb
from abaqusConstants import INTEGRATION_POINT

# ── Material definitions ───────────────────────────────────────────────────────
MAT_NAME_TO_IDX = {
    'MAT_BOTTOM_DIE_SI':    0,
    'MAT_SOC_SI':           1,
    'MAT_MICROBUMP_SOLDER': 2,
    'MAT_UF_E03_CTE25':     3,
}
# Fixed material properties (UF E and CTE are sample-specific, set after parsing INP)
_MAT_E_FIXED   = {0: 130000.0, 1: 130000.0, 2: 47000.0}   # MPa
_MAT_NU_FIXED  = {0: 0.28,     1: 0.28,     2: 0.36,    3: 0.33}
_MAT_CTE_FIXED = {0: 3.1e-6,   1: 3.1e-6,   2: 2.25e-5}   # 1/degC; UF from INP

# BC node sets -> bit flags for node_type computation
BC_NODESETS = {
    'SET_XSYMM_X0': 0x01,   # XSYMM: U1=0
    'SET_YSYMM_Y0': 0x02,   # YSYMM: U2=0
    'SET_ORIGIN':   0x04,   # single-point: U3=0
}
# Bit pattern -> categorical node_type (M0040-specific; origin node is 0x07 = all three)
BITS_TO_TYPE = {0: 0, 1: 1, 2: 2, 3: 3, 7: 4}


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_inp_uf_props(inp_path):
    """Parse UF Young's modulus and CTE from INP file."""
    with open(inp_path, 'r') as f:
        content = f.read()
    mat_match = re.search(
        r'\*Material[^\n]*name\s*=\s*MAT_UF_E03_CTE25[^\n]*\n(.*?)(?=\*Material|\Z)',
        content, re.IGNORECASE | re.DOTALL)
    if not mat_match:
        raise ValueError("Cannot find MAT_UF_E03_CTE25 in " + inp_path)
    block = mat_match.group(1)

    e_m = re.search(r'\*Elastic[^\n]*\n\s*([0-9.eE+\-]+)', block, re.IGNORECASE)
    if not e_m:
        raise ValueError("Cannot parse *Elastic for MAT_UF in " + inp_path)

    cte_m = re.search(r'\*Expansion[^\n]*\n\s*([0-9.eE+\-]+)', block, re.IGNORECASE)
    if not cte_m:
        raise ValueError("Cannot parse *Expansion for MAT_UF in " + inp_path)

    return float(e_m.group(1)), float(cte_m.group(1))


def get_nodeset_nodes(odb, instance, set_name):
    """Return nodes from a named set, checking instance then assembly level."""
    if set_name in instance.nodeSets:
        return instance.nodeSets[set_name].nodes
    for key in (set_name, '{}.{}'.format(instance.name, set_name)):
        if key in odb.rootAssembly.nodeSets:
            return odb.rootAssembly.nodeSets[key].nodes
    return []


# ── Args ──────────────────────────────────────────────────────────────────────
if len(sys.argv) != 4:
    print("Usage: abaqus python extract_odb_m0040.py <odb_path> <inp_path> <npz_path>")
    sys.exit(1)

odb_path, inp_path, npz_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Parse sample-specific UF properties from INP
E_uf, CTE_uf = parse_inp_uf_props(inp_path)
print("UF properties: E={:.1f} MPa, CTE={:.3e} 1/degC".format(E_uf, CTE_uf))

MAT_E   = [_MAT_E_FIXED.get(i, E_uf)   for i in range(4)]
MAT_NU  = [_MAT_NU_FIXED[i]            for i in range(4)]
MAT_CTE = [_MAT_CTE_FIXED.get(i, CTE_uf) for i in range(4)]

# ── Open ODB ──────────────────────────────────────────────────────────────────
print("Opening: {}".format(odb_path))
odb      = openOdb(odb_path, readOnly=True)
instance = list(odb.rootAssembly.instances.values())[0]
print("  Instance: {}".format(instance.name))

# ── Node mapping (sorted by label for cross-sample determinism) ───────────────
labels       = sorted([n.label for n in instance.nodes])
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
N = len(labels)
print("  Nodes: {}".format(N))

mesh_pos = np.zeros((N, 3), dtype=np.float32)
for node in instance.nodes:
    mesh_pos[label_to_idx[node.label]] = node.coordinates

# ── Element mapping (sorted by label) ────────────────────────────────────────
elem_labels_sorted = sorted([e.label for e in instance.elements])
elem_label_to_eidx = {lbl: i for i, lbl in enumerate(elem_labels_sorted)}
M = len(elem_labels_sorted)
print("  Elements: {}".format(M))

elem_conn = np.zeros((M, 8), dtype=np.int32)
for e in instance.elements:
    ei = elem_label_to_eidx[e.label]
    for j, lbl in enumerate(e.connectivity):
        elem_conn[ei, j] = label_to_idx[lbl]

# ── Element material from section assignments ─────────────────────────────────
elem_label_to_mat = {}
try:
    for sa in instance.sectionAssignments:
        sec_name = sa.sectionName
        try:
            mat_name = odb.sections[sec_name].material
        except Exception:
            mat_name = sec_name
        mat_idx = MAT_NAME_TO_IDX.get(mat_name, -1)
        if mat_idx < 0:   # fuzzy match fallback
            for k, v in MAT_NAME_TO_IDX.items():
                if k.lower() in mat_name.lower() or mat_name.lower() in k.lower():
                    mat_idx = v
                    break
        for elem in sa.region.elements:
            elem_label_to_mat[elem.label] = mat_idx
except Exception as ex:
    print("  Warning: sectionAssignments failed: {}".format(ex))

elem_mat = np.array([elem_label_to_mat.get(lbl, 0)
                     for lbl in elem_labels_sorted], dtype=np.int32)
if (elem_mat < 0).any():
    print("  WARNING: {} elements have no material assignment".format(int((elem_mat < 0).sum())))

# ── Element properties ────────────────────────────────────────────────────────
elem_E   = np.array([MAT_E[m]   for m in elem_mat], dtype=np.float32)
elem_nu  = np.array([MAT_NU[m]  for m in elem_mat], dtype=np.float32)
elem_CTE = np.array([MAT_CTE[m] for m in elem_mat], dtype=np.float32)

# ── BC node_type ──────────────────────────────────────────────────────────────
bits = np.zeros(N, dtype=np.int8)
for set_name, bit in BC_NODESETS.items():
    for n in get_nodeset_nodes(odb, instance, set_name):
        bits[label_to_idx[n.label]] |= bit

node_type = np.zeros(N, dtype=np.int8)
for i in range(N):
    b = int(bits[i])
    if b not in BITS_TO_TYPE:
        print("  Warning: unexpected BC bit pattern {:d} at node {:d}".format(
            b, labels[i]))
    node_type[i] = BITS_TO_TYPE.get(b, b)

# ── Frame extraction ──────────────────────────────────────────────────────────
step   = list(odb.steps.values())[0]
frames = step.frames
T      = len(frames)
print("  Step: {}  Frames: {}".format(step.name, T))

world_pos      = np.zeros((T, N, 3), dtype=np.float32)
elem_vm_stress = np.zeros((T, M),    dtype=np.float32)
elem_peeq      = np.zeros((T, M),    dtype=np.float32)

ip_cnt_last = np.zeros(M, dtype=np.int32)   # for validation

for fi, frame in enumerate(frames):
    print("  Frame {}/{}  t={:.4f}".format(fi, T - 1, frame.frameValue))

    # Displacement -> world_pos
    world_pos[fi] = mesh_pos.copy()
    if 'U' in frame.fieldOutputs:
        for v in frame.fieldOutputs['U'].getSubset(region=instance).values:
            idx = label_to_idx[v.nodeLabel]
            world_pos[fi, idx] = mesh_pos[idx] + np.array(v.data, dtype=np.float32)

    # Stress: sum over integration points, then average -> von Mises
    s6_sum = np.zeros((M, 6), dtype=np.float64)
    ip_cnt = np.zeros(M, dtype=np.int32)
    if 'S' in frame.fieldOutputs:
        fo_s = frame.fieldOutputs['S'].getSubset(region=instance,
                                                  position=INTEGRATION_POINT)
        for v in fo_s.values:
            ei = elem_label_to_eidx[v.elementLabel]
            s6_sum[ei] += np.array(v.data, dtype=np.float64)
            ip_cnt[ei] += 1
        valid = ip_cnt > 0
        s6_sum[valid] /= ip_cnt[valid, np.newaxis]
        s = s6_sum
        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1])**2 +
                             (s[:, 1] - s[:, 2])**2 +
                             (s[:, 2] - s[:, 0])**2 +
                             6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)))
        elem_vm_stress[fi] = vm.astype(np.float32)
    if fi == T - 1:
        ip_cnt_last = ip_cnt.copy()

    # PEEQ: average over integration points per element
    if 'PEEQ' in frame.fieldOutputs:
        pq_sum = np.zeros(M, dtype=np.float64)
        pq_cnt = np.zeros(M, dtype=np.int32)
        fo_pq = frame.fieldOutputs['PEEQ'].getSubset(region=instance,
                                                      position=INTEGRATION_POINT)
        for v in fo_pq.values:
            ei = elem_label_to_eidx[v.elementLabel]
            d  = v.data
            pq_sum[ei] += float(d) if not hasattr(d, '__len__') else float(d[0])
            pq_cnt[ei] += 1
        valid_pq = pq_cnt > 0
        pq_sum[valid_pq] /= pq_cnt[valid_pq]
        elem_peeq[fi] = pq_sum.astype(np.float32)

velocity = np.diff(world_pos, axis=0).astype(np.float32)   # (T-1, N, 3)

# ── Validation ────────────────────────────────────────────────────────────────
MAT_NAMES = ['Si_bottom', 'Si_SoC', 'Solder', 'UF']
print("  --- Validation ---")
for m in range(4):
    print("  mat {:d} {:9s}: {:4d} elems".format(m, MAT_NAMES[m], int((elem_mat == m).sum())))
print("  ip_cnt (last frame): min={} max={}".format(
    int(ip_cnt_last.min()), int(ip_cnt_last.max())))
solder_mask = elem_mat == 2
if solder_mask.any():
    print("  PEEQ solder max: {:.6f}".format(float(elem_peeq[:, solder_mask].max())))
    print("  PEEQ non-solder max: {:.6f}".format(float(elem_peeq[:, ~solder_mask].max())))
for t in range(5):
    print("  node_type {:d}: {:4d} nodes".format(t, int((node_type == t).sum())))

odb.close()

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(npz_path))
os.makedirs(out_dir, exist_ok=True)

np.savez_compressed(
    npz_path,
    mesh_pos       = mesh_pos,
    world_pos      = world_pos,
    velocity       = velocity,
    node_type      = node_type,
    elem_conn      = elem_conn,
    elem_mat       = elem_mat,
    elem_E         = elem_E,
    elem_nu        = elem_nu,
    elem_CTE       = elem_CTE,
    elem_vm_stress = elem_vm_stress,
    elem_peeq      = elem_peeq,
)

print("Saved: {}".format(npz_path))
print("  N={} nodes | M={} elems | T={} frames".format(N, M, T))
print("  world_pos {}  elem_vm_stress {}  elem_peeq {}".format(
    world_pos.shape, elem_vm_stress.shape, elem_peeq.shape))
