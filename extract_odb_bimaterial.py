"""
Extract Abaqus ODB -> NPZ for bi-material beam.
Called internally by generate_bimaterial_beam.py via:
    abaqus python extract_odb_bimaterial.py <odb> <npz> <E1> <E2> <nu> <split_i> <NX> <NY> <NZ>

NPZ fields (compatible with preprocess_abaqus.py + extra material fields):
    mesh_pos        (N, 3)        reference node positions
    world_pos       (F, N, 3)     positions at each frame (F=20)
    velocity        (F-1, N, 3)   displacement increments
    stress_vm       (F, N)        von Mises stress
    mesh_edge_index (2, E)        directed edge connectivity (0-indexed)
    node_type       (N,)          0=free, 1=clamped, 2=loaded tip
    fixed_mask      (N,)          bool, True for clamped nodes
    elem_conn       (N_elem, 8)   C3D8 connectivity (0-indexed)
    n_nodes_per_elem              8
    elem_mat        (N_elem,)     material index: 0=Mat1, 1=Mat2  [NEW]
    mat_E           (2,)          [E1, E2]                         [NEW]
    mat_nu          (2,)          [nu, nu]                         [NEW]
"""

import sys
import os
import numpy as np
from odbAccess import openOdb

# ── Args ──────────────────────────────────────────────────────────────────────
odb_path = sys.argv[1]
npz_path = sys.argv[2]
E1       = float(sys.argv[3])
E2       = float(sys.argv[4])
nu       = float(sys.argv[5])
SPLIT_J  = int(sys.argv[6])
NX       = int(sys.argv[7])
NY       = int(sys.argv[8])
NZ       = int(sys.argv[9])

# ── Open ODB ──────────────────────────────────────────────────────────────────
odb      = openOdb(odb_path, readOnly=True)
instance = list(odb.rootAssembly.instances.values())[0]

# Build label -> 0-indexed map (sort for determinism)
labels       = sorted([n.label for n in instance.nodes])
label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
N            = len(labels)

# Reference positions
mesh_pos = np.zeros((N, 3), dtype=np.float32)
for node in instance.nodes:
    mesh_pos[label_to_idx[node.label]] = node.coordinates

# ── Field extraction ──────────────────────────────────────────────────────────
step     = list(odb.steps.values())[0]
F        = len(step.frames)          # should be 20

world_pos = np.zeros((F, N, 3), dtype=np.float32)
S_raw     = np.zeros((F, N, 6), dtype=np.float32)  # S11,S22,S33,S12,S13,S23

for fi, frame in enumerate(step.frames):
    # Displacements
    U = frame.fieldOutputs['U'].getSubset(region=instance)
    for v in U.values:
        idx = label_to_idx[v.nodeLabel]
        world_pos[fi, idx] = mesh_pos[idx] + np.array(v.data, dtype=np.float32)

    # Stress tensor (averaged at nodes by Abaqus)
    if 'S' in frame.fieldOutputs:
        S = frame.fieldOutputs['S'].getSubset(region=instance)
        for v in S.values:
            if hasattr(v, 'nodeLabel'):
                idx = label_to_idx[v.nodeLabel]
                S_raw[fi, idx] = np.array(v.data, dtype=np.float32)

odb.close()

# Von Mises from stress tensor components (S11,S22,S33,S12,S13,S23)
s11 = S_raw[..., 0]; s22 = S_raw[..., 1]; s33 = S_raw[..., 2]
s12 = S_raw[..., 3]; s13 = S_raw[..., 4]; s23 = S_raw[..., 5]
stress_vm = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2)
                    + 3.0 * (s12**2 + s13**2 + s23**2)).astype(np.float32)

velocity = np.diff(world_pos, axis=0).astype(np.float32)  # (F-1, N, 3)

# ── Mesh topology ─────────────────────────────────────────────────────────────
def nid_1idx(i, j, k):
    return i * (NY+1) * (NZ+1) + j * (NZ+1) + k + 1

# C3D8 element edges (12 per element, pairs of local node indices 0-7)
C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),  # bottom face
    (4,5),(5,6),(6,7),(7,4),  # top face
    (0,4),(1,5),(2,6),(3,7),  # verticals
]

edge_set   = set()
elem_conns = []
elem_mats  = []

for I in range(NX):
    for J in range(NY):
        for K in range(NZ):
            c1 = [
                nid_1idx(I,   J,   K  ), nid_1idx(I+1, J,   K  ),
                nid_1idx(I+1, J+1, K  ), nid_1idx(I,   J+1, K  ),
                nid_1idx(I,   J,   K+1), nid_1idx(I+1, J,   K+1),
                nid_1idx(I+1, J+1, K+1), nid_1idx(I,   J+1, K+1),
            ]
            c0 = [label_to_idx[lbl] for lbl in c1]   # 0-indexed
            elem_conns.append(c0)
            elem_mats.append(0 if J < SPLIT_J else 1)

            for (a, b) in C3D8_EDGES:
                u, v = c0[a], c0[b]
                edge_set.add((u, v))
                edge_set.add((v, u))

elem_conn_arr = np.array(elem_conns, dtype=np.int32)   # (N_elem, 8)
elem_mat_arr  = np.array(elem_mats,  dtype=np.int32)   # (N_elem,)

edges = np.array(sorted(edge_set), dtype=np.int64).T    # (2, E)

# ── Node type & fixed mask ────────────────────────────────────────────────────
# 0=free interior, 1=clamped (x=0), 2=loaded tip (x=L)
node_type  = np.zeros(N, dtype=np.int64)
fixed_mask = np.zeros(N, dtype=bool)

for j in range(NY+1):
    for k in range(NZ+1):
        clamp_idx = label_to_idx[nid_1idx(0,  j, k)]
        tip_idx   = label_to_idx[nid_1idx(NX, j, k)]
        node_type[clamp_idx]  = 1
        fixed_mask[clamp_idx] = True
        node_type[tip_idx]    = 2

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(npz_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

np.savez(
    npz_path,
    mesh_pos        = mesh_pos,
    world_pos       = world_pos,
    velocity        = velocity,
    stress_vm       = stress_vm,
    mesh_edge_index = edges,
    node_type       = node_type,
    fixed_mask      = fixed_mask,
    elem_conn       = elem_conn_arr,
    n_nodes_per_elem= np.array(8),
    elem_mat        = elem_mat_arr,
    mat_E           = np.array([E1, E2], dtype=np.float64),
    mat_nu          = np.array([nu, nu], dtype=np.float64),
)

print("Saved: {}  N={}  F={}  N_elem={}  E={}".format(
    npz_path, N, F, len(elem_conns), edges.shape[1]))
