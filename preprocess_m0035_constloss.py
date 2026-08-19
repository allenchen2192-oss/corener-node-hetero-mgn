"""
preprocess_m0035_constloss.py
=============================
Generate per-sample topology files for the constitutive consistency loss.

For each S{id}.npz, writes S{id}_topo.pt to ./04_preprocessed_pt_m0035_constloss/
containing:
  elem_conn  (E, 8)    int64   - 0-indexed element connectivity
  elem_mat   (E,)      int64   - material index (0=Si_bot, 1=Si_SoC, 2=Solder, 3=UF)
  elem_B     (E, 3, 8) float32 - physical shape-fn gradients at center IP  (J^{-1} @ dN/dxi)
  elem_vol   (E,)      float32 - element volume |det(J)| * 8  [mm^3]
  elem_E     (E,)      float32 - Young's modulus per element  [MPa]
  elem_nu    (E,)      float32 - Poisson's ratio per element
  elem_CTE   (E,)      float32 - CTE per element              [1/°C]

Usage:
    python preprocess_m0035_constloss.py           # skip existing
    python preprocess_m0035_constloss.py --force   # overwrite
"""

import os
import sys

import numpy as np
import torch
from tqdm import tqdm

INPUT_DIR  = "./02_abaqus_npz_m0035"
OUTPUT_DIR = "./04_preprocessed_pt_m0035_constloss"
FORCE      = "--force" in sys.argv

# Isoparametric shape-function derivatives at center IP (xi=eta=zeta=0)
# dN_i/dxi_alpha = XI[i]/8 etc.
XI   = np.array([-1,  1,  1, -1, -1,  1,  1, -1], dtype=np.float64)
ETA  = np.array([-1, -1,  1,  1, -1, -1,  1,  1], dtype=np.float64)
ZETA = np.array([-1, -1, -1, -1,  1,  1,  1,  1], dtype=np.float64)
PARAM = np.stack([XI, ETA, ZETA], axis=0)   # (3, 8)  — divide by 8 when computing J

SI_E   = 130000.0   # MPa
SI_NU  = 0.28
SI_CTE = 3.1e-6     # /°C


def compute_topo(elem_conn, elem_mat, mesh_pos, node_mat, node_E, node_nu, node_CTE,
                 E_uf):
    """
    Compute per-element B-matrices, volumes, and material properties.
    Returns dict of numpy arrays.
    """
    n_elem  = elem_conn.shape[0]
    elem_B   = np.empty((n_elem, 3, 8), dtype=np.float32)
    elem_vol = np.empty(n_elem,          dtype=np.float32)
    elem_E_  = np.empty(n_elem,          dtype=np.float32)
    elem_nu_ = np.empty(n_elem,          dtype=np.float32)
    elem_CTE_= np.empty(n_elem,          dtype=np.float32)

    for e in range(n_elem):
        conn = elem_conn[e]
        mat  = int(elem_mat[e])
        xyz  = mesh_pos[conn].astype(np.float64)           # (8, 3)

        J    = (PARAM / 8.0) @ xyz                         # (3, 3)
        dN   = np.linalg.solve(J, PARAM / 8.0)            # (3, 8)  J^{-1} @ dNdxi
        V_e  = 8.0 * abs(np.linalg.det(J))

        elem_B[e]   = dN.astype(np.float32)
        elem_vol[e] = float(V_e)

        # Material properties per element
        if mat == 0 or mat == 1:                            # Si
            elem_E_[e]  = SI_E
            elem_nu_[e] = SI_NU
            elem_CTE_[e]= SI_CTE
        elif mat == 3:                                      # UF: find a UF node
            elem_E_[e]  = E_uf
            for i in conn:
                if node_mat[i] == 3:
                    elem_nu_[e]  = float(node_nu[i])
                    elem_CTE_[e] = float(node_CTE[i])
                    break
        else:                                               # Solder (mat=2)
            n0 = int(conn[0])
            elem_E_[e]  = float(node_E[n0])
            elem_nu_[e] = float(node_nu[n0])
            elem_CTE_[e]= float(node_CTE[n0])

    return {
        "elem_conn": torch.from_numpy(elem_conn.astype(np.int64)),
        "elem_mat":  torch.from_numpy(elem_mat.astype(np.int64)),
        "elem_B":    torch.from_numpy(elem_B),
        "elem_vol":  torch.from_numpy(elem_vol),
        "elem_E":    torch.from_numpy(elem_E_),
        "elem_nu":   torch.from_numpy(elem_nu_),
        "elem_CTE":  torch.from_numpy(elem_CTE_),
    }


os.makedirs(OUTPUT_DIR, exist_ok=True)

npz_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".npz"))
print(f"Found {len(npz_files)} NPZ files in {INPUT_DIR}")

for fname in tqdm(npz_files):
    sid      = fname[:-4]   # strip .npz
    out_path = os.path.join(OUTPUT_DIR, f"{sid}_topo.pt")
    if os.path.exists(out_path) and not FORCE:
        continue

    d = np.load(os.path.join(INPUT_DIR, fname))
    topo = compute_topo(
        d["elem_conn"].astype(np.int32),
        d["elem_mat"].astype(np.int32),
        d["mesh_pos"].astype(np.float64),
        d["node_mat"].astype(np.int32),
        d["node_E"].astype(np.float64),
        d["node_nu"].astype(np.float64),
        d["node_CTE"].astype(np.float64),
        float(d["E_uf_MPa"]),
    )
    torch.save(topo, out_path)

print(f"\nDone. Topology files saved to {OUTPUT_DIR}/")
n_done = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith("_topo.pt")])
print(f"  {n_done} files written.")