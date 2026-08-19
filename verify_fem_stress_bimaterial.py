"""
verify_fem_stress_bimaterial.py
================================
Validate fem_stress.py B-matrix implementation against Abaqus GT:
  1. Load GT world_pos from NPZ
  2. Compute stress via compute_nodal_vm_stress (B-matrix, per-element E)
  3. Compare with Abaqus GT stress_vm

Expected: R² ≈ 1.0, relative error < 5%  (small gap from reduced integration vs full)
"""

import numpy as np
import torch
from fem_stress import compute_nodal_vm_stress

NPZ_DIR  = "./02_abaqus_npz_bimaterial"
N_SAMPLES = 3   # number of samples to check
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

import os
files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))[:N_SAMPLES]

for fname in files:
    npz = np.load(os.path.join(NPZ_DIR, fname))

    ref_pos   = torch.from_numpy(npz["mesh_pos"]).to(DEVICE)          # (N, 3)
    world_pos = torch.from_numpy(npz["world_pos"]).to(DEVICE)         # (T, N, 3)
    stress_gt = torch.from_numpy(npz["stress_vm"]).to(DEVICE)         # (T, N)
    elem_conn = torch.from_numpy(npz["elem_conn"].astype(np.int64)).to(DEVICE)  # (E, 8)
    elem_E    = torch.from_numpy(npz["mat_E"][npz["elem_mat"]]).to(DEVICE)      # (E,)
    elem_nu   = torch.from_numpy(npz["mat_nu"][npz["elem_mat"]]).to(DEVICE)     # (E,)

    T = world_pos.shape[0]
    print(f"\n{'='*60}")
    print(f"Sample: {fname}  |  N={ref_pos.shape[0]}  E={elem_conn.shape[0]}  T={T}")
    print(f"mat_E={npz['mat_E']}  mat_nu={npz['mat_nu']}")
    print(f"{'t':>4}  {'GT mean':>10}  {'BM mean':>10}  {'rel err%':>9}  {'R2':>8}  {'max err%':>9}")
    print("-" * 60)

    for t in range(T):
        disp = world_pos[t] - ref_pos          # (N, 3)
        gt   = stress_gt[t]                    # (N,)

        with torch.no_grad():
            bm = compute_nodal_vm_stress(
                disp, ref_pos, elem_conn, npe=8,
                elem_E=elem_E, elem_nu=elem_nu,
            )                                  # (N,)

        gt_np = gt.cpu().float().numpy()
        bm_np = bm.cpu().float().numpy()

        # Skip t=0 (all zeros)
        if gt_np.mean() < 1e-6:
            print(f"  {t:>2}  [skip t=0, all zero]")
            continue

        rel_err = np.abs(bm_np - gt_np).mean() / (gt_np.mean() + 1e-10) * 100
        max_err = np.abs(bm_np - gt_np).max()  / (gt_np.max()  + 1e-10) * 100

        ss_res = ((gt_np - bm_np)**2).sum()
        ss_tot = ((gt_np - gt_np.mean())**2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        print(f"  {t:>2}  {gt_np.mean():>10.2f}  {bm_np.mean():>10.2f}"
              f"  {rel_err:>8.2f}%  {r2:>8.4f}  {max_err:>8.2f}%")

