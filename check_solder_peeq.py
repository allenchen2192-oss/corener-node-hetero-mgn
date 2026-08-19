"""
check_solder_peeq.py
====================
Check whether microbump (Solder, mat=2) nodes undergo plastic deformation
across all samples and frames in the M0035 dataset.

Usage:
    python check_solder_peeq.py
"""

import os
import numpy as np
from glob import glob

NPZ_DIR  = "./02_abaqus_npz_m0035"
N_REPORT = 5   # detailed per-sample report for first N samples

npz_files = sorted(glob(os.path.join(NPZ_DIR, "S*.npz")))
print(f"Found {len(npz_files)} samples\n")

# Global accumulators across all samples
global_peeq_max   = []   # max PEEQ per sample (at final frame)
global_any_plastic = 0   # samples where any solder node has PEEQ > 0 at final frame
global_n_samples   = 0

for i, fpath in enumerate(npz_files):
    sid = os.path.splitext(os.path.basename(fpath))[0]
    d   = np.load(fpath, allow_pickle=True)

    node_mat = d["node_mat"].astype(np.int32)   # (N,)
    peeq     = d["peeq"]                         # (T, N) or (T, N, 1)
    if peeq.ndim == 3:
        peeq = peeq[:, :, 0]

    solder_mask = (node_mat == 2)
    n_solder    = solder_mask.sum()
    T           = peeq.shape[0]

    if n_solder == 0:
        print(f"  {sid}: no Solder nodes")
        continue

    peeq_solder = peeq[:, solder_mask]   # (T, n_solder)

    # Final frame stats
    peeq_final     = peeq_solder[-1]
    peeq_max_final = peeq_final.max()
    peeq_mean_final= peeq_final.mean()
    n_plastic_final= (peeq_final > 0).sum()

    global_peeq_max.append(peeq_max_final)
    if peeq_max_final > 0:
        global_any_plastic += 1
    global_n_samples += 1

    # Detailed report for first N_REPORT samples
    if i < N_REPORT:
        print(f"  {sid}  n_solder={n_solder}  frames={T}")
        print(f"  {'Frame':>6}  {'PEEQ_max':>12}  {'PEEQ_mean':>12}  {'n_nodes > 0':>12}")
        for fi in range(T):
            pm  = peeq_solder[fi].max()
            pmn = peeq_solder[fi].mean()
            np_ = (peeq_solder[fi] > 0).sum()
            print(f"  {fi:>6}  {pm:>12.4e}  {pmn:>12.4e}  {np_:>12d}")
        print()

# Global summary
global_peeq_max = np.array(global_peeq_max)
print("=" * 55)
print(f"GLOBAL SUMMARY (all {global_n_samples} samples, final frame)")
print("=" * 55)
print(f"  Samples with any solder PEEQ > 0 : {global_any_plastic} / {global_n_samples}")
print(f"  PEEQ_max across all samples       : {global_peeq_max.max():.4e}")
print(f"  PEEQ_max mean across samples      : {global_peeq_max.mean():.4e}")
print(f"  PEEQ_max median across samples    : {np.median(global_peeq_max):.4e}")
print(f"  Samples with PEEQ_max > 1e-6      : {(global_peeq_max > 1e-6).sum()}")
print(f"  Samples with PEEQ_max > 1e-3      : {(global_peeq_max > 1e-3).sum()}")
print(f"  Samples with PEEQ_max > 0.01      : {(global_peeq_max > 0.01).sum()}")
print("\nDone.")