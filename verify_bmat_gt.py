"""
verify_bmat_gt.py
=================
Verify that B-matrix(GT displacement) ≈ GT stress (Abaqus) for Si and UF nodes.

Loads ground-truth world_pos from NPZ, computes B-matrix stress,
and compares against Abaqus stress field.

Usage:
    python verify_bmat_gt.py
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_DIR   = "./02_abaqus_npz_m0035"
TOPO_DIR  = "./04_preprocessed_pt_m0035_constloss"
SAMPLE_IDS = [f"S{i:04d}" for i in range(1, 6)]   # S0001–S0005

T_REF    = 250.0
T_FINAL  = 20.0
N_FRAMES = 21

def dT_at_frame(t):
    return (T_FINAL - T_REF) / (N_FRAMES - 1) * t


# ── B-matrix stress (copied from scatter_bmat_h256.py) ───────────────────────

def bmat_stress_np(u_next, topo, node_mat, dT):
    N   = u_next.shape[0]
    ec  = np.asarray(topo["elem_conn"])
    em  = np.asarray(topo["elem_mat"])
    B   = np.asarray(topo["elem_B"])
    vol = np.asarray(topo["elem_vol"])
    E_e = np.asarray(topo["elem_E"])
    nu_e= np.asarray(topo["elem_nu"])
    cte = np.asarray(topo["elem_CTE"])

    el = em != 2
    if not el.any():
        return np.zeros((N, 6), dtype=np.float32)

    ec_el  = ec[el];  B_el  = B[el].astype(np.float64)
    vol_el = vol[el]; E_el  = E_e[el].astype(np.float64)
    nu_el  = nu_e[el].astype(np.float64)
    cte_el = cte[el].astype(np.float64)
    em_el  = em[el]

    u_elem = u_next[ec_el].astype(np.float64)
    grad   = np.matmul(B_el, u_elem)

    eps_th = cte_el * dT
    e11 = grad[:, 0, 0] - eps_th
    e22 = grad[:, 1, 1] - eps_th
    e33 = grad[:, 2, 2] - eps_th
    g12 = grad[:, 1, 0] + grad[:, 0, 1]
    g13 = grad[:, 2, 0] + grad[:, 0, 2]
    g23 = grad[:, 2, 1] + grad[:, 1, 2]

    lam = E_el * nu_el / ((1 + nu_el) * (1 - 2*nu_el))
    mu  = E_el / (2 * (1 + nu_el))
    tr  = e11 + e22 + e33

    sig_el = np.stack([
        lam*tr + 2*mu*e11,
        lam*tr + 2*mu*e22,
        lam*tr + 2*mu*e33,
        mu*g12, mu*g13, mu*g23,
    ], axis=1).astype(np.float32)

    node_mat_el = node_mat[ec_el]
    matched     = (node_mat_el == em_el[:, None]).astype(np.float32)

    ec_flat  = ec_el.reshape(-1)
    vol_rep  = (vol_el[:, None] * matched).reshape(-1)
    vw_rep   = (vol_el[:, None, None] * sig_el[:, None, :] * matched[:, :, None]).reshape(-1, 6)

    sig_num = np.zeros((N, 6), dtype=np.float32)
    vol_sum = np.zeros(N,      dtype=np.float32)
    np.add.at(sig_num, ec_flat, vw_rep)
    np.add.at(vol_sum, ec_flat, vol_rep)

    sig_out = np.zeros((N, 6), dtype=np.float32)
    has_w = vol_sum > 0
    sig_out[has_w] = sig_num[has_w] / vol_sum[has_w, None]
    return sig_out


def von_mises(s):
    xx, yy, zz = s[..., 0], s[..., 1], s[..., 2]
    xy, xz, yz = s[..., 3], s[..., 4], s[..., 5]
    return np.sqrt(np.maximum(
        0.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2 + 6*(xy**2+xz**2+yz**2)), 0.0))


def r2(pred, gt):
    ss_res = np.sum((gt - pred)**2)
    ss_tot = np.sum((gt - gt.mean())**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def mae(pred, gt):
    return np.mean(np.abs(pred - gt))


# ── Main ──────────────────────────────────────────────────────────────────────

vm_bmat_si, vm_gt_si = [], []
vm_bmat_uf, vm_gt_uf = [], []

for sid in SAMPLE_IDS:
    npz_path  = os.path.join(NPZ_DIR,  f"{sid}.npz")
    topo_path = os.path.join(TOPO_DIR, f"{sid}_topo.pt")
    if not os.path.exists(npz_path) or not os.path.exists(topo_path):
        print(f"  MISSING: {sid}"); continue

    d         = np.load(npz_path, allow_pickle=True)
    mesh_pos  = d["mesh_pos"].astype(np.float32)
    world_pos = d["world_pos"].astype(np.float32)
    stress_gt = d["stress"].astype(np.float32)   # (T, N, 6) MPa
    node_mat  = d["node_mat"].astype(np.int32)

    si_mask = (node_mat == 0) | (node_mat == 1)
    uf_mask =  node_mat == 3

    topo = torch.load(topo_path, map_location="cpu", weights_only=False)

    T = world_pos.shape[0]
    for t in range(1, T):   # frames 1..20
        u_gt  = world_pos[t] - mesh_pos        # GT displacement (m)
        dT    = dT_at_frame(t)
        sig_b = bmat_stress_np(u_gt, topo, node_mat, dT)   # (N, 6) MPa

        vm_bmat_si.append(von_mises(sig_b[si_mask]))
        vm_gt_si.append(  von_mises(stress_gt[t][si_mask]))
        vm_bmat_uf.append(von_mises(sig_b[uf_mask]))
        vm_gt_uf.append(  von_mises(stress_gt[t][uf_mask]))

    print(f"  {sid} done")

vm_bmat_si = np.concatenate(vm_bmat_si)
vm_gt_si   = np.concatenate(vm_gt_si)
vm_bmat_uf = np.concatenate(vm_bmat_uf)
vm_gt_uf   = np.concatenate(vm_gt_uf)

print(f"\n{'':6s}  {'R²':>8}  {'MAE (MPa)':>12}")
print("-" * 32)
print(f"{'Si':6s}  {r2(vm_bmat_si, vm_gt_si):8.4f}  {mae(vm_bmat_si, vm_gt_si):12.4f}")
print(f"{'UF':6s}  {r2(vm_bmat_uf, vm_gt_uf):8.4f}  {mae(vm_bmat_uf, vm_gt_uf):12.4f}")

# ── Scatter plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("B-matrix(GT disp) vs Abaqus GT stress  [S0001–S0005, frames 1–20]",
             fontsize=12, fontweight="bold")

for ax, bmat, gt, label, color in [
    (axes[0], vm_bmat_si, vm_gt_si, "Si",  "#C0392B"),
    (axes[1], vm_bmat_uf, vm_gt_uf, "UF",  "#8E44AD"),
]:
    n = len(bmat)
    idx = np.random.default_rng(0).choice(n, min(n, 30000), replace=False)
    ax.scatter(gt[idx], bmat[idx], s=1, alpha=0.3, color=color, rasterized=True)
    vmin = min(gt.min(), bmat.min()); vmax = max(gt.max(), bmat.max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    ax.set_xlabel("GT Abaqus VM Stress (MPa)"); ax.set_ylabel("B-matrix VM Stress (MPa)")
    ax.set_title(f"{label}  R²={r2(bmat,gt):.4f}  MAE={mae(bmat,gt):.3f} MPa",
                 fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")

plt.tight_layout()
plt.savefig("./verify_bmat_gt.png", dpi=150, bbox_inches="tight")
print("\nSaved → ./verify_bmat_gt.png")
