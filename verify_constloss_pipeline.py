"""
verify_constloss_pipeline.py
============================
Verifies that σ_const computed from GT displacement (using the SAME B-matrices
and pipeline as train_m0035_constloss.py) matches Abaqus GT stress.

Mirrors _constitutive_loss() exactly, but substitutes GT velocity for predicted.

Usage:
    python verify_constloss_pipeline.py
"""

import os
import json
import torch
import numpy as np
from glob import glob

PT_DIR    = "./04_preprocessed_pt_m0035"
TOPO_DIR  = "./04_preprocessed_pt_m0035_constloss"
STATS_FILE = os.path.join(PT_DIR, "node_stats_m0035.json")
N_SAMPLES = 5    # sample IDs to check
N_FRAMES  = 5    # frames per sample (0 = all)

# ── Normalization stats ──────────────────────────────────────────────────────

with open(STATS_FILE) as f:
    ns = json.load(f)

v_mean         = torch.tensor(ns["vel_mean"],       dtype=torch.float32)
v_std          = torch.tensor(ns["vel_std"],        dtype=torch.float32)
stress_mean_Si = torch.tensor(ns["stress_mean_Si"], dtype=torch.float32)
stress_std_Si  = torch.tensor(ns["stress_std_Si"],  dtype=torch.float32)
stress_mean_UF = torch.tensor(ns["stress_mean_UF"], dtype=torch.float32)
stress_std_UF  = torch.tensor(ns["stress_std_UF"],  dtype=torch.float32)

MAT_NAMES = {0: "Si_bot", 1: "Si_SoC", 3: "UF"}
LABELS    = ["S11", "S22", "S33", "S12", "S13", "S23"]

# ── Core computation (mirrors _constitutive_loss) ────────────────────────────

def compute_sigma_const(graph, topo):
    """
    Compute constitutive stress from GT displacement.
    Returns sig_const (N,6) [MPa], has_weight (N,), node_mat (N,),
            si_mask (N,), uf_mask (N,).
    """
    # GT displacement at *next* frame: u = world_pos_current + vel_gt - mesh_pos
    vel_gt_phys = graph.y[:, :3] * v_std + v_mean       # (N, 3)
    u_next = graph.world_pos + vel_gt_phys - graph.mesh_pos  # (N, 3)

    ec    = topo["elem_conn"].long()    # (E, 8)
    em    = topo["elem_mat"].long()     # (E,)
    B     = topo["elem_B"].float()      # (E, 3, 8)
    vol   = topo["elem_vol"].float()    # (E,)
    E_e   = topo["elem_E"].float()      # (E,)
    nu_e  = topo["elem_nu"].float()     # (E,)
    cte_e = topo["elem_CTE"].float()    # (E,)

    # delta_T: stored as scalar or 1-D tensor; handle both
    dT_raw = graph.delta_T_next
    dT = float(dT_raw.item() if hasattr(dT_raw, "item") else dT_raw)

    # Elastic elements only (skip Solder = mat 2)
    el       = em != 2
    ec_el    = ec[el];   B_el  = B[el];   vol_el = vol[el]
    E_el     = E_e[el];  nu_el = nu_e[el]; cte_el = cte_e[el]; em_el = em[el]
    n_el     = el.sum().item()

    dT_el = torch.full((n_el,), dT)

    # Gather node displacements (E_el, 8, 3)
    u_elem = u_next[ec_el].float()       # (E_el, 8, 3)

    # Displacement gradient: grad[α,β] = dNα·uβ  →  (E_el, 3, 3)
    grad = torch.bmm(B_el, u_elem)

    # Mechanical strains (subtract thermal diagonal)
    eps_th = cte_el * dT_el              # (E_el,)
    e11 = grad[:, 0, 0] - eps_th
    e22 = grad[:, 1, 1] - eps_th
    e33 = grad[:, 2, 2] - eps_th
    g12 = grad[:, 1, 0] + grad[:, 0, 1]   # engineering shear
    g13 = grad[:, 2, 0] + grad[:, 0, 2]
    g23 = grad[:, 2, 1] + grad[:, 1, 2]

    # Hooke's law  σ = λ·tr(ε)·I + 2μ·ε
    lam = E_el * nu_el / ((1 + nu_el) * (1 - 2 * nu_el))
    mu  = E_el / (2 * (1 + nu_el))
    tr  = e11 + e22 + e33

    sig_el = torch.stack([
        lam * tr + 2 * mu * e11,
        lam * tr + 2 * mu * e22,
        lam * tr + 2 * mu * e33,
        mu * g12, mu * g13, mu * g23,
    ], dim=1)   # (E_el, 6)  MPa

    # Material-matched volume-weighted scatter to nodes
    node_mat   = graph.node_mat.long()   # (N,)
    nm_el      = node_mat[ec_el]         # (E_el, 8)
    matched    = (nm_el == em_el.unsqueeze(1)).float()   # (E_el, 8)

    ec_flat      = ec_el.reshape(-1)                                         # (E_el*8,)
    matched_flat = matched.reshape(-1)                                        # (E_el*8,)
    vol_rep      = vol_el.unsqueeze(1).expand(-1, 8).reshape(-1) * matched_flat
    vw_rep       = (vol_el[:, None] * sig_el).unsqueeze(1)\
                   .expand(-1, 8, -1).reshape(-1, 6) \
                   * matched_flat.unsqueeze(-1)

    N_nodes  = u_next.shape[0]
    sig_num  = torch.zeros(N_nodes, 6)
    vol_sum  = torch.zeros(N_nodes)
    sig_num.scatter_add_(0, ec_flat.unsqueeze(-1).expand(-1, 6), vw_rep)
    vol_sum.scatter_add_(0, ec_flat, vol_rep)

    has_weight = vol_sum > 0
    sig_const  = torch.zeros(N_nodes, 6)
    sig_const[has_weight] = sig_num[has_weight] / vol_sum[has_weight, None]

    si_mask = (node_mat == 0) | (node_mat == 1)
    uf_mask = (node_mat == 3)
    return sig_const, has_weight, node_mat, si_mask, uf_mask


# ── Main loop ────────────────────────────────────────────────────────────────

# Each S####.pt is a list of dicts, one per timestep (same as training Dataset)
pt_files = sorted(glob(os.path.join(PT_DIR, "S*.pt")))
# Exclude non-sample files (stats json etc.)
pt_files = [f for f in pt_files if os.path.basename(f).startswith("S")
            and os.path.basename(f).endswith(".pt")]
all_sids = sorted(set(os.path.splitext(os.path.basename(f))[0] for f in pt_files))
sids     = all_sids[:N_SAMPLES]

# Accumulators: per-material abs error tensors, and per-component
abs_errs  = {k: [] for k in [0, 1, 3]}  # mat_id → list of (N_nodes, 6) tensors
comp_errs = []   # (n_nodes, 6) across all samples/frames for combined stats

print(f"{'='*65}")
print(f"Pipeline verification: σ_const(u_GT) vs GT stress")
print(f"  PT dir   : {PT_DIR}")
print(f"  Topo dir : {TOPO_DIR}")
print(f"  Samples  : {sids}")
print(f"{'='*65}\n")

for sid in sids:
    topo_path = os.path.join(TOPO_DIR, f"{sid}_topo.pt")
    if not os.path.exists(topo_path):
        print(f"  [SKIP] {sid} — no _topo.pt found")
        continue
    topo = torch.load(topo_path, map_location="cpu", weights_only=False)

    pt_path = os.path.join(PT_DIR, f"{sid}.pt")
    sample_data = torch.load(pt_path, map_location="cpu", weights_only=False)
    # Filter to step_index > 0 (same as Dataset) and limit to N_FRAMES
    items = [item for item in sample_data if int(item["graph"].step_index.item()) > 0]
    if N_FRAMES > 0:
        items = items[:N_FRAMES]
    print(f"  Sample {sid}  ({len(items)} frames)")

    for item in items:
        graph = item["graph"]
        fi    = int(graph.step_index.item())

        sig_const, has_w, node_mat, si_mask, uf_mask = compute_sigma_const(graph, topo)

        # GT stress in physical MPa (de-normalize from y[:, 3:9])
        y = graph.y
        stress_gt   = torch.zeros(y.shape[0], 6)
        stress_gt_n = y[:, 3:9]
        if si_mask.any():
            stress_gt[si_mask] = stress_gt_n[si_mask] * stress_std_Si + stress_mean_Si
        if uf_mask.any():
            stress_gt[uf_mask] = stress_gt_n[uf_mask] * stress_std_UF + stress_mean_UF

        cmp_mask = has_w & (si_mask | uf_mask)
        if not cmp_mask.any():
            print(f"    step {fi:2d}: no comparable nodes")
            continue

        dT_raw = graph.delta_T_next
        dT = float(dT_raw.item() if hasattr(dT_raw, "item") else dT_raw)

        print(f"    step {fi:2d}  ΔT={dT:+6.1f}°C  elastic+weighted nodes={has_w.sum().item()}")

        # Per-material error
        for mat_id in [0, 1, 3]:
            mask = cmp_mask & (node_mat == mat_id)
            if not mask.any():
                continue
            err = (sig_const[mask] - stress_gt[mask]).abs()   # (n, 6)
            abs_errs[mat_id].append(err)

        err_all = (sig_const[cmp_mask] - stress_gt[cmp_mask]).abs()
        comp_errs.append(err_all)

# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print("PER-MATERIAL SUMMARY  (all samples × frames, abs error in MPa)")
print(f"{'='*65}")
print(f"  {'Material':<10}  {'abs_mean':>10}  {'abs_max':>10}  {'n_nodes':>8}")
for mat_id in [0, 1, 3]:
    if not abs_errs[mat_id]:
        print(f"  {MAT_NAMES[mat_id]:<10}  — no data —")
        continue
    errs = torch.cat(abs_errs[mat_id], dim=0)   # (N_total, 6)
    print(f"  {MAT_NAMES[mat_id]:<10}  {errs.mean().item():>10.4f}  {errs.max().item():>10.4f}  {errs.shape[0]:>8}")

print(f"\n{'='*65}")
print("PER-COMPONENT SUMMARY  (Si + UF combined)")
print(f"{'='*65}")
if comp_errs:
    errs_all = torch.cat(comp_errs, dim=0)   # (N_total, 6)
    print(f"  {'Comp':<6}  {'abs_mean':>10}  {'abs_max':>10}")
    for c, lbl in enumerate(LABELS):
        print(f"  {lbl:<6}  {errs_all[:, c].mean().item():>10.4f}  {errs_all[:, c].max().item():>10.4f}")
else:
    print("  No data.")

print(f"\nDone.")