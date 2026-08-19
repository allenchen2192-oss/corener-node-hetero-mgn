# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
compare_errors_rect.py
=======================
Rollout error comparison across three groups using epoch-661 checkpoint:
  - Training (in-dist) : sample_00000 .. sample_00009
  - Test     (in-dist) : sample_02000 .. sample_02009
  - OOD (rect)         : sample_00000 .. sample_00009  (04_preprocessed_pt_abaqus_rect)

Metrics (10 samples each, mean ± std):
  Displacement Error (%) = mean_nodes(||pred - gt||) / mean_nodes(||gt_disp||) * 100
  Stress Error       (%) = mean_nodes(|pred_vm - gt_vm|) / mean_nodes(gt_vm) * 100
    - pred_vm: scatter linear fit on predicted positions
    - gt_vm  : directly from Abaqus NPZ stress_vm

Output:
  compare_errors_rect.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 6    # disp_vec(3) + prev_vel(3)
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3    # velocity only
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 64

DATA_DIR      = "./04_preprocessed_pt_abaqus_prevvel"   # cube (train + test)
RECT_DATA_DIR = "./04_preprocessed_pt_abaqus_rect"      # rect OOD

NPZ_DIR       = "./02_abaqus_npz"                       # cube NPZ (for GT stress)
RECT_NPZ_DIR  = "./02_abaqus_npz_rect"                  # rect NPZ (for GT stress)

CKPT_PATH  = "./checkpoints_abaqus_prevvel_1000data_1000nodes"
CKPT_EPOCH = 661   # None = latest (epoch 661)

N_SAMPLES = 10
# First 10 samples that pass max_nodes=1000 filter (same as training loader)
TRAIN_INDICES = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11]
TEST_INDICES  = list(range(2000, 2000 + N_SAMPLES))
OOD_INDICES   = list(range(0, N_SAMPLES))

OUTPUT_PNG = "compare_errors_rect.png"


# ── Material / physics ────────────────────────────────────────────────────────

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


def compute_edge_features(pos, edge_index):
    row, col = edge_index
    d = pos[col] - pos[row]   # dst - src (matches preprocess_abaqus.py)
    n = torch.norm(d, dim=1, keepdim=True)
    return torch.cat([d, n], dim=1)


# ── FEM-consistent stress (C3D8 / C3D10 / C3D20) ────────────────────────────

def _gauss_hex_222():
    g = 1.0 / np.sqrt(3.0)
    pts, wts = [], []
    for zi in [-g, g]:
        for ei in [-g, g]:
            for xi in [-g, g]:
                pts.append([xi, ei, zi]); wts.append(1.0)
    return np.array(pts), np.array(wts)   # (8,3), (8,)

def _gauss_tet4():
    a = (5.0 + 3.0*np.sqrt(5.0)) / 20.0
    b = (5.0 - np.sqrt(5.0))     / 20.0
    pts = np.array([[a,b,b],[b,a,b],[b,b,a],[b,b,b]])
    wts = np.full(4, 1.0/24.0)
    return pts, wts   # (4,3), (4,)

_XN8  = np.array([-1., 1., 1.,-1.,-1., 1., 1.,-1.])
_EN8  = np.array([-1.,-1., 1., 1.,-1.,-1., 1., 1.])
_ZN8  = np.array([-1.,-1.,-1.,-1., 1., 1., 1., 1.])

def _dN_hex8(xi, eta, zeta):
    return np.stack([
        0.125 * _XN8 * (1+_EN8*eta)  * (1+_ZN8*zeta),
        0.125 * _EN8 * (1+_XN8*xi)   * (1+_ZN8*zeta),
        0.125 * _ZN8 * (1+_XN8*xi)   * (1+_EN8*eta),
    ])   # (3, 8)

def _dN_tet10(xi, eta, zeta):
    L4 = 1.0 - xi - eta - zeta
    dN_dxi   = np.array([4*xi-1, 0, 0, -(4*L4-1),
                          4*eta, 0, 4*zeta, 4*(L4-xi), -4*eta, -4*zeta])
    dN_deta  = np.array([0, 4*eta-1, 0, -(4*L4-1),
                          4*xi, 4*zeta, 0, -4*xi, 4*(L4-eta), -4*zeta])
    dN_dzeta = np.array([0, 0, 4*zeta-1, -(4*L4-1),
                          0, 4*eta, 4*xi, -4*xi, -4*eta, 4*(L4-zeta)])
    return np.stack([dN_dxi, dN_deta, dN_dzeta])   # (3, 10)

def _dN_hex20(xi, eta, zeta):
    dN = np.zeros((3, 20))
    for i in range(8):
        x0, e0, z0 = _XN8[i], _EN8[i], _ZN8[i]
        s = x0*xi + e0*eta + z0*zeta - 2
        f1 = (1+e0*eta)*(1+z0*zeta)
        f2 = (1+x0*xi)*(1+z0*zeta)
        f3 = (1+x0*xi)*(1+e0*eta)
        dN[0,i] = 0.125*(x0*f1*s + (1+x0*xi)*f1*x0)
        dN[1,i] = 0.125*(e0*f2*s + (1+e0*eta)*f2*e0)
        dN[2,i] = 0.125*(z0*f3*s + (1+z0*zeta)*f3*z0)
    # xi=0 midside: nodes 8,10,12,14 with (eta0,zeta0)
    for idx, e0, z0 in [(8,-1.,-1.),(10,+1.,-1.),(12,-1.,+1.),(14,+1.,+1.)]:
        dN[0,idx] = 0.25*(-2*xi)*(1+e0*eta)*(1+z0*zeta)
        dN[1,idx] = 0.25*(1-xi*xi)*e0*(1+z0*zeta)
        dN[2,idx] = 0.25*(1-xi*xi)*(1+e0*eta)*z0
    # eta=0 midside: nodes 9,11,13,15 with (xi0,zeta0)
    for idx, x0, z0 in [(9,+1.,-1.),(11,-1.,-1.),(13,+1.,+1.),(15,-1.,+1.)]:
        dN[0,idx] = 0.25*x0*(1-eta*eta)*(1+z0*zeta)
        dN[1,idx] = 0.25*(1+x0*xi)*(-2*eta)*(1+z0*zeta)
        dN[2,idx] = 0.25*(1+x0*xi)*(1-eta*eta)*z0
    # zeta=0 midside: nodes 16,17,18,19 with (xi0,eta0)
    for idx, x0, e0 in [(16,-1.,-1.),(17,+1.,-1.),(18,+1.,+1.),(19,-1.,+1.)]:
        dN[0,idx] = 0.25*x0*(1+e0*eta)*(1-zeta*zeta)
        dN[1,idx] = 0.25*(1+x0*xi)*e0*(1-zeta*zeta)
        dN[2,idx] = 0.25*(1+x0*xi)*(1+e0*eta)*(-2*zeta)
    return dN   # (3, 20)

# ── Gauss-to-node extrapolation matrices ─────────────────────────────────────
# These replicate the Abaqus approach: evaluate the Lagrange basis of the
# Gauss-point field at each node coordinate, rather than averaging GP values.

def _extrap_matrix_hex8():
    """8×8: maps 2×2×2 Gauss-point values → 8 corner-node values (hex8/hex20)."""
    sqrt3 = np.sqrt(3.0)
    g = 1.0 / sqrt3
    # GP coords [xi, eta, zeta] — same loop order as _gauss_hex_222
    gp = np.array([[xi, ei, zi]
                   for zi in [-g, g] for ei in [-g, g] for xi in [-g, g]])  # (8,3)
    # Node coords — same order as _XN8/_EN8/_ZN8
    nd = np.stack([_XN8, _EN8, _ZN8], axis=1).astype(float)  # (8,3)
    # 3-D weight = product of 1-D Lagrange factors across each axis:
    #   L(node_coord_i, gp_coord_j) = (1 + node_i * sign(gp_j) * sqrt3) / 2
    M = np.ones((8, 8))
    for dim in range(3):
        M *= (1.0 + np.outer(nd[:, dim], np.sign(gp[:, dim])) * sqrt3) / 2.0
    return M  # M[node, gp]


def _extrap_matrix_hex20():
    """20×8: corner rows from hex8 matrix; midside rows as adjacent-corner averages."""
    M8  = _extrap_matrix_hex8()
    M20 = np.zeros((20, 8))
    M20[:8] = M8
    # Midside node i lies at the midpoint of two corner nodes (from _dN_hex20 ordering)
    midside_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # xi=0 edges, bottom face  (nodes 8-11)
        (4, 5), (5, 6), (6, 7), (7, 4),   # xi=0 edges, top face     (nodes 12-15)
        (0, 4), (1, 5), (2, 6), (3, 7),   # zeta=0 vertical edges    (nodes 16-19)
    ]
    for mi, (c0, c1) in enumerate(midside_pairs):
        M20[8 + mi] = 0.5 * (M8[c0] + M8[c1])
    return M20  # M[node, gp]


def _extrap_matrix_tet10():
    """10×4: linear-polynomial extrapolation from 4 tet Gauss points → 10 node coords."""
    a = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    b = (5.0 - np.sqrt(5.0)) / 20.0
    gp = np.array([[a, b, b], [b, a, b], [b, b, a], [b, b, b]])  # (4,3)
    # Tet10 node coords in (xi,eta,zeta) — derived from _dN_tet10 shape functions
    nd = np.array([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.],  # corners 0-3
        [.5, .5, 0.], [0., .5, .5], [.5, 0., .5],                 # midside 4-6
        [.5, 0., 0.], [0., .5, 0.], [0., 0., .5],                 # midside 7-9
    ])
    def aug(c): return np.hstack([np.ones((len(c), 1)), c])
    return aug(nd) @ np.linalg.inv(aug(gp))  # (10,4): fit linear field at GPs, eval at nodes


_EXTRAP_HEX8  = _extrap_matrix_hex8()
_EXTRAP_HEX20 = _extrap_matrix_hex20()
_EXTRAP_TET10 = _extrap_matrix_tet10()


def _fem_vm(ref_np, world_np, elem_conn, gauss_pts, dN_fn, extrap_matrix):
    """FEM von Mises (Pa): stress at each Gauss point, extrapolated to nodes."""
    u    = (world_np - ref_np).astype(np.float64)
    ref  = ref_np.astype(np.float64)
    N    = ref.shape[0]
    X_e  = ref[elem_conn]    # (E, npe, 3)
    u_e  = u[elem_conn]      # (E, npe, 3)
    E_el = X_e.shape[0]
    n_gp = len(gauss_pts)

    vm_gp = np.zeros((E_el, n_gp))
    for gi, gp in enumerate(gauss_pts):
        dN   = dN_fn(*gp)                                    # (3, npe)
        J    = np.einsum('ik,ekj->eij', dN, X_e)            # (E, 3, 3)
        Jinv = np.linalg.inv(J)                              # (E, 3, 3)
        dNdx = np.einsum('eij,jk->eik', Jinv, dN)           # (E, 3, npe)
        dudx = np.einsum('eik,ekj->eij', dNdx, u_e)         # (E, 3, 3)
        exx = dudx[:,0,0]; eyy = dudx[:,1,1]; ezz = dudx[:,2,2]
        exy = 0.5*(dudx[:,0,1]+dudx[:,1,0])
        exz = 0.5*(dudx[:,0,2]+dudx[:,2,0])
        eyz = 0.5*(dudx[:,1,2]+dudx[:,2,1])
        tr  = exx+eyy+ezz
        sxx = LAME1*tr+2*MU*exx; syy = LAME1*tr+2*MU*eyy; szz = LAME1*tr+2*MU*ezz
        sxy = 2*MU*exy; sxz = 2*MU*exz; syz = 2*MU*eyz
        vm_gp[:, gi] = np.sqrt(0.5*((sxx-syy)**2+(syy-szz)**2+(szz-sxx)**2
                                     +6*(sxy**2+sxz**2+syz**2))+1e-60)

    # Extrapolate GP values to nodes; clip to keep σ_vm ≥ 0
    vm_node_e = np.clip(vm_gp @ extrap_matrix.T, 0.0, None)  # (E_el, npe)

    vm_sum   = np.zeros(N)
    vm_count = np.zeros(N, dtype=np.int32)
    np.add.at(vm_sum,   elem_conn.flatten(), vm_node_e.flatten())
    np.add.at(vm_count, elem_conn.flatten(), 1)
    return (vm_sum / np.maximum(vm_count, 1)).astype(np.float32)

def compute_vm_stress_fem(world_np, ref_np, elem_conn, npe):
    if npe == 8:
        gp, _ = _gauss_hex_222(); fn = _dN_hex8;  M = _EXTRAP_HEX8
    elif npe == 10:
        gp, _ = _gauss_tet4();    fn = _dN_tet10; M = _EXTRAP_TET10
    elif npe == 20:
        gp, _ = _gauss_hex_222(); fn = _dN_hex20; M = _EXTRAP_HEX20
    else:
        raise ValueError(f"Unsupported npe={npe}")
    return _fem_vm(ref_np, world_np, elem_conn, gp, fn, M)


# ── Global setup ─────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

# Normalization stats — always from cube training data
node_stats  = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean      = node_stats["velocity_mean"].to(device)
v_std       = node_stats["velocity_std"].to(device)
v_mean_cpu  = v_mean.cpu()
v_std_cpu   = v_std.cpu()

edge_stats  = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref  = edge_stats["edge_mean"].to(device)[:4]
e_std_ref   = edge_stats["edge_std"].to(device)[:4]
e_mean_wld  = edge_stats["edge_mean"].to(device)[4:]
e_std_wld   = edge_stats["edge_std"].to(device)[4:]

empty_wef   = torch.zeros(0, NUM_EDGE_FEATURES, device=device)


# ── Model ─────────────────────────────────────────────────────────────────────

model = HybridMeshGraphNet(
    NUM_INPUT_FEATURES, NUM_EDGE_FEATURES, NUM_OUTPUT_FEATURES,
    processor_size=PROCESSOR_SIZE,
    hidden_dim_processor=HIDDEN_DIM,
    hidden_dim_node_encoder=HIDDEN_DIM,
    hidden_dim_edge_encoder=HIDDEN_DIM,
    hidden_dim_node_decoder=HIDDEN_DIM,
    mlp_activation_fn="relu",
    do_concat_trick=False,
    num_processor_checkpoint_segments=0,
    recompute_activation=False,
).to(device)
model.eval()

print(f"Loading checkpoint: {CKPT_PATH}  epoch={CKPT_EPOCH}")
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)


# ── Rollout ───────────────────────────────────────────────────────────────────

@torch.inference_mode()
def rollout_one(pt_path, npz_path):
    """
    Run autoregressive rollout for one sample.
    Seed: GT position at t=1, prev_vel = GT vel at t=0.
    Returns disp_errs (T-1,) and stress_errs (T-1,) arrays.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    T    = len(data)   # = 19

    # Per-sample mesh topology
    ref_pos    = data[0]["graph"].mesh_pos.to(device)
    mesh_ei    = data[0]["graph"].edge_index.to(device)
    ref_np     = ref_pos.cpu().numpy()
    elem_conn  = data[0]["elem_conn"].numpy()
    npe        = int(data[0]["n_nodes_per_elem"])

    ref_feat_n = (compute_edge_features(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref

    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    npz       = np.load(npz_path)
    gt_stress = [npz["stress_vm"][t + 1] for t in range(T)]

    gt_vel_0 = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur      = gt_wp_list[1].to(device)
    prev_vel = gt_vel_0

    disp_errs   = []
    stress_errs = []

    for k in range(T - 1):
        world_feat = (compute_edge_features(cur, mesh_ei) - e_mean_wld) / e_std_wld
        mesh_ef    = torch.cat([ref_feat_n, world_feat], dim=1)
        node_x     = torch.cat([cur - ref_pos, prev_vel], dim=1)
        g          = Data(x=node_x, edge_index=mesh_ei, num_nodes=ref_pos.shape[0])

        out      = model(node_x, mesh_ef, empty_wef, g)
        vel      = out[:, :3] * v_std + v_mean
        prev_vel = vel
        cur      = cur + vel

        pred_np = cur.cpu().numpy()
        gt_np   = gt_wp_list[k + 2].numpy()

        disp_pred = np.linalg.norm(pred_np - gt_np,  axis=1)
        disp_gt   = np.linalg.norm(gt_np   - ref_np, axis=1)
        disp_errs.append(disp_pred.sum() / (disp_gt.sum() + 1e-30) * 100.0)

        pred_vm = compute_vm_stress_fem(pred_np, ref_np, elem_conn, npe)
        gt_vm   = gt_stress[k + 1].astype(np.float32)
        stress_errs.append(
            np.abs(pred_vm - gt_vm).mean() / (gt_vm.mean() + 1e-30) * 100.0
        )

    return np.array(disp_errs), np.array(stress_errs)


@torch.inference_mode()
def baseline_one(pt_path, npz_path):
    """
    Compute scatter linear fit stress on GT positions vs Abaqus GT stress.
    No rollout — uses ground-truth world_pos at every timestep.
    Returns stress_baseline (T-1,): method discrepancy only, no displacement error.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    T    = len(data)

    ref_np    = data[0]["graph"].mesh_pos.numpy()
    elem_conn = data[0]["elem_conn"].numpy()
    npe       = int(data[0]["n_nodes_per_elem"])

    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    npz       = np.load(npz_path)
    gt_stress = [npz["stress_vm"][t + 1] for t in range(T)]

    stress_baseline = []
    for k in range(T - 1):
        gt_wp_np   = gt_wp_list[k + 2].numpy()
        fem_vm     = compute_vm_stress_fem(gt_wp_np, ref_np, elem_conn, npe)
        gt_vm      = gt_stress[k + 1].astype(np.float32)
        stress_baseline.append(
            np.abs(fem_vm - gt_vm).mean() / (gt_vm.mean() + 1e-30) * 100.0
        )

    return np.array(stress_baseline)


def get_files(indices, data_dir, npz_dir, npz_prefix="sample_"):
    all_pt = sorted(
        f for f in os.listdir(data_dir) if f.startswith("sample_") and f.endswith(".pt")
    )
    pairs = []
    for i in indices:
        if i >= len(all_pt):
            print(f"  [skip] index {i} out of range")
            continue
        fname     = all_pt[i]
        sample_id = fname.replace(".pt", "")
        pt_path   = os.path.join(data_dir, fname)
        npz_path  = os.path.join(npz_dir,  sample_id + ".npz")
        if not os.path.exists(npz_path):
            print(f"  [skip] NPZ not found: {npz_path}")
            continue
        pairs.append((pt_path, npz_path))
    return pairs


# ── Evaluate all three groups ─────────────────────────────────────────────────

groups = {
    "Training (in-dist)":    (TRAIN_INDICES, DATA_DIR,      NPZ_DIR),
    "Test (in-dist)":        (TEST_INDICES,  DATA_DIR,      NPZ_DIR),
    "OOD (out-of-dist)":     (OOD_INDICES,   RECT_DATA_DIR, RECT_NPZ_DIR),
}

results = {}
for label, (indices, data_dir, npz_dir) in groups.items():
    files = get_files(indices, data_dir, npz_dir)
    print(f"\n[{label}] {len(files)} samples")
    disp_list, stress_list = [], []
    for pt_path, npz_path in files:
        d, s = rollout_one(pt_path, npz_path)
        disp_list.append(d)
        stress_list.append(s)
        print(f"  {os.path.basename(pt_path)}: final disp={d[-1]:.2f}%  stress={s[-1]:.2f}%")
    results[label] = {
        "disp":   np.stack(disp_list,   axis=0),   # (N_SAMPLES, T-1)
        "stress": np.stack(stress_list, axis=0),
    }
    print(f"  → mean final: disp={results[label]['disp'][:,-1].mean():.2f}%  "
          f"stress={results[label]['stress'][:,-1].mean():.2f}%")

# ── Scatter fit baseline (GT pos, in-dist) ────────────────────────────────────
print("\n[Scatter Fit Baseline] GT positions, in-dist samples")
baseline_files = get_files(TRAIN_INDICES, DATA_DIR, NPZ_DIR)
baseline_list  = []
for pt_path, npz_path in baseline_files:
    b = baseline_one(pt_path, npz_path)
    baseline_list.append(b)
    print(f"  {os.path.basename(pt_path)}: mean={b.mean():.2f}%  final={b[-1]:.2f}%")
baseline_arr = np.stack(baseline_list, axis=0)   # (N_SAMPLES, T-1)
print(f"  → overall mean: {baseline_arr.mean():.2f}%")


# ── Plot ──────────────────────────────────────────────────────────────────────

# t_axis: t=2..T  (T-1 prediction steps, seed was at t=1)
T_steps = results["Training (in-dist)"]["disp"].shape[1]
t_axis  = np.arange(2, T_steps + 2)

colors = {
    "Training (in-dist)": "tab:blue",
    "Test (in-dist)":     "tab:green",
    "OOD (out-of-dist)":  "tab:red",
}
labels_short = {
    "Training (in-dist)": "Training (in-dist)",
    "Test (in-dist)":     "Test (in-dist)",
    "OOD (out-of-dist)":  "OOD (out-of-dist)",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Rollout Error Comparison (mean ± std over {N_SAMPLES} samples)", fontsize=16)

for label in ["Training (in-dist)", "Test (in-dist)", "OOD (out-of-dist)"]:
    d   = results[label]["disp"]
    s   = results[label]["stress"]
    c   = colors[label]
    lbl = labels_short[label]

    d_mean, d_std = d.mean(axis=0), d.std(axis=0)
    s_mean, s_std = s.mean(axis=0), s.std(axis=0)

    axes[0].plot(t_axis, d_mean, color=c, linewidth=2, label=lbl)
    axes[0].fill_between(t_axis, d_mean - d_std, d_mean + d_std, color=c, alpha=0.2)

    axes[1].plot(t_axis, s_mean, color=c, linewidth=2, label=lbl)
    axes[1].fill_between(t_axis, s_mean - s_std, s_mean + s_std, color=c, alpha=0.2)


for ax, title, ylabel in [
    (axes[0], "Displacement Error (%)", "Displacement Error (%)"),
    (axes[1], "Stress Error (%)",       "Stress Error (%)"),
]:
    ax.set_xlabel("Time step t", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_ylim(0, 50)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
print(f"\n[saved] {OUTPUT_PNG}")
