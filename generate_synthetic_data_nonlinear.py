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
generate_synthetic_data_nonlinear.py
=====================================
Generate synthetic training data with a quadratic displacement field,
producing spatially varying (per-node) von Mises stress.

Displacement field:
  u_i(x,y,z,t) = λ(t)/H * φ(x,y,z) @ a_i
  φ = [1, x, y, z, x², y², z², xy, xz, yz]   (10 terms)
  A ∈ R^(3×10), entries ~ Uniform[-coeff_range, coeff_range]

Key optimisation:
  dphi derivatives depend only on ref_pos (constant across samples/timesteps)
  → precomputed once; per-sample cost is matrix multiplies only.
"""

import math
import os
import time

import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import save_json


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_mesh_edge_index(n: int) -> torch.Tensor:
    edges = []
    def nid(i, j, k): return k*n*n + j*n + i
    for k in range(n):
        for j in range(n):
            for i in range(n):
                s = nid(i, j, k)
                if i+1 < n: t = nid(i+1,j,k); edges.extend([[s,t],[t,s]])
                if j+1 < n: t = nid(i,j+1,k); edges.extend([[s,t],[t,s]])
                if k+1 < n: t = nid(i,j,k+1); edges.extend([[s,t],[t,s]])
    return torch.tensor(edges, dtype=torch.long).T


def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed constants (depend only on ref_pos, computed once)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_geometry(ref_pos: torch.Tensor, mesh_ei: torch.Tensor,
                        e_mean: torch.Tensor, e_std: torch.Tensor):
    """
    Returns tensors that are constant across all samples and timesteps.
      phi_ref  : (N, 10)   polynomial features at ref nodes
      ref_norm : (E, 4)    normalised reference edge features
      r_ij     : (E, 3)    reference relative positions (col - row)
      row, col : (E,)      edge endpoint indices
      XtX_inv  : (N, 3, 3) precomputed inverse of scatter normal matrix
                            — same formula as train.py compute_vm_stress_per_node,
                            so GT stress uses the identical computation
    """
    x, y, z = ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2]
    N  = ref_pos.shape[0]
    z0 = torch.zeros(N)
    z1 = torch.ones(N)

    phi_ref = torch.stack([z1, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z], dim=1)  # (N,10)

    ref_feat = compute_edge_features(ref_pos, mesh_ei)           # (E, 4)
    ref_norm = (ref_feat - e_mean) / e_std                       # (E, 4)

    # Scatter normal matrix (identical to train.py, constant across all samples/timesteps)
    row, col = mesh_ei
    r_ij = ref_pos[col] - ref_pos[row]                           # (E, 3)
    r_r  = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)              # (E, 3, 3)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    XtX = torch.zeros(N, 3, 3)
    XtX.scatter_add_(0, row_exp, r_r)
    XtX = XtX + 1e-6 * torch.eye(3).unsqueeze(0)
    XtX_inv = torch.linalg.inv(XtX)                             # (N, 3, 3)

    return phi_ref, ref_norm, r_ij, row, col, XtX_inv


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample vectorised computation (over all timesteps at once)
# ─────────────────────────────────────────────────────────────────────────────

def vm_stress_from_dU(dU: torch.Tensor, lame1: float, mu: float) -> torch.Tensor:
    """dU: (..., 3, 3)  →  vm: (...,)"""
    tr  = dU[..., 0, 0] + dU[..., 1, 1] + dU[..., 2, 2]
    sx  = lame1 * tr + 2*mu * dU[..., 0, 0]
    sy  = lame1 * tr + 2*mu * dU[..., 1, 1]
    sz  = lame1 * tr + 2*mu * dU[..., 2, 2]
    txy = mu * (dU[..., 0, 1] + dU[..., 1, 0])
    txz = mu * (dU[..., 0, 2] + dU[..., 2, 0])
    tyz = mu * (dU[..., 1, 2] + dU[..., 2, 1])
    return torch.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                           + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)


def compute_sample(A, phi_ref, lam_vals, dlam_vals, H, lame1, mu,
                   r_ij, row, col, XtX_inv):
    """
    Vectorised over all T-2 timesteps.
    Stress is computed via the same scatter linear fit as train.py
    (compute_vm_stress_per_node), so GT stress and predicted stress use
    identical numerics → loss_stress → 0 when velocity prediction is perfect.

    Returns:
      vel_all    : (T-2, N, 3)   velocity per timestep
      stress_all : (T-2, N, 1)   per-node von Mises stress (scatter fit)
    """
    N   = phi_ref.shape[0]
    T2  = lam_vals.shape[0]
    E   = r_ij.shape[0]

    base_u  = (1.0 / H) * (phi_ref @ A.T)            # (N, 3)
    vel_all = dlam_vals[:, None, None] * base_u       # (T-2, N, 3)

    # Displacement at each timestep: u[t, n] = lam[t]/H * phi[n] @ A.T
    u_all = lam_vals[:, None, None] * base_u          # (T-2, N, 3)

    # Scatter gradient fit — same formula as train.py ─────────────────────────
    du_ij_all = u_all[:, col, :] - u_all[:, row, :]  # (T-2, E, 3)

    # r_ij ⊗ du_ij  per edge per timestep
    r_du_all = r_ij[None, :, :, None] * du_ij_all[:, :, None, :]  # (T-2, E, 3, 3)

    # Accumulate into XtY: (T-2, N, 3, 3)
    row_4d = row[None, :, None, None].expand(T2, E, 3, 3)
    XtY_all = torch.zeros(T2, N, 3, 3)
    XtY_all.scatter_add_(1, row_4d, r_du_all)

    # dU[t, n, k, l] = ∂u_k/∂x_l  (XtX_inv @ XtY, then transpose last two dims)
    result  = torch.einsum('nij,tnjk->tnik', XtX_inv, XtY_all)   # (T-2, N, 3, 3)
    dU_all  = result.permute(0, 1, 3, 2)                          # (T-2, N, 3, 3)

    stress_all = vm_stress_from_dU(dU_all, lame1, mu).unsqueeze(-1)  # (T-2, N, 1)

    return vel_all, stress_all


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    output_dir:   str   = "./04_preprocessed_pt_nonlinear",
    num_samples:  int   = 2010,
    num_time_steps: int = 21,
    seed:         int   = 42,
    coeff_range:  float = 0.02,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = torch.Generator()
    rng.manual_seed(seed)

    # ── Geometry ──────────────────────────────────────────────────────────────
    L, H, n = 1.0, 1.0, 4
    c = torch.linspace(0, L, n)
    gx, gy, gz = torch.meshgrid(c, c, c, indexing='ij')
    ref_pos = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1)
    N = ref_pos.shape[0]

    mesh_ei = build_mesh_edge_index(n)
    print(f"[geometry] N={N} nodes, E_mesh={mesh_ei.shape[1]} edges")

    ref_feat = compute_edge_features(ref_pos, mesh_ei)
    e_mean   = ref_feat.mean(dim=0)
    e_std    = ref_feat.std(dim=0).clamp(min=1e-8)

    # ── Precompute constants ───────────────────────────────────────────────────
    phi_ref, ref_norm, r_ij, row, col, XtX_inv = precompute_geometry(ref_pos, mesh_ei, e_mean, e_std)

    # ── Material ──────────────────────────────────────────────────────────────
    E_mod, nu = 70e9, 0.33
    lame1 = E_mod * nu / ((1+nu) * (1-2*nu))
    mu    = E_mod / (2*(1+nu))

    T = num_time_steps
    # Precompute all λ and Δλ values (constant across samples)
    lam_seq  = torch.tensor([math.sin(math.pi/2*t/(T-1)) for t in range(T)])
    lam_vals = lam_seq[2:]                     # lam at t+1 for t=1..T-2  (T-2,)
    dlam_vals = lam_seq[1:] - lam_seq[:-1]    # Δlam for t=1..T-2        (T-2,)
    dlam_vals = dlam_vals[1:]                  # align with t=1..T-2

    print(f"[generating] {num_samples} samples × {T-2} time-steps ...")
    t0 = time.time()

    # ── Pass 1: collect stats ──────────────────────────────────────────────────
    A_list = []
    all_vel, all_stress = [], []

    for s in range(num_samples):
        A = torch.rand(3, 10, generator=rng) * (2*coeff_range) - coeff_range
        A_list.append(A)
        vel_all, stress_all = compute_sample(A, phi_ref, lam_vals, dlam_vals,
                                             H, lame1, mu, r_ij, row, col, XtX_inv)
        all_vel.append(vel_all)
        all_stress.append(stress_all)

    vel_stack    = torch.stack(all_vel,    dim=0)   # (S, T-2, N, 3)
    stress_stack = torch.stack(all_stress, dim=0)   # (S, T-2, N, 1)

    v_mean = vel_stack.mean(dim=(0,1,2))
    v_std  = vel_stack.std(dim=(0,1,2)).clamp(min=1e-8)
    s_mean = stress_stack.mean(dim=(0,1,2))
    s_std  = stress_stack.std(dim=(0,1,2)).clamp(min=1e-8)

    print(f"[stats] vel  mean={[f'{v:.3e}' for v in v_mean.tolist()]}, "
          f"std={[f'{v:.3e}' for v in v_std.tolist()]}")
    print(f"[stats] stress mean={s_mean.item():.3e}, std={s_std.item():.3e}")
    sample0 = stress_stack[0, 0, :, 0]
    print(f"[stress] sample0 t0: min={sample0.min():.3e}, max={sample0.max():.3e}, "
          f"ratio={sample0.max()/sample0.min().clamp(min=1e-10):.2f}x")

    # ── Pass 2: save ──────────────────────────────────────────────────────────
    for s, A in enumerate(A_list):
        # World positions for all timesteps: (T, N, 3)
        lam_all = lam_seq.unsqueeze(-1).unsqueeze(-1)          # (T, 1, 1)
        base_u  = (1.0/H) * (phi_ref @ A.T)                   # (N, 3)
        wp_all  = ref_pos.unsqueeze(0) + lam_all * base_u      # (T, N, 3)

        vel_s    = all_vel[s]       # (T-2, N, 3)
        stress_s = all_stress[s]    # (T-2, N, 1)

        sample_data = []
        for t_idx in range(T - 2):
            wp_t = wp_all[t_idx + 1]   # world_pos at time t

            # World edge features (normalised)
            world_feat = compute_edge_features(wp_t, mesh_ei)
            world_norm = (world_feat - e_mean) / e_std
            mesh_ef    = torch.cat([ref_norm, world_norm], dim=1)   # (E, 8)

            vel    = vel_s[t_idx]       # (N, 3)
            stress = stress_s[t_idx]    # (N, 1)

            y = torch.cat([
                (vel    - v_mean) / v_std,
                (stress - s_mean) / s_std,
            ], dim=1)   # (N, 4)

            graph = Data(
                x          = wp_t - ref_pos,
                edge_index = mesh_ei,
                edge_attr  = mesh_ef,
                y          = y,
                mesh_pos   = ref_pos,
                world_pos  = wp_t,
            )
            sample_data.append({
                "graph"              : graph,
                "mesh_edge_features" : mesh_ef,
                "world_edge_features": torch.zeros(0, 8),
            })

        pt_path = os.path.join(output_dir, f"sample_{s:05d}.pt")
        torch.save(sample_data, pt_path)
        if s % 200 == 0 or s == num_samples - 1:
            elapsed = time.time() - t0
            print(f"  saved {s+1:4d}/{num_samples}  ({elapsed:.1f}s elapsed)")

    # ── Stats files ───────────────────────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    edge_d = {"edge_mean": e_mean, "edge_std": e_std}
    node_d = {"velocity_mean": v_mean, "velocity_std": v_std,
               "stress_mean": s_mean, "stress_std": s_std}
    for p in ["edge_stats.json", "./outputs/edge_stats.json",
              os.path.join(output_dir, "edge_stats.json")]:
        save_json(edge_d, p)
    for p in ["node_stats.json", "./outputs/node_stats.json",
              os.path.join(output_dir, "node_stats.json")]:
        save_json(node_d, p)

    print(f"\n完成！{num_samples} samples，total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    generate(
        output_dir      = "./04_preprocessed_pt_nonlinear",
        num_samples     = 2010,
        num_time_steps  = 21,
        seed            = 42,
        coeff_range     = 0.02,
    )
