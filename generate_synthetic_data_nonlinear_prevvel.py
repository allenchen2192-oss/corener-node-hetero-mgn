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
generate_synthetic_data_nonlinear_prevvel.py
=============================================
加 Δx_{t-1}（前一步速度）的版本。

與 generate_synthetic_data_nonlinear.py 的唯一差異：
  graph.x = [x_t (3), prev_vel (3)]   → num_input_features = 6

  prev_vel 定義：
    t_idx = 0  : wp_all[1] - ref_pos  (從靜止到 t=1 的位移增量)
    t_idx > 0  : vel_s[t_idx - 1]     (前一步的物理速度，與 x_t 同單位)

輸出目錄: ./04_preprocessed_pt_nonlinear_prevvel
"""

import math
import os
import time

import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import save_json


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers  (identical to generate_synthetic_data_nonlinear.py)
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


def precompute_geometry(ref_pos, mesh_ei, e_mean, e_std):
    x, y, z = ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2]
    N  = ref_pos.shape[0]
    z1 = torch.ones(N)
    phi_ref = torch.stack([z1, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z], dim=1)
    ref_feat = compute_edge_features(ref_pos, mesh_ei)
    ref_norm = (ref_feat - e_mean) / e_std
    row, col = mesh_ei
    r_ij = ref_pos[col] - ref_pos[row]
    r_r  = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    XtX = torch.zeros(N, 3, 3)
    XtX.scatter_add_(0, row_exp, r_r)
    XtX = XtX + 1e-6 * torch.eye(3).unsqueeze(0)
    XtX_inv = torch.linalg.inv(XtX)
    return phi_ref, ref_norm, r_ij, row, col, XtX_inv


def vm_stress_from_dU(dU, lame1, mu):
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
    N, T2, E = phi_ref.shape[0], lam_vals.shape[0], r_ij.shape[0]
    base_u  = (1.0 / H) * (phi_ref @ A.T)
    vel_all = dlam_vals[:, None, None] * base_u
    u_all   = lam_vals[:, None, None]  * base_u
    du_ij_all = u_all[:, col, :] - u_all[:, row, :]
    r_du_all  = r_ij[None, :, :, None] * du_ij_all[:, :, None, :]
    row_4d    = row[None, :, None, None].expand(T2, E, 3, 3)
    XtY_all   = torch.zeros(T2, N, 3, 3)
    XtY_all.scatter_add_(1, row_4d, r_du_all)
    result    = torch.einsum('nij,tnjk->tnik', XtX_inv, XtY_all)
    dU_all    = result.permute(0, 1, 3, 2)
    stress_all = vm_stress_from_dU(dU_all, lame1, mu).unsqueeze(-1)
    return vel_all, stress_all


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    output_dir:     str   = "./04_preprocessed_pt_nonlinear_prevvel",
    num_samples:    int   = 2010,
    num_time_steps: int   = 21,
    seed:           int   = 42,
    coeff_range:    float = 0.02,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = torch.Generator()
    rng.manual_seed(seed)

    L, H, n = 1.0, 1.0, 4
    c = torch.linspace(0, L, n)
    gx, gy, gz = torch.meshgrid(c, c, c, indexing='ij')
    ref_pos = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1)
    N = ref_pos.shape[0]

    mesh_ei  = build_mesh_edge_index(n)
    ref_feat = compute_edge_features(ref_pos, mesh_ei)
    e_mean   = ref_feat.mean(dim=0)
    e_std    = ref_feat.std(dim=0).clamp(min=1e-8)

    phi_ref, ref_norm, r_ij, row, col, XtX_inv = precompute_geometry(
        ref_pos, mesh_ei, e_mean, e_std
    )

    E_mod, nu = 70e9, 0.33
    lame1 = E_mod * nu / ((1+nu) * (1-2*nu))
    mu    = E_mod / (2*(1+nu))

    T = num_time_steps
    lam_seq   = torch.tensor([math.sin(math.pi/2*t/(T-1)) for t in range(T)])
    lam_vals  = lam_seq[2:]
    dlam_vals = (lam_seq[1:] - lam_seq[:-1])[1:]

    print(f"[geometry] N={N}, E_mesh={mesh_ei.shape[1]}")
    print(f"[generating] {num_samples} samples × {T-2} timesteps ...")
    t0 = time.time()

    # ── Pass 1: compute stats ─────────────────────────────────────────────────
    A_list = []
    all_vel, all_stress = [], []
    for s in range(num_samples):
        A = torch.rand(3, 10, generator=rng) * (2*coeff_range) - coeff_range
        A_list.append(A)
        vel_all, stress_all = compute_sample(
            A, phi_ref, lam_vals, dlam_vals, H, lame1, mu, r_ij, row, col, XtX_inv
        )
        all_vel.append(vel_all)
        all_stress.append(stress_all)

    vel_stack    = torch.stack(all_vel,    dim=0)
    stress_stack = torch.stack(all_stress, dim=0)
    v_mean = vel_stack.mean(dim=(0,1,2))
    v_std  = vel_stack.std(dim=(0,1,2)).clamp(min=1e-8)
    s_mean = stress_stack.mean(dim=(0,1,2))
    s_std  = stress_stack.std(dim=(0,1,2)).clamp(min=1e-8)

    print(f"[stats] vel  mean={[f'{v:.3e}' for v in v_mean.tolist()]}, "
          f"std={[f'{v:.3e}' for v in v_std.tolist()]}")
    print(f"[stats] stress mean={s_mean.item():.3e}, std={s_std.item():.3e}")

    # ── Pass 2: save ──────────────────────────────────────────────────────────
    for s, A in enumerate(A_list):
        lam_all = lam_seq.unsqueeze(-1).unsqueeze(-1)
        base_u  = (1.0/H) * (phi_ref @ A.T)
        wp_all  = ref_pos.unsqueeze(0) + lam_all * base_u

        vel_s    = all_vel[s]
        stress_s = all_stress[s]

        sample_data = []
        for t_idx in range(T - 2):
            wp_t = wp_all[t_idx + 1]

            # ── Δx_{t-1}（前一步速度）────────────────────────────────────────
            # t_idx=0: 物體從靜止(x_0=0)運動到 t=1，prev_vel = x_1 - x_0 = x_1
            # t_idx>0: 前一步的物理速度 vel_s[t_idx-1]
            if t_idx == 0:
                prev_vel = wp_t - ref_pos          # = x_1，shape (N, 3)
            else:
                prev_vel = vel_s[t_idx - 1]        # (N, 3)，物理單位

            x_t    = wp_t - ref_pos                # (N, 3) 當前位移
            node_x = torch.cat([x_t, prev_vel], dim=1)   # (N, 6)

            world_feat = compute_edge_features(wp_t, mesh_ei)
            world_norm = (world_feat - e_mean) / e_std
            mesh_ef    = torch.cat([ref_norm, world_norm], dim=1)

            vel    = vel_s[t_idx]
            stress = stress_s[t_idx]
            y = torch.cat([
                (vel    - v_mean) / v_std,
                (stress - s_mean) / s_std,
            ], dim=1)

            graph = Data(
                x          = node_x,
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
            print(f"  saved {s+1:4d}/{num_samples}  ({time.time()-t0:.1f}s elapsed)")

    # ── Stats files ───────────────────────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    edge_d = {"edge_mean": e_mean, "edge_std": e_std}
    node_d = {"velocity_mean": v_mean, "velocity_std": v_std,
               "stress_mean": s_mean, "stress_std": s_std}
    for p in ["edge_stats_prevvel.json", "./outputs/edge_stats_prevvel.json",
              os.path.join(output_dir, "edge_stats.json")]:
        save_json(edge_d, p)
    for p in ["node_stats_prevvel.json", "./outputs/node_stats_prevvel.json",
              os.path.join(output_dir, "node_stats.json")]:
        save_json(node_d, p)

    print(f"\n完成！{num_samples} samples，total {time.time()-t0:.1f}s")
    print(f"輸出: {output_dir}")
    print(f"node stats: node_stats_prevvel.json / edge_stats_prevvel.json")


if __name__ == "__main__":
    generate(
        output_dir      = "./04_preprocessed_pt_nonlinear_prevvel",
        num_samples     = 2100,
        num_time_steps  = 21,
        seed            = 42,
        coeff_range     = 0.02,
    )
