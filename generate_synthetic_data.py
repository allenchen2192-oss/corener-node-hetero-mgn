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
generate_synthetic_data.py
==========================
生成合成訓練資料（解析線性位移場），用於測試 MGN training pipeline。

位移場（見 Problem Setup slide）:
  u_i(x, y, z, t) = λ(t) * (1/H) * (a_i0 + a_i1*x + a_i2*y + a_i3*z)
  其中 i ∈ {x, y, z}，λ(t) = sin(π*t / (2*T))
  A 矩陣 (3×4) 係數從 Uniform[-0.02, 0.02] 隨機取樣，每個 sample 不同

Edge connectivity (v3 style):
  mesh edges : face-adjacent structured grid (6 neighbors per interior node)
  world edges: radius graph on world positions (r = 1.08 * spacing), mesh fallback if empty
"""

import math
import os

import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import save_json


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_mesh_edge_index(n: int) -> torch.Tensor:
    """Face-adjacent structured grid edges for n×n×n grid, bidirectional."""
    edges = []
    def node_id(i, j, k):
        return k * n * n + j * n + i
    for k in range(n):
        for j in range(n):
            for i in range(n):
                s = node_id(i, j, k)
                if i + 1 < n:
                    t = node_id(i + 1, j, k)
                    edges.extend([[s, t], [t, s]])
                if j + 1 < n:
                    t = node_id(i, j + 1, k)
                    edges.extend([[s, t], [t, s]])
                if k + 1 < n:
                    t = node_id(i, j, k + 1)
                    edges.extend([[s, t], [t, s]])
    return torch.tensor(edges, dtype=torch.long).T   # (2, E)


def radius_edge_index(pos: torch.Tensor, r: float) -> torch.Tensor:
    """Radius-based edge index, no self-loops, bidirectional."""
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)   # (N, N, 3)
    dist = torch.norm(diff, dim=-1)               # (N, N)
    mask = (dist < r) & (dist > 0)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)         # (2, E)


def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """[relative_displacement(3), distance(1)] for each directed edge → (E, 4)."""
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def make_8dim_ef(edge_index: torch.Tensor, ref_pos: torch.Tensor, world_pos: torch.Tensor,
                 e_mean: torch.Tensor, e_std: torch.Tensor) -> torch.Tensor:
    """[ref_norm(4) | world_norm(4)] = 8-dim edge features."""
    ref_feat   = compute_edge_features(ref_pos,   edge_index)
    world_feat = compute_edge_features(world_pos, edge_index)
    ref_norm   = (ref_feat   - e_mean) / e_std
    world_norm = (world_feat - e_mean) / e_std
    return torch.cat([ref_norm, world_norm], dim=1)   # (E, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Analytic displacement field
# ─────────────────────────────────────────────────────────────────────────────

def lambda_t(t: int, T: int) -> float:
    """λ_n = sin(π/2 * n/(N-1)),  n=0..N-1, N=T"""
    return math.sin(math.pi / 2 * t / (T - 1)) if T > 1 else 0.0


def displacement(xyzh: torch.Tensor, A: torch.Tensor, lam: float, H: float) -> torch.Tensor:
    """
    u(x,y,z,t) = λ(t)/H * A @ [1, x, y, z]^T
    xyzh: (N, 4) = [1, x, y, z] for each node
    A   : (3, 4) random coefficient matrix
    Returns (N, 3)
    """
    return (lam / H) * (xyzh @ A.T)   # (N, 3)


def analytic_stress(A: torch.Tensor, lam: float, H: float,
                    lame1: float, mu: float) -> float:
    """Von Mises stress at lambda=lam from linear displacement gradient."""
    dU = (lam / H) * A[:, 1:]   # (3, 3): ∂u_i/∂x_j

    eps_xx = dU[0, 0].item()
    eps_yy = dU[1, 1].item()
    eps_zz = dU[2, 2].item()
    eps_xy = 0.5 * (dU[0, 1] + dU[1, 0]).item()
    eps_xz = 0.5 * (dU[0, 2] + dU[2, 0]).item()
    eps_yz = 0.5 * (dU[1, 2] + dU[2, 1]).item()

    tr_eps = eps_xx + eps_yy + eps_zz
    sig_xx = lame1 * tr_eps + 2 * mu * eps_xx
    sig_yy = lame1 * tr_eps + 2 * mu * eps_yy
    sig_zz = lame1 * tr_eps + 2 * mu * eps_zz
    sig_xy = 2 * mu * eps_xy
    sig_xz = 2 * mu * eps_xz
    sig_yz = 2 * mu * eps_yz

    vm = math.sqrt(0.5 * (
        (sig_xx - sig_yy)**2 + (sig_yy - sig_zz)**2 + (sig_zz - sig_xx)**2
        + 6 * (sig_xy**2 + sig_xz**2 + sig_yz**2)
    ))
    return vm


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    output_dir: str = "./04_preprocessed_pt",
    num_samples: int = 100,
    num_time_steps: int = 100,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = torch.Generator()
    rng.manual_seed(seed)

    # ── 1. Geometry: 4×4×4 regular grid ──────────────────────────────────────
    L = 1.0
    H = L
    n = 4

    c = torch.linspace(0, L, n)
    gx, gy, gz = torch.meshgrid(c, c, c, indexing='ij')
    ref_pos = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1)  # (64, 3)
    N = ref_pos.shape[0]   # 64

    xyzh = torch.cat([torch.ones(N, 1), ref_pos], dim=1)   # (N, 4)

    # ── 2. Mesh edges: face-adjacent structured grid ──────────────────────────
    mesh_ei = build_mesh_edge_index(n)   # (2, E_mesh)
    E_mesh  = mesh_ei.shape[1]
    print(f"[geometry] N={N} nodes, E_mesh={E_mesh} mesh edges (face-adjacent)")

    # ── 3. Reference edge features & stats (from mesh edges, static) ─────────
    ref_feat      = compute_edge_features(ref_pos, mesh_ei)   # (E_mesh, 4)
    e_mean        = ref_feat.mean(dim=0)
    e_std         = ref_feat.std(dim=0).clamp(min=1e-8)

    # ── 4. Material: aluminium alloy ──────────────────────────────────────────
    E_mod = 70e9
    nu    = 0.33
    lame1 = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E_mod / (2 * (1 + nu))

    T = num_time_steps
    pairs_per_sample = T - 2
    print(f"\n[generating] {num_samples} samples × {pairs_per_sample} time-steps (skip t=0) ...")

    A_list       = []
    all_vel_t    = []
    all_stress_t = []

    for s in range(num_samples):
        A = torch.rand(3, 4, generator=rng) * 0.04 - 0.02   # (3, 4), entries in [-0.02, 0.02]
        A_list.append(A)

        for t in range(1, T - 1):
            lam_cur  = lambda_t(t,     T)
            lam_next = lambda_t(t + 1, T)
            dlam     = lam_next - lam_cur

            vel = (dlam / H) * (xyzh @ A.T)   # (N, 3)
            all_vel_t.append(vel)

            vm = analytic_stress(A, lam_next, H, lame1, mu)
            all_stress_t.append(torch.full((N, 1), vm))

    # ── 5. Global normalization stats ─────────────────────────────────────────
    vel_stack    = torch.stack(all_vel_t,    dim=0)
    stress_stack = torch.stack(all_stress_t, dim=0)

    v_mean = vel_stack.mean(dim=(0, 1))
    v_std  = vel_stack.std(dim=(0, 1)).clamp(min=1e-8)
    s_mean = stress_stack.mean(dim=(0, 1))
    s_std  = stress_stack.std(dim=(0, 1)).clamp(min=1e-8)

    print(f"[stats] vel    mean={[f'{v:.3e}' for v in v_mean.tolist()]}, "
          f"std={[f'{v:.3e}' for v in v_std.tolist()]}")
    print(f"[stats] stress mean={s_mean.item():.3e},  std={s_std.item():.3e}")

    # ── 6. Build and save each sample ────────────────────────────────────────
    for s, A in enumerate(A_list):
        wp = [ref_pos + displacement(xyzh, A, lambda_t(t, T), H) for t in range(T)]

        sample_data = []
        for t_idx, t in enumerate(range(1, T - 1)):
            wp_t = wp[t]

            idx    = s * (T - 2) + t_idx
            vel    = all_vel_t[idx]
            stress = all_stress_t[idx]

            # 8-dim edge features: [ref_norm(4) | world_norm(4)]
            mesh_ef  = make_8dim_ef(mesh_ei, ref_pos, wp_t, e_mean, e_std)
            world_ef = torch.zeros(0, 8)   # no world edges for small-deformation linear field

            # Normalized targets
            y = torch.cat([
                (vel    - v_mean) / v_std,
                (stress - s_mean) / s_std,
            ], dim=1)

            disp_vec = wp_t - ref_pos   # (N, 3), matches _make_node_x in train.py
            graph = Data(
                x          = disp_vec,
                edge_index = mesh_ei,
                edge_attr  = mesh_ef,
                y          = y,
                mesh_pos   = ref_pos,
                world_pos  = wp_t,
            )

            sample_data.append({
                "graph"              : graph,
                "mesh_edge_features" : mesh_ef,
                "world_edge_features": world_ef,
            })

        pt_path = os.path.join(output_dir, f"sample_{s:05d}.pt")
        torch.save(sample_data, pt_path)
        if s % 50 == 0 or s == num_samples - 1:
            print(f"  saved sample {s+1:4d}/{num_samples}  → {pt_path}")

    # ── 7. Save normalization stats ───────────────────────────────────────────
    os.makedirs("./outputs", exist_ok=True)
    for path in ["edge_stats.json", "./outputs/edge_stats.json"]:
        save_json({"edge_mean": e_mean, "edge_std": e_std}, path)
    for path in ["node_stats.json", "./outputs/node_stats.json"]:
        save_json({
            "velocity_mean": v_mean, "velocity_std": v_std,
            "stress_mean"  : s_mean, "stress_std"  : s_std,
        }, path)
    print("\n[saved] edge_stats.json, node_stats.json  (current dir + ./outputs/)")
    print(f"\n完成！共 {num_samples} 個 sample。")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate(
        output_dir     = "./04_preprocessed_pt",
        num_samples    = 2010,
        num_time_steps = 21,
        seed           = 42,
    )
