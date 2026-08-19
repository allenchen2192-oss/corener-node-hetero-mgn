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
generate_ood_data.py
====================
生成 Out-of-Distribution (OOD) 測試資料：
  - A 矩陣每個元素的絕對值在 [0.02, 0.05]（訓練集為 [-0.02, 0.02]）
  - Normalization stats 沿用訓練集的 edge_stats.json / node_stats.json
  - 輸出到 ./04_preprocessed_pt_ood
"""

import json
import math
import os

import torch
from torch_geometric.data import Data


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (same as generate_synthetic_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_mesh_edge_index(n: int) -> torch.Tensor:
    edges = []
    def node_id(i, j, k):
        return k * n * n + j * n + i
    for k in range(n):
        for j in range(n):
            for i in range(n):
                s = node_id(i, j, k)
                if i + 1 < n:
                    t = node_id(i + 1, j, k); edges.extend([[s, t], [t, s]])
                if j + 1 < n:
                    t = node_id(i, j + 1, k); edges.extend([[s, t], [t, s]])
                if k + 1 < n:
                    t = node_id(i, j, k + 1); edges.extend([[s, t], [t, s]])
    return torch.tensor(edges, dtype=torch.long).T


def compute_edge_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def make_8dim_ef(edge_index, ref_pos, world_pos, e_mean, e_std):
    ref_feat   = compute_edge_features(ref_pos,   edge_index)
    world_feat = compute_edge_features(world_pos, edge_index)
    ref_norm   = (ref_feat   - e_mean) / e_std
    world_norm = (world_feat - e_mean) / e_std
    return torch.cat([ref_norm, world_norm], dim=1)


def lambda_t(t: int, T: int) -> float:
    return math.sin(math.pi / 2 * t / (T - 1)) if T > 1 else 0.0


def displacement(xyzh, A, lam, H):
    return (lam / H) * (xyzh @ A.T)


def analytic_stress(A, lam, H, lame1, mu):
    dU = (lam / H) * A[:, 1:]
    eps_xx = dU[0, 0].item(); eps_yy = dU[1, 1].item(); eps_zz = dU[2, 2].item()
    eps_xy = 0.5 * (dU[0, 1] + dU[1, 0]).item()
    eps_xz = 0.5 * (dU[0, 2] + dU[2, 0]).item()
    eps_yz = 0.5 * (dU[1, 2] + dU[2, 1]).item()
    tr_eps = eps_xx + eps_yy + eps_zz
    sig_xx = lame1 * tr_eps + 2 * mu * eps_xx
    sig_yy = lame1 * tr_eps + 2 * mu * eps_yy
    sig_zz = lame1 * tr_eps + 2 * mu * eps_zz
    sig_xy = 2 * mu * eps_xy; sig_xz = 2 * mu * eps_xz; sig_yz = 2 * mu * eps_yz
    vm = math.sqrt(0.5 * (
        (sig_xx - sig_yy)**2 + (sig_yy - sig_zz)**2 + (sig_zz - sig_xx)**2
        + 6 * (sig_xy**2 + sig_xz**2 + sig_yz**2)
    ))
    return vm


# ─────────────────────────────────────────────────────────────────────────────
# OOD A matrix sampler: |A_ij| in [low, high], sign random
# ─────────────────────────────────────────────────────────────────────────────

def sample_ood_A(rng: torch.Generator, low: float = 0.02, high: float = 0.05) -> torch.Tensor:
    """每個元素絕對值在 [low, high]，符號隨機。"""
    mag  = torch.rand(3, 4, generator=rng) * (high - low) + low   # magnitude in [low, high]
    sign = torch.sign(torch.rand(3, 4, generator=rng) - 0.5)      # ±1
    return mag * sign


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_ood(
    output_dir:    str   = "./04_preprocessed_pt_ood",
    num_samples:   int   = 10,
    num_time_steps: int  = 21,
    seed:          int   = 2024,
    stats_dir:     str   = ".",          # 讀取訓練集 stats 的目錄
    ood_low:       float = 0.02,
    ood_high:      float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = torch.Generator()
    rng.manual_seed(seed)

    # ── 1. Geometry ────────────────────────────────────────────────────────────
    L, H, n = 1.0, 1.0, 4
    c = torch.linspace(0, L, n)
    gx, gy, gz = torch.meshgrid(c, c, c, indexing='ij')
    ref_pos = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1)  # (64, 3)
    N = ref_pos.shape[0]
    xyzh = torch.cat([torch.ones(N, 1), ref_pos], dim=1)   # (N, 4)

    # ── 2. Mesh edges ──────────────────────────────────────────────────────────
    mesh_ei = build_mesh_edge_index(n)

    # ── 3. Load training normalization stats ───────────────────────────────────
    edge_path = os.path.join(stats_dir, "edge_stats.json")
    node_path = os.path.join(stats_dir, "node_stats.json")
    with open(edge_path) as f:
        edge_stats = json.load(f)
    with open(node_path) as f:
        node_stats = json.load(f)

    e_mean = torch.tensor(edge_stats["edge_mean"])
    e_std  = torch.tensor(edge_stats["edge_std"])
    v_mean = torch.tensor(node_stats["velocity_mean"])
    v_std  = torch.tensor(node_stats["velocity_std"])
    s_mean = torch.tensor(node_stats["stress_mean"])
    s_std  = torch.tensor(node_stats["stress_std"])

    print(f"[stats loaded] edge_stats from {edge_path}")
    print(f"               node_stats from {node_path}")

    # ── 4. Material ────────────────────────────────────────────────────────────
    E_mod = 70e9; nu = 0.33
    lame1 = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E_mod / (2 * (1 + nu))

    T = num_time_steps

    # ── 5. Generate & save ─────────────────────────────────────────────────────
    print(f"\n[generating] {num_samples} OOD samples  |A_ij| in [{ood_low}, {ood_high}]")
    A_list = []
    for s in range(num_samples):
        A = sample_ood_A(rng, ood_low, ood_high)
        A_list.append(A)
        print(f"  sample {s:05d}  A =\n{A.numpy()}")

    print()
    for s, A in enumerate(A_list):
        wp = [ref_pos + displacement(xyzh, A, lambda_t(t, T), H) for t in range(T)]

        sample_data = []
        for t in range(1, T - 1):
            wp_t = wp[t]
            lam_next = lambda_t(t + 1, T)
            dlam     = lam_next - lambda_t(t, T)

            vel    = (dlam / H) * (xyzh @ A.T)
            vm     = analytic_stress(A, lam_next, H, lame1, mu)
            stress = torch.full((N, 1), vm)

            mesh_ef  = make_8dim_ef(mesh_ei, ref_pos, wp_t, e_mean, e_std)
            world_ef = torch.zeros(0, 8)

            y = torch.cat([
                (vel    - v_mean) / v_std,
                (stress - s_mean) / s_std,
            ], dim=1)

            disp_vec = wp_t - ref_pos
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
        print(f"  saved → {pt_path}")

    print(f"\n完成！{num_samples} 個 OOD sample 存至 {output_dir}")


if __name__ == "__main__":
    generate_ood(
        output_dir    = "./04_preprocessed_pt_ood",
        num_samples   = 10,
        num_time_steps = 21,
        seed          = 2024,
        stats_dir     = ".",
        ood_low       = 0.02,
        ood_high      = 0.05,
    )
