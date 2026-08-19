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
export_predictions_paraview_prevvel.py
=======================================
Run rollout with the prevvel model and export GT + predicted results
to VTU / PVD format for side-by-side comparison in ParaView.

Output structure:
  export_pred_paraview_prevvel/
    sample_XXXXX/
      gt/
        t_000.vtu  t_001.vtu  ...
        sample_XXXXX_gt.pvd
      pred/
        t_000.vtu  t_001.vtu  ...
        sample_XXXXX_pred.pvd

Fields per node:
  displacement   (3-vector)
  disp_mag       (scalar)
  von_mises      (scalar, Pa)
  disp_error     (scalar, pred only)

Usage:
  Edit SAMPLE_INDICES and CKPT_EPOCH, then run:
    python export_predictions_paraview_prevvel.py
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 6    # x_t(3) + prev_vel(3)
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 64

DATA_DIR   = "./04_preprocessed_pt_nonlinear_prevvel"
CKPT_PATH  = "./checkpoints_nonlinear_prevvel"
CKPT_EPOCH = None          # None = latest; set e.g. 1000 for a specific epoch
OUTPUT_DIR = "./export_pred_paraview_prevvel"
N_GRID     = 4             # 4×4×4 mesh

# Samples to export (mix of train and test)
SAMPLE_INDICES = [0, 1, 2, 2000, 2001, 2002]


# ── Material constants ────────────────────────────────────────────────────────

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


# ── VTU / PVD writers ─────────────────────────────────────────────────────────

def build_hex_cells(n: int):
    def nid(i, j, k):
        return k * n * n + j * n + i
    conn = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                conn.extend([
                    nid(i,   j,   k  ), nid(i+1, j,   k  ),
                    nid(i+1, j+1, k  ), nid(i,   j+1, k  ),
                    nid(i,   j,   k+1), nid(i+1, j,   k+1),
                    nid(i+1, j+1, k+1), nid(i,   j+1, k+1),
                ])
    n_cells = (n - 1) ** 3
    connectivity = np.array(conn, dtype=np.int32)
    offsets      = np.arange(8, 8 * (n_cells + 1), 8, dtype=np.int32)
    types        = np.full(n_cells, 12, dtype=np.uint8)
    return connectivity, offsets, types


def _arr_str(arr):
    return " ".join(f"{v:.6g}" for v in np.asarray(arr, dtype=np.float32).ravel())


def write_vtu(path, points, connectivity, offsets, types, point_data):
    N, n_cells = points.shape[0], len(types)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{N}" NumberOfCells="{n_cells}">',
        '      <Points>',
        '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">',
        f'          {_arr_str(points)}',
        '        </DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int32" Name="connectivity" format="ascii">',
        f'          {_arr_str(connectivity)}',
        '        </DataArray>',
        '        <DataArray type="Int32" Name="offsets" format="ascii">',
        f'          {_arr_str(offsets)}',
        '        </DataArray>',
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        f'          {_arr_str(types)}',
        '        </DataArray>',
        '      </Cells>',
        '      <PointData>',
    ]
    for name, arr in point_data.items():
        arr = np.asarray(arr, dtype=np.float32)
        nc  = arr.shape[1] if arr.ndim > 1 else None
        if nc:
            lines.append(f'        <DataArray type="Float32" Name="{name}" '
                         f'NumberOfComponents="{nc}" format="ascii">')
        else:
            lines.append(f'        <DataArray type="Float32" Name="{name}" format="ascii">')
        lines.append(f'          {_arr_str(arr)}')
        lines.append('        </DataArray>')
    lines += ['      </PointData>', '    </Piece>', '  </UnstructuredGrid>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_pvd(path, vtu_entries):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t_val, vtu_rel in vtu_entries:
        lines.append(f'    <DataSet timestep="{t_val:.4f}" group="" part="0" file="{vtu_rel}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Physics helpers ───────────────────────────────────────────────────────────

def compute_edge_features(pos, edge_index):
    row, col = edge_index
    disp = pos[row] - pos[col]
    dist = torch.norm(disp, dim=1, keepdim=True)
    return torch.cat([disp, dist], dim=1)


def precompute_xtx_inv(ref_pos, mesh_ei):
    N = ref_pos.shape[0]
    row, col = mesh_ei
    r_ij = ref_pos[col] - ref_pos[row]
    r_r  = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    XtX = torch.zeros(N, 3, 3, device=ref_pos.device)
    XtX.scatter_add_(0, row_exp, r_r)
    XtX = XtX + 1e-6 * torch.eye(3, device=ref_pos.device).unsqueeze(0)
    return torch.linalg.inv(XtX), r_ij, row, col


def compute_vm_stress(wp, ref, XtX_inv, r_ij, row, col):
    u     = wp - ref
    du_ij = u[col] - u[row]
    r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    row_exp = row.view(-1, 1, 1).expand_as(r_du)
    XtY = torch.zeros(wp.shape[0], 3, 3, device=wp.device)
    XtY.scatter_add_(0, row_exp, r_du)
    dU  = torch.bmm(XtX_inv, XtY).permute(0, 2, 1)
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy  = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz  = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                              + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)


# ── Setup ─────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean     = node_stats["velocity_mean"].to(device)
v_std      = node_stats["velocity_std"].to(device)
s_mean_val = node_stats["stress_mean"].item()
s_std_val  = node_stats["stress_std"].item()
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean = edge_stats["edge_mean"].to(device)[:4]
e_std  = edge_stats["edge_std"].to(device)[:4]

first   = torch.load(f"{DATA_DIR}/sample_00000.pt", map_location="cpu", weights_only=False)
E_mesh  = first[0]["mesh_edge_features"].shape[0]
mesh_ei = first[0]["graph"].edge_index[:, :E_mesh].to(device)
ref_pos = first[0]["graph"].mesh_pos.to(device)

ref_feat_n = (compute_edge_features(ref_pos, mesh_ei) - e_mean) / e_std
empty_wef  = torch.zeros(0, NUM_EDGE_FEATURES, device=device)

XtX_inv, r_ij, row_ei, col_ei = precompute_xtx_inv(ref_pos, mesh_ei)
connectivity, offsets, types   = build_hex_cells(N_GRID)

print(f"[mesh] N={ref_pos.shape[0]}, rollout_steps={len(first)}")


def make_graph(wp, prev_vel):
    wf = (compute_edge_features(wp, mesh_ei) - e_mean) / e_std
    ef = torch.cat([ref_feat_n, wf], dim=1)
    x  = torch.cat([wp - ref_pos, prev_vel], dim=1)   # (N, 6)
    return Data(x=x, edge_index=mesh_ei, edge_attr=ef), ef


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

ep_tag = f" epoch={CKPT_EPOCH}" if CKPT_EPOCH else " (latest)"
print(f"Loading checkpoint: {CKPT_PATH}{ep_tag}")
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)


# ── Export ────────────────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
ref_np = ref_pos.cpu().numpy()

for idx in SAMPLE_INDICES:
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range")
        continue

    fname     = all_files[idx]
    sample_id = fname.replace(".pt", "")
    fpath     = os.path.join(DATA_DIR, fname)

    gt_dir   = os.path.join(OUTPUT_DIR, sample_id, "gt")
    pred_dir = os.path.join(OUTPUT_DIR, sample_id, "pred")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    data = torch.load(fpath, map_location="cpu", weights_only=False)
    T    = len(data)
    print(f"\n[{sample_id}] {T} steps → {os.path.join(OUTPUT_DIR, sample_id)}")

    # Reconstruct GT world-pos sequence
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    gt_stress_list = [
        (step["graph"].y[:, 3] * s_std_val + s_mean_val).numpy()
        for step in data
    ]

    # ── Run rollout ───────────────────────────────────────────────────────────
    cur      = gt_wp_list[0].to(device)
    prev_vel = data[0]["graph"].x[:, 3:].to(device)   # GT prev_vel for first step
    pred_wp_list = [gt_wp_list[0].numpy()]             # t=0 same as GT

    with torch.inference_mode():
        for k in range(T):
            g, ef    = make_graph(cur, prev_vel)
            out      = model(g.x, ef, empty_wef, g)
            vel      = out[:, :3] * v_std + v_mean
            prev_vel = vel
            cur      = cur + vel
            pred_wp_list.append(cur.cpu().numpy())

    # ── Write VTU files ───────────────────────────────────────────────────────
    gt_pvd, pred_pvd = [], []

    for t_idx in range(T):
        vtu_name = f"t_{t_idx:03d}.vtu"

        # GT
        gt_pos  = gt_wp_list[t_idx + 1].numpy().astype(np.float32)
        gt_disp = gt_pos - ref_np
        gt_vm   = gt_stress_list[t_idx].astype(np.float32)

        write_vtu(
            os.path.join(gt_dir, vtu_name),
            gt_pos, connectivity, offsets, types,
            {
                "displacement": gt_disp,
                "disp_mag":     np.linalg.norm(gt_disp, axis=1),
                "von_mises":    gt_vm,
            },
        )
        gt_pvd.append((t_idx, vtu_name))

        # Pred
        pred_pos  = pred_wp_list[t_idx + 1].astype(np.float32)
        pred_disp = pred_pos - ref_np
        pred_vm   = compute_vm_stress(
            torch.from_numpy(pred_pos).to(device),
            ref_pos, XtX_inv, r_ij, row_ei, col_ei,
        ).cpu().numpy().astype(np.float32)
        disp_err  = np.linalg.norm(pred_pos - gt_pos, axis=1).astype(np.float32)

        write_vtu(
            os.path.join(pred_dir, vtu_name),
            pred_pos, connectivity, offsets, types,
            {
                "displacement": pred_disp,
                "disp_mag":     np.linalg.norm(pred_disp, axis=1),
                "von_mises":    pred_vm,
                "disp_error":   disp_err,
            },
        )
        pred_pvd.append((t_idx, vtu_name))

    write_pvd(os.path.join(gt_dir,   f"{sample_id}_gt.pvd"),   gt_pvd)
    write_pvd(os.path.join(pred_dir, f"{sample_id}_pred.pvd"), pred_pvd)

    final_disp_err = np.linalg.norm(
        pred_wp_list[-1] - gt_wp_list[-1].numpy(), axis=1
    ).mean()
    print(f"  final mean node error: {final_disp_err:.6f} m")
    print(f"  GT   PVD → {os.path.join(gt_dir,   sample_id + '_gt.pvd')}")
    print(f"  Pred PVD → {os.path.join(pred_dir, sample_id + '_pred.pvd')}")

print("\n完成！用 ParaView 開各 sample 下 gt/ 和 pred/ 裡的 .pvd 檔。")
