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
export_to_paraview.py
=====================
Export synthetic dataset samples to VTU format for visualization in ParaView.

Outputs:
  export_paraview/
    sample_XXXXX/
      t_000.vtu   ← one file per timestep (deformed mesh + fields)
      t_001.vtu
      ...
      sample_XXXXX.pvd  ← time-series collection, open this in ParaView

Fields exported per node:
  - displacement   (3-component vector)
  - disp_mag       (scalar, displacement magnitude)
  - von_mises      (scalar, denormalized von Mises stress)
  - velocity       (3-component vector, Lagrangian velocity at this step)

Usage:
  python export_to_paraview.py
  (edit SAMPLE_INDICES, DATA_DIR, N_GRID at the bottom as needed)
"""

import os
import numpy as np
import torch

from physicsnemo.datapipes.gnn.utils import load_json


# ─────────────────────────────────────────────────────────────────────────────
# VTU writer (pure Python, no extra dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def build_hex_cells(n: int):
    """
    Build VTK hexahedral cell connectivity for an n×n×n node grid.
    node_id(i, j, k) = k*n*n + j*n + i

    Returns:
      connectivity : (n_cells * 8,) int array
      offsets      : (n_cells,)     int array  (cumulative)
      types        : (n_cells,)     uint8 array (12 = VTK_HEXAHEDRON)
    """
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
    types        = np.full(n_cells, 12, dtype=np.uint8)   # VTK_HEXAHEDRON
    return connectivity, offsets, types


def _arr_str(arr: np.ndarray) -> str:
    """Flat array → space-separated string."""
    return " ".join(f"{v:.6g}" for v in arr.ravel())


def write_vtu(path: str, points: np.ndarray, connectivity, offsets, types,
              point_data: dict):
    """
    Write a single VTU (XML UnstructuredGrid) file.

    points     : (N, 3) float32
    point_data : {name: array}  — 1D (scalar) or (N,3) (vector)
    """
    N       = points.shape[0]
    n_cells = len(types)

    lines = ['<?xml version="1.0"?>',
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
             '      <PointData>']

    for name, arr in point_data.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            lines.append(f'        <DataArray type="Float32" Name="{name}" format="ascii">')
        else:
            nc = arr.shape[1]
            lines.append(f'        <DataArray type="Float32" Name="{name}" '
                         f'NumberOfComponents="{nc}" format="ascii">')
        lines.append(f'          {_arr_str(arr)}')
        lines.append('        </DataArray>')

    lines += ['      </PointData>',
              '    </Piece>',
              '  </UnstructuredGrid>',
              '</VTKFile>']

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_pvd(path: str, vtu_entries: list):
    """
    Write a PVD collection file for time-series animation.
    vtu_entries : list of (time_value, relative_vtu_path)
    """
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '  <Collection>']
    for t_val, vtu_rel in vtu_entries:
        lines.append(f'    <DataSet timestep="{t_val:.4f}" group="" part="0" '
                     f'file="{vtu_rel}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_samples(
    data_dir:       str   = "./04_preprocessed_pt_nonlinear",
    output_dir:     str   = "./export_paraview",
    sample_indices: list  = [0, 1, 2],
    n_grid:         int   = 4,       # grid resolution (4 → 4×4×4 nodes)
):
    os.makedirs(output_dir, exist_ok=True)

    # Load normalization stats
    node_stats = load_json("node_stats.json")
    v_mean = node_stats["velocity_mean"].numpy()
    v_std  = node_stats["velocity_std"].numpy()
    s_mean = node_stats["stress_mean"].numpy()
    s_std  = node_stats["stress_std"].numpy()

    # Build hex cell topology (same for all samples/timesteps)
    connectivity, offsets, types = build_hex_cells(n_grid)

    # List all sample files
    all_files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith("sample_") and f.endswith(".pt")
    )

    for idx in sample_indices:
        if idx >= len(all_files):
            print(f"[skip] sample index {idx} out of range ({len(all_files)} files)")
            continue

        fname     = all_files[idx]
        fpath     = os.path.join(data_dir, fname)
        sample_id = fname.replace(".pt", "")
        out_dir   = os.path.join(output_dir, sample_id)
        os.makedirs(out_dir, exist_ok=True)

        data = torch.load(fpath, map_location="cpu", weights_only=False)
        print(f"[{sample_id}] {len(data)} timesteps → {out_dir}")

        pvd_entries = []

        for t_idx, step in enumerate(data):
            graph = step["graph"]

            # World positions (deformed mesh coordinates)
            world_pos = graph.world_pos.numpy().astype(np.float32)   # (N, 3)
            ref_pos   = graph.mesh_pos.numpy().astype(np.float32)    # (N, 3)

            # Displacement
            disp     = world_pos - ref_pos                            # (N, 3)
            disp_mag = np.linalg.norm(disp, axis=1)                  # (N,)

            # Denormalize targets: y = [vel_norm(3), stress_norm(1)]
            y = graph.y.numpy()                                       # (N, 4)
            vel      = y[:, :3] * v_std + v_mean                     # (N, 3)
            stress   = (y[:, 3:4] * s_std + s_mean).squeeze(1)      # (N,)

            vtu_name = f"t_{t_idx:03d}.vtu"
            vtu_path = os.path.join(out_dir, vtu_name)

            write_vtu(
                path         = vtu_path,
                points       = world_pos,
                connectivity = connectivity,
                offsets      = offsets,
                types        = types,
                point_data   = {
                    "displacement" : disp,
                    "disp_mag"     : disp_mag,
                    "von_mises"    : stress,
                    "velocity"     : vel,
                },
            )

            pvd_entries.append((t_idx, vtu_name))

        pvd_path = os.path.join(out_dir, f"{sample_id}.pvd")
        write_pvd(pvd_path, pvd_entries)
        print(f"  → open in ParaView: {pvd_path}")

    print("\n完成！用 ParaView 開啟各 sample 資料夾內的 .pvd 檔。")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    export_samples(
        data_dir       = "./04_preprocessed_pt_nonlinear",
        output_dir     = "./export_paraview",
        sample_indices = [0, 1, 2],   # 改這裡選要看哪幾個 sample
        n_grid         = 4,
    )
