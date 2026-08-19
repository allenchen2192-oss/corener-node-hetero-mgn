#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import meshio
import numpy as np


def build_hex_cells(nx: int, ny: int, nz: int) -> np.ndarray:
    def node_id(i: int, j: int, k: int) -> int:
        return k * nx * ny + j * nx + i

    hex_cells = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                n000 = node_id(i, j, k)
                n100 = node_id(i + 1, j, k)
                n110 = node_id(i + 1, j + 1, k)
                n010 = node_id(i, j + 1, k)
                n001 = node_id(i, j, k + 1)
                n101 = node_id(i + 1, j, k + 1)
                n111 = node_id(i + 1, j + 1, k + 1)
                n011 = node_id(i, j + 1, k + 1)
                hex_cells.append([n000, n100, n110, n010, n001, n101, n111, n011])
    return np.asarray(hex_cells, dtype=np.int64)


def infer_cube_resolution(num_nodes: int) -> int:
    n = round(num_nodes ** (1.0 / 3.0))
    if n ** 3 != num_nodes:
        raise ValueError(
            f"Expected a structured cube with n^3 nodes, but got {num_nodes} nodes."
        )
    return n


def write_pvd(pvd_path: Path, entries: Iterable[tuple[float, str]]) -> None:
    with pvd_path.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for tval, fname in entries:
            f.write(f'    <DataSet timestep="{tval:.8f}" group="" part="0" file="{fname}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


def load_case_ids_from_txt(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find case list file:\n{path}")
    case_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not case_ids:
        raise ValueError(f"Case list file is empty:\n{path}")
    return case_ids


def export_one_case(case_id: str, seq_path: Path, out_dir: Path) -> None:
    if not seq_path.exists():
        raise FileNotFoundError(f"Cannot find seq file:\n{seq_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(seq_path)

    mesh_pos = data["mesh_pos"]
    world_pos = data["world_pos"]
    disp = data["disp"]
    disp_mag = data["disp_mag"]
    stress_vm = data["stress_vm"]
    fixed_mask = data["fixed_mask"]
    node_type = data["node_type"]
    time_arr = data["time"]
    frame_id = data["frame_id"]

    num_frames = world_pos.shape[0]
    num_nodes = mesh_pos.shape[0]
    n = infer_cube_resolution(num_nodes)
    hex_cells = build_hex_cells(n, n, n)

    pvd_entries: List[tuple[float, str]] = []

    print(f"[{case_id}] num_frames={num_frames}, num_nodes={num_nodes}, inferred_grid={n}x{n}x{n}")
    for t in range(num_frames):
        frame_min = float(stress_vm[t].min())
        frame_max = float(stress_vm[t].max())
        print(f"[{case_id}] frame {t:04d}: stress_vm min = {frame_min:.6e}, max = {frame_max:.6e}")

        mesh = meshio.Mesh(
            points=world_pos[t].astype(np.float32),
            cells=[("hexahedron", hex_cells)],
            point_data={
                "U": disp[t].astype(np.float32),
                "disp_mag": disp_mag[t].astype(np.float32),
                "stress_vm": stress_vm[t].astype(np.float32),
                "fixed_mask": fixed_mask.astype(np.int32),
                "node_type": node_type.astype(np.int32),
            },
            field_data={
                "time_value": np.array([float(time_arr[t])], dtype=np.float64),
                "frame_id": np.array([int(frame_id[t])], dtype=np.int32),
            },
        )

        vtu_name = f"frame_{t:04d}.vtu"
        mesh.write(out_dir / vtu_name)
        pvd_entries.append((float(time_arr[t]), vtu_name))

    pvd_path = out_dir / f"{case_id}.pvd"
    write_pvd(pvd_path, pvd_entries)

    print(f"[{case_id}] Done.")
    print(f"[{case_id}] SEQ_PATH: {seq_path}")
    print(f"[{case_id}] OUT_DIR : {out_dir}")
    print(f"[{case_id}] Open in ParaView: {pvd_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export analytic cube seq_case_*.npz files to ParaView VTU/PVD for Abaqus_version3."
    )
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        default=Path(r"C:\Users\User\Desktop\Abaqus_version3\train_dataset\preprocess\cases_v3"),
        help="Root folder containing 02_seq_npz and 05_checks.",
    )
    parser.add_argument(
        "--analytic-root",
        type=Path,
        default=Path(r"C:\Users\User\Desktop\Abaqus_version3\train_dataset\analytic\cases_v3"),
        help="Root folder where paraview_case_xxxxx folders will be written.",
    )
    parser.add_argument(
        "--case-id",
        nargs="+",
        help="One or more case IDs to export, e.g. --case-id case_00000 case_00017",
    )
    parser.add_argument(
        "--case-list-file",
        type=Path,
        default=None,
        help="Optional text file with one case_id per line. If omitted and --case-id is not given, uses 05_checks/paraview_case_ids.txt",
    )
    args = parser.parse_args()

    preprocess_root = args.preprocess_root
    analytic_root = args.analytic_root
    seq_root = preprocess_root / "02_seq_npz"

    if args.case_id:
        case_ids = args.case_id
    else:
        case_list_file = args.case_list_file or (preprocess_root / "05_checks" / "paraview_case_ids.txt")
        case_ids = load_case_ids_from_txt(case_list_file)

    analytic_root.mkdir(parents=True, exist_ok=True)

    print(f"preprocess_root = {preprocess_root}")
    print(f"analytic_root   = {analytic_root}")
    print(f"num_cases       = {len(case_ids)}")

    for case_id in case_ids:
        seq_path = seq_root / f"seq_{case_id}.npz"
        out_dir = analytic_root / f"paraview_{case_id}"
        export_one_case(case_id=case_id, seq_path=seq_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
