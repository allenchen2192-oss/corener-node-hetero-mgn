#!/usr/bin/env python3
"""Preprocess analytic cube dataset v3.

Pipeline:
    00_manifest/case_table.csv
        -> 02_seq_npz/seq_case_*.npz
        -> 03_graph_npz/graph_case_*.npz
        -> 04_preprocessed_pt/sample_*.pt
        -> 04_preprocessed_pt/node_stats.json
        -> 04_preprocessed_pt/edge_stats.json
        -> 05_checks/*.csv

Important notes:
- sequence keys intentionally follow the user's working v2 style, e.g. use "velocity"
  instead of "velocity_step".
- by default, sample_*.pt stores NORMALIZED graph.y / mesh_edge_features / world_edge_features,
  which is closer to NVIDIA's official deforming_plate preprocessing flow.
- world-edge handling is configurable:
    * mesh_fallback (default): if no world edges are found, duplicate mesh edges as fallback
      for maximum HybridMeshGraphNet compatibility.
    * empty: keep world edges/features empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

EDGE_DIM = 8
EPS = 1.0e-8
CASE_HEADER = [
    "case_id",
    "ax0", "ax1", "ax2", "ax3",
    "ay0", "ay1", "ay2", "ay3",
    "az0", "az1", "az2", "az3",
    "H",
    "num_frames",
    "n_nodes_axis",
    "time_profile",
    "E",
    "nu",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    coeff: np.ndarray
    H: float
    num_frames: int
    n_nodes_axis: int
    time_profile: str
    E: float
    nu: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def quarter_sine(num_frames: int) -> np.ndarray:
    n = np.arange(num_frames, dtype=np.float64)
    return np.sin(0.5 * math.pi * n / (num_frames - 1))


def build_structured_grid(n: int, H: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, H, n)
    ys = np.linspace(0.0, H, n)
    zs = np.linspace(0.0, H, n)
    coords = []
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                coords.append([x, y, z])
    mesh_pos = np.asarray(coords, dtype=np.float32)
    fixed_mask = np.zeros((mesh_pos.shape[0],), dtype=bool)
    node_type = np.zeros((mesh_pos.shape[0],), dtype=np.int64)
    return mesh_pos, fixed_mask, node_type


def build_mesh_edge_index(n: int) -> np.ndarray:
    def node_id(i: int, j: int, k: int) -> int:
        return k * n * n + j * n + i

    edges = []
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
    return np.asarray(edges, dtype=np.int64).T


def radius_edge_index(world_pos: np.ndarray, radius: float) -> np.ndarray:
    num_nodes = world_pos.shape[0]
    edges = []
    for i in range(num_nodes):
        diff = world_pos - world_pos[i]
        dist = np.linalg.norm(diff, axis=1)
        neighbors = np.where((dist <= radius) & (dist > 0.0))[0]
        for j in neighbors:
            edges.append([i, j])
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def displacement_field(mesh_pos: np.ndarray, lam: float, coeff: np.ndarray, H: float) -> np.ndarray:
    x = mesh_pos[:, 0:1].astype(np.float32)
    y = mesh_pos[:, 1:2].astype(np.float32)
    z = mesh_pos[:, 2:3].astype(np.float32)
    one = np.ones_like(x)
    basis = np.concatenate([one, x, y, z], axis=1)   # [N, 4]
    disp = lam * (basis @ coeff.T) / H               # [N, 3]
    return disp.astype(np.float32)


def compute_displacement_sequence(mesh_pos: np.ndarray, lambdas: np.ndarray, coeff: np.ndarray, H: float) -> np.ndarray:
    return np.stack([displacement_field(mesh_pos, lam, coeff, H) for lam in lambdas], axis=0).astype(np.float32)


def compute_world_pos_sequence(mesh_pos: np.ndarray, disp: np.ndarray) -> np.ndarray:
    return (mesh_pos[None, :, :] + disp).astype(np.float32)


def compute_velocity_sequence(world_pos: np.ndarray) -> np.ndarray:
    return (world_pos[1:] - world_pos[:-1]).astype(np.float32)


def displacement_gradient_at_points(points: np.ndarray, lam: float, coeff: np.ndarray, H: float) -> np.ndarray:
    num_nodes = points.shape[0]
    grad = np.zeros((num_nodes, 3, 3), dtype=np.float64)
    grad_const = (lam / H) * coeff[:, 1:4]
    grad[:] = grad_const[None, :, :]
    return grad


def strain_from_grad_u(grad_u: np.ndarray) -> np.ndarray:
    return 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))


def stress_from_strain(strain: np.ndarray, E: float, nu: float) -> np.ndarray:
    mu = E / (2.0 * (1.0 + nu))
    lam_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    tr = np.trace(strain, axis1=1, axis2=2)[:, None, None]
    eye = np.eye(3, dtype=np.float64)[None, :, :]
    sigma = lam_lame * tr * eye + 2.0 * mu * strain
    return sigma


def von_mises_from_stress(stress: np.ndarray) -> np.ndarray:
    sxx = stress[:, 0, 0]
    syy = stress[:, 1, 1]
    szz = stress[:, 2, 2]
    sxy = stress[:, 0, 1]
    syz = stress[:, 1, 2]
    szx = stress[:, 2, 0]
    vm = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )
    return vm.astype(np.float32)


def compute_stress_vm_sequence(mesh_pos: np.ndarray, lambdas: np.ndarray, coeff: np.ndarray, H: float, E: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    vm_frames = []
    strain6_frames = []
    for lam in lambdas:
        grad_u = displacement_gradient_at_points(mesh_pos, lam, coeff, H)
        strain = strain_from_grad_u(grad_u)
        stress = stress_from_strain(strain, E, nu)
        vm = von_mises_from_stress(stress)
        strain6 = np.stack([
            strain[:, 0, 0], strain[:, 1, 1], strain[:, 2, 2],
            strain[:, 0, 1], strain[:, 1, 2], strain[:, 2, 0],
        ], axis=1).astype(np.float32)
        vm_frames.append(vm)
        strain6_frames.append(strain6)
    return np.stack(vm_frames, axis=0).astype(np.float32), np.stack(strain6_frames, axis=0).astype(np.float32)


def edge_features_from_positions(edge_index: np.ndarray, mesh_pos: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.empty((0, EDGE_DIM), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    d_mesh = mesh_pos[dst] - mesh_pos[src]
    n_mesh = np.linalg.norm(d_mesh, axis=1, keepdims=True)
    d_world = world_pos[dst] - world_pos[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    return np.concatenate([d_mesh, n_mesh, d_world, n_world], axis=1).astype(np.float32)


def robust_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-8, 1.0, std).astype(np.float32)
    return mean, std


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "cases_v3"
    parser = argparse.ArgumentParser(description="Preprocess analytic cube dataset v3")
    parser.add_argument("--root", type=Path, default=default_root, help="Root folder containing 00_manifest/case_table.csv")
    parser.add_argument("--world-radius", type=float, default=None, help="Radius for optional dynamic world edges. Default = 1.08 * grid spacing")
    parser.add_argument("--world-edge-policy", choices=["mesh_fallback", "empty"], default="mesh_fallback", help="How to handle missing world edges")
    parser.add_argument("--normalize-pt", choices=["true", "false"], default="true", help="Store normalized targets/features in sample_*.pt (official-like)")
    parser.add_argument("--case-indices", type=str, default="", help="Optional comma-separated case indices, e.g. 0,3,10")
    parser.add_argument("--paraview-count", type=int, default=5)
    parser.add_argument("--paraview-seed", type=int, default=42)
    return parser.parse_args()


def parse_case_table(case_table_path: Path) -> List[CaseSpec]:
    if not case_table_path.exists():
        raise FileNotFoundError(f"Cannot find {case_table_path}")
    with case_table_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("case_table.csv is empty")

    specs: List[CaseSpec] = []
    for row in rows:
        coeff = np.asarray([
            [float(row["ax0"]), float(row["ax1"]), float(row["ax2"]), float(row["ax3"])],
            [float(row["ay0"]), float(row["ay1"]), float(row["ay2"]), float(row["ay3"])],
            [float(row["az0"]), float(row["az1"]), float(row["az2"]), float(row["az3"])],
        ], dtype=np.float32)
        specs.append(CaseSpec(
            case_id=row["case_id"],
            coeff=coeff,
            H=float(row["H"]),
            num_frames=int(row["num_frames"]),
            n_nodes_axis=int(row["n_nodes_axis"]),
            time_profile=row["time_profile"],
            E=float(row["E"]),
            nu=float(row["nu"]),
        ))
    return specs


def filter_specs(specs: List[CaseSpec], case_indices_arg: str) -> List[CaseSpec]:
    if not case_indices_arg.strip():
        return specs
    wanted = {int(x.strip()) for x in case_indices_arg.split(",") if x.strip()}
    filtered = [spec for i, spec in enumerate(specs) if i in wanted]
    if not filtered:
        raise ValueError(f"No matching cases for --case-indices={case_indices_arg!r}")
    return filtered


def save_seq_npz(
    out_path: Path,
    mesh_pos: np.ndarray,
    world_pos: np.ndarray,
    disp: np.ndarray,
    disp_mag: np.ndarray,
    velocity: np.ndarray,
    stress_vm: np.ndarray,
    strain_6: np.ndarray,
    fixed_mask: np.ndarray,
    node_type: np.ndarray,
    coeff: np.ndarray,
    lambdas: np.ndarray,
    H: float,
    E: float,
    nu: float,
) -> None:
    num_frames = world_pos.shape[0]
    np.savez_compressed(
        out_path,
        mesh_pos=mesh_pos,
        world_pos=world_pos,
        disp=disp,
        disp_mag=disp_mag,
        velocity=velocity,
        stress_vm=stress_vm,
        strain_6=strain_6,
        fixed_mask=fixed_mask.astype(np.bool_),
        node_type=node_type.astype(np.int64),
        coeff=coeff.astype(np.float32),
        lambda_arr=lambdas.astype(np.float32),
        H=np.asarray(H, dtype=np.float32),
        E=np.asarray(E, dtype=np.float32),
        nu=np.asarray(nu, dtype=np.float32),
        time=np.linspace(0.0, 1.0, num_frames, dtype=np.float32),
        frame_id=np.arange(num_frames, dtype=np.int64),
    )


def save_graph_npz(
    out_path: Path,
    mesh_pos: np.ndarray,
    world_pos: np.ndarray,
    velocity: np.ndarray,
    stress_vm: np.ndarray,
    node_type: np.ndarray,
    fixed_mask: np.ndarray,
    mesh_edge_index: np.ndarray,
    lambdas: np.ndarray,
    coeff: np.ndarray,
    H: float,
    E: float,
    nu: float,
    world_radius: float,
    world_edge_policy: str,
) -> None:
    num_frames = world_pos.shape[0]
    num_steps = num_frames - 1
    payload: Dict[str, np.ndarray] = {
        "mesh_pos": mesh_pos,
        "node_type": node_type.astype(np.int64),
        "fixed_mask": fixed_mask.astype(np.bool_),
        "mesh_edge_index": mesh_edge_index.astype(np.int64),
        "coeff": coeff.astype(np.float32),
        "lambda_arr": lambdas.astype(np.float32),
        "H": np.asarray(H, dtype=np.float32),
        "E": np.asarray(E, dtype=np.float32),
        "nu": np.asarray(nu, dtype=np.float32),
        "time": np.linspace(0.0, 1.0, num_frames, dtype=np.float32),
        "frame_id": np.arange(num_frames, dtype=np.int64),
        "world_edge_count": np.zeros((num_steps,), dtype=np.int64),
    }

    for t in range(num_steps):
        x_t = world_pos[t]
        y_t = np.concatenate([velocity[t], stress_vm[t + 1][:, None]], axis=1).astype(np.float32)

        world_edge_index = radius_edge_index(x_t, radius=world_radius)
        if world_edge_index.shape[1] == 0 and world_edge_policy == "mesh_fallback":
            world_edge_index = mesh_edge_index.copy()
        payload["world_edge_count"][t] = world_edge_index.shape[1]

        edge_index = np.concatenate([mesh_edge_index, world_edge_index], axis=1).astype(np.int64)
        mesh_feats = edge_features_from_positions(mesh_edge_index, mesh_pos, x_t)
        world_feats = edge_features_from_positions(world_edge_index, mesh_pos, x_t)

        payload[f"x_{t:03d}"] = x_t.astype(np.float32)
        payload[f"y_{t:03d}"] = y_t.astype(np.float32)
        payload[f"edge_index_{t:03d}"] = edge_index
        payload[f"world_edge_index_{t:03d}"] = world_edge_index.astype(np.int64)
        payload[f"mesh_edge_features_{t:03d}"] = mesh_feats
        payload[f"world_edge_features_{t:03d}"] = world_feats

    np.savez_compressed(out_path, **payload)


def build_pt_sample(
    mesh_pos: np.ndarray,
    world_pos: np.ndarray,
    velocity: np.ndarray,
    stress_vm: np.ndarray,
    node_type: np.ndarray,
    fixed_mask: np.ndarray,
    mesh_edge_index: np.ndarray,
    coeff: np.ndarray,
    world_radius: float,
    world_edge_policy: str,
    normalize_pt: bool,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    edge_mean: np.ndarray,
    edge_std: np.ndarray,
) -> Tuple[List[Dict[str, torch.Tensor]], np.ndarray, np.ndarray]:
    num_frames = world_pos.shape[0]
    num_steps = num_frames - 1

    steps: List[Dict[str, torch.Tensor]] = []
    all_y = []
    all_edge_feats = []

    for t in range(num_steps):
        x_t = world_pos[t].astype(np.float32)
        y_raw = np.concatenate([velocity[t], stress_vm[t + 1][:, None]], axis=1).astype(np.float32)

        world_edge_index = radius_edge_index(x_t, radius=world_radius)
        if world_edge_index.shape[1] == 0 and world_edge_policy == "mesh_fallback":
            world_edge_index = mesh_edge_index.copy()

        edge_index = np.concatenate([mesh_edge_index, world_edge_index], axis=1).astype(np.int64)
        mesh_feats_raw = edge_features_from_positions(mesh_edge_index, mesh_pos, x_t)
        world_feats_raw = edge_features_from_positions(world_edge_index, mesh_pos, x_t)

        if normalize_pt:
            y_t = ((y_raw - target_mean[None, :]) / target_std[None, :]).astype(np.float32)
            mesh_feats = ((mesh_feats_raw - edge_mean[None, :]) / edge_std[None, :]).astype(np.float32)
            world_feats = ((world_feats_raw - edge_mean[None, :]) / edge_std[None, :]).astype(np.float32)
        else:
            y_t = y_raw
            mesh_feats = mesh_feats_raw
            world_feats = world_feats_raw

        graph = Data(
            x=torch.from_numpy(x_t),
            y=torch.from_numpy(y_t),
            edge_index=torch.from_numpy(edge_index),
            num_nodes=x_t.shape[0],
        )
        graph.mesh_pos = torch.from_numpy(mesh_pos.astype(np.float32))
        graph.world_pos = torch.from_numpy(x_t)
        graph.node_type = torch.from_numpy(node_type.astype(np.int64))
        graph.fixed_mask = torch.from_numpy(fixed_mask.astype(np.bool_))
        graph.case_coeff = torch.from_numpy(coeff.astype(np.float32))
        graph.step_index = torch.tensor([t], dtype=torch.int64)

        item = {
            "graph": graph,
            "mesh_edge_features": torch.from_numpy(mesh_feats),
            "world_edge_features": torch.from_numpy(world_feats),
        }
        steps.append(item)

        all_y.append(y_raw)
        if mesh_feats_raw.shape[0] > 0:
            all_edge_feats.append(mesh_feats_raw)
        if world_feats_raw.shape[0] > 0:
            all_edge_feats.append(world_feats_raw)

    y_cat = np.concatenate(all_y, axis=0) if all_y else np.empty((0, 4), dtype=np.float32)
    edge_cat = np.concatenate(all_edge_feats, axis=0) if all_edge_feats else np.empty((0, EDGE_DIM), dtype=np.float32)
    return steps, y_cat, edge_cat


def case_numeric_id(case_id: str) -> int:
    try:
        return int(case_id.split("_")[-1])
    except Exception as exc:
        raise ValueError(f"Invalid case_id format: {case_id!r}. Expected e.g. case_00000") from exc


def main() -> None:
    args = parse_args()
    normalize_pt = args.normalize_pt.lower() == "true"

    root = args.root
    manifest_dir = root / "00_manifest"
    seq_dir = root / "02_seq_npz"
    graph_dir = root / "03_graph_npz"
    pt_dir = root / "04_preprocessed_pt"
    checks_dir = root / "05_checks"
    for d in [manifest_dir, seq_dir, graph_dir, pt_dir, checks_dir]:
        ensure_dir(d)

    case_table_path = manifest_dir / "case_table.csv"
    specs = filter_specs(parse_case_table(case_table_path), args.case_indices)
    ref = specs[0]

    mesh_pos, fixed_mask, node_type = build_structured_grid(n=ref.n_nodes_axis, H=ref.H)
    mesh_edge_index = build_mesh_edge_index(ref.n_nodes_axis)
    spacing = ref.H / (ref.n_nodes_axis - 1)
    world_radius = args.world_radius if args.world_radius is not None else 1.08 * spacing

    summary_rows = []
    all_y_train = []
    all_edge_train = []
    seq_cache = []

    # Pass 1: build raw seq + graph npz and collect RAW stats from the selected training cases.
    for spec in specs:
        lambdas = quarter_sine(num_frames=spec.num_frames)
        disp = compute_displacement_sequence(mesh_pos, lambdas, spec.coeff, spec.H)
        world_pos = compute_world_pos_sequence(mesh_pos, disp)
        disp_mag = np.linalg.norm(disp, axis=2).astype(np.float32)
        velocity = compute_velocity_sequence(world_pos)
        stress_vm, strain_6 = compute_stress_vm_sequence(mesh_pos, lambdas, spec.coeff, spec.H, spec.E, spec.nu)

        seq_path = seq_dir / f"seq_{spec.case_id}.npz"
        graph_path = graph_dir / f"graph_{spec.case_id}.npz"

        save_seq_npz(
            out_path=seq_path,
            mesh_pos=mesh_pos,
            world_pos=world_pos,
            disp=disp,
            disp_mag=disp_mag,
            velocity=velocity,
            stress_vm=stress_vm,
            strain_6=strain_6,
            fixed_mask=fixed_mask,
            node_type=node_type,
            coeff=spec.coeff,
            lambdas=lambdas,
            H=spec.H,
            E=spec.E,
            nu=spec.nu,
        )
        save_graph_npz(
            out_path=graph_path,
            mesh_pos=mesh_pos,
            world_pos=world_pos,
            velocity=velocity,
            stress_vm=stress_vm,
            node_type=node_type,
            fixed_mask=fixed_mask,
            mesh_edge_index=mesh_edge_index,
            lambdas=lambdas,
            coeff=spec.coeff,
            H=spec.H,
            E=spec.E,
            nu=spec.nu,
            world_radius=world_radius,
            world_edge_policy=args.world_edge_policy,
        )

        seq_cache.append((spec, world_pos, velocity, stress_vm, strain_6))

        # collect RAW stats like the current custom v2 approach; normalization happens after stats are known
        all_y_train.append(np.concatenate([velocity.reshape(-1, 3), stress_vm[1:].reshape(-1, 1)], axis=1).astype(np.float32))
        for t in range(spec.num_frames - 1):
            x_t = world_pos[t]
            world_edge_index = radius_edge_index(x_t, radius=world_radius)
            if world_edge_index.shape[1] == 0 and args.world_edge_policy == "mesh_fallback":
                world_edge_index = mesh_edge_index.copy()
            mesh_feats_raw = edge_features_from_positions(mesh_edge_index, mesh_pos, x_t)
            world_feats_raw = edge_features_from_positions(world_edge_index, mesh_pos, x_t)
            if mesh_feats_raw.shape[0] > 0:
                all_edge_train.append(mesh_feats_raw)
            if world_feats_raw.shape[0] > 0:
                all_edge_train.append(world_feats_raw)

        abs_strain_max = np.max(np.abs(strain_6), axis=(0, 1))
        summary_rows.append({
            "case_id": spec.case_id,
            "num_nodes": int(mesh_pos.shape[0]),
            "num_mesh_edges": int(mesh_edge_index.shape[1]),
            "num_steps": int(spec.num_frames - 1),
            "world_radius": float(world_radius),
            "sample_file": f"sample_{case_numeric_id(spec.case_id):05d}.pt",
            "all_free_ok": bool(fixed_mask.sum() == 0),
            "disp_mag_final_max": float(disp_mag[-1].max()),
            "stress_vm_final_max": float(stress_vm[-1].max()),
            "abs_eps_xx_max": float(abs_strain_max[0]),
            "abs_eps_yy_max": float(abs_strain_max[1]),
            "abs_eps_zz_max": float(abs_strain_max[2]),
            "abs_eps_xy_max": float(abs_strain_max[3]),
            "abs_eps_yz_max": float(abs_strain_max[4]),
            "abs_eps_zx_max": float(abs_strain_max[5]),
        })

    y_all = np.concatenate(all_y_train, axis=0) if all_y_train else np.empty((0, 4), dtype=np.float32)
    edge_all = np.concatenate(all_edge_train, axis=0) if all_edge_train else np.empty((0, EDGE_DIM), dtype=np.float32)
    y_mean, y_std = robust_stats(y_all)
    edge_mean, edge_std = robust_stats(edge_all)

    node_stats = {
        "velocity_mean": y_mean[:3].tolist(),
        "velocity_std": y_std[:3].tolist(),
        "stress_mean": float(y_mean[3]) if y_mean.shape[0] > 3 else 0.0,
        "stress_std": float(y_std[3]) if y_std.shape[0] > 3 else 1.0,
    }
    edge_stats = {
        "edge_mean": edge_mean.tolist(),
        "edge_std": edge_std.tolist(),
    }
    (pt_dir / "node_stats.json").write_text(json.dumps(node_stats, indent=2), encoding="utf-8")
    (pt_dir / "edge_stats.json").write_text(json.dumps(edge_stats, indent=2), encoding="utf-8")

    # Pass 2: build .pt samples using chosen normalization behavior.
    for spec, world_pos, velocity, stress_vm, strain_6 in seq_cache:
        steps, _, _ = build_pt_sample(
            mesh_pos=mesh_pos,
            world_pos=world_pos,
            velocity=velocity,
            stress_vm=stress_vm,
            node_type=node_type,
            fixed_mask=fixed_mask,
            mesh_edge_index=mesh_edge_index,
            coeff=spec.coeff,
            world_radius=world_radius,
            world_edge_policy=args.world_edge_policy,
            normalize_pt=normalize_pt,
            target_mean=y_mean,
            target_std=y_std,
            edge_mean=edge_mean,
            edge_std=edge_std,
        )
        out_pt = pt_dir / f"sample_{case_numeric_id(spec.case_id):05d}.pt"
        torch.save(steps, out_pt)

    # checks
    seq_summary_path = checks_dir / "seq_summary.csv"
    with seq_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    all_free_path = checks_dir / "all_free_check.csv"
    with all_free_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "all_free_ok"])
        for row in summary_rows:
            writer.writerow([row["case_id"], row["all_free_ok"]])

    case_stats_path = checks_dir / "case_stats.csv"
    with case_stats_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "disp_mag_final_max",
            "stress_vm_final_max",
            "abs_eps_xx_max",
            "abs_eps_yy_max",
            "abs_eps_zz_max",
            "abs_eps_xy_max",
            "abs_eps_yz_max",
            "abs_eps_zx_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    rng = np.random.default_rng(args.paraview_seed)
    n_pick = min(args.paraview_count, len(specs))
    picks = sorted(rng.choice(len(specs), size=n_pick, replace=False).tolist()) if n_pick > 0 else []
    with (checks_dir / "paraview_case_ids.txt").open("w", encoding="utf-8") as f:
        for idx in picks:
            f.write(f"{specs[idx].case_id}\n")

    print(f"Processed {len(specs)} cases")
    print(f"Wrote node stats to: {pt_dir / 'node_stats.json'}")
    print(f"Wrote edge stats to: {pt_dir / 'edge_stats.json'}")
    print(f"Wrote summary to: {seq_summary_path}")
    print(f"Stored sample_*.pt with normalize_pt={normalize_pt}, world_edge_policy={args.world_edge_policy}")


if __name__ == "__main__":
    main()
