"""
export_4case_paraview.py
=========================
Run rollout for all 4 cases and export GT + predictions to VTU/PVD for ParaView.

Output structure:
  export_4case_paraview/
    sample_XXXXX/
      gt/     t_000.vtu ...  sample_XXXXX_gt.pvd
      case1/  t_000.vtu ...  sample_XXXXX_case1.pvd
      case2/  t_000.vtu ...  sample_XXXXX_case2.pvd
      case3/  t_000.vtu ...  sample_XXXXX_case3.pvd
      case4/  t_000.vtu ...  sample_XXXXX_case4.pvd

GT PointData:
  displacement, disp_mag, abaqus_stress, bmat_stress

Pred PointData (all cases):
  displacement, disp_mag, bmat_stress_pred, stress_err_bmat, disp_error

Pred PointData (Case 2 only, additional):
  direct_stress_pred, stress_err_direct
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint
from fem_stress import compute_nodal_vm_stress

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = "./04_preprocessed_pt_bimaterial"
NPZ_DIR    = "./02_abaqus_npz_bimaterial"
OUTPUT_DIR = "./export_4case_paraview"

SAMPLE_INDICES = [0, 1, 2, 1000, 1001, 1002]

CASE_CONFIGS = [
    {"name": "case1", "ckpt": "./checkpoints_bimaterial_v2",    "n_out": 3, "direct_stress": False},
    {"name": "case2", "ckpt": "./checkpoints_bimaterial_case2", "n_out": 4, "direct_stress": True},
    {"name": "case3", "ckpt": "./checkpoints_bimaterial_case3", "n_out": 3, "direct_stress": False},
    {"name": "case4", "ckpt": "./checkpoints_bimaterial_v6",    "n_out": 3, "direct_stress": False},
]

HIDDEN   = 128
NUM_EDGE = 9

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

# ── Stats ──────────────────────────────────────────────────────────────────────

node_stats  = load_json(os.path.join(DATA_DIR, "node_stats_bimaterial.json"))
v_mean      = node_stats["velocity_mean"].to(device)
v_std       = node_stats["velocity_std"].to(device)
d_mean      = node_stats["disp_mean"].to(device)
d_std       = node_stats["disp_std"].to(device)
bmat_s_mean = torch.tensor(node_stats.get("bmat_stress_mean", 0.0),
                            dtype=torch.float32, device=device)
bmat_s_std  = torch.tensor(node_stats.get("bmat_stress_std",  1.0),
                            dtype=torch.float32, device=device).clamp(min=1e-6)
v_mean_cpu  = v_mean.cpu()
v_std_cpu   = v_std.cpu()

edge_stats = load_json("edge_stats_bimaterial.json")
e_mean = edge_stats["edge_mean"].to(device)
e_std  = edge_stats["edge_std"].to(device)

# ── VTU / PVD helpers ─────────────────────────────────────────────────────────

_FACE_LOCAL = [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]]


def build_surface_cells(points_np, elem_conn_np):
    face_map = {}
    for elem in elem_conn_np:
        for face_local in _FACE_LOCAL:
            g   = elem[np.array(face_local)]
            key = tuple(sorted(g.tolist()))
            face_map[key] = None if key in face_map else g.copy()
    surface_faces = [v for v in face_map.values() if v is not None]
    mesh_center   = points_np.mean(axis=0)
    conn_list, off_list, type_list = [], [], []
    cur_off = 0
    for face_nodes in surface_faces:
        pts    = points_np[face_nodes]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.dot(normal, pts.mean(axis=0) - mesh_center) < 0:
            face_nodes = face_nodes[::-1].copy()
        conn_list.extend(face_nodes.tolist())
        cur_off += 4
        off_list.append(cur_off)
        type_list.append(9)
    return (np.array(conn_list, dtype=np.int32),
            np.array(off_list,  dtype=np.int32),
            np.array(type_list, dtype=np.uint8))


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


def write_pvd(path, entries):
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t_val, vtu_rel in entries:
        lines.append(f'    <DataSet timestep="{t_val:.4f}" group="" part="0" file="{vtu_rel}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines))

# ── Rollout helpers ────────────────────────────────────────────────────────────

def build_multigraph(elem_conn, elem_mat, mat_E):
    seen = set()
    rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E[mat_idx])
        for (a, b) in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst, int(mat_idx))
                if key not in seen:
                    seen.add(key)
                    rows.append(src); cols.append(dst); E_vals.append(E_elem)
    return np.array([rows, cols], dtype=np.int64), np.array(E_vals, dtype=np.float32)


def build_ef(mesh_pos, world_pos, edge_index, edge_E, E_ref):
    src, dst = edge_index
    d_mesh  = mesh_pos[dst]  - mesh_pos[src]
    d_world = world_pos[dst] - world_pos[src]
    ef = torch.cat([
        d_mesh,  d_mesh.norm(dim=1, keepdim=True),
        d_world, d_world.norm(dim=1, keepdim=True),
        (edge_E / E_ref).unsqueeze(1),
    ], dim=1)
    return (ef - e_mean) / e_std


def load_model(ckpt, n_out):
    m = HybridMeshGraphNet(
        9, NUM_EDGE, n_out,
        processor_size=4,
        hidden_dim_processor=HIDDEN, hidden_dim_node_encoder=HIDDEN,
        hidden_dim_edge_encoder=HIDDEN, hidden_dim_node_decoder=HIDDEN,
        mlp_activation_fn="relu", do_concat_trick=False,
        num_processor_checkpoint_segments=0, recompute_activation=False,
    ).to(device)
    m.eval()
    load_checkpoint(ckpt, models=m, device=device)
    return m


def run_rollout(model, data, ref_pos, node_type, edge_index, edge_E, E1,
                gt_wp, T, N, direct_stress):
    """Returns pred_wp (list of T+1 tensors) and pred_direct_stress (list, only if direct_stress=True)."""
    empty_wef = torch.zeros(0, NUM_EDGE, device=device)
    cur      = gt_wp[1].to(device)
    prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    pred_wp  = [gt_wp[0], gt_wp[1]]
    direct_s_list = []

    with torch.inference_mode():
        for _ in range(T - 1):
            ef  = build_ef(ref_pos, cur, edge_index, edge_E, E1)
            nx  = torch.cat([
                (cur - ref_pos - d_mean) / d_std,
                (prev_vel - v_mean) / v_std,
                node_type,
            ], dim=1)
            out      = model(nx, ef, empty_wef,
                             Data(x=nx, edge_index=edge_index, num_nodes=N))
            vel      = out[:, :3] * v_std + v_mean
            prev_vel = vel
            cur      = cur + vel
            pred_wp.append(cur.clone())
            if direct_stress:
                direct_s_list.append(
                    (out[:, 3] * bmat_s_std + bmat_s_mean).cpu().numpy()
                )

    return pred_wp, direct_s_list

# ── Load all models ────────────────────────────────────────────────────────────

print("Loading models...")
models = []
for cfg in CASE_CONFIGS:
    m = load_model(cfg["ckpt"], cfg["n_out"])
    models.append(m)
    print(f"  {cfg['name']}: {cfg['ckpt']} (n_out={cfg['n_out']})")

# ── Main export loop ───────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"\nFound {len(all_files)} .pt files")

for idx in SAMPLE_INDICES:
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range"); continue

    fname   = all_files[idx]
    sid_str = fname.replace(".pt", "")
    print(f"\n{'='*60}")
    print(f"[{sid_str}]")

    data = torch.load(os.path.join(DATA_DIR, fname), map_location="cpu", weights_only=False)
    npz  = np.load(os.path.join(NPZ_DIR, sid_str + ".npz"))
    T    = len(data)
    N    = data[0]["graph"].num_nodes
    E1   = float(npz["mat_E"][0])

    # Topology
    ei_np, eE_np = build_multigraph(npz["elem_conn"], npz["elem_mat"], npz["mat_E"])
    edge_index   = torch.from_numpy(ei_np).to(device)
    edge_E       = torch.from_numpy(eE_np).to(device)
    ref_pos      = data[0]["graph"].mesh_pos.to(device)
    node_type    = data[0]["graph"].x[:, 6:9].to(device)
    ref_np       = ref_pos.cpu().numpy()

    # B-matrix material arrays
    elem_mat_idx = npz["elem_mat"]
    elem_E_t     = torch.from_numpy(npz["mat_E"][elem_mat_idx].astype(np.float64)).to(device)
    elem_nu_t    = torch.from_numpy(npz["mat_nu"][elem_mat_idx].astype(np.float64)).to(device)
    elem_conn_t  = torch.from_numpy(npz["elem_conn"].astype(np.int64)).to(device)

    # VTK surface cells (once per sample)
    connectivity, offsets, types = build_surface_cells(ref_np, npz["elem_conn"])

    # GT positions
    gt_wp = [data[0]["graph"].world_pos.clone()]
    for step in data:
        gt_wp.append(gt_wp[-1] + step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)

    gt_abaqus_stress = npz["stress_vm"]   # (T+1, N) Abaqus nodal stress

    # ── Write GT ──────────────────────────────────────────────────────────────
    gt_dir = os.path.join(OUTPUT_DIR, sid_str, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    gt_pvd = []

    for t in range(T):
        gt_pos_np = gt_wp[t + 1].numpy().astype(np.float32)
        gt_abaqus = gt_abaqus_stress[t + 1].astype(np.float32)
        gt_bmat   = data[t]["graph"].bmat_stress.numpy().astype(np.float32)
        vtu_name  = f"t_{t:03d}.vtu"
        write_vtu(
            os.path.join(gt_dir, vtu_name),
            gt_pos_np, connectivity, offsets, types,
            {
                "displacement":  (gt_pos_np - ref_np),
                "disp_mag":      np.linalg.norm(gt_pos_np - ref_np, axis=1),
                "abaqus_stress": gt_abaqus,
                "bmat_stress":   gt_bmat,
            },
        )
        gt_pvd.append((t, vtu_name))
    write_pvd(os.path.join(gt_dir, f"{sid_str}_gt.pvd"), gt_pvd)
    print(f"  GT written ({T} frames)")

    # ── Write each case ───────────────────────────────────────────────────────
    for cfg, model in zip(CASE_CONFIGS, models):
        case_name = cfg["name"]
        print(f"  Running {case_name}...")

        pred_wp, direct_s_list = run_rollout(
            model, data, ref_pos, node_type, edge_index, edge_E, E1,
            gt_wp, T, N, cfg["direct_stress"]
        )

        case_dir = os.path.join(OUTPUT_DIR, sid_str, case_name)
        os.makedirs(case_dir, exist_ok=True)
        case_pvd = []

        for t in range(T):
            gt_pos_np   = gt_wp[t + 1].numpy().astype(np.float32)
            pred_pos_np = pred_wp[t + 1].cpu().numpy().astype(np.float32)
            gt_bmat     = data[t]["graph"].bmat_stress.numpy().astype(np.float32)
            vtu_name    = f"t_{t:03d}.vtu"

            # B-matrix stress from predicted positions
            with torch.inference_mode():
                pred_bmat = compute_nodal_vm_stress(
                    torch.from_numpy(pred_pos_np - ref_np).to(device),
                    ref_pos, elem_conn_t, 8,
                    elem_E=elem_E_t, elem_nu=elem_nu_t,
                ).cpu().numpy().astype(np.float32)

            point_data = {
                "displacement":     (pred_pos_np - ref_np),
                "disp_mag":         np.linalg.norm(pred_pos_np - ref_np, axis=1),
                "bmat_stress_pred": pred_bmat,
                "stress_err_bmat":  np.abs(pred_bmat - gt_bmat),
                "disp_error":       np.linalg.norm(pred_pos_np - gt_pos_np, axis=1),
            }

            if cfg["direct_stress"]:
                direct_s = direct_s_list[t - 1].astype(np.float32) if t > 0 else gt_bmat
                point_data["direct_stress_pred"] = direct_s
                point_data["stress_err_direct"]  = np.abs(direct_s - gt_bmat)

            write_vtu(
                os.path.join(case_dir, vtu_name),
                pred_pos_np, connectivity, offsets, types,
                point_data,
            )
            case_pvd.append((t, vtu_name))

        write_pvd(os.path.join(case_dir, f"{sid_str}_{case_name}.pvd"), case_pvd)

        final_disp_err = np.linalg.norm(
            pred_wp[-1].cpu().numpy() - gt_wp[-1].numpy(), axis=1
        ).mean()
        print(f"    final mean disp error: {final_disp_err*1000:.3f} mm")

print("\n完成！ParaView 開各 sample 下各 case 的 .pvd 檔。")
