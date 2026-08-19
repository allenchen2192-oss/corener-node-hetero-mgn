"""
export_abaqus_paraview.py
==========================
Run rollout with the Abaqus prevvel model and export GT + predicted results
to VTU / PVD format for side-by-side comparison in ParaView.

Output structure:
  export_abaqus_paraview/
    sample_XXXXX/
      gt/    t_000.vtu  t_001.vtu  ...  sample_XXXXX_gt.pvd
      pred/  t_000.vtu  t_001.vtu  ...  sample_XXXXX_pred.pvd

PointData fields:
  displacement  (vector, 3 components, m)
  disp_mag      (scalar, m)
  von_mises     (scalar, Pa)  — GT: from Abaqus labels; Pred: gradient-fitting
  disp_error    (scalar, m)   — pred only: ||pred - gt||

The mesh is exported as VTK_LINE cells (wireframe) from the skeleton edge
connectivity stored in graph.edge_index.

Usage:
  python export_abaqus_paraview.py
"""

import os
import numpy as np
import torch
from scipy.spatial import Delaunay
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

NUM_INPUT_FEATURES  = 6    # disp_vec(3) + prev_vel(3)
NUM_EDGE_FEATURES   = 8
NUM_OUTPUT_FEATURES = 3
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 64

DATA_DIR   = "./04_preprocessed_pt_abaqus_rect"
CKPT_PATH  = "./checkpoints_abaqus_prevvel_1000data_1000nodes"
CKPT_EPOCH = 661          # None = latest (epoch 661)
OUTPUT_DIR = "./export_abaqus_paraview_rect"

# All 10 rectangular prism OOD samples
SAMPLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

NPZ_DIR = "./02_abaqus_npz_rect"  # for elem_conn (exact Abaqus surface topology)


# ── VTU / PVD helpers ─────────────────────────────────────────────────────────

def build_surface_cells(points_np, tol=1e-4):
    """Build surface cells for each axis-aligned face of a box-shaped FEM mesh.

    If the face nodes form a structured 2-D grid, builds VTK_QUAD (type 9)
    cells that match the FEM face connectivity exactly → no interpolation
    artifacts.  Falls back to Delaunay VTK_TRIANGLE (type 5) for
    non-structured faces.  Connectivity computed once on the reference mesh
    and reused for every frame.
    """
    mins   = points_np.min(axis=0)
    maxs   = points_np.max(axis=0)
    ranges = np.where((maxs - mins) > 0, maxs - mins, 1.0)
    p_norm = (points_np - mins) / ranges   # normalize to [0, 1]^3

    # (axis, boundary_val, 2D_proj_axes, flip_winding)
    # flip=True  → outward normal needs reversal: [a,e,c,b] instead of [a,b,c,e]
    # x=0:-x  x=1:+x  y=0:-y  y=1:+y  z=0:-z  z=1:+z
    face_specs = [
        (0, 0.0, [1, 2], True),   # x=0  outward=-x  → flip
        (0, 1.0, [1, 2], False),  # x=1  outward=+x
        (1, 0.0, [0, 2], False),  # y=0  outward=-y
        (1, 1.0, [0, 2], True),   # y=1  outward=+y  → flip
        (2, 0.0, [0, 1], True),   # z=0  outward=-z  → flip
        (2, 1.0, [0, 1], False),  # z=1  outward=+z
    ]

    conn_list, off_list, type_list = [], [], []
    cur_off = 0

    for axis, val, proj, flip in face_specs:
        mask = np.abs(p_norm[:, axis] - val) < tol
        idx  = np.where(mask)[0]
        if len(idx) < 3:
            continue

        pts_2d = p_norm[idx][:, proj]
        u0 = np.sort(np.unique(np.round(pts_2d[:, 0], 8)))
        u1 = np.sort(np.unique(np.round(pts_2d[:, 1], 8)))
        n0, n1 = len(u0), len(u1)

        if n0 * n1 == len(idx):
            # Structured grid → VTK_QUAD cells
            p0_idx = np.searchsorted(u0, np.round(pts_2d[:, 0], 8))
            p1_idx = np.searchsorted(u1, np.round(pts_2d[:, 1], 8))
            grid = np.full((n0, n1), -1, dtype=np.int32)
            for k in range(len(idx)):
                grid[p0_idx[k], p1_idx[k]] = idx[k]
            for i in range(n0 - 1):
                for j in range(n1 - 1):
                    a, b, c, e = grid[i,j], grid[i+1,j], grid[i+1,j+1], grid[i,j+1]
                    if a >= 0 and b >= 0 and c >= 0 and e >= 0:
                        quad = [a, e, c, b] if flip else [a, b, c, e]
                        conn_list.extend(quad)
                        cur_off += 4
                        off_list.append(cur_off)
                        type_list.append(9)   # VTK_QUAD
        else:
            # Unstructured → Delaunay VTK_TRIANGLE fallback
            tri = Delaunay(pts_2d)
            for t in tri.simplices:
                conn_list.extend(idx[t].tolist())
                cur_off += 3
                off_list.append(cur_off)
                type_list.append(5)           # VTK_TRIANGLE

    return (np.array(conn_list, dtype=np.int32),
            np.array(off_list,  dtype=np.int32),
            np.array(type_list, dtype=np.uint8))


# Face local-node indices for each element type (corner nodes only)
# C3D8 / C3D20: 6 quad faces;  C3D10: 4 triangle faces
_FACE_LOCAL_IDX = {
    8:  [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
    10: [[0,1,2],[0,3,1],[1,3,2],[0,2,3]],
    20: [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
}
_FACE_VTK_TYPE = {8: 9, 10: 5, 20: 9}   # QUAD=9, TRI=5
_FACE_CORNER_N = {8: 8, 10: 4, 20: 8}   # leading corner nodes per element


def build_surface_cells_from_elems(points_np, elem_conn_np, n_nodes_per_elem, debug_label=""):
    """Build VTK surface cells from element connectivity (exact Abaqus-matching faces).

    Boundary faces (appear in exactly one element) are extracted, winding is
    corrected for outward normals, and per-node normals are computed for smooth
    shading in ParaView.

    Returns (connectivity, offsets, types, point_normals).
    Falls back to Delaunay build_surface_cells for unknown element types.
    """
    face_defs = _FACE_LOCAL_IDX.get(n_nodes_per_elem)
    vtk_type  = _FACE_VTK_TYPE.get(n_nodes_per_elem)
    corner_n  = _FACE_CORNER_N.get(n_nodes_per_elem)

    if face_defs is None:
        print(f"  [{debug_label}] unknown npe={n_nodes_per_elem}, Delaunay fallback")
        conn, offs, types = build_surface_cells(points_np)
        return conn, offs, types, np.zeros((points_np.shape[0], 3), dtype=np.float32)

    conn_corners = elem_conn_np[:, :corner_n]
    mesh_center  = points_np.mean(axis=0)

    # Detect boundary faces: sorted key appears in exactly one element
    face_map = {}
    for e_idx in range(len(conn_corners)):
        elem = conn_corners[e_idx]
        for face_local in face_defs:
            g   = elem[np.array(face_local)]
            key = tuple(sorted(g.tolist()))
            face_map[key] = None if key in face_map else g.copy()

    surface_faces = [v for v in face_map.values() if v is not None]
    if not surface_faces:
        print(f"  [{debug_label}] no boundary faces detected!")
        return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.uint8), np.zeros((points_np.shape[0], 3), dtype=np.float32))

    N = points_np.shape[0]
    point_normals = np.zeros((N, 3), dtype=np.float64)
    conn_list, off_list, type_list = [], [], []
    cur_off = 0

    for face_nodes in surface_faces:
        pts      = points_np[face_nodes]
        face_ctr = pts.mean(axis=0)
        normal   = np.cross(pts[1] - pts[0], pts[2] - pts[0]).astype(np.float64)
        nlen     = np.linalg.norm(normal)
        if nlen < 1e-30:
            continue
        normal /= nlen
        if np.dot(normal, face_ctr - mesh_center) < 0:
            face_nodes = face_nodes[::-1].copy()
            normal     = -normal
        conn_list.extend(face_nodes.tolist())
        cur_off += len(face_nodes)
        off_list.append(cur_off)
        type_list.append(vtk_type)
        for ni in face_nodes:
            point_normals[ni] += normal

    norms = np.linalg.norm(point_normals, axis=1, keepdims=True)
    point_normals /= np.where(norms < 1e-12, 1.0, norms)

    print(f"  [{debug_label}] npe={n_nodes_per_elem}, "
          f"{len(elem_conn_np)} elems → {len(surface_faces)} surface faces")

    return (np.array(conn_list, dtype=np.int32),
            np.array(off_list,  dtype=np.int32),
            np.array(type_list, dtype=np.uint8),
            point_normals.astype(np.float32))


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
    """4D: [disp_vec(3), dist(1)], convention dst-src to match preprocess_abaqus.py."""
    row, col = edge_index   # row=src, col=dst
    d = pos[col] - pos[row]  # dst - src
    n = torch.norm(d, dim=1, keepdim=True)
    return torch.cat([d, n], dim=1)


E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


def compute_vm_stress(wp, ref, mesh_ei):
    """Scatter linear fit von Mises stress, returns (N,) tensor in Pa."""
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    u     = wp.float() - ref.float()
    du_ij = u[col] - u[row]
    r_r   = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    row_exp = row.view(-1, 1, 1).expand_as(r_r)
    N = wp.shape[0]
    XtX = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtY = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtX.scatter_add_(0, row_exp, r_r)
    XtY.scatter_add_(0, row_exp, r_du)
    XtX = XtX + 1e-6 * torch.eye(3, device=wp.device, dtype=torch.float32).unsqueeze(0)
    dU  = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx  = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy  = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz  = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
                              + 6 * (txy**2 + txz**2 + tyz**2)) + 1e-30)


# ── Load normalization stats ──────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean     = node_stats["velocity_mean"].to(device)    # (3,)
v_std      = node_stats["velocity_std"].to(device)     # (3,)
s_mean_val = float(node_stats["stress_mean"])
s_std_val  = float(node_stats["stress_std"])
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

edge_stats  = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
# edge features = [d_mesh(3), n_mesh(1), d_world(3), n_world(1)]
# use first 4 dims to normalize ref edge features, last 4 for world edge features
e_mean_ref = edge_stats["edge_mean"].to(device)[:4]
e_std_ref  = edge_stats["edge_std"].to(device)[:4]
e_mean_wld = edge_stats["edge_mean"].to(device)[4:]
e_std_wld  = edge_stats["edge_std"].to(device)[4:]

empty_wef = torch.zeros(0, NUM_EDGE_FEATURES, device=device)


# ── Load model ────────────────────────────────────────────────────────────────

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
print(f"Loading checkpoint from {CKPT_PATH}{ep_tag}")
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)


# ── Main export loop ──────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"Found {len(all_files)} sample files in {DATA_DIR}")

for idx in SAMPLE_INDICES:
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range ({len(all_files)} files)")
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
    N    = data[0]["graph"].num_nodes
    print(f"\n[{sample_id}] N={N}, T={T} → {os.path.join(OUTPUT_DIR, sample_id)}")

    # Per-sample topology (each Abaqus sample may have different node count)
    ref_pos = data[0]["graph"].mesh_pos.to(device)    # (N, 3)
    mesh_ei = data[0]["graph"].edge_index.to(device)  # (2, E)
    ref_np  = ref_pos.cpu().numpy()

    # Load elem_conn from NPZ for exact surface topology matching Abaqus
    npz_d            = np.load(os.path.join(NPZ_DIR, sample_id + ".npz"))
    elem_conn_np     = npz_d["elem_conn"]
    n_nodes_per_elem = int(npz_d["n_nodes_per_elem"])
    connectivity, offsets, types, ref_normals = build_surface_cells_from_elems(
        ref_np, elem_conn_np, n_nodes_per_elem, debug_label=sample_id
    )

    # Static (reference) edge features — constant throughout rollout
    ref_feat_n = (compute_edge_features(ref_pos, mesh_ei) - e_mean_ref) / e_std_ref  # (E, 4)

    # ── Reconstruct GT world-pos sequence from velocity labels ────────────────
    # data[t]["graph"].world_pos = world_pos[t]
    # data[t]["graph"].y[:, :3]  = normalized velocity[t] = world_pos[t+1] - world_pos[t]
    gt_wp_list = [data[0]["graph"].world_pos.clone()]
    for step in data:
        vel_gt = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_wp_list.append(gt_wp_list[-1] + vel_gt)

    # GT stress: read directly from NPZ because .pt files were built from old
    # NPZ that had all-zero stress; stress_vm[t+1] matches world_pos[t+1].
    stress_vm_np   = npz_d["stress_vm"]                        # (F, N) raw Pa
    gt_stress_list = [stress_vm_np[t + 1].astype(np.float32)
                      for t in range(len(data))]

    # ── Autoregressive rollout ────────────────────────────────────────────────
    # Seed with GT state at t=1: position x_I^(1) + prev_vel = GT vel at t=0
    gt_vel_0 = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    cur      = gt_wp_list[1].to(device)
    prev_vel = gt_vel_0
    pred_wp_list = [gt_wp_list[0].numpy(), gt_wp_list[1].numpy()]  # t=0,1: GT seed

    with torch.inference_mode():
        for k in range(T - 1):  # predict t=2..T
            world_feat = (compute_edge_features(cur, mesh_ei) - e_mean_wld) / e_std_wld  # (E, 4)
            mesh_ef    = torch.cat([ref_feat_n, world_feat], dim=1)                       # (E, 8)

            node_x = torch.cat([cur - ref_pos, prev_vel], dim=1)  # (N, 6)
            g      = Data(x=node_x, edge_index=mesh_ei, num_nodes=N)
            out    = model(node_x, mesh_ef, empty_wef, g)

            vel      = out[:, :3] * v_std + v_mean
            prev_vel = vel
            cur      = cur + vel
            pred_wp_list.append(cur.cpu().numpy())

    # ── Write VTU + PVD ───────────────────────────────────────────────────────
    gt_pvd, pred_pvd = [], []

    for t_idx in range(T):
        vtu_name = f"t_{t_idx:03d}.vtu"

        # Ground truth
        gt_pos    = gt_wp_list[t_idx + 1].numpy().astype(np.float32)
        gt_disp   = gt_pos - ref_np
        gt_vm     = gt_stress_list[t_idx].astype(np.float32)
        # Scatter linear fit stress on GT positions — same method as pred for comparison
        gt_vm_computed = compute_vm_stress(
            torch.from_numpy(gt_pos).to(device),
            ref_pos, mesh_ei,
        ).cpu().numpy().astype(np.float32)

        write_vtu(
            os.path.join(gt_dir, vtu_name),
            gt_pos, connectivity, offsets, types,
            {
                "displacement":      gt_disp,
                "disp_mag":          np.linalg.norm(gt_disp, axis=1),
                "von_mises":         gt_vm,
                "von_mises_computed": gt_vm_computed,
            },
        )
        gt_pvd.append((t_idx, vtu_name))

        # Prediction
        pred_pos  = pred_wp_list[t_idx + 1].astype(np.float32)
        pred_disp = pred_pos - ref_np
        pred_vm   = compute_vm_stress(
            torch.from_numpy(pred_pos).to(device),
            ref_pos, mesh_ei,
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

    final_err = np.linalg.norm(
        pred_wp_list[-1] - gt_wp_list[-1].numpy(), axis=1
    ).mean()
    print(f"  final mean node error: {final_err:.6f} m")
    print(f"  GT   PVD → {gt_dir}/{sample_id}_gt.pvd")
    print(f"  Pred PVD → {pred_dir}/{sample_id}_pred.pvd")

print("\n完成！用 ParaView 開各 sample 下 gt/ 和 pred/ 裡的 .pvd 檔。")
