"""
export_beam_paraview.py
=======================
Export beam bending (no rod) GT + model predictions to VTU/PVD for ParaView.

Model: checkpoints_3Dbeam_no_rod  (train_prevvel.py, config_beam_no_rod)
  Input  : disp(3, normalized) + prev_vel(3, normalized) + node_type_oh(2)  = 8D
  Output : velocity(3)  = 3D
  Edges  : ref_ef(4D, normalized) + world_ef(4D, normalized) → cat → 8D mesh_ef

Rollout seed: t=1 (first load step; t=0 is undeformed)
  - wp  = world_pos[1]
  - prev_vel = world_pos[1] - world_pos[0]
Predicts t=2, 3, ..., T-1.

Output:
  export_beam_paraview_no_rod/
    sample_XXXXX/
      gt/   t_000.vtu ...  sample_XXXXX_gt.pvd
      pred/ t_000.vtu ...  sample_XXXXX_pred.pvd

PointData fields:
  GT   : displacement(vec), disp_mag, von_mises_computed
  Pred : displacement(vec), disp_mag, von_mises_computed, disp_error
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/04_preprocessed_pt_beam_prevvel"
NPZ_DIR    = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/02_beam_npz"
CKPT_PATH  = "./checkpoints_3Dbeam_no_rod_stress05_persample_clamp1e6"
CKPT_EPOCH = None          # None = latest epoch
OUTPUT_DIR = "./export_beam_paraview_no_rod_persample_clamp1e6"
USE_PER_SAMPLE_NORM = True

NUM_INPUT_FEATURES  = 8   # disp(3,norm) + prev_vel(3,norm) + node_type_oh(2)
NUM_EDGE_FEATURES   = 8   # ref(4D,norm) cat world(4D,norm)
NUM_OUTPUT_FEATURES = 3   # velocity(3)
PROCESSOR_SIZE      = 4
HIDDEN_DIM          = 128

# Test samples to export (index into sorted sample_*.pt list)
SAMPLE_INDICES = [1000, 1001, 1002, 1003, 1004, 1005]


# ── VTU / PVD helpers ─────────────────────────────────────────────────────────

_FACE_LOCAL_IDX = {
    8: [[0,1,2,3],[4,7,6,5],[0,4,5,1],[1,5,6,2],[2,6,7,3],[3,7,4,0]],
}
_FACE_VTK_TYPE  = {8: 9}   # VTK_QUAD
_FACE_CORNER_N  = {8: 8}


def build_surface_cells(points_np, elem_conn_np, npe):
    face_defs = _FACE_LOCAL_IDX.get(npe)
    vtk_type  = _FACE_VTK_TYPE.get(npe)
    if face_defs is None:
        raise ValueError(f"Unsupported npe={npe}")

    mesh_center = points_np.mean(axis=0)
    face_map = {}
    for elem in elem_conn_np:
        for fd in face_defs:
            g   = elem[np.array(fd)]
            key = tuple(sorted(g.tolist()))
            face_map[key] = None if key in face_map else g.copy()

    surface = [v for v in face_map.values() if v is not None]
    conn_list, off_list, type_list = [], [], []
    cur_off = 0
    for face_nodes in surface:
        pts    = points_np[face_nodes]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.dot(normal, pts.mean(0) - mesh_center) < 0:
            face_nodes = face_nodes[::-1].copy()
        conn_list.extend(face_nodes.tolist())
        cur_off += len(face_nodes)
        off_list.append(cur_off)
        type_list.append(vtk_type)
    print(f"    {len(elem_conn_np)} elems → {len(surface)} surface faces (npe={npe})")
    return (np.array(conn_list, dtype=np.int32),
            np.array(off_list,  dtype=np.int32),
            np.array(type_list, dtype=np.uint8))


def _arr_str(arr):
    return " ".join(f"{v:.6g}" for v in np.asarray(arr, dtype=np.float32).ravel())


def write_vtu(path, points, connectivity, offsets, types, point_data):
    N, nc = points.shape[0], len(types)
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{N}" NumberOfCells="{nc}">',
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
        if arr.ndim > 1:
            lines.append(f'        <DataArray type="Float32" Name="{name}" '
                         f'NumberOfComponents="{arr.shape[1]}" format="ascii">')
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


# ── Physics: Von Mises stress via GFDM ───────────────────────────────────────

E_MOD, NU = 70e9, 0.33
LAME1 = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU    = E_MOD / (2 * (1 + NU))


def compute_vm_stress(wp, ref, mesh_ei):
    """Returns (N,) Von Mises stress tensor in Pa."""
    row, col = mesh_ei
    r_ij  = ref[col].float() - ref[row].float()
    u     = wp.float() - ref.float()
    du_ij = u[col] - u[row]
    r_r   = r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
    r_du  = r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
    row_exp = row.view(-1,1,1).expand_as(r_r)
    N = wp.shape[0]
    XtX = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtY = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
    XtX.scatter_add_(0, row_exp, r_r)
    XtY.scatter_add_(0, row_exp, r_du)
    XtX += 1e-6 * torch.eye(3, device=wp.device).unsqueeze(0)
    dU   = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
    tr   = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
    sx   = LAME1 * tr + 2 * MU * dU[:, 0, 0]
    sy   = LAME1 * tr + 2 * MU * dU[:, 1, 1]
    sz   = LAME1 * tr + 2 * MU * dU[:, 2, 2]
    txy  = MU * (dU[:, 0, 1] + dU[:, 1, 0])
    txz  = MU * (dU[:, 0, 2] + dU[:, 2, 0])
    tyz  = MU * (dU[:, 1, 2] + dU[:, 2, 1])
    return torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                              + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)


def compute_edge_features(pos, edge_index):
    row, col = edge_index
    d = pos[col] - pos[row]
    return torch.cat([d, torch.norm(d, dim=1, keepdim=True)], dim=1)


# ── Load stats ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

node_stats = load_json(os.path.join(DATA_DIR, "node_stats.json"))
v_mean = node_stats["velocity_mean"].to(device)   # (3,)
v_std  = node_stats["velocity_std"].to(device)    # (3,)
d_mean = node_stats["disp_mean"].to(device)       # (3,)
d_std  = node_stats["disp_std"].to(device)        # (3,)

edge_stats  = load_json(os.path.join(DATA_DIR, "edge_stats.json"))
e_mean_ref  = edge_stats["edge_mean"].to(device)[:4]
e_std_ref   = edge_stats["edge_std"].to(device)[:4]
e_mean_wld  = edge_stats["edge_mean"].to(device)[4:]
e_std_wld   = edge_stats["edge_std"].to(device)[4:]

v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()
empty_wef  = torch.zeros(0, NUM_EDGE_FEATURES, device=device)


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

print(f"Loading checkpoint from {CKPT_PATH} (epoch={CKPT_EPOCH or 'latest'})")
load_checkpoint(CKPT_PATH, models=model, device=device, epoch=CKPT_EPOCH)
print("Checkpoint loaded.")


# ── Export loop ───────────────────────────────────────────────────────────────

all_files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith("sample_") and f.endswith(".pt"))
print(f"Found {len(all_files)} sample files in {DATA_DIR}")

for idx in SAMPLE_INDICES:
    if idx >= len(all_files):
        print(f"[skip] index {idx} out of range")
        continue

    fname     = all_files[idx]
    sample_id = fname.replace(".pt", "")   # e.g. "sample_01000"
    fpath     = os.path.join(DATA_DIR, fname)

    gt_dir   = os.path.join(OUTPUT_DIR, sample_id, "gt")
    pred_dir = os.path.join(OUTPUT_DIR, sample_id, "pred")
    os.makedirs(gt_dir,   exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    data = torch.load(fpath, map_location="cpu", weights_only=False)
    T    = len(data)   # number of transitions (F-1)
    N    = data[0]["graph"].num_nodes
    print(f"\n[{sample_id}] N={N}, transitions={T}")

    # ── Mesh geometry ─────────────────────────────────────────────────────────
    ref_pos     = data[0]["graph"].mesh_pos.to(device)   # (N, 3)
    mesh_ei     = data[0]["graph"].edge_index.to(device) # (2, E)
    ref_np      = ref_pos.cpu().numpy()
    node_type_oh = data[0]["graph"].x[:, 6:8].to(device) # (N, 2), constant

    ref_feat_raw  = compute_edge_features(ref_pos, mesh_ei)
    ref_feat_norm = (ref_feat_raw - e_mean_ref) / e_std_ref   # (E, 4) normalized

    # ── Surface topology from NPZ ─────────────────────────────────────────────
    npz_id   = sample_id.replace("sample_", "sample_")   # same naming
    npz_path = os.path.join(NPZ_DIR, npz_id + ".npz")
    if not os.path.exists(npz_path):
        # Beam NPZ might have a different numeric ID
        print(f"  [warn] NPZ not found: {npz_path}, skipping")
        continue
    npz_d    = np.load(npz_path)
    elem_conn_np  = npz_d["elem_conn"]
    npe           = int(npz_d["n_nodes_per_elem"])
    connectivity, offsets, types = build_surface_cells(ref_np, elem_conn_np, npe)

    # ── GT world positions: reconstruct from stored velocities ────────────────
    # data[t]["graph"].y = normalized velocity at step t (t→t+1)
    # world_pos at step 0 stored as world_pos field
    gt_world_pos = [data[0]["graph"].world_pos.clone()]   # t=0
    for step in data:
        vel_raw = step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu
        gt_world_pos.append(gt_world_pos[-1] + vel_raw)
    # gt_world_pos[0..T]: world_pos at t=0 to t=T

    # Also load GT Von Mises from NPZ for comparison
    gt_stress_np = npz_d["stress_vm"]   # (F, N) in Pa

    # ── Autoregressive rollout seeded at t=1 ─────────────────────────────────
    # Seed: wp at t=1, prev_vel = vel from t=0→t=1
    wp_seed  = gt_world_pos[1].to(device)
    pv_seed  = (gt_world_pos[1] - gt_world_pos[0]).to(device)

    pred_world_pos = [gt_world_pos[0].numpy(),   # t=0: GT (undeformed seed)
                      gt_world_pos[1].numpy()]   # t=1: GT seed
    wp_cur  = wp_seed
    prev_vel = pv_seed

    with torch.no_grad():
        for _ in range(T - 1):   # predict t=2..T
            # Normalize inputs (matches train_prevvel.py forward())
            if USE_PER_SAMPLE_NORM:
                node_scale    = prev_vel.norm(dim=1, keepdim=True).mean().clamp(min=1e-6)
                disp_norm     = (wp_cur - ref_pos) / node_scale
                prev_vel_norm = prev_vel / node_scale
            else:
                node_scale    = None
                disp_norm     = ((wp_cur - ref_pos) - d_mean) / d_std
                prev_vel_norm = (prev_vel - v_mean) / v_std
            node_x = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)  # (N, 8)

            # Edge features: ref(4D,norm) + world(4D,norm) → 8D
            wf      = (compute_edge_features(wp_cur, mesh_ei) - e_mean_wld) / e_std_wld
            mesh_ef = torch.cat([ref_feat_norm, wf], dim=1)   # (E, 8)

            g_in = Data(x=node_x, edge_index=mesh_ei, num_nodes=N,
                        mesh_pos=ref_pos, world_pos=wp_cur)
            pred     = model(node_x, mesh_ef, empty_wef, g_in)
            if USE_PER_SAMPLE_NORM:
                vel_phys = pred.float() * node_scale
            else:
                vel_phys = pred.float() * v_std + v_mean
            wp_next  = wp_cur + vel_phys

            pred_world_pos.append(wp_next.cpu().numpy())
            wp_cur   = wp_next
            prev_vel = vel_phys

    # ── Write VTU + PVD ───────────────────────────────────────────────────────
    gt_pvd, pred_pvd = [], []

    for t_idx in range(T):
        vtu_name = f"t_{t_idx:03d}.vtu"

        # GT frame: world_pos[t_idx+1]  (skip t=0 which is undeformed)
        gt_pos   = gt_world_pos[t_idx + 1].numpy().astype(np.float32)
        gt_disp  = gt_pos - ref_np
        gt_vm_np = gt_stress_np[t_idx + 1].astype(np.float32) if t_idx + 1 < gt_stress_np.shape[0] else np.zeros(N, np.float32)
        gt_vm_computed = compute_vm_stress(
            torch.from_numpy(gt_pos).to(device), ref_pos, mesh_ei
        ).cpu().numpy().astype(np.float32)

        write_vtu(os.path.join(gt_dir, vtu_name), gt_pos,
                  connectivity, offsets, types, {
                      "displacement":       gt_disp,
                      "disp_mag":           np.linalg.norm(gt_disp, axis=1),
                      "von_mises_abaqus":   gt_vm_np,
                      "von_mises_computed": gt_vm_computed,
                  })
        gt_pvd.append((t_idx, vtu_name))

        # Pred frame: pred_world_pos[t_idx+1]
        pred_pos  = pred_world_pos[t_idx + 1].astype(np.float32)
        pred_disp = pred_pos - ref_np
        pred_vm   = compute_vm_stress(
            torch.from_numpy(pred_pos).to(device), ref_pos, mesh_ei
        ).cpu().numpy().astype(np.float32)
        disp_err  = np.linalg.norm(pred_pos - gt_pos, axis=1).astype(np.float32)

        write_vtu(os.path.join(pred_dir, vtu_name), pred_pos,
                  connectivity, offsets, types, {
                      "displacement":       pred_disp,
                      "disp_mag":           np.linalg.norm(pred_disp, axis=1),
                      "von_mises_computed": pred_vm,
                      "disp_error":         disp_err,
                  })
        pred_pvd.append((t_idx, vtu_name))

    write_pvd(os.path.join(gt_dir,   f"{sample_id}_gt.pvd"),   gt_pvd)
    write_pvd(os.path.join(pred_dir, f"{sample_id}_pred.pvd"), pred_pvd)

    final_err = np.linalg.norm(
        pred_world_pos[-1] - gt_world_pos[-1].numpy(), axis=1
    ).mean()
    print(f"  final mean disp error : {final_err:.4e} m")
    print(f"  GT   → {gt_dir}/{sample_id}_gt.pvd")
    print(f"  Pred → {pred_dir}/{sample_id}_pred.pvd")

print(f"\n完成！用 ParaView 開 {OUTPUT_DIR}/<sample>/gt/ 或 pred/ 裡的 .pvd 檔。")
