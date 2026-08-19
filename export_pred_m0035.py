"""
export_pred_m0035.py
====================
Autoregressive rollout for M0035: model uses its own predictions as input
at each step (no GT injection after frame 0).

Rollout:
  frame 0 → model → frame 1 → model → frame 2 → ... → frame 20

Edge features are recomputed from current deformed positions at every step.

Output:
  export_pred_m0035/
    S0001/
      surface_gt/    S0001_surface_gt.pvd
      surface_pred/  S0001_surface_pred.pvd
      solder_gt/     ...
      solder_pred/   ...
      si_gt/         ...
      si_pred/       ...
      uf_gt/         ...
      uf_pred/       ...

Usage:
  python export_pred_m0035.py
"""

import os
import sys
import json
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
from export_abaqus_paraview import (
    build_surface_cells_from_elems,
    write_vtu,
    write_pvd,
)
from export_gt_m0035 import (
    von_mises,
    build_hex_cells,
    export_frames,
)
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint


# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR        = "./02_abaqus_npz_m0035"
PT_DIR         = "./04_preprocessed_pt_m0035"
CKPT_PATH      = "./checkpoints_m0035_h256"
NODE_STATS     = "./04_preprocessed_pt_m0035/node_stats_m0035.json"
EDGE_STATS     = "./edge_stats_m0035.json"
OUTPUT_DIR     = "./export_pred_m0035_h256_ep860"
EXPORT_EPOCH   = 860

NUM_INPUT   = 17
NUM_EDGE    = 9
NUM_OUTPUT  = 10
PROC_SIZE   = 15
HIDDEN_DIM  = 256

SAMPLE_IDS  = ["S0001", "S0002", "S0201", "S0202"]

# Must match preprocess_m0035.py
E_REF        = 130000.0
MAT_E_FIXED  = np.array([130000.0, 130000.0, 47000.0, np.nan], dtype=np.float32)
C3D8_EDGES   = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


# ── Helpers (copied from preprocess) ─────────────────────────────────────────

def build_multigraph(elem_conn, elem_mat, mat_E_sample):
    seen = {}
    rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E_sample[mat_idx])
        for (a, b) in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst, int(mat_idx))
                if key not in seen:
                    seen[key] = True
                    rows.append(src)
                    cols.append(dst)
                    E_vals.append(E_elem)
    return (np.array([rows, cols], dtype=np.int64),
            np.array(E_vals, dtype=np.float32))


def compute_edge_feats(mesh_pos, world_pos_t, edge_index, edge_E):
    src, dst = edge_index[0], edge_index[1]
    d_mesh  = mesh_pos[dst]    - mesh_pos[src]
    n_mesh  = np.linalg.norm(d_mesh,  axis=1, keepdims=True)
    d_world = world_pos_t[dst] - world_pos_t[src]
    n_world = np.linalg.norm(d_world, axis=1, keepdims=True)
    E_col   = (edge_E / E_REF)[:, None]
    return np.concatenate([d_mesh, n_mesh, d_world, n_world, E_col],
                          axis=1).astype(np.float32)


# ── Stats ─────────────────────────────────────────────────────────────────────

def load_stats():
    with open(NODE_STATS) as f:
        ns = json.load(f)
    with open(EDGE_STATS) as f:
        es = json.load(f)
    return {
        "disp_mean":      np.array(ns["disp_mean"],           dtype=np.float32),
        "disp_std":       np.array(ns["disp_std"],            dtype=np.float32),
        "vel_mean":       np.array(ns["vel_mean"],            dtype=np.float32),
        "vel_std":        np.array(ns["vel_std"],             dtype=np.float32),
        "s_mean_Si":      np.array(ns["stress_mean_Si"],      dtype=np.float32),
        "s_std_Si":       np.array(ns["stress_std_Si"],       dtype=np.float32),
        "s_mean_So":      np.array(ns["stress_mean_Solder"],  dtype=np.float32),
        "s_std_So":       np.array(ns["stress_std_Solder"],   dtype=np.float32),
        "s_mean_UF":      np.array(ns["stress_mean_UF"],      dtype=np.float32),
        "s_std_UF":       np.array(ns["stress_std_UF"],       dtype=np.float32),
        "peeq_mean":      float(ns["peeq_mean"]),
        "peeq_std":       float(ns["peeq_std"]),
        "edge_mean":      np.array(es["edge_mean"],           dtype=np.float32),
        "edge_std":       np.array(es["edge_std"],            dtype=np.float32),
    }


# ── Denormalize prediction ────────────────────────────────────────────────────

def denorm(pred_np, node_mat, st):
    """pred_np (N,10) normalized → vel_phys (N,3), stress_phys (N,6), peeq_phys (N,)"""
    N = pred_np.shape[0]
    si  = (node_mat == 0) | (node_mat == 1)
    sol =  node_mat == 2
    uf  =  node_mat == 3

    vel_phys = pred_np[:, :3] * st["vel_std"] + st["vel_mean"]

    s = np.zeros((N, 6), dtype=np.float32)
    s[si]  = pred_np[si,  3:9] * st["s_std_Si"] + st["s_mean_Si"]
    s[sol] = pred_np[sol, 3:9] * st["s_std_So"] + st["s_mean_So"]
    s[uf]  = pred_np[uf,  3:9] * st["s_std_UF"] + st["s_mean_UF"]

    p = np.zeros(N, dtype=np.float32)
    p[sol] = pred_np[sol, 9] * st["peeq_std"] + st["peeq_mean"]

    return vel_phys, s, p


# ── Point data for VTK ────────────────────────────────────────────────────────

def make_pdata(world_pos_t, mesh_pos, stress_t, peeq_t):
    disp = (world_pos_t - mesh_pos).astype(np.float32) * 1e3   # m → mm
    return {
        "displacement": disp,
        "disp_mag":     np.linalg.norm(disp, axis=1),
        "von_mises":    von_mises(stress_t),
        "stress_xx":    stress_t[:, 0],
        "stress_yy":    stress_t[:, 1],
        "stress_zz":    stress_t[:, 2],
        "stress_xy":    stress_t[:, 3],
        "stress_xz":    stress_t[:, 4],
        "stress_yz":    stress_t[:, 5],
        "PEEQ":         peeq_t,
    }


# ── Export one sample ─────────────────────────────────────────────────────────

@torch.no_grad()
def export_sample(sid, model, st, device):
    npz_path = os.path.join(NPZ_DIR, f"{sid}.npz")
    pt_path  = os.path.join(PT_DIR,  f"{sid}.pt")
    if not os.path.exists(npz_path):
        print(f"  MISSING {npz_path}"); return

    d         = np.load(npz_path, allow_pickle=True)
    mesh_pos  = d["mesh_pos"].astype(np.float32)       # (N, 3)
    world_pos = d["world_pos"].astype(np.float32)      # (21, N, 3)
    stress_gt = d["stress"].astype(np.float32)         # (21, N, 6)
    peeq_gt   = d["peeq"].astype(np.float32)           # (21, N)
    node_mat  = d["node_mat"].astype(np.int32)
    elem_conn = d["elem_conn"].astype(np.int32)
    elem_mat  = d["elem_mat"].astype(np.int32)
    node_CTE  = d["node_CTE"].astype(np.float32)
    E_uf      = float(d["E_uf_MPa"])

    N       = mesh_pos.shape[0]
    T       = world_pos.shape[0]   # 21 frames
    si      = (node_mat == 0) | (node_mat == 1)
    sol     =  node_mat == 2
    uf_mask =  node_mat == 3

    # Build multigraph (same as preprocessing, per-sample E_uf)
    mat_E_s    = MAT_E_FIXED.copy();  mat_E_s[3] = E_uf
    edge_index, edge_E = build_multigraph(elem_conn, elem_mat.astype(np.int64), mat_E_s)
    ei_tensor  = torch.from_numpy(edge_index)

    # Constant node feature slices (mat_oh + CTE_norm) from .pt frame 0
    step_data = torch.load(pt_path, map_location="cpu", weights_only=False)
    # indices 13:17 of graph.x are [mat_oh(3), CTE_norm(1)] — constant across time
    const_feat = step_data[0]["graph"].x[:, 13:].numpy()   # (N, 4)

    # ── Rollout starting from GT frame 1 ─────────────────────────────────────
    # Model was trained on step_index≥1 (step 0 skipped: prev_vel=0 is OOD).
    # Use GT state at frame 1 as initial condition, then rollout frames 2-20.
    x1      = step_data[1]["graph"].x.numpy()                        # (N, 17)
    wp_curr = step_data[1]["graph"].world_pos.numpy().astype(np.float32)  # frame 1

    disp_n = x1[:, 0:3].copy()
    pv_n   = x1[:, 3:6].copy()
    s_n    = x1[:, 6:12].copy()
    peeq_n = x1[:, 12:13].copy()

    # Denorm GT frame 1 for storage
    s1_phys = np.zeros((N, 6), dtype=np.float32)
    s1_phys[si]      = s_n[si]      * st["s_std_Si"] + st["s_mean_Si"]
    s1_phys[sol]     = s_n[sol]     * st["s_std_So"] + st["s_mean_So"]
    s1_phys[uf_mask] = s_n[uf_mask] * st["s_std_UF"] + st["s_mean_UF"]
    p1_phys = np.zeros(N, dtype=np.float32)
    p1_phys[sol] = peeq_n[sol, 0] * st["peeq_std"] + st["peeq_mean"]

    # Frame 0 and 1 from GT; frames 2-20 from rollout
    wp_pred_all = [world_pos[0].copy(), wp_curr.copy()]
    s_pred_all  = [stress_gt[0].copy(), s1_phys]
    p_pred_all  = [peeq_gt[0].copy(),   p1_phys]

    for _t in range(T - 2):   # 19 rollout steps → frames 2-20
        # Recompute edge features from current deformed geometry
        ef_raw  = compute_edge_feats(mesh_pos, wp_curr, edge_index, edge_E)
        ef_norm = ((ef_raw - st["edge_mean"]) / st["edge_std"]).astype(np.float32)

        # Assemble node features
        node_x = np.concatenate(
            [disp_n, pv_n, s_n, peeq_n, const_feat], axis=1
        ).astype(np.float32)   # (N, 17)

        graph = Data(
            x          = torch.from_numpy(node_x),
            edge_index = ei_tensor,
            num_nodes  = N,
        ).to(device)
        mef = torch.from_numpy(ef_norm).to(device)
        wef = torch.zeros(0, NUM_EDGE, device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred = model(graph.x.float(), mef, wef, graph)
        pred_np = pred.float().cpu().numpy()   # (N, 10)

        vel_phys, s_phys, p_phys = denorm(pred_np, node_mat, st)

        # Update physical state
        wp_curr = wp_curr + vel_phys

        # Update normalized state for next step
        disp_n   = ((wp_curr - mesh_pos) - st["disp_mean"]) / st["disp_std"]
        pv_n     = pred_np[:, :3].copy()      # vel output is already normalized
        s_n      = pred_np[:, 3:9].copy()
        peeq_n   = pred_np[:, 9:10].copy()

        wp_pred_all.append(wp_curr.copy())
        s_pred_all.append(s_phys)
        p_pred_all.append(p_phys)

    print(f"  {sid}: rollout done (frame 0-1 GT, frames 2-{T-1} predicted)")

    # ── Build VTK cells ───────────────────────────────────────────────────────
    surf_conn, surf_offs, surf_types, _ = build_surface_cells_from_elems(
        mesh_pos, elem_conn, elem_conn.shape[1], debug_label=f"{sid}/surface"
    )
    mat_cells = {}
    for name, mask in [("solder", elem_mat==2),
                        ("si",    (elem_mat==0)|(elem_mat==1)),
                        ("uf",    elem_mat==3)]:
        elems = elem_conn[mask]
        if len(elems) > 0:
            mat_cells[name] = build_hex_cells(elems)

    base    = os.path.join(OUTPUT_DIR, sid)
    t_vals  = list(range(T))

    for tag, wp_seq, s_seq, p_seq in [
        ("gt",   [world_pos[t] for t in range(T)], list(stress_gt), list(peeq_gt)),
        ("pred", wp_pred_all, s_pred_all, p_pred_all),
    ]:
        pd_seq = [make_pdata(wp, mesh_pos, s, p)
                  for wp, s, p in zip(wp_seq, s_seq, p_seq)]

        export_frames(
            os.path.join(base, f"surface_{tag}"),
            f"{sid}_surface_{tag}.pvd",
            wp_seq, surf_conn, surf_offs, surf_types, pd_seq, t_vals,
        )
        for name, (conn, offs, types) in mat_cells.items():
            export_frames(
                os.path.join(base, f"{name}_{tag}"),
                f"{sid}_{name}_{tag}.pvd",
                wp_seq, conn, offs, types, pd_seq, t_vals,
            )

    print(f"  {sid}: exported → {base}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    st = load_stats()

    model = HybridMeshGraphNet(
        NUM_INPUT, NUM_EDGE, NUM_OUTPUT,
        processor_size=PROC_SIZE,
        hidden_dim_processor=HIDDEN_DIM,
        hidden_dim_node_encoder=HIDDEN_DIM,
        hidden_dim_edge_encoder=HIDDEN_DIM,
        hidden_dim_node_decoder=HIDDEN_DIM,
    ).to(device)

    epoch = load_checkpoint(CKPT_PATH, models=model, device=device, epoch=EXPORT_EPOCH)
    model.eval()
    print(f"Loaded checkpoint: epoch {epoch}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sid in SAMPLE_IDS:
        print(f"Exporting {sid} ...")
        export_sample(sid, model, st, device)

    print("Done.")


if __name__ == "__main__":
    main()