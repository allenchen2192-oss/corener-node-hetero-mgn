"""
compare_stress_node_finalstep.py
=================================
Compare HeteroMGN (with element nodes) vs Corner-Node MGN (no element nodes)
on final-step node-level VM stress.

Conversion: HeteroMGN element-level pred -> elem_to_node_mean -> node stress.
Same GT for both: elem_to_node_mean(elem_vm_stress[T], elem_conn, N).
Metric: Relative MAE (%) = MAE / mean(GT) * 100%

Node groups (inclusive, interface nodes counted in all touching materials):
  Overall | Si | Solder | UF

Usage:
  python compare_stress_node_finalstep.py            # 10 train + 20 test
  python compare_stress_node_finalstep.py 20 20
"""

import os, sys, json, glob
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData, Data

from physicsnemo.utils import load_checkpoint
from train_m0040 import HeteroMGN
from train_m0040_cornernode import CornerNodeMGN

# ── Config ────────────────────────────────────────────────────────────────────

NPZ_DIR       = "./02_abaqus_npz_m0040"
HETERO_STATS  = "./04_preprocessed_hetero_m0040/stats_m0040.json"
GLOBALVM_STATS= "./04_preprocessed_hetero_m0040_globalvm/stats_m0040_globalvm.json"
CORNER_STATS  = "./04_preprocessed_cornernode_m0040/stats_cornernode_m0040.json"
HETERO_CKPT   = "./checkpoints_m0040"
GLOBALVM_CKPT = "./checkpoints_m0040_globalvm"
CORNER_CKPT   = "./checkpoints_m0040_cornernode"
OUT_PNG       = "./compare_stress_node_finalstep.png"

TRAIN_IDS = [f"S{i:04d}" for i in range(1,   201)]   # all 200 train
TEST_IDS  = [f"S{i:04d}" for i in range(201, 221)]   # all 20 test

HETERO_CFG = SimpleNamespace(
    hidden_dim=256, processor_size=10,
    num_node_features=11, num_edge_features=8,
    num_elem_features=8, num_node_outputs=3, num_elem_outputs=2,
)
CORNER_CFG = SimpleNamespace(
    hidden_dim=256, processor_size=10,
    num_node_features=13, num_edge_features=14, num_outputs=5,
)

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_singlegraph(elem_conn):
    seen = set(); rows, cols = [], []
    for c in elem_conn:
        for a, b in C3D8_EDGES:
            u, v = int(c[a]), int(c[b])
            for s, d in ((u,v),(v,u)):
                if (s,d) not in seen:
                    seen.add((s,d)); rows.append(s); cols.append(d)
    return np.array([rows, cols], dtype=np.int64)

def build_b02(elem_conn):
    M, npe = elem_conn.shape
    ei = np.repeat(np.arange(M, dtype=np.int64), npe)
    ni = elem_conn.reshape(-1).astype(np.int64)
    return np.stack([ni, ei]), np.stack([ei, ni])

def build_multigraph(elem_conn, elem_mat):
    seen = {}; rows, cols, eidx, mids = [], [], [], []
    for ei, conn in enumerate(elem_conn):
        mid = int(elem_mat[ei])
        for a, b in C3D8_EDGES:
            u, v = int(conn[a]), int(conn[b])
            for s, t in ((u,v),(v,u)):
                key = (s, t, mid)
                if key not in seen:
                    seen[key] = True
                    rows.append(s); cols.append(t)
                    eidx.append(ei); mids.append(mid)
    return (np.array([rows, cols], dtype=np.int64),
            np.array(eidx, dtype=np.int64),
            np.array(mids, dtype=np.int64))

def geom_feats(mesh_pos, world_pos, ei):
    s, d = ei[0], ei[1]
    dm = mesh_pos[d] - mesh_pos[s]; nm = np.linalg.norm(dm, axis=1, keepdims=True)
    dw = world_pos[d] - world_pos[s]; nw = np.linalg.norm(dw, axis=1, keepdims=True)
    return np.concatenate([dm, nm, dw, nw], axis=1).astype(np.float32)

def node_type_oh(node_type, n=5):
    oh = np.zeros((len(node_type), n), dtype=np.float32)
    for i, nt in enumerate(node_type):
        if 0 <= int(nt) < n: oh[i, int(nt)] = 1.0
    return oh

def mat_oh_3(elem_mat):
    oh = np.zeros((len(elem_mat), 3), dtype=np.float32)
    oh[:,0] = ((elem_mat==0)|(elem_mat==1)).astype(np.float32)
    oh[:,1] = (elem_mat==2).astype(np.float32)
    oh[:,2] = (elem_mat==3).astype(np.float32)
    return oh

def e2n_mean(elem_vals, elem_conn, N):
    ns = np.zeros(N, np.float64); nc = np.zeros(N, np.int64)
    np.add.at(ns, elem_conn.ravel(), np.repeat(elem_vals, 8))
    np.add.at(nc, elem_conn.ravel(), np.ones(elem_conn.size, np.int64))
    return (ns / np.maximum(nc, 1)).astype(np.float32)

# ── Load stats ────────────────────────────────────────────────────────────────

with open(HETERO_STATS) as f:
    hs = json.load(f)
with open(GLOBALVM_STATS) as f:
    gs = json.load(f)
with open(CORNER_STATS) as f:
    cs = json.load(f)

# HeteroMGN stats
h_disp_mean = np.array(hs["disp_mean"],  dtype=np.float32)
h_disp_std  = np.array(hs["disp_std"],   dtype=np.float32)
h_vel_mean  = np.array(hs["vel_mean"],   dtype=np.float32)
h_vel_std   = np.array(hs["vel_std"],    dtype=np.float32)
h_edge_mean = np.array(hs["edge_mean"],  dtype=np.float32)
h_edge_std  = np.array(hs["edge_std"],   dtype=np.float32)
h_CTE_mean  = float(hs["CTE_mean"]); h_CTE_std = float(hs["CTE_std"])
h_nu_mean   = float(hs["nu_mean"]);  h_nu_std  = float(hs["nu_std"])
h_E_MIN     = float(hs["E_min"]);    h_E_MAX   = float(hs["E_max"])
h_peeq_mean = float(hs["peeq_mean_solder"])
h_peeq_std  = float(hs["peeq_std_solder"])
H_VM_MEAN = {0: hs["vm_mean_Si"], 1: hs["vm_mean_Si"],
             2: hs["vm_mean_Solder"], 3: hs["vm_mean_UF"]}
H_VM_STD  = {0: hs["vm_std_Si"],  1: hs["vm_std_Si"],
             2: hs["vm_std_Solder"],  3: hs["vm_std_UF"]}

# GlobalVM stats
g_disp_mean  = np.array(gs["disp_mean"],  dtype=np.float32)
g_disp_std   = np.array(gs["disp_std"],   dtype=np.float32)
g_vel_mean   = np.array(gs["vel_mean"],   dtype=np.float32)
g_vel_std    = np.array(gs["vel_std"],    dtype=np.float32)
g_edge_mean  = np.array(gs["edge_mean"],  dtype=np.float32)
g_edge_std   = np.array(gs["edge_std"],   dtype=np.float32)
g_CTE_mean   = float(gs["CTE_mean"]); g_CTE_std = float(gs["CTE_std"])
g_nu_mean    = float(gs["nu_mean"]);  g_nu_std  = float(gs["nu_std"])
g_E_MIN      = float(gs["E_min"]);    g_E_MAX   = float(gs["E_max"])
g_peeq_mean  = float(gs["peeq_mean_solder"])
g_peeq_std   = float(gs["peeq_std_solder"])
g_vm_mean    = float(gs["vm_mean_global"])
g_vm_std     = float(gs["vm_std_global"])

# CornerNode stats
c_disp_mean  = np.array(cs["disp_mean"], dtype=np.float32)
c_disp_std   = np.array(cs["disp_std"],  dtype=np.float32)
c_vel_mean   = np.array(cs["vel_mean"],  dtype=np.float32)
c_vel_std    = np.array(cs["vel_std"],   dtype=np.float32)
c_geom_mean  = np.array(cs["geom_mean"], dtype=np.float32)
c_geom_std   = np.array(cs["geom_std"],  dtype=np.float32)
c_vm_mean    = float(cs["vm_node_mean"])
c_vm_std     = float(cs["vm_node_std"])
c_peeq_mean  = float(cs["peeq_mean_solder"])
c_peeq_std   = float(cs["peeq_std_solder"])
c_CTE_mean   = float(cs["CTE_mean"]); c_CTE_std = float(cs["CTE_std"])
c_nu_mean    = float(cs["nu_mean"]);  c_nu_std  = float(cs["nu_std"])
c_E_MIN      = float(cs["E_min"]);    c_E_MAX   = float(cs["E_max"])

# ── Load models ───────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

h_ckpt_files = glob.glob(os.path.join(HETERO_CKPT, "HeteroMGN.0.*.pt"))
h_epoch = max((int(os.path.basename(f).split(".")[2]) for f in h_ckpt_files), default=0)

g_ckpt_files = glob.glob(os.path.join(GLOBALVM_CKPT, "HeteroMGN.0.*.pt"))
g_epoch = max((int(os.path.basename(f).split(".")[2]) for f in g_ckpt_files), default=0)

c_ckpt_files = glob.glob(os.path.join(CORNER_CKPT, "CornerNodeMGN.0.*.pt"))
c_epoch = max((int(os.path.basename(f).split(".")[2]) for f in c_ckpt_files), default=0)

print(f"HeteroMGN epoch: {h_epoch}  |  GlobalVM epoch: {g_epoch}  |  CornerNodeMGN epoch: {c_epoch}")

hetero_model = HeteroMGN(HETERO_CFG).to(device)
hetero_model.eval()
load_checkpoint(HETERO_CKPT, models=hetero_model, device=device)

globalvm_model = HeteroMGN(HETERO_CFG).to(device)
globalvm_model.eval()
load_checkpoint(GLOBALVM_CKPT, models=globalvm_model, device=device)

corner_model = CornerNodeMGN(CORNER_CFG).to(device)
corner_model.eval()
load_checkpoint(CORNER_CKPT, models=corner_model, device=device)

# ── Per-sample rollout → final-step node stress ───────────────────────────────

def process_sample(sid):
    npz_path = os.path.join(NPZ_DIR, f"{sid}.npz")
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path)

    mesh_pos       = d["mesh_pos"]
    world_pos      = d["world_pos"]       # (21, N, 3)
    velocity       = d["velocity"]        # (20, N, 3)
    elem_vm_stress = d["elem_vm_stress"]  # (21, M)
    elem_peeq      = d["elem_peeq"]       # (21, M)
    elem_conn      = d["elem_conn"]
    elem_mat       = d["elem_mat"]
    elem_E         = d["elem_E"]
    elem_CTE       = d["elem_CTE"]
    elem_nu        = d["elem_nu"]
    node_type      = d["node_type"]

    T = velocity.shape[0]   # 20
    N = mesh_pos.shape[0]
    M = elem_mat.shape[0]
    sol_mask = elem_mat == 2
    si_mask  = (elem_mat == 0) | (elem_mat == 1)
    uf_mask  = elem_mat == 3

    # GT node stress at final step (shared by both models)
    gt_vm_node = e2n_mean(elem_vm_stress[T].astype(np.float32), elem_conn, N)

    # Inclusive node masks per material
    all_mask = np.ones(N, bool)
    si_node  = np.zeros(N, bool)
    sol_node = np.zeros(N, bool)
    uf_node  = np.zeros(N, bool)
    si_node[np.unique(elem_conn[si_mask].ravel())]  = True
    sol_node[np.unique(elem_conn[sol_mask].ravel())] = True
    uf_node[np.unique(elem_conn[uf_mask].ravel())]  = True
    solder_node_mask = sol_node   # for CornerNode PEEQ state

    # ── HeteroMGN rollout ─────────────────────────────────────────────────────
    h_edge_index = build_singlegraph(elem_conn)
    h_n2e, h_e2n = build_b02(elem_conn)
    h_ei_t  = torch.from_numpy(h_edge_index).to(device)
    h_n2e_t = torch.from_numpy(h_n2e).to(device)
    h_e2n_t = torch.from_numpy(h_e2n).to(device)

    h_mat_oh = mat_oh_3(elem_mat)
    h_E_n    = ((elem_E - h_E_MIN) / (h_E_MAX - h_E_MIN)).astype(np.float32)[:, None]
    h_CTE_n  = ((elem_CTE - h_CTE_mean) / h_CTE_std).astype(np.float32)[:, None]
    h_nu_n   = ((elem_nu  - h_nu_mean)  / h_nu_std).astype(np.float32)[:, None]
    h_elem_static = np.concatenate([h_mat_oh, h_E_n, h_CTE_n, h_nu_n], axis=1)
    h_vm_mean_e = np.array([H_VM_MEAN[int(m)] for m in elem_mat], dtype=np.float32)
    h_vm_std_e  = np.array([H_VM_STD[int(m)]  for m in elem_mat], dtype=np.float32)
    h_nt_oh = node_type_oh(node_type)

    h_cur_pos  = world_pos[1].astype(np.float32)
    h_prev_vel = velocity[0].astype(np.float32)
    h_vm_elem  = elem_vm_stress[1].astype(np.float32)
    h_peeq_e   = elem_peeq[1].astype(np.float32)

    for fi in range(1, T):
        with torch.inference_mode():
            d_n   = (h_cur_pos - mesh_pos - h_disp_mean) / h_disp_std
            pv_n  = (h_prev_vel - h_vel_mean) / h_vel_std
            node_x = np.concatenate([d_n, pv_n, h_nt_oh], axis=1)

            vm_n = ((h_vm_elem - h_vm_mean_e) / h_vm_std_e)[:, None]
            pq_n = np.zeros(M, dtype=np.float32)
            pq_n[sol_mask] = (h_peeq_e[sol_mask] - h_peeq_mean) / h_peeq_std
            elem_x = np.concatenate([h_elem_static, vm_n, pq_n[:, None]], axis=1)

            ef = ((geom_feats(mesh_pos, h_cur_pos, h_edge_index)
                   - h_edge_mean) / h_edge_std).astype(np.float32)

            hd = HeteroData()
            hd["node"].x    = torch.from_numpy(node_x).to(device)
            hd["element"].x = torch.from_numpy(elem_x).to(device)
            hd["node",    "mesh", "node"   ].edge_index = h_ei_t
            hd["node",    "mesh", "node"   ].edge_attr  = torch.from_numpy(ef).to(device)
            hd["node",    "in",   "element"].edge_index = h_n2e_t
            hd["element", "has",  "node"   ].edge_index = h_e2n_t

            vel_pred, elem_pred = hetero_model(hd)

        h_prev_vel = (vel_pred.cpu().numpy() * h_vel_std + h_vel_mean).astype(np.float32)
        h_cur_pos  = h_cur_pos + h_prev_vel
        h_vm_elem  = (elem_pred[:, 0].cpu().numpy() * h_vm_std_e + h_vm_mean_e).astype(np.float32)
        h_peeq_e   = np.zeros(M, dtype=np.float32)
        h_peeq_e[sol_mask] = np.clip(
            elem_pred[sol_mask, 1].cpu().numpy() * h_peeq_std + h_peeq_mean, 0.0, None
        )

    # Convert final element prediction to node stress
    hetero_vm_node = e2n_mean(h_vm_elem, elem_conn, N)

    # ── GlobalVM rollout ──────────────────────────────────────────────────────
    g_mat_oh = mat_oh_3(elem_mat)
    g_E_n    = ((elem_E   - g_E_MIN)    / (g_E_MAX   - g_E_MIN)).astype(np.float32)[:, None]
    g_CTE_n  = ((elem_CTE - g_CTE_mean) / g_CTE_std).astype(np.float32)[:, None]
    g_nu_n   = ((elem_nu  - g_nu_mean)  / g_nu_std).astype(np.float32)[:, None]
    g_elem_static = np.concatenate([g_mat_oh, g_E_n, g_CTE_n, g_nu_n], axis=1)

    g_cur_pos  = world_pos[1].astype(np.float32)
    g_prev_vel = velocity[0].astype(np.float32)
    g_vm_elem  = elem_vm_stress[1].astype(np.float32)
    g_peeq_e   = elem_peeq[1].astype(np.float32)

    for fi in range(1, T):
        with torch.inference_mode():
            d_n    = (g_cur_pos - mesh_pos - g_disp_mean) / g_disp_std
            pv_n   = (g_prev_vel - g_vel_mean) / g_vel_std
            node_x = np.concatenate([d_n, pv_n, h_nt_oh], axis=1)

            vm_n = ((g_vm_elem - g_vm_mean) / g_vm_std)[:, None]
            pq_n = np.zeros(M, dtype=np.float32)
            pq_n[sol_mask] = (g_peeq_e[sol_mask] - g_peeq_mean) / g_peeq_std
            elem_x = np.concatenate([g_elem_static, vm_n, pq_n[:, None]], axis=1)

            ef = ((geom_feats(mesh_pos, g_cur_pos, h_edge_index)
                   - g_edge_mean) / g_edge_std).astype(np.float32)

            hd = HeteroData()
            hd["node"].x    = torch.from_numpy(node_x).to(device)
            hd["element"].x = torch.from_numpy(elem_x).to(device)
            hd["node",    "mesh", "node"   ].edge_index = h_ei_t
            hd["node",    "mesh", "node"   ].edge_attr  = torch.from_numpy(ef).to(device)
            hd["node",    "in",   "element"].edge_index = h_n2e_t
            hd["element", "has",  "node"   ].edge_index = h_e2n_t

            vel_pred, elem_pred = globalvm_model(hd)

        g_prev_vel = (vel_pred.cpu().numpy() * g_vel_std + g_vel_mean).astype(np.float32)
        g_cur_pos  = g_cur_pos + g_prev_vel
        g_vm_elem  = (elem_pred[:, 0].cpu().numpy() * g_vm_std + g_vm_mean).astype(np.float32)
        g_peeq_e   = np.zeros(M, dtype=np.float32)
        g_peeq_e[sol_mask] = np.clip(
            elem_pred[sol_mask, 1].cpu().numpy() * g_peeq_std + g_peeq_mean, 0.0, None
        )

    globalvm_vm_node = e2n_mean(g_vm_elem, elem_conn, N)

    # ── CornerNode rollout ────────────────────────────────────────────────────
    c_ei, c_eidx, c_mids = build_multigraph(elem_conn, elem_mat)
    c_mat_oh = np.zeros((len(c_mids), 3), dtype=np.float32)
    c_mat_oh[:,0] = ((c_mids==0)|(c_mids==1)).astype(np.float32)
    c_mat_oh[:,1] = (c_mids==2).astype(np.float32)
    c_mat_oh[:,2] = (c_mids==3).astype(np.float32)
    c_E_n   = ((elem_E[c_eidx] - c_E_MIN) / (c_E_MAX - c_E_MIN))[:, None].astype(np.float32)
    c_CTE_n = ((elem_CTE[c_eidx] - c_CTE_mean) / c_CTE_std)[:, None].astype(np.float32)
    c_nu_n  = ((elem_nu[c_eidx]  - c_nu_mean)  / c_nu_std)[:, None].astype(np.float32)
    c_mat_ef = np.concatenate([c_mat_oh, c_E_n, c_CTE_n, c_nu_n], axis=1)
    c_ei_t  = torch.from_numpy(c_ei).to(device)
    c_nt_oh = node_type_oh(node_type)

    c_cur_pos   = world_pos[1].astype(np.float32)
    c_prev_vel  = velocity[0].astype(np.float32)
    c_vm_node   = e2n_mean(elem_vm_stress[1].astype(np.float32), elem_conn, N)
    c_peeq_node = np.zeros(N, dtype=np.float32)
    c_peeq_node[solder_node_mask] = e2n_mean(
        elem_peeq[1].astype(np.float32), elem_conn, N
    )[solder_node_mask]

    for fi in range(1, T):
        with torch.inference_mode():
            d_n  = (c_cur_pos - mesh_pos - c_disp_mean) / c_disp_std
            pv_n = (c_prev_vel - c_vel_mean) / c_vel_std
            vm_n = ((c_vm_node - c_vm_mean) / c_vm_std)[:, None]
            pq_n = np.zeros(N, dtype=np.float32)
            pq_n[solder_node_mask] = (
                (c_peeq_node[solder_node_mask] - c_peeq_mean) / c_peeq_std
            )
            node_x = np.concatenate([d_n, pv_n, c_nt_oh, vm_n, pq_n[:, None]], axis=1)

            geom_n = ((geom_feats(mesh_pos, c_cur_pos, c_ei)
                       - c_geom_mean) / c_geom_std).astype(np.float32)
            edge_attr = np.concatenate([geom_n, c_mat_ef], axis=1)

            data = Data(
                x          = torch.from_numpy(node_x).to(device),
                edge_index = c_ei_t,
                edge_attr  = torch.from_numpy(edge_attr).to(device),
            )
            pred = corner_model(data)

        pred_np = pred.cpu().numpy()
        c_prev_vel = (pred_np[:, 0:3] * c_vel_std + c_vel_mean).astype(np.float32)
        c_cur_pos  = c_cur_pos + c_prev_vel
        c_vm_node  = (pred_np[:, 3] * c_vm_std + c_vm_mean).astype(np.float32)
        c_peeq_node = np.zeros(N, dtype=np.float32)
        c_peeq_node[solder_node_mask] = np.clip(
            pred_np[solder_node_mask, 4] * c_peeq_std + c_peeq_mean, 0.0, None
        )

    corner_vm_node = c_vm_node   # already node-level

    return {
        "gt":       gt_vm_node,
        "hetero":   hetero_vm_node,
        "globalvm": globalvm_vm_node,
        "corner":   corner_vm_node,
        "masks":    {"Overall": all_mask, "Si": si_node, "Solder": sol_node, "UF": uf_node},
    }


# ── Collect across samples ────────────────────────────────────────────────────

GROUP_KEYS = ["Overall", "Si", "Solder", "UF"]

def collect(sample_ids, label):
    buckets = {g: {"gt": [], "hetero": [], "globalvm": [], "corner": []} for g in GROUP_KEYS}
    for sid in sample_ids:
        print(f"  [{label}] {sid} ...", end=" ", flush=True)
        res = process_sample(sid)
        if res is None:
            print("MISSING"); continue
        for g in GROUP_KEYS:
            m = res["masks"][g]
            buckets[g]["gt"].append(res["gt"][m])
            buckets[g]["hetero"].append(res["hetero"][m])
            buckets[g]["globalvm"].append(res["globalvm"][m])
            buckets[g]["corner"].append(res["corner"][m])
        print("done")
    return {g: {k: np.concatenate(v) for k, v in buckets[g].items()} for g in GROUP_KEYS}


print(f"\nCollecting {len(TRAIN_IDS)} train samples ...")
train_data = collect(TRAIN_IDS, "train")
print(f"Collecting {len(TEST_IDS)} test samples ...")
test_data  = collect(TEST_IDS,  "test")


# ── Compute metrics ───────────────────────────────────────────────────────────

def rel_mae(gt, pred):
    return np.mean(np.abs(gt - pred)) / np.mean(gt) * 100.0

print("\n=== Relative MAE (%) ===")
print(f"{'Group':<10}  {'Hetero-Tr':>10} {'GlobalVM-Tr':>12} {'Corner-Tr':>10}  "
      f"{'Hetero-Te':>10} {'GlobalVM-Te':>12} {'Corner-Te':>10}")

results = {}
for g in GROUP_KEYS:
    h_tr = rel_mae(train_data[g]["gt"], train_data[g]["hetero"])
    gv_tr= rel_mae(train_data[g]["gt"], train_data[g]["globalvm"])
    c_tr = rel_mae(train_data[g]["gt"], train_data[g]["corner"])
    h_te = rel_mae(test_data[g]["gt"],  test_data[g]["hetero"])
    gv_te= rel_mae(test_data[g]["gt"],  test_data[g]["globalvm"])
    c_te = rel_mae(test_data[g]["gt"],  test_data[g]["corner"])
    results[g] = (h_tr, gv_tr, c_tr, h_te, gv_te, c_te)
    print(f"{g:<10}  {h_tr:>9.2f}% {gv_tr:>11.2f}% {c_tr:>9.2f}%  "
          f"{h_te:>9.2f}% {gv_te:>11.2f}% {c_te:>9.2f}%")


# ── Plot grouped bar chart ────────────────────────────────────────────────────

x     = np.arange(len(GROUP_KEYS))
width = 0.13
colors = {
    "HeteroMGN Train":   "#2471a3",
    "HeteroMGN Test":    "#5dade2",
    "GlobalVM Train":    "#1e8449",
    "GlobalVM Test":     "#58d68d",
    "CornerNode Train":  "#ca6f1e",
    "CornerNode Test":   "#f0b27a",
}

fig, ax = plt.subplots(figsize=(13, 5))

bars = [
    ("HeteroMGN Train",  [results[g][0] for g in GROUP_KEYS], -2.5),
    ("HeteroMGN Test",   [results[g][3] for g in GROUP_KEYS], -1.5),
    ("GlobalVM Train",   [results[g][1] for g in GROUP_KEYS], -0.5),
    ("GlobalVM Test",    [results[g][4] for g in GROUP_KEYS], +0.5),
    ("CornerNode Train", [results[g][2] for g in GROUP_KEYS], +1.5),
    ("CornerNode Test",  [results[g][5] for g in GROUP_KEYS], +2.5),
]

for label, vals, offset in bars:
    rects = ax.bar(x + offset * width, vals, width,
                   label=label, color=colors[label], edgecolor="white", linewidth=0.5)
    for rect, v in zip(rects, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.05,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(GROUP_KEYS, fontsize=11)
ax.set_ylabel("Relative MAE (%)", fontsize=11)
ax.set_title(
    f"Node-Level VM Stress Error — Final Step t=20\n"
    f"HeteroMGN Per-Material (ep {h_epoch}) vs HeteroMGN Global VM (ep {g_epoch}) "
    f"vs Corner-Node MGN (ep {c_epoch})",
    fontsize=11, fontweight="bold",
)
ax.legend(fontsize=9, ncol=3)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
