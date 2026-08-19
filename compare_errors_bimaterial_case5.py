"""
compare_errors_bimaterial_case5.py
=====================================
Scatter Pred vs GT for bimaterial beam — Case 5 (Hetero MGN, elem-level stress).

2 rows (Train / Test) x 3 cols (Displacement | VM Stress Mat1 | VM Stress Mat2)
Train: N_TRAIN random samples from indices 0-999    (seed=42)
Test:  N_TEST  random samples from indices 1000-end (seed=42)

GT element stress = compute_elem_vm_stress from GT positions at t+1
                    (identical to training GT — Gauss-point averaged C3D8)
Pred stress       = model elem_dec output, denormalized
                    (collected from rollout frames 2..T, skip seed frame 1)

Usage:
  python compare_errors_bimaterial_case5.py          # default 10+10
  python compare_errors_bimaterial_case5.py 20 20    # custom
"""

import os
import sys
import glob
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, MessagePassing

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.utils import load_checkpoint
from fem_stress import _D_batch, _gauss_c3d8, _gauss_stress_at

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = "./04_preprocessed_pt_bimaterial"
NPZ_DIR   = "./02_abaqus_npz_bimaterial"
CKPT_PATH = "./checkpoints_bimaterial_hetero"
OUT_PNG   = "./compare_errors_bimaterial_case5_finalstep.png"

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_TEST  = int(sys.argv[2]) if len(sys.argv) > 2 else 10

ELEM_E_MIN = 50.0e9
ELEM_E_MAX = 200.0e9

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

# ── Model definition (same architecture as train_bimaterial_hetero.py) ────────

class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x): return self.net(x)


class _NodeEdgeConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim, hidden):
        super().__init__(aggr="sum")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    def message(self, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))


class HeteroMGN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.node_enc = _MLP(cfg.num_node_features, h, h)
        self.elem_enc = _MLP(cfg.num_elem_features, h, h)
        self.edge_enc = _MLP(cfg.num_edge_features, h, h)
        self.node_node_conv = nn.ModuleList([_NodeEdgeConv(h, h, h, h) for _ in range(cfg.processor_size)])
        self.node_elem_conv = nn.ModuleList([SAGEConv((h, h), h, aggr="mean") for _ in range(cfg.processor_size)])
        self.elem_node_conv = nn.ModuleList([SAGEConv((h, h), h, aggr="mean") for _ in range(cfg.processor_size)])
        self.node_mlp = nn.ModuleList([_MLP(h, h, h) for _ in range(cfg.processor_size)])
        self.elem_mlp = nn.ModuleList([_MLP(h, h, h) for _ in range(cfg.processor_size)])
        self.node_dec = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cfg.num_node_outputs))
        self.elem_dec = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, data):
        x_n = self.node_enc(data["node"].x)
        x_e = self.elem_enc(data["element"].x)
        ei_nn  = data["node",    "mesh", "node"   ].edge_index
        ea_nn  = data["node",    "mesh", "node"   ].edge_attr[:, :8]   # drop E_norm col
        ei_n2e = data["node",    "in",   "element"].edge_index
        ei_e2n = data["element", "has",  "node"   ].edge_index
        ea_emb = self.edge_enc(ea_nn)
        for i in range(len(self.node_node_conv)):
            msg_nn  = self.node_node_conv[i](x_n, ei_nn, ea_emb)
            msg_n2e = self.node_elem_conv[i]((x_n, x_e), ei_n2e)
            msg_e2n = self.elem_node_conv[i]((x_e, x_n), ei_e2n)
            x_n = self.node_mlp[i](msg_nn + msg_e2n + x_n)
            x_e = self.elem_mlp[i](msg_n2e + x_e)
        return self.node_dec(x_n), self.elem_dec(x_e)


MODEL_CFG = SimpleNamespace(
    hidden_dim=128, processor_size=4,
    num_node_features=9, num_edge_features=8,
    num_elem_features=3, num_node_outputs=3,
)

# ── Load stats ────────────────────────────────────────────────────────────────

node_stats = load_json(os.path.join(DATA_DIR, "node_stats_bimaterial.json"))
v_mean_cpu = node_stats["velocity_mean"]          # (3,) tensor on CPU
v_std_cpu  = node_stats["velocity_std"]
d_mean_cpu = node_stats["disp_mean"]
d_std_cpu  = node_stats["disp_std"]
elem_s_mean = float(node_stats.get("bmat_stress_mean", 0.0))
elem_s_std  = max(float(node_stats.get("bmat_stress_std", 1.0)), 1e-6)

edge_stats = load_json(os.path.join(DATA_DIR, "edge_stats_bimaterial.json"))
e_mean_cpu = edge_stats["edge_mean"]              # (9,) tensor
e_std_cpu  = edge_stats["edge_std"]

# ── Load model ────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

v_mean = v_mean_cpu.to(device); v_std = v_std_cpu.to(device)
d_mean = d_mean_cpu.to(device); d_std = d_std_cpu.to(device)
e_mean = e_mean_cpu.to(device); e_std = e_std_cpu.to(device)

model = HeteroMGN(MODEL_CFG).to(device)
model.eval()
load_checkpoint(CKPT_PATH, models=model, device=device)

ckpt_files = glob.glob(os.path.join(CKPT_PATH, "HeteroMGN.0.*.pt"))
latest_epoch = (max(int(os.path.basename(f).split(".")[2]) for f in ckpt_files)
                if ckpt_files else 0)
print(f"Checkpoint epoch: {latest_epoch}")

# ── Random sample selection ───────────────────────────────────────────────────

all_files = sorted(f for f in os.listdir(DATA_DIR)
                   if f.startswith("sample_") and f.endswith(".pt"))
print(f"Found {len(all_files)} .pt files")

rng = np.random.default_rng(42)
train_files = [f for f in all_files if int(f.split("_")[1].split(".")[0]) < 1000]
test_files  = [f for f in all_files if int(f.split("_")[1].split(".")[0]) >= 1000]
train_sel = sorted(rng.choice(train_files, min(N_TRAIN, len(train_files)), replace=False).tolist())
test_sel  = sorted(rng.choice(test_files,  min(N_TEST,  len(test_files)),  replace=False).tolist())
print(f"Train ({len(train_sel)}): {train_sel[0]} .. {train_sel[-1]}")
print(f"Test  ({len(test_sel)}): {test_sel[0]} .. {test_sel[-1]}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_elem_vm_stress(disp_t, ref_pos, elem_conn, elem_E, elem_nu):
    """Element-level von Mises stress — identical to training GT computation."""
    dev = disp_t.device
    xe  = ref_pos[elem_conn].to(torch.float64)
    ue  = disp_t[elem_conn].to(torch.float64)
    D   = _D_batch(elem_E.to(dev), elem_nu.to(dev))
    sig_sum = torch.zeros(elem_conn.shape[0], 6, dtype=torch.float64, device=dev)
    for dNdxi_cpu, _ in _gauss_c3d8():
        sig_sum += _gauss_stress_at(xe, ue, dNdxi_cpu, D)
    sig_avg = sig_sum / 8.0
    s = sig_avg
    vm = torch.sqrt(0.5 * (
        (s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2 +
        6.0*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2)
    ) + 1e-30)
    return vm.float()


def build_singlegraph(elem_conn, elem_mat, mat_E):
    seen = set(); rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E[mat_idx])
        for a, b in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u,v),(v,u)):
                if (src,dst) not in seen:
                    seen.add((src,dst))
                    rows.append(src); cols.append(dst); E_vals.append(E_elem)
    return np.array([rows,cols], dtype=np.int64), np.array(E_vals, dtype=np.float32)


def build_b02_edges(elem_conn):
    n_elem, npe = elem_conn.shape
    elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    return (torch.from_numpy(np.stack([node_ids, elem_ids])),
            torch.from_numpy(np.stack([elem_ids, node_ids])))


def build_ef(ref_pos, cur, edge_index, edge_E, E_ref):
    src, dst = edge_index
    d_mesh  = ref_pos[dst] - ref_pos[src]
    d_world = cur[dst]     - cur[src]
    ef = torch.cat([
        d_mesh,  d_mesh.norm(dim=1, keepdim=True),
        d_world, d_world.norm(dim=1, keepdim=True),
        (edge_E / E_ref).unsqueeze(1),
    ], dim=1)   # (E, 9) — model slices to [:, :8] internally
    return (ef - e_mean) / e_std


def r2_mae(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    ss_res = np.sum((gt - pred)**2)
    ss_tot = np.sum((gt - gt.mean())**2)
    return 1.0 - ss_res / (ss_tot + 1e-30), np.mean(np.abs(gt - pred))


def elem_to_node_mean(elem_vals, elem_conn, N):
    ns = np.zeros(N, np.float64); nc = np.zeros(N, np.int64)
    np.add.at(ns, elem_conn.ravel(), np.repeat(elem_vals, 8))
    np.add.at(nc, elem_conn.ravel(), np.ones(elem_conn.size, np.int64))
    return (ns / np.maximum(nc, 1)).astype(np.float32)


# ── Single-sample rollout ─────────────────────────────────────────────────────

def rollout_sample(fname):
    data = torch.load(os.path.join(DATA_DIR, fname), map_location="cpu", weights_only=False)
    sid  = fname.replace(".pt", "")
    npz  = np.load(os.path.join(NPZ_DIR, sid + ".npz"))

    T = len(data)                           # number of training steps
    E1 = float(npz["mat_E"][0])

    elem_conn_np = npz["elem_conn"].astype(np.int64)    # (M, 8)
    elem_mat_np  = npz["elem_mat"].astype(np.int64)     # (M,)
    mat_E        = npz["mat_E"].astype(np.float64)      # (2,)
    mat_nu       = npz["mat_nu"].astype(np.float64)     # (2,)

    mat1_mask = elem_mat_np == 0
    mat2_mask = elem_mat_np == 1

    N = int(data[0]["graph"].mesh_pos.shape[0])
    mat1_node = np.zeros(N, bool)
    mat2_node = np.zeros(N, bool)
    mat1_node[np.unique(elem_conn_np[mat1_mask].ravel())] = True
    mat2_node[np.unique(elem_conn_np[mat2_mask].ravel())] = True

    # Graph topology
    ei_np, eE_np = build_singlegraph(elem_conn_np, elem_mat_np, mat_E)
    edge_index   = torch.from_numpy(ei_np).to(device)
    edge_E       = torch.from_numpy(eE_np).to(device)
    n2e, e2n     = build_b02_edges(elem_conn_np)
    n2e = n2e.to(device); e2n = e2n.to(device)

    # Static element features
    elem_E_arr   = mat_E[elem_mat_np]
    elem_nu_arr  = mat_nu[elem_mat_np]
    mat_type     = torch.from_numpy(elem_mat_np.astype(np.float32)).unsqueeze(1).to(device)
    E_norm       = torch.from_numpy(
        ((elem_E_arr - ELEM_E_MIN) / (ELEM_E_MAX - ELEM_E_MIN)).astype(np.float32)
    ).unsqueeze(1).to(device)
    elem_base    = torch.cat([mat_type, E_norm], dim=1)   # (M, 2)

    elem_conn_t  = torch.from_numpy(elem_conn_np).to(device)
    elem_E_t     = torch.from_numpy(elem_E_arr).to(device)
    elem_nu_t    = torch.from_numpy(elem_nu_arr).to(device)

    ref_pos   = data[0]["graph"].mesh_pos.to(device)   # (N, 3)
    node_type = data[0]["graph"].x[:, 6:9].to(device)  # (N, 3) one-hot

    # Reconstruct GT world positions
    gt_wp = [data[0]["graph"].world_pos.clone()]        # t=0
    for step in data:
        gt_wp.append(gt_wp[-1] + step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)
    # gt_wp[t+1] = GT position at frame t+1

    # Seed at t=1: compute GT element stress at frame 1
    disp_t1 = (gt_wp[1].to(device) - ref_pos).float()
    with torch.inference_mode():
        stress_seed = compute_elem_vm_stress(disp_t1, ref_pos, elem_conn_t, elem_E_t, elem_nu_t)
    prev_stress_norm = ((stress_seed - elem_s_mean) / elem_s_std).unsqueeze(1)  # (M, 1)

    cur      = gt_wp[1].to(device)
    prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
    pred_wp  = [gt_wp[0], gt_wp[1]]
    pred_elem_stress = []   # pred_elem_stress[i] = elem stress at pred_wp[i+2]

    with torch.inference_mode():
        for _ in range(T - 1):
            disp_n = (cur - ref_pos - d_mean) / d_std
            vel_n  = (prev_vel - v_mean) / v_std
            node_x = torch.cat([disp_n, vel_n, node_type], dim=1)

            ef     = build_ef(ref_pos, cur, edge_index, edge_E, E1)
            elem_x = torch.cat([elem_base, prev_stress_norm], dim=1)   # (M, 3)

            hd = HeteroData()
            hd["node"].x    = node_x
            hd["element"].x = elem_x
            hd["node",    "mesh", "node"   ].edge_index = edge_index
            hd["node",    "mesh", "node"   ].edge_attr  = ef
            hd["node",    "in",   "element"].edge_index = n2e
            hd["element", "has",  "node"   ].edge_index = e2n

            vel_pred, stress_pred = model(hd)
            prev_stress_norm = stress_pred.detach()

            vel      = vel_pred * v_std + v_mean
            prev_vel = vel
            cur      = cur + vel
            pred_wp.append(cur.clone())
            pred_elem_stress.append(
                (stress_pred.squeeze(1) * elem_s_std + elem_s_mean).cpu().numpy()
            )

    # Collect final step only (t = T-1, i.e., frame T)
    ref_np = ref_pos.cpu().numpy()
    out = {k: [] for k in
           ["disp_gt","disp_pred","vm_gt_m1","vm_pred_m1","vm_gt_m2","vm_pred_m2"]}

    for t in [T - 1]:
        gt_np   = gt_wp[t + 1].numpy()       # GT position at frame t+1
        pred_np = pred_wp[t + 1].cpu().numpy()

        # Displacement magnitude (meters)
        out["disp_gt"].append(np.linalg.norm(gt_np - ref_np, axis=1))
        out["disp_pred"].append(np.linalg.norm(pred_np - ref_np, axis=1))

        # GT element stress: same as training GT (Gauss-averaged C3D8 from GT positions)
        disp_gt_tp1 = torch.from_numpy((gt_np - ref_np).astype(np.float32)).to(device)
        with torch.inference_mode():
            gt_vm = compute_elem_vm_stress(
                disp_gt_tp1, ref_pos, elem_conn_t, elem_E_t, elem_nu_t
            ).cpu().numpy()   # (M,) in Pa

        # Pred element stress from rollout (rollout step t-1 → frame t+1)
        pred_vm = pred_elem_stress[t - 1]   # (M,) in Pa

        # Convert Pa → MPa, elem_to_node_mean, split by inclusive node masks
        gt_vm_node   = elem_to_node_mean(gt_vm   / 1e6, elem_conn_np, N)
        pred_vm_node = elem_to_node_mean(pred_vm / 1e6, elem_conn_np, N)
        out["vm_gt_m1"].append(gt_vm_node[mat1_node])
        out["vm_pred_m1"].append(pred_vm_node[mat1_node])
        out["vm_gt_m2"].append(gt_vm_node[mat2_node])
        out["vm_pred_m2"].append(pred_vm_node[mat2_node])

    return {k: np.concatenate(v) for k, v in out.items()}


# ── Collect all samples ───────────────────────────────────────────────────────

def collect(fnames, label):
    keys = ["disp_gt","disp_pred","vm_gt_m1","vm_pred_m1","vm_gt_m2","vm_pred_m2"]
    arrays = {k: [] for k in keys}
    for fname in fnames:
        print(f"  [{label}] {fname} ...", end=" ", flush=True)
        try:
            res = rollout_sample(fname)
            for k in keys:
                arrays[k].append(res[k])
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
    return {k: np.concatenate(v) for k, v in arrays.items() if v}


print(f"\nCollecting {len(train_sel)} train samples ...")
train_data = collect(train_sel, "train")
print(f"Collecting {len(test_sel)} test samples ...")
test_data  = collect(test_sel,  "test")

# ── Plot ──────────────────────────────────────────────────────────────────────

COLS = [
    ("disp_gt",  "disp_pred",  "Displacement",      "mm",  "steelblue"),
    ("vm_gt_m1", "vm_pred_m1", "VM Stress — Mat 1", "MPa", "#c0392b"),
    ("vm_gt_m2", "vm_pred_m2", "VM Stress — Mat 2", "MPa", "#1a5276"),
]

fig, axes = plt.subplots(2, 3, figsize=(24, 16))
fig.suptitle(
    f"Bimaterial Case 5 (Hetero MGN) — Epoch {latest_epoch}\n"
    f"Pred vs GT  (final step only, GT = Gauss C3D8 from GT positions)",
    fontsize=20, fontweight="bold",
)

ROW_DATA  = [train_data, test_data]
ROW_LABEL = [
    f"Train ({train_sel[0]}..{train_sel[-1]})",
    f"Test  ({test_sel[0]}..{test_sel[-1]})",
]

for row, (d, rlabel) in enumerate(zip(ROW_DATA, ROW_LABEL)):
    for col, (gt_key, pd_key, title, unit, color) in enumerate(COLS):
        ax   = axes[row, col]
        gt   = d[gt_key].copy()
        pred = d[pd_key].copy()

        if unit == "mm":     # meters → mm
            gt *= 1e3; pred *= 1e3

        if len(gt) > 80000:
            idx  = rng.choice(len(gt), 80000, replace=False)
            gt   = gt[idx]; pred = pred[idx]

        ax.scatter(gt, pred, s=6, alpha=0.35, color=color, linewidths=0)
        lo = min(gt.min(), pred.min())
        hi = max(gt.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')

        if col == 0:   # Displacement — scientific notation
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(16)
            ax.yaxis.get_offset_text().set_fontsize(16)

        r2, mae = r2_mae(gt, pred)
        ax.text(0.04, 0.96,
                f"$R^2$={r2:.4f}\nMAE={mae:.4e} {unit}",
                transform=ax.transAxes, va="top", ha="left", fontsize=20,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        if row == 0:
            ax.set_title(title, fontsize=20)
        ax.set_xlabel(f"GT ({unit})", fontsize=20)
        if col == 0:
            ax.set_ylabel(f"{rlabel}\nPred ({unit})", fontsize=20)
        else:
            ax.set_ylabel(f"Pred ({unit})", fontsize=20)
        ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
