"""
compare_errors_bimaterial_4case.py
====================================
4-way rollout comparison on bimaterial beam:

  Case 1 (v2)  : vel-only baseline
  Case 2       : 4-output (vel + direct stress pred) — stress eval via B-matrix on rollout pos
  Case 3       : vel-only output + B-matrix stress in training loss
  Case 4 (v6)  : vel-only + strain proxy loss

Metrics (B-matrix stress for ALL cases, for fair comparison):
  Displacement Error (%) = mean(|pred_pos - gt_pos|) / mean(|gt_disp|) * 100
  Stress Error (%)       = mean(|BM(pred) - BM(gt)|) / mean(BM(gt)) * 100
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from types import SimpleNamespace
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGEConv, MessagePassing

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint
from fem_stress import compute_nodal_vm_stress, _D_batch, _gauss_c3d8, _gauss_stress_at

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = "./04_preprocessed_pt_bimaterial"
NPZ_DIR  = "./02_abaqus_npz_bimaterial"
OUT_PNG  = "./compare_errors_bimaterial_4case.png"

TRAIN_INDICES = list(range(50))
TEST_INDICES  = list(range(1000, 1050))

P_MIN = 0   # N — set > 0 to filter small-P samples

HIDDEN   = 128
NUM_EDGE = 9

ELEM_E_MIN = 50.0e9
ELEM_E_MAX = 200.0e9

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

# ── Model configs ──────────────────────────────────────────────────────────────

MODELS = [
    {
        "label":  "Case 1: vel-only (v2)",
        "ckpt":   "./checkpoints_bimaterial_v2",
        "n_out":  3,
        "color":  "steelblue",
        "marker": "o",
    },
    {
        "label":        "Case 2: 4-output (vel+stress direct)",
        "ckpt":         "./checkpoints_bimaterial_case2",
        "n_out":        4,
        "direct_stress": True,   # use out[:,3] as stress (denorm), skip B-matrix from pred pos
        "color":        "darkorange",
        "marker":       "^",
    },
    {
        "label":  "Case 3: vel + B-matrix loss",
        "ckpt":   "./checkpoints_bimaterial_case3",
        "n_out":  3,
        "color":  "forestgreen",
        "marker": "s",
    },
    {
        "label":  "Case 4: vel + strain proxy (v6)",
        "ckpt":   "./checkpoints_bimaterial_v6",
        "n_out":  3,
        "color":  "tomato",
        "marker": "D",
    },
]

# ── Stats ──────────────────────────────────────────────────────────────────────

node_stats = load_json(os.path.join(DATA_DIR, "node_stats_bimaterial.json"))
v_mean = node_stats["velocity_mean"].to(device)
v_std  = node_stats["velocity_std"].to(device)
d_mean = node_stats["disp_mean"].to(device)
d_std  = node_stats["disp_std"].to(device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()
bmat_s_mean = torch.tensor(node_stats.get("bmat_stress_mean", 0.0),
                            dtype=torch.float32, device=device)
bmat_s_std  = torch.tensor(node_stats.get("bmat_stress_std",  1.0),
                            dtype=torch.float32, device=device).clamp(min=1e-6)

# elem stress denorm (same stats as bmat_stress — used in hetero training)
elem_s_mean = float(node_stats.get("bmat_stress_mean", 0.0))
elem_s_std  = max(float(node_stats.get("bmat_stress_std",  1.0)), 1e-6)

edge_stats = load_json("edge_stats_bimaterial.json")
e_mean = edge_stats["edge_mean"].to(device)
e_std  = edge_stats["edge_std"].to(device)

# Per-sample P values
sample_P = {}
meta_path = os.path.join(NPZ_DIR, "sample_metadata.csv")
with open(meta_path) as f:
    for row in csv.DictReader(f):
        sample_P[int(row["sample_id"])] = float(row["P_N"])
print(f"Loaded P for {len(sample_P)} samples")

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_multigraph(elem_conn, elem_mat, mat_E):
    """Original multigraph — key includes mat_idx; used for Cases 1-4 checkpoints."""
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


def build_singlegraph(elem_conn, elem_mat, mat_E):
    """Fixed single-graph — key is (src, dst) only; matches hetero preprocessing fix."""
    seen = set()
    rows, cols, E_vals = [], [], []
    for c0, mat_idx in zip(elem_conn, elem_mat):
        E_elem = float(mat_E[mat_idx])
        for (a, b) in C3D8_EDGES:
            u, v = int(c0[a]), int(c0[b])
            for src, dst in ((u, v), (v, u)):
                key = (src, dst)
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


# ── HeteroMGN (Case 5) ────────────────────────────────────────────────────────

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
        ea_nn  = data["node",    "mesh", "node"   ].edge_attr[:, :8]
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


def compute_elem_vm_stress(disp_t, ref_pos, elem_conn, elem_E, elem_nu):
    """Per-element von Mises stress via C3D8 Gauss-point averaging."""
    dev = disp_t.device
    xe  = ref_pos[elem_conn].to(torch.float64)
    ue  = disp_t[elem_conn].to(torch.float64)
    D   = _D_batch(elem_E.to(dev), elem_nu.to(dev))
    sig_sum = torch.zeros(elem_conn.shape[0], 6, dtype=torch.float64, device=dev)
    for dNdxi_cpu, _ in _gauss_c3d8():
        sig_sum += _gauss_stress_at(xe, ue, dNdxi_cpu, D)
    sig_avg = sig_sum / 8.0
    s  = sig_avg
    vm = torch.sqrt(0.5 * (
        (s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2 +
        6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
    ) + 1e-30)
    return vm.float()


def build_b02_edges(elem_conn):
    n_elem, npe = elem_conn.shape
    elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    n2e = torch.from_numpy(np.stack([node_ids, elem_ids]))
    e2n = torch.from_numpy(np.stack([elem_ids, node_ids]))
    return n2e, e2n


def load_hetero_model(ckpt):
    cfg = SimpleNamespace(
        hidden_dim=128, processor_size=4,
        num_node_features=9, num_edge_features=8,
        num_elem_features=3, num_node_outputs=3,
    )
    m = HeteroMGN(cfg).to(device)
    m.eval()
    load_checkpoint(ckpt, models=m, device=device)
    return m


def eval_hetero_samples(model, indices, all_files):
    disp_acc   = defaultdict(list)
    stress_acc = defaultdict(list)

    for idx in indices:
        if idx >= len(all_files):
            print(f"  [skip] index {idx} out of range"); continue
        fname   = all_files[idx]
        sid_str = fname.replace(".pt", "")
        sid     = int(sid_str.split("_")[1])

        if P_MIN > 0 and sample_P.get(sid, 0) < P_MIN:
            continue

        data = torch.load(os.path.join(DATA_DIR, fname), map_location="cpu", weights_only=False)
        npz  = np.load(os.path.join(NPZ_DIR, sid_str + ".npz"))
        T    = len(data)
        N    = data[0]["graph"].num_nodes
        E1   = float(npz["mat_E"][0])

        elem_conn_np = npz["elem_conn"].astype(np.int64)
        elem_mat_np  = npz["elem_mat"].astype(np.int64)
        mat_E        = npz["mat_E"].astype(np.float64)
        mat_nu       = npz["mat_nu"].astype(np.float64)

        # Node-node edges — single-graph (matches fixed hetero preprocessing)
        ei_np, eE_np = build_singlegraph(elem_conn_np, elem_mat_np, mat_E)
        edge_index = torch.from_numpy(ei_np).to(device)
        edge_E     = torch.from_numpy(eE_np).to(device)

        # B_{0,2} edges
        n2e, e2n = build_b02_edges(elem_conn_np)
        n2e = n2e.to(device);  e2n = e2n.to(device)

        # Element features (static base per sample; prev_stress added per rollout step)
        elem_E_arr  = mat_E[elem_mat_np]
        mat_type    = torch.from_numpy(elem_mat_np.astype(np.float32)).unsqueeze(1).to(device)
        E_norm_e    = torch.from_numpy(
            ((elem_E_arr - ELEM_E_MIN) / (ELEM_E_MAX - ELEM_E_MIN)).astype(np.float32)
        ).unsqueeze(1).to(device)
        elem_base   = torch.cat([mat_type, E_norm_e], dim=1)   # (E_elem, 2) static

        elem_E_t    = torch.from_numpy(elem_E_arr).to(device)
        elem_nu_t   = torch.from_numpy(mat_nu[elem_mat_np]).to(device)
        elem_conn_t = torch.from_numpy(elem_conn_np).to(device)

        ref_pos   = data[0]["graph"].mesh_pos.to(device)
        node_type = data[0]["graph"].x[:, 6:9].to(device)

        gt_wp = [data[0]["graph"].world_pos.clone()]
        for step in data:
            gt_wp.append(gt_wp[-1] + step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)

        cur      = gt_wp[1].to(device)
        prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
        pred_wp  = [gt_wp[0], gt_wp[1]]

        pred_elem_stress = []   # model direct element stress output per rollout step
        # initialize prev_stress from GT stress at t=1 (matches training: prev_stress=GT_stress_at_t)
        with torch.inference_mode():
            disp_t1 = (gt_wp[1].to(device) - ref_pos).float()
            gt_stress_t1 = compute_elem_vm_stress(disp_t1, ref_pos, elem_conn_t, elem_E_t, elem_nu_t)
        prev_stress_norm = ((gt_stress_t1 - elem_s_mean) / elem_s_std).unsqueeze(1)

        with torch.inference_mode():
            for _ in range(T - 1):
                disp_n = (cur - ref_pos - d_mean) / d_std
                vel_n  = (prev_vel - v_mean) / v_std
                node_x = torch.cat([disp_n, vel_n, node_type], dim=1)

                ea = build_ef(ref_pos, cur, edge_index, edge_E, E1)

                # Concatenate static material features with carried-forward stress
                elem_x = torch.cat([elem_base, prev_stress_norm], dim=1)  # (E_elem, 3)

                hd = HeteroData()
                hd["node"].x    = node_x
                hd["element"].x = elem_x
                hd["node",    "mesh", "node"   ].edge_index = edge_index
                hd["node",    "mesh", "node"   ].edge_attr  = ea
                hd["node",    "in",   "element"].edge_index = n2e
                hd["element", "has",  "node"   ].edge_index = e2n

                vel_pred, stress_pred = model(hd)
                prev_stress_norm = stress_pred.detach()   # (E_elem, 1) carry forward
                vel      = vel_pred * v_std + v_mean
                prev_vel = vel
                cur      = cur + vel
                pred_wp.append(cur.clone())
                # denormalize element stress: shape (E_elem,)
                pred_elem_stress.append(
                    (stress_pred.squeeze(1) * elem_s_std + elem_s_mean).cpu().numpy()
                )

        ref_np = ref_pos.cpu().numpy()

        for t in range(T):
            gt_np   = gt_wp[t + 1].numpy()
            pred_np = pred_wp[t + 1].cpu().numpy()

            gt_disp = np.linalg.norm(gt_np - ref_np, axis=1).mean()
            d_err   = np.linalg.norm(pred_np - gt_np, axis=1).mean()
            disp_acc[t + 1].append(d_err / max(gt_disp, 1e-10) * 100)

            # GT element-level stress from GT positions
            with torch.inference_mode():
                gt_elem_vm = compute_elem_vm_stress(
                    torch.from_numpy(gt_np - ref_np).to(device),
                    ref_pos, elem_conn_t, elem_E_t, elem_nu_t,
                ).cpu().numpy()

            if t > 0:
                # pred_elem_stress[i] corresponds to rollout step i → pred_wp[i+2]
                # t maps to rollout step t-1
                pred_vm_elem = pred_elem_stress[t - 1]
            else:
                pred_vm_elem = gt_elem_vm  # t=1 is seeded from GT, no stress prediction yet

            s_err = np.abs(pred_vm_elem - gt_elem_vm).mean()
            stress_acc[t + 1].append(s_err / max(gt_elem_vm.mean(), 1e-10) * 100)

        print(f"  [{idx:4d}] P={sample_P.get(sid,0)/1e3:.0f}kN  "
              f"disp={disp_acc[T][-1]:.2f}%  stress={stress_acc[T][-1]:.2f}%")

    T_vals = sorted(disp_acc.keys())
    return (T_vals,
            [np.mean(disp_acc[t]) for t in T_vals],
            [np.mean(stress_acc[t]) for t in T_vals])


def eval_samples(model, indices, all_files, direct_stress=False):
    disp_acc   = defaultdict(list)
    stress_acc = defaultdict(list)
    empty_wef  = torch.zeros(0, NUM_EDGE, device=device)

    for idx in indices:
        if idx >= len(all_files):
            print(f"  [skip] index {idx} out of range"); continue
        fname   = all_files[idx]
        sid_str = fname.replace(".pt", "")
        sid     = int(sid_str.split("_")[1])

        if P_MIN > 0 and sample_P.get(sid, 0) < P_MIN:
            print(f"  [skip] {sid_str}  P={sample_P.get(sid,0)/1e3:.0f}kN < {P_MIN/1e3:.0f}kN")
            continue

        data = torch.load(os.path.join(DATA_DIR, fname),
                          map_location="cpu", weights_only=False)
        npz  = np.load(os.path.join(NPZ_DIR, sid_str + ".npz"))
        T    = len(data)
        N    = data[0]["graph"].num_nodes
        E1   = float(npz["mat_E"][0])

        ei_np, eE_np = build_singlegraph(npz["elem_conn"], npz["elem_mat"], npz["mat_E"])
        edge_index  = torch.from_numpy(ei_np).to(device)
        edge_E      = torch.from_numpy(eE_np).to(device)
        ref_pos     = data[0]["graph"].mesh_pos.to(device)
        node_type   = data[0]["graph"].x[:, 6:9].to(device)

        elem_mat_idx = npz["elem_mat"]
        elem_E_t  = torch.from_numpy(npz["mat_E"][elem_mat_idx].astype(np.float64)).to(device)
        elem_nu_t = torch.from_numpy(npz["mat_nu"][elem_mat_idx].astype(np.float64)).to(device)
        elem_conn_t = torch.from_numpy(npz["elem_conn"].astype(np.int64)).to(device)

        # GT positions (t=0..T)
        gt_wp = [data[0]["graph"].world_pos.clone()]
        for step in data:
            gt_wp.append(gt_wp[-1] + step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)

        # Autoregressive rollout seeded at t=1
        cur      = gt_wp[1].to(device)
        prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
        pred_wp  = [gt_wp[0], gt_wp[1]]

        pred_stress_direct = []   # filled only when direct_stress=True
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
                    # out[:,3] is normalized bmat stress — denormalize to physical units
                    pred_stress_direct.append(
                        (out[:, 3] * bmat_s_std + bmat_s_mean).cpu().numpy()
                    )

        ref_np = ref_pos.cpu().numpy()

        for t in range(T):
            gt_np   = gt_wp[t + 1].numpy()
            pred_np = pred_wp[t + 1].cpu().numpy()

            gt_disp = np.linalg.norm(gt_np - ref_np, axis=1).mean()
            d_err   = np.linalg.norm(pred_np - gt_np, axis=1).mean()
            disp_acc[t + 1].append(d_err / max(gt_disp, 1e-10) * 100)

            with torch.inference_mode():
                gt_vm = compute_nodal_vm_stress(
                    torch.from_numpy(gt_np - ref_np).to(device),
                    ref_pos, elem_conn_t, 8,
                    elem_E=elem_E_t, elem_nu=elem_nu_t,
                ).cpu().numpy()

            if direct_stress and t > 0:
                # pred_stress_direct[i] = output from rollout step i → corresponds to pred_wp[i+2]
                # evaluation t maps to rollout step t-1
                pred_vm = pred_stress_direct[t - 1]
            else:
                with torch.inference_mode():
                    pred_vm = compute_nodal_vm_stress(
                        torch.from_numpy(pred_np - ref_np).to(device),
                        ref_pos, elem_conn_t, 8,
                        elem_E=elem_E_t, elem_nu=elem_nu_t,
                    ).cpu().numpy()

            s_err = np.abs(pred_vm - gt_vm).mean()
            stress_acc[t + 1].append(s_err / max(gt_vm.mean(), 1e-10) * 100)

        print(f"  [{idx:4d}] P={sample_P.get(sid,0)/1e3:.0f}kN  "
              f"disp={disp_acc[T][-1]:.2f}%  stress={stress_acc[T][-1]:.2f}%")

    T_vals = sorted(disp_acc.keys())
    return (T_vals,
            [np.mean(disp_acc[t]) for t in T_vals],
            [np.mean(stress_acc[t]) for t in T_vals])


# ── Main ───────────────────────────────────────────────────────────────────────

all_files = sorted(
    f for f in os.listdir(DATA_DIR)
    if f.startswith("sample_") and f.endswith(".pt")
)
print(f"Found {len(all_files)} .pt files in {DATA_DIR}")

results = {}
for m_cfg in MODELS:
    print(f"\n{'='*60}")
    print(f"=== {m_cfg['label']} ===")
    model = load_model(m_cfg["ckpt"], m_cfg["n_out"])

    ds = m_cfg.get("direct_stress", False)
    print(f"  Train ({len(TRAIN_INDICES)} samples)...")
    t_vals, d_train, s_train = eval_samples(model, TRAIN_INDICES, all_files, direct_stress=ds)
    print(f"  Test  ({len(TEST_INDICES)} samples)...")
    _,      d_test,  s_test  = eval_samples(model, TEST_INDICES,  all_files, direct_stress=ds)

    results[m_cfg["label"]] = {
        "color":  m_cfg["color"],
        "marker": m_cfg["marker"],
        "t":      t_vals,
        "disp_train":   d_train, "disp_test":   d_test,
        "stress_train": s_train, "stress_test": s_test,
    }
    print(f"  → final disp   train={d_train[-1]:.2f}%  test={d_test[-1]:.2f}%")
    print(f"  → final stress train={s_train[-1]:.2f}%  test={s_test[-1]:.2f}%")
    del model
    torch.cuda.empty_cache()

# ── Case 5: Hetero MGN ─────────────────────────────────────────────────────────

HETERO_CKPT = "./checkpoints_bimaterial_hetero"

if os.path.isdir(HETERO_CKPT):
    print(f"\n{'='*60}")
    print(f"=== Case 5: Hetero MGN (element-level stress) ===")
    hetero_model = load_hetero_model(HETERO_CKPT)
    print(f"  Train ({len(TRAIN_INDICES)} samples)...")
    t_vals, d_train, s_train = eval_hetero_samples(hetero_model, TRAIN_INDICES, all_files)
    print(f"  Test  ({len(TEST_INDICES)} samples)...")
    _,      d_test,  s_test  = eval_hetero_samples(hetero_model, TEST_INDICES,  all_files)
    results["Case 5: Hetero MGN (elem stress)"] = {
        "color":  "mediumpurple",
        "marker": "P",
        "t":      t_vals,
        "disp_train":   d_train, "disp_test":   d_test,
        "stress_train": s_train, "stress_test": s_test,
    }
    print(f"  → final disp   train={d_train[-1]:.2f}%  test={d_test[-1]:.2f}%")
    print(f"  → final stress train={s_train[-1]:.2f}%  test={s_test[-1]:.2f}%")
    del hetero_model
    torch.cuda.empty_cache()
else:
    print(f"\n[skip] Hetero checkpoint not found: {HETERO_CKPT}")

# ── Plot ───────────────────────────────────────────────────────────────────────

filter_str = f"P ≥ {P_MIN/1e3:.0f} kN" if P_MIN > 0 else ""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
n_cases = len(results)
title_line1 = f"Bimaterial beam — {n_cases}-case rollout comparison"
if filter_str:
    title_line1 += f" ({filter_str})"
fig.suptitle(
    f"{title_line1}\ntrain n={len(TRAIN_INDICES)}, test n={len(TEST_INDICES)}  |  stress metric: B-matrix",
    fontsize=12,
)

for label, res in results.items():
    c, mk, t = res["color"], res["marker"], res["t"]
    axes[0].plot(t, res["disp_train"],  color=c, marker=mk, ms=4, ls="-",
                 label=f"{label} — Train")
    axes[0].plot(t, res["disp_test"],   color=c, marker=mk, ms=4, ls="--",
                 markerfacecolor="none", label=f"{label} — Test")
    axes[1].plot(t, res["stress_train"], color=c, marker=mk, ms=4, ls="-",
                 label=f"{label} — Train")
    axes[1].plot(t, res["stress_test"],  color=c, marker=mk, ms=4, ls="--",
                 markerfacecolor="none", label=f"{label} — Test")

for ax, title in zip(axes, ["Displacement Error (%)", "Von Mises Stress Error (%)"]):
    ax.set_title(title)
    ax.set_xlabel("Time step  t")
    ax.set_ylabel(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PNG}")
