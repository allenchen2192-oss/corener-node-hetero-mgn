"""
compare_errors_v2v3.py
======================
Compare v2 (9D node features, no P) vs v3 (10D, log-P normalization)
on autoregressive rollout error.

v2: checkpoints_bimaterial_v2  -- trained without P feature
v3: checkpoints_bimaterial_v3  -- trained with log(P) normalized feature

Metrics:
  Displacement Error (%) = mean_node(|pred_pos - gt_pos|) / mean_node(|gt_disp|) * 100
  Stress Error (%)       = mean_node(|BM(pred) - BM(gt)|) / mean_node(BM(gt)) * 100
  (BM = FEM B-matrix stress, per-element E, Gauss-to-node Lagrange extrapolation)
"""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch_geometric.data import Data

from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from physicsnemo.utils import load_checkpoint
from fem_stress import compute_nodal_vm_stress

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR   = "./04_preprocessed_pt_bimaterial"
NPZ_DIR    = "./02_abaqus_npz_bimaterial"
OUT_PNG    = "./compare_errors_v2.png"

TRAIN_INDICES = list(range(100))
TEST_INDICES  = list(range(1000, 1100))

HIDDEN    = 128
NUM_EDGE  = 9

C3D8_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device}")

# ── Load stats ─────────────────────────────────────────────────────────────────

node_stats = load_json(os.path.join(DATA_DIR, "node_stats_bimaterial.json"))
v_mean = node_stats["velocity_mean"].to(device)
v_std  = node_stats["velocity_std"].to(device)
d_mean = node_stats["disp_mean"].to(device)
d_std  = node_stats["disp_std"].to(device)
v_mean_cpu = v_mean.cpu()
v_std_cpu  = v_std.cpu()

P_log_mean = float(node_stats["P_log_mean"])
P_log_std  = float(node_stats["P_log_std"])

edge_stats = load_json("edge_stats_bimaterial.json")
e_mean = edge_stats["edge_mean"].to(device)
e_std  = edge_stats["edge_std"].to(device)

# Load per-sample P values from metadata CSV
sample_P = {}
meta_path = os.path.join(NPZ_DIR, "sample_metadata.csv")
with open(meta_path) as f:
    for row in csv.DictReader(f):
        sample_P[int(row["sample_id"])] = float(row["P_N"])

print(f"Loaded P for {len(sample_P)} samples  "
      f"(min={min(sample_P.values()):.0f} N, max={max(sample_P.values()):.0f} N)")

# ── Model configs ──────────────────────────────────────────────────────────────

MODELS = [
    {
        "label":      "v2  (9D, no P feature)",
        "ckpt":       "./checkpoints_bimaterial_v2",
        "color":      "steelblue",
        "marker":     "o",
        "num_inputs": 9,
        "proc_size":  4,
        "use_p":      False,
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

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


def node_E_from_npz(npz, N):
    E1, E2 = float(npz["mat_E"][0]), float(npz["mat_E"][1])
    mat_count = np.zeros((N, 2), dtype=np.int32)
    for c0, m in zip(npz["elem_conn"], npz["elem_mat"]):
        for ni in c0:
            mat_count[ni, int(m)] += 1
    mat_id = np.argmax(mat_count, axis=1)
    return torch.from_numpy(
        np.where(mat_id == 0, E1, E2).astype(np.float32)
    ).to(device)


def load_model(ckpt, num_inputs, proc_size=4):
    m = HybridMeshGraphNet(
        num_inputs, NUM_EDGE, 3,
        processor_size=proc_size,
        hidden_dim_processor=HIDDEN, hidden_dim_node_encoder=HIDDEN,
        hidden_dim_edge_encoder=HIDDEN, hidden_dim_node_decoder=HIDDEN,
        mlp_activation_fn="relu", do_concat_trick=False,
        num_processor_checkpoint_segments=0, recompute_activation=False,
    ).to(device)
    m.eval()
    load_checkpoint(ckpt, models=m, device=device)
    return m


def eval_samples(model, indices, use_p, all_files):
    disp_acc   = defaultdict(list)
    stress_acc = defaultdict(list)
    empty_wef  = torch.zeros(0, NUM_EDGE, device=device)

    for idx in indices:
        if idx >= len(all_files):
            print(f"  [skip] index {idx} out of range")
            continue
        fname   = all_files[idx]
        sid_str = fname.replace(".pt", "")
        sid     = int(sid_str.split("_")[1])

        data = torch.load(os.path.join(DATA_DIR, fname),
                          map_location="cpu", weights_only=False)
        npz  = np.load(os.path.join(NPZ_DIR, sid_str + ".npz"))
        T    = len(data)
        N    = data[0]["graph"].num_nodes
        E1   = float(npz["mat_E"][0])

        ei_np, eE_np = build_multigraph(npz["elem_conn"], npz["elem_mat"], npz["mat_E"])
        edge_index = torch.from_numpy(ei_np).to(device)
        edge_E     = torch.from_numpy(eE_np).to(device)
        ref_pos    = data[0]["graph"].mesh_pos.to(device)
        node_type  = data[0]["graph"].x[:, 6:9].to(device)

        # B-matrix: per-element material arrays
        elem_mat_idx = npz["elem_mat"]
        elem_E_t  = torch.from_numpy(npz["mat_E"][elem_mat_idx].astype(np.float64)).to(device)
        elem_nu_t = torch.from_numpy(npz["mat_nu"][elem_mat_idx].astype(np.float64)).to(device)
        elem_conn_t = torch.from_numpy(npz["elem_conn"].astype(np.int64)).to(device)

        if use_p:
            P_val  = sample_P.get(sid, 500000.0)
            p_norm = float((np.log(max(P_val, 1.0)) - P_log_mean) / P_log_std)
            p_col  = torch.full((N, 1), p_norm, device=device, dtype=torch.float32)

        # Build GT position sequence
        gt_wp = [data[0]["graph"].world_pos.clone()]
        for step in data:
            gt_wp.append(gt_wp[-1] + step["graph"].y[:, :3] * v_std_cpu + v_mean_cpu)

        # Autoregressive rollout seeded at t=1
        cur      = gt_wp[1].to(device)
        prev_vel = (data[0]["graph"].y[:, :3] * v_std_cpu + v_mean_cpu).to(device)
        pred_wp  = [gt_wp[0], gt_wp[1]]

        with torch.inference_mode():
            for _ in range(T - 1):
                ef = build_ef(ref_pos, cur, edge_index, edge_E, E1)
                disp_n = (cur - ref_pos - d_mean) / d_std
                vel_n  = (prev_vel - v_mean) / v_std
                if use_p:
                    nx = torch.cat([disp_n, vel_n, node_type, p_col], dim=1)
                else:
                    nx = torch.cat([disp_n, vel_n, node_type], dim=1)

                out      = model(nx, ef, empty_wef,
                                 Data(x=nx, edge_index=edge_index, num_nodes=N))
                vel      = out[:, :3] * v_std + v_mean
                prev_vel = vel
                cur      = cur + vel
                pred_wp.append(cur.clone())

        ref_np = ref_pos.cpu().numpy()

        for t in range(T):
            gt_np   = gt_wp[t + 1].numpy()
            pred_np = pred_wp[t + 1].cpu().numpy()

            gt_disp = np.linalg.norm(gt_np - ref_np, axis=1).mean()
            d_err   = np.linalg.norm(pred_np - gt_np, axis=1).mean()
            disp_acc[t + 1].append(d_err / max(gt_disp, 1e-10) * 100)

            with torch.inference_mode():
                gt_disp_t   = torch.from_numpy(gt_np - ref_np).to(device)
                pred_disp_t = torch.from_numpy(pred_np - ref_np).to(device)
                gt_vm   = compute_nodal_vm_stress(
                    gt_disp_t, ref_pos, elem_conn_t, 8,
                    elem_E=elem_E_t, elem_nu=elem_nu_t,
                ).cpu().numpy()
                pred_vm = compute_nodal_vm_stress(
                    pred_disp_t, ref_pos, elem_conn_t, 8,
                    elem_E=elem_E_t, elem_nu=elem_nu_t,
                ).cpu().numpy()

            s_err = np.abs(pred_vm - gt_vm).mean()
            stress_acc[t + 1].append(s_err / max(gt_vm.mean(), 1e-10) * 100)

        print(f"  [{idx:4d}] P={sample_P.get(sid,0)/1e3:.0f}kN  "
              f"disp_err={disp_acc[T][-1]:.2f}%  stress_err={stress_acc[T][-1]:.2f}%")

    T_vals = sorted(disp_acc.keys())
    return (
        T_vals,
        [np.mean(disp_acc[t])   for t in T_vals],
        [np.mean(stress_acc[t]) for t in T_vals],
    )


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
    model = load_model(m_cfg["ckpt"], m_cfg["num_inputs"], m_cfg["proc_size"])

    print(f"  Evaluating {len(TRAIN_INDICES)} training samples ...")
    t_vals, d_train, s_train = eval_samples(model, TRAIN_INDICES, m_cfg["use_p"], all_files)

    print(f"  Evaluating {len(TEST_INDICES)} test samples ...")
    _,      d_test,  s_test  = eval_samples(model, TEST_INDICES,  m_cfg["use_p"], all_files)

    results[m_cfg["label"]] = {
        "color": m_cfg["color"], "marker": m_cfg["marker"],
        "t": t_vals,
        "disp_train": d_train, "disp_test": d_test,
        "stress_train": s_train, "stress_test": s_test,
    }
    del model
    torch.cuda.empty_cache()

# ── Plot ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"v2 — rollout error (B-matrix stress)  "
    f"(train n={len(TRAIN_INDICES)}, test n={len(TEST_INDICES)})",
    fontsize=13,
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

for ax, title in zip(axes, ["Displacement Error (%)", "Von Mises Stress Error (%, B-matrix)"]):
    ax.set_title(title)
    ax.set_xlabel("Time step  t")
    ax.set_ylabel(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PNG}")
