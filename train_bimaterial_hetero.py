"""
train_bimaterial_hetero.py
Bipartite Heterogeneous GNN for bimaterial beam — Hasse Graph Trick (Theorem 8.4).

Graph structure (TDL CC framework):
  node  → node    : mesh edges  (9D edge features, SAGEConv)
  node  → element : B_{0,2}     (node i belongs to element j)
  element → node  : B_{0,2}^T  (element j contains node i)

Node features    : disp(3) + prev_vel(3) + node_type_oh(3)     [9D, normalized]
Element features : mat_type(1) + E_norm(1) + prev_stress(1)   [3D, teacher forcing]

Prediction targets:
  corner node  : velocity (3D)          — same as existing pipeline
  element node : von Mises stress (1D)  — avoids interface GT averaging problem
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import SAGEConv, MessagePassing
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint

from fem_stress import _D_batch, _gauss_c3d8, _gauss_stress_at


# ─── Per-element stress (Gauss-averaged) ─────────────────────────────────────

def compute_elem_vm_stress(disp_t, ref_pos, elem_conn, elem_E, elem_nu):
    """
    Per-element von Mises stress for C3D8 via Gauss-point averaging.
    C3D8: all 8 Gauss points have equal weight = 1.0 → simple arithmetic mean.

    disp_t    : (N, 3) float32  — displacement at time t
    ref_pos   : (N, 3) float32  — reference mesh positions
    elem_conn : (E_elem, 8) int64
    elem_E    : (E_elem,) float64
    elem_nu   : (E_elem,) float64
    Returns   : (E_elem,) float32 von Mises stress
    """
    dev = disp_t.device
    xe  = ref_pos[elem_conn].to(torch.float64)          # (E_elem, 8, 3)
    ue  = disp_t[elem_conn].to(torch.float64)           # (E_elem, 8, 3)
    D   = _D_batch(elem_E.to(dev), elem_nu.to(dev))     # (E_elem, 6, 6)

    sig_sum = torch.zeros(elem_conn.shape[0], 6, dtype=torch.float64, device=dev)
    for dNdxi_cpu, _ in _gauss_c3d8():                  # weight=1.0 for all 8 pts
        sig_sum += _gauss_stress_at(xe, ue, dNdxi_cpu, D)
    sig_avg = sig_sum / 8.0                              # (E_elem, 6)

    s  = sig_avg
    vm = torch.sqrt(0.5 * (
        (s[:, 0] - s[:, 1])**2 +
        (s[:, 1] - s[:, 2])**2 +
        (s[:, 2] - s[:, 0])**2 +
        6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
    ) + 1e-30)
    return vm.float()


# ─── B_{0,2} edge construction ───────────────────────────────────────────────

def build_b02_edges(elem_conn: np.ndarray):
    """
    Build B_{0,2} incidence edges from node to element and back.

    elem_conn : (E_elem, 8) int64 numpy array
    Returns   : (n2e, e2n) each (2, E_elem*8) int64 tensors
                  n2e: source=node_id, dst=elem_id  (node→element)
                  e2n: source=elem_id, dst=node_id  (element→node)
    """
    n_elem, npe = elem_conn.shape
    elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), npe)
    node_ids = elem_conn.reshape(-1).astype(np.int64)
    n2e = torch.from_numpy(np.stack([node_ids, elem_ids]))   # (2, E*8)
    e2n = torch.from_numpy(np.stack([elem_ids, node_ids]))   # (2, E*8)
    return n2e, e2n


# ─── Dataset ─────────────────────────────────────────────────────────────────

class HeteroDataset(torch.utils.data.Dataset):
    """
    Loads preprocessed .pt files and raw .npz (for elem_conn/mat).
    Builds HeteroData with element nodes; normalizes node features at build time.

    node_stats keys used:
        disp_mean, disp_std       — for displacement features
        velocity_mean, velocity_std — for prev_vel features
        bmat_stress_mean/std      — passed in via elem_stats

    elem_stats: {"mean": float, "std": float} — for element stress GT
    """

    def __init__(self, preprocess_dir, raw_dir, sample_indices, node_stats, elem_stats,
                 elem_E_min, elem_E_max, cache_path=None):
        self.hdata = []

        # Load from disk cache if available
        if cache_path and os.path.exists(cache_path):
            print(f"Loading HeteroData from cache: {cache_path}")
            self.hdata = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"  {len(self.hdata)} timesteps loaded from cache.")
            return

        # Node feature normalization tensors (built once)
        d_mean = torch.tensor(node_stats["disp_mean"],     dtype=torch.float32)  # (3,)
        d_std  = torch.tensor(node_stats["disp_std"],      dtype=torch.float32).clamp(min=1e-10)
        v_mean = torch.tensor(node_stats["velocity_mean"], dtype=torch.float32)  # (3,)
        v_std  = torch.tensor(node_stats["velocity_std"],  dtype=torch.float32).clamp(min=1e-10)
        e_mean = float(elem_stats["mean"])
        e_std  = max(float(elem_stats["std"]), 1e-10)
        E_range = float(elem_E_max - elem_E_min)  # for min-max normalization of E

        print(f"Building HeteroData for {len(sample_indices)} samples ...")
        for sid in tqdm(sample_indices):
            pt_path  = os.path.join(preprocess_dir, f"sample_{sid:05d}.pt")
            npz_path = os.path.join(raw_dir,        f"sample_{sid:05d}.npz")
            if not os.path.exists(pt_path) or not os.path.exists(npz_path):
                continue

            # Topology from raw .npz (elem_conn/mat not stored in .pt)
            raw          = np.load(npz_path)
            elem_conn_np = raw["elem_conn"].astype(np.int64)   # (E_elem, 8)
            elem_mat_np  = raw["elem_mat"].astype(np.int64)    # (E_elem,)
            mat_E        = raw["mat_E"].astype(np.float64)     # (n_mat,)
            mat_nu       = raw["mat_nu"].astype(np.float64)    # (n_mat,)

            E_ref = float(mat_E[0])  # consistent with preprocess_bimaterial.py
            elem_E_arr  = mat_E[elem_mat_np]    # (E_elem,)
            elem_nu_arr = mat_nu[elem_mat_np]   # (E_elem,)

            n2e, e2n  = build_b02_edges(elem_conn_np)
            ec_t      = torch.from_numpy(elem_conn_np)
            elem_E_t  = torch.from_numpy(elem_E_arr)
            elem_nu_t = torch.from_numpy(elem_nu_arr)

            # Static element features [mat_type, E_norm] — same for all timesteps
            mat_type  = torch.from_numpy(elem_mat_np.astype(np.float32)).unsqueeze(1)                          # (E, 1)
            E_norm    = torch.from_numpy(((elem_E_arr - elem_E_min) / E_range).astype(np.float32)).unsqueeze(1)  # (E, 1)
            elem_base = torch.cat([mat_type, E_norm], dim=1)                                                    # (E, 2)

            steps = torch.load(pt_path, map_location="cpu", weights_only=False)
            for item in steps:
                t = int(item["graph"].step_index.item())
                if t == 0:
                    continue  # skip t=0: all zeros, model can't infer dynamics

                graph = item["graph"]

                # Normalize node features (graph.x is raw, normalized at runtime in orig pipeline)
                x_raw  = graph.x                                # (N, 10) raw
                disp_n = (x_raw[:, :3]  - d_mean) / d_std      # (N, 3)
                vel_n  = (x_raw[:, 3:6] - v_mean) / v_std      # (N, 3)
                nt_oh  = x_raw[:, 6:9]                          # (N, 3) one-hot, no normalization
                node_x = torch.cat([disp_n, vel_n, nt_oh], dim=1)  # (N, 9)

                ref_pos  = graph.mesh_pos.float()                        # (N, 3)

                # GT stress at t (teacher-forced prev_stress input feature)
                disp_t    = (graph.world_pos - graph.mesh_pos).float()
                gt_str_t  = compute_elem_vm_stress(disp_t, ref_pos, ec_t, elem_E_t, elem_nu_t)
                prev_s_n  = (gt_str_t - e_mean) / e_std               # (E,) normalized

                # GT stress at t+1 (target, consistent with velocity target convention)
                vel_t    = (graph.y[:, :3] * v_std + v_mean).float()    # physical velocity at t
                disp_tp1 = (graph.world_pos + vel_t - graph.mesh_pos).float()  # disp at t+1
                elem_vm   = compute_elem_vm_stress(disp_tp1, ref_pos, ec_t, elem_E_t, elem_nu_t)
                elem_vm_n = (elem_vm - e_mean) / e_std                   # (E,) normalized

                # Element node features: static material props + current GT stress (teacher forcing)
                elem_x = torch.cat([elem_base, prev_s_n.unsqueeze(1)], dim=1)  # (E, 3)

                hd = HeteroData()

                # Corner nodes
                hd["node"].x = node_x                   # (N, 9) normalized
                hd["node"].y = graph.y[:, :3]           # velocity target (N, 3) already normalized

                # Element nodes
                hd["element"].x = elem_x                          # (E, 3)
                hd["element"].y = elem_vm_n.unsqueeze(1)          # (E, 1) normalized

                # Edges
                # node→node: multigraph mesh edges (edge_attr from item, already normalized)
                hd["node",    "mesh", "node"   ].edge_index = graph.edge_index
                hd["node",    "mesh", "node"   ].edge_attr  = item["mesh_edge_features"]  # (E_mg, 9)
                # B_{0,2}: node↔element
                hd["node",    "in",   "element"].edge_index = n2e
                hd["element", "has",  "node"   ].edge_index = e2n

                self.hdata.append(hd)

        print(f"  {len(self.hdata)} timesteps loaded.")

        # Save to disk cache for future runs
        if cache_path:
            print(f"Saving HeteroData cache to: {cache_path} ...")
            torch.save(self.hdata, cache_path)
            print(f"  Cache saved ({len(self.hdata)} items).")

    def __len__(self):
        return len(self.hdata)

    def __getitem__(self, idx):
        return self.hdata[idx]


# ─── Model ───────────────────────────────────────────────────────────────────

class NodeEdgeConv(MessagePassing):
    """
    MLP-based message passing with edge features for node→node edges.
    message(v→u) = MLP(cat(x_v, edge_attr_{v→u}))
    Matches the original MeshGraphNet edge conv design.
    """

    def __init__(self, node_dim, edge_dim, out_dim, hidden):
        super().__init__(aggr="sum")
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))


class MLP(nn.Module):
    """MLP with SiLU and LayerNorm (internal normalization per layer)."""

    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class HeteroMGN(nn.Module):
    """
    Bipartite Heterogeneous MeshGraphNet (Hasse Graph Trick, Theorem 8.4).

    Three edge types per processor layer:
      node → node    : NodeEdgeConv with 9D mesh edge features (MLP-based, like original MGN)
      node → element : SAGEConv  (B_{0,2}, no edge features)
      element → node : SAGEConv  (B_{0,2}^T, no edge features)

    Separate encoders and decoders for each node type.
    """

    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim

        # Encoders
        self.node_enc = MLP(cfg.num_node_features, h, h)   # 9 → h
        self.elem_enc = MLP(cfg.num_elem_features, h, h)   # 3 → h
        self.edge_enc = MLP(cfg.num_edge_features, h, h)   # 8 → h  (encoded once, reused each layer)

        # Processor layers (explicit per edge type for clarity)
        self.node_node_conv = nn.ModuleList([
            NodeEdgeConv(h, h, h, h)   # node_dim=h, edge_dim=h (after edge encoding)
            for _ in range(cfg.processor_size)
        ])
        self.node_elem_conv = nn.ModuleList([
            SAGEConv((h, h), h, aggr="mean")               # node → element
            for _ in range(cfg.processor_size)
        ])
        self.elem_node_conv = nn.ModuleList([
            SAGEConv((h, h), h, aggr="mean")               # element → node
            for _ in range(cfg.processor_size)
        ])

        # Post-message residual update MLPs
        self.node_mlp = nn.ModuleList([MLP(h, h, h) for _ in range(cfg.processor_size)])
        self.elem_mlp = nn.ModuleList([MLP(h, h, h) for _ in range(cfg.processor_size)])

        # Decoders
        self.node_dec = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, cfg.num_node_outputs)
        )
        self.elem_dec = nn.Sequential(
            nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1)
        )

    def forward(self, data):
        x_n = self.node_enc(data["node"].x)
        x_e = self.elem_enc(data["element"].x)

        ei_nn  = data["node",    "mesh", "node"   ].edge_index
        ea_nn  = data["node",    "mesh", "node"   ].edge_attr[:, :8]  # (E_mesh, 8): drop E_norm at [:, 8]
        ei_n2e = data["node",    "in",   "element"].edge_index
        ei_e2n = data["element", "has",  "node"   ].edge_index

        ea_emb = self.edge_enc(ea_nn)   # (E_mesh, h) — encode once, reuse each processor layer

        for i in range(len(self.node_node_conv)):
            msg_nn  = self.node_node_conv[i](x_n, ei_nn, ea_emb)  # (N, h) — with encoded edge feats
            msg_n2e = self.node_elem_conv[i]((x_n, x_e), ei_n2e)  # (E, h)
            msg_e2n = self.elem_node_conv[i]((x_e, x_n), ei_e2n)  # (N, h)

            # Nodes receive from mesh neighbors + elements; elements receive from nodes
            x_n = self.node_mlp[i](msg_nn + msg_e2n + x_n)
            x_e = self.elem_mlp[i](msg_n2e + x_e)

        vel_pred    = self.node_dec(x_n)   # (N, 3)
        stress_pred = self.elem_dec(x_e)   # (E_elem, 1)
        return vel_pred, stress_pred


# ─── Training ────────────────────────────────────────────────────────────────

@hydra.main(config_path="conf", config_name="config_bimaterial_hetero", version_base=None)
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist   = DistributedManager()
    device = dist.device

    if dist.rank == 0:
        writer = SummaryWriter(to_absolute_path(cfg.tensorboard_log_dir))
        print(f"Device: {device}  |  world_size: {dist.world_size}")

    # ── Normalization stats ───────────────────────────────────────────────────
    stats_path = os.path.join(to_absolute_path(cfg.preprocess_output_dir),
                              "node_stats_bimaterial.json")
    with open(stats_path) as f:
        node_stats = json.load(f)

    elem_stats = {
        "mean": float(node_stats["bmat_stress_mean"]),
        "std":  float(node_stats["bmat_stress_std"]),
    }

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_indices = list(range(cfg.train_start, cfg.train_end))
    cache_path = os.path.join(
        to_absolute_path(cfg.preprocess_output_dir),
        f"hetero_cache_{cfg.train_start}_{cfg.train_end}.pt",
    )
    dataset = HeteroDataset(
        preprocess_dir=to_absolute_path(cfg.preprocess_output_dir),
        raw_dir       =to_absolute_path(cfg.raw_data_dir),
        sample_indices=train_indices,
        node_stats    =node_stats,
        elem_stats    =elem_stats,
        elem_E_min    =float(cfg.elem_E_min),
        elem_E_max    =float(cfg.elem_E_max),
        cache_path    =cache_path,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, drop_last=True,
        num_replicas=dist.world_size, rank=dist.rank,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size     =cfg.batch_size,
        sampler        =sampler,
        num_workers    =cfg.num_dataloader_workers,
        collate_fn         =Batch.from_data_list,
        pin_memory         =True,
        persistent_workers =(cfg.num_dataloader_workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model     = HeteroMGN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate ** epoch
    )
    scaler = GradScaler()

    os.makedirs(to_absolute_path(cfg.ckpt_path), exist_ok=True)
    epoch_init = load_checkpoint(
        to_absolute_path(cfg.ckpt_path),
        models=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, device=device,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(epoch_init, cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_vel = total_stress = total_n = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Input noise on normalized displacement (equiv. to existing pipeline's noise_std * d_std / d_std)
            if cfg.input_noise_std > 0:
                node_x = batch["node"].x.clone()
                node_x[:, :3] += torch.randn_like(node_x[:, :3]) * cfg.input_noise_std
                batch["node"].x = node_x

            with autocast("cuda", enabled=cfg.amp):
                vel_pred, stress_pred = model(batch)

                vel_gt    = batch["node"].y
                stress_gt = batch["element"].y

                loss_vel    = F.mse_loss(vel_pred, vel_gt)
                loss_stress = F.mse_loss(stress_pred, stress_gt)
                loss        = cfg.w_vel * loss_vel + cfg.w_stress * loss_stress

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            total_vel    += loss_vel.item()
            total_stress += loss_stress.item()
            total_n      += 1

        scheduler.step()

        if dist.rank == 0:
            avg_vel    = total_vel    / max(total_n, 1)
            avg_stress = total_stress / max(total_n, 1)
            print(f"Epoch {epoch:4d} | vel={avg_vel:.3e}  stress={avg_stress:.3e}")
            writer.add_scalar("loss/vel",    avg_vel,    epoch)
            writer.add_scalar("loss/stress", avg_stress, epoch)

            if epoch % cfg.ckpt_interval == 0:
                save_checkpoint(
                    to_absolute_path(cfg.ckpt_path),
                    models=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, epoch=epoch,
                )

    if dist.rank == 0:
        writer.close()


if __name__ == "__main__":
    main()
