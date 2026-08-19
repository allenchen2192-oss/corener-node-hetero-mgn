"""
Train MeshGraphNet on bi-material beam (Multigraph, E as edge feature).

Differences from train_prevvel.py:
  - node features: 9D  = [disp(3), prev_vel(3), node_type_oh(3)]
  - edge features: 9D  = [d_mesh(3), n_mesh(1), d_world(3), n_world(1), E_norm(1)]
  - Multigraph: interface nodes have TWO directed edges with different E
  - stress loss disabled by default (GFDM assumes uniform E; bi-material needs
    per-node E → omit for now, set w_stress=0 in config)
  - stats files: node_stats_bimaterial.json, edge_stats_bimaterial.json
  - train indices: 0-999, test indices: 1000-1099
"""

import os
import time

import hydra
from hydra.utils import to_absolute_path
import torch
from tqdm import tqdm

from omegaconf import DictConfig

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.datapipes.gnn.utils import load_json
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

import numpy as np
from fem_stress import (_D_batch, _gauss_c3d8, _elem_nodal_stress_c3d8,
                        _vm_from_nodal)


def collate_fn(batch):
    """Pre-batch in DataLoader worker to overlap CPU/GPU work."""
    graphs    = [seq[0]["graph"]              for seq in batch]
    mesh_efs  = [seq[0]["mesh_edge_features"] for seq in batch]
    world_efs = [seq[0]["world_edge_features"] for seq in batch]
    return (
        Batch.from_data_list(graphs),
        torch.cat(mesh_efs,  dim=0),
        torch.cat(world_efs, dim=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BimaterialDataset(torch.utils.data.Dataset):
    """In-memory dataset for bi-material Multigraph samples."""

    def __init__(self, sample_dir, sample_indices):
        self.data = []
        all_files = sorted(
            [os.path.join(sample_dir, f)
             for f in os.listdir(sample_dir)
             if f.startswith("sample_") and f.endswith(".pt")]
        )
        sample_files = [all_files[i] for i in sample_indices]
        print(f"Loading {len(sample_files)} sample files into memory ...")
        for f in sample_files:
            sample_data = torch.load(f, map_location="cpu", weights_only=False)
            # Skip t=0: prev_vel=0, model can't infer dynamics
            self.data.extend(
                item for item in sample_data
                if int(item["graph"].step_index.item()) > 0
            )
        print(f"  {len(self.data)} timesteps loaded.")

    def __getitem__(self, idx):
        return [self.data[idx]]

    def __len__(self):
        return len(self.data)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class BimaterialTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp  = cfg.amp

        mlp_act = "silu" if cfg.recompute_activation else "relu"

        sample_dir = to_absolute_path(cfg.preprocess_output_dir)

        train_indices = list(range(cfg.train_start, cfg.train_end))
        dataset = BimaterialDataset(sample_dir, train_indices)

        sampler = DistributedSampler(
            dataset, shuffle=True, drop_last=True,
            num_replicas=self.dist.world_size, rank=self.dist.rank,
        )
        n_workers = cfg.num_dataloader_workers
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=sampler,
            pin_memory=False, num_workers=n_workers,
            collate_fn=collate_fn,
            persistent_workers=(n_workers > 0),
        )
        self.sampler = sampler

        self.model = HybridMeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            processor_size=cfg.processor_size,
            hidden_dim_processor=cfg.hidden_dim,
            hidden_dim_node_encoder=cfg.hidden_dim,
            hidden_dim_edge_encoder=cfg.hidden_dim,
            hidden_dim_node_decoder=cfg.hidden_dim,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        ).to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()

        self.criterion    = torch.nn.MSELoss()
        self.w_vel        = cfg.w_vel
        self.w_stress     = cfg.w_stress
        self.w_strain     = getattr(cfg, "w_strain", 0.0)
        self._train_step  = 0
        self._stress_every = getattr(cfg, "stress_every", 5)
        self._stage1_threshold = getattr(cfg, "w_stress_stage1_threshold", None)
        self._stage2_threshold = getattr(cfg, "w_stress_stage2_threshold", None)
        self._warmup_stage = 0  # 0: w_stress=0, 1: w_stress=0.5, 2: w_stress=final

        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except (ImportError, AttributeError):
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, foreach=True
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        # If ckpt_path is empty (epoch 0), optionally warm-start model weights
        # from a pre-trained velocity-only checkpoint (optimizer state is NOT loaded,
        # so the scheduler/Adam moments reset — appropriate when switching loss objectives).
        init_ckpt = getattr(cfg, "init_ckpt_path", None)
        if self.epoch_init == 0 and init_ckpt:
            load_checkpoint(
                to_absolute_path(init_ckpt),
                models=self.model,
                device=self.dist.device,
            )
            if self.dist.rank == 0:
                print(f"[warm-start] loaded model weights from {init_ckpt}")

        # Two-stage stress warmup
        if self._stage1_threshold is not None:
            if self.epoch_init == 0:
                self.w_stress = 0.0   # fresh start: no stress loss
            else:
                self._warmup_stage = 2  # resuming: assume fully warmed up

        if self.dist.rank == 0:
            self.writer = SummaryWriter(
                log_dir=to_absolute_path(cfg.tensorboard_log_dir)
            )

        self.input_noise_std  = getattr(cfg, "input_noise_std", 0.0)
        self.use_p_feature    = getattr(cfg, "use_p_feature", True)
        self.use_bmat_stress  = getattr(cfg, "use_bmat_stress", False)

        self._load_stats(cfg)
        # Precompute fixed-mesh GFDM components once (all samples share the same mesh)
        first_step = dataset.data[0]
        self._precompute_gfdm(
            first_step["graph"].mesh_pos,
            first_step["graph"].mesh_edge_index,
        )
        if self.use_bmat_stress:
            self._setup_bmat(cfg)

    def _setup_bmat(self, cfg):
        """Load fixed mesh topology for B-matrix stress (elem_conn/mat shared across all samples)."""
        npz_dir = to_absolute_path(getattr(cfg, "npz_dir", "./02_abaqus_npz_bimaterial"))
        npz0 = np.load(os.path.join(npz_dir, "sample_00000.npz"))
        dev = self.dist.device
        self._ec_bm  = torch.from_numpy(npz0["elem_conn"].astype(np.int64)).to(dev)  # (E, 8)
        self._em_bm  = torch.from_numpy(npz0["elem_mat"].astype(np.int64)).to(dev)   # (E,)
        self._nu_bm  = float(npz0["mat_nu"][0])   # nu is same for both materials
        self._gp_bm  = _gauss_c3d8()
        print(f"[bmat] elem_conn {tuple(self._ec_bm.shape)}, "
              f"elem_mat {tuple(self._em_bm.shape)}, nu={self._nu_bm}")

    def _bmat_vm_batch(self, wp, ref, mat_E_flat):
        """
        B-matrix von Mises stress for a batched graph.
        All graphs share the same elem_conn / elem_mat (fixed beam mesh).

        wp:          (B*N, 3)  world positions
        ref:         (B*N, 3)  reference positions
        mat_E_flat:  (2B,)     concatenated per-graph mat_E from PyG batch
        Returns:     (B*N,)    float32 von Mises stress
        """
        dev   = wp.device
        N     = self._N_single
        B     = wp.shape[0] // N
        ec    = self._ec_bm          # (E_elem, 8)
        E_e   = ec.shape[0]

        # Per-element E for all graphs: mat_E (B,2) -> elem_E (B, E_elem) -> (B*E_elem,)
        mat_E  = mat_E_flat.view(B, 2).to(torch.float64)
        elem_E = mat_E[:, self._em_bm].reshape(B * E_e)                         # (B*E_e,)
        elem_nu = torch.full((B * E_e,), self._nu_bm, device=dev, dtype=torch.float64)
        D_all  = _D_batch(elem_E, elem_nu)                                       # (B*E_e, 6, 6)

        # xe, ue for all graphs (vectorised over batch)
        ref_f  = ref.to(torch.float64).view(B, N, 3)
        disp_f = wp.to(torch.float64).view(B, N, 3) - ref_f
        xe_all = ref_f[:, ec, :].reshape(B * E_e, 8, 3)                         # (B*E_e, 8, 3)
        ue_all = disp_f[:, ec, :].reshape(B * E_e, 8, 3)                        # (B*E_e, 8, 3)

        # Element-NODAL stress via Gauss-to-node extrapolation
        sig_n = _elem_nodal_stress_c3d8(xe_all, ue_all, self._gp_bm, D_all)    # (B*E_e, 8, 6)

        # Vectorised scatter-average over all graphs at once (avoids B Python iterations).
        # Offset each graph's elem_conn by g*N so all graphs scatter into a single (B*N, 6) tensor.
        offsets  = torch.arange(B, device=dev, dtype=torch.int64) * N           # (B,)
        ec_batch = ec.unsqueeze(0) + offsets[:, None, None]                     # (B, E_e, 8)
        ec_flat  = ec_batch.reshape(B * E_e, 8)                                 # (B*E_e, 8)

        node_sig   = torch.zeros(B * N, 6, device=dev, dtype=torch.float64)
        node_count = torch.zeros(B * N,    device=dev, dtype=torch.float64)
        for k in range(8):                                                       # 8 kernel launches (was B*8)
            idx = ec_flat[:, k]                                                  # (B*E_e,)
            node_sig.scatter_add_(0, idx.unsqueeze(1).expand(-1, 6), sig_n[:, k, :])
            node_count.scatter_add_(0, idx,
                                    torch.ones(B * E_e, device=dev, dtype=torch.float64))
        node_sig = node_sig / node_count.clamp(min=1).unsqueeze(1)              # (B*N, 6)
        return _vm_from_nodal(node_sig).to(wp.dtype)                            # (B*N,)

    def _load_stats(self, cfg):
        data_dir   = to_absolute_path(cfg.preprocess_output_dir)
        node_stats = load_json(
            os.path.join(data_dir, "node_stats_bimaterial.json")
        )
        dev = self.dist.device
        self.v_mean = node_stats["velocity_mean"].to(dev)
        self.v_std  = node_stats["velocity_std"].to(dev)
        self.s_mean = node_stats["stress_mean"].to(dev)
        self.s_std  = node_stats["stress_std"].to(dev)
        self.d_mean = node_stats["disp_mean"].to(dev)
        self.d_std  = node_stats["disp_std"].to(dev)
        self.bmat_s_mean = torch.tensor(node_stats.get("bmat_stress_mean", 0.0),
                                        dtype=torch.float32, device=dev)
        self.bmat_s_std  = torch.tensor(node_stats.get("bmat_stress_std",  1.0),
                                        dtype=torch.float32, device=dev).clamp(min=1e-6)

    def _precompute_gfdm(self, mesh_pos, mesh_ei):
        """Precompute XtX_inv and weighted positions for the fixed mesh (once at startup)."""
        dev = self.dist.device
        ref = mesh_pos.to(dev).float()
        row = mesh_ei[0].to(dev)
        col = mesh_ei[1].to(dev)
        N   = ref.shape[0]

        r_ij  = ref[col] - ref[row]                                       # (E, 3)
        w_ij  = 1.0 / r_ij.pow(2).sum(-1, keepdim=True).clamp(min=1e-12) # (E, 1)
        wr_ij = w_ij * r_ij                                               # (E, 3)

        r_r   = wr_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)                  # (E, 3, 3)
        row_e = row.view(-1, 1, 1).expand_as(r_r)
        XtX   = torch.zeros(N, 3, 3, device=dev, dtype=torch.float32)
        XtX.scatter_add_(0, row_e, r_r)
        XtX  += 1e-6 * torch.eye(3, device=dev).unsqueeze(0)

        self._xtx_inv  = torch.linalg.inv(XtX)   # (N, 3, 3)
        self._wr_ij    = wr_ij                    # (E, 3)
        self._N_single = N
        self._E_single = int(row.shape[0])

    def _vm_stress_fast(self, wp, ref, mesh_ei, node_E, nu=0.33):
        """GFDM von Mises stress using precomputed XtX_inv (avoids redundant scatter + solve)."""
        N   = wp.shape[0]
        B   = N // self._N_single                  # graphs in this batch
        row, col = mesh_ei

        u     = wp.float() - ref.float()
        du_ij = u[col] - u[row]                    # (B*E, 3)

        wr_ij = self._wr_ij.repeat(B, 1)           # (B*E, 3)
        r_du  = wr_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)  # (B*E, 3, 3)
        row_e = row.view(-1, 1, 1).expand_as(r_du)
        XtY   = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
        XtY.scatter_add_(0, row_e, r_du)

        xtx_inv = self._xtx_inv.repeat(B, 1, 1)   # (B*N, 3, 3)
        dU      = torch.bmm(xtx_inv, XtY).permute(0, 2, 1)  # (N, 3, 3)

        E   = node_E.float()
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu  = E / (2 * (1 + nu))
        tr  = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
        sx  = lam * tr + 2 * mu * dU[:, 0, 0]
        sy  = lam * tr + 2 * mu * dU[:, 1, 1]
        sz  = lam * tr + 2 * mu * dU[:, 2, 2]
        txy = mu * (dU[:, 0, 1] + dU[:, 1, 0])
        txz = mu * (dU[:, 0, 2] + dU[:, 2, 0])
        tyz = mu * (dU[:, 1, 2] + dU[:, 2, 1])
        return torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                                  + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)

    def _vm_stress_bimaterial(self, wp, ref, mesh_ei, node_E, nu=0.33):
        """GFDM von Mises stress with per-node E (uses deduplicated mesh_ei)."""
        N = wp.shape[0]
        row, col = mesh_ei
        r_ij  = ref[col].float() - ref[row].float()
        u     = wp.float() - ref.float()
        du_ij = u[col] - u[row]
        w_ij  = 1.0 / (r_ij.pow(2).sum(-1, keepdim=True).clamp(min=1e-12))
        r_r   = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)
        r_du  = w_ij.unsqueeze(-1) * r_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)
        row_e = row.view(-1, 1, 1).expand_as(r_r)
        XtX   = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
        XtY   = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
        XtX.scatter_add_(0, row_e, r_r)
        XtY.scatter_add_(0, row_e, r_du)
        XtX  = XtX + 1e-6 * torch.eye(3, device=wp.device).unsqueeze(0)
        dU   = torch.linalg.solve(XtX, XtY).permute(0, 2, 1)
        E    = node_E.float()
        lam  = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu   = E / (2 * (1 + nu))
        tr   = dU[:, 0, 0] + dU[:, 1, 1] + dU[:, 2, 2]
        sx   = lam * tr + 2 * mu * dU[:, 0, 0]
        sy   = lam * tr + 2 * mu * dU[:, 1, 1]
        sz   = lam * tr + 2 * mu * dU[:, 2, 2]
        txy  = mu * (dU[:, 0, 1] + dU[:, 1, 0])
        txz  = mu * (dU[:, 0, 2] + dU[:, 2, 0])
        tyz  = mu * (dU[:, 1, 2] + dU[:, 2, 1])
        return torch.sqrt(0.5 * ((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2
                                  + 6*(txy**2 + txz**2 + tyz**2)) + 1e-30)

    def _strain_proxy_loss(self, pred, gt_vel, graph):
        """MSE on normalized velocity differences along mesh edges (strain proxy).
        Equivalent to training on displacement differences at next timestep,
        because the current displacement cancels when differencing pred vs GT.
        """
        mesh_ei = graph.mesh_edge_index.to(self.dist.device)
        src, dst = mesh_ei
        dv_pred = pred.float()[dst] - pred.float()[src]   # (E, 3)
        dv_gt   = gt_vel[dst]       - gt_vel[src]          # (E, 3)
        return self.criterion(dv_pred, dv_gt)

    def forward(self, graph, mesh_ef, world_ef):
        # Save clean displacement before noise — GFDM is sensitive to position error
        # because w=1/h² amplifies noise through short edges (E*noise/h → huge stress)
        clean_disp = graph.x[:, :3].to(self.dist.device).float()

        # Add input noise to displacement part only.
        # input_noise_std is in normalized-space units (σ), so multiply by d_std
        # to get physical-space noise → noise level is consistent across all P values.
        if self.model.training and self.input_noise_std > 0:
            graph = graph.clone()
            graph.x = graph.x.clone()
            noise_scale = self.input_noise_std * self.d_std   # (3,) physical units
            graph.x[:, :3] = (graph.x[:, :3]
                              + torch.randn_like(graph.x[:, :3]) * noise_scale)

        x = graph.x.to(self.dist.device)

        # Normalize node features at runtime
        disp_norm     = (x[:, :3]  - self.d_mean) / self.d_std
        prev_vel_norm = (x[:, 3:6] - self.v_mean) / self.v_std
        node_type_oh  = x[:, 6:9]   # already binary, no normalization needed
        if self.use_p_feature:
            p_norm = x[:, 9:10]     # already normalized in preprocessing
            node_x = torch.cat([disp_norm, prev_vel_norm, node_type_oh, p_norm], dim=1)  # (N, 10)
        else:
            node_x = torch.cat([disp_norm, prev_vel_norm, node_type_oh], dim=1)  # (N, 9)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_ef, world_ef, graph)

        gt_vel   = graph.y[:, :3].to(self.dist.device).float()
        loss_vel = self.criterion(pred.float(), gt_vel)

        loss_stress = torch.zeros((), device=self.dist.device)
        if self.w_stress > 0 and self._train_step % self._stress_every == 0:
            ref_all  = graph.mesh_pos.to(self.dist.device)
            vel_phys = pred.float() * self.v_std + self.v_mean
            wp_pred  = ref_all + clean_disp + vel_phys

            if self.use_bmat_stress:
                # B-matrix loss: BM(pred) vs precomputed BM(GT) from graph.bmat_stress
                mat_E_flat = graph.mat_E.to(self.dist.device)
                vm_pred    = self._bmat_vm_batch(wp_pred, ref_all, mat_E_flat)
                vm_gt      = graph.bmat_stress.to(self.dist.device).float()
                vm_pred_n  = (vm_pred - self.bmat_s_mean) / self.bmat_s_std
                vm_gt_n    = (vm_gt   - self.bmat_s_mean) / self.bmat_s_std
                loss_stress = self.criterion(vm_pred_n, vm_gt_n)
            else:
                # GFDM loss: GFDM(pred) vs GFDM(GT stored in y[:,3])
                mesh_ei = graph.mesh_edge_index.to(self.dist.device)
                node_E  = graph.node_E.to(self.dist.device).float()
                vm_pred = self._vm_stress_fast(wp_pred, ref_all, mesh_ei, node_E)
                vm_pred_n   = (vm_pred - self.s_mean.float()) / self.s_std.float()
                vm_target_n = graph.y[:, 3].to(self.dist.device).float()
                loss_stress = self.criterion(vm_pred_n, vm_target_n)

        loss_strain = torch.zeros((), device=self.dist.device)
        if self.w_strain > 0:
            loss_strain = self._strain_proxy_loss(
                pred, graph.y[:, :3].to(self.dist.device).float(), graph
            )

        loss = (self.w_vel    * loss_vel.float()
              + self.w_stress * loss_stress.float()
              + self.w_strain * loss_strain.float())
        return loss, loss_vel, loss_stress, loss_strain

    def train_step(self, graph, mesh_ef, world_ef, epoch):
        mesh_ef  = mesh_ef.to(self.dist.device)
        world_ef = world_ef.to(self.dist.device)
        self.optimizer.zero_grad()
        self._train_step += 1
        loss, loss_vel, loss_stress, loss_strain = self.forward(graph, mesh_ef, world_ef)
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
        self.scheduler.step()
        return loss, loss_vel, loss_stress, loss_strain


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config_bimaterial")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = BimaterialTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info(
        "Training started... [bimaterial Multigraph: E as edge feature]"
    )

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.sampler.set_epoch(epoch)
        trainer.w_strain = cfg.w_strain * epoch / max(cfg.epochs - 1, 1)
        start = time.time()
        step_count = epoch * len(trainer.dataloader)

        progress_bar = tqdm(
            trainer.dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            leave=False,
        )
        vel_loss_sum, n_steps = 0.0, 0
        for batch in progress_bar:
            batched_graph, batched_mesh_ef, batched_world_ef = batch
            batched_graph = batched_graph.to(dist.device)

            loss, loss_vel, loss_stress, loss_strain = trainer.train_step(
                batched_graph, batched_mesh_ef, batched_world_ef, epoch
            )
            del batched_graph, batched_mesh_ef, batched_world_ef

            vel_loss_sum += loss_vel.item()
            n_steps      += 1

            if dist.rank == 0 and step_count % 10 == 0:
                trainer.writer.add_scalar("loss/step",        loss.item(),        step_count)
                trainer.writer.add_scalar("loss_vel/step",    loss_vel.item(),    step_count)
                trainer.writer.add_scalar("loss_strain/step", loss_strain.item(), step_count)

            step_count += 1
            progress_bar.set_postfix(
                loss=f"{loss.item():.3e}",
                vel=f"{loss_vel.item():.3e}",
                stress=f"{loss_stress.item():.3e}",
                strain=f"{loss_strain.item():.3e}",
                w_s=f"{trainer.w_stress:.1f}",
                w_str=f"{trainer.w_strain:.3f}",
            )

        epoch_avg_vel = vel_loss_sum / max(n_steps, 1)

        # Two-stage stress warmup
        if trainer._stage1_threshold is not None and trainer._warmup_stage == 0:
            if epoch_avg_vel < trainer._stage1_threshold:
                trainer.w_stress = 0.5
                trainer._warmup_stage = 1
                rank_zero_logger.info(
                    f"[stress warmup] epoch {epoch+1}: avg vel_loss={epoch_avg_vel:.3e} "
                    f"< {trainer._stage1_threshold:.1e} → w_stress: 0 → 0.5"
                )
        if trainer._stage2_threshold is not None and trainer._warmup_stage == 1:
            if epoch_avg_vel < trainer._stage2_threshold:
                trainer.w_stress = cfg.w_stress
                trainer._warmup_stage = 2
                rank_zero_logger.info(
                    f"[stress warmup] epoch {epoch+1}: avg vel_loss={epoch_avg_vel:.3e} "
                    f"< {trainer._stage2_threshold:.1e} → w_stress: 0.5 → {cfg.w_stress}"
                )

        torch.cuda.empty_cache()
        rank_zero_logger.info(
            f"epoch: {epoch + 1}, loss: {loss:10.3e}, "
            f"loss_vel: {loss_vel:10.3e}, w_stress: {trainer.w_stress}, "
            f"time per epoch: {(time.time() - start):10.3e}"
        )
        if dist.rank == 0:
            trainer.writer.add_scalar("loss",     loss.detach().cpu().item(),     epoch)
            trainer.writer.add_scalar("loss_vel", loss_vel.detach().cpu().item(), epoch)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            trainer.writer.add_scalar("learning_rate", current_lr, epoch)

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            logger.info(f"Saved checkpoint (epoch {epoch + 1})")
        torch.cuda.empty_cache()

    rank_zero_logger.info("Training completed!")
    if dist.rank == 0:
        trainer.writer.close()


if __name__ == "__main__":
    main()
