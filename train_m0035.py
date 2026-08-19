"""
Train MeshGraphNet on M0035 IC package (4-material, elastoplastic solder).

Node features (17D, pre-normalized in preprocessing):
  disp(3) + prev_vel(3) + stress(6) + peeq(1) + mat_oh(3) + CTE_norm(1)

Edge features (9D, Multigraph):
  d_mesh(3) + n_mesh(1) + d_world(3) + n_world(1) + E_elem/E_ref(1)

Output (10D):
  vel(3) + stress_next(6) + peeq_next(1)

Loss:
  L = w_vel    * MSE(vel_all)
    + w_stress * MSE(stress_all)   <- direct for all materials
    + w_peeq   * MSE(peeq[Solder])

Direct MSE for all stress: GFDM was found to be inaccurate for multi-material
domains (interface contamination corrupts Si nodes with ~1000% error).

Two-stage stress warmup:
  stage 0 → 1: vel_loss < w_stress_stage1_threshold  → w_stress = 0.5, w_peeq = 0.5*final
  stage 1 → 2: vel_loss < w_stress_stage2_threshold  → w_stress = final, w_peeq = final
"""

import json
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
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
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

class M0035Dataset(torch.utils.data.Dataset):
    """In-memory dataset: loads all timestep dicts from S*.pt files."""

    def __init__(self, sample_dir, sample_ids):
        self.data = []
        print(f"Loading {len(sample_ids)} sample files into memory ...")
        for sid in sample_ids:
            pt_path = os.path.join(sample_dir, f"{sid}.pt")
            if not os.path.exists(pt_path):
                print(f"  MISSING: {pt_path}")
                continue
            sample_data = torch.load(pt_path, map_location="cpu", weights_only=False)
            # Skip t=0: prev_vel=0 normalizes to a misleading non-zero vector
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

class M0035Trainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp  = cfg.amp

        mlp_act    = "silu" if cfg.recompute_activation else "relu"
        sample_dir = to_absolute_path(cfg.preprocess_output_dir)

        train_ids = [f"S{i:04d}" for i in range(1, cfg.train_n + 1)]
        dataset   = M0035Dataset(sample_dir, train_ids)

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
        self.criterion = torch.nn.MSELoss()
        self.w_vel    = cfg.w_vel
        self.w_stress = cfg.w_stress
        self.w_peeq   = cfg.w_peeq
        self._train_step      = 0
        self._stress_every    = getattr(cfg, "stress_every", 1)
        self._stage1_threshold = getattr(cfg, "w_stress_stage1_threshold", None)
        self._stage2_threshold = getattr(cfg, "w_stress_stage2_threshold", None)
        self._warmup_stage    = 0
        self._cfg_w_peeq      = cfg.w_peeq

        # Load normalization stats and precompute GFDM (fixed mesh → once at startup)
        self._load_stats(cfg)
        first_step = dataset.data[0]
        self._precompute_gfdm(
            first_step["graph"].mesh_pos,
            first_step["graph"].mesh_edge_index,
        )

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

        init_ckpt = getattr(cfg, "init_ckpt_path", None)
        if self.epoch_init == 0 and init_ckpt:
            load_checkpoint(
                to_absolute_path(init_ckpt),
                models=self.model,
                device=self.dist.device,
            )
            if self.dist.rank == 0:
                print(f"[warm-start] loaded model weights from {init_ckpt}")

        # Fresh start: disable stress/peeq loss until vel converges
        if self._stage1_threshold is not None:
            if self.epoch_init == 0:
                self.w_stress = 0.0
                self.w_peeq   = 0.0
            else:
                self._warmup_stage = 2   # resuming: assume fully warmed up

        if self.dist.rank == 0:
            self.writer = SummaryWriter(
                log_dir=to_absolute_path(cfg.tensorboard_log_dir)
            )

        self.input_noise_std = getattr(cfg, "input_noise_std", 0.0)

    # ── stats & GFDM precompute ──────────────────────────────────────────────

    def _load_stats(self, cfg):
        """Load normalization statistics from preprocessing JSON."""
        sample_dir = to_absolute_path(cfg.preprocess_output_dir)
        with open(os.path.join(sample_dir, "node_stats_m0035.json")) as f:
            ns = json.load(f)
        dev = self.dist.device
        self.d_mean = torch.tensor(ns["disp_mean"], dtype=torch.float32).to(dev)  # (3,)
        self.d_std  = torch.tensor(ns["disp_std"],  dtype=torch.float32).to(dev)
        self.v_mean = torch.tensor(ns["vel_mean"],  dtype=torch.float32).to(dev)  # (3,)
        self.v_std  = torch.tensor(ns["vel_std"],   dtype=torch.float32).to(dev)
        # Per-material stress stats for GFDM output normalization
        self.stress_mean_Si     = torch.tensor(ns["stress_mean_Si"],     dtype=torch.float32).to(dev)  # (6,)
        self.stress_std_Si      = torch.tensor(ns["stress_std_Si"],      dtype=torch.float32).to(dev)
        self.stress_mean_Solder = torch.tensor(ns["stress_mean_Solder"], dtype=torch.float32).to(dev)
        self.stress_std_Solder  = torch.tensor(ns["stress_std_Solder"],  dtype=torch.float32).to(dev)
        self.stress_mean_UF     = torch.tensor(ns["stress_mean_UF"],     dtype=torch.float32).to(dev)
        self.stress_std_UF      = torch.tensor(ns["stress_std_UF"],      dtype=torch.float32).to(dev)

    def _precompute_gfdm(self, mesh_pos, mesh_ei):
        """Precompute XtX_inv and weighted displacements for the fixed mesh (once at startup)."""
        dev = self.dist.device
        ref = mesh_pos.to(dev).float()
        N   = ref.shape[0]
        row, col = mesh_ei[0].to(dev), mesh_ei[1].to(dev)

        r_ij  = ref[col] - ref[row]                                        # (E, 3)
        w_ij  = 1.0 / r_ij.pow(2).sum(-1, keepdim=True).clamp(min=1e-12)  # (E, 1)
        wr_ij = w_ij * r_ij                                                # (E, 3)

        r_r   = wr_ij.unsqueeze(-1) * r_ij.unsqueeze(-2)                   # (E, 3, 3)
        row_e = row.view(-1, 1, 1).expand_as(r_r)
        XtX   = torch.zeros(N, 3, 3, device=dev, dtype=torch.float32)
        XtX.scatter_add_(0, row_e, r_r)
        XtX  += 1e-6 * torch.eye(3, device=dev).unsqueeze(0)

        self._xtx_inv  = torch.linalg.inv(XtX)   # (N, 3, 3)
        self._wr_ij    = wr_ij                    # (E, 3)
        self._N_single = N
        self._E_single = int(row.shape[0])

    def _gfdm_stress_elastic(self, wp, ref, mesh_ei, node_E, node_nu, node_CTE, delta_T_node):
        """
        GFDM full 6-component stress tensor with isotropic thermal correction.
        σ = C:(ε_total − α·ΔT·I)  where ε_total = sym(∇u) from GFDM.

        Returns (N, 6): [S11, S22, S33, S12, S13, S23] in physical units (MPa).
        Uses precomputed XtX_inv for efficiency.
        """
        N   = wp.shape[0]
        B   = N // self._N_single
        row, col = mesh_ei

        u     = wp.float() - ref.float()             # displacement field
        du_ij = u[col] - u[row]                       # (B*E, 3)

        wr_ij = self._wr_ij.repeat(B, 1)             # (B*E, 3)
        r_du  = wr_ij.unsqueeze(-1) * du_ij.unsqueeze(-2)  # (B*E, 3, 3)
        row_e = row.view(-1, 1, 1).expand_as(r_du)
        XtY   = torch.zeros(N, 3, 3, device=wp.device, dtype=torch.float32)
        XtY.scatter_add_(0, row_e, r_du)

        xtx_inv = self._xtx_inv.repeat(B, 1, 1)
        dU      = torch.bmm(xtx_inv, XtY).permute(0, 2, 1)  # (N, 3, 3) displacement gradient ∂u_i/∂x_j

        E   = node_E.float()            # (N,)
        nu  = node_nu.float()           # (N,)
        alp = node_CTE.float()          # (N,)
        dT  = delta_T_node.float()      # (N,) ΔT at frame t+1

        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu  = E / (2 * (1 + nu))

        # Mechanical strains = total strains − isotropic thermal strains
        eps_th = alp * dT                       # (N,) scalar per node
        eps11  = dU[:, 0, 0] - eps_th
        eps22  = dU[:, 1, 1] - eps_th
        eps33  = dU[:, 2, 2] - eps_th
        tr_mech = eps11 + eps22 + eps33

        s11 = lam * tr_mech + 2 * mu * eps11
        s22 = lam * tr_mech + 2 * mu * eps22
        s33 = lam * tr_mech + 2 * mu * eps33
        s12 = mu * (dU[:, 0, 1] + dU[:, 1, 0])  # shear: no thermal correction
        s13 = mu * (dU[:, 0, 2] + dU[:, 2, 0])
        s23 = mu * (dU[:, 1, 2] + dU[:, 2, 1])

        return torch.stack([s11, s22, s33, s12, s13, s23], dim=1)  # (N, 6)

    def _normalize_stress_permat(self, stress_phys, node_mat):
        """Normalize physical stress (MPa) to normalized space using per-material stats."""
        dev      = stress_phys.device
        stress_n = torch.zeros_like(stress_phys)
        si_mask  = (node_mat == 0) | (node_mat == 1)
        uf_mask  = node_mat == 3
        if si_mask.any():
            stress_n[si_mask] = (stress_phys[si_mask] - self.stress_mean_Si) / self.stress_std_Si
        if uf_mask.any():
            stress_n[uf_mask] = (stress_phys[uf_mask] - self.stress_mean_UF) / self.stress_std_UF
        return stress_n

    # ── forward / train_step ────────────────────────────────────────────────

    def forward(self, graph, mesh_ef, world_ef):
        # graph.x is (N, 17), fully normalized in preprocessing
        node_x = graph.x.to(self.dist.device).float()

        # Noise on normalized displacement (training only; clean disp used for GFDM below)
        if self.model.training and self.input_noise_std > 0:
            node_x = node_x.clone()
            node_x[:, :3] = (node_x[:, :3]
                             + torch.randn_like(node_x[:, :3]) * self.input_noise_std)

        with autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(node_x, mesh_ef, world_ef, graph)

        pred = pred.float()
        y    = graph.y.to(self.dist.device).float()   # (N, 10)

        loss_vel    = self.criterion(pred[:, :3], y[:, :3])
        loss_stress = torch.zeros((), device=self.dist.device)
        loss_peeq   = torch.zeros((), device=self.dist.device)

        if self.w_stress > 0 and self._train_step % self._stress_every == 0:
            # Direct MSE for all stress components (all materials)
            loss_stress = self.criterion(pred[:, 3:9], y[:, 3:9])

            # PEEQ: solder only
            if self.w_peeq > 0:
                solder_mask = (graph.node_mat == 2).to(self.dist.device)
                if solder_mask.any():
                    loss_peeq = self.criterion(
                        pred[solder_mask, 9:10], y[solder_mask, 9:10]
                    )

        loss = (self.w_vel    * loss_vel
              + self.w_stress * loss_stress
              + self.w_peeq   * loss_peeq)
        return loss, loss_vel, loss_stress, loss_peeq

    def train_step(self, graph, mesh_ef, world_ef, epoch):
        mesh_ef  = mesh_ef.to(self.dist.device)
        world_ef = world_ef.to(self.dist.device)
        self.optimizer.zero_grad()
        self._train_step += 1

        loss, loss_vel, loss_stress, loss_peeq = self.forward(graph, mesh_ef, world_ef)

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
        return loss, loss_vel, loss_stress, loss_peeq


# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config_m0035")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger           = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    trainer = M0035Trainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Training started... [M0035 Multigraph: elastoplastic solder + GFDM elastic]")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.sampler.set_epoch(epoch)
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

            loss, loss_vel, loss_stress, loss_peeq = trainer.train_step(
                batched_graph, batched_mesh_ef, batched_world_ef, epoch
            )
            del batched_graph, batched_mesh_ef, batched_world_ef

            vel_loss_sum += loss_vel.item()
            n_steps      += 1

            if dist.rank == 0 and step_count % 10 == 0:
                trainer.writer.add_scalar("loss/step",        loss.item(),        step_count)
                trainer.writer.add_scalar("loss_vel/step",    loss_vel.item(),    step_count)
                trainer.writer.add_scalar("loss_stress/step", loss_stress.item(), step_count)
                trainer.writer.add_scalar("loss_peeq/step",   loss_peeq.item(),   step_count)

            step_count += 1
            progress_bar.set_postfix(
                loss=f"{loss.item():.3e}",
                vel=f"{loss_vel.item():.3e}",
                s=f"{loss_stress.item():.3e}",
                peeq=f"{loss_peeq.item():.3e}",
                ws=f"{trainer.w_stress:.2f}",
            )

        epoch_avg_vel = vel_loss_sum / max(n_steps, 1)

        # Two-stage stress + peeq warmup
        if trainer._stage1_threshold is not None and trainer._warmup_stage == 0:
            if epoch_avg_vel < trainer._stage1_threshold:
                trainer.w_stress = 0.5 * cfg.w_stress
                trainer.w_peeq   = 0.5 * trainer._cfg_w_peeq
                trainer._warmup_stage = 1
                rank_zero_logger.info(
                    f"[warmup] epoch {epoch+1}: vel_loss={epoch_avg_vel:.3e} "
                    f"→ w_stress=0.5, w_peeq={trainer.w_peeq:.2f}"
                )
        if trainer._stage2_threshold is not None and trainer._warmup_stage == 1:
            if epoch_avg_vel < trainer._stage2_threshold:
                trainer.w_stress = cfg.w_stress
                trainer.w_peeq   = trainer._cfg_w_peeq
                trainer._warmup_stage = 2
                rank_zero_logger.info(
                    f"[warmup] epoch {epoch+1}: vel_loss={epoch_avg_vel:.3e} "
                    f"→ w_stress={cfg.w_stress}, w_peeq={trainer._cfg_w_peeq}"
                )

        torch.cuda.empty_cache()
        rank_zero_logger.info(
            f"epoch: {epoch+1}, loss: {loss:10.3e}, vel: {loss_vel:10.3e}, "
            f"stress: {loss_stress:10.3e}, peeq: {loss_peeq:10.3e}, "
            f"w_s={trainer.w_stress:.2f}, w_p={trainer.w_peeq:.2f}, "
            f"time: {(time.time() - start):.1f}s"
        )
        if dist.rank == 0:
            trainer.writer.add_scalar("loss",        loss.detach().cpu().item(),        epoch)
            trainer.writer.add_scalar("loss_vel",    loss_vel.detach().cpu().item(),    epoch)
            trainer.writer.add_scalar("loss_stress", loss_stress.detach().cpu().item(), epoch)
            trainer.writer.add_scalar("loss_peeq",   loss_peeq.detach().cpu().item(),   epoch)
            trainer.writer.add_scalar("learning_rate",
                                      trainer.optimizer.param_groups[0]["lr"], epoch)

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