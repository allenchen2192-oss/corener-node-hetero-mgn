# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import hydra
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from deforming_plate_dataset import DeformingPlateDataset
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.utils import load_checkpoint
from helpers import add_world_edges


def extract_surface_triangles(tets):
    # tets: (N_tet, 4) array of indices
    # Returns: (N_surface_tri, 3) array of triangle indices
    faces = np.concatenate(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ],
        axis=0,
    )
    # Sort each face so that duplicates can be found
    faces = np.sort(faces, axis=1)
    # Find unique faces and their counts
    faces_tuple = [tuple(face) for face in faces]
    from collections import Counter

    face_counts = Counter(faces_tuple)
    # Surface faces appear only once
    surface_faces = np.array(
        [face for face, count in face_counts.items() if count == 1]
    )
    return surface_faces


class MGNRollout:
    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        self.num_test_time_steps = cfg.num_test_time_steps
        self.frame_skip = cfg.frame_skip
        self.logger = logger

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = DeformingPlateDataset(
            name="deforming_plate_test",
            data_dir=to_absolute_path(cfg.data_dir),
            split="test",
            num_samples=cfg.num_test_samples,
            num_steps=cfg.num_test_time_steps,
        )

        # instantiate dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda batch: batch[0],
        )

        # instantiate the model
        self.model = HybridMeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        if cfg.jit:
            self.model = torch.compile(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )

    @torch.inference_mode()
    def predict(self):
        # 增加 self.pred_stress 和 self.exact_stress 來儲存應力
        self.pred, self.exact, self.faces, self.graphs = [], [], [], []
        self.pred_stress, self.exact_stress = [], [] 

        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }
        for i, (
            graph,
            cells,
            moving_points_mask,
            object_points_mask,
            clamped_points_mask,
        ) in enumerate(self.dataloader):
            graph = graph.to(self.device)
            moving_points_mask = moving_points_mask.to(self.device)
            object_points_mask = object_points_mask.to(self.device)
            clamped_points_mask = clamped_points_mask.to(self.device)

            # --- 計算真實數值 (Ground Truth) ---
            # denormalize data
            exact_velocity_denormalized = self.dataset.denormalize(
                graph.y[:, 0:3],
                stats["velocity_mean"],
                stats["velocity_std"],
            )
            exact_next_world_pos = exact_velocity_denormalized + graph.world_pos[:, 0:3]
            # --------------------------------

            # inference step
            if i % (self.num_test_time_steps - 1) != 0:
                graph.world_pos = self.pred[i - 1][:, 0:3]
            graph, mesh_edge_features, world_edge_features = add_world_edges(graph)
            pred_i = self.model(
                graph.x, mesh_edge_features, world_edge_features, graph
            )  # predict

            # --- 分離速度與應力 ---
            pred_velocity = pred_i[:, 0:3]
            if pred_i.shape[1] > 3:
                # 假設模型輸出的第 4 個特徵之後是應力
                pred_stress_i = pred_i[:, 3:] 
            else:
                pred_stress_i = torch.zeros((pred_i.shape[0], 1)).to(self.device)
            
            # denormalize prediction
            pred_velocity_denormalized = self.dataset.denormalize(
                pred_velocity,
                stats["velocity_mean"],
                stats["velocity_std"],
            )

            # --- 邊界條件與積分 ---
            # do not update the "wall_boundary" & "outflow" nodes
            moving_points_mask_3d = torch.cat(
                (moving_points_mask, moving_points_mask, moving_points_mask), dim=-1
            ).to(self.device)
            pred_velocity_denormalized = torch.where(
                moving_points_mask_3d,
                pred_velocity_denormalized,
                torch.zeros_like(pred_velocity_denormalized),
            )

            # integration
            pred_world_pos_denormalized = (
                pred_velocity_denormalized.squeeze(0) + graph.world_pos[:, 0:3]
            )  # Note that the world_pos is not normalized
            
            # assign boundary conditions to the object points
            pred_world_pos_denormalized = torch.where(
                object_points_mask, exact_next_world_pos, pred_world_pos_denormalized
            )
            pred_world_pos_denormalized = torch.where(
                clamped_points_mask, exact_next_world_pos, pred_world_pos_denormalized
            )

            # --- 儲存結果 ---
            self.pred.append(pred_world_pos_denormalized.squeeze(0))
            self.exact.append(exact_next_world_pos.squeeze(0))
            
            # 將應力也存入 list
            self.pred_stress.append(pred_stress_i.squeeze(0))
            
            # 如果 ground truth (graph.y) 也有應力標籤，可以在這裡提取
            if graph.y.shape[1] > 3:
                self.exact_stress.append(graph.y[:, 3:])
            else:
                self.exact_stress.append(torch.zeros_like(pred_stress_i.squeeze(0)))

            self.faces.append(torch.squeeze(cells))
            self.graphs.append(graph)

        self.pred = [pred.cpu() for pred in self.pred]
        self.exact = [exact.cpu() for exact in self.exact]
        self.pred_stress = [stress.cpu() for stress in self.pred_stress]
        self.exact_stress = [stress.cpu() for stress in self.exact_stress]
        self.graphs = [graph.cpu() for graph in self.graphs]
        self.faces = [face.cpu().numpy() for face in self.faces]

    var_identifier = {"ux": 0, "uy": 1, "uz": 2, "disp_mag": -1}

    def get_raw_data(self, idx):
        # Support for displacement magnitude
        if idx == -1:  # -1 will be used for disp_mag
            self.pred_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.pred]
            self.exact_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.exact]
        else:
            self.pred_i = [var[:, idx] for var in self.pred]
            self.exact_i = [var[:, idx] for var in self.exact]
        return self.graphs, self.faces, self.pred_i, self.exact_i

    def init_animation(self, idx):
        # Support for displacement magnitude
        if idx == -1:  # -1 will be used for disp_mag
            self.pred_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.pred]
            self.exact_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.exact]
        else:
            self.pred_i = [var[:, idx] for var in self.pred]
            self.exact_i = [var[:, idx] for var in self.exact]

        # fig configs
        plt.rcParams["image.cmap"] = "inferno"
        self.fig, self.ax = plt.subplots(1, 2, figsize=(16, 9))

        # Set background color to black
        self.fig.set_facecolor("black")
        self.ax[0].set_facecolor("black")
        self.ax[1].set_facecolor("black")

        # make animations dir
        if not os.path.exists("./animations"):
            os.makedirs("./animations")

    def animate(self, num):
        num *= self.frame_skip
        graph = self.graphs[num]
        y_star = self.pred_i[num].numpy()
        y_exact = self.exact_i[num].numpy()
        cells = self.faces[num]
        surface_tris = extract_surface_triangles(cells)

        # For predicted mesh
        mesh_pos_pred = self.pred[num][:, 0:3].numpy()
        # For ground truth mesh
        mesh_pos_exact = self.exact[num][:, 0:3].numpy()

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.ax[0].cla()
        self.ax[0] = self.fig.add_subplot(1, 2, 1, projection="3d")
        tris = mesh_pos_pred[surface_tris]
        # Use a solid metallic color (e.g., 'silver')
        col = Poly3DCollection(tris, facecolor="silver", edgecolor="k", linewidths=0.05)
        self.ax[0].add_collection3d(col)
        self.ax[0].auto_scale_xyz(
            mesh_pos_pred[:, 0], mesh_pos_pred[:, 1], mesh_pos_pred[:, 2]
        )
        self.ax[0].set_title("Predicted Deformed Mesh", color="white")

        self.ax[1].cla()
        self.ax[1] = self.fig.add_subplot(1, 2, 2, projection="3d")
        tris = mesh_pos_exact[surface_tris]
        col = Poly3DCollection(tris, facecolor="silver", edgecolor="k", linewidths=0.05)
        self.ax[1].add_collection3d(col)
        self.ax[1].auto_scale_xyz(
            mesh_pos_exact[:, 0], mesh_pos_exact[:, 1], mesh_pos_exact[:, 2]
        )
        self.ax[1].set_title("True Deformed Mesh", color="white")

        # Adjust subplots to minimize empty space
        self.ax[0].set_aspect("auto", adjustable="box")
        self.ax[1].set_aspect("auto", adjustable="box")
        self.ax[0].autoscale(enable=True, tight=True)
        self.ax[1].autoscale(enable=True, tight=True)
        self.fig.subplots_adjust(
            left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.2, hspace=0.05
        )

        # After plotting both meshes, set axis limits for predicted to match exact from the first frame
        if not hasattr(self, "xlim"):
            self.xlim = self.ax[1].get_xlim()
            self.ylim = self.ax[1].get_ylim()
            self.zlim = self.ax[1].get_zlim()
        self.ax[0].set_xlim(self.xlim)
        self.ax[0].set_ylim(self.ylim)
        self.ax[0].set_zlim(self.zlim)
        self.ax[1].set_xlim(self.xlim)
        self.ax[1].set_ylim(self.ylim)
        self.ax[1].set_zlim(self.zlim)

        return self.fig

    def export_to_paraview(self, output_dir="paraview_output"):
        import meshio
        import os
        import numpy as np

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.logger.info(f"Exporting files to {output_dir} for ParaView...")

        # 每個 case 包含的步數
        steps_per_case = self.num_test_time_steps - 1 

        for i in range(len(self.pred)):
            case_idx = i // steps_per_case
            step_idx = i % steps_per_case

            # 網格連接關係
            tets = self.faces[i]
            cells = [("tetra", tets)]

            # 取出當前的預測與真實位置
            pred_points = self.pred[i][:, 0:3].numpy()
            exact_points = self.exact[i][:, 0:3].numpy()
            
            # 取出當前 Case 的「第 0 步」作為未變形的初始參考座標
            initial_points = self.exact[case_idx * steps_per_case][:, 0:3].numpy()

            # 計算位移大小 (當前位置 - 初始位置 的直線距離)
            pred_disp_mag = np.linalg.norm(pred_points - initial_points, axis=1)
            exact_disp_mag = np.linalg.norm(exact_points - initial_points, axis=1)

            # 其他數據
            pred_stress_np = self.pred_stress[i].numpy()
            exact_stress_np = self.exact_stress[i].numpy()
            error_np = np.linalg.norm(pred_points - exact_points, axis=1)

            # ==========================================
            # 1. 輸出預測結果 (加入 Displacement_Mag)
            # ==========================================
            pred_point_data = {
                "Stress": pred_stress_np,       
                "Position_Error": error_np,
                "Displacement_Mag": pred_disp_mag  # <--- 新增的位移大小
            }
            mesh_pred = meshio.Mesh(points=pred_points, cells=cells, point_data=pred_point_data)
            pred_filename = os.path.join(output_dir, f"case_{case_idx:03d}_step_{step_idx:03d}_pred.vtu")
            mesh_pred.write(pred_filename)

            # ==========================================
            # 2. 輸出真實結果 (加入 Displacement_Mag)
            # ==========================================
            exact_point_data = {
                "Stress": exact_stress_np,
                "Displacement_Mag": exact_disp_mag  # <--- 新增的位移大小
            }
            mesh_exact = meshio.Mesh(points=exact_points, cells=cells, point_data=exact_point_data)
            exact_filename = os.path.join(output_dir, f"case_{case_idx:03d}_step_{step_idx:03d}_exact.vtu")
            mesh_exact.write(exact_filename)
            
        self.logger.info("ParaView export complete! Prediction and Exact files are separated.")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()
    logger.info("Rollout started...")
    
    rollout = MGNRollout(cfg, logger)
    rollout.predict()

    # 輸出 VTU 檔案至 paraview_output 資料夾
    rollout.export_to_paraview(output_dir="paraview_output_3Dbeam")

    # 保留原本的 GIF 動畫生成
    idx = [rollout.var_identifier.get(k, -1) for k in cfg.viz_vars]
    for k, i in zip(cfg.viz_vars, idx):
        rollout.init_animation(i)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate,
            frames=len(rollout.graphs) // cfg.frame_skip,
            interval=cfg.frame_interval,
        )
        ani.save(f"animations/animation_{k}.gif")
        logger.info(f"Created animation for {k}")


if __name__ == "__main__":
    main()