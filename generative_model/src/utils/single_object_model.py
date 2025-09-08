"""
NOTE: set_global_transformation might have bugs!!! 

(This function is not used in the codebase, but I'm keeping it here.)

"""

import os
import trimesh as tm
from trimesh import Trimesh
import plotly.graph_objects as go
import torch
import pytorch3d.structures
import pytorch3d.ops
import numpy as np

from kaolin.metrics.trianglemesh import (
    CUSTOM_index_vertices_by_faces as index_vertices_by_faces,
    compute_sdf
)

from typing import Union, Tuple


class SingleObjectModel:
    def __init__(self, data_root_path: str, batch_size: int, num_samples: int = 2000, device="cuda"):
        self.device = device
        self.batch_size = batch_size
        self.data_root_path = data_root_path
        self.num_samples = num_samples
        self.object_code = None

        self.object_mesh: Trimesh = None
        self.object_verts = None
        self.object_faces = None
        self.object_face_verts = None
        self.surface_points_tensor = None

        self.scale_choice = torch.tensor(
            [1.0], dtype=torch.float, device=self.device)
        self.scale_factor = None 

        self.global_translation = torch.zeros(
            (batch_size, 3), dtype=torch.float, device=self.device)
        self.global_rotation = torch.eye(
            3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

    def initialize(self, object_code: str,) -> None:

        self.object_code = object_code

        object_mesh = tm.load(
            os.path.join(self.data_root_path, self.object_code,
                         "coacd", "decomposed.obj"),
            force="mesh",
            process=False
        )

        self.object_mesh = object_mesh

        self.object_min_y = np.min(object_mesh.vertices[:, 1])
        self.object_min_y = torch.tensor(
            self.object_min_y, dtype=torch.float32, device=self.device)
        self.object_min_y = self.object_min_y.repeat(self.batch_size)

        self.object_verts = torch.tensor(
            self.object_mesh.vertices, dtype=torch.float32, device=self.device
        )
        self.object_faces = torch.tensor(
            self.object_mesh.faces, dtype=torch.long, device=self.device
        )
        self.object_face_verts = index_vertices_by_faces(
            self.object_verts, self.object_faces
        )

        self.scale_factor = self.scale_choice[
            torch.randint(
                0,
                self.scale_choice.shape[0],
                (self.batch_size, ),
                device=self.device
            )
        ]  # shape: (batch_size,)

        if self.num_samples != 0:
            vertices = self.object_verts
            faces = self.object_faces.to(dtype=torch.float)
            mesh = pytorch3d.structures.Meshes(
                vertices.unsqueeze(0), faces.unsqueeze(0)
            )
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=100 * self.num_samples
            )
            surface_points = pytorch3d.ops.sample_farthest_points(
                dense_point_cloud, K=self.num_samples
            )[0][0]
            surface_points = surface_points.to(
                dtype=torch.float, device=self.device)
            self.surface_points_tensor = surface_points.unsqueeze(0).repeat_interleave(
                self.batch_size, dim=0
            )

    def set_global_transformation(self, translation: torch.Tensor, rotation: torch.Tensor) -> None:
        if translation.shape != (self.batch_size, 3):
            raise ValueError(
                f"Translation tensor must have shape ({self.batch_size}, 3), got {translation.shape}")

        if rotation.shape != (self.batch_size, 3, 3):
            raise ValueError(
                f"Rotation tensor must have shape ({self.batch_size}, 3, 3), got {rotation.shape}")

        # Update global translation and rotation
        self.global_translation = translation.to(self.device)
        self.global_rotation = rotation.to(self.device)

    def get_surface_points_global(self) -> torch.Tensor:

        if self.surface_points_tensor is None:
            return None

        local_points = self.surface_points_tensor  # (B, N, 3)

        scaled_local_points = local_points * self.scale_factor.view(-1, 1, 1)

        global_points = scaled_local_points @ self.global_rotation.transpose(
            1, 2) + self.global_translation.unsqueeze(1)

        return global_points  # shape (batch_size, num_samples, 3)

    def cal_distance(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute signed distances and normals from input points to the object's surface.

        Args:
            x (torch.Tensor): Input points of shape (batch_size, n_points, 3).

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - (distance, normals) if with_closest_points is False
                - (distance, normals, closest_points) if True

            distance (torch.Tensor): (batch_size, n_points) signed distances.
            normals (torch.Tensor): (batch_size, n_points, 3) surface normals.
            closest_points (torch.Tensor, optional): (batch_size, n_points, 3) closest surface points.
        """
        # Shapes
        batch_size, n_points, _ = x.shape
        assert batch_size == self.batch_size, "Input x's batch_size != self.batch_size."

        x_local = (x - self.global_translation.unsqueeze(1)
                   )@self.global_rotation

        x_scaled = x_local / self.scale_factor.view(-1, 1, 1)

        dis, dis_signs, normal_local, _ = compute_sdf(
            x_scaled.reshape(-1, 3),
            self.object_face_verts
        )

        # reshape to (batch_size, n_points)
        dis = dis.reshape(batch_size, n_points)
        dis_signs = dis_signs.reshape(batch_size, n_points)
        normal_local = normal_local.reshape(batch_size, n_points, 3)

        dis_sqrt = torch.sqrt(dis + 1e-8)
        signed_distance = dis_sqrt * (-dis_signs)

        normals_local = normal_local * dis_signs.unsqueeze(2)

        signed_distance = signed_distance * self.scale_factor.view(-1, 1)

        normals = torch.matmul(
            normals_local, self.global_rotation.transpose(1, 2))

        return signed_distance, normals

    def get_plotly_data(
        self,
        i: int = 0,
        color='lightgreen',
        opacity=0.5,
        pose=None
    ):

        if (self.scale_factor is not None) and (0 <= i < self.batch_size):
            scale_i = self.scale_factor[i].item()
        else:
            scale_i = 1.0

        vertices_local = self.object_mesh.vertices * scale_i

        R_i = self.global_rotation[i].detach().cpu().numpy()
        T_i = self.global_translation[i].detach().cpu().numpy()

        vertices_global = vertices_local @ R_i.T
        vertices_global += T_i

        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices_global = vertices_global @ pose[:3, :3].T + pose[:3, 3]

        data = go.Mesh3d(
            x=vertices_global[:, 0],
            y=vertices_global[:, 1],
            z=vertices_global[:, 2],
            i=self.object_mesh.faces[:, 0],
            j=self.object_mesh.faces[:, 1],
            k=self.object_mesh.faces[:, 2],
            color=color,
            opacity=opacity
        )
        return [data]
