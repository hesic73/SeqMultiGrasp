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

from typing import List, Union, Tuple


class ObjectModel:

    def __init__(self, data_root_path: str, batch_size_each: int, num_samples: int = 2000, device="cuda"):

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.object_mesh_list = None
        self.object_verts_list = None
        self.object_faces_list = None
        self.object_face_verts_list = None

        # we don't need to normalize and rescale the mesh
        self.scale_choice = torch.tensor(
            [1.0], dtype=torch.float, device=self.device)

    def initialize(self, object_code_list: Union[str, List[str]]) -> None:
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list: List[Trimesh] = []
        self.object_min_y = []
        self.object_verts_list = []
        self.object_faces_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        self.object_scale_list = []
        for object_code in object_code_list:
            self.object_scale_tensor.append(self.scale_choice[torch.randint(
                0, self.scale_choice.shape[0], (self.batch_size_each, ), device=self.device)])
            object_mesh = tm.load(os.path.join(
                self.data_root_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False)

            # object_mesh.apply_transform(np.array([
            #     [0, -1, 0, 0],
            #     [0, 0, 1, 0],
            #     [-1, 0, 0, 0],
            #     [0, 0, 0, 1],
            # ]))
            self.object_mesh_list.append(object_mesh)
            min_y = np.min(object_mesh.vertices[:, 1])
            self.object_min_y.append(min_y)
            self.object_verts_list.append(torch.tensor(
                self.object_mesh_list[-1].vertices, dtype=torch.float32, device=self.device))
            self.object_faces_list.append(torch.tensor(
                self.object_mesh_list[-1].faces).long().to(self.device))
            self.object_face_verts_list.append(index_vertices_by_faces(
                self.object_verts_list[-1], self.object_faces_list[-1]))
            if self.num_samples != 0:
                vertices = torch.tensor(
                    self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(
                    self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
                mesh = pytorch3d.structures.Meshes(
                    vertices.unsqueeze(0), faces.unsqueeze(0))
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                    mesh, num_samples=100 * self.num_samples)
                surface_points = pytorch3d.ops.sample_farthest_points(
                    dense_point_cloud, K=self.num_samples)[0][0]
                surface_points.to(dtype=float, device=self.device)
                self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)
        self.object_min_y = torch.tensor(
            self.object_min_y, dtype=torch.float, device=self.device)
        self.object_min_y = torch.repeat_interleave(
            self.object_min_y, self.batch_size_each)  # (n_objects * batch_size_each)

        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(
                self.batch_size_each, dim=0)  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x: torch.Tensor, with_closest_points: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Compute signed distances and normals from input points to the object's surface.

        Args:
            x (torch.Tensor): Input points of shape (n_objects * batch_size_each, n_points, 3).
            with_closest_points (bool, optional): If True, also return closest surface points.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                - (distance, normals) if with_closest_points is False
                - (distance, normals, closest_points) if True

            distance (torch.Tensor): (n_objects * batch_size_each, n_points) signed distances.
            normals (torch.Tensor): (n_objects * batch_size_each, n_points, 3) surface normals.
            closest_points (torch.Tensor, optional): (n_objects * batch_size_each, n_points, 3) closest surface points.
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)  # (n_object, batch_size_each * n_points, 3)

        for i in range(len(self.object_mesh_list)):
            face_verts = self.object_face_verts_list[i]
            dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)
            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)
        if with_closest_points:
            closest_points = (torch.stack(closest_points) *
                              scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals

    def get_plotly_data(self, i: int, color='lightgreen', opacity=0.5, pose=None):
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index,
                                               i % self.batch_size_each].detach().cpu().numpy()
        mesh: Trimesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=mesh.faces[:, 0],
                         j=mesh.faces[:, 1], k=mesh.faces[:, 2], color=color, opacity=opacity)
        return [data]
