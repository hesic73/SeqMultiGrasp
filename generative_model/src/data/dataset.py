import torch
import numpy as np
import os
import h5py
import trimesh
from trimesh.sample import sample_surface
from torch.utils.data import Dataset
from typing import List, Dict, Optional


class MultiGraspDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 meshdata_path: str,
                 object_lists: Optional[List[str]] = None,
                 num_surface_points: int = 3000,
                 sample_surface_points_seed: int = 42,
                 sampled_points: int = 1024):

        self.num_surface_points = num_surface_points
        self.sampled_points = sampled_points

        if object_lists is None:
            object_lists = os.listdir(meshdata_path)
        self.object_lists = object_lists
        self.object_index_map = {name: i for i,
                                 name in enumerate(object_lists)}

        object_vertices_list = []
        for object_name in object_lists:
            mesh_path = os.path.join(
                meshdata_path, object_name, f"{object_name}.stl")
            assert os.path.exists(
                mesh_path), f"Mesh file not found: {mesh_path}"
            mesh: trimesh.Trimesh = trimesh.load_mesh(mesh_path)
            points = sample_surface(
                mesh, self.num_surface_points, seed=sample_surface_points_seed)[0]
            object_vertices_list.append(
                torch.tensor(points, dtype=torch.float32))

        self.object_vertices_tensor = torch.stack(
            object_vertices_list)  # [num_objects, num_surface_points, 3]

        with h5py.File(data_path, 'r') as h5py_file:
            self.object_0_indices = np.array([self.object_index_map[name.decode(
                'utf-8')] for name in h5py_file['object_0_name'][:]])
            self.object_1_indices = np.array([self.object_index_map[name.decode(
                'utf-8')] for name in h5py_file['object_1_name'][:]])
            self.object_0_poses = torch.tensor(
                h5py_file['object_0_pose'][:], dtype=torch.float32)
            self.object_1_poses = torch.tensor(
                h5py_file['object_1_pose'][:], dtype=torch.float32)
            self.qpos = torch.tensor(h5py_file['qpos'][:], dtype=torch.float32)

        self._size = len(self.object_0_indices)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        object_0_idx = self.object_0_indices[idx]
        object_1_idx = self.object_1_indices[idx]
        object_0_pose = self.object_0_poses[idx]
        object_1_pose = self.object_1_poses[idx]
        qpos = self.qpos[idx]

        object_0_points_all = self.object_vertices_tensor[object_0_idx]
        object_1_points_all = self.object_vertices_tensor[object_1_idx]

        idx0 = torch.randperm(self.num_surface_points)[:self.sampled_points]
        idx1 = torch.randperm(self.num_surface_points)[:self.sampled_points]

        object_0_sampled = object_0_points_all[idx0]
        object_1_sampled = object_1_points_all[idx1]

        # object_0_sampled = object_0_points_all[:self.sampled_points]
        # object_1_sampled = object_1_points_all[:self.sampled_points]

        return {
            "object_0_pose": object_0_pose,  # => [6]
            "object_1_pose": object_1_pose,  # => [6]
            "qpos": qpos,  # => [16]
            "object_0_points": object_0_sampled,  # => [sampled_points, 3]
            "object_1_points": object_1_sampled,  # => [sampled_points, 3]
        }
