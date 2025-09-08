import os
import torch
import numpy as np
import transforms3d
import trimesh

import plotly.graph_objects as go
import pytorch_kinematics as pk
from urdf_parser_py.urdf import Robot, Link

from typing import Optional
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


class HandVisualizer:
    def __init__(self,
                 urdf_path: str,
                 device='cpu'):

        self.device = device

        self.chain = pk.build_chain_from_urdf(
            open(urdf_path, 'r').read()
        ).to(dtype=torch.float, device=device)
        self.robot: Robot = Robot.from_xml_file(urdf_path)

        self.mesh = {}
        for link in self.robot.links:
            if link.visual is None or link.visual.geometry is None:
                continue

            visual = link.visual
            filename = visual.geometry.filename
            file_path = os.path.join(
                os.path.dirname(os.path.dirname(urdf_path)),
                filename.lstrip('./')
            )

            link_mesh = trimesh.load(file_path, process=False)
            vertices = torch.tensor(link_mesh.vertices,
                                    dtype=torch.float32,
                                    device=device)
            faces = torch.tensor(link_mesh.faces,
                                 dtype=torch.long,
                                 device=device)

            if hasattr(visual.geometry, 'scale') and visual.geometry.scale:
                scale = torch.tensor(visual.geometry.scale,
                                     dtype=torch.float32, device=device)
                vertices = vertices * scale

            translation = torch.tensor(getattr(visual.origin, 'xyz', [0, 0, 0]),
                                       dtype=torch.float32, device=device)
            rpy = getattr(visual.origin, 'rpy', [0, 0, 0])
            rotation = torch.tensor(transforms3d.euler.euler2mat(*rpy),
                                    dtype=torch.float32,
                                    device=device)
            vertices = vertices @ rotation.T + translation

            self.mesh[link.name] = {
                "vertices": vertices,
                "faces": faces
            }

        self.link_name_to_index = {
            name: i for i, name in enumerate(self.mesh.keys())
        }

        self.hand_pose: torch.Tensor = None
        self.global_translation = None
        self.global_rotation = None
        self.fk_results = None

        self.object_verts = None
        self.object_faces = None
        self.object_translation = None
        self.object_rotation = None
        self.object_scales = None

    def set_hand_parameters(self, hand_pose: torch.Tensor):
        self.hand_pose = hand_pose.to(self.device)
        batch_size = self.hand_pose.shape[0]

        self.global_translation = self.hand_pose[:, 0:3]

        rot_6d = self.hand_pose[:, 3:9]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            rot_6d)

        joint_angles = self.hand_pose[:, 9:]
        self.fk_results = self.chain.forward_kinematics(joint_angles)

    def load_object_mesh(self, mesh_path: str):
        obj_mesh = trimesh.load(mesh_path, process=False)
        self.object_verts = torch.tensor(
            obj_mesh.vertices, dtype=torch.float32, device=self.device
        )
        self.object_faces = torch.tensor(
            obj_mesh.faces, dtype=torch.long, device=self.device
        )

    def set_object_parameters(self,
                              object_translation: Optional[torch.Tensor] = None,
                              object_rotation: Optional[torch.Tensor] = None,
                              object_scales: Optional[torch.Tensor] = None):
        if object_translation is not None:
            self.object_translation = object_translation.to(self.device)

        if object_rotation is not None:
            self.object_rotation = object_rotation.to(self.device)

        if object_scales is not None:
            self.object_scales = object_scales.to(self.device)

    def get_plotly_data(self,
                        i: int,
                        opacity: float = 0.5,
                        color='lightblue',
                        object_color='lightgreen',
                        use_visual_mesh=True):
        data = []
        for link_name, mesh_data in self.mesh.items():
            link_vertices_local = mesh_data["vertices"]

            transform_mat = self.fk_results[link_name].get_matrix()[i]

            ones = torch.ones((link_vertices_local.shape[0], 1),
                              dtype=torch.float32, device=self.device)
            link_vertices_local_homo = torch.cat(
                [link_vertices_local, ones], dim=1
            )
            link_vertices_world = (
                transform_mat @ link_vertices_local_homo.T).T[:, :3]

            link_vertices_world = link_vertices_world @ self.global_rotation[i].T \
                + self.global_translation[i]

            v = link_vertices_world.detach().cpu().numpy()
            f = mesh_data["faces"].detach().cpu().numpy()

            data.append(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=color,
                opacity=opacity,
                name=link_name
            ))

        if self.object_verts is not None and self.object_faces is not None:
            bsize = self.hand_pose.shape[0]
            if self.object_translation is None:
                self.object_translation = torch.zeros(
                    bsize, 3, device=self.device)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(3, dtype=torch.float,
                                                 device=self.device).unsqueeze(0).repeat(bsize, 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(bsize, device=self.device)

            obj_v = self.object_verts * self.object_scales[i]
            obj_v = (obj_v.unsqueeze(0) @ self.object_rotation[i].T).squeeze(0)
            obj_v = obj_v + self.object_translation[i]
            obj_v = (
                obj_v @ self.global_rotation[i].T) + self.global_translation[i]

            v_obj = obj_v.detach().cpu().numpy()
            f_obj = self.object_faces.detach().cpu().numpy()

            data.append(go.Mesh3d(
                x=v_obj[:, 0], y=v_obj[:, 1], z=v_obj[:, 2],
                i=f_obj[:, 0], j=f_obj[:, 1], k=f_obj[:, 2],
                color=object_color,
                opacity=opacity,
                name='Object'
            ))

        return data

    def get_trimesh_data(self, i: int) -> trimesh.Trimesh:
        combined_mesh = trimesh.Trimesh()

        for link_name, mesh_data in self.mesh.items():
            transform_mat = self.fk_results[link_name].get_matrix()[i]

            vertices_local = mesh_data["vertices"]
            ones = torch.ones((vertices_local.shape[0], 1),
                              dtype=torch.float32, device=self.device)
            v_homo = torch.cat([vertices_local, ones], dim=1)
            v_world = (transform_mat @ v_homo.T).T[:, :3]
            v_world = v_world @ self.global_rotation[i].T + \
                self.global_translation[i]

            v_np = v_world.detach().cpu().numpy()
            f_np = mesh_data["faces"].detach().cpu().numpy()

            link_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
            link_mesh.visual.face_colors = [173, 216, 230, 100]
            combined_mesh = combined_mesh + link_mesh

        if self.object_verts is not None and self.object_faces is not None:
            bsize = self.hand_pose.shape[0]
            if self.object_translation is None:
                self.object_translation = torch.zeros(
                    bsize, 3, device=self.device)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(3, dtype=torch.float,
                                                 device=self.device).unsqueeze(0).repeat(bsize, 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(bsize, device=self.device)

            obj_v = self.object_verts * self.object_scales[i]
            obj_v = (obj_v.unsqueeze(0) @ self.object_rotation[i].T).squeeze(0)
            obj_v = obj_v + self.object_translation[i]
            obj_v = (
                obj_v @ self.global_rotation[i].T) + self.global_translation[i]

            v_np = obj_v.detach().cpu().numpy()
            f_np = self.object_faces.detach().cpu().numpy()
            obj_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
            obj_mesh.visual.face_colors = [144, 238, 144, 100]

            combined_mesh = combined_mesh + obj_mesh

        return combined_mesh
