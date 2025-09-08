from kaolin.metrics.trianglemesh import (
    CUSTOM_index_vertices_by_faces as index_vertices_by_faces,
    compute_sdf
)
import plotly.graph_objects as go
import pytorch3d.ops
import pytorch3d.structures
from urdf_parser_py.urdf import Robot, Box, Sphere, Link
import pytorch_kinematics as pk
from src.utils.rot6d_utils import robust_compute_rotation_matrix_from_ortho6d
import trimesh as tm
import transforms3d
import torch
import numpy as np
import os

from typing import List, Union, Optional, Tuple, Dict


class HandModel:
    def __init__(self,
                 urdf_path: str,
                 contact_candidates: Dict[str, torch.Tensor],
                 n_surface_points: int,
                 device='cpu'
                 ):
        self.device = device
        self.chain = pk.build_chain_from_urdf(
            open(urdf_path).read()).to(dtype=torch.float, device=device)
        self.robot: Robot = Robot.from_xml_file(urdf_path)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        self.mesh: Dict[str, Dict[str, torch.Tensor]] = {}
        areas = {}
        for link in self.robot.links:
            link: Link
            if link.visual is None or link.collision is None:
                continue
            self.mesh[link.name] = {}
            # load collision mesh
            collision = link.collision

            if type(collision.geometry) == Sphere:
                link_mesh = tm.primitives.Sphere(
                    radius=collision.geometry.radius)
                self.mesh[link.name]['radius'] = collision.geometry.radius
            elif type(collision.geometry) == Box:
                # link_mesh = tm.primitives.Box(extents=collision.geometry.size)
                link_mesh = tm.load_mesh(os.path.join(os.path.dirname(
                    urdf_path), 'meshes', 'box.obj'), process=False)
                link_mesh.vertices *= np.array(collision.geometry.size) / 2
            else:
                raise NotImplementedError(
                    f"Unsupported collision geometry: {collision.geometry}")

            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(
                link_mesh.faces, dtype=torch.long, device=device)

            if hasattr(collision.geometry, 'scale') and collision.geometry.scale is None:
                collision.geometry.scale = [1, 1, 1]

            scale = torch.tensor(getattr(collision.geometry, 'scale', [
                                 1, 1, 1]), dtype=torch.float, device=device)
            translation = torch.tensor(getattr(collision.origin, 'xyz', [
                                       0, 0, 0]), dtype=torch.float, device=device)
            rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(collision.origin, 'rpy', [0, 0, 0])),
                                    dtype=torch.float, device=device)
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update({
                'vertices': vertices,
                'faces': faces,
            })
            if 'radius' not in self.mesh[link.name]:
                self.mesh[link.name]['face_verts'] = index_vertices_by_faces(
                    vertices, faces)
            areas[link.name] = tm.Trimesh(
                vertices.cpu().numpy(), faces.cpu().numpy()).area.item()
            # load visual mesh
            visual = link.visual
            filename = os.path.join(os.path.dirname(
                os.path.dirname(urdf_path)), visual.geometry.filename[10:])
            link_mesh = tm.load_mesh(filename)
            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(
                link_mesh.faces, dtype=torch.long, device=device)
            if hasattr(visual.geometry, 'scale') and visual.geometry.scale is None:
                visual.geometry.scale = [1, 1, 1]
            scale = torch.tensor(getattr(visual.geometry, 'scale', [
                                 1, 1, 1]), dtype=torch.float, device=device)
            translation = torch.tensor(
                getattr(visual.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device)
            rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(visual.origin, 'rpy', [0, 0, 0])),
                                    dtype=torch.float, device=device)
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update({
                'visual_vertices': vertices,
                'visual_faces': faces,
            })

            self.mesh[link.name].update({
                'contact_candidates': contact_candidates[link.name],
            })

        self.joints_lower = torch.tensor(
            [joint.limit.lower for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)
        self.joints_upper = torch.tensor(
            [joint.limit.upper for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)

        # sample surface points
        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(
            areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples[list(num_samples.keys())[0]
                    ] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor(
                    [], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(
                0), self.mesh[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(
                dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]['surface_points'] = surface_points

        self.link_name_to_link_index = dict(
            zip([link_name for link_name in self.mesh], range(len(self.mesh))))
        self.surface_points_link_indices = torch.cat([self.link_name_to_link_index[link_name] * torch.ones(
            self.mesh[link_name]['surface_points'].shape[0], dtype=torch.long, device=device) for link_name in self.mesh])

        self.contact_candidates = [
            self.mesh[link_name]['contact_candidates'] for link_name in self.mesh]
        self.global_index_to_link_index = sum(
            [[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)], [])
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=device)
        self.n_contact_candidates = self.contact_candidates.shape[0]

        # build collision mask
        self.adjacency_mask = torch.zeros(
            [len(self.mesh), len(self.mesh)], dtype=torch.bool, device=device)
        for joint in self.robot.joints:
            parent_id = self.link_name_to_link_index[joint.parent]
            child_id = self.link_name_to_link_index[joint.child]
            self.adjacency_mask[parent_id, child_id] = True
            self.adjacency_mask[child_id, parent_id] = True
        self.adjacency_mask[self.link_name_to_link_index['base_link'],
                            self.link_name_to_link_index['link_13.0']] = True
        self.adjacency_mask[self.link_name_to_link_index['link_13.0'],
                            self.link_name_to_link_index['base_link']] = True

        self.hand_pose: torch.Tensor = None  # (B, 3+6+`n_dofs`)
        self.contact_point_indices: torch.Tensor = None  # (B, n_contact)
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None

    def set_parameters(self, hand_pose: torch.Tensor, contact_point_indices: Optional[torch.Tensor] = None):
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9])
        self.current_status = self.chain.forward_kinematics(
            self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(
                batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = self.current_status[link_name].get_matrix().unsqueeze(
                    1).expand(batch_size, n_contact, 4, 4)
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat([self.contact_points, torch.ones(
                batch_size, n_contact, 1, dtype=torch.float, device=self.device)], dim=2)
            self.contact_points = (
                transforms @ self.contact_points.unsqueeze(3))[:, :, :3, 0]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(
                1, 2) + self.global_translation.unsqueeze(1)

    def cal_distance(self, x: torch.Tensor):
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            # (total_batch_size * num_samples, 3)
            x_local = x_local.reshape(-1, 3)
            if 'radius' not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                dis_local = self.mesh[link_name]['radius'] - \
                    x_local.norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def cal_self_distance(self) -> torch.Tensor:
        # get surface points
        x = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            x.append(self.current_status[link_name].transform_points(
                self.mesh[link_name]['surface_points']))
            if 1 < batch_size != x[-1].shape[0]:
                x[-1] = x[-1].expand(batch_size, n_surface_points, 3)
        # (total_batch_size, n_surface_points, 3)
        x = torch.cat(x, dim=-2).to(self.device)
        if len(x.shape) == 2:
            x = x.expand(1, x.shape[0], x.shape[1])
        # cal distance
        dis = []
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            # if len(matrix) != len(x):
            #     matrix = matrix.expand(len(x), 4, 4)
            # x_local = transform_points_inverse(x, matrix)
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            # (total_batch_size * n_surface_points, 3)
            x_local = x_local.reshape(-1, 3)
            if 'radius' in self.mesh[link_name]:
                radius = self.mesh[link_name]['radius']
                # (total_batch_size * n_surface_points,)
                dis_local = radius - (x_local.square().sum(-1) + 1e-8).sqrt()
            else:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = (dis_local + 1e-8).sqrt()
                dis_local = dis_local * (-dis_signs)
            # (total_batch_size, n_surface_points)
            dis_local = dis_local.reshape(x.shape[0], x.shape[1])
            is_adjacent = self.adjacency_mask[self.link_name_to_link_index[link_name],
                                              self.surface_points_link_indices]  # (n_surface_points,)
            dis_local[:, is_adjacent | (
                self.link_name_to_link_index[link_name] == self.surface_points_link_indices)] = -float('inf')
            dis.append(dis_local)
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self) -> torch.Tensor:
        dis = self.cal_self_distance()
        dis[dis <= 0] = 0
        E_spen = dis.sum(-1)
        return E_spen

    def get_contact_candidates_world(self):
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            contact_candidates = self.mesh[link_name]['contact_candidates']
            n_surface_points = contact_candidates.shape[0]

            if n_surface_points > 0:
                points.append(self.current_status[link_name].transform_points(
                    contact_candidates))

            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_surface_points(self):
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            points.append(self.current_status[link_name].transform_points(
                self.mesh[link_name]['surface_points']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_plotly_data(self,
                        i: int,
                        *,
                        opacity: float = 0.5,
                        color='lightblue',
                        with_contact_points: bool = False,
                        with_contact_candidates: bool = False,
                        visual: bool = False):
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['visual_vertices' if visual else 'vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['visual_faces' if visual else 'faces'].detach(
            ).cpu()
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], text=[
                        link_name] * len(v), color=color, opacity=opacity, hovertemplate='%{text}'))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            data.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2],
                                     mode='markers', marker=dict(color='red', size=5)))

        if with_contact_candidates:
            # (n_objects, n_contact_candidates, 3)
            contact_candidates = self.get_contact_candidates_world()[
                i].detach().cpu()

            data.append(go.Scatter3d(x=contact_candidates[:, 0], y=contact_candidates[:, 1], z=contact_candidates[:, 2],
                                     mode='markers', marker=dict(color='blue', size=5)))
        return data

    @property
    def joint_names(self):
        return self.chain.get_joint_parameter_names()

    def forward_kinematics(self, qpos_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        compute forward kinematics

        Args:
        - qpos_dict: dict[str, torch.Tensor[batch_size]], joint_name -> qpos

        Returns:
        - link_translations: dict[str, torch.Tensor[batch_size, 3]], link name -> translation
        - link_rotations: dict[str, torch.Tensor[batch_size, 3, 3]], link name -> rotation
        """

        device = qpos_dict[self.joint_names[0]].device
        batch_size = qpos_dict[self.joint_names[0]].shape[0]
        assert all([qpos_dict[joint_name].shape[0] ==
                   batch_size for joint_name in qpos_dict])
        assert all([qpos_dict[joint_name].device ==
                   device for joint_name in qpos_dict])

        qpos = torch.cat([qpos_dict[joint_name].unsqueeze(1)
                         for joint_name in self.joint_names], dim=1)  # (batch_size, n_dofs)

        ts = self.chain.forward_kinematics(qpos)

        link_translations = {}
        link_rotations = {}

        for link_name in self.mesh:
            t = ts[link_name].get_matrix()  # (batch_size, 4, 4)

            link_translations[link_name] = t[:, :3, 3]
            link_rotations[link_name] = t[:, :3, :3]

        return link_translations, link_rotations
