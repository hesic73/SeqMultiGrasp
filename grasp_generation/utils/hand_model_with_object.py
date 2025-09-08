from kaolin.metrics.trianglemesh import (
    CUSTOM_index_vertices_by_faces as index_vertices_by_faces,
    compute_sdf
)

import plotly.graph_objects as go
import pytorch3d.ops
import pytorch3d.structures
from urdf_parser_py.urdf import Robot, Box, Sphere, Link
import pytorch_kinematics as pk
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import trimesh as tm
import transforms3d
import torch
import numpy as np
import os
import trimesh

from typing import List, Union, Optional, Tuple, Dict

from src.consts import MESHDATA_PATH


class HandModelWithObject:
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
                # Example: load a box mesh and scale it
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

        # sample surface points from hand links
        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(
            areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        # Adjust rounding errors
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
            surface_points = surface_points.to(
                dtype=torch.float, device=device)
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
        # Add any special adjacency if needed
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

        self.object_verts = None
        self.object_faces = None
        self.object_face_verts = None
        self.object_surface_points = None

        # Pose per batch
        self.object_translation = None  # (B, 3)
        self.object_rotation = None     # (B, 3, 3)

        self.object_scales = None

    def initialize_object(self,
                          object_code: str,
                          object_scales: Optional[torch.Tensor] = None,
                          num_samples: int = 2000
                          ):

        device = self.device

        object_mesh_path = os.path.join(
            MESHDATA_PATH, object_code, "coacd", "decomposed.obj")

        object_mesh = trimesh.load(
            object_mesh_path, force="mesh", process=False)

        self.object_verts = torch.tensor(
            object_mesh.vertices, dtype=torch.float32, device=device)
        self.object_faces = torch.tensor(
            object_mesh.faces, dtype=torch.long, device=device)
        self.object_face_verts = index_vertices_by_faces(
            self.object_verts, self.object_faces
        )

        if object_scales is None:
            self.object_scales = None
        else:
            self.object_scales = object_scales.to(device).float()

        if num_samples > 0:
            mesh = pytorch3d.structures.Meshes(
                self.object_verts.unsqueeze(0), self.object_faces.unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
                mesh, num_samples=100 * num_samples)
            object_surface_points = pytorch3d.ops.sample_farthest_points(
                dense_point_cloud, K=num_samples)[0][0]
            self.object_surface_points = object_surface_points.to(
                dtype=torch.float, device=device)
        else:
            self.object_surface_points = torch.tensor(
                [], dtype=torch.float, device=device).reshape(0, 3)

    def set_parameters(self,
                       hand_pose: torch.Tensor,
                       contact_point_indices: Optional[torch.Tensor] = None,
                       object_translation: Optional[torch.Tensor] = None,
                       object_rotation: Optional[torch.Tensor] = None):
        """
        Set the hand and object parameters for the entire batch.

        Args:
            hand_pose (torch.Tensor): shape (B, 3 + 6 + n_dofs)
            contact_point_indices (torch.Tensor, optional): (B, n_contact)
            object_translation (torch.Tensor, optional): (B, 3)
            object_rotation (torch.Tensor, optional): (B, 3, 3)
            object_scales (torch.Tensor, optional): (B,) uniform scale factors.
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9])
        self.current_status = self.chain.forward_kinematics(
            self.hand_pose[:, 9:])

        if object_translation is not None:
            self.object_translation = object_translation.to(self.device)
        if object_rotation is not None:
            self.object_rotation = object_rotation.to(self.device)

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
        """
        Compute the signed distance for query points x wrt the hand + object.

        Args:
            x (torch.Tensor): (B, N, 3)

        Returns:
            dis (torch.Tensor): (B, N), the signed distances (max-of-min SDF).
        """
        batch_size, n_points, _ = x.shape

        x_hand = (x - self.global_translation.unsqueeze(1)
                  ) @ self.global_rotation

        dis_list = []

        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (
                x_hand - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)

            if 'radius' not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8) * (-dis_signs)
            else:
                radius = self.mesh[link_name]['radius']
                dis_local = radius - x_local.norm(dim=1)

            dis_list.append(dis_local.reshape(batch_size, n_points))

        if self.object_verts is not None and self.object_faces is not None:
            if self.object_translation is None:
                self.object_translation = torch.zeros(
                    batch_size, 3, device=self.device)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(
                    3, dtype=torch.float, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(
                    batch_size, device=self.device)

            x_object_local = (x_hand - self.object_translation.unsqueeze(1)) \
                @ self.object_rotation
            x_object_local_scaled = x_object_local / \
                self.object_scales.view(-1, 1, 1)

            x_object_local_scaled = x_object_local_scaled.reshape(-1, 3)
            face_verts = self.object_face_verts
            dis_local, dis_signs, _, _ = compute_sdf(
                x_object_local_scaled, face_verts)
            dis_local = torch.sqrt(dis_local + 1e-8) * (-dis_signs)
            dis_local = dis_local.reshape(batch_size, n_points)

            dis_local = dis_local * self.object_scales.view(-1, 1)
            dis_list.append(dis_local)

        if len(dis_list) == 0:
            return torch.zeros(batch_size, n_points, device=self.device)
        dis = torch.max(torch.stack(dis_list, dim=0), dim=0)[0]
        return dis

    def cal_self_distance(self) -> torch.Tensor:
        """
        Compute the SDF for the hand + object surface points w.r.t. the hand + object,
        ignoring adjacency for self-penetration checks.

        Returns:
            dis (torch.Tensor): (B, total_points)
        """
        batch_size = self.global_translation.shape[0]
        hand_surface_points = []
        for link_name in self.mesh:
            pts = self.mesh[link_name]['surface_points']
            if pts.shape[0] == 0:
                continue
            v = self.current_status[link_name].transform_points(pts)
            if 1 < batch_size != v.shape[0]:
                v = v.expand(batch_size, pts.shape[0], 3)
            hand_surface_points.append(v)

        if len(hand_surface_points) > 0:
            hand_surface_points = torch.cat(
                hand_surface_points, dim=-2).to(self.device)
        else:
            hand_surface_points = torch.empty(
                batch_size, 0, 3, device=self.device)

        if self.object_surface_points is not None and self.object_surface_points.shape[0] > 0:
            if self.object_translation is None:
                self.object_translation = torch.zeros(
                    batch_size, 3, device=self.device)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(
                    3, dtype=torch.float, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(
                    batch_size, device=self.device)

            obj_pts = self.object_surface_points.unsqueeze(0)
            obj_pts = obj_pts * self.object_scales.view(-1, 1, 1)
            obj_pts = obj_pts @ self.object_rotation.transpose(1, 2)
            obj_pts = obj_pts + self.object_translation.unsqueeze(1)
        else:
            obj_pts = torch.empty(batch_size, 0, 3, device=self.device)

        x = torch.cat([hand_surface_points, obj_pts], dim=1)
        if x.dim() == 2:  # batch=1 edge case
            x = x.unsqueeze(0)

        dis_list = []
        total_points = x.shape[1]

        # distance from each link
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)
            if 'radius' in self.mesh[link_name]:
                radius = self.mesh[link_name]['radius']
                dis_local = radius - x_local.norm(dim=1)
            else:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = (dis_local + 1e-8).sqrt() * (-dis_signs)
            dis_local = dis_local.reshape(batch_size, total_points)

            # adjacency mask for hand points
            n_hand_points = hand_surface_points.shape[1]
            if total_points > 0 and n_hand_points > 0:
                surface_points_link_indices = self.surface_points_link_indices
                is_adjacent = self.adjacency_mask[
                    self.link_name_to_link_index[link_name],
                    surface_points_link_indices
                ] if surface_points_link_indices.shape[0] > 0 else torch.zeros(
                    n_hand_points, dtype=torch.bool, device=self.device)

                is_adjacent = is_adjacent | (
                    self.link_name_to_link_index[link_name] == surface_points_link_indices)

                # Pad for object points
                is_adjacent_padded = torch.cat([
                    is_adjacent,
                    torch.zeros(total_points - n_hand_points,
                                dtype=torch.bool, device=self.device)
                ])
                dis_local[:, is_adjacent_padded] = -float('inf')

            dis_list.append(dis_local)

        # distance from the scaled object
        if self.object_verts is not None and self.object_faces is not None:
            if self.object_translation is None:
                self.object_translation = torch.zeros(
                    batch_size, 3, device=self.device)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(
                    3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(
                    batch_size, device=self.device)

            x_object_local = (x - self.object_translation.unsqueeze(1)) \
                @ self.object_rotation
            # scale down
            x_object_local_scaled = x_object_local / \
                self.object_scales.view(batch_size, 1, 1)
            x_object_local_scaled = x_object_local_scaled.reshape(-1, 3)
            face_verts = self.object_face_verts
            dis_local, dis_signs, _, _ = compute_sdf(
                x_object_local_scaled, face_verts)
            dis_local = (dis_local + 1e-8).sqrt() * (-dis_signs)
            dis_local = dis_local.reshape(batch_size, total_points)
            # scale back
            dis_local = dis_local * self.object_scales.view(batch_size, 1)

            dis_list.append(dis_local)

        if len(dis_list) == 0:
            # no surfaces
            return torch.zeros(batch_size, total_points, device=self.device)

        dis = torch.max(torch.stack(dis_list, dim=0), dim=0)[0]
        return dis

    def self_penetration(self) -> torch.Tensor:
        dis = self.cal_self_distance()
        dis[dis <= 0] = 0
        E_spen = dis.sum(-1)
        return E_spen

    def get_contact_candidates_world(self):
        """
        Return the contact candidates in the world frame (B, #candidates, 3).
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            contact_candidates = self.mesh[link_name]['contact_candidates']
            n_surface_points = contact_candidates.shape[0]

            if n_surface_points > 0:
                v = self.current_status[link_name].transform_points(
                    contact_candidates)
                if 1 < batch_size != v.shape[0]:
                    v = v.expand(batch_size, n_surface_points, 3)
                points.append(v)

        if len(points) == 0:
            return torch.empty(batch_size, 0, 3, device=self.device)

        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(
            1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_surface_points(self):
        """
        Return all hand + object surface points in the *global* frame.
        """
        batch_size = self.global_translation.shape[0]

        # Hand points
        points = []
        for link_name in self.mesh:
            spts = self.mesh[link_name]['surface_points']
            if spts.shape[0] == 0:
                continue
            v = self.current_status[link_name].transform_points(spts)
            if batch_size != v.shape[0]:
                v = v.expand(batch_size, spts.shape[0], 3)
            points.append(v)

        if len(points) > 0:
            points = torch.cat(points, dim=-2).to(self.device)
        else:
            points = torch.empty(batch_size, 0, 3, device=self.device)

        # Object points
        if self.object_surface_points is not None and self.object_surface_points.shape[0] > 0:
            if self.object_translation is None:
                self.object_translation = torch.zeros(
                    batch_size, 3, device=self.device)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(
                    3, dtype=torch.float, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(batch_size, device=self.device)

            obj_pts = self.object_surface_points.unsqueeze(0)
            # scale
            obj_pts = obj_pts * self.object_scales.view(batch_size, 1, 1)
            # rotate
            obj_pts = obj_pts @ self.object_rotation.transpose(1, 2)
            # translate
            obj_pts = obj_pts + self.object_translation.unsqueeze(1)
        else:
            obj_pts = torch.empty(batch_size, 0, 3, device=self.device)

        # Concatenate
        x = torch.cat([points, obj_pts], dim=1)
        # Now apply the global rotation + translation
        x = x @ self.global_rotation.transpose(1, 2) + \
            self.global_translation.unsqueeze(1)
        return x

    def get_plotly_data(self,
                        i: int,
                        *,
                        opacity: float = 0.5,
                        color='lightblue',
                        with_contact_points: bool = False,
                        with_contact_candidates: bool = False,
                        visual: bool = False,
                        object_color='lightgreen'):
        data = []
        # Plot hand links
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['visual_vertices' if visual else 'vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['visual_faces' if visual else 'faces'].detach(
            ).cpu()
            data.append(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                text=[link_name] * len(v),
                color=color,
                opacity=opacity,
                hovertemplate='%{text}'
            ))

        # Plot object mesh
        if self.object_verts is not None and self.object_faces is not None:
            # If no transforms specified, default
            if self.object_translation is None:
                self.object_translation = torch.zeros_like(
                    self.global_translation)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(3, dtype=torch.float, device=self.device).unsqueeze(
                    0).repeat(self.global_translation.shape[0], 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(
                    self.global_translation.shape[0], device=self.device
                )

            # 1) Scale
            v_obj = self.object_verts * self.object_scales[i]
            # 2) Rotate by object_rotation[i]
            v_obj = (v_obj.unsqueeze(0) @ self.object_rotation[i].T).squeeze(0)
            # 3) Translate by object_translation[i]
            v_obj = v_obj + self.object_translation[i]
            # 4) Then global transform
            v_obj = (
                v_obj @ self.global_rotation[i].T) + self.global_translation[i]

            v_obj = v_obj.detach().cpu()
            f_obj = self.object_faces.detach().cpu()
            data.append(go.Mesh3d(
                x=v_obj[:, 0],
                y=v_obj[:, 1],
                z=v_obj[:, 2],
                i=f_obj[:, 0],
                j=f_obj[:, 1],
                k=f_obj[:, 2],
                text=['object'] * len(v_obj),
                color=object_color,
                opacity=opacity,
                hovertemplate='%{text}'
            ))

        # Contact points
        if with_contact_points and self.contact_points is not None:
            contact_points_world = self.contact_points
            contact_points = contact_points_world[i].detach().cpu()
            data.append(go.Scatter3d(
                x=contact_points[:, 0], y=contact_points[:,
                                                         1], z=contact_points[:, 2],
                mode='markers', marker=dict(color='red', size=5)
            ))

        # Contact candidates
        if with_contact_candidates:
            contact_candidates = self.get_contact_candidates_world()[
                i].detach().cpu()
            data.append(go.Scatter3d(
                x=contact_candidates[:, 0], y=contact_candidates[:,
                                                                 1], z=contact_candidates[:, 2],
                mode='markers', marker=dict(color='blue', size=5)
            ))

        return data

    def get_trimesh_data(self, i: int):
        """
        Generate a combined trimesh object for visualization.

        Args:
            i (int): Index of the batch for which to generate the mesh.

        Returns:
            trimesh.Trimesh: Combined mesh of the hand and object, including contact points as spheres.
        """
        combined_mesh = trimesh.Trimesh()

        # Add hand links
        for link_name in self.mesh:
            vertices = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if vertices.dim() == 3:  # Batch dimension
                vertices = vertices[i]
            vertices = vertices @ self.global_rotation[i].T + \
                self.global_translation[i]
            vertices = vertices.detach().cpu().numpy()
            faces = self.mesh[link_name]['faces'].detach().cpu().numpy()

            link_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            # Light blue, semi-transparent
            link_mesh.visual.face_colors = [173, 216, 230, 100]
            combined_mesh = combined_mesh + link_mesh

        # Add object mesh
        if self.object_verts is not None and self.object_faces is not None:
            if self.object_translation is None:
                self.object_translation = torch.zeros_like(
                    self.global_translation)
            if self.object_rotation is None:
                self.object_rotation = torch.eye(3, dtype=torch.float, device=self.device).unsqueeze(
                    0).repeat(self.global_translation.shape[0], 1, 1)
            if self.object_scales is None:
                self.object_scales = torch.ones(
                    self.global_translation.shape[0], device=self.device
                )

            # 1) scale
            obj_vertices = self.object_verts * self.object_scales[i]
            # 2) rotate
            obj_vertices = (obj_vertices.unsqueeze(0) @
                            self.object_rotation[i].T).squeeze(0)
            # 3) translate
            obj_vertices = obj_vertices + self.object_translation[i]
            # 4) global transform
            obj_vertices = (obj_vertices @
                            self.global_rotation[i].T) + self.global_translation[i]

            obj_vertices = obj_vertices.detach().cpu().numpy()
            obj_faces = self.object_faces.detach().cpu().numpy()

            object_mesh = trimesh.Trimesh(vertices=obj_vertices,
                                          faces=obj_faces)
            # Light green, semi-transparent
            object_mesh.visual.face_colors = [144, 238, 144, 100]
            combined_mesh = combined_mesh + object_mesh

        # Add contact points as spheres
        contact_points_world = self.contact_points
        if contact_points_world is not None:
            contact_points = contact_points_world[i].detach().cpu().numpy()
            for cp in contact_points:
                sphere = trimesh.primitives.Sphere(radius=0.002, center=cp)
                sphere.visual.face_colors = [255, 0, 0, 255]  # Solid red
                combined_mesh = sphere + combined_mesh

        return combined_mesh
