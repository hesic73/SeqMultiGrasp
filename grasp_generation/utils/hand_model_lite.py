import os
import torch
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import pytorch_kinematics as pk
from pytorch_kinematics.frame import Frame
import trimesh
import numpy as np
import transforms3d

from typing import Dict


class HandModelURDFLite:
    def __init__(
        self,
        urdf_path: str,
        device='cpu',
        contact_candidates: Dict = None,
        collision_spheres: Dict = None,
    ):
        """
        Create a Lite Hand Model for a URDF robot

        Parameters
        ----------
        urdf_path: str
            path to urdf file
        device: str | torch.Device
            device for torch tensors
        contact_candidates: Dict or None
            A dictionary keyed by link name, each value a (N, 3) tensor of candidate 
            contact points in link coordinates. If None, no contact points will be available.
        collision_spheres: Dict or None
            A dictionary keyed by link name, each value is a list of dict like
            {"center": [...], "radius": ...}, describing spheres in link coordinates.
            If None, no collision spheres will be available.
        """
        self.device = device
        self.chain = pk.build_chain_from_urdf(
            open(urdf_path).read()).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        self.mesh = {}
        base_path = os.path.dirname(os.path.dirname(urdf_path))

        def build_mesh_recurse(body: Frame):
            if (len(body.link.visuals) > 0):
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in body.link.visuals:
                    scale = torch.tensor(
                        [1, 1, 1], dtype=torch.float, device=device)
                    if visual.geom_type == "box":
                        link_mesh = trimesh.primitives.Box(
                            extents=2*visual.geom_param)
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(
                            radius=visual.geom_param[0],
                            height=visual.geom_param[1]*2
                        ).apply_translation((0, 0, -visual.geom_param[1]))
                    else:
                        relative_path = visual.geom_param[0].replace(
                            "package://", "")
                        link_mesh = trimesh.load_mesh(
                            os.path.join(base_path, relative_path), process=False
                        )
                        if visual.geom_param[1] is not None:
                            scale = (visual.geom_param[1]).to(
                                dtype=torch.float, device=device
                            )
                    vertices = torch.tensor(
                        link_mesh.vertices, dtype=torch.float, device=device
                    )
                    faces = torch.tensor(
                        link_mesh.faces, dtype=torch.float, device=device
                    )
                    pos = visual.offset.to(dtype=torch.float, device=device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)
                link_vertices = torch.cat(link_vertices, dim=0)
                link_faces = torch.cat(link_faces, dim=0)
                self.mesh[body.link.name] = {
                    'vertices': link_vertices,
                    'faces': link_faces
                }
            for children in body.children:
                build_mesh_recurse(children)

        build_mesh_recurse(self.chain._root)

        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body: Frame):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                self.joints_lower.append(torch.tensor(
                    body.joint.limits[0], dtype=torch.float, device=device))
                self.joints_upper.append(torch.tensor(
                    body.joint.limits[1], dtype=torch.float, device=device))
            for children in body.children:
                set_joint_range_recurse(children)

        set_joint_range_recurse(self.chain._root)

        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        # contact points
        self.contact_candidates = contact_candidates
        self.contact_point_indices = None
        self.contact_points = None

        # collision spheres
        self.collision_spheres = collision_spheres

        self.hand_pose = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None

    def set_parameters(
        self,
        hand_pose: torch.Tensor,
        contact_point_indices: torch.Tensor = None
    ):
        """
        Set translation, rotation, and joint angles of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, N) torch.LongTensor (optional)
            Indices into the contact_candidates that select which points to use as contact points.
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation: torch.Tensor = robust_compute_rotation_matrix_from_ortho6d(
            self.hand_pose[:, 3:9]
        )
        self.current_status = self.chain.forward_kinematics(
            self.hand_pose[:, 9:]
        )

        self.contact_point_indices = contact_point_indices
        self.contact_points = None
        if self.contact_candidates is not None and contact_point_indices is not None:
            all_candidates = []
            link_list = []
            for link_name, cpoints in self.contact_candidates.items():
                all_candidates.append(cpoints.to(self.device))
                link_list.extend([link_name] * cpoints.shape[0])
            all_candidates = torch.cat(all_candidates, dim=0)

            batch_size, n_contact = contact_point_indices.shape
            selected_points = all_candidates[contact_point_indices]

            link_indices = [
                link_list[idx.item()] for idx in contact_point_indices.flatten()
            ]
            link_indices = [
                link_indices[i * n_contact:(i + 1) * n_contact]
                for i in range(batch_size)
            ]

            contact_points_world = torch.zeros_like(
                selected_points, device=self.device)
            for b in range(batch_size):
                for j in range(n_contact):
                    link_name = link_indices[b][j]
                    T = self.current_status[link_name].get_matrix()[b]  # (4,4)
                    pt = torch.cat(
                        [selected_points[b, j], torch.tensor(
                            [1.0], device=self.device)],
                        dim=0
                    )
                    pt_world = T @ pt  # (4,)
                    contact_points_world[b, j] = pt_world[:3]

            contact_points_world = (
                contact_points_world @ self.global_rotation.transpose(1, 2)
                + self.global_translation.unsqueeze(1)
            )
            self.contact_points = contact_points_world

    def get_trimesh_data(
        self,
        i: int,
        link_directions=None
    ):
        data = trimesh.Trimesh()

        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            link_mesh = trimesh.Trimesh(vertices=v, faces=f)
            link_mesh.visual.face_colors = [173, 216, 230, 100]  # RGBA
            data = data + link_mesh

        if self.contact_points is not None:
            cpoints = self.contact_points[i].detach().cpu().numpy()
            for cp in cpoints:
                sphere = trimesh.primitives.Sphere(radius=0.002, center=cp)
                sphere.visual.face_colors = [255, 0, 0, 255]
                data = data + sphere

        if self.collision_spheres is not None:
            for link_name, sphere_list in self.collision_spheres.items():
                T_link = self.current_status[link_name].get_matrix()[i]

                for sphere_def in sphere_list:
                    center_local = torch.tensor(
                        sphere_def["center"], dtype=torch.float, device=self.device
                    )
                    radius = sphere_def["radius"]

                    pt_hom = torch.cat(
                        [center_local, torch.ones(1, device=self.device)], dim=0)
                    pt_world_chain = T_link @ pt_hom  # 4D
                    center_in_chain = pt_world_chain[:3]

                    center_in_world = (
                        center_in_chain @ self.global_rotation[i].T
                        + self.global_translation[i]
                    )

                    center_in_world = center_in_world.detach().cpu().numpy()
                    col_sphere = trimesh.primitives.Sphere(
                        radius=radius, center=center_in_world)
                    col_sphere.visual.face_colors = [0, 255, 0, 80]
                    data = data + col_sphere

        if link_directions is not None:
            for link_name, info in link_directions.items():
                if link_name not in self.current_status:
                    continue

                directions_local = info.get("directions", [])
                color = info.get("color", [255, 0, 0, 255])  # RGBA
                arrow_radius = info.get("arrow_radius", 0.002)
                head_ratio = info.get("head_ratio", 0.2)

                T_link = self.current_status[link_name].get_matrix()[
                    i]  # (4,4)
                T_link_np = T_link.detach().cpu().numpy()

                origin_in_chain = T_link_np[:3, 3]  # (3,)
                R_global = self.global_rotation[i].detach(
                ).cpu().numpy()  # (3,3)
                t_global = self.global_translation[i].detach(
                ).cpu().numpy()  # (3,)

                origin_in_world = origin_in_chain @ R_global.T + t_global

                R_link = T_link_np[:3, :3]

                for d_local in directions_local:
                    d_local = np.array(d_local, dtype=np.float32)
                    d_chain = R_link @ d_local  # (3,)
                    d_world = d_chain @ R_global.T  # (3,)

                    arrow_geo = create_arrow_geometry(
                        origin_in_world=origin_in_world,
                        direction_in_world=d_world*0.05,
                        arrow_radius=arrow_radius,
                        head_ratio=head_ratio,
                        color_rgba=color
                    )
                    if arrow_geo is not None:
                        data = data + arrow_geo

        return data


def create_arrow_geometry(
    origin_in_world: np.ndarray,
    direction_in_world: np.ndarray,
    arrow_radius: float = 0.002,
    head_ratio: float = 0.2,
    color_rgba=[255, 0, 0, 255],
):
    length = np.linalg.norm(direction_in_world)
    if length < 1e-8:
        return None

    direction_unit = direction_in_world / length

    head_length = length * head_ratio
    shaft_length = length - head_length

    shaft = trimesh.creation.cylinder(
        radius=arrow_radius,
        height=shaft_length,
        sections=12
    )
    shaft.apply_translation([0, 0, shaft_length / 2.0])

    cone = trimesh.creation.cone(
        radius=arrow_radius * 2.0,
        height=head_length,
        sections=12
    )
    cone.apply_translation([0, 0, shaft_length + head_length / 2.0])

    arrow_geometry = shaft + cone

    z_axis = np.array([0, 0, 1], dtype=np.float32)
    rot_axis = np.cross(z_axis, direction_unit)
    dot_zz = np.dot(z_axis, direction_unit)
    if np.linalg.norm(rot_axis) < 1e-8 and dot_zz < 0:
        R = transforms3d.axangles.axangle2mat([1, 0, 0], np.pi)
    elif np.linalg.norm(rot_axis) < 1e-8:
        R = np.eye(3, dtype=np.float32)
    else:
        angle = np.arccos(np.clip(dot_zz, -1.0, 1.0))
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        R = transforms3d.axangles.axangle2mat(rot_axis, angle)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = origin_in_world
    arrow_geometry.apply_transform(T)

    arrow_geometry.visual.face_colors = color_rgba

    return arrow_geometry
