import os
import torch
import pytorch_kinematics as pk
from pytorch_kinematics.frame import Frame
import trimesh


class RobotModelURDFLite:
    def __init__(
        self,
        urdf_path: str,
        device: str = 'cpu',
        collision_spheres: dict = None,
    ):
        """
        A Lite Robot Model for a URDF robot that omits contact points and optional
        global transform, but can optionally handle collision spheres.

        Parameters
        ----------
        urdf_path: str
            Path to URDF file.
        device: str or torch.Device
            Which device to load the model into (e.g., 'cpu', 'cuda').
        collision_spheres: dict or None
            Optional dictionary describing collision spheres. The format should be:
                {
                  "link_name1": [
                      {"center": [x, y, z], "radius": r},
                      {"center": [x2, y2, z2], "radius": r2},
                      ...
                  ],
                  "link_name2": [
                      ...
                  ],
                  ...
                }
            Each sphere is defined in the local coordinates of its link.
        """
        self.device = device
        self.collision_spheres = collision_spheres  # store for optional visualization

        # Build kinematic chain from URDF
        with open(urdf_path, 'r') as f:
            urdf_xml = f.read()

        self.chain = pk.build_chain_from_urdf(
            urdf_xml).to(dtype=torch.float, device=device)
        self.n_dofs = len(self.chain.get_joint_parameter_names())

        # Build local mesh geometry for each link
        self.mesh = {}
        base_path = os.path.dirname(os.path.abspath(urdf_path))

        def build_mesh_recurse(body: Frame):
            not_none_visuals = [
                visual for visual in body.link.visuals if visual.geom_type is not None]
            if len(not_none_visuals) > 0:
                link_vertices = []
                link_faces = []
                n_link_vertices = 0
                for visual in not_none_visuals:
                    scale = torch.tensor(
                        [1, 1, 1], dtype=torch.float, device=device)
                    # Create geometry
                    if visual.geom_type == "box":
                        link_mesh = trimesh.primitives.Box(
                            extents=2*visual.geom_param)
                    elif visual.geom_type == "capsule":
                        link_mesh = trimesh.primitives.Capsule(
                            radius=visual.geom_param[0],
                            height=visual.geom_param[1]*2
                        ).apply_translation((0, 0, -visual.geom_param[1]))
                    else:
                        # e.g. mesh file, stl, obj, etc.
                        relative_path = visual.geom_param[0].replace(
                            "package://", "")
                        link_mesh = trimesh.load(
                            os.path.join(base_path, relative_path), force='mesh'
                        )
                        # Possibly scale the mesh
                        if visual.geom_param[1] is not None:
                            scale = visual.geom_param[1]
                            if isinstance(scale, list):
                                scale = torch.tensor(
                                    scale, dtype=torch.float, device=device)
                    # Convert to torch tensors
                    vertices = torch.tensor(
                        link_mesh.vertices, dtype=torch.float, device=device
                    )
                    faces = torch.tensor(
                        link_mesh.faces, dtype=torch.float, device=device
                    )
                    # Apply local offset transform (position + orientation)
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

            for child in body.children:
                build_mesh_recurse(child)

        build_mesh_recurse(self.chain._root)

        # Record joint info (names, lower bounds, upper bounds)
        self.joint_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body: Frame):
            if body.joint.joint_type != "fixed":
                self.joint_names.append(body.joint.name)
                self.joints_lower.append(
                    torch.tensor(
                        body.joint.limits[0], dtype=torch.float, device=device)
                )
                self.joints_upper.append(
                    torch.tensor(
                        body.joint.limits[1], dtype=torch.float, device=device)
                )
            for child in body.children:
                set_joint_range_recurse(child)

        set_joint_range_recurse(self.chain._root)

        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

        # Will hold forward kinematics results
        self.current_status = None

    def set_parameters(self, qpos: torch.Tensor):
        """
        Set the joint angles for the robot (batch-compatible).

        Parameters
        ----------
        qpos : (B, n_dofs) torch.FloatTensor
            Joint angles for each dof. B is batch size.
        """
        # Forward kinematics
        self.current_status = self.chain.forward_kinematics(qpos)

    def get_trimesh_data(self, i: int = 0) -> trimesh.Trimesh:
        """
        Retrieve a combined trimesh of the entire robot for a given batch index.
        Includes optional collision sphere visualization if self.collision_spheres is set.

        Parameters
        ----------
        i: int
            Which batch index to visualize.

        Returns
        -------
        trimesh.Trimesh
            A single trimesh with all links (and optionally collision spheres) 
            combined for the i-th sample in the batch.
        """
        # Start with an empty trimesh
        combined_mesh = trimesh.Trimesh()

        # If we haven't run forward kinematics yet, no geometry can be produced
        if self.current_status is None:
            print("Warning: No current_status found. Please call set_parameters() first.")
            return combined_mesh

        # 1. Combine all link meshes
        for link_name, link_mesh_data in self.mesh.items():
            # Convert link vertices from link-local coords -> world coords
            # using the transformation matrix at the i-th sample
            v = self.current_status[link_name].transform_points(
                link_mesh_data['vertices'])
            if len(v.shape) == 3:
                v = v[i]  # select batch i if B > 1
            v = v.detach().cpu()

            f = link_mesh_data['faces'].detach().cpu()
            link_mesh = trimesh.Trimesh(vertices=v, faces=f)
            link_mesh.visual.face_colors = [
                173, 216, 230, 100]  # Light blue RGBA
            combined_mesh = combined_mesh + link_mesh

        # 2. Optionally visualize collision spheres
        if self.collision_spheres is not None:
            for link_name, sphere_list in self.collision_spheres.items():
                # Skip if link is not in current_status (e.g. urdf mismatch)
                if link_name not in self.current_status:
                    continue

                # Link transform for batch index i
                T_link = self.current_status[link_name].get_matrix()[
                    i]  # (4,4)

                for sphere_def in sphere_list:
                    center_local = torch.tensor(
                        sphere_def["center"], dtype=torch.float, device=self.device
                    )
                    radius = sphere_def["radius"]

                    # Transform local center to world coords
                    center_local_h = torch.cat(
                        [center_local, torch.ones(1, device=self.device)], dim=0
                    )
                    # (4,) in homogeneous coords
                    center_world_h = T_link @ center_local_h
                    center_world = center_world_h[:3].detach().cpu().numpy()

                    # Create a sphere mesh
                    sphere_mesh = trimesh.primitives.Sphere(
                        radius=radius, center=center_world)
                    sphere_mesh.visual.face_colors = [
                        0, 255, 0, 80]  # semi-transparent green
                    combined_mesh = combined_mesh + sphere_mesh

        return combined_mesh
