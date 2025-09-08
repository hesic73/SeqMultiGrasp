import torch
import transforms3d
import math
import pytorch3d.structures
import pytorch3d.ops
from pytorch3d import transforms
import trimesh as tm
import numpy as np

from .hand_model import HandModel
from .hand_model_with_object import HandModelWithObject
from .object_model import ObjectModel
from .single_object_model import SingleObjectModel


from typing import List, Tuple, Sequence



def random_rotation_around_direction(total_batch_size: int, direction: torch.Tensor, perturbation_strength: float):

    device = direction.device

    direction = direction / torch.norm(direction, dim=0, keepdim=True)

    perturb_x = perturbation_strength * \
        (torch.rand([total_batch_size], device=device) - 0.5)
    perturb_y = perturbation_strength * \
        (torch.rand([total_batch_size], device=device) - 0.5)
    perturb_z = perturbation_strength * \
        (torch.rand([total_batch_size], device=device) - 0.5)

    perturbed_direction = torch.stack([
        direction[0] + perturb_x,
        direction[1] + perturb_y,
        direction[2] + perturb_z
    ], dim=1)

    perturbed_direction = perturbed_direction / \
        torch.norm(perturbed_direction, dim=1, keepdim=True)

    angle = math.pi / 2 * \
        torch.rand([total_batch_size], device=device) - math.pi / 4

    rotation_matrices = transforms.axis_angle_to_matrix(
        perturbed_direction * angle.unsqueeze(-1))

    return rotation_matrices


def initialize_convex_hull(
    hand_model: HandModel,
    object_model: ObjectModel,
    distance_lower: float,
    distance_upper: float,
    theta_lower: float,
    theta_upper: float,
    jitter_strength: float,
    n_contact: int,
    contact_candidates_weight: torch.Tensor,
):
    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each

    translation = torch.zeros([total_batch_size, 3],
                              dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3],
                           dtype=torch.float, device=device)

    for i in range(n_objects):

        # get inflated convex hull
        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces(
        )]

        # Inflate the hull slightly
        vertices += 0.2 * vertices / \
            np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(
            mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(
            vertices.unsqueeze(0), faces.unsqueeze(0))

        # sample points
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
            mesh_pytorch3d, num_samples=100 * batch_size_each)
        p = pytorch3d.ops.sample_farthest_points(
            dense_point_cloud, K=batch_size_each)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(
            p.detach().cpu().numpy())
        closest_points = torch.tensor(
            closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        # sample parameters
        distance = distance_lower + (distance_upper - distance_lower) * torch.rand(
            [batch_size_each], dtype=torch.float, device=device
        )
        deviate_theta = theta_lower + (theta_upper - theta_lower) * torch.rand(
            [batch_size_each], dtype=torch.float, device=device
        )
        process_theta = 2 * math.pi * \
            torch.rand([batch_size_each], dtype=torch.float, device=device)
        rotate_theta = 2 * math.pi * \
            torch.rand([batch_size_each], dtype=torch.float, device=device)

        # solve transformation
        rotation_local = torch.zeros(
            [batch_size_each, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros(
            [batch_size_each, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'),
                dtype=torch.float, device=device
            )
            rotation_global[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1], n[j, 0]) - math.pi / 2,
                    -math.acos(n[j, 2]),
                    0,
                    axes='rzxz'
                ),
                dtype=torch.float, device=device
            )

        translation[i * batch_size_each: (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * (
            rotation_global @ rotation_local @ torch.tensor(
                [0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
        ).squeeze(2)

        rotation_hand = torch.tensor(
            transforms3d.euler.euler2mat(-np.pi /
                                         2, -np.pi / 2, 0, axes='rzyz'),
            dtype=torch.float, device=device
        )
        rotation[i * batch_size_each: (
            i + 1) * batch_size_each] = rotation_global @ rotation_local @ rotation_hand

    joint_angles_mu = torch.tensor(
        [0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 1.4, 0, 0, 0],
        dtype=torch.float, device=device
    )
    joint_angles_sigma = jitter_strength * \
        (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros(
        [total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            joint_angles_mu[i],
            joint_angles_sigma[i],
            hand_model.joints_lower[i] - 1e-6,
            hand_model.joints_upper[i] + 1e-6
        )

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    weight_matrix = contact_candidates_weight.unsqueeze(
        0).expand(total_batch_size, -1)

    # Use these indices to select from the candidate tensor
    contact_point_indices = torch.multinomial(
        weight_matrix, n_contact, replacement=False)

    hand_model.set_parameters(hand_pose, contact_point_indices)


def single_initialize_convex_hull(
    hand_model: HandModel,
    object_model: SingleObjectModel,
    distance_lower: float,
    distance_upper: float,
    theta_lower: float,
    theta_upper: float,
    jitter_strength: float,
    n_contact: int,
    contact_candidates_weight: torch.Tensor,
):
    device = hand_model.device
    total_batch_size = object_model.batch_size

    # get inflated convex hull
    mesh_origin = object_model.object_mesh.convex_hull
    vertices = mesh_origin.vertices.copy()
    faces = mesh_origin.faces
    vertices *= object_model.scale_factor.max().item()
    mesh_origin = tm.Trimesh(vertices, faces)
    mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces(
    )]

    # Inflate the hull slightly
    vertices += 0.2 * vertices / \
        np.linalg.norm(vertices, axis=1, keepdims=True)
    mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
    vertices = torch.tensor(
        mesh.vertices, dtype=torch.float, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
    mesh_pytorch3d = pytorch3d.structures.Meshes(
        vertices.unsqueeze(0), faces.unsqueeze(0))

    # sample points
    dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
        mesh_pytorch3d, num_samples=100 * total_batch_size)
    p = pytorch3d.ops.sample_farthest_points(
        dense_point_cloud, K=total_batch_size)[0][0]
    closest_points, _, _ = mesh_origin.nearest.on_surface(
        p.detach().cpu().numpy())
    closest_points = torch.tensor(
        closest_points, dtype=torch.float, device=device)
    n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

    # sample parameters
    distance = distance_lower + (distance_upper - distance_lower) * torch.rand(
        [total_batch_size], dtype=torch.float, device=device
    )
    deviate_theta = theta_lower + (theta_upper - theta_lower) * torch.rand(
        [total_batch_size], dtype=torch.float, device=device
    )
    process_theta = 2 * math.pi * \
        torch.rand([total_batch_size], dtype=torch.float, device=device)
    rotate_theta = 2 * math.pi * \
        torch.rand([total_batch_size], dtype=torch.float, device=device)

    # solve transformation
    rotation_local = torch.zeros(
        [total_batch_size, 3, 3], dtype=torch.float, device=device)
    rotation_global = torch.zeros(
        [total_batch_size, 3, 3], dtype=torch.float, device=device)
    for j in range(total_batch_size):
        rotation_local[j] = torch.tensor(
            transforms3d.euler.euler2mat(
                process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'),
            dtype=torch.float, device=device
        )
        rotation_global[j] = torch.tensor(
            transforms3d.euler.euler2mat(
                math.atan2(n[j, 1], n[j, 0]) - math.pi / 2,
                -math.acos(n[j, 2]),
                0,
                axes='rzxz'
            ),
            dtype=torch.float, device=device
        )

    translation = p - distance.unsqueeze(1) * (
        rotation_global @ rotation_local @ torch.tensor(
            [0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
    ).squeeze(2)

    rotation_hand = torch.tensor(
        transforms3d.euler.euler2mat(-np.pi /
                                     2, -np.pi / 2, 0, axes='rzyz'),
        dtype=torch.float, device=device
    )
    rotation = rotation_global @ rotation_local @ rotation_hand

    joint_angles_mu = torch.tensor(
        [0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 1.4, 0, 0, 0],
        dtype=torch.float, device=device
    )
    joint_angles_sigma = jitter_strength * \
        (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros(
        [total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            joint_angles_mu[i],
            joint_angles_sigma[i],
            hand_model.joints_lower[i] - 1e-6,
            hand_model.joints_upper[i] + 1e-6
        )

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    weight_matrix = contact_candidates_weight.unsqueeze(
        0).expand(total_batch_size, -1)

    # Use these indices to select from the candidate tensor
    contact_point_indices = torch.multinomial(
        weight_matrix, n_contact, replacement=False)

    hand_model.set_parameters(hand_pose, contact_point_indices)


def initialize_seq_multi_grasp(
    hand_model: HandModelWithObject,
    total_batch_size: int,
    n_contact: int,
    jitter_strength: float,
    contact_candidates_weight: torch.Tensor,
    # New arguments (similar to initialize_multi_grasp):
    translation_loc: Sequence[float],
    translation_scale: Sequence[float],
    rotation_rpy: Sequence[float],
    qpos_loc: Sequence[float],
    # Optional: If you want to mask out certain joints
    active_joint_mask: torch.Tensor = None,
):
    
    device = hand_model.device

    # 1) Sample random translation around translation_loc
    translation_loc = torch.tensor(
        translation_loc, dtype=torch.float, device=device)
    translation_scale = torch.tensor(
        translation_scale, dtype=torch.float, device=device)

    translation = torch.randn([total_batch_size, 3], device=device)
    translation = translation * translation_scale + translation_loc

    # 2) Convert base RPY to rotation matrix (like initialize_multi_grasp)
    R_hand_base_np = transforms3d.euler.euler2mat(
        *rotation_rpy, axes='rzyz')  # or 'sxyz'
    R_hand_base = torch.tensor(
        R_hand_base_np, dtype=torch.float, device=device)

    # For demonstration, let’s pick a direction vector around +X (as in initialize_multi_grasp)
    direction_vector = torch.tensor([1.0, 0.0, 0.0], device=device)
    direction_vector = R_hand_base @ direction_vector
    direction_vector = -direction_vector  # As in the original code

    # Generate random rotation around that direction
    perturbation_strength = 0.5
    R_around_dir = random_rotation_around_direction(
        total_batch_size, direction_vector, perturbation_strength
    )

    # Combine with the base rotation
    R_hand = R_hand_base.unsqueeze(0).expand(total_batch_size, -1, -1)
    rotation = torch.bmm(R_around_dir, R_hand)  # (B, 3, 3)

    # 3) Sample joint angles around qpos_loc
    qpos_loc = torch.tensor(qpos_loc, dtype=torch.float, device=device)
    joint_angles_sigma = jitter_strength * \
        (hand_model.joints_upper - hand_model.joints_lower)

    # Initialize a random joint angle tensor
    joint_angles = torch.zeros(
        [total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            mean=qpos_loc[i],
            std=joint_angles_sigma[i],
            a=hand_model.joints_lower[i] - 1e-6,
            b=hand_model.joints_upper[i] + 1e-6
        )

    # 4) If using a mask, only apply the new joint angles to the “active” ones
    #    and keep the existing angles for the “inactive” ones
    if active_joint_mask is not None:
        # Grab the original angles from the existing hand pose
        # (If the hand_model was freshly created, you might do something else here)
        original_joint_angles = hand_model.hand_pose[:, 9:].clone().detach()
        joint_angles = torch.where(
            active_joint_mask,  # shape (B, n_dofs)
            joint_angles,
            original_joint_angles
        )

    # 5) Construct final hand pose
    #    Rotation: we only store the first 2 columns in 6D format (like ortho6D)
    rotation_6d = rotation.transpose(1, 2)[:, :2].reshape(-1, 6)
    hand_pose = torch.cat([translation, rotation_6d, joint_angles], dim=1)
    hand_pose.requires_grad_()

    # 6) Randomly select contact point indices
    weight_matrix = contact_candidates_weight.unsqueeze(
        0).expand(total_batch_size, -1)

    # Use these indices to select from the candidate tensor
    contact_point_indices = torch.multinomial(
        weight_matrix, n_contact, replacement=False)

    # 7) Finally set parameters on the hand_model
    hand_model.set_parameters(hand_pose, contact_point_indices)


# as in MultiGrasp
def initialize_multi_grasp(
    hand_model: HandModel,
    batch_size: int,
    n_contact: int,
    contact_candidates_weight: torch.Tensor,
    qpos_loc: Sequence[float],
    jitter_strength: float,
):
    device = hand_model.device

    R_palm_down = transforms.euler_angles_to_matrix(torch.tensor(
        [0.0, 0.0, -torch.pi / 2]).unsqueeze(0).tile([batch_size, 1]).to(device), "XYZ")

    z = torch.tensor([0, 0, 1], device=device, dtype=torch.float32).unsqueeze(
        0).tile([batch_size, 1])

    v = torch.zeros([batch_size, 3], dtype=torch.float, device=device)
    v[:, 0] = torch.rand(batch_size, device=device)-0.5
    v[:, 2] = torch.rand(batch_size, device=device)*0.7+0.3
    v[:, 1] = torch.rand(batch_size, device=device)*0.6-0.3
    v = v / torch.norm(v, dim=1).view(-1, 1)  # Normalize
    # print(v)

    axis = torch.linalg.cross(z, v)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-12) * \
        torch.acos(torch.clamp(torch.sum(z * v, dim=1), -1, 1)).unsqueeze(-1)
    R_upper_sphere = transforms.axis_angle_to_matrix(axis)
    R = torch.matmul(R_upper_sphere, R_palm_down)
    R6 = compute_ortho6d_from_rotation_matrix(R)

    translation = torch.zeros([batch_size, 3],
                              dtype=torch.float, device=device)

    translation[:, 0] = -(torch.rand(batch_size, device=device) * 0.2 + 0.05)

    translation = torch.matmul(R, translation.unsqueeze(-1)).squeeze(-1)

    rot6d_tensor = R6.clone()+torch.normal(0, 0.1,
                                           [batch_size, 6], dtype=torch.float32, device=device)

    joint_angles_mu = torch.tensor(qpos_loc,
                                   dtype=torch.float, device=device)
    joint_angles_sigma = jitter_strength * \
        (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros(
        [batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
                                    hand_model.joints_lower[i] - 1e-6, hand_model.joints_upper[i] + 1e-6)

    hand_pose = torch.cat([translation, rot6d_tensor, joint_angles], dim=1)
    hand_pose = hand_pose.contiguous()
    hand_pose.requires_grad_()

    weight_matrix = contact_candidates_weight.unsqueeze(
        0).expand(batch_size, -1)

    # Use these indices to select from the candidate tensor
    contact_point_indices = torch.multinomial(
        weight_matrix, n_contact, replacement=False)

    hand_model.set_parameters(hand_pose, contact_point_indices)


def compute_ortho6d_from_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.transpose(-1, -2)[:, 0:2].reshape([-1, 6])


def initialize_side_grasp(
    hand_model: HandModel,
    batch_size: int,
    n_contact: int,
    contact_candidates_weight: torch.Tensor,
    qpos_loc: Sequence[float],
    jitter_strength: float,
):
   
    device = hand_model.device

    R_side_base = transforms.euler_angles_to_matrix(
        torch.tensor([0.0, 0.0, 0.0]).unsqueeze(
            0).expand(batch_size, 3).to(device),
        convention="XYZ"
    )  # shape (B,3,3)

    y_ref = torch.tensor([0.0, 1.0, 0.0], device=device,
                         dtype=torch.float32).unsqueeze(0)
    y_ref = y_ref.expand(batch_size, 3)  # (B,3)

    v = torch.zeros(batch_size, 3, device=device)
    v[:, 0] = torch.rand(batch_size, device=device) - 0.5
    v[:, 2] = torch.rand(batch_size, device=device) - 0.5
    v[:, 1] = torch.rand(batch_size, device=device) * 0.5 + 0.3
    v = v / (v.norm(dim=1, keepdim=True) + 1e-12)

    axis = torch.linalg.cross(y_ref, v)  # (B,3)
    axis_norm = axis.norm(dim=1, keepdim=True) + 1e-12
    axis = axis / axis_norm
    angle = torch.acos(torch.clamp(
        (y_ref * v).sum(dim=1), min=-1.0, max=1.0))
    axis = axis * angle.unsqueeze(-1)  # axis-angle
    R_upper_sphere = transforms.axis_angle_to_matrix(axis)  # (B,3,3)

    R = torch.matmul(R_upper_sphere, R_side_base)  # (B,3,3)

    R6 = compute_ortho6d_from_rotation_matrix(R)
    rot6d_tensor = R6 + torch.normal(
        mean=0.0, std=0.1, size=(batch_size, 6), device=device
    )

    translation_local = torch.zeros((batch_size, 3), device=device)
    translation_local[:, 0] = - \
        (torch.rand(batch_size, device=device)*0.2 + 0.05)

    translation = (R @ translation_local.unsqueeze(-1)).squeeze(-1)

    joint_angles_mu = torch.tensor(qpos_loc, dtype=torch.float, device=device)
    joint_angles_sigma = jitter_strength * \
        (hand_model.joints_upper - hand_model.joints_lower)
    joint_angles = torch.zeros(
        (batch_size, hand_model.n_dofs), dtype=torch.float, device=device)

    for i in range(hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i],
            mean=joint_angles_mu[i],
            std=joint_angles_sigma[i],
            a=hand_model.joints_lower[i] - 1e-6,
            b=hand_model.joints_upper[i] + 1e-6
        )

    hand_pose = torch.cat([translation, rot6d_tensor, joint_angles], dim=1)
    hand_pose.requires_grad_()

    weight_matrix = contact_candidates_weight.unsqueeze(
        0).expand(batch_size, -1)
    contact_point_indices = torch.multinomial(
        weight_matrix, n_contact, replacement=False)

    hand_model.set_parameters(hand_pose, contact_point_indices)
