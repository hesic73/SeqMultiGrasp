import torch
import numpy as np

from typing import List, Dict, Tuple, Callable, Optional

from pytorch3d.transforms import matrix_to_axis_angle,  axis_angle_to_matrix


from src.utils import torch_3d_utils, rot6d_utils
from src.utils.misc_utils import convert_xyz_rpy_to_xyz_rot6d, convert_xyz_rot6d_to_xyz_rpy

from src.utils.keypoints_utils import HandModelKeypoints

from enum import Enum


class RotationRepresentation(Enum):
    rot6d = 'rot6d'
    rpy = 'rpy'
    quaternion = 'quaternion'
    rotation_matrix = 'rotation_matrix'
    axis_angle = 'axis_angle'


def _convert_pose(pose: torch.Tensor, rotation_representation: RotationRepresentation) -> torch.Tensor:
    if rotation_representation == RotationRepresentation.rot6d:
        return convert_xyz_rpy_to_xyz_rot6d(pose)
    elif rotation_representation == RotationRepresentation.rpy:
        return pose
    elif rotation_representation == RotationRepresentation.quaternion:
        xyz, rpy = pose[:, :3], pose[:, 3:]
        quat = torch_3d_utils.rpy_to_quaternion(rpy)
        return torch.cat([xyz, quat], dim=1)
    elif rotation_representation == RotationRepresentation.rotation_matrix:
        xyz, rpy = pose[:, :3], pose[:, 3:]
        rotmat = torch_3d_utils.rpy_to_rotation_matrix(rpy)
        rotmat = rotmat.flatten(start_dim=1)
        return torch.cat([xyz, rotmat], dim=1)
    elif rotation_representation == RotationRepresentation.axis_angle:
        xyz, rpy = pose[:, :3], pose[:, 3:]
        rotmat = torch_3d_utils.rpy_to_rotation_matrix(rpy)
        axis_angle = matrix_to_axis_angle(rotmat)
        return torch.cat([xyz, axis_angle], dim=1)
    else:
        raise NotImplementedError(
            f"Unsupported rotation representation: {rotation_representation}")


# https://github.com/PKU-EPIC/DexGraspNet2/blob/26ecd76121e3c8218ad53db9840cf34f6b81b076/src/utils/util.py#L24
def proper_svd(rot: torch.Tensor):
    """
    Compute proper SVD of a rotation matrix.

    rot: (B, 3, 3)
    return: Rotation matrix (B, 3, 3) with det = 1
    """

    u, s, v = torch.svd(rot.double())

    with torch.no_grad():
        sign = torch.sign(torch.det(torch.einsum(
            'bij,bkj->bik', u, v)))
        diag = torch.stack([
            torch.ones_like(s[:, 0]),
            torch.ones_like(s[:, 1]),
            sign
        ], dim=-1)
        diag = torch.diag_embed(diag)

    return torch.einsum('bij,bjk,blk->bil', u, diag, v).to(rot.dtype)


def _convert_pose_back(pose: torch.Tensor, rotation_representation: RotationRepresentation) -> torch.Tensor:
    if rotation_representation == RotationRepresentation.rot6d:
        return convert_xyz_rot6d_to_xyz_rpy(pose)
    elif rotation_representation == RotationRepresentation.rpy:
        return pose
    elif rotation_representation == RotationRepresentation.quaternion:
        xyz, quat = pose[:, :3], pose[:, 3:]

        quat = torch.nn.functional.normalize(quat, p=2, dim=1)

        rpy = torch_3d_utils.quaternion_to_rpy(quat)
        return torch.cat([xyz, rpy], dim=1)
    elif rotation_representation == RotationRepresentation.rotation_matrix:
        xyz, rotmat = pose[:, :3], pose[:, 3:]
        rotmat = rotmat.view(-1, 3, 3)

        rotmat = proper_svd(rotmat)

        rpy = torch_3d_utils.rotation_matrix_to_rpy(rotmat)
        return torch.cat([xyz, rpy], dim=1)

    elif rotation_representation == RotationRepresentation.axis_angle:

        xyz, axis_angle = pose[:, :3], pose[:, 3:]
        rotmat = axis_angle_to_matrix(axis_angle)

        rpy = torch_3d_utils.rotation_matrix_to_rpy(rotmat)
        return torch.cat([xyz, rpy], dim=1)
    else:
        raise NotImplementedError(
            f"Unsupported rotation representation: {rotation_representation}")


def make_process_batch_fn(
        use_keypoints: bool,
        q_noise_var: Optional[float],
        hand_model_keypoints: Optional[HandModelKeypoints],
        x0_normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]],
        rot_augmentation: bool,
        device,
        rotation_representation: RotationRepresentation,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    def process_batch_fn(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if device is not None:
            for key in batch:
                batch[key] = batch[key].to(device)

        qpos = batch["qpos"]  # [B, 16]
        if q_noise_var is not None:
            qpos = qpos + torch.randn_like(qpos) * q_noise_var
        batch["qpos"] = qpos

        if rot_augmentation:

            object_0_points = batch["object_0_points"]  # [B, N, 3]
            object_1_points = batch["object_1_points"]  # [B, N, 3]

            batch_size = object_0_points.shape[0]

            rand_rot_0 = _make_random_rotation_matrix_z(
                batch_size, device)  # [B, 3, 3]
            rand_rot_1 = _make_random_rotation_matrix_z(
                batch_size, device)  # [B, 3, 3]

            object_0_points_aug = torch.matmul(
                object_0_points, rand_rot_0.transpose(1, 2))
            object_1_points_aug = torch.matmul(
                object_1_points, rand_rot_1.transpose(1, 2))
            batch["object_0_points"] = object_0_points_aug
            batch["object_1_points"] = object_1_points_aug

            object_0_pose = batch["object_0_pose"]  # [B, 6]
            object_1_pose = batch["object_1_pose"]  # [B, 6]

            obj0_trans = object_0_pose[:, :3]  # [B, 3]
            obj0_rpy = object_0_pose[:, 3:6]   # [B, 3]
            obj0_rot = torch_3d_utils.rpy_to_rotation_matrix(
                obj0_rpy)  # [B, 3, 3]
            new_obj0_trans = torch.bmm(obj0_trans.unsqueeze(
                1), rand_rot_0.transpose(1, 2)).squeeze(1)  # [B, 3]
            new_obj0_rot = torch.bmm(rand_rot_0, obj0_rot)  # [B, 3, 3]
            new_obj0_rpy = torch_3d_utils.rotation_matrix_to_rpy(
                new_obj0_rot)  # [B, 3]
            object_0_pose_aug = torch.cat(
                [new_obj0_trans, new_obj0_rpy], dim=1)  # [B, 6]

            obj1_trans = object_1_pose[:, :3]  # [B, 3]
            obj1_rpy = object_1_pose[:, 3:6]   # [B, 3]
            obj1_rot = torch_3d_utils.rpy_to_rotation_matrix(
                obj1_rpy)  # [B, 3, 3]
            new_obj1_trans = torch.bmm(obj1_trans.unsqueeze(
                1), rand_rot_1.transpose(1, 2)).squeeze(1)  # [B, 3]
            new_obj1_rot = torch.bmm(rand_rot_1, obj1_rot)  # [B, 3, 3]
            new_obj1_rpy = torch_3d_utils.rotation_matrix_to_rpy(
                new_obj1_rot)  # [B, 3]
            object_1_pose_aug = torch.cat(
                [new_obj1_trans, new_obj1_rpy], dim=1)  # [B, 6]

            batch["object_0_pose"] = object_0_pose_aug
            batch["object_1_pose"] = object_1_pose_aug

        object_0_pose_representation = _convert_pose(
            batch["object_0_pose"], rotation_representation)
        object_1_pose_representation = _convert_pose(
            batch["object_1_pose"], rotation_representation)

        if use_keypoints:
            assert hand_model_keypoints is not None
            keypoints = hand_model_keypoints.compute_keypoints(
                qpos)  # [B, 16, 3]
            keypoints_flattened = keypoints.flatten(start_dim=1)  # [B, 48]
            x0 = torch.cat([object_0_pose_representation, object_1_pose_representation, keypoints_flattened],
                           dim=1)  # [B, 9+9+48]
        else:
            x0 = torch.cat([object_0_pose_representation, object_1_pose_representation, qpos],
                           dim=1)  # [B, 9+9+16]

        if x0_normalizer is not None:
            x0 = x0_normalizer(x0)

        batch["x0"] = x0

        return batch

    return process_batch_fn


def _make_random_rotation_matrix_z(batch_size: int, device) -> torch.Tensor:
    rand_rpy = torch.zeros(batch_size, 3, device=device)
    rand_rpy[:, 2] = 2 * np.pi * torch.rand(batch_size, device=device)
    rand_rot = torch_3d_utils.rpy_to_rotation_matrix(rand_rpy)  # [B, 3, 3]
    return rand_rot


def _make_x0_mean_std(stats_path: str,
                      use_keypoints: bool,
                      device,
                      rotation_representation: RotationRepresentation
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    """

    stats = np.load(stats_path)

    obj0_xyz_mean = torch.tensor(
        stats['obj0_xyz_mean'], dtype=torch.float32, device=device)
    obj0_xyz_std = torch.tensor(
        stats['obj0_xyz_std'],  dtype=torch.float32, device=device)
    obj1_xyz_mean = torch.tensor(
        stats['obj1_xyz_mean'], dtype=torch.float32, device=device)
    obj1_xyz_std = torch.tensor(
        stats['obj1_xyz_std'],  dtype=torch.float32, device=device)

    if rotation_representation == RotationRepresentation.rot6d:
        obj0_rot_mean = torch.tensor(
            stats['obj0_rot6d_mean'], dtype=torch.float32, device=device)
        obj0_rot_std = torch.tensor(
            stats['obj0_rot6d_std'],  dtype=torch.float32, device=device)
        obj1_rot_mean = torch.tensor(
            stats['obj1_rot6d_mean'], dtype=torch.float32, device=device)
        obj1_rot_std = torch.tensor(
            stats['obj1_rot6d_std'],  dtype=torch.float32, device=device)

    elif rotation_representation == RotationRepresentation.rpy:
        obj0_rot_mean = torch.tensor(
            stats['obj0_rpy_mean'], dtype=torch.float32, device=device)
        obj0_rot_std = torch.tensor(
            stats['obj0_rpy_std'],  dtype=torch.float32, device=device)
        obj1_rot_mean = torch.tensor(
            stats['obj1_rpy_mean'], dtype=torch.float32, device=device)
        obj1_rot_std = torch.tensor(
            stats['obj1_rpy_std'],  dtype=torch.float32, device=device)

    elif rotation_representation == RotationRepresentation.quaternion:
        obj0_rot_mean = torch.tensor(
            stats['obj0_quat_mean'], dtype=torch.float32, device=device)
        obj0_rot_std = torch.tensor(
            stats['obj0_quat_std'],  dtype=torch.float32, device=device)
        obj1_rot_mean = torch.tensor(
            stats['obj1_quat_mean'], dtype=torch.float32, device=device)
        obj1_rot_std = torch.tensor(
            stats['obj1_quat_std'],  dtype=torch.float32, device=device)

    elif rotation_representation == RotationRepresentation.rotation_matrix:
        obj0_rot_mean = torch.tensor(
            stats['obj0_rot_mat_mean'], dtype=torch.float32, device=device)
        obj0_rot_std = torch.tensor(
            stats['obj0_rot_mat_std'],  dtype=torch.float32, device=device)
        obj1_rot_mean = torch.tensor(
            stats['obj1_rot_mat_mean'], dtype=torch.float32, device=device)
        obj1_rot_std = torch.tensor(
            stats['obj1_rot_mat_std'],  dtype=torch.float32, device=device)
    elif rotation_representation == RotationRepresentation.axis_angle:
        obj0_rot_mean = torch.tensor(
            stats['obj0_axis_angle_mean'], dtype=torch.float32, device=device)
        obj0_rot_std = torch.tensor(
            stats['obj0_axis_angle_std'],  dtype=torch.float32, device=device)
        obj1_rot_mean = torch.tensor(
            stats['obj1_axis_angle_mean'], dtype=torch.float32, device=device)
        obj1_rot_std = torch.tensor(
            stats['obj1_axis_angle_std'],  dtype=torch.float32, device=device)
    else:
        raise NotImplementedError(
            f"Unsupported rotation representation: {rotation_representation}")

    if not use_keypoints:
        qpos_mean = torch.tensor(
            stats['qpos_mean'], dtype=torch.float32, device=device)
        qpos_std = torch.tensor(
            stats['qpos_std'],  dtype=torch.float32, device=device)

        x0_mean = torch.cat([obj0_xyz_mean, obj0_rot_mean,
                             obj1_xyz_mean, obj1_rot_mean,
                             qpos_mean], dim=0)
        x0_std = torch.cat([obj0_xyz_std, obj0_rot_std,
                            obj1_xyz_std, obj1_rot_std,
                            qpos_std], dim=0)
    else:
        keypoints_xyz_mean = torch.tensor(
            stats['keypoints_xyz_mean'], dtype=torch.float32, device=device)
        keypoints_xyz_std = torch.tensor(
            stats['keypoints_xyz_std'],  dtype=torch.float32, device=device)

        x0_mean = torch.cat([obj0_xyz_mean, obj0_rot_mean,
                             obj1_xyz_mean, obj1_rot_mean,
                             keypoints_xyz_mean], dim=0)
        x0_std = torch.cat([obj0_xyz_std, obj0_rot_std,
                            obj1_xyz_std, obj1_rot_std,
                            keypoints_xyz_std], dim=0)

    return x0_mean, x0_std


def make_x0_normalizer(stats_path: str, use_keypoints: bool, device, rotation_representation: RotationRepresentation = RotationRepresentation.rot6d,) -> Callable[[torch.Tensor], torch.Tensor]:
    x0_mean, x0_std = _make_x0_mean_std(
        stats_path, use_keypoints, device, rotation_representation)

    def x0_normalizer(x0: torch.Tensor) -> torch.Tensor:
        return (x0 - x0_mean) / x0_std

    return x0_normalizer


def make_x0_unnormalizer(stats_path: str, use_keypoints: bool, device, rotation_representation: RotationRepresentation = RotationRepresentation.rot6d,) -> Callable[[torch.Tensor], torch.Tensor]:
    x0_mean, x0_std = _make_x0_mean_std(
        stats_path, use_keypoints, device, rotation_representation)

    def x0_unnormalizer(x0: torch.Tensor) -> torch.Tensor:
        return x0 * x0_std + x0_mean

    return x0_unnormalizer
