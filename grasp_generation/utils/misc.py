import numpy as np

import torch
import transforms3d

import os


from src.utils import torch_3d_utils

from utils import rot6d

from typing import Dict, List, Sequence, Tuple, Any, Optional

from src.consts import MESHDATA_PATH

import trimesh


def inverse_pose(translation: Sequence[float], rpy: Sequence[float]) -> Tuple[List[float], List[float]]:
    """
    Compute the inverse of a pose given its translation and RPY.

    Args:
        translation (list): Translation vector [x, y, z].
        rpy (list): RPY angles [roll, pitch, yaw].

    Returns:
        tuple: (inverse_translation, inverse_rpy)
    """
    # Convert RPY to rotation matrix
    rotation_matrix = transforms3d.euler.euler2mat(
        rpy[0], rpy[1], rpy[2], axes='sxyz')

    # Compute the inverse rotation
    inverse_rot = rotation_matrix.T

    # Compute the inverse translation
    inverse_translation = np.dot(-inverse_rot, translation)

    # Convert the inverse rotation matrix back to RPY
    inverse_rpy = transforms3d.euler.mat2euler(inverse_rot, axes='sxyz')

    return inverse_translation.tolist(), inverse_rpy


def make_contact_candidates_weight(
    contact_candidates: Dict[str, List[List[float]]],
    active_links: Optional[List[str]] = None,
    link_base_weight: Optional[Dict[str, float]] = None
):
    contact_candidates_weight = []

    for k, v in contact_candidates.items():
        for _ in v:
            if active_links is None or k in active_links:
                w = 1.0 if link_base_weight is None else link_base_weight.get(
                    k, 1.0)
                contact_candidates_weight.append(w / len(v))
            else:
                contact_candidates_weight.append(0.0)

    contact_candidates_weight = torch.tensor(
        contact_candidates_weight, dtype=torch.float)  # shape (n_contact_candidates,)

    # normalize the weights
    contact_candidates_weight /= contact_candidates_weight.sum()

    return contact_candidates_weight


_translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
_rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
_joint_names = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',
    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0',
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0',
    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
]


def get_translation_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _translation_names]


def get_rpy_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _rot_names]


def get_joint_positions_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _joint_names]


def extract_hand_pose_and_scale_tensor(
    data_dict: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size = data_dict.shape[0]

    hand_state = []
    scale_tensor = []

    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']

        translation = get_translation_from_qpos(qpos)
        rotation_rpy = get_rpy_from_qpos(qpos)
        joints = get_joint_positions_from_qpos(qpos)

        rot_matrix = transforms3d.euler.euler2mat(*rotation_rpy)
        rot_6d = rot_matrix[:, :2].T.ravel().tolist()

        hand_pose = torch.tensor(
            translation + rot_6d + joints, dtype=torch.float, device=device
        )
        hand_state.append(hand_pose)
        scale_tensor.append(scale)

    hand_state = torch.stack(hand_state).to(device)
    scale_tensor = torch.tensor(
        scale_tensor, dtype=torch.float, device=device)

    return hand_state, scale_tensor


def extract_hand_pose_and_scale_tensor_two_objects(
        data_dict: np.ndarray,
        device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = data_dict.shape[0]

    hand_state = []
    scale_tensor = []

    for i in range(batch_size):
        object_0_qpos = data_dict[i]['object_0_qpos']
        object_0_joints = np.array(
            get_joint_positions_from_qpos(object_0_qpos))

        object_1_qpos = data_dict[i]['object_1_qpos']
        object_1_scale = data_dict[i]['object_1_scale']

        translation = get_translation_from_qpos(object_1_qpos)
        rotation_rpy = get_rpy_from_qpos(object_1_qpos)
        joints = np.array(
            get_joint_positions_from_qpos(object_1_qpos))

        active_joint_mask: np.ndarray = data_dict[i]['active_joint_mask']

        active_joint_mask = active_joint_mask.astype(bool)

        joints[~active_joint_mask] = object_0_joints[~active_joint_mask]

        joints = joints.tolist()

        rot_matrix = transforms3d.euler.euler2mat(*rotation_rpy)
        rot_6d = rot_matrix[:, :2].T.ravel().tolist()

        hand_pose = torch.tensor(
            translation + rot_6d + joints, dtype=torch.float, device=device
        )
        hand_state.append(hand_pose)
        scale_tensor.append(object_1_scale)

    hand_state = torch.stack(hand_state).to(device)
    scale_tensor = torch.tensor(
        scale_tensor, dtype=torch.float, device=device).reshape(1, -1)

    return hand_state, scale_tensor


def hand_pose_to_dict(hand_pose: torch.Tensor) -> Dict[str, float]:
    assert len(hand_pose.shape) == 1
    assert hand_pose.shape[0] == 25

    hand_pose = hand_pose.detach().cpu()

    qpos = dict(zip(_joint_names, hand_pose[9:].tolist()))
    rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(
        hand_pose[3:9].unsqueeze(0))[0]
    euler = transforms3d.euler.mat2euler(rot, axes='sxyz')
    qpos.update(dict(zip(_rot_names, euler)))
    qpos.update(dict(zip(_translation_names, hand_pose[:3].tolist())))

    return qpos


def dict_to_hand_pose(qpos: Dict[str, float], device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Convert a dictionary containing joint positions, rotation (RPY), and translation back into a hand pose tensor.

    Args:
        qpos (dict): Dictionary containing joint positions, rotation (RPY), and translation.
        device (torch.device): Target device for the resulting tensor.

    Returns:
        torch.Tensor: Hand pose tensor with structure [translation, rotation (6D), joint positions].
    """
    # Extract translation
    translation = [qpos[name] for name in _translation_names]

    # Extract rotation matrix from RPY and convert it to 6D representation
    rpy = [qpos[name] for name in _rot_names]
    rot_matrix = transforms3d.euler.euler2mat(*rpy, axes='sxyz')
    rot_6d = rot_matrix[:, :2].T.ravel().tolist()

    # Extract joint positions
    joints = [qpos[name] for name in _joint_names]

    # Construct the hand pose tensor
    hand_pose = torch.tensor(
        translation + rot_6d + joints, dtype=torch.float, device=device
    )

    return hand_pose


def convert_xyz_rpy_to_xyz_rot6d(pose: torch.Tensor) -> torch.Tensor:
    """Convert pose from xyz+rpy to xyz+rot6d

    Args:
        pose (torch.Tensor): pose tensor with shape (B, 6). xyz+rpy

    Returns:
        torch.Tensor: pose tensor with shape (B, 9)
    """

    xyz, rpy = pose[:, :3], pose[:, 3:]

    rotmat: torch.Tensor = torch_3d_utils.rpy_to_rotation_matrix(rpy)

    rot6d_tensor = rotmat[:, :, :2].transpose(1, 2).contiguous().view(-1, 6)

    return torch.cat([xyz, rot6d_tensor], dim=1)


def convert_xyz_rot6d_to_xyz_rpy(pose: torch.Tensor) -> torch.Tensor:
    """Convert pose from xyz+rot6d to xyz+rpy


    Args:
        pose (torch.Tensor): pose tensor with shape (B, 9). xyz+rot6d

    Returns:
        torch.Tensor: pose tensor with shape (B, 6). xyz+rpy
    """

    xyz, rot6d_tensor = pose[:, :3], pose[:, 3:]

    rotmat = rot6d.robust_compute_rotation_matrix_from_ortho6d(
        rot6d_tensor)

    rpy = torch_3d_utils.rotation_matrix_to_rpy(rotmat)

    return torch.cat([xyz, rpy], dim=1)
