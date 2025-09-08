import torch
import torch.nn as nn

import random
import os


from . import rot6d_utils, torch_3d_utils


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def convert_rpy_to_rot6d(rpy: torch.Tensor) -> torch.Tensor:
    """Convert rpy to rot6d

    Args:
        rpy (torch.Tensor): rpy tensor with shape (B, 3). rpy

    Returns:
        torch.Tensor: rot6d tensor with shape (B, 6)
    """

    rotmat: torch.Tensor = torch_3d_utils.rpy_to_rotation_matrix(rpy)

    rot6d_tensor = rotmat[:, :, :2].transpose(1, 2).contiguous().view(-1, 6)

    return rot6d_tensor


def convert_xyz_rot6d_to_xyz_rpy(pose: torch.Tensor) -> torch.Tensor:
    """Convert pose from xyz+rot6d to xyz+rpy


    Args:
        pose (torch.Tensor): pose tensor with shape (B, 9). xyz+rot6d

    Returns:
        torch.Tensor: pose tensor with shape (B, 6). xyz+rpy
    """

    xyz, rot6d_tensor = pose[:, :3], pose[:, 3:]

    rotmat = rot6d_utils.robust_compute_rotation_matrix_from_ortho6d(
        rot6d_tensor)

    rpy = torch_3d_utils.rotation_matrix_to_rpy(rotmat)

    return torch.cat([xyz, rpy], dim=1)


class AverageMeter:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value: float):
        self._sum += value
        self._count += 1

    def reset(self):
        self._sum = 0
        self._count = 0

    def avg(self):
        return self._sum / self._count


def load_checkpoint(model: nn.Module, ckpt_path: str):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict['model'])
    print(f"Checkpoint loaded from: {ckpt_path}")
