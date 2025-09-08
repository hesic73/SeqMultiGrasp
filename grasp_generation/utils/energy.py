"""
Last modified date: 2022.03.11
Author: mzhmxzh
Description: energy functions
"""

import torch

from .hand_model import HandModel
from .object_model import ObjectModel
from .single_object_model import SingleObjectModel

from typing import Optional


def cal_energy(hand_model: HandModel,
               object_model: SingleObjectModel,
               w_dis: float = 100.0,
               w_pen: float = 100.0,
               w_spen: float = 10.0,
               w_joints: float = 1.0,
               levitate: bool = False,  # if False, hand-table penetration is considered
               object_min_y: Optional[torch.Tensor] = None,
               ground_offset: float = 0.01,
               verbose: bool = False):

    # E_dis
    # (batch_size, n_contact, 3)
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(
        hand_model.contact_points)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                         dtype=torch.float, device=device)
    g = torch.cat([torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
                   (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)],
                  dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc: torch.Tensor = norm * norm

    # E_joints
    E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
        torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (
            hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)

    # E_pen
    object_scale = object_model.scale_factor.unsqueeze(1).unsqueeze(2)

    object_surface_points = object_model.surface_points_tensor * \
        object_scale  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    if not levitate:
        hand_verts = hand_model.get_surface_points()  # (batch_size, n_surface_points, 3)
        hand_verts_height = hand_verts[:, :, 1]
        assert object_min_y is not None

        table_penetration_energy = torch.relu(
            object_min_y.unsqueeze(-1)+ground_offset-hand_verts_height).sum(dim=-1).sum(dim=-1)  # (batch_size,)

        E_pen += table_penetration_energy

    # E_spen
    E_spen = hand_model.self_penetration()

    E_total: torch.Tensor = E_fc + w_dis * E_dis + w_pen * \
        E_pen + w_spen * E_spen + w_joints * E_joints

    if verbose:
        return E_total, E_fc, E_dis, E_pen, E_spen, E_joints
    else:
        return E_total
