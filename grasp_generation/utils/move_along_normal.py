
import numpy as np
import torch


from utils.hand_model import HandModel
from utils.hand_model_with_object import HandModelWithObject
from utils.object_model import ObjectModel
from utils.single_object_model import SingleObjectModel


from typing import Dict, List, Union


def move_along_normal(
    hand_model: Union[HandModel, HandModelWithObject],
    object_model: Union[ObjectModel, SingleObjectModel],
    hand_pose: torch.Tensor,
    device: torch.device,
    thres_cont: float,
    dis_move: float,
    grad_move: float
):
    """
    Reference:
    - DexGraspNet: https://arxiv.org/abs/2210.02697
    - https://github.com/PKU-EPIC/DexGraspNet/blob/bdb423c37d7e90e395e9d07cdd8d6aaa793bc06b/grasp_generation/scripts/validate_grasps.py#L97-L126

    """

    assert len(hand_pose.shape) == 2
    assert hand_pose.shape[1] == 25

    batch_size = hand_pose.shape[0]

    assert not hand_pose.requires_grad

    hand_pose.requires_grad_(True)

    hand_model.set_parameters(hand_pose)

    n_links = len(hand_model.mesh)
    contact_points_hand = torch.zeros((batch_size, n_links, 3)).to(device)
    contact_normals = torch.zeros((batch_size, n_links, 3)).to(device)

    for i, link_name in enumerate(hand_model.mesh):
        if len(hand_model.mesh[link_name]['surface_points']) == 0:
            continue

        surface_points = hand_model.current_status[link_name].transform_points(
            hand_model.mesh[link_name]['surface_points']).expand(batch_size, -1, 3)
        surface_points = surface_points @ hand_model.global_rotation.transpose(
            1, 2) + hand_model.global_translation.unsqueeze(1)

        distances, normals = object_model.cal_distance(
            surface_points)
        nearest_point_index = distances.argmax(dim=1)
        nearest_distances = torch.gather(
            distances, 1, nearest_point_index.unsqueeze(1))
        nearest_points_hand = torch.gather(
            surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
        nearest_normals = torch.gather(
            normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
        admited = -nearest_distances < thres_cont
        admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points_hand[:, i:i+1, :] = torch.where(
            admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])
        contact_normals[:, i:i+1, :] = torch.where(
            admited, nearest_normals, contact_normals[:, i:i+1, :])

    target_points = contact_points_hand + contact_normals * dis_move
    loss = (target_points.detach().clone() -
            contact_points_hand).square().sum()
    loss.backward()

    with torch.no_grad():
        hand_pose[:, 9:] += hand_pose.grad[:, 9:] * grad_move
        hand_pose.grad.zero_()

    hand_pose.requires_grad_(False)

    return hand_pose
