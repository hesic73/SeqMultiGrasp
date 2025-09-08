import os
import sys


import numpy as np
import transforms3d
import trimesh as tm
import torch
import yaml

from urdf_parser_py.urdf import Robot
import pytorch_kinematics as pk


from typing import Dict, Tuple, List, Union


class SimpleHandModel:
    def __init__(self, urdf_path: str, device='cpu'):

        self.device = torch.device(device)

        with open(urdf_path, 'r') as f:
            urdf_string = f.read()
        self.chain = pk.build_chain_from_urdf(
            urdf_string).to(dtype=torch.float, device=device)

        self.robot = Robot.from_xml_file(urdf_path)

        revolve_joints = [
            joint for joint in self.robot.joints if joint.joint_type == 'revolute']

        self.joint_names = [joint.name for joint in revolve_joints]

        self.joints_lower = torch.tensor([joint.limit.lower for joint in revolve_joints],
                                         dtype=torch.float, device=device)
        self.joints_upper = torch.tensor([joint.limit.upper for joint in revolve_joints],
                                         dtype=torch.float, device=device)

    def forward_kinematics(self, qpos_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        first_joint = self.joint_names[0]
        device = qpos_dict[first_joint].device
        batch_size = qpos_dict[first_joint].shape[0]
        for jn in self.joint_names:
            assert qpos_dict[jn].shape[0] == batch_size, \
                f"Joint {jn} batch size does not match {first_joint}"
            assert qpos_dict[jn].device == device, \
                f"Joint {jn} is not on the same device as {first_joint}"

        qpos_list = []
        for jn in self.joint_names:
            qpos_list.append(qpos_dict[jn].unsqueeze(1))  # (batch_size, 1)
        qpos = torch.cat(qpos_list, dim=1).to(dtype=torch.float, device=device)

        ts = self.chain.forward_kinematics(qpos)

        link_translations = {}
        link_rotations = {}
        for joint in self.robot.joints:
            link_name = joint.child
            # (batch_size, 4, 4)
            mat = ts[link_name].get_matrix()
            link_translations[link_name] = mat[:,
                                               :3, 3]        # (batch_size, 3)
            # (batch_size, 3, 3)
            link_rotations[link_name] = mat[:, :3, :3]

        return link_translations, link_rotations

    def clamp_qpos(
        self,
        qpos_dict: Dict[str, torch.Tensor],
    ):

        for joint_index, joint_name in enumerate(self.joint_names):
            lower = self.joints_lower[joint_index]
            upper = self.joints_upper[joint_index]
            qpos_dict[joint_name][:] = torch.clamp(
                qpos_dict[joint_name], lower, upper)


class WidthMapper:
    """
    class to map width to robot qpos
    """

    def __init__(
        self,
        hand_model: SimpleHandModel,
        meta_path: str,
    ):
        """
        initialize the class

        Args:
        - robot_model: RobotModel, robot model
        - meta_path: str, path to robot meta file
        """
        self._robot_model: SimpleHandModel = hand_model
        self._robot_meta: dict = yaml.safe_load(open(meta_path, 'r'))

    @property
    def device(self):
        return self._robot_model.device

    def _get_fingertips(
        self,
        link_translations: dict,
        link_rotations: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        get fingertips

        Returns:
        - thumb: torch.Tensor, thumb fingertip, (batch_size, 3)
        - others: torch.Tensor, other fingertips, (batch_size, n_others, 3)
        - thumb_normal: torch.Tensor, thumb normal, (batch_size, 3)
        - other_normals: torch.Tensor, other normals, (batch_size, n_others, 3)
        """
        thumb_link = self._robot_meta['fingertip_link']['thumb']
        other_links = self._robot_meta['fingertip_link']['others']
        n_others = len(other_links)
        # get fingertip
        thumb = link_translations[thumb_link]
        others = torch.stack([link_translations[link]
                             for link in other_links], dim=1)
        device = thumb.device
        # get fingertip normals
        thumb_normal = torch.tensor(self._robot_meta['fingertip_normal']['thumb'],                  # (3,)
                                    dtype=torch.float32, device=device)
        other_normals = torch.tensor(self._robot_meta['fingertip_normal']['others'],                # (n_others, 3)
                                     dtype=torch.float32, device=device)
        # (batch_size, 3, 3)
        thumb_rotation = link_rotations[thumb_link]
        # (batch_size, n_others, 3, 3)
        other_rotations = torch.stack(
            [link_rotations[link] for link in other_links], dim=1)
        thumb_normal = (thumb_rotation @ thumb_normal.reshape(1, 3, 1)
                        ).squeeze(-1)                 # (batch_size, 3)
        other_normals = (other_rotations @ other_normals.reshape(1,
                         n_others, 3, 1)).squeeze(-1)    # (batch_size, n_others, 3)
        return thumb, others, thumb_normal, other_normals

    def squeeze_fingers(
        self,
        qpos_dict: dict,
        delta_width_thumb: float,
        delta_width_others: Union[torch.Tensor, float],
        keep_z: bool = False,
        active_links: list[str] = None,
    ):
        """
        squeeze fingers by a certain amount

        Args:
        - qpos_dict: dict, batched qpos, {joint_name: torch.Tensor, ...}
        - delta_width_thumb: float, amount to squeeze thumb
        - delta_width_others: torch.Tensor or float, amount to squeeze others. If float, applies the same value to all fingertips.
        - keep_z: bool, whether to keep the z component of the normals zero
        - active_links: Optional[List[str]], list of active links whose qpos will be updated

        Returns:
        - squeezed_qpos_dict: dict, squeezed qpos, {joint_name: torch.Tensor, ...}
        - targets: torch.Tensor, fingertip targets, (batch_size, n_fingertips, 3)
        """
        # compute fingertip positions and normals
        link_translations, link_rotations = self._robot_model.forward_kinematics(
            qpos_dict)
        thumb, others, thumb_normal, other_normals = self._get_fingertips(
            link_translations, link_rotations)

        # compute fingertip targets
        if keep_z:
            thumb_normal[..., 2] = 0
            other_normals[..., 2] = 0
            thumb_normal /= thumb_normal.norm(dim=-1, keepdim=True)
            other_normals /= other_normals.norm(dim=-1, keepdim=True)

        thumb_target = thumb + thumb_normal * delta_width_thumb
        if isinstance(delta_width_others, float):
            other_targets = others + other_normals * delta_width_others
        else:
            other_targets = others + other_normals * \
                delta_width_others.unsqueeze(
                    0).unsqueeze(-1)  # handle 1D tensor input

        # Build active joint mask
        if active_links:
            active_mask = torch.tensor([1 if joint_name in active_links else 0 for joint_name in self._robot_model.joint_names],
                                       dtype=torch.float32, device=qpos_dict[next(iter(qpos_dict))].device)
        else:
            active_mask = torch.ones(len(self._robot_model.joint_names), dtype=torch.float32,
                                     device=qpos_dict[next(iter(qpos_dict))].device)

        # optimize towards the targets
        qpos = torch.stack(list(qpos_dict.values()), dim=1)
        qpos.requires_grad = True
        qpos_dict = {joint_name: qpos[:, i]
                     for i, joint_name in enumerate(qpos_dict.keys())}

        for step in range(20):
            link_translations, link_rotations = self._robot_model.forward_kinematics(
                qpos_dict)
            thumb, others, _, _ = self._get_fingertips(
                link_translations, link_rotations)

            loss = torch.sum((thumb - thumb_target) ** 2, dim=1) + \
                torch.sum((others - other_targets) ** 2, dim=[1, 2])
            loss.sum().backward()

            with torch.no_grad():
                qpos -= 20 * (qpos.grad * active_mask)
                qpos.grad.zero_()
                self._robot_model.clamp_qpos(qpos_dict)

        # detach qpos
        qpos = qpos.detach()
        qpos_dict = {joint_name: qpos[:, i]
                     for i, joint_name in enumerate(qpos_dict.keys())}
        # return the optimized robot pose
        targets = torch.cat([thumb_target.unsqueeze(1), other_targets], dim=1)
        return qpos_dict, targets


def make_width_mapper(
        urdf_path: str,
        meta_path: str,
        device='cpu',
):
    """
    make width mapper

    Args:
    - urdf_path: str, path to urdf file
    - meta_path: str, path to meta file
    - device: str, device to use

    Returns:
    - width_mapper: WidthMapper, width mapper
    """
    hand_model = SimpleHandModel(urdf_path, device=device)
    width_mapper = WidthMapper(hand_model, meta_path)
    return width_mapper
