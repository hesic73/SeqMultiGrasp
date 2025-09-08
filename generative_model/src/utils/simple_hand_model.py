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
