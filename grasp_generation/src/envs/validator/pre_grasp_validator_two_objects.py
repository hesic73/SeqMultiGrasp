import torch
import numpy as np


from typing import Dict, List, Tuple, Union, Optional, Sequence

from sapien.physx import PhysxMaterial

from mani_skill.utils.structs import Link,  Actor, SimConfig, GPUMemoryConfig, SceneConfig, Pose
from mani_skill.utils.registration import register_env

from src.envs.utils.actor_builder import VariedActorBuilder
from src.envs.utils.actor_utils import add_actor_from_file, add_actor_from_file_with_varied_scale
from src.utils import torch_3d_utils
from src.envs.validator.common import _MAX_EPISODE_STEPS, ValidatorBase

from src.envs.utils.misc import reorder_joint_indices
from src.consts import _DENSITY, _FRICTION_COEFFICIENT

import os


from src.consts import get_object_mesh_path


@register_env("PreGraspValidatorTwoObjects-v0", max_episode_steps=_MAX_EPISODE_STEPS)
class PreGraspValidatorTwoObjectsV0(ValidatorBase):

    def __init__(self,
                 object_0_name: str,
                 object_1_name: str,
                 hand_state: torch.Tensor,
                 object_0_pose_in_hand_frame: torch.Tensor,
                 object_0_scale: Optional[Sequence[float]] = None,
                 object_1_scale: Optional[Sequence[float]] = None,
                 *args,
                 **kwargs):

        self.object_0_name = object_0_name
        self.object_1_name = object_1_name

        self.object_0_scale = object_0_scale
        self.object_1_scale = object_1_scale

        self.hand_state = hand_state

        self._initial_object_0_pose_in_hand_frame = object_0_pose_in_hand_frame

        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):

        # placeholder
        initial_pose_object_0 = Pose.create_from_pq(p=[1, 0, 0])

        physical_material = PhysxMaterial(
            static_friction=_FRICTION_COEFFICIENT, dynamic_friction=_FRICTION_COEFFICIENT, restitution=0.0)

        file_path_object_0 = get_object_mesh_path(self.object_0_name)
        builder_object_0 = VariedActorBuilder()
        builder_object_0.set_scene(self.scene)

        self.object_0 = add_actor_from_file_with_varied_scale(
            builder_object_0, file_path_object_0, self.object_0_name, scale=self.object_0_scale,
            physical_material=physical_material,
            initial_pose=initial_pose_object_0,
            color=[1, 0, 0],
            density=_DENSITY,
        )

        initial_pose_object_1 = Pose.create_from_pq()

        file_path_object_1 = get_object_mesh_path(self.object_1_name)
        builder_object_1 = VariedActorBuilder()
        builder_object_1.set_scene(self.scene)

        self.object_1 = add_actor_from_file_with_varied_scale(
            builder_object_1, file_path_object_1, self.object_1_name, scale=self.object_1_scale,
            physical_material=physical_material,
            initial_pose=initial_pose_object_1,
            color=[0, 1, 0],
            density=_DENSITY,
        )

    def _first_initialize_episode(self, env_idx: torch.Tensor, options: dict):

        super()._first_initialize_episode(env_idx, options)
        assert self.hand_state.shape == (self.num_envs, 3+3+16)
        self.hand_state = self.hand_state.to(self.device)

        assert self._initial_object_0_pose_in_hand_frame.shape == (
            self.num_envs, 7)
        self._initial_object_0_pose_in_hand_frame = Pose(
            self._initial_object_0_pose_in_hand_frame.to(self.device))

        self._episode_steps = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.int32)  # (n, 1)

        xyz, rpy, qpos = torch.split(self.hand_state, [3, 3, 16], dim=-1)

        quat = torch_3d_utils.rpy_to_quaternion(rpy)

        hand_pose = Pose.create_from_pq(p=xyz, q=quat)

        # NOTE (hsc): Here we assume that the object is initially at the origin
        self._initial_object_1_pose_in_hand_frame = hand_pose.inv()

        joint_names = [joint.get_name()
                       for joint in self.agent.robot.get_active_joints()]

        self.joint_indices = reorder_joint_indices(joint_names)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        self.object_1.set_pose(Pose.create_from_pq())

        hand_state = self.hand_state[env_idx]

        xyz, rpy, qpos = torch.split(hand_state, [3, 3, 16], dim=-1)

        quat = torch_3d_utils.rpy_to_quaternion(rpy)

        self.agent.robot.set_pose(Pose.create_from_pq(
            p=xyz, q=quat))

        qpos = qpos[:, self.joint_indices]
        self.agent.robot.set_qpos(qpos)


        object_0_pose_in_hand_frame = self._initial_object_0_pose_in_hand_frame[env_idx]
        robot_pose = self.agent.robot.get_pose()[env_idx]
        object_0_pose = robot_pose * object_0_pose_in_hand_frame

        self.object_0.set_pose(object_0_pose)

    def evaluate(self, **kwargs) -> dict:
        last_step = (self._episode_steps ==
                     _MAX_EPISODE_STEPS).squeeze(-1)  # (n,)
        success = self._is_successful(
            self.object_0, self._initial_object_0_pose_in_hand_frame) & self._is_successful(self.object_1, self._initial_object_1_pose_in_hand_frame) & last_step
        fail = ~success & last_step

        # joint 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15
        qpos = self.agent.robot.get_qpos()

        hand_pose = self.agent.robot.get_pose()
        object_0_pose = self.object_0.pose

        object_0_pose_in_hand_frame: Pose = hand_pose.inv()*object_0_pose

        object_1_pose = self.object_1.pose

        object_1_pose_in_hand_frame: Pose = hand_pose.inv()*object_1_pose

        return {
            'success': success,
            'fail': fail,
            'qpos': qpos,
            'object_0_pose_in_hand_frame': object_0_pose_in_hand_frame.raw_pose,
            'object_1_pose_in_hand_frame': object_1_pose_in_hand_frame.raw_pose,

        }
