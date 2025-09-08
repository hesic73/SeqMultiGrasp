import torch
import numpy as np


from typing import Dict, List, Tuple, Union, Optional, Sequence

from mani_skill.utils.structs import Link,  Actor, SimConfig, GPUMemoryConfig, SceneConfig, Pose
from mani_skill.utils.registration import register_env

from src.envs.utils.actor_utils import add_actor_from_file, add_actor_from_file_with_varied_scale
from src.envs.utils.actor_builder import VariedActorBuilder
from src.utils import torch_3d_utils
from src.envs.validator.common import _MAX_EPISODE_STEPS, ValidatorBase

from src.envs.utils.misc import reorder_joint_indices
from src.consts import _DENSITY, _FRICTION_COEFFICIENT


from sapien.physx import PhysxMaterial


from src.consts import get_object_mesh_path


@register_env("PreGraspValidator-v0", max_episode_steps=_MAX_EPISODE_STEPS)
class PreGraspValidatorV0(ValidatorBase):

    def __init__(self,
                 object_name: str,
                 hand_state: torch.Tensor,
                 object_scale: Optional[Sequence[float]] = None,
                 *args,
                 **kwargs):

        self.object_name = object_name
        self.object_scale = object_scale

        self.hand_state = hand_state

        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        initial_pose = Pose.create_from_pq()
        file_path = get_object_mesh_path(self.object_name)

        # builder = self.scene.create_actor_builder()

        # physical_material = PhysxMaterial(
        #     static_friction=0.2, dynamic_friction=0.2, restitution=0.0)

        # self.object = add_actor_from_file(
        #     builder, file_path, self.object_name, scale=1.0,
        #     physical_material=physical_material,
        #     initial_pose=initial_pose,
        #     color=[1, 0, 0])

        builder = VariedActorBuilder()
        builder.set_scene(self.scene)

        physical_material = PhysxMaterial(
            static_friction=_FRICTION_COEFFICIENT, dynamic_friction=_FRICTION_COEFFICIENT, restitution=0.0)

        self.object = add_actor_from_file_with_varied_scale(
            builder, file_path, self.object_name, scale=self.object_scale,
            physical_material=physical_material,
            initial_pose=initial_pose,
            color=[1, 0, 0],
            density=_DENSITY,
        )

    def _first_initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._first_initialize_episode(env_idx, options)

        assert self.hand_state.shape == (self.num_envs, 3+3+16)
        self.hand_state = self.hand_state.to(self.device)

        xyz, rpy, qpos = torch.split(self.hand_state, [3, 3, 16], dim=-1)

        quat = torch_3d_utils.rpy_to_quaternion(rpy)

        hand_pose = Pose.create_from_pq(p=xyz, q=quat)

        # NOTE (hsc): Here we assume that the object is initially at the origin
        self._initial_object_pose_in_hand_frame = hand_pose.inv()

        joint_names = [joint.get_name()
                       for joint in self.agent.robot.get_active_joints()]

        self.joint_indices = reorder_joint_indices(joint_names)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        super()._initialize_episode(env_idx, options)
        # set object pose
        self.object.set_pose(Pose.create_from_pq())

        # set hand pose

        hand_state = self.hand_state[env_idx]

        xyz, rpy, qpos = torch.split(hand_state, [3, 3, 16], dim=-1)

        quat = torch_3d_utils.rpy_to_quaternion(rpy)

        self.agent.robot.set_pose(Pose.create_from_pq(
            p=xyz, q=quat))

        qpos = qpos[:, self.joint_indices]
        self.agent.robot.set_qpos(qpos)

    def evaluate(self, **kwargs) -> dict:

        last_step = (self._episode_steps ==
                     _MAX_EPISODE_STEPS).squeeze(-1)  # (n,)
        success = self._is_successful(
            self.object, self._initial_object_pose_in_hand_frame) & last_step

        fail = ~success & last_step

        # joint 0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15
        qpos = self.agent.robot.get_qpos()

        hand_pose = self.agent.robot.get_pose()
        object_pose = self.object.pose

        object_pose_in_hand_frame: Pose = hand_pose.inv()*object_pose

        return {
            'success': success,
            'fail': fail,
            'qpos': qpos,
            'object_pose_in_hand_frame': object_pose_in_hand_frame.raw_pose,
        }
