from .base import TwoObjectsBase

from typing import Any, Dict, List, Optional, Union, Callable

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

from sapien.physx import PhysxMaterial


import sapien


from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose, Actor, SimConfig, GPUMemoryConfig

from src.envs.utils.actor_builder import VariedActorBuilder


from mani_skill.utils.registration import register_env

from src.consts import get_object_mesh_path

from gymnasium.envs.registration import WrapperSpec

from .base import TwoObjectsBase

from typing import Sequence, List

from src.envs.utils.misc import reorder_joint_indices


from src.consts import _DENSITY, _FRICTION_COEFFICIENT


# TODO (hsc): scale of the objects
@register_env("TwoObjects-v0")
class TwoObjectsV0(TwoObjectsBase):

    def __init__(
        self,
        *args,
        object_0_name: str,
        object_1_name: str,
        object_0_xy: Sequence[float],
        object_1_xy: Sequence[float],
        initial_joint_positions: Optional[List[float]] = None,
        **kwargs,
    ):

        self.object_0_name = object_0_name
        self.object_1_name = object_1_name

        self.object_0_xy = object_0_xy
        self.object_1_xy = object_1_xy

        self.object_0_quat = None
        self.object_1_quat = None

        self.initial_joint_positions = initial_joint_positions

        assert self.initial_joint_positions is None or (isinstance(
            self.initial_joint_positions, list) and len(self.initial_joint_positions) == 23)

        super().__init__(
            *args,
            **kwargs,
        )

    def _make_object_0(self) -> Actor:
        physical_material = PhysxMaterial(
            static_friction=_FRICTION_COEFFICIENT,
            dynamic_friction=_FRICTION_COEFFICIENT,
            restitution=0.0,
        )

        builder = VariedActorBuilder()
        builder.set_scene(self.scene)

        stl_file_path = get_object_mesh_path(self.object_0_name)

        builder.add_convex_collision_from_file(
            stl_file_path,
            material=physical_material,
            density=_DENSITY,
        )

        glb_file_path = stl_file_path.replace('.stl', '.glb')
        if os.path.exists(glb_file_path):
            builder.add_visual_from_file(
                glb_file_path)
        else:
            builder.add_visual_from_file(
                stl_file_path)

        actor = builder.build_with_variation(
            name=f"{self.object_0_name}_0",
        )

        return actor

    def _make_object_1(self) -> Actor:

        physical_material = PhysxMaterial(
            static_friction=_FRICTION_COEFFICIENT,
            dynamic_friction=_FRICTION_COEFFICIENT,
            restitution=0.0,
        )

        builder = VariedActorBuilder()
        builder.set_scene(self.scene)

        stl_file_path = get_object_mesh_path(self.object_1_name)

        builder.add_convex_collision_from_file(
            stl_file_path,
            material=physical_material,
            density=_DENSITY,
        )

        glb_file_path = stl_file_path.replace('.stl', '.glb')
        if os.path.exists(glb_file_path):
            builder.add_visual_from_file(
                glb_file_path)
        else:
            builder.add_visual_from_file(
                stl_file_path)

        actor = builder.build_with_variation(
            name=f"{self.object_1_name}_1",
        )

        return actor

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            [0.3, 0.3, self.z_top+0.3], [0.0, 0.0, self.z_top+0.1])
        return [CameraConfig("render_camera", pose, 1920, 1440, 1, 0.01, 100)]

    def _first_initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.object_0_xy = torch.tensor(
            self.object_0_xy, device=self.device)  # (2,)
        self.object_1_xy = torch.tensor(
            self.object_1_xy, device=self.device)  # (2,)

        if self.initial_joint_positions is not None:
            hand_joint_names = [joint.get_name()
                                for joint in self.agent.robot.get_active_joints()]
            hand_joint_names = hand_joint_names[7:]

            hand_joint_indices = reorder_joint_indices(hand_joint_names)

            hand_initial_joint_positions = self.initial_joint_positions[7:]
            hand_initial_joint_positions = [
                hand_initial_joint_positions[i] for i in hand_joint_indices]
            self.initial_joint_positions[7:] = hand_initial_joint_positions

        super()._first_initialize_episode(env_idx, options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)

    def _initialize_actors(self, env_idx: torch.Tensor, options: dict):
        self.table_scene.initialize(env_idx)

        b = len(env_idx)

        new_object_0_xyz = torch.zeros((b, 3), device=self.device)
        new_object_0_xyz[:, :2] = self.object_0_xy.unsqueeze(0)
        new_object_0_xyz[:, 2] = self._object_0_z_on_ground[env_idx]

        new_object_1_xyz = torch.zeros((b, 3), device=self.device)
        new_object_1_xyz[:, :2] = self.object_1_xy.unsqueeze(0)
        new_object_1_xyz[:, 2] = self._object_1_z_on_ground[env_idx]

        self.object_0.set_pose(
            Pose.create_from_pq(
                p=new_object_0_xyz,
                q=self.object_0_quat
            )
        )

        self.object_1.set_pose(
            Pose.create_from_pq(
                p=new_object_1_xyz,
                q=self.object_1_quat
            )
        )

    def _initialize_agent(self, env_idx, options):

        if self.control_mode == 'pd_joint_pos':
            initial_joint_positions = self.initial_joint_positions or self.agent.keyframes[
                'rest'].qpos
            self.agent.reset(initial_joint_positions)
        else:
            raise ValueError(
                f"Unsupported control mode: {self.control_mode}")

    def _after_control_step(self):
        super()._after_control_step()

    def evaluate(self, **kwargs) -> Dict:
        info = super().evaluate(**kwargs)

        info.update({
            'object_0_name': self.object_0_name,
            'object_1_name': self.object_1_name,
        })

        return info
