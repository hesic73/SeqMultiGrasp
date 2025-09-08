
from src.envs.utils.actor_utils import build_static_box_with_collision

from src.agents import FrankaAllegoRight


from typing import Any, Dict, List, Optional, Union, Callable

import os
import numpy as np
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Link,  Actor, SimConfig, GPUMemoryConfig, SceneConfig, Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from sapien.physx import PhysxMaterial


class TwoObjectsBase(BaseEnv):
    SUPPORTED_ROBOTS = [FrankaAllegoRight.uid]
    SUPPORTED_REWARD_MODES = ["none"]

    agent: Union[FrankaAllegoRight]

    table_static_friction = 0.5
    table_dynamic_friction = 0.5
    table_restitution = 0.0

    def __init__(
        self,
        *args,
        num_envs=1,
        control_freq: int = 50,
        robot_uid: str = FrankaAllegoRight.uid,
        z_top: float = 0.083,
        **kwargs,
    ):

        self._my_control_freq = control_freq

        self._initialized = False

        self.z_top = z_top

        super().__init__(
            *args,
            robot_uids=robot_uid,
            num_envs=num_envs,
            **kwargs,
        )

    def _after_reconfigure(self, options: dict):

        object_0_collision_mesh = self.object_0.get_first_collision_mesh()
        self._object_0_z = -object_0_collision_mesh.bounding_box.bounds[0, 2]

        object_1_collision_mesh = self.object_1.get_first_collision_mesh()
        self._object_1_z = -object_1_collision_mesh.bounding_box.bounds[0, 2]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            [0.0, 0.3, 0.2], [0.0, 0.0, 0.05])
        return [
            CameraConfig(
                uid="render_camera",
                pose=pose,
                width=1280,
                height=960,
                fov=1,
                near=0.01,
                far=100,
                shader_pack='default',
            )
        ]

    @property
    def _default_sim_config(self):
        gpu_memory_config = GPUMemoryConfig(
            max_rigid_contact_count=self.num_envs *
            max(1024, self.num_envs) * 8,
            max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
            found_lost_pairs_capacity=2**26,
        )
        scene_config = SceneConfig(
            contact_offset=0.02,
            solver_position_iterations=25,
            cpu_workers=min(os.cpu_count(), 4))
        return SimConfig(sim_freq=200,
                         control_freq=self._my_control_freq,
                         scene_config=scene_config,
                         gpu_memory_config=gpu_memory_config)

    def _load_agent(self, options):
        initial_agent_poses = Pose.create_from_pq(p=[-0.5, 0, 0.0])
        return super()._load_agent(options, initial_agent_poses)

    def _make_object_0(self) -> Actor:
        raise NotImplementedError

    def _make_object_1(self) -> Actor:
        raise NotImplementedError

    @property
    def _object_0_z_on_ground(self) -> torch.Tensor:
        return torch.tensor([self._object_0_z + self.z_top], device=self.device).repeat(self.num_envs)

    @property
    def _object_1_z_on_ground(self) -> torch.Tensor:
        return torch.tensor([self._object_1_z + self.z_top], device=self.device).repeat(self.num_envs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self._static_box_pos = np.array([0.0, 0.0, self.z_top/2])
        self._static_box_dims = np.array([0.4, 0.4, self.z_top])

        self.static_box = build_static_box_with_collision(
            builder=self.scene.create_actor_builder(),
            half_sizes=self._static_box_dims/2,
            color=(88/255, 87/255, 86/255),
            name="static_box",
            initial_pose=Pose.create_from_pq(
                p=self._static_box_pos),
            physical_material=PhysxMaterial(
                static_friction=self.table_static_friction,
                dynamic_friction=self.table_dynamic_friction,
                restitution=self.table_restitution,
            )
        )
        self.object_0 = self._make_object_0()
        self.object_1 = self._make_object_1()

    def _first_initialize_episode(self, env_idx: torch.Tensor, options: dict):

        self._episode_steps = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.int32)  # (n, 1)

        self._object_0_init_pos = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self._object_1_init_pos = torch.zeros(
            (self.num_envs, 3), device=self.device)

        self._object_0_init_quat = torch.zeros(
            (self.num_envs, 4), device=self.device)
        self._object_1_init_quat = torch.zeros(
            (self.num_envs, 4), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self._initialized:
            self._first_initialize_episode(env_idx, options)
            self._initialized = True

        self._initialize_actors(env_idx, options=options)
        self._initialize_agent(env_idx, options=options)

        self._object_0_init_pos[env_idx] = self.object_0.pose.p[env_idx]
        self._object_1_init_pos[env_idx] = self.object_1.pose.p[env_idx]

        self._object_0_init_quat[env_idx] = self.object_0.pose.q[env_idx]
        self._object_1_init_quat[env_idx] = self.object_1.pose.q[env_idx]

        self._episode_steps[env_idx] = 0

    def _initialize_actors(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    def _initialize_agent(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    def _after_control_step(self):
        self._episode_steps += 1

    def evaluate(self, **kwargs) -> dict:

        root_pose = self.agent.robot.get_root_pose()

        root_pose_inv = root_pose.inv()

        table_pose = self.table_scene.table.pose
        static_box_pose = self.static_box.pose
        object_0_pose = self.object_0.pose
        object_1_pose = self.object_1.pose

        object_0_lifted_height = self.object_0.pose.p[...,
                                                      2] - self._object_0_z_on_ground
        assert object_0_lifted_height.shape == (self.num_envs,)

        object_1_lifted_height = self.object_1.pose.p[...,
                                                      2] - self._object_1_z_on_ground
        assert object_1_lifted_height.shape == (self.num_envs,)

        table_pose_in_root_frame = root_pose_inv * table_pose
        static_box_pose_in_root_frame = root_pose_inv * static_box_pose
        object_0_pose_in_root_frame = root_pose_inv * object_0_pose
        object_1_pose_in_root_frame = root_pose_inv * object_1_pose

        return dict(
            object_0_lifted_height=object_0_lifted_height,
            object_1_lifted_height=object_1_lifted_height,
            table_pose_in_root_frame=table_pose_in_root_frame,
            static_box_pose_in_root_frame=static_box_pose_in_root_frame,
            object_0_pose_in_root_frame=object_0_pose_in_root_frame,
            object_1_pose_in_root_frame=object_1_pose_in_root_frame,
            z_top=self.z_top,
            static_box_dims=self._static_box_dims,
        )


class OneObjectBase(BaseEnv):
    SUPPORTED_ROBOTS = [FrankaAllegoRight.uid]
    SUPPORTED_REWARD_MODES = ["none"]

    agent: Union[FrankaAllegoRight]

    table_static_friction = 0.5
    table_dynamic_friction = 0.5
    table_restitution = 0.0

    def __init__(
        self,
        *args,
        num_envs=1,
        control_freq: int = 50,
        robot_uid: str = FrankaAllegoRight.uid,
        initial_agent_poses=Pose.create_from_pq(p=[-0.5, 0, 0.0]),
        static_box_pos=np.array([0.0, 0.0, 0.083/2]),
        static_box_dims=np.array([0.5, 0.8, 0.083]),
        static_box_color=(88/255, 87/255, 86/255),
        **kwargs,
    ):

        self._my_control_freq = control_freq

        self._initialized = False

        self.z_top = float(static_box_dims[2])
        self._initial_agent_poses = initial_agent_poses
        self._static_box_pos = static_box_pos
        self._static_box_dims = static_box_dims
        self._static_box_color = static_box_color

        super().__init__(
            *args,
            robot_uids=robot_uid,
            num_envs=num_envs,
            **kwargs,
        )

    def _after_reconfigure(self, options: dict):

        object_collision_mesh = self.object.get_first_collision_mesh()
        self._object_z = -object_collision_mesh.bounding_box.bounds[0, 2]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            [0.0, 0.3, 0.2], [0.0, 0.0, 0.05])
        return [
            CameraConfig(
                uid="render_camera",
                pose=pose,
                width=1280,
                height=960,
                fov=1,
                near=0.01,
                far=100,
                shader_pack='default',
            )
        ]

    @property
    def _default_sim_config(self):
        gpu_memory_config = GPUMemoryConfig(
            max_rigid_contact_count=self.num_envs *
            max(1024, self.num_envs) * 8,
            max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
            found_lost_pairs_capacity=2**26,
        )
        scene_config = SceneConfig(
            contact_offset=0.02,
            solver_position_iterations=25,
            cpu_workers=min(os.cpu_count(), 4))
        return SimConfig(sim_freq=200,
                         control_freq=self._my_control_freq,
                         scene_config=scene_config,
                         gpu_memory_config=gpu_memory_config)

    def _load_agent(self, options):
        return super()._load_agent(options, self._initial_agent_poses)

    def _make_object(self) -> Actor:
        raise NotImplementedError

    @property
    def _object_z_on_ground(self) -> torch.Tensor:
        return torch.tensor([self._object_z + self.z_top], device=self.device).repeat(self.num_envs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        self.static_box = build_static_box_with_collision(
            builder=self.scene.create_actor_builder(),
            half_sizes=self._static_box_dims/2,
            color=self._static_box_color,
            name="static_box",
            initial_pose=Pose.create_from_pq(
                p=self._static_box_pos),
            physical_material=PhysxMaterial(
                static_friction=self.table_static_friction,
                dynamic_friction=self.table_dynamic_friction,
                restitution=self.table_restitution,
            )
        )
        self.object = self._make_object()

    def _first_initialize_episode(self, env_idx: torch.Tensor, options: dict):

        self._episode_steps = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.int32)  # (n, 1)

        self._object_init_pos = torch.zeros(
            (self.num_envs, 3), device=self.device)
        self._object_init_quat = torch.zeros(
            (self.num_envs, 4), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if not self._initialized:
            self._first_initialize_episode(env_idx, options)
            self._initialized = True

        self._initialize_actors(env_idx, options=options)
        self._initialize_agent(env_idx, options=options)

        self._object_init_pos[env_idx] = self.object.pose.p[env_idx]

        self._object_init_quat[env_idx] = self.object.pose.q[env_idx]

        self._episode_steps[env_idx] = 0

    def _initialize_actors(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    def _initialize_agent(self, env_idx: torch.Tensor, options: dict):
        raise NotImplementedError

    def _after_control_step(self):
        self._episode_steps += 1

    def evaluate(self, **kwargs) -> dict:

        root_pose = self.agent.robot.get_root_pose()

        root_pose_inv = root_pose.inv()

        table_pose = self.table_scene.table.pose
        static_box_pose = self.static_box.pose
        object_pose = self.object.pose

        object_lifted_height = self.object.pose.p[...,
                                                  2] - self._object_z_on_ground

        table_pose_in_root_frame = root_pose_inv * table_pose
        static_box_pose_in_root_frame = root_pose_inv * static_box_pose
        object_pose_in_root_frame = root_pose_inv * object_pose

        return dict(
            object_lifted_height=object_lifted_height,
            table_pose_in_root_frame=table_pose_in_root_frame,
            static_box_pose_in_root_frame=static_box_pose_in_root_frame,
            object_pose_in_root_frame=object_pose_in_root_frame,
            z_top=self.z_top,
            static_box_dims=self._static_box_dims,
        )
