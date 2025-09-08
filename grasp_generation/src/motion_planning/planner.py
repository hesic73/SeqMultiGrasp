

import numpy as np

import os
import torch

from gymnasium import Env as GymEnv


from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose

import sapien

# cuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


from loguru import logger


from src.motion_planning.utils.curobo_utils import (
    make_world_config_empty,
    make_world_config_two_objects,
    make_world_config_object_1,
)
from src.motion_planning.utils.misc import allegro_hand_reorder_joint_indices, sapien_pose_to_curobo_pose
from src.consts import get_object_mesh_path

from typing import List, Dict, Tuple, Optional, Union


class Planner:

    def __init__(self, env: GymEnv, vis: bool = True) -> None:
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent = self.base_env.agent
        self.robot = self.env_agent.robot

        self.joint_names = [joint.get_name()
                            for joint in self.robot.get_active_joints()]
        assert len(self.joint_names) == 23
        self.hand_joint_names = self.joint_names[7:]
        self.hand_joint_indices = allegro_hand_reorder_joint_indices(
            self.hand_joint_names)

        self._target_arm_qpos = self.arm_joint_positions.copy()
        self._target_hand_qpos = self.hand_joint_positions.copy()

        self.tensor_args = TensorDeviceType()
        self.vis = vis
        self.n_dof = 7
        self.ee_link_name = "base_link"

        self._setup_motion_gen()

        self.control_mode = self.base_env.control_mode
        assert self.control_mode == "pd_joint_pos"

        self._target_hand_qpos = self.hand_joint_positions.copy()

    # Helper functions to get the state of the robot

    @property
    def arm_joint_positions(self) -> np.ndarray:
        """
        (7,) panda_joint1 to panda_joint7
        """
        return self.robot.get_qpos().cpu().squeeze(0).numpy()[:7]

    @property
    def hand_joint_positions(self) -> np.ndarray:
        """
        (16,) joint_0.0 to joint_15.0
        """
        tmp = self.robot.get_qpos().cpu().squeeze(0).numpy()[7:]
        return tmp[self.hand_joint_indices]

    def get_end_effector_pose(self, root_frame: bool = False) -> sapien.Pose:

        ee_link = self.robot.find_link_by_name(self.ee_link_name)
        pose = ee_link.pose  # sapien Pose
        pose = to_sapien_pose(pose)
        if root_frame:
            root_pose = to_sapien_pose(self.env_agent.robot.get_root_pose())
            pose = root_pose.inv() * pose
        return pose

    # Helper functions to get the state of the objects

    def get_object_0_pose(self, root_frame: bool = False) -> sapien.Pose:
        object_0_pose = self.base_env.object_0.pose
        object_0_pose = to_sapien_pose(object_0_pose)
        if root_frame:
            root_pose = self.root_pose
            object_0_pose = root_pose.inv()*object_0_pose
        return object_0_pose

    def get_object_1_pose(self, root_frame: bool = False) -> sapien.Pose:
        object_1_pose = self.base_env.object_1.pose
        object_1_pose = to_sapien_pose(object_1_pose)
        if root_frame:
            root_pose = self.root_pose
            object_1_pose = root_pose.inv()*object_1_pose
        return object_1_pose

    # Motion planning functions

    def _setup_motion_gen(self):
        robot_file = "franka_allegro_right.yml"
        robot_cfg_path = join_path(get_robot_configs_path(), robot_file)
        robot_cfg_data = load_yaml(robot_cfg_path)

        # Update `lock_joints` with current joint positions
        lock_joints_update = self._get_hand_joint_positions_dict()
        robot_cfg_data["robot_cfg"]["kinematics"]["lock_joints"] = lock_joints_update

        # Convert the modified data into a RobotConfig object
        self.robot_cfg = RobotConfig.from_dict(robot_cfg_data["robot_cfg"])

        assert self.robot_cfg.kinematics.kinematics_config.joint_limits.position.shape == (
            2, self.n_dof)

        world_cfg = self._make_world_cfg_two_objects()

        motion_gen_config = MotionGenConfig.load_from_robot_config(robot_cfg=self.robot_cfg,
                                                                   world_model=world_cfg,
                                                                   tensor_args=self.tensor_args,
                                                                   interpolation_dt=1/self.base_env.control_freq,
                                                                   self_collision_check=True,
                                                                   collision_activation_distance=0.01,
                                                                   )

        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(
            warmup_js_trajopt=False,
        )

    def _make_world_cfg_two_objects(self) -> WorldConfig:
        info = self.base_env.evaluate()

        self.root_pose = to_sapien_pose(self.env_agent.robot.get_root_pose())

        object_0_name = info["object_0_name"]
        object_1_name = info["object_1_name"]

        object_0_pose_in_root_frame = info["object_0_pose_in_root_frame"].raw_pose.squeeze(
            0).cpu().numpy()
        object_1_pose_in_root_frame = info["object_1_pose_in_root_frame"].raw_pose.squeeze(
            0).cpu().numpy()

        static_box_pose_in_root_frame = info["static_box_pose_in_root_frame"].raw_pose.squeeze(
            0).cpu().numpy()

        static_box_dims = info["static_box_dims"]

        world_cfg = make_world_config_two_objects(
            object_0_file_path=get_object_mesh_path(object_0_name),
            object_0_pose=object_0_pose_in_root_frame,
            object_1_file_path=get_object_mesh_path(object_1_name),
            object_1_pose=object_1_pose_in_root_frame,
            sponge_pose=static_box_pose_in_root_frame,
            sponge_dims=static_box_dims,
        )
        return world_cfg

    def _make_world_cfg_object_1(self) -> WorldConfig:
        info = self.base_env.evaluate()

        self.root_pose = to_sapien_pose(self.env_agent.robot.get_root_pose())

        object_1_name = info["object_1_name"]

        object_1_pose_in_root_frame = info["object_1_pose_in_root_frame"].raw_pose.squeeze(
            0).cpu().numpy()

        static_box_pose_in_root_frame = info["static_box_pose_in_root_frame"].raw_pose.squeeze(
            0).cpu().numpy()

        static_box_dims = info["static_box_dims"]

        world_cfg = make_world_config_object_1(
            object_1_file_path=get_object_mesh_path(object_1_name),
            object_1_pose=object_1_pose_in_root_frame,
            sponge_pose=static_box_pose_in_root_frame,
            sponge_dims=static_box_dims,
        )

        return world_cfg

    def update_world_empty(self):

        info = self.base_env.evaluate()

        static_box_pose_in_root_frame = info["static_box_pose_in_root_frame"].raw_pose.squeeze(
            0).cpu().numpy()

        static_box_dims = info["static_box_dims"]

        world_cfg = make_world_config_empty(
            sponge_pose=static_box_pose_in_root_frame,
            sponge_dims=static_box_dims,)

        world_cfg = WorldConfig()

        # https://github.com/NVlabs/curobo/issues/263
        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(world_cfg)

    def update_world_two_objects(self):
        world_cfg = self._make_world_cfg_two_objects()
        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(world_cfg)

    def update_world_object_1(self):
        world_cfg = self._make_world_cfg_object_1()
        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(world_cfg)

    def _compute_arm_trajectory(self, pose: List[float],
                                use_delta: bool = False,
                                time_dilation_factor: Optional[float] = None,
                                ) -> np.ndarray:
        goal_pose = Pose.from_list(pose)
        if use_delta:
            current_pose = self.get_end_effector_pose(root_frame=True)
            current_pose = sapien_pose_to_curobo_pose(current_pose)
            goal_pose = goal_pose.multiply(current_pose)

        start_state = JointState.from_position(
            self.robot.get_qpos().cuda(),
            joint_names=self.joint_names,
        )
        start_state = self.motion_gen.get_active_js(start_state)

        result = self.motion_gen.plan_single(
            start_state, goal_pose, MotionGenPlanConfig(
                enable_graph=False,
                enable_opt=True,
                max_attempts=1,
                time_dilation_factor=time_dilation_factor,
            ))

        if not result.success.item():
            logger.error(
                f"Failed to plan to the target pose: {result.status}.")
            raise RuntimeError("Failed to plan to the target pose")

        traj = result.get_interpolated_plan()
        traj_position = traj.position.cpu().numpy()

        return traj_position

    def _execute_arm_trajectory(self, traj_position: np.ndarray) -> None:
        n_steps = traj_position.shape[0]

        for i in range(n_steps):
            arm_qpos = traj_position[i, :7]
            hand_qpos = self._target_hand_qpos

            action = np.hstack([arm_qpos, hand_qpos])
            action = torch.from_numpy(action).float()

            obs, reward, terminated, truncated, info = self.env.step(
                action)
            if self.vis:
                viewer = self.base_env.render_human()

    def move_to_pose(
        self,
            pose: Union[List[float], sapien.Pose],
            use_delta: bool = False,
            time_dilation_factor: Optional[float] = None,
    ) -> None:
        if isinstance(pose, sapien.Pose):
            pose = pose.get_p().tolist() + pose.get_q().tolist()
        traj_position = self._compute_arm_trajectory(
            pose, use_delta, time_dilation_factor)
        self._execute_arm_trajectory(traj_position)

    def hand_move_to_qpos(self,
                          hand_qpos: np.ndarray,
                          interpolation_steps: int = 20,
                          mask: Optional[np.ndarray] = None) -> None:
        """
        Move the hand to the target joint positions (`hand_qpos`) with interpolation.

        Parameters:
            hand_qpos (np.ndarray): Target hand joint positions (shape: (16,)).
            interpolation_steps (int): Number of interpolation steps (default: 20).
            mask (np.ndarray or None): Optional boolean mask (shape: (16,)).
                                    If provided, only the elements where mask is True will be updated.

        Returns:
            None
        """
        assert hand_qpos.shape == (16,), "hand_qpos must have shape (16,)"
        if mask is not None:
            assert mask.shape == (16,), "mask must have shape (16,)"
            assert mask.dtype == bool, "mask must be a boolean array"

        current_hand_qpos = self.hand_joint_positions

        traj = np.linspace(current_hand_qpos, hand_qpos, interpolation_steps)

        for i in range(interpolation_steps):
            arm_qpos = self.arm_joint_positions

            if mask is not None:
                hand_qpos = np.where(mask, traj[i], self._target_hand_qpos)
            else:
                hand_qpos = traj[i]

            self._target_hand_qpos = hand_qpos

            action = np.hstack([arm_qpos, hand_qpos])
            action = torch.from_numpy(action).float()

            obs, reward, terminated, truncated, info = self.env.step(action)
            if self.vis:
                viewer = self.base_env.render_human()

    def _get_joint_positions_dict(self) -> Dict[str, float]:
        qpos = self.robot.get_qpos().cpu().squeeze(0).tolist()
        return dict(zip(self.joint_names, qpos))

    def _get_hand_joint_positions_dict(self) -> Dict[str, float]:
        joint_positions_dict = self._get_joint_positions_dict()
        hand_joint_positions_dict = {
            joint_name: joint_positions_dict[joint_name]
            for joint_name in self.hand_joint_names
        }
        return hand_joint_positions_dict