from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from .utils import _validate_tabletop, _load_data
from src.consts import HAND_URDF_PATH, HAND_WIDTH_MAPPER_META_PATH
from src.utils.width_mapper import WidthMapper, make_width_mapper
from src.consts import get_object_mesh_path
from mani_skill.utils.structs import Pose as ManiSkillPose
import os

import torch
import numpy as np

import h5py

import transforms3d

import trimesh

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuRoboPose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.types.robot import JointState
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.util.logger import setup_logger

setup_logger('warning')


_HAND_INITIAL_JOINT_POSITIONS = [0., 0.7, 0., 0.4, 0., 0.7, 0., 0.4, 0., 0.7, 0., 0.4, 1.,
                                 0., 0., 0.]


def validate_one_object_tabletop(
    data_path: str,
    rotate_hand: bool = False,
    debug: bool = False,
) -> List[int]:

    object_name, hand_pose_relative_to_object, hand_qpos = _load_data(
        data_path)

    # debug
    if debug:
        hand_pose_relative_to_object = hand_pose_relative_to_object[:1, :]
        hand_qpos = hand_qpos[:1, :]

    if object_name is None:
        return []

    tensor_args = TensorDeviceType()

    hand_pose_relative_to_object = hand_pose_relative_to_object.to(
        tensor_args.dtype).to(tensor_args.device)
    hand_qpos = hand_qpos.to(tensor_args.dtype).to(tensor_args.device)

    initial_agent_poses = ManiSkillPose.create_from_pq(p=[-0.5, 0, 0.0])
    z_top = 0.083
    static_box_pos = np.array([0.0, 0.0, z_top/2])
    static_box_dims = np.array([0.5, 0.8, z_top])

    object_xy = [0.0, 0.0]

    object_mesh: trimesh.Trimesh = trimesh.load(
        get_object_mesh_path(object_name))
    object_z = -object_mesh.bounding_box.bounds[0, 2]

    initial_object_pose = ManiSkillPose.create_from_pq(
        p=[object_xy[0], object_xy[1], object_z+z_top])

    object_pose_in_root_frame = (initial_agent_poses.inv()*initial_object_pose)

    static_box_pose_in_root_frame = (initial_agent_poses.inv(
    )*ManiSkillPose.create_from_pq(p=static_box_pos)).raw_pose.squeeze(
        0).cpu().numpy()

    world_cfg_object = make_world_config_object(
        object_file_path=get_object_mesh_path(object_name),
        object_pose=object_pose_in_root_frame.raw_pose.squeeze(
            0).cpu().numpy().tolist(),
        sponge_pose=static_box_pose_in_root_frame.tolist(),
        sponge_dims=static_box_dims.tolist(),
    )

    ik_solver = initialize_ik_solver(world_cfg_object, tensor_args)

    # Setup WidthMapper
    width_mapper = make_width_mapper(
        urdf_path=HAND_URDF_PATH,
        meta_path=HAND_WIDTH_MAPPER_META_PATH,
        device=tensor_args.device,
    )

    if not debug:
        output_dir = None
        save_video = False
    else:
        output_dir = 'mp_logs'
        save_video = True

    success_indices = _validate_tabletop(
        ik_solver=ik_solver,
        width_mapper=width_mapper,
        tensor_args=tensor_args,
        object_name=object_name,
        object_xy=object_xy,
        hand_initial_joint_positions=_HAND_INITIAL_JOINT_POSITIONS,
        static_box_pose_in_root_frame=static_box_pose_in_root_frame,
        static_box_dims=static_box_dims,
        initial_agent_poses=initial_agent_poses,
        object_pose_in_root_frame=object_pose_in_root_frame,
        hand_pose_relative_to_object=hand_pose_relative_to_object,
        hand_qpos=hand_qpos,
        output_dir=output_dir,
        save_video=save_video,
        rotate_hand=rotate_hand,
    )

    return success_indices


def initialize_ik_solver(world_cfg: WorldConfig, tensor_args: TensorDeviceType) -> IKSolver:
    robot_file = "franka_allegro_right.yml"
    robot_cfg_path = join_path(get_robot_configs_path(), robot_file)
    robot_cfg_data = load_yaml(robot_cfg_path)
    robot_cfg = RobotConfig.from_dict(robot_cfg_data["robot_cfg"])

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    return IKSolver(ik_config)


def make_world_config_object(
    object_file_path: str,
    object_pose: List[float],
    sponge_pose: List[float],
    sponge_dims: List[float],
):
    world_dict = {
        "mesh": {
            "object": {
                "file_path": object_file_path,
                "pose": object_pose,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2.418,  1.209, 0.9196429],
                "pose": [0.38, 0, -0.9196429/2]+transforms3d.euler.euler2quat(0, 0, np.pi / 2).tolist(),
            },
            'sponge': {
                "dims": sponge_dims,
                "pose": sponge_pose,
            }
        },
    }

    world_cfg = WorldConfig.from_dict(world_dict)

    return world_cfg
