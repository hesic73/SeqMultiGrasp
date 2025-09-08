from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)

import numpy as np
from transforms3d.euler import euler2quat

from typing import List, Tuple


def make_world_config_empty(sponge_pose: List[float],
                            sponge_dims: List[float]):
    world_dict = {
        "cuboid": {
            "table": {
                "dims": [2.418,  1.209, 0.9196429],
                "pose": [0.38, 0, -0.9196429/2]+euler2quat(0, 0, np.pi / 2).tolist(),
            },
            'sponge': {
                "dims": sponge_dims,
                "pose": sponge_pose,
            }
        },
    }

    world_cfg = WorldConfig.from_dict(world_dict)
    return world_cfg


def make_world_config_two_objects(
    object_0_file_path: str,
    object_0_pose: List[float],
    object_1_file_path: str,
    object_1_pose:  List[float],
    sponge_pose: List[float],
    sponge_dims: List[float],
):
    world_dict = {
        "mesh": {
            "object_0": {
                "file_path": object_0_file_path,
                "pose": object_0_pose,
            },
            "object_1": {
                "file_path": object_1_file_path,
                "pose": object_1_pose,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2.418,  1.209, 0.9196429],
                "pose": [0.38, 0, -0.9196429/2]+euler2quat(0, 0, np.pi / 2).tolist(),
            },
            'sponge': {
                "dims": sponge_dims,
                "pose": sponge_pose,
            }
        },
    }

    world_cfg = WorldConfig.from_dict(world_dict)

    return world_cfg


# NOTE (hsc): 等我有空了试试motion_gen.attach_objects_to_robot
def make_world_config_object_1(
    object_1_file_path: str,
    object_1_pose: List[float],
    sponge_pose: List[float],
    sponge_dims: List[float],
):
    world_dict = {
        "mesh": {
            "object_1": {
                "file_path": object_1_file_path,
                "pose": object_1_pose,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2.418,  1.209, 0.9196429],
                "pose": [0.38, 0, -0.9196429/2]+euler2quat(0, 0, np.pi / 2).tolist(),
            },
            'sponge': {
                "dims": sponge_dims,
                "pose": sponge_pose,
            }
        },
    }

    world_cfg = WorldConfig.from_dict(world_dict)

    return world_cfg
