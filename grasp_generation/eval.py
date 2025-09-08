import argparse
from datetime import datetime
import os
import h5py

import json

import torch
import random

import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv


import numpy as np
import sapien
from utils.maniskill_utils import (
    convert_qpos_to_maniskill,
    maniskill_transform_translation_rpy,
)
from utils.misc import (
    inverse_pose,
)
from src.motion_planning.utils.np_3d_utils import rpy_to_quaternion
from src.motion_planning.evaluate import evaluate_grasp
from typing import Tuple
from curobo.util.logger import setup_logger

from loguru import logger

setup_logger('warning')

INITIAL_JOINT_POSITIONS = [
    -0.14377305, -0.41615936, 0.15588209, -2.7992177, 0.09156579,
    2.3855371, 1.5030054, 0., 0.7, 0., 0.4, 0., 0.7, 0., 0.4, 0., 0.7, 0., 0.4, 1.,
    0., 0., 0.
]


 


def load_data(data_path: str, index: int) -> Tuple[str, str, sapien.Pose, sapien.Pose, np.ndarray]:
    if data_path.endswith('.npy'):
        return load_data_npy(data_path, index)
    elif data_path.endswith('.h5'):
        return load_data_hdf5(data_path, index)
    else:
        raise ValueError(f"Unsupported data file format: {data_path}")


def load_data_npy(data_path: str, index: int) -> Tuple[str, str, sapien.Pose, sapien.Pose, np.ndarray]:
    grasp_data = np.load(data_path, allow_pickle=True)
    record = grasp_data[index]

    object_0_name = record['object_0_code']
    object_1_name = record['object_1_code']

    object_0_qpos = record['object_0_qpos']
    object_1_qpos = record['object_1_qpos']

    object_0_trans, object_0_rpy, object_0_hand_qpos = convert_qpos_to_maniskill(
        object_0_qpos)
    object_1_trans, object_1_rpy, object_1_hand_qpos = convert_qpos_to_maniskill(
        object_1_qpos)

    hand_quat_relative_to_object_0 = rpy_to_quaternion(object_0_rpy)
    hand_pose_relative_to_object_0 = sapien.Pose(
        p=object_0_trans, q=hand_quat_relative_to_object_0)

    hand_quat_relative_to_object_1 = rpy_to_quaternion(object_1_rpy)
    hand_pose_relative_to_object_1 = sapien.Pose(
        p=object_1_trans, q=hand_quat_relative_to_object_1)

    grasp_qpos = np.array(object_1_hand_qpos)

    return object_0_name, object_1_name, hand_pose_relative_to_object_0, hand_pose_relative_to_object_1, grasp_qpos


def load_data_hdf5(data_path: str, index: int) -> Tuple[str, str, sapien.Pose, sapien.Pose, np.ndarray]:
    with h5py.File(data_path, 'r') as f:
        object_0_name = f['object_0_name'][index].decode('utf-8')
        object_1_name = f['object_1_name'][index].decode('utf-8')
        object_0_pose = np.array(f['object_0_pose'][index])
        object_1_pose = np.array(f['object_1_pose'][index])

        object_0_trans, object_0_rpy = inverse_pose(
            object_0_pose[:3], object_0_pose[3:])
        object_1_trans, object_1_rpy = inverse_pose(
            object_1_pose[:3], object_1_pose[3:])

        object_0_trans, object_0_rpy = maniskill_transform_translation_rpy(
            object_0_trans, object_0_rpy)
        object_1_trans, object_1_rpy = maniskill_transform_translation_rpy(
            object_1_trans, object_1_rpy)

        object_0_quat = rpy_to_quaternion(object_0_rpy)
        object_1_quat = rpy_to_quaternion(object_1_rpy)

        hand_pose_relative_to_object_0 = sapien.Pose(
            p=object_0_trans, q=object_0_quat)
        hand_pose_relative_to_object_1 = sapien.Pose(
            p=object_1_trans, q=object_1_quat)

        grasp_qpos = np.array(f['qpos'][index])

    return object_0_name, object_1_name, hand_pose_relative_to_object_0, hand_pose_relative_to_object_1, grasp_qpos


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple grasp attempts.")
    parser.add_argument("--data_path", type=str,
                        required=True, help="Path to the data file.")
    parser.add_argument("--n", type=int, required=True,
                        help="Number of grasps to evaluate.")
    parser.add_argument("--sim_backend", type=str, default="cpu",
                        help="Simulation backend (default: cpu).")
    parser.add_argument("--control_freq", type=int, default=20,
                        help="Control frequency (default: 20).")
    parser.add_argument("--vis", action="store_true",
                        help="Enable visualization.")
    parser.add_argument("--z_top", type=float, default=0.10,
                        help="Top Z value (default: 0.10).")
    parser.add_argument("--env_seed", type=int, default=114514,
                        help="Seed for environment (default: 114514).")

    parser.add_argument("--output_dir", type=str,
                        required=True, help="Directory to save logs.")

    parser.add_argument("--save_video", action="store_true",
                        help="Save video of the grasp attempt.")

    parser.add_argument("--randomize_object_pose", action="store_true",
                        help="Randomize object pose.")

    

    args = parser.parse_args()

    root_dir = args.output_dir
    root_dir = os.path.abspath(root_dir)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    run_info = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "results": [],
    }

    if args.data_path.endswith('.npy'):
        data_size = len(np.load(args.data_path, allow_pickle=True))
    elif args.data_path.endswith('.h5'):
        with h5py.File(args.data_path, 'r') as f:
            data_size = len(f['object_0_name'])
    else:
        raise ValueError("Unsupported file format.")

    rng = random.Random(args.env_seed)
    sampled_indices = rng.sample(range(data_size), args.n)

    success_count = 0
    failure_count = 0

    first_idx = sampled_indices[0]
    object_0_name, object_1_name, _, _, _ = load_data(
        args.data_path, first_idx)

    OBJECT_0_XY = [0.0, -0.1]
    OBJECT_1_XY = [0.0, 0.1]

    env = gym.make(
        "TwoObjects-v0",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        control_freq=args.control_freq,
        object_0_name=object_0_name,
        object_1_name=object_1_name,
        object_0_xy=OBJECT_0_XY.copy(),
        object_1_xy=OBJECT_1_XY.copy(),
        robot_uid="franka_allegro_right",
        initial_joint_positions=INITIAL_JOINT_POSITIONS.copy(),
        z_top=args.z_top,
    )

    unwrapped_env: BaseEnv = env.unwrapped

    object_pose_random_engine = random.Random(args.env_seed)

    for i, index in enumerate(sampled_indices):
        _object_0_name, _object_1_name, hand_pose_relative_to_object_0, hand_pose_relative_to_object_1, grasp_qpos = load_data(
            args.data_path, index)

        assert object_0_name == _object_0_name and object_1_name == _object_1_name

        assert isinstance(unwrapped_env.object_0_xy, torch.Tensor)
        assert isinstance(unwrapped_env.object_1_xy, torch.Tensor)
        if args.randomize_object_pose:
            d = 0.01
            object_0_xy_perturb = object_pose_random_engine.uniform(
                -d/2, d/2), object_pose_random_engine.uniform(-d/2, d/2)
            object_1_xy_perturb = object_pose_random_engine.uniform(
                -d/2, d/2), object_pose_random_engine.uniform(-d/2, d/2)
            unwrapped_env.object_0_xy = torch.tensor(
                OBJECT_0_XY, device=unwrapped_env.device) + torch.tensor(object_0_xy_perturb, device=unwrapped_env.device)
            unwrapped_env.object_1_xy = torch.tensor(
                OBJECT_1_XY, device=unwrapped_env.device) + torch.tensor(object_1_xy_perturb, device=unwrapped_env.device)

        env.reset()
        try:
            results = evaluate_grasp(
                env=env,
                grasp_qpos=grasp_qpos,
                hand_pose_relative_to_object_0=hand_pose_relative_to_object_0,
                hand_pose_relative_to_object_1=hand_pose_relative_to_object_1,
                output_dir=os.path.join(root_dir, f"run_{i}"),
                vis=args.vis,
                env_seed=args.env_seed,
                save_video=args.save_video
            )
            success = results["success"]
        except Exception as e:
            logger.error(f"Grasp {i} failed with error: {e}")
            success = False

        success_count += success
        failure_count += not success
        run_info["results"].append({
            "run_index": i,
            "data_index": index,
            "success": success
        })
        print(f"Grasp {i} success: {success}")

    total_attempts = success_count + failure_count
    success_rate = success_count / total_attempts if total_attempts > 0 else 0.0
    run_info["total_success"] = success_count
    run_info["total_failure"] = failure_count
    run_info["success_rate"] = success_rate

    print(
        f"Total Success: {success_count}, Total Failure: {failure_count}, Success Rate: {success_rate:.2%}")

    

    info_path = os.path.join(root_dir, "run_info.txt")
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=4)


if __name__ == "__main__":
    main()