import copy
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import sapien

import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode

from src.motion_planning.planner import Planner
from src.utils.width_mapper import make_width_mapper, WidthMapper
from src.consts import HAND_WIDTH_MAPPER_META_PATH, HAND_URDF_PATH

from src.envs.evaluator import TwoObjectsV0
from src.envs.wrappers.record_action import RecordAction

from loguru import logger
import os


_TIME_DILATION_FACTOR = 0.5


_MASK_OBJECT_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                           0, 0, 1, 1, 1, 1], dtype=bool)

_MASK_OBJECT_1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                           1, 1, 0, 0, 0, 0], dtype=bool)

_ACTIVE_LINKS_OBJECT_0 = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0',
]

_ACTIVE_LINKS_OBJECT_1 = [
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0',
]


def evaluate_grasp(
        env: gym.Env,
    grasp_qpos: np.ndarray,
    hand_pose_relative_to_object_0: sapien.Pose,
    hand_pose_relative_to_object_1: sapien.Pose,
    output_dir: str,
    vis: bool = False,
    env_seed: Optional[int] = None,
    save_video: bool = False,
) -> Dict[str, Any]:

    os.makedirs(output_dir, exist_ok=True)

    actions = []

    def _record_action_func(action):
        actions.append(action)

    unwrapped_env: BaseEnv = env.unwrapped

    control_freq = int(unwrapped_env.control_freq)
    logger.info(f"Control frequency: {control_freq}")

    env = RecordAction(env, _record_action_func)
    env = RecordEpisode(env, output_dir=output_dir, save_video=save_video,
                        video_fps=control_freq)

    env.reset(seed=env_seed)

    object_0_mass = unwrapped_env.object_0._bodies[0].mass
    object_1_mass = unwrapped_env.object_1._bodies[0].mass
    logger.info(f"Object 0 mass: {object_0_mass*1000:.2f}g")
    logger.info(f"Object 1 mass: {object_1_mass*1000:.2f}g")

    if vis:
        viewer = unwrapped_env.render_human()

    planner = Planner(env, vis=vis)
    width_mapper = make_width_mapper(
        HAND_URDF_PATH, HAND_WIDTH_MAPPER_META_PATH)

    np.save(os.path.join(output_dir, "initial_joint_positions.npy"), {
        'arm_joint_positions': planner.arm_joint_positions,
        'hand_joint_positions': planner.hand_joint_positions,
    })

    # 让物体稳定
    for i in range(control_freq):
        obs, reward, terminated, truncated, info = unwrapped_env.step(
            np.concatenate([planner.arm_joint_positions, planner.hand_joint_positions], axis=0))

    try:
        # Object 0
        object_0_pose = planner.get_object_0_pose(root_frame=True)

        object_0_arm_grasp_pose = object_0_pose * hand_pose_relative_to_object_0

        n_object_0 = normalize(hand_pose_relative_to_object_0.get_p())
        object_0_arm_pre_grasp_pose = copy.deepcopy(object_0_arm_grasp_pose)
        object_0_arm_pre_grasp_pose.set_p(
            object_0_arm_grasp_pose.get_p() + n_object_0 * 0.05
        )

        planner.move_to_pose(object_0_arm_pre_grasp_pose)
        planner.update_world_empty()

        planner.move_to_pose(object_0_arm_grasp_pose)

        object_0_pre_grasp_qpos = compute_object_0_pre_grasp_pose(
            width_mapper, grasp_qpos)

        planner.hand_move_to_qpos(
            object_0_pre_grasp_qpos,
            interpolation_steps=control_freq,
            mask=_MASK_OBJECT_0,
        )

        # planner.hand_move_to_qpos(
        #     grasp_qpos,
        #     interpolation_steps=int(control_freq),
        #     mask=_MASK_OBJECT_0,
        # )

        object_0_target_grasp_qpos = compute_object_0_target_grasp_pose(
            width_mapper, grasp_qpos
        )

        planner.hand_move_to_qpos(
            object_0_target_grasp_qpos,
            interpolation_steps=int(control_freq),
            mask=_MASK_OBJECT_0,
        )
        planner.update_world_empty()

        planner.move_to_pose([0, 0, 0.1, 1, 0, 0, 0], use_delta=True,
                             time_dilation_factor=_TIME_DILATION_FACTOR,)

        planner._setup_motion_gen()

        planner.update_world_object_1()
        object_1_pose = planner.get_object_1_pose(root_frame=True)

        object_1_arm_grasp_pose = object_1_pose * hand_pose_relative_to_object_1
        n_object_1 = normalize(
            hand_pose_relative_to_object_1.get_p(), keep_z=True)
        object_1_arm_pre_grasp_pose = copy.deepcopy(object_1_arm_grasp_pose)

        object_1_arm_pre_grasp_pose.set_p(
            object_1_arm_grasp_pose.get_p() + n_object_1 * 0.2
        )

        planner.move_to_pose(
            object_1_arm_pre_grasp_pose,
            time_dilation_factor=_TIME_DILATION_FACTOR,
        )
        planner.update_world_empty()
        planner.move_to_pose(
            object_1_arm_grasp_pose,
            time_dilation_factor=_TIME_DILATION_FACTOR,
        )

        object_1_pre_grasp_qpos = compute_object_1_pre_grasp_pose(
            width_mapper, grasp_qpos)

        planner.hand_move_to_qpos(
            object_1_pre_grasp_qpos,
            interpolation_steps=control_freq,
            mask=_MASK_OBJECT_1,
        )

        # planner.hand_move_to_qpos(
        #     grasp_qpos,
        #     interpolation_steps=int(control_freq),
        #     mask=_MASK_OBJECT_1,
        # )

        object_1_target_grasp_qpos = compute_object_1_target_grasp_pose(
            width_mapper, grasp_qpos
        )

        planner.hand_move_to_qpos(
            object_1_target_grasp_qpos,
            interpolation_steps=int(control_freq),
            mask=_MASK_OBJECT_1,
        )
        planner.update_world_empty()
        planner.move_to_pose([0, 0, 0.2, 1, 0, 0, 0],
                             use_delta=True, time_dilation_factor=_TIME_DILATION_FACTOR,)

        # Evaluation results
        info = unwrapped_env.evaluate()

        object_0_lifted_height = info["object_0_lifted_height"].item()
        object_1_lifted_height = info["object_1_lifted_height"].item()

        logger.info(
            f"Object 0 lifted height: {object_0_lifted_height*100:.2f}cm")
        logger.info(
            f"Object 1 lifted height: {object_1_lifted_height*100:.2f}cm")
    finally:
        if actions:
            actions = np.stack(actions, axis=0)
        else:
            actions = np.zeros((0, 23))
            logger.warning("No actions recorded.")
        np.save(os.path.join(output_dir, "actions.npy"), actions)
        env.flush_video()

    success = object_0_lifted_height > 0.1 and object_1_lifted_height > 0.1

    return {
        "object_0_lifted_height": object_0_lifted_height,
        "object_1_lifted_height": object_1_lifted_height,
        "success": success,
    }


def normalize(v: np.ndarray, keep_z: bool = False) -> np.ndarray:
    if keep_z:
        v_xy_only = v.copy()
        v_xy_only[2] = 0
        return v_xy_only / np.linalg.norm(v_xy_only)
    else:
        return v / np.linalg.norm(v)


def make_qpos_dict(qpos: torch.Tensor) -> Dict[str, torch.Tensor]:
    assert len(qpos.shape) == 2 and qpos.shape[-1] == 16

    d = {
        f"joint_{i}.0": qpos[:, i] for i in range(16)
    }

    return d


def compute_object_0_target_grasp_pose(width_mapper: WidthMapper, qpos: np.ndarray) -> np.ndarray:
    qpos_dict = make_qpos_dict(
        torch.from_numpy(qpos).unsqueeze(0))

    # squeeze index and thumb fingers
    squeezed_qpos_dict, _ = width_mapper.squeeze_fingers(
        qpos_dict, 0.03, torch.tensor([0.03, 0.03, 0.00], device=width_mapper.device), keep_z=True, active_links=_ACTIVE_LINKS_OBJECT_0)

    squeezed_qpos = np.stack(
        [squeezed_qpos_dict[f"joint_{i}.0"].item() for i in range(16)], axis=0)

    return squeezed_qpos


def compute_object_0_pre_grasp_pose(width_mapper: WidthMapper, qpos: np.ndarray) -> np.ndarray:
    qpos_dict = make_qpos_dict(
        torch.from_numpy(qpos).unsqueeze(0))

    # squeeze index and thumb fingers
    squeezed_qpos_dict, _ = width_mapper.squeeze_fingers(
        qpos_dict, -0.025, torch.tensor([-0.025, -0.025, 0.00], device=width_mapper.device), keep_z=True, active_links=_ACTIVE_LINKS_OBJECT_0)

    squeezed_qpos = np.stack(
        [squeezed_qpos_dict[f"joint_{i}.0"].item() for i in range(16)], axis=0)

    return squeezed_qpos


def compute_object_1_target_grasp_pose(width_mapper: WidthMapper, qpos: np.ndarray) -> np.ndarray:
    qpos_dict = make_qpos_dict(
        torch.from_numpy(qpos).unsqueeze(0))

    # squeeze middle and ring fingers
    squeezed_qpos_dict, _ = width_mapper.squeeze_fingers(
        qpos_dict, 0.00, torch.tensor([0, 0.00, 0.03], device=width_mapper.device), keep_z=True, active_links=_ACTIVE_LINKS_OBJECT_1)

    squeezed_qpos = np.stack(
        [squeezed_qpos_dict[f"joint_{i}.0"].item() for i in range(16)], axis=0)

    return squeezed_qpos


def compute_object_1_pre_grasp_pose(width_mapper: WidthMapper, qpos: np.ndarray) -> np.ndarray:
    qpos_dict = make_qpos_dict(
        torch.from_numpy(qpos).unsqueeze(0))

    # squeeze middle and ring fingers
    squeezed_qpos_dict, _ = width_mapper.squeeze_fingers(
        qpos_dict, 0.00, torch.tensor([0, 0, -0.025], device=width_mapper.device), keep_z=True, active_links=_ACTIVE_LINKS_OBJECT_1)

    squeezed_qpos = np.stack(
        [squeezed_qpos_dict[f"joint_{i}.0"].item() for i in range(16)], axis=0)

    return squeezed_qpos
