import gymnasium as gym
from gymnasium import Env as GymEnv


from mani_skill.envs.sapien_env import BaseEnv
from src.envs import PreGraspValidatorV0, PreGraspValidatorTwoObjectsV0
from src.envs.utils.misc import VideoRecorder
from src.envs.wrappers.pre_step import _PreStepOrAnyDoneWrapper

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from loguru import logger

from typing import Sequence

import json

from utils.maniskill_utils import convert_qpos_to_maniskill
from utils.misc import (
    extract_hand_pose_and_scale_tensor,
)

from utils.single_object_model import SingleObjectModel
from utils.hand_model import HandModel
from utils.move_along_normal import move_along_normal
from utils.misc import (
    hand_pose_to_dict,
)

from src.consts import HAND_URDF_PATH, CONTACT_CANDIDATES_PATH, MESHDATA_PATH


def validate_grasp(data_path: str, save_path: str):
    data_dict = np.load(data_path, allow_pickle=True)

    if len(data_dict) == 0:
        logger.warning("Data list is empty")
        np.save(save_path, [])
        return

    object_name = data_dict[0]['object_code']
    batch_size = len(data_dict)

    device = "cuda"

    dex_grasp_net_hand_pose, object_scale = extract_hand_pose_and_scale_tensor(
        data_dict=data_dict, device=device)

    contact_candidates = json.load(open(CONTACT_CANDIDATES_PATH, 'r'))
    contact_candidates = {
        k: torch.tensor(v, dtype=torch.float, device=device)
        for k, v in contact_candidates.items()
    }

    hand_model = HandModel(
        urdf_path=HAND_URDF_PATH,
        contact_candidates=contact_candidates,
        n_surface_points=1000,
        device=device
    )

    object_model = SingleObjectModel(
        data_root_path=MESHDATA_PATH,
        batch_size=batch_size,
        num_samples=2000,
        device=device
    )
    object_model.initialize(object_name)

    object_model.scale_factor = object_scale

    moved_hand_pose = move_along_normal(
        hand_model=hand_model,
        object_model=object_model,
        hand_pose=dex_grasp_net_hand_pose,
        device=device,
        thres_cont=1e-3,
        dis_move=1e-3,
        grad_move=5e2,
    )

    target_joint_position = moved_hand_pose[:, 9:]

    hand_state = []
    for d in data_dict:
        trans, rpy, hand_qpos = convert_qpos_to_maniskill(
            d['qpos'])
        maniskill_hand_state = np.concatenate([trans, rpy, hand_qpos])
        hand_state.append(maniskill_hand_state)
    hand_state = torch.from_numpy(np.stack(hand_state, axis=0)).float()

    env = gym.make(
        "PreGraspValidator-v0",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode='pd_joint_pos',
        render_mode="rgb_array",
        sim_backend='gpu',
        num_envs=batch_size,
        object_name=object_name,
        hand_state=hand_state,
        object_scale=object_scale.cpu().numpy(),
    )

    base_env: BaseEnv = env.unwrapped
    hand_state = hand_state.to(base_env.device)
    target_joint_position = target_joint_position.to(base_env.device)

    validated = torch.ones(batch_size, dtype=torch.bool,
                           device=base_env.device)

    maniskill_qpos = None
    maniskill_object_pose_in_hand_frame = None

    gravities = [
        np.array([0, 0, -9.81]),
        np.array([0, 0, 9.81]),
        np.array([0, -9.81, 0]),
        np.array([0, 9.81, 0]),
        np.array([-9.81, 0, 0]),
        np.array([9.81, 0, 0]),
    ]

    for i, g in enumerate(gravities):
        logger.info(f"Gravity: {g}")
        base_env.sim_config.scene_config.gravity = g
        env.reset(seed=0, options={'reconfigure': True})

        while True:
            observation, reward, terminated, truncated, info = env.step(
                target_joint_position)
            done = terminated | truncated
            if done.any():
                assert done.all()
                success = info['success']
                fail = info['fail']

                if i == 0:
                    maniskill_qpos = info['qpos']
                    maniskill_object_pose_in_hand_frame = info['object_pose_in_hand_frame']

                if fail.any():
                    fail_idx = fail.nonzero(as_tuple=True)[0]
                    success_rate = 1 - len(fail_idx)/batch_size
                    logger.info(f"Success rate: {success_rate:.2f}")
                    logger.info(f"Failed: {fail_idx.tolist()}")
                validated = validated & success
                break

    n_validated = validated.sum().item()
    logger.info(f"Validated: {n_validated}/{batch_size}")

    validated_idx = validated.nonzero(as_tuple=True)[0].cpu().numpy()
    validated_data_dict = data_dict[validated_idx]

    maniskill_qpos = maniskill_qpos.cpu().numpy()[validated_idx]
    maniskill_object_pose_in_hand_frame = maniskill_object_pose_in_hand_frame.cpu().numpy()[
        validated_idx]

    for i, d in enumerate(validated_data_dict):
        d['maniskill_qpos'] = maniskill_qpos[i]
        d['maniskill_pose_in_hand_frame'] = maniskill_object_pose_in_hand_frame[i]
        d['qpos_moved'] = hand_pose_to_dict(
            moved_hand_pose[i].cpu())

    np.save(save_path, validated_data_dict)
    logger.info(f"Saved to {save_path}")

    env.close()


def validate_grasp_two_objects(data_path: str, save_path: str):
    data_dict = np.load(data_path, allow_pickle=True)

    object_0_name = data_dict[0]['object_0_code']
    object_1_name = data_dict[0]['object_1_code']

    hand_state = []
    target_joint_position = []
    object_0_pose_in_hand_frame = []

    object_0_scale = []
    object_1_scale = []

    for d in data_dict:
        hand_state.append(d['maniskill_hand_state'])
        target_joint_position.append(d['target_joint_position'])
        object_0_pose_in_hand_frame.append(
            d['maniskill_object_0_pose_in_hand_frame'])

        object_0_scale.append(d['object_0_scale'])
        object_1_scale.append(d['object_1_scale'])

    hand_state = torch.from_numpy(np.stack(hand_state, axis=0)).float()
    target_joint_position = torch.from_numpy(
        np.stack(target_joint_position, axis=0)).float()
    object_0_pose_in_hand_frame = torch.from_numpy(
        np.stack(object_0_pose_in_hand_frame, axis=0)).float()

    object_0_scale = np.stack(object_0_scale, axis=0)
    object_1_scale = np.stack(object_1_scale, axis=0)

    batch_size = hand_state.shape[0]

    env = gym.make(
        "PreGraspValidatorTwoObjects-v0",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode='pd_joint_pos',
        render_mode="rgb_array",
        sim_backend='auto',
        num_envs=batch_size,
        object_0_name=object_0_name,
        object_1_name=object_1_name,
        hand_state=hand_state,
        object_0_pose_in_hand_frame=object_0_pose_in_hand_frame,
        object_0_scale=object_0_scale,
        object_1_scale=object_1_scale,
    )

    base_env: BaseEnv = env.unwrapped

    hand_state = hand_state.to(base_env.device)
    target_joint_position = target_joint_position.to(base_env.device)

    validated = torch.ones(batch_size, dtype=torch.bool,
                           device=base_env.device)

    maniskill_qpos = None
    maniskill_object_0_pose_in_hand_frame = None
    maniskill_object_1_pose_in_hand_frame = None

    gravities = [
        np.array([0, 0, -9.81]),
        np.array([0, 0, 9.81]),
        np.array([0, -9.81, 0]),
        np.array([0, 9.81, 0]),
        np.array([-9.81, 0, 0]),
        np.array([9.81, 0, 0]),
    ]

    for i, g in enumerate(gravities):
        logger.info(f"Gravity: {g}")
        base_env.sim_config.scene_config.gravity = g
        env.reset(seed=0, options={'reconfigure': True})

        while True:
            observation, reward, terminated, truncated, info = env.step(
                target_joint_position)
            done = terminated | truncated
            if done.any():
                assert done.all()
                success = info['success']
                fail = info['fail']

                if i == 0:
                    maniskill_qpos = info['qpos']
                    maniskill_object_0_pose_in_hand_frame = info['object_0_pose_in_hand_frame']
                    maniskill_object_1_pose_in_hand_frame = info['object_1_pose_in_hand_frame']

                if fail.any():
                    fail_idx = fail.nonzero(as_tuple=True)[0]
                    success_rate = 1 - len(fail_idx)/batch_size
                    logger.info(f"Success rate: {success_rate:.2f}")
                    logger.info(f"Failed: {fail_idx.tolist()}")
                validated = validated & success
                break

    n_validated = validated.sum().item()
    logger.info(f"Validated: {n_validated}/{batch_size}")

    validated_idx = validated.nonzero(as_tuple=True)[0].cpu().numpy()
    validated_data_dict = data_dict[validated_idx]

    maniskill_qpos = maniskill_qpos.cpu().numpy()[validated_idx]
    maniskill_object_0_pose_in_hand_frame = maniskill_object_0_pose_in_hand_frame.cpu().numpy()[
        validated_idx]
    maniskill_object_1_pose_in_hand_frame = maniskill_object_1_pose_in_hand_frame.cpu().numpy()[
        validated_idx]

    for i, d in enumerate(validated_data_dict):
        d['maniskill_qpos'] = maniskill_qpos[i]
        d['maniskill_object_0_pose_in_hand_frame'] = maniskill_object_0_pose_in_hand_frame[i]
        d['maniskill_object_1_pose_in_hand_frame'] = maniskill_object_1_pose_in_hand_frame[i]

    np.save(save_path, validated_data_dict)
    logger.info(f"Saved to {save_path}")


# for debugging
def render_validate_grasp(
    object_name: str,
    object_scale: Sequence[float],
    hand_state: torch.Tensor,
    target_joint_position: torch.Tensor,
    save_path: str,
):
    video_recorder = VideoRecorder()

    def record_frame_func(env: GymEnv):
        frame: torch.Tensor = env.render()
        if len(frame.shape) > 3:
            assert len(frame.shape) == 4
            frame = frame[0]

        # uint8
        assert len(frame.shape) == 3  # (H, W, C)
        frame = frame.cpu().numpy()
        video_recorder.record_frame(frame)

    env = gym.make(
        "PreGraspValidator-v0",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode='pd_joint_pos',
        render_mode="rgb_array",
        sim_backend='auto',
        num_envs=1,
        object_name=object_name,
        hand_state=hand_state,
        object_scale=object_scale,
    )

    env = _PreStepOrAnyDoneWrapper(env, func=record_frame_func)

    base_env: BaseEnv = env.unwrapped
    hand_state = hand_state.to(base_env.device)
    target_joint_position = target_joint_position.to(base_env.device)

    gravities = [
        np.array([0, 0, -9.81]),
        np.array([0, 0, 9.81]),
        np.array([0, -9.81, 0]),
        np.array([0, 9.81, 0]),
        np.array([-9.81, 0, 0]),
        np.array([9.81, 0, 0]),
    ]

    for i, g in enumerate(gravities):
        logger.info(f"Gravity: {g}")
        base_env.sim_config.scene_config.gravity = g
        env.reset(seed=0, options={'reconfigure': True})

        while True:
            observation, reward, terminated, truncated, info = env.step(
                target_joint_position)
            done = terminated | truncated
            if done.any():
                success = info['success'].item()
                fail = info['fail'].item()
                if success:
                    logger.info("Success")
                elif fail:
                    logger.info("Fail")

                break

    video_recorder.save(save_path)
    logger.info(f"Saved to {save_path}")


def render_validate_grasp_two_objects(
    object_0_name: str,
    object_1_name: str,
    object_0_scale: Sequence[float],
    object_1_scale: Sequence[float],
    hand_state: torch.Tensor,
    target_joint_position: torch.Tensor,
    object_0_pose_in_hand_frame: torch.Tensor,
    save_path: str,
):
    video_recorder = VideoRecorder()

    def record_frame_func(env: GymEnv):
        frame: torch.Tensor = env.render()
        if len(frame.shape) > 3:
            assert len(frame.shape) == 4
            frame = frame[0]

        # uint8
        assert len(frame.shape) == 3  # (H, W, C)
        frame = frame.cpu().numpy()
        video_recorder.record_frame(frame)

    env = gym.make(
        "PreGraspValidatorTwoObjects-v0",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode='pd_joint_pos',
        render_mode="rgb_array",
        sim_backend='auto',
        num_envs=1,
        object_0_name=object_0_name,
        object_1_name=object_1_name,
        hand_state=hand_state,
        object_0_pose_in_hand_frame=object_0_pose_in_hand_frame,
        object_0_scale=object_0_scale,
        object_1_scale=object_1_scale,
    )

    env = _PreStepOrAnyDoneWrapper(env, func=record_frame_func)

    base_env: BaseEnv = env.unwrapped
    hand_state = hand_state.to(base_env.device)
    target_joint_position = target_joint_position.to(base_env.device)

    gravities = [
        np.array([0, 0, -9.81]),
        np.array([0, 0, 9.81]),
        np.array([0, -9.81, 0]),
        np.array([0, 9.81, 0]),
        np.array([-9.81, 0, 0]),
        np.array([9.81, 0, 0]),
    ]

    for i, g in enumerate(gravities):
        logger.info(f"Gravity: {g}")
        base_env.sim_config.scene_config.gravity = g
        env.reset(seed=0, options={'reconfigure': True})

        while True:
            observation, reward, terminated, truncated, info = env.step(
                target_joint_position)
            done = terminated | truncated
            if done.any():
                success = info['success'].item()
                fail = info['fail'].item()
                if success:
                    logger.info("Success")
                elif fail:
                    logger.info("Fail")

                break

    video_recorder.save(save_path)
    logger.info(f"Saved to {save_path}")
