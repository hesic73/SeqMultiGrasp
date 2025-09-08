import torch
import numpy as np

import copy
import h5py

import transforms3d

import gymnasium as gym

from mani_skill.utils.structs import Pose as ManiSkillPose
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers import RecordEpisode

from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuRoboPose
from curobo.geom.types import WorldConfig

from src.utils.width_mapper import WidthMapper
from src.utils import torch_3d_utils
from src.utils.misc import maniskill_transform_translation_rpy_batched
from src.envs.evaluator import OneObjectV0


from typing import Any, Dict, List, Optional, Union, Callable, Tuple


def run_interpolated_trajectory(
    env: gym.Env,
    success_indices: torch.Tensor,
    qpos_hand_pre_grasp_pose: torch.Tensor,  # pre-grasp arm pose
    qpos_hand_grasp_pose: torch.Tensor,      # grasp arm pose
    qpos_hand_lift_pose: torch.Tensor,       # lift arm pose
    qpos_hand_rotated_pose: Optional[torch.Tensor],    # rotated arm pose

    pre_grasp_hand_qpos: torch.Tensor,
    grasp_hand_qpos: torch.Tensor,
    target_grasp_hand_qpos: torch.Tensor,

    initial_joint_positions: torch.Tensor,

    steps_per_phase: int = 50,
    rotate_hand: bool = False,
):


    device = initial_joint_positions.device
    dtype = initial_joint_positions.dtype
    batch_size = initial_joint_positions.shape[0]

    arm_dof = 7
    hand_dof = 16

    initial_arm_qpos = initial_joint_positions[:, :arm_dof]
    initial_hand_qpos = initial_joint_positions[:, arm_dof:arm_dof+hand_dof]

    def linear_interpolate(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
        return start + alpha * (end - start)

    def step_env_with_interpolation(
        arm_start: torch.Tensor,
        arm_end: torch.Tensor,
        hand_start: torch.Tensor,
        hand_end: torch.Tensor,
        n_steps: int
    ):

        for i in range(n_steps):
            alpha = i / max(n_steps - 1, 1)

            action = torch.zeros(batch_size, arm_dof +
                                 hand_dof, device=device, dtype=dtype)

            arm_interp = linear_interpolate(arm_start, arm_end, alpha)
            hand_interp = linear_interpolate(hand_start, hand_end, alpha)

            success_mask = success_indices

            # arm
            action[success_mask, :arm_dof] = arm_interp[success_mask]
            action[~success_mask, :arm_dof] = initial_arm_qpos[~success_mask]

            # hand
            action[success_mask, arm_dof:] = hand_interp[success_mask]
            action[~success_mask, arm_dof:] = initial_hand_qpos[~success_mask]

            env.step(action)

    step_env_with_interpolation(
        arm_start=qpos_hand_pre_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=initial_hand_qpos,
        hand_end=initial_hand_qpos,
        n_steps=steps_per_phase
    )

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=initial_hand_qpos,
        hand_end=pre_grasp_hand_qpos,
        n_steps=steps_per_phase
    )

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=pre_grasp_hand_qpos,
        hand_end=grasp_hand_qpos,
        n_steps=steps_per_phase
    )

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_grasp_pose,
        hand_start=grasp_hand_qpos,
        hand_end=target_grasp_hand_qpos,
        n_steps=steps_per_phase
    )

    step_env_with_interpolation(
        arm_start=qpos_hand_grasp_pose,
        arm_end=qpos_hand_lift_pose,
        hand_start=target_grasp_hand_qpos,
        hand_end=target_grasp_hand_qpos,
        n_steps=steps_per_phase
    )

    arm_end_qpos = qpos_hand_lift_pose

    if rotate_hand:
        step_env_with_interpolation(
            arm_start=qpos_hand_lift_pose,
            arm_end=qpos_hand_rotated_pose,
            hand_start=target_grasp_hand_qpos,
            hand_end=target_grasp_hand_qpos,
            n_steps=steps_per_phase
        )
        arm_end_qpos = qpos_hand_rotated_pose

    step_env_with_interpolation(
        arm_start=arm_end_qpos,
        arm_end=arm_end_qpos,
        hand_start=target_grasp_hand_qpos,
        hand_end=target_grasp_hand_qpos,
        n_steps=steps_per_phase
    )



def _validate_tabletop(
    ik_solver: IKSolver,
    width_mapper: WidthMapper,
    tensor_args: TensorDeviceType,
    object_name: str,
    object_xy: List[float],
    hand_initial_joint_positions: List[float],
    static_box_pose_in_root_frame: np.ndarray,
    static_box_dims: np.ndarray,
    initial_agent_poses: ManiSkillPose,
    object_pose_in_root_frame: ManiSkillPose,
    hand_pose_relative_to_object: torch.Tensor,
    hand_qpos: torch.Tensor,
    output_dir: Optional[str] = None,
    save_video: bool = False,
    rotate_hand: bool = False,
) -> List[int]:

    object_pose_in_root_frame = ManiSkillPose(object_pose_in_root_frame.raw_pose.to(
        tensor_args.device).to(tensor_args.dtype))

    assert hand_pose_relative_to_object.shape[0] == hand_qpos.shape[0]
    batch_size = hand_pose_relative_to_object.shape[0]
    if save_video:
        assert batch_size == 1

    success_indices = torch.ones(
        batch_size, dtype=torch.bool, device=tensor_args.device)

    hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_rpy = hand_pose_relative_to_object.split([
        3, 3], dim=-1)
    hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_rpy = maniskill_transform_translation_rpy_batched(
        hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_rpy)

    hand_pose_relative_to_object_quat = torch_3d_utils.rpy_to_quaternion(
        hand_pose_relative_to_object_rpy)

    hand_pose_relative_to_object: ManiSkillPose = ManiSkillPose.create_from_pq(
        hand_pose_relative_to_object_xyz, hand_pose_relative_to_object_quat)

    hand_grasp_pose = object_pose_in_root_frame*hand_pose_relative_to_object

    norm_object = normalize(hand_pose_relative_to_object.get_p())

    hand_pre_grasp_pose = copy.deepcopy(hand_grasp_pose)
    hand_pre_grasp_pose.set_p(hand_grasp_pose.get_p() + 0.05*norm_object)

    qpos_hand_pre_grasp_pose, hand_pre_grasp_pose_success = compute_qpos_and_success(
        ik_solver, hand_pre_grasp_pose)

    success_indices = success_indices & hand_pre_grasp_pose_success

    world_cfg_no_object = make_world_config_empty(
        sponge_pose=static_box_pose_in_root_frame.tolist(),
        sponge_dims=static_box_dims.tolist(),
    )

    ik_solver.world_coll_checker.clear_cache()
    ik_solver.update_world(world_cfg_no_object)

    qpos_hand_grasp_pose, hand_grasp_pose_success = compute_qpos_and_success(
        ik_solver, hand_grasp_pose)

    success_indices = success_indices & hand_grasp_pose_success

    hand_lift_pose = copy.deepcopy(hand_grasp_pose)
    hand_lift_pose.p[:, 2] += 0.2

    qpos_hand_lift_pose, hand_lift_pose_success = compute_qpos_and_success(
        ik_solver, hand_lift_pose)

    success_indices = success_indices & hand_lift_pose_success

    if rotate_hand:
        hand_rotated_pose = copy.deepcopy(hand_grasp_pose)
        hand_rotated_pose.q = torch_3d_utils.rpy_to_quaternion(
            torch.tensor([torch.pi/2, 0, torch.pi/2], dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)).repeat(batch_size, 1)

        qpos_rotated_hand_pose, hand_rotated_pose_success = compute_qpos_and_success(
            ik_solver, hand_rotated_pose)
        success_indices = success_indices & hand_rotated_pose_success
    else:
        qpos_rotated_hand_pose = None

    if not success_indices.any():
        return []


    hand_qpos_dict = hand_qpos_to_dict(hand_qpos)

    pre_grasp_hand_qpos_dict, _ = width_mapper.squeeze_fingers(hand_qpos_dict,
                                                               delta_width_thumb=-0.025,
                                                               delta_width_others=-0.025,
                                                               keep_z=True,
                                                               )

    pre_grasp_hand_qpos = dict_to_hand_qpos(pre_grasp_hand_qpos_dict)

    target_grasp_hand_qpos_dict, _ = width_mapper.squeeze_fingers(hand_qpos_dict,
                                                                  delta_width_thumb=0.03,
                                                                  delta_width_others=0.03,
                                                                  keep_z=True,
                                                                  )
    target_grasp_hand_qpos = dict_to_hand_qpos(target_grasp_hand_qpos_dict)

    sim_backend = "gpu"

    env_seed = 0

    initial_joint_positions = torch.cat(
        [qpos_hand_pre_grasp_pose,
         torch.tensor(hand_initial_joint_positions).to(tensor_args.device).to(tensor_args.dtype).unsqueeze(0).repeat(
             batch_size, 1)],
        dim=-1
    )


    static_box_pose = initial_agent_poses*ManiSkillPose.create_from_pq(
        p=static_box_pose_in_root_frame[:3], q=static_box_pose_in_root_frame[3:])
    static_box_pos = static_box_pose.p.squeeze(
        0).cpu().numpy()

    env = gym.make(
        "OneObject-v0",
        num_envs=batch_size,
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend=sim_backend,
        object_name=object_name,
        object_xy=object_xy,
        robot_uid="franka_allegro_right",
        initial_joint_positions=initial_joint_positions.clone(),
        initial_agent_poses=initial_agent_poses,
        static_box_pos=static_box_pos,
        static_box_dims=static_box_dims,
    )
    unwrapped_env: BaseEnv = env.unwrapped
    env_agent = unwrapped_env.agent
    robot = env_agent.robot
    control_freq = int(unwrapped_env.control_freq)

    if output_dir is not None:
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            save_trajectory=False,
            save_video=save_video,
            max_steps_per_video=500,
            video_fps=control_freq,
        )

    env.reset(seed=env_seed)

    run_interpolated_trajectory(
        env=env,
        success_indices=success_indices,
        qpos_hand_pre_grasp_pose=qpos_hand_pre_grasp_pose,
        qpos_hand_grasp_pose=qpos_hand_grasp_pose,
        qpos_hand_lift_pose=qpos_hand_lift_pose,
        qpos_hand_rotated_pose=qpos_rotated_hand_pose,

        pre_grasp_hand_qpos=pre_grasp_hand_qpos,
        grasp_hand_qpos=hand_qpos,
        target_grasp_hand_qpos=target_grasp_hand_qpos,

        initial_joint_positions=initial_joint_positions,
        steps_per_phase=50,
        rotate_hand=rotate_hand,
    )

    info = unwrapped_env.evaluate()
    object_lifted_height = info["object_lifted_height"]
    n_contact = info["n_contact"]

    lifted = object_lifted_height > 0.1
    lifted = lifted.to(success_indices.device)

    success_indices: torch.Tensor = success_indices & lifted  # (B,)

    env.close()

    return success_indices.cpu().nonzero().squeeze(1).tolist()


def hand_qpos_to_dict(qpos: torch.Tensor) -> Dict[str, torch.Tensor]:
    assert len(qpos.shape) == 2 and qpos.shape[-1] == 16

    d = {
        f"joint_{i}.0": qpos[:, i] for i in range(16)
    }

    return d


def dict_to_hand_qpos(d: Dict[str, torch.Tensor]) -> torch.Tensor:
    qpos = torch.stack([d[f"joint_{i}.0"] for i in range(16)], dim=-1)
    return qpos


def normalize(v: torch.Tensor):
    return v / torch.norm(v, dim=-1, keepdim=True)


def compute_qpos_and_success(
    ik_solver: IKSolver,
    pose: ManiSkillPose,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (b, 7) (b, )
    """
    result = ik_solver.solve_batch(CuRoboPose(
        pose.p, pose.q))
    return result.solution.squeeze(1), result.success.squeeze(1)


def make_world_config_empty(sponge_pose: List[float],
                            sponge_dims: List[float]):
    world_dict = {
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


def _load_data(data_path: str):
    if data_path.endswith('.npy'):
        return _load_data_npy(data_path)
    elif data_path.endswith('.h5'):
        return _load_data_hdf5(data_path)
    else:
        raise ValueError(f"Unsupported data path: {data_path}")


_translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
_rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
_joint_names = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',
    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0',
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0',
    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
]


def get_translation_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _translation_names]


def get_rpy_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _rot_names]


def get_joint_positions_from_qpos(qpos: Dict[str, float]) -> List[float]:
    return [qpos[name] for name in _joint_names]


def _load_data_npy(data_path: str):
    data_list = np.load(data_path, allow_pickle=True)

    if len(data_list) == 0:
        return None, None, None

    object_name = data_list[0]['object_code']

    n = len(data_list)

    hand_pose_relative_to_object = []
    hand_qpos = []

    for data_item in data_list:
        qpos_dict = data_item['qpos']
        trans = get_translation_from_qpos(qpos_dict)
        rpy = get_rpy_from_qpos(qpos_dict)
        joints = get_joint_positions_from_qpos(qpos_dict)

        hand_pose_relative_to_object.append(trans+rpy)
        hand_qpos.append(joints)

    hand_pose_relative_to_object = np.array(hand_pose_relative_to_object)
    hand_pose_relative_to_object = torch.from_numpy(
        hand_pose_relative_to_object)

    hand_qpos = np.array(hand_qpos)
    hand_qpos = torch.from_numpy(hand_qpos)

    return object_name, hand_pose_relative_to_object, hand_qpos


def _load_data_hdf5(data_path: str):
    with h5py.File(data_path, 'r') as f:

        if len(f['object_name']) == 0:
            return None, None, None
        object_name = f['object_name'][0].decode('utf-8')

        hand_pose_relative_to_object = f["pose"]
        hand_pose_relative_to_object = np.array(hand_pose_relative_to_object)
        hand_pose_relative_to_object = torch.from_numpy(
            hand_pose_relative_to_object)

        hand_qpos = f["qpos"]
        hand_qpos = np.array(hand_qpos)
        hand_qpos = torch.from_numpy(hand_qpos)

    return object_name, hand_pose_relative_to_object, hand_qpos
