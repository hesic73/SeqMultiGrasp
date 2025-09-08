import numpy as np
import torch

from scipy.spatial.transform import Rotation as R
from utils.misc import (
    inverse_pose,
    get_rpy_from_qpos,
    get_translation_from_qpos,
    get_joint_positions_from_qpos,
    make_contact_candidates_weight,
)

from src.utils.torch_3d_utils import rpy_to_rotation_matrix, rotation_matrix_to_rpy

from typing import Dict, Tuple, List


def convert_qpos_to_maniskill(qpos: Dict[str, float]) -> Tuple[List[float], List[float], List[float]]:
    """
    Convert qpos data to ManiSkill coordinate system, including translation, RPY, and hand joint positions.

    Parameters
    ----------
    qpos : dict
        Dictionary containing object qpos data.

    Returns
    -------
    translation_maniskill : ndarray of shape (3,)
        Translation vector in ManiSkill's coordinate system.
    rpy_maniskill : ndarray of shape (3,)
        Euler angles (roll, pitch, yaw) in ManiSkill's coordinate system.
    hand_qpos_maniskill : list of float
        Joint positions of the hand in ManiSkill's coordinate system.
    """
    translation = get_translation_from_qpos(qpos)
    rpy = get_rpy_from_qpos(qpos)

    translation_maniskill, rpy_maniskill = maniskill_transform_translation_rpy(
        translation, rpy)

    hand_qpos_maniskill = get_joint_positions_from_qpos(qpos)

    return translation_maniskill.tolist(), rpy_maniskill.tolist(), hand_qpos_maniskill


def _debug_print_in_maniskill(data_path: str, index: int):
    """
    Convert DexGraspNet data for two objects (object_0, object_1) into ManiSkill's coordinate system.
    """

    grasp_data = np.load(data_path, allow_pickle=True)
    record = grasp_data[index]

    object_0_qpos = record['object_0_qpos']
    object_1_qpos = record['object_1_qpos']

    object_0_trans_maniskill, object_0_rpy_maniskill, object_0_hand_qpos_maniskill = convert_qpos_to_maniskill(
        object_0_qpos
    )

    object_1_trans_maniskill, object_1_rpy_maniskill, object_1_hand_qpos_maniskill = convert_qpos_to_maniskill(
        object_1_qpos
    )

    print("Object 0:")
    print("Translation (ManiSkill):")
    print(repr(object_0_trans_maniskill))
    print("RPY (ManiSkill):")
    print(repr(object_0_rpy_maniskill))
    print("Hand Qpos (ManiSkill):")
    print(repr(object_0_hand_qpos_maniskill))

    print("Object 1:")
    print("Translation (ManiSkill):")
    print(repr(object_1_trans_maniskill))
    print("RPY (ManiSkill):")
    print(repr(object_1_rpy_maniskill))
    print("Hand Qpos (ManiSkill):")
    print(repr(object_1_hand_qpos_maniskill))


def maniskill_transform_translation_rpy(translation, rpy) -> Tuple[np.ndarray, np.ndarray]:
    # Initial transformation offset
    P = np.array([
        [0, 0, 1],  # x_s -> z_t
        [1, 0, 0],  # y_s -> x_t
        [0, 1, 0]   # z_s -> y_t
    ])

    R_source = R.from_euler('xyz', rpy).as_matrix()

    R_target = P @ R_source

    rpy_t = R.from_matrix(R_target).as_euler('xyz')

    translation_t = P @ np.array(translation)

    return translation_t, rpy_t
