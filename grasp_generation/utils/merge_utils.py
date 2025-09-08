import numpy as np

import h5py

import json

from utils.dataset_utils import (
    contact_point_indices_to_contact_mask,
    contact_mask_to_joint_indices,
)
from utils.misc import (
    inverse_pose,
)

from src.utils.hdf5_utils import HDF5Buffer

from src.consts import CONTACT_CANDIDATES_PATH


def try_merge_grasp(
    grasp_0: dict,
    grasp_1: dict,
    contact_candidates: dict,
    silence: bool = False,
):
    object_0_code = grasp_0['object_name'].decode('utf-8')
    object_1_code = grasp_1['object_name'].decode('utf-8')

    object_0_scale = grasp_0['scale']
    object_1_scale = grasp_1['scale']

    object_0_pose = grasp_0['pose']
    object_1_pose = grasp_1['pose']

    object_0_translation, object_0_rotation_rpy = object_0_pose[:3], object_0_pose[3:]
    object_1_translation, object_1_rotation_rpy = object_1_pose[:3], object_1_pose[3:]

    inverse_object_0_translation, inverse_object_0_rotation_rpy = inverse_pose(
        object_0_translation, object_0_rotation_rpy)
    object_0_relative_pose = np.concatenate(
        [inverse_object_0_translation, inverse_object_0_rotation_rpy])

    inverse_object_1_translation, inverse_object_1_rotation_rpy = inverse_pose(
        object_1_translation, object_1_rotation_rpy)
    object_1_relative_pose = np.concatenate(
        [inverse_object_1_translation, inverse_object_1_rotation_rpy])

    object_0_qpos = grasp_0['qpos']
    object_1_qpos = grasp_1['qpos']

    object_0_contact_point_indices = grasp_0['contact_point_indices']
    object_1_contact_point_indices = grasp_1['contact_point_indices']

    object_0_contact_mask = contact_point_indices_to_contact_mask(
        object_0_contact_point_indices, contact_candidates)
    object_1_contact_mask = contact_point_indices_to_contact_mask(
        object_1_contact_point_indices, contact_candidates)

    if np.sum(object_0_contact_mask) < 2 or np.sum(object_1_contact_mask) < 2:
        if not silence:
            print("Not enough contact points!")
        return None

    thumb_index = 0
    index_index = 1
    middle_index = 2
    ring_index = 3

    if object_0_contact_mask[thumb_index] or object_0_contact_mask[index_index] or object_0_contact_mask[middle_index]:
        assert object_0_contact_mask[ring_index] == 0
        object_0_contact_mask[thumb_index] = 1
        object_0_contact_mask[index_index] = 1
        object_0_contact_mask[middle_index] = 1

    if np.any(object_0_contact_mask.astype(np.int32) + object_1_contact_mask.astype(np.int32) > 1):
        if not silence:
            print("Contact mask overlap!")
        return None

    if np.sum(object_0_contact_mask.astype(np.int32) + object_1_contact_mask.astype(np.int32)) < 5:
        if not silence:
            print("Not enough total contact points!")
        return None

    object_0_active_joint_mask = np.array(contact_mask_to_joint_indices(
        object_0_contact_mask))
    object_1_active_joint_mask = np.array(contact_mask_to_joint_indices(
        object_1_contact_mask))

    joints = np.zeros(16)
    joints[object_0_active_joint_mask] = object_0_qpos[object_0_active_joint_mask]
    joints[object_1_active_joint_mask] = object_1_qpos[object_1_active_joint_mask]

    return {
        'object_0_name': object_0_code,
        'object_0_scale': object_0_scale,
        'object_0_pose': object_0_relative_pose,
        'object_0_qpos': object_0_qpos,
        'object_0_contact_point_indices': object_0_contact_point_indices,

        'object_1_name': object_1_code,
        'object_1_scale': object_1_scale,
        'object_1_pose': object_1_relative_pose,
        'object_1_qpos': object_1_qpos,
        'object_1_contact_point_indices': object_1_contact_point_indices,

        'qpos': joints,
    }


def merge_grasps_from_hdf5(
    merged_grasps_save_path: str,
    hdf5_path_0: str,
    hdf5_path_1: str,
    buffer_size: int = 1000,
    silence: bool = False,
):
    """
    Merge grasps from two HDF5 files and save them into a new file using HDF5Buffer.

    Parameters:
    - merged_grasps_save_path (str): Path to save the merged HDF5 file.
    - hdf5_path_0 (str): Path to the first HDF5 file.
    - hdf5_path_1 (str): Path to the second HDF5 file.
    - buffer_size (int): Buffer size for writing data to the output file.
    - silence (bool): Whether to print debug information.
    """
    contact_candidates = json.load(open(CONTACT_CANDIDATES_PATH, 'r'))

    with h5py.File(hdf5_path_0, 'r') as f0, h5py.File(hdf5_path_1, 'r') as f1:
        object_0_grasps = {key: f0[key][:] for key in f0.keys()}
        object_1_grasps = {key: f1[key][:] for key in f1.keys()}

    # Initialize the HDF5 buffer
    buffer = HDF5Buffer(merged_grasps_save_path, buffer_size)

    n_success = 0
    n_total = 0

    for idx_object_0 in range(len(object_0_grasps['object_name'])):
        grasp_0 = {key: object_0_grasps[key][idx_object_0]
                   for key in object_0_grasps}

        for idx_object_1 in range(len(object_1_grasps['object_name'])):
            grasp_1 = {key: object_1_grasps[key][idx_object_1]
                       for key in object_1_grasps}

            merged_grasp = try_merge_grasp(
                grasp_0, grasp_1, contact_candidates, silence=silence)

            n_total += 1
            if merged_grasp is not None:
                # Append merged grasp to the buffer
                buffer.append(merged_grasp)
                n_success += 1

    # Ensure all remaining data in the buffer is written to disk
    buffer.close()

    return {
        'merge_success_rate': n_success / n_total,
    }


def sanity_check_hdf5(filename: str):
    """
    Check if the HDF5 file matches the expected format.

    Expected format:
    - Datasets:
        - "object_name": dtype=bytes, shape=(N,), all values should be identical
        - "scale": dtype=float32, shape=(N,)
        - "pose": dtype=float32, shape=(N, ..., 6)
        - "qpos": dtype=float32, shape=(N, ..., 16)
        - "contact_point_indices": dtype=int32, shape=(N, ...)
    - All datasets must have the same size along the first dimension (N).

    Parameters:
    - filename (str): The HDF5 file to check.

    Raises:
    - ValueError: If the file does not match the expected format.
    """
    expected_datasets = {
        "object_name": "S",  # Bytes string
        "scale": np.float32,
        "pose": np.float32,
        "qpos": np.float32,
        "contact_point_indices": np.int32,
    }

    try:
        with h5py.File(filename, "r") as f:
            # Check if all expected datasets exist
            for key, expected_dtype in expected_datasets.items():
                if key not in f:
                    raise ValueError(
                        f"Dataset '{key}' is missing from the HDF5 file.")

                dset = f[key]

                # Check dtype
                if key == "object_name":
                    if not np.issubdtype(dset.dtype, np.dtype("S").type):
                        raise ValueError(
                            f"Dataset '{key}' has incorrect dtype: expected bytes, got {dset.dtype}.")
                else:
                    if dset.dtype != expected_dtype:
                        raise ValueError(
                            f"Dataset '{key}' has incorrect dtype: expected {expected_dtype}, got {dset.dtype}.")

                # Check shape
                if len(dset.shape) < 1:
                    raise ValueError(
                        f"Dataset '{key}' has an invalid shape: {dset.shape}. Expected at least one dimension.")

            # Check that all datasets have the same size along the first dimension
            first_dim_sizes = {key: f[key].shape[0]
                               for key in expected_datasets.keys()}
            if len(set(first_dim_sizes.values())) > 1:
                raise ValueError(
                    f"Inconsistent first dimension sizes: {first_dim_sizes}.")

            # Check specific conditions for each dataset
            # Check 'object_name': all values should be identical
            object_name_values = f["object_name"][:]
            if not np.all(object_name_values == object_name_values[0]):
                raise ValueError(
                    "Dataset 'object_name' contains non-identical values.")

            # Check 'pose': last dimension should be 6
            if f["pose"].shape[-1] != 6:
                raise ValueError(
                    f"Dataset 'pose' has incorrect last dimension: expected 6, got {f['pose'].shape[-1]}.")

            # Check 'qpos': last dimension should be 16
            if f["qpos"].shape[-1] != 16:
                raise ValueError(
                    f"Dataset 'qpos' has incorrect last dimension: expected 16, got {f['qpos'].shape[-1]}.")

    except Exception as e:
        raise ValueError(f"Sanity check failed: {e}")
