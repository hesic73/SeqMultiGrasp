import tyro
from typing import List, Dict
import numpy as np
import os
from pathlib import Path
import tempfile

from utils.merge_utils import merge_grasps_from_hdf5, sanity_check_hdf5
from utils.misc import (
    get_translation_from_qpos,
    get_rpy_from_qpos,
    get_joint_positions_from_qpos,
)

from src.utils.hdf5_utils import append_to_hdf5


def main(
    path_0: str,
    path_1: str,
    save_path: str,
    buffer_size: int = 1000,
):
    save_path_parent = Path(save_path).parent
    if not save_path_parent.exists():
        save_path_parent.mkdir(parents=True)

    data_0 = np.load(path_0, allow_pickle=True)
    data_1 = np.load(path_1, allow_pickle=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path_0 = os.path.join(tmpdir, "tmp_0.h5")
        hdf5_path_1 = os.path.join(tmpdir, "tmp_1.h5")

        extract_and_save_to_hdf5(data_0, hdf5_path_0)
        extract_and_save_to_hdf5(data_1, hdf5_path_1)

        sanity_check_hdf5(hdf5_path_0)
        sanity_check_hdf5(hdf5_path_1)

        # Merge grasps
        merge_stats = merge_grasps_from_hdf5(
            save_path,
            hdf5_path_0,
            hdf5_path_1,
            buffer_size,
            silence=True,
        )
    print(f"Merged grasps saved to {save_path}")


def extract_and_save_to_hdf5(data_list: List[Dict], hdf5_file: str):
    object_names = []
    scales = []
    poses = []
    qposes = []
    contact_point_indicess = []

    for item in data_list:
        object_code = item["object_code"]
        scale = item["scale"]
        qpos = item["qpos"]

        translation = get_translation_from_qpos(qpos)
        rotation_rpy = get_rpy_from_qpos(qpos)
        joints = get_joint_positions_from_qpos(qpos)

        pose = np.concatenate([translation, rotation_rpy])  # (6,)
        qpose = joints  # (16,)

        object_names.append(object_code)
        scales.append(scale)
        poses.append(pose)
        qposes.append(qpose)
        contact_point_indicess.append(item["contact_point_indices"])

    data_to_save = {
        "object_name": np.array(object_names, dtype="S"),
        "scale": np.array(scales, dtype=np.float32),
        "pose": np.array(poses, dtype=np.float32),
        "qpos": np.array(qposes, dtype=np.float32),
        "contact_point_indices": np.array(contact_point_indicess, dtype=np.int32),
    }

    append_to_hdf5(hdf5_file, data_to_save)


if __name__ == "__main__":
    tyro.cli(main)
