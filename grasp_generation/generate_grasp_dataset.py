import os
import numpy as np
import h5py
import pandas as pd

import argparse
from loguru import logger
from pathlib import Path
from random import randint, seed as random_seed
from typing import List, Dict

from src.utils.misc import run_command, timing_decorator
from src.utils.hdf5_utils import append_to_hdf5
from src.consts import EXPERIMENTS_PATH

from utils.misc import (
    get_translation_from_qpos,
    get_rpy_from_qpos,
    get_joint_positions_from_qpos,
)


def _list_to_hydra_str(values: List[float]) -> str:
    return "[" + ",".join([str(x) for x in values]) + "]"


def _get_dataset_length(hdf5_path: str) -> int:
    if not os.path.exists(hdf5_path):
        return 0
    with h5py.File(hdf5_path, "r") as f:
        return len(f["object_name"])


def _extract_and_append_to_hdf5(data_list: List[Dict], hdf5_path: str) -> None:
    object_names = []
    scales = []
    poses = []
    qposes = []
    contact_point_indices_list = []

    for item in data_list:
        qpos = item["qpos"]

        translation = get_translation_from_qpos(qpos)
        rotation_rpy = get_rpy_from_qpos(qpos)
        joints = get_joint_positions_from_qpos(qpos)

        pose = np.concatenate([translation, rotation_rpy])
        qpose = joints

        object_names.append(item["object_code"])
        scales.append(item["scale"])
        poses.append(pose)
        qposes.append(qpose)
        contact_point_indices_list.append(item["contact_point_indices"])

    data_to_save = {
        "object_name": np.array(object_names, dtype="S"),
        "scale": np.array(scales, dtype=np.float32),
        "pose": np.array(poses, dtype=np.float32),
        "qpos": np.array(qposes, dtype=np.float32),
        "contact_point_indices": np.array(contact_point_indices_list, dtype=np.int32),
    }

    append_to_hdf5(hdf5_path, data_to_save)
    logger.info(f"Saved {len(object_names)} samples to {hdf5_path}")


@timing_decorator
def main(
    object_name: str,
    n_contact: int,
    active_links: str,
    hdf5_path: str,
    qpos_loc: List[float],
    ground_offset: float = 0.01,
    n_iter: int = 6000,
    batch_size: int = 256,
    initialization_method: str = "multi_grasp",
    gpu: int = 0,
    target_length: int = 100,
    script_seed: int = 42,
    disable_tqdm: bool = False,
    resume_seed: int = None,
    seed_offset: int = 0,
):

    assert len(qpos_loc) == 16 and all(
        isinstance(x, (float, int)) for x in qpos_loc
    )

    Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)

    assert initialization_method in ("multi_grasp", "side_grasp"), (
        f"Unsupported initialization_method: {initialization_method!r}. "
        "Choose 'multi_grasp' or 'side_grasp'."
    )

    random_seed(script_seed)
    np.random.seed(script_seed)

    qpos_loc_str = _list_to_hydra_str(qpos_loc)

    run_name_prefix = f"{object_name}_run"

    current_length = _get_dataset_length(hdf5_path)

    total_n = 0
    total_success = 0
    total_validated = 0
    total_tabletop_validated = 0

    # Advance the RNG to resume_seed if requested
    seed = randint(0, 10000) + seed_offset
    if resume_seed is not None:
        while seed != resume_seed:
            seed = randint(0, 10000) + seed_offset

    try:
        while current_length < target_length:
            logger.info(f"Seed: {seed}")

            run_name = f"{run_name_prefix}_seed{seed}"

            command = (
                f"python main.py"
                f" object_code={object_name}"
                f" name={run_name}"
                f" active_links={active_links}"
                f" qpos_loc={qpos_loc_str}"
                f" n_contact={n_contact}"
                f" n_iter={n_iter}"
                f" batch_size={batch_size}"
                f" ground_offset={ground_offset}"
                f" initialization_method={initialization_method}"
                f" gpu={gpu}"
                f" seed={seed}"
                f" disable_tqdm={disable_tqdm}"
                f" use_writer=false"
            )

            run_command(command, redirect=True)

            seed = randint(0, 10000) + seed_offset

            result_dir = os.path.join(EXPERIMENTS_PATH, run_name, "results")
            info_path = os.path.join(result_dir, "info.csv")
            info = pd.read_csv(info_path)

            n_success = info.at[0, "number of successful grasps"]
            n_validated = info.at[0, "number of validated grasps"]
            n_tabletop_validated = info.at[0, "number of tabletop validated grasps"]

            total_n += batch_size
            total_success += n_success
            total_validated += n_validated
            total_tabletop_validated += n_tabletop_validated

            validated_npy = os.path.join(
                result_dir, f"{object_name}_tabletop_validated.npy"
            )
            data_list = np.load(validated_npy, allow_pickle=True)

            if len(data_list) == 0:
                logger.warning(
                    f"No tabletop-validated grasps in this run; "
                    f"current dataset length: {current_length}"
                )
                continue

            _extract_and_append_to_hdf5(data_list, hdf5_path)

            current_length = _get_dataset_length(hdf5_path)
            logger.info(f"Dataset length: {current_length}/{target_length}")

    finally:
        print(f"Total grasps attempted:          {total_n}")
        print(f"Total successful grasps:         {total_success}")
        print(f"Total validated grasps:          {total_validated}")
        print(f"Total tabletop validated grasps: {total_tabletop_validated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a grasp dataset.")
    parser.add_argument("--object_name", required=True)
    parser.add_argument("--n_contact", type=int, required=True)
    parser.add_argument("--active_links", required=True)
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--qpos_loc", required=True)
    parser.add_argument("--ground_offset", type=float, default=0.00)
    parser.add_argument("--n_iter", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--initialization_method", default="multi_grasp", choices=["multi_grasp", "side_grasp"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_length", type=int, default=100)
    parser.add_argument("--script_seed", type=int, default=42)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--resume_seed", type=int, default=None)
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()

    qpos_loc = [float(x) for x in args.qpos_loc.split(",")]

    main(
        object_name=args.object_name,
        n_contact=args.n_contact,
        active_links=args.active_links,
        hdf5_path=args.hdf5_path,
        qpos_loc=qpos_loc,
        ground_offset=args.ground_offset,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        initialization_method=args.initialization_method,
        gpu=args.gpu,
        target_length=args.target_length,
        script_seed=args.script_seed,
        disable_tqdm=args.disable_tqdm,
        resume_seed=args.resume_seed,
        seed_offset=args.seed_offset,
    )
