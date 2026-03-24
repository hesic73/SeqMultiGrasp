import os
import argparse
import h5py
import numpy as np
import torch

from src.utils.misc_utils import convert_rpy_to_rot6d


def main():
    parser = argparse.ArgumentParser(description="Compute per-field mean & std for object poses, qpos, and keypoints.")
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--out_path", default="stats_rot6d_qpos.npz", type=str)
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        ans = input(f"File {args.out_path} already exists. Overwrite? [y/N] ")
        if ans.lower() != 'y':
            return

    print(f"Loading data from {args.data_path} ...")
    with h5py.File(args.data_path, 'r') as f:
        obj0_rpy = f['object_0_pose'][:]
        obj1_rpy = f['object_1_pose'][:]
        qpos = f['qpos'][:]
        keypoints_xyz = None
        if 'keypoints' in f:
            keypoints = f['keypoints'][:]
            keypoints_xyz = keypoints.reshape(len(keypoints), -1)

    obj0_xyz, obj0_rpy = obj0_rpy[:, :3], obj0_rpy[:, 3:]
    obj1_xyz, obj1_rpy = obj1_rpy[:, :3], obj1_rpy[:, 3:]

    print("Converting rpy -> rot6d ...")
    obj0_rot6d = convert_rpy_to_rot6d(torch.from_numpy(obj0_rpy).float()).cpu().numpy()
    obj1_rot6d = convert_rpy_to_rot6d(torch.from_numpy(obj1_rpy).float()).cpu().numpy()

    stats = {
        "obj0_xyz_mean": obj0_xyz.mean(axis=0),
        "obj0_xyz_std": obj0_xyz.std(axis=0),
        "obj0_rot6d_mean": obj0_rot6d.mean(axis=0),
        "obj0_rot6d_std": obj0_rot6d.std(axis=0),
        "obj1_xyz_mean": obj1_xyz.mean(axis=0),
        "obj1_xyz_std": obj1_xyz.std(axis=0),
        "obj1_rot6d_mean": obj1_rot6d.mean(axis=0),
        "obj1_rot6d_std": obj1_rot6d.std(axis=0),
        "qpos_mean": qpos.mean(axis=0),
        "qpos_std": qpos.std(axis=0),
    }

    if keypoints_xyz is not None:
        stats.update({
            "keypoints_xyz_mean": keypoints_xyz.mean(axis=0),
            "keypoints_xyz_std": np.clip(keypoints_xyz.std(axis=0), 1e-3, None),
        })

    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"Saving stats to {args.out_path}")
    np.savez(args.out_path, **stats)
    print("Done.")


if __name__ == "__main__":
    main()
