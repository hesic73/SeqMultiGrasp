import os

import h5py
import numpy as np
import trimesh
import transforms3d
import tyro

from allegro_visualization.create_allegro_scene import create_allegro_scene
from src.consts import MESHDATA_PATH, HAND_URDF_PATH


def _load_data_from_hdf5(hdf5_path: str, index: int, mesh_dir: str):
    with h5py.File(hdf5_path, 'r') as f:
        hand_state = {
            'translation': [0.0, 0.0, 0.0],
            'rotation': [0.0, 0.0, 0.0],
            'joint_angles': f['qpos'][index].tolist()
        }
        objects = []
        obj_idx = 0
        while f"object_{obj_idx}_name" in f:
            obj_name = f[f"object_{obj_idx}_name"][index].decode('utf-8')
            obj_scale = float(f[f"object_{obj_idx}_scale"][index])
            obj_pose = f[f"object_{obj_idx}_pose"][index]

            mesh_path = os.path.join(mesh_dir, f"{obj_name}", 'coacd/decomposed.obj')
            objects.append({
                'mesh': trimesh.load(mesh_path),
                'scale': obj_scale,
                'pose': {
                    'translation': obj_pose[:3].tolist(),
                    'rotation': obj_pose[3:].tolist()
                },
                'color': [100, 100, 100, 100]
            })

            obj_idx += 1

    return hand_state, objects


def main(hdf5_path: str, index: int = 0) -> None:
    hand_state, objects = _load_data_from_hdf5(hdf5_path, index, mesh_dir=MESHDATA_PATH)

    scene = create_allegro_scene(HAND_URDF_PATH, hand_state, objects)

    global_transform = np.eye(4)
    global_transform[:3, :3] = transforms3d.euler.euler2mat(0, 0, -np.pi / 2)
    scene.apply_transform(global_transform)

    scene.show()


if __name__ == "__main__":
    tyro.cli(main)


