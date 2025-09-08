import os
import json

import numpy as np
import torch
import trimesh
import transforms3d
import tyro

from utils.hand_model_lite import HandModelURDFLite
from utils.misc import (
    get_translation_from_qpos,
    get_rpy_from_qpos,
    get_joint_positions_from_qpos,
)
from src.consts import MESHDATA_PATH, HAND_URDF_PATH, CONTACT_CANDIDATES_PATH


def main(data_path: str, index: int = 0, device: str = "cuda") -> None:
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    contact_candidates = json.load(open(CONTACT_CANDIDATES_PATH, 'r'))
    contact_candidates = {
        k: torch.tensor(v, dtype=torch.float, device=device) for k, v in contact_candidates.items()
    }

    hand_model = HandModelURDFLite(
        urdf_path=HAND_URDF_PATH,
        contact_candidates=contact_candidates,
    )

    grasp_data = np.load(data_path, allow_pickle=True)
    record = grasp_data[index]

    grasp_code = record['object_code']
    mesh_path = os.path.join(MESHDATA_PATH, grasp_code, 'coacd/decomposed.obj')
    object_mesh_origin = trimesh.load(mesh_path)
    object_mesh_origin.visual.face_colors = [100, 100, 100, 100]

    qpos = record['qpos']
    contact_point_indices = record['contact_point_indices']

    translation = get_translation_from_qpos(qpos)
    euler_rpy = np.array(get_rpy_from_qpos(qpos))
    rot_mat = transforms3d.euler.euler2mat(*euler_rpy)
    rot6 = rot_mat[:, :2].T.ravel().tolist()
    joint_positions = get_joint_positions_from_qpos(qpos)

    hand_pose = torch.tensor(translation + rot6 + joint_positions, dtype=torch.float, device="cpu").unsqueeze(0)
    contact_point_indices_t = torch.tensor(contact_point_indices, dtype=torch.long, device="cpu").unsqueeze(0)

    hand_model.set_parameters(hand_pose, contact_point_indices_t)
    hand_mesh = hand_model.get_trimesh_data(0)

    object_scale = record.get('scale', 1.0)
    object_mesh = object_mesh_origin.copy().apply_scale(object_scale)

    scene = trimesh.Scene([object_mesh, hand_mesh])
    scene.show()


if __name__ == "__main__":
    tyro.cli(main)


