# inference.py

import os
import numpy as np
import torch
import torch.nn as nn
import trimesh
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

from diffusers import DDPMScheduler
from src.network.unet import UNetModel
from src.network.model import MyModel
from src.utils.misc_utils import set_seed, load_checkpoint
from src.utils.pytorch3d_utils import batched_sample_points
from src.utils.hdf5_utils import create_hdf5
from src.utils.keypoints_utils import HandModelKeypoints

from src.data.utils import make_x0_unnormalizer, RotationRepresentation, _convert_pose_back

from src.consts import CONFIG_PATH

_MESHDATA_PATH = str(Path(__file__).parent.parent / "grasp_generation" / "assets" / "objects")


@torch.no_grad()
def sample_diffusion(model: nn.Module, scheduler: DDPMScheduler, pc: torch.Tensor, d_x: int,
                     device="cuda", num_inference_steps=100):
    """
    Given a batched point cloud `pc` of shape [B, 2, 1024, 3] (example),
    run the diffusion sampling process to produce x0 of shape [B, d_x].
    """
    model.eval()

    batch_size = pc.shape[0]

    x = torch.randn((batch_size, d_x), device=device)

    scheduler.set_timesteps(num_inference_steps)

    for t in reversed(scheduler.timesteps):
        t_tensor = torch.tensor(
            [t] * batch_size, device=device, dtype=torch.long)
        noise_pred = model(x, t_tensor, pc)
        x = scheduler.step(
            model_output=noise_pred,
            timestep=t,
            sample=x
        ).prev_sample

    return x


def create_model_and_scheduler(cfg: DictConfig, device="cuda"):

    unet = UNetModel(**cfg.model)

    model = MyModel(unet).to(device)

    noise_scheduler = DDPMScheduler(**cfg.scheduler)

    return model, noise_scheduler


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="inference")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"

    rotation_representation = cfg.rotation_representation
    rotation_representation = RotationRepresentation(rotation_representation)

    if rotation_representation == RotationRepresentation.rot6d:
        d_rot = 6
    elif rotation_representation == RotationRepresentation.quaternion:
        d_rot = 4
    elif rotation_representation == RotationRepresentation.rpy:
        d_rot = 3
    elif rotation_representation == RotationRepresentation.rotation_matrix:
        d_rot = 9
    elif rotation_representation == RotationRepresentation.axis_angle:
        d_rot = 3
    else:
        raise ValueError("Invalid rotation_representation")

    model, noise_scheduler = create_model_and_scheduler(cfg, device=device)

    load_checkpoint(model, cfg.checkpoint_path)

    mesh_paths = []
    assert len(cfg.objects) == 2, "Only support two objects"
    for obj_name in cfg.objects:
        mesh_paths.append(os.path.join(
            _MESHDATA_PATH, obj_name, f"{obj_name}.stl"))

    pcs = []
    for mp in mesh_paths:
        mesh: trimesh.Trimesh = trimesh.load_mesh(mp)
        pts = batched_sample_points(
            mesh, batch_size=cfg.batch_size, n_points=1024, device=device)
        pcs.append(pts)

    pc = torch.stack([pcs[0], pcs[1]], dim=1)

    pc_stats_path = cfg.get("pc_stats_path", None)
    if pc_stats_path is not None:
        assert os.path.isfile(pc_stats_path)
        with open(pc_stats_path, 'r') as f:
            pc_stats = json.load(f)
            pc_loc = torch.tensor(pc_stats['pc_loc'], device=device).float()
            pc_scale = torch.tensor(
                pc_stats['pc_scale'], device=device).float()
            pc = (pc - pc_loc) / pc_scale

    x0 = sample_diffusion(
        model=model,
        scheduler=noise_scheduler,
        pc=pc,
        d_x=cfg.model.d_x,
        device=device,
        num_inference_steps=cfg.num_inference_steps
    )

    if cfg.get("stats_path", None):
        assert os.path.exists(cfg.stats_path)
        x0_unnormalizer = make_x0_unnormalizer(
            cfg.stats_path, cfg.use_keypoints, device, rotation_representation)
        x0 = x0_unnormalizer(x0)

    if not cfg.use_keypoints:
        object_0_pose_representation, object_1_pose_representation, grasp_qpos = torch.split(
            x0, [3+d_rot, 3+d_rot, 16], dim=1)
    else:
        object_0_pose_representation, object_1_pose_representation, keypoints = torch.split(
            x0, [3+d_rot, 3+d_rot, x0.shape[1] - 6-2*d_rot], dim=1)
        keypoints = keypoints.view(-1, keypoints.shape[1] // 3, 3)
        hand_model_keypoints = HandModelKeypoints(device=device)
        grasp_qpos = hand_model_keypoints.fit(keypoints)

    obj0_xyz_rpy = _convert_pose_back(
        object_0_pose_representation, rotation_representation
    )
    obj1_xyz_rpy = _convert_pose_back(
        object_1_pose_representation, rotation_representation
    )

    assert cfg.get("output_path", None) is not None
    assert not os.path.exists(cfg.output_path)

    data = {
        'object_0_name': np.array([cfg.objects[0]]*cfg.batch_size),
        'object_0_scale': np.ones(cfg.batch_size),
        'object_0_pose': obj0_xyz_rpy.cpu().numpy(),

        'object_1_name': np.array([cfg.objects[1]]*cfg.batch_size),
        'object_1_scale': np.ones(cfg.batch_size),
        'object_1_pose': obj1_xyz_rpy.cpu().numpy(),

        'qpos': grasp_qpos.cpu().numpy()
    }

    create_hdf5(cfg.output_path, data)


if __name__ == "__main__":
    main()