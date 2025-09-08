import torch

from tqdm import tqdm, trange

from src.utils.hand_model import HandModel
from src.utils.single_object_model import SingleObjectModel


def refine(
        hand_model: HandModel,
        hand_qpos: torch.Tensor,
        object_0_trans: torch.Tensor,
        object_0_rot6d: torch.Tensor,
        object_1_trans: torch.Tensor,
        object_1_rot6d: torch.Tensor,
        n_steps: int = 100,
):

    batch_size, _ = hand_qpos.shape
    device = hand_qpos.device

    hand_joints = hand_qpos.clone()
    hand_joints.requires_grad = True

    optimizer = torch.optim.Adam([

        {"params": [hand_joints], "lr": 1e-2},

        {"params": [object_0_trans], "lr": 1e-5},
        {"params": [object_0_rot6d], "lr": 1e-5},

        {"params": [object_1_trans], "lr": 1e-5},
        {"params": [object_1_rot6d], "lr": 1e-5},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_steps, 1e-4)

    for refine_step in trange(n_steps):
        contact_thres = (3 - 2 * refine_step / n_steps) * 0.01

        optimizer.zero_grad()

        q = torch.cat(
            [torch.zeros(batch_size, 9, device=device), hand_joints], dim=1)

        hand_model.set_parameters(q)

    return hand_joints.detach()
