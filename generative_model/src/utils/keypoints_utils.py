import torch
import json
from typing import Dict
from src.consts import ALLEGRO_HAND_URDF_PATH, ALLEGRO_HAND_KEYPOINTS_PATH
from src.utils.simple_hand_model import SimpleHandModel

from tqdm import trange, tqdm

from typing import List


class HandModelKeypoints:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = SimpleHandModel(ALLEGRO_HAND_URDF_PATH, device=device)
        self.keypoints_dict = self._load_keypoints(ALLEGRO_HAND_KEYPOINTS_PATH)

    def _load_keypoints(self, path: str) -> Dict[str, torch.Tensor]:
        with open(path, 'r') as f:
            keypoints_dict = json.load(f)
        return {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in keypoints_dict.items() if v}

    def compute_keypoints(self, qpos: torch.Tensor) -> torch.Tensor:
        assert len(
            qpos.shape) == 2 and qpos.shape[-1] == 16, "qpos must have shape (batch_size, 16)"

        qpos_dict = self.hand_qpos_to_dict(qpos)
        translations, rotations = self.model.forward_kinematics(qpos_dict)

        all_keypoints_hand_frame = []
        for name, keypoints in self.keypoints_dict.items():
            t = translations[name]  # (b, 3)
            r = rotations[name]  # (b, 3, 3)
            keypoints = keypoints.unsqueeze(0)  # (1, n, 3)
            # (b, n, 3)
            keypoints_hand_frame = keypoints @ r.transpose(
                1, 2) + t.unsqueeze(1)
            all_keypoints_hand_frame.append(keypoints_hand_frame)

        all_keypoints_hand_frame = torch.cat(
            all_keypoints_hand_frame, dim=1)  # (b, n_keypoints, 3)
        return all_keypoints_hand_frame

    def fit(
        self,
        target_keypoints: torch.Tensor,  # (batch_size, n_keypoints, 3)
        q_init: List[float] = None,
        n_steps: int = 200,
    ) -> torch.Tensor:

        batch_size = target_keypoints.shape[0]

        if q_init is None:
            q_init = [0.0]*16

        qpos = torch.tensor(q_init, dtype=torch.float32,
                            device=self.device).repeat(batch_size, 1)

        qpos.requires_grad = True

        optimizer = torch.optim.Adam(
            [
                {"params": qpos, "lr": 0.1}
            ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, n_steps, 0.01)

        for step in trange(n_steps):
            optimizer.zero_grad()
            keypoints_pred = self.compute_keypoints(qpos)
            loss = (keypoints_pred - target_keypoints).norm(dim=-
                                                            1).mean(dim=-1).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 25 == 24:
                tqdm.write(
                    f"Step {step} | Mean L1 (m): {loss.item() / batch_size}")

        return qpos.detach()

    @staticmethod
    def hand_qpos_to_dict(qpos: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {f"joint_{i}.0": qpos[:, i] for i in range(16)}

    @staticmethod
    def dict_to_hand_qpos(d: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([d[f"joint_{i}.0"] for i in range(16)], dim=-1)
