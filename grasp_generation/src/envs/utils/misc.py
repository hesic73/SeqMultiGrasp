import numpy as np

from typing import Any, Dict, Optional, Sequence, Union, Tuple, Type, Callable, List

import imageio.v3 as iio


class VideoRecorder:
    def __init__(self):
        self._frames: List[np.ndarray] = []

    def record_frame(self, frame: np.ndarray):
        self._frames.append(frame)

    def reset(self):
        self._frames = []

    def save(self, path: str, fps: int = 30):
        if not self._frames:
            raise ValueError("No frames to save")

        # Save the frames as a video using imageio
        iio.imwrite(path, self._frames, fps=fps, codec='libx264')


def reorder_joint_indices(joint_names: List[str]) -> List[int]:
    indexed_joints = [(name, i) for i, name in enumerate(joint_names)]
    sorted_joints = sorted(indexed_joints, key=lambda x: int(
        x[0].split('_')[1].split('.')[0]))
    reordered_indices = [index for _, index in sorted_joints]
    return reordered_indices
