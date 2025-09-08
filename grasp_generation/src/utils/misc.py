from functools import wraps
import time
import subprocess
import os
import sys

import torch

from src.utils.torch_3d_utils import rpy_to_rotation_matrix, rotation_matrix_to_rpy


def run_command(command: str, redirect: bool = False):
    """Run a shell command and check for errors."""
    print(f"Running command:\n{command}")

    if redirect:
        stdout = sys.stdout
        stderr = sys.stderr
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    process.wait()  # Wait for the process to finish
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}")


def maniskill_transform_translation_rpy_batched(translations, rpys):
    """
    Batched version of maniskill_transform_translation_rpy in PyTorch.

    Args:
        translations (torch.Tensor): Tensor of shape (batch_size, 3) representing translations.
        rpys (torch.Tensor): Tensor of shape (batch_size, 3) representing roll, pitch, yaw angles.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Transformed translations of shape (batch_size, 3).
            - Transformed RPYs of shape (batch_size, 3).
    """
    # Define the transformation matrix P
    P = torch.tensor([
        [0, 0, 1],  # x_s -> z_t
        [1, 0, 0],  # y_s -> x_t
        [0, 1, 0]   # z_s -> y_t
    ], dtype=torch.float32, device=translations.device)

    # Convert rpys (roll, pitch, yaw) to rotation matrices
    R_source_batch = rpy_to_rotation_matrix(rpys)

    # Perform the transformation: R_target = P @ R_source
    R_target_batch = torch.matmul(P, R_source_batch)

    # Convert back to RPY (roll, pitch, yaw) using the helper function
    rpy_t_batch = rotation_matrix_to_rpy(R_target_batch)

    # Transform translations: translation_t = P @ translation
    translation_t_batch = torch.matmul(translations, P.T)

    return translation_t_batch, rpy_t_batch


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print("\n" + "="*40)
            print(
                f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            print("="*40 + "\n")
    return wrapper
