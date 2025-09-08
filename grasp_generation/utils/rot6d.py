import torch
import torch.nn.functional as F


@torch.jit.script
def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v, dim=1, eps=1e-8)


@torch.jit.script
def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.cross(u, v, dim=1)


@torch.jit.script
def compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    x_raw = poses[:, 0:3]  # (batch, 3)
    y_raw = poses[:, 3:6]  # (batch, 3)

    x = normalize_vector(x_raw)  # (batch, 3)
    z = cross_product(x, y_raw)  # (batch, 3)
    z = normalize_vector(z)  # (batch, 3)
    y = cross_product(z, x)  # (batch, 3)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # (batch, 3, 3)
    return matrix


@torch.jit.script
def robust_compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # (batch, 3)
    y_raw = poses[:, 3:6]  # (batch, 3)

    x = normalize_vector(x_raw)  # (batch, 3)
    y = normalize_vector(y_raw)  # (batch, 3)
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # (batch, 3, 3)
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix
