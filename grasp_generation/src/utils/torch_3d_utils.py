import torch


@torch.jit.script
def quaternion_to_rpy(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # Clamp to [-1, 1]
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    euler_angles: torch.Tensor = torch.stack((roll, pitch, yaw), dim=-1)

    return euler_angles


@torch.jit.script
def rpy_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    r, p, y = torch.unbind(euler, dim=-1)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)

    return quaternion


@torch.jit.script
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Performs batch quaternion multiplication.

    This function returns the quaternion product of q1 and q2, representing
    a rotation where q2 is applied first, followed by q1.

    Parameters:
    - q1: A tensor of shape (..., 4), representing the first quaternion or a batch of quaternions.
    - q2: A tensor of shape (..., 4), representing the second quaternion or a batch of quaternions.

    Returns:
    - quaternion: A tensor of shape (..., 4), representing the product of q1 and q2,
      normalized to unit length to prevent numerical errors.
    """
    # Extract individual components
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)

    # Compute the product components
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack components back into a tensor
    quaternion = torch.stack((w, x, y, z), dim=-1)

    # Normalize the quaternion to unit length
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)

    return quaternion


@torch.jit.script
def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Computes the inverse of a quaternion.

    For unit quaternions representing rotations, the inverse is equal to the conjugate.
    This function supports batched inputs and normalizes the quaternion to prevent numerical errors.

    Parameters:
    - q: A tensor of shape (..., 4), representing a quaternion or a batch of quaternions.

    Returns:
    - q_inv: A tensor of shape (..., 4), representing the inverse of the input quaternion(s).
    """
    # Normalize the quaternion to unit length
    q = torch.nn.functional.normalize(q, dim=-1)

    # Extract individual components
    w, x, y, z = torch.unbind(q, dim=-1)

    # Compute the conjugate (inverse for unit quaternions)
    q_inv = torch.stack((w, -x, -y, -z), dim=-1)

    return q_inv


@torch.jit.script
def rpy_to_rotation_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """
    Converts batched roll, pitch, yaw angles to the corresponding rotation matrix.

    The rotation is assumed to be applied in the order:
        Rz(yaw) * Ry(pitch) * Rx(roll)

    Parameters:
    - rpy: A tensor of shape (..., 3), where rpy[..., 0] = roll, rpy[..., 1] = pitch,
      rpy[..., 2] = yaw.

    Returns:
    - R: A tensor of shape (..., 3, 3) representing the rotation matrices.
    """
    # Unpack angles
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    # Precompute sines/cosines
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    # Rotation matrix in row-major form for Rz(yaw) * Ry(pitch) * Rx(roll)
    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr

    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr

    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr

    # Stack into a single tensor of shape (..., 3, 3)
    R = torch.stack(
        [
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1),
        ],
        dim=-2,
    )

    return R


@torch.jit.script
def rotation_matrix_to_rpy(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a batched 3×3 rotation matrix to roll, pitch, yaw angles.

    Assumes the matrix was constructed by the sequence:
        Rz(yaw) * Ry(pitch) * Rx(roll)

    Parameters:
    - R: A tensor of shape (..., 3, 3) representing rotation matrices.

    Returns:
    - rpy: A tensor of shape (..., 3), where rpy[..., 0] = roll, rpy[..., 1] = pitch,
      rpy[..., 2] = yaw.
    """
    # For R = Rz(yaw) * Ry(pitch) * Rx(roll):
    #   R[2,0] = -sin(pitch)
    #   R[2,1] =  sin(roll)*cos(pitch)
    #   R[2,2] =  cos(roll)*cos(pitch)
    #   R[1,0] =  sin(yaw)*cos(pitch)
    #   R[0,0] =  cos(yaw)*cos(pitch)

    # pitch = asin(-R[2,0])
    sinp = -R[..., 2, 0]
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # roll = atan2(R[2,1], R[2,2])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])

    # yaw = atan2(R[1,0], R[0,0])
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])

    # Stack results into (..., 3)
    rpy = torch.stack((roll, pitch, yaw), dim=-1)

    return rpy
