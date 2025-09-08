import numpy as np


def quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to roll, pitch, and yaw (RPY) angles.
    Args:
    - quaternion: A numpy array of shape (4,).
    Returns:
    - euler_angles: A numpy array of shape (3,), representing roll, pitch, and yaw.
    """
    w, x, y, z = quaternion

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    euler_angles = np.array([roll, pitch, yaw])

    return euler_angles


def rpy_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Converts roll, pitch, and yaw (RPY) angles to a quaternion.
    Args:
    - euler: A numpy array of shape (3,), representing roll, pitch, and yaw.
    Returns:
    - quaternion: A numpy array of shape (4,), representing the quaternion (w, x, y, z).
    """
    r, p, y = euler

    cy = np.cos(y * 0.5)
    sy = np.sin(y * 0.5)
    cp = np.cos(p * 0.5)
    sp = np.sin(p * 0.5)
    cr = np.cos(r * 0.5)
    sr = np.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = np.array([qw, qx, qy, qz])

    return quaternion


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiplies two quaternions.
    Args:
    - q1: A numpy array of shape (4,).
    - q2: A numpy array of shape (4,).
    Returns:
    - quaternion: A numpy array of shape (4,).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    quaternion = np.array([w, x, y, z])
    quaternion /= np.linalg.norm(quaternion)

    return quaternion


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a quaternion.
    Args:
    - q: A numpy array of shape (4,).
    Returns:
    - q_inv: A numpy array of shape (4,).
    """
    w, x, y, z = q
    q_inv = np.array([w, -x, -y, -z])
    q_inv /= np.linalg.norm(q)

    return q_inv


def quaternion_rotate_z(q: np.ndarray) -> np.ndarray:
    """
    Rotates the z-normal vector (0, 0, 1) using the given quaternion.
    Args:
    - q: A numpy array of shape (4,).
    Returns:
    - rotated_normal: A numpy array of shape (3,).
    """
    w, x, y, z = q
    q /= np.linalg.norm(q)

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ])

    z_normal = np.array([0.0, 0.0, 1.0])

    rotated_normal = R @ z_normal

    return rotated_normal
