import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import trimesh


def batched_sample_points(mesh: trimesh.Trimesh, batch_size: int, n_points: int, device="cuda"):
    """
    True batched sampling of points from a mesh, ensuring independent sampling for each batch.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        batch_size (int): Number of batches.
        n_points (int): Number of points to sample per batch.
        device (str, optional): Device to use for PyTorch tensors. Defaults to "cuda".

    Returns:
        torch.Tensor: Sampled point cloud of shape (batch_size, n_points, 3).
    """
    # Convert trimesh to PyTorch3D Meshes object
    verts = torch.tensor(mesh.vertices, dtype=torch.float32,
                         device=device)  # (V, 3)
    faces = torch.tensor(mesh.faces, dtype=torch.int64,
                         device=device)       # (F, 3)
    verts_batch = verts.unsqueeze(0).repeat(
        batch_size, 1, 1)               # (B, V, 3)
    faces_batch = faces.unsqueeze(0).repeat(
        batch_size, 1, 1)               # (B, F, 3)

    # Create batched mesh
    pytorch3d_mesh = Meshes(verts_batch, faces_batch)

    # Perform independent sampling for each batch
    sampled_points = sample_points_from_meshes(
        pytorch3d_mesh, num_samples=n_points)  # (B, n_points, 3)

    return sampled_points
