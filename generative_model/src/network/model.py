import torch
import torch.nn as nn

from src.network.pointnet2 import PointNet2


class MyModel(nn.Module):
    def __init__(self, model: nn.Module):

        super().__init__()
        self.encoder = PointNet2()  # PointNet2
        self.model = model      # UNetModel


        n_obj = 2
        embd_len = 32
        # 1 x n_obj x 1
        embd_indexer = torch.arange(n_obj).unsqueeze(0).unsqueeze(2).cuda()
        self.register_buffer("embd_indexer", embd_indexer)

        self.obj_embd = nn.Embedding(n_obj, embd_len)

    def forward(self, x: torch.Tensor, t: torch.Tensor, point_cloud: torch.Tensor):
        """
        Args:
            x (torch.Tensor):         [B, d_x]
            t (torch.Tensor):         [B]
            point_cloud (torch.Tensor): [B, M, N, C]
               - B: batch size
               - M: number of objects
               - N: number of points per object (e.g. 1024)
               - C: point cloud coordinate dimension (usually 3)

        Returns:
            pred_noise: [B, d_x]
        """
        B, M, N, C = point_cloud.shape
        pc_reshaped = point_cloud.view(B * M, N, C)  # => [B*M, N, C]

        pc_features: torch.Tensor = self.encoder(
            pc_reshaped)      # => [B*M, feat_dim, n_samples]

        # => [B, M, feat_dim, n_samples]
        pc_features = pc_features.view(B, M, -1, pc_features.shape[-1])
        # => [B, M, n_samples, feat_dim]
        pc_features = pc_features.permute(0, 1, 3, 2)

        obj_embd = self.obj_embd(self.embd_indexer.tile(
            [B, 1, self.encoder.out_samples]))  # [B, M,n_samples, embd_len]

        # [B, M, n_samples, feat_dim+embd_len]
        object_features = torch.cat([pc_features, obj_embd], dim=-1)

        # => [B, M*n_samples, feat_dim+embd_len]
        object_features = object_features.reshape(
            B, M * object_features.shape[2], -1)

        pred_noise = self.model(x, t, object_features)
        return pred_noise
