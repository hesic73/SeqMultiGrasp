"""
The code is borrowed from Yuyang Li's codebase.
"""

from typing import Optional
from einops import rearrange
import torch
import torch.nn as nn
from .utils import timestep_embedding, ResBlock, SpatialTransformer


class UNetModel(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_model: int,
        nblocks: int,
        resblock_dropout: float,
        transformer_num_heads: int,
        transformer_dim_head: int,
        transformer_dropout: float,
        transformer_depth: int,
        transformer_mult_ff: float,
        time_embed_mult: int,
        context_dim: Optional[int] = None,
        use_position_embedding: bool = True,
    ):
        super(UNetModel, self).__init__()

        self.d_x = d_x
        self.d_model = d_model
        self.nblocks = nblocks
        self.resblock_dropout = resblock_dropout
        self.transformer_num_heads = transformer_num_heads
        self.transformer_dim_head = transformer_dim_head
        self.transformer_dropout = transformer_dropout
        self.transformer_depth = transformer_depth
        self.transformer_mult_ff = transformer_mult_ff
        self.context_dim = context_dim
        self.use_position_embedding = use_position_embedding

        time_embed_dim = self.d_model * time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for _ in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model,
                    self.transformer_num_heads,
                    self.transformer_dim_head,
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        """ Apply the model to an input batch

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature

        Return:
            the denoised target data, i.e., $x_{t-1}$
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        # time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)

        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h)  # <B, d_model, L>

        # prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX)  # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0)  # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        # reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h
