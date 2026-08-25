from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - import guard for symbolic-only envs
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


def require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for model training/extraction. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


if nn is not None:

    class TurbDeflexBlock(nn.Module):
        """Deflex-style block for tensors shaped [B, T, N, d]."""

        def __init__(self, embedding_dim: int, n_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.spatial_attn = nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout)
            self.temporal_attn = nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout)
            self.ffn = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim * 4, embedding_dim),
                nn.Dropout(dropout),
            )
            self.norm_spatial = nn.LayerNorm(embedding_dim)
            self.norm_temporal = nn.LayerNorm(embedding_dim)
            self.norm_ffn = nn.LayerNorm(embedding_dim)

        def forward(self, x):
            b, t, n, d = x.shape
            spatial_in = x.reshape(b * t, n, d)
            spatial_seq = spatial_in.transpose(0, 1)
            spatial_out, _ = self.spatial_attn(spatial_seq, spatial_seq, spatial_seq, need_weights=False)
            spatial_out = spatial_out.transpose(0, 1)
            x = self.norm_spatial((spatial_in + spatial_out).reshape(b, t, n, d))

            temporal_in = x.permute(0, 2, 1, 3).contiguous().reshape(b * n, t, d)
            temporal_seq = temporal_in.transpose(0, 1)
            temporal_out, _ = self.temporal_attn(temporal_seq, temporal_seq, temporal_seq, need_weights=False)
            temporal_out = temporal_out.transpose(0, 1)
            x = self.norm_temporal(
                (temporal_in + temporal_out).reshape(b, n, t, d).permute(0, 2, 1, 3).contiguous()
            )
            return self.norm_ffn(x + self.ffn(x))

else:

    class TurbDeflexBlock:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()
