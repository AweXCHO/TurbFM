from __future__ import annotations

try:
    import torch
except Exception:
    torch = None


def patchify_frames(frames, patch_size: int = 16):
    """Convert [B,T,C,H,W] frames into [B,T,N,C,p,p] patches."""
    if torch is None:
        raise RuntimeError("PyTorch is required for patchify_frames.")
    b, t, c, h, w = frames.shape
    p = patch_size
    patches = frames.unfold(3, p, p).unfold(4, p, p)
    patches = patches.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
    return patches.reshape(b, t, -1, c, p, p)
