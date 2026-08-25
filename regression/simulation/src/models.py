from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .deflex_blocks import require_torch

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@contextmanager
def _cwd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


if nn is not None:

    class CNNPatchEncoder(nn.Module):
        def __init__(self, embedding_dim: int = 64, patch_size: int = 16):
            super().__init__()
            self.patch_size = patch_size
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.proj = nn.Linear(64, embedding_dim)

        def forward(self, frames):
            b, t, c, h, w = frames.shape
            p = self.patch_size
            patches = frames.unfold(3, p, p).unfold(4, p, p)
            patches = patches.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
            gh, gw = patches.shape[2], patches.shape[3]
            patches = patches.reshape(b * t * gh * gw, c, p, p)
            feat = self.net(patches).flatten(1)
            feat = self.proj(feat)
            return feat.reshape(b, t, gh * gw, -1)


    class SwinMAEPatchEncoder(nn.Module):
        """Use 83/swin_mae.py as the image encoder and map its feature map to 14x14 patch embeddings."""

        def __init__(
            self,
            encoder_dir: str | Path,
            embedding_dim: int = 64,
            feature_index: int = 1,
            feature_dim: int = 512,
            target_grid: int = 14,
            freeze: bool = True,
        ):
            super().__init__()
            encoder_dir = Path(encoder_dir).resolve()
            if not (encoder_dir / "swin_mae.py").exists():
                raise FileNotFoundError(f"Cannot find swin_mae.py in {encoder_dir}")
            sys.path.insert(0, str(encoder_dir))
            try:
                from swin_mae import swin_mae
            finally:
                if sys.path[0] == str(encoder_dir):
                    sys.path.pop(0)

            # 83/swin_mae.py loads '83-checkpoint-499.pth' by relative path in __init__.
            with _cwd(encoder_dir):
                self.backbone = swin_mae()
            self._patch_legacy_fft()
            self.feature_index = feature_index
            self.target_grid = target_grid
            self.freeze = freeze
            if freeze:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.backbone.eval()

            self.proj = nn.Linear(feature_dim, embedding_dim)

        def _patch_legacy_fft(self):
            if hasattr(torch.fft, "fft2"):
                return
            patch_embed = getattr(self.backbone, "patch_embed", None)
            if patch_embed is None or not hasattr(patch_embed, "extract_freq_token"):
                return

            def extract_freq_token_legacy(module, x, target_H, target_W):
                b, c, h, w = x.shape
                freq = torch.rfft(x, signal_ndim=2, normalized=True, onesided=False)
                freq_mag = torch.sqrt((freq**2).sum(dim=-1) + 1e-12)
                low_h = h // 4
                low_w = w // 4
                low_freq = freq_mag[:, :, :low_h, :low_w]
                resized = F.interpolate(low_freq, size=(target_H, target_W), mode="bilinear", align_corners=False)
                freq_token = module.freq_proj(resized)
                return freq_token.permute(0, 2, 3, 1).contiguous()

            patch_embed.extract_freq_token = types.MethodType(extract_freq_token_legacy, patch_embed)

        def train(self, mode: bool = True):
            super().train(mode)
            if self.freeze:
                self.backbone.eval()
            return self

        def forward(self, frames):
            self._patch_legacy_fft()
            b, t, c, h, w = frames.shape
            flat = frames.reshape(b * t, c, h, w)
            grad_enabled = torch.is_grad_enabled() and not self.freeze
            with torch.set_grad_enabled(grad_enabled):
                feats = self.backbone(flat)
            if isinstance(feats, (tuple, list)):
                idx = min(max(self.feature_index, 0), len(feats) - 1)
                feat = feats[idx]
            else:
                feat = feats
            if feat.ndim != 4:
                raise RuntimeError(f"Expected Swin feature map [B,H,W,C] or [B,C,H,W], got {tuple(feat.shape)}")
            # 83/swin_mae.py returns [B,H,W,C]. Accept [B,C,H,W] too.
            if feat.shape[1] not in (7, 14, 28, 56) and feat.shape[-1] in (7, 14, 28, 56):
                feat = feat.permute(0, 2, 3, 1).contiguous()
            feat = feat.permute(0, 3, 1, 2).contiguous()
            if feat.shape[-2:] != (self.target_grid, self.target_grid):
                feat = F.adaptive_avg_pool2d(feat, (self.target_grid, self.target_grid))
            feat = feat.permute(0, 2, 3, 1).contiguous()
            feat = self.proj(feat)
            return feat.reshape(b, t, self.target_grid * self.target_grid, -1)


    class Bottleneck(nn.Module):
        def __init__(self, embedding_dim: int = 64, latent_dim: int = 4):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, latent_dim),
            )

        def forward(self, x):
            return self.net(x)


    class PositiveMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            hidden = max(32, input_dim * 4)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return F.softplus(self.net(x).squeeze(-1)) + 1e-12


    class LogMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            hidden = max(32, input_dim * 8)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


    class LearnableDispPowerLawHead(nn.Module):
        def __init__(
            self,
            extra_dim: int = 0,
            init_log_const: float = 0.0,
            init_aoa_exp: float = 1.0,
            init_delta_exp: float = -2.0,
            residual_scale: float = 0.0,
        ):
            super().__init__()
            self.log_const = nn.Parameter(torch.tensor(float(init_log_const)))
            self.aoa_exp = nn.Parameter(torch.tensor(float(init_aoa_exp)))
            self.delta_exp = nn.Parameter(torch.tensor(float(init_delta_exp)))
            self.residual_scale = float(residual_scale)
            if extra_dim > 0:
                self.extra_coef = nn.Parameter(torch.zeros(extra_dim))
            else:
                self.register_parameter("extra_coef", None)

        def forward(self, logAOA_latent, logDelta, extras=None):
            out = self.log_const + self.aoa_exp * logAOA_latent + self.delta_exp * logDelta
            if self.extra_coef is not None and extras is not None and self.residual_scale != 0.0:
                out = out + self.residual_scale * (extras * self.extra_coef).sum(dim=-1)
            return out


    class LatentTurbulenceModel(nn.Module):
        def __init__(
            self,
            encoder: nn.Module,
            embedding_dim: int = 64,
            latent_dim: int = 4,
            deflex_blocks: int = 2,
            attention_heads: int = 4,
            dropout: float = 0.1,
            mu_logJ: float = 0.0,
            std_logJ: float = 1.0,
            mu_logAOA: float = 0.0,
            std_logAOA: float = 1.0,
            disp_head_inputs: str = "rawH_delta",
            disp_head_type: str = "mlp",
            disp_powerlaw_init_log_const: float = 0.0,
            disp_powerlaw_init_aoa_exp: float = 1.0,
            disp_powerlaw_init_delta_exp: float = -2.0,
            disp_powerlaw_residual_scale: float = 0.0,
        ):
            super().__init__()
            from .deflex_blocks import TurbDeflexBlock

            self.eps = 1e-30
            self.encoder = encoder
            self.disp_head_inputs = disp_head_inputs
            self.disp_head_type = disp_head_type
            self.register_buffer("mu_logJ", torch.tensor(float(mu_logJ)))
            self.register_buffer("std_logJ", torch.tensor(float(std_logJ)))
            self.register_buffer("mu_logAOA", torch.tensor(float(mu_logAOA)))
            self.register_buffer("std_logAOA", torch.tensor(float(std_logAOA)))
            self.blocks = nn.ModuleList(
                [
                    TurbDeflexBlock(embedding_dim, n_heads=attention_heads, dropout=dropout)
                    for _ in range(deflex_blocks)
                ]
            )
            self.bottleneck = Bottleneck(embedding_dim, latent_dim)
            self.aoa_head = LogMLP(2)
            if disp_head_inputs == "rawH_delta":
                disp_input_dim = latent_dim + 1
                disp_extra_dim = 0
            elif disp_head_inputs == "aoa_delta":
                disp_input_dim = 2
                disp_extra_dim = 0
            elif disp_head_inputs == "aoa_extra_delta":
                disp_input_dim = latent_dim
                disp_extra_dim = max(0, latent_dim - 2)
            else:
                raise ValueError(f"Unknown disp_head_inputs: {disp_head_inputs}")
            if disp_head_type == "mlp":
                self.disp_head = LogMLP(disp_input_dim)
            elif disp_head_type == "powerlaw":
                if disp_head_inputs not in {"aoa_delta", "aoa_extra_delta"}:
                    raise ValueError("disp_head_type='powerlaw' requires disp_head_inputs='aoa_delta' or 'aoa_extra_delta'")
                self.disp_head = LearnableDispPowerLawHead(
                    extra_dim=disp_extra_dim,
                    init_log_const=disp_powerlaw_init_log_const,
                    init_aoa_exp=disp_powerlaw_init_aoa_exp,
                    init_delta_exp=disp_powerlaw_init_delta_exp,
                    residual_scale=disp_powerlaw_residual_scale,
                )
            else:
                raise ValueError(f"Unknown disp_head_type: {disp_head_type}")

        def forward(self, frames, D_aperture, alpha, Delta_theta):
            z_frame = self.encoder(frames)
            h_frame_repr = z_frame
            for block in self.blocks:
                h_frame_repr = block(h_frame_repr)
            patch_repr = h_frame_repr.mean(dim=1)
            raw_H = self.bottleneck(patch_repr)
            zJ = raw_H[..., 0]
            zAOA = raw_H[..., 1]
            logJ_latent = self.mu_logJ + self.std_logJ * zJ
            logAOA_latent = self.mu_logAOA + self.std_logAOA * zAOA
            J_latent = torch.exp(logJ_latent)
            AOA_latent = torch.exp(logAOA_latent)
            logD = torch.log(D_aperture + self.eps)
            logDelta = torch.log(Delta_theta + self.eps)
            aoa_in = torch.stack([logJ_latent, logD], dim=-1)
            disp_extras = None
            if self.disp_head_inputs == "rawH_delta":
                disp_in = torch.cat([raw_H, logDelta.unsqueeze(-1)], dim=-1)
            elif self.disp_head_inputs == "aoa_delta":
                disp_in = torch.stack([logAOA_latent, logDelta], dim=-1)
            elif self.disp_head_inputs == "aoa_extra_delta":
                extras = raw_H[..., 2:]
                disp_extras = extras
                disp_in = torch.cat([logAOA_latent.unsqueeze(-1), extras, logDelta.unsqueeze(-1)], dim=-1)
            else:
                raise ValueError(f"Unknown disp_head_inputs: {self.disp_head_inputs}")
            aoa_log_hat = self.aoa_head(aoa_in)
            if self.disp_head_type == "powerlaw":
                disp_log_hat = self.disp_head(logAOA_latent, logDelta, disp_extras)
            else:
                disp_log_hat = self.disp_head(disp_in)
            return {
                "H": raw_H,
                "raw_H": raw_H,
                "zJ": zJ,
                "zAOA": zAOA,
                "logJ_latent": logJ_latent,
                "logAOA_latent": logAOA_latent,
                "J_latent": J_latent,
                "AOA_latent": AOA_latent,
                "H_frame_repr": h_frame_repr,
                "aoa_log_hat": aoa_log_hat,
                "disp_log_hat": disp_log_hat,
                "aoa_hat": torch.exp(aoa_log_hat),
                "disp_hat": torch.exp(disp_log_hat),
                "disp_powerlaw_log_const": getattr(self.disp_head, "log_const", raw_H.new_tensor(float("nan"))),
                "disp_powerlaw_aoa_exp": getattr(self.disp_head, "aoa_exp", raw_H.new_tensor(float("nan"))),
                "disp_powerlaw_delta_exp": getattr(self.disp_head, "delta_exp", raw_H.new_tensor(float("nan"))),
            }


def build_encoder(config: dict):
    require_torch()
    model_cfg = config.get("model", config)
    name = model_cfg.get("encoder", "swin_mae")
    if name == "swin_mae":
        return SwinMAEPatchEncoder(
            encoder_dir=model_cfg["external_encoder_dir"],
            embedding_dim=int(model_cfg.get("embedding_dim", 64)),
            feature_index=int(model_cfg.get("swin_feature_index", 1)),
            feature_dim=int(model_cfg.get("swin_feature_dim", 512)),
            freeze=bool(model_cfg.get("freeze_encoder", True)),
        )
    if name == "cnn":
        return CNNPatchEncoder(
            embedding_dim=int(model_cfg.get("embedding_dim", 64)),
            patch_size=int(config.get("data", {}).get("patch_size", 16)),
        )
    raise ValueError(f"Unknown encoder: {name}")


def build_model(config: dict):
    require_torch()
    model_cfg = config.get("model", config)
    encoder = build_encoder(config)
    return LatentTurbulenceModel(
        encoder=encoder,
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        latent_dim=int(model_cfg.get("latent_dim", 4)),
        deflex_blocks=int(model_cfg.get("deflex_blocks", 2)),
        attention_heads=int(model_cfg.get("attention_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mu_logJ=float(model_cfg.get("mu_logJ", 0.0)),
        std_logJ=float(model_cfg.get("std_logJ", 1.0)),
        mu_logAOA=float(model_cfg.get("mu_logAOA", 0.0)),
        std_logAOA=float(model_cfg.get("std_logAOA", 1.0)),
        disp_head_inputs=str(model_cfg.get("disp_head_inputs", "rawH_delta")),
        disp_head_type=str(model_cfg.get("disp_head_type", "mlp")),
        disp_powerlaw_init_log_const=float(model_cfg.get("disp_powerlaw_init_log_const", 0.0)),
        disp_powerlaw_init_aoa_exp=float(model_cfg.get("disp_powerlaw_init_aoa_exp", 1.0)),
        disp_powerlaw_init_delta_exp=float(model_cfg.get("disp_powerlaw_init_delta_exp", -2.0)),
        disp_powerlaw_residual_scale=float(model_cfg.get("disp_powerlaw_residual_scale", 0.0)),
    )
