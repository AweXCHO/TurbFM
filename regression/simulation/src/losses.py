from __future__ import annotations

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


if nn is not None:

    EPS = 1e-30


    def corr_loss(x, y, eps: float = 1e-8):
        x = x.reshape(-1)
        y = y.reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        denom = x.std(unbiased=False) * y.std(unbiased=False)
        if denom.detach().item() < eps:
            return x.new_zeros(())
        corr = (x * y).mean() / (denom + eps)
        return 1.0 - corr.abs()


    def weighted_mean(loss, weight, eps: float = 1e-8):
        return (loss * weight).sum() / weight.sum().clamp_min(eps)


    def disp_sample_weight(disp, min_disp_var: float = 1e-4, min_weight: float = 0.1):
        weight = torch.sqrt(torch.clamp(disp / float(min_disp_var), max=1.0))
        return torch.clamp(weight, min=float(min_weight), max=1.0)


    class AOAFormulaLoss(nn.Module):
        def __init__(self, mu_logJ: float = 0.0, std_logJ: float = 1.0, c0_init: float = 0.0):
            super().__init__()
            self.register_buffer("mu_logJ", torch.tensor(float(mu_logJ)))
            self.register_buffer("std_logJ", torch.tensor(float(std_logJ)))
            self.c0 = nn.Parameter(torch.tensor(float(c0_init)))

        def forward(self, zJ, D_aperture, aoa_true):
            logJ_latent = self.mu_logJ + self.std_logJ * zJ
            logD = torch.log(D_aperture + EPS)
            logAOA_formula = self.c0 + logJ_latent - (1.0 / 3.0) * logD
            logAOA_true = torch.log(aoa_true + EPS)
            return F.smooth_l1_loss(logAOA_formula, logAOA_true)


    class DispFormulaLoss(nn.Module):
        """Soft prior: disp_var is proportional to AOA_latent / Delta_theta^2."""

        def __init__(self, c_init: float = 1.0):
            super().__init__()
            self.log_c = nn.Parameter(torch.log(torch.tensor(float(c_init)).clamp_min(EPS)))

        def forward(self, log_aoa_latent, delta_theta, disp_true):
            log_delta = torch.log(delta_theta + EPS)
            log_disp_formula = self.log_c + log_aoa_latent - 2.0 * log_delta
            log_disp_true = torch.log(disp_true + EPS)
            return F.smooth_l1_loss(log_disp_formula, log_disp_true)


    def squared_corr(x, y, eps: float = 1e-8):
        x = x.reshape(-1)
        y = y.reshape(-1)
        x = x - x.mean()
        y = y - y.mean()
        denom = x.std(unbiased=False) * y.std(unbiased=False)
        if denom.detach().item() < eps:
            return x.new_zeros(())
        corr = (x * y).mean() / (denom + eps)
        return corr.square()


    def total_loss(outputs, batch, aoa_formula_loss, disp_formula_loss, weights):
        patch_mask = batch.get("patch_mask")
        if patch_mask is None:
            patch_mask = torch.ones_like(outputs["zJ"], dtype=torch.bool)
        else:
            patch_mask = patch_mask.bool()
        raw_H = outputs["raw_H"][patch_mask]
        zJ = outputs["zJ"][patch_mask]
        zAOA = outputs["zAOA"][patch_mask]
        aoa_log_hat = outputs["aoa_log_hat"][patch_mask]
        disp_log_hat = outputs["disp_log_hat"][patch_mask]
        logJ_latent = outputs["logJ_latent"][patch_mask]
        logAOA_latent = outputs["logAOA_latent"][patch_mask]
        aoa = batch["AOA_var"][patch_mask]
        disp = batch["disp_var"][patch_mask]
        J = batch["J"][patch_mask]
        D = batch["D_aperture"][patch_mask]
        delta_theta = batch["Delta_theta"][patch_mask]
        log_aoa = torch.log(aoa + EPS)
        log_disp = torch.log(disp + EPS)
        logJ = torch.log(J + EPS)
        logD = torch.log(D + EPS)

        loss_aoa_task = F.smooth_l1_loss(aoa_log_hat, log_aoa)
        disp_loss_raw = F.smooth_l1_loss(disp_log_hat, log_disp, reduction="none")
        if weights.get("use_disp_weight", False):
            w_disp = disp_sample_weight(
                disp,
                min_disp_var=float(weights.get("min_disp_var", 1e-4)),
                min_weight=float(weights.get("min_disp_weight", 0.1)),
            )
            loss_disp_task = weighted_mean(disp_loss_raw, w_disp)
            disp_weight_mean = w_disp.mean()
        else:
            loss_disp_task = disp_loss_raw.mean()
            disp_weight_mean = disp_loss_raw.new_ones(())
        loss_task = loss_aoa_task + loss_disp_task
        if weights.get("aoa_formula", 0.0) != 0.0:
            loss_aoa_formula = aoa_formula_loss(zJ, D, aoa)
        else:
            loss_aoa_formula = raw_H.new_zeros(())
        if weights.get("disp_formula", 0.0) != 0.0:
            loss_disp_formula = disp_formula_loss(logAOA_latent, delta_theta, disp)
        else:
            loss_disp_formula = raw_H.new_zeros(())
        loss_J_align = F.smooth_l1_loss(logJ_latent, logJ)
        loss_AOA_align = F.smooth_l1_loss(logAOA_latent, log_aoa)
        loss_align = loss_J_align + loss_AOA_align
        loss_corr = corr_loss(zJ, logJ) + corr_loss(zAOA, log_aoa)
        loss_J_residual_D = squared_corr(logJ_latent - logJ, logD)
        loss_temp = raw_H.new_zeros(())
        loss_compact = (raw_H**2).mean()

        total = (
            weights.get("task", 1.0) * loss_task
            + weights.get("aoa_formula", 0.2) * loss_aoa_formula
            + weights.get("disp_formula", 0.0) * loss_disp_formula
            + weights.get("corr", 0.05) * loss_corr
            + weights.get("align", weights.get("latent_align", 0.0)) * loss_align
            + weights.get("j_residual_d", 0.0) * loss_J_residual_D
            + weights.get("temporal", 0.0) * loss_temp
            + weights.get("compact", 0.001) * loss_compact
        )
        return total, {
            "loss_total": total.detach(),
            "loss_task": loss_task.detach(),
            "loss_aoa_task": loss_aoa_task.detach(),
            "loss_disp_task": loss_disp_task.detach(),
            "disp_weight_mean": disp_weight_mean.detach(),
            "loss_aoa_formula": loss_aoa_formula.detach(),
            "loss_disp_formula": loss_disp_formula.detach(),
            "loss_align": loss_align.detach(),
            "loss_J_align": loss_J_align.detach(),
            "loss_AOA_align": loss_AOA_align.detach(),
            "loss_corr": loss_corr.detach(),
            "loss_J_residual_D": loss_J_residual_D.detach(),
            "loss_temp": loss_temp.detach(),
            "loss_compact": loss_compact.detach(),
        }

else:

    class AOAFormulaLoss:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()

    class DispFormulaLoss:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()

    def total_loss(*args, **kwargs):  # pragma: no cover
        require_torch()
