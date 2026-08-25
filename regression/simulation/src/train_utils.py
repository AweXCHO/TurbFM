from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

from .metrics import pearson_corr, regression_metrics


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def append_log(path: str | Path, row: Dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def evaluate_model(model, loader, device):
    import torch

    model.eval()
    aoa_true, aoa_pred, disp_true, disp_pred, delta_all = [], [], [], [], []
    zJ_all, zAOA_all, logJ_latent_all, logAOA_latent_all, J_all, D_all = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if hasattr(value, "to") and key != "frames"
            }
            out = model(frames, tensors["D_aperture"], tensors["alpha"], tensors["Delta_theta"])
            patch_mask = tensors.get("patch_mask")
            if patch_mask is None:
                patch_mask = torch.ones_like(out["zJ"], dtype=torch.bool)
            else:
                patch_mask = patch_mask.bool()
            aoa_true.append(tensor_to_numpy(tensors["AOA_var"][patch_mask]))
            aoa_pred.append(tensor_to_numpy(out["aoa_hat"][patch_mask]))
            disp_true.append(tensor_to_numpy(tensors["disp_var"][patch_mask]))
            disp_pred.append(tensor_to_numpy(out["disp_hat"][patch_mask]))
            delta_all.append(tensor_to_numpy(tensors["Delta_theta"][patch_mask]))
            zJ_all.append(tensor_to_numpy(out["zJ"][patch_mask]))
            zAOA_all.append(tensor_to_numpy(out["zAOA"][patch_mask]))
            logJ_latent_all.append(tensor_to_numpy(out["logJ_latent"][patch_mask]))
            logAOA_latent_all.append(tensor_to_numpy(out["logAOA_latent"][patch_mask]))
            J_all.append(tensor_to_numpy(tensors["J"][patch_mask]))
            D_all.append(tensor_to_numpy(tensors["D_aperture"][patch_mask]))

    aoa_true = np.concatenate([x.reshape(-1) for x in aoa_true])
    aoa_pred = np.concatenate([x.reshape(-1) for x in aoa_pred])
    disp_true = np.concatenate([x.reshape(-1) for x in disp_true])
    disp_pred = np.concatenate([x.reshape(-1) for x in disp_pred])
    delta_all = np.concatenate([x.reshape(-1) for x in delta_all])
    zJ_all = np.concatenate([x.reshape(-1) for x in zJ_all])
    zAOA_all = np.concatenate([x.reshape(-1) for x in zAOA_all])
    logJ_latent_all = np.concatenate([x.reshape(-1) for x in logJ_latent_all])
    logAOA_latent_all = np.concatenate([x.reshape(-1) for x in logAOA_latent_all])
    J_all = np.concatenate([x.reshape(-1) for x in J_all])
    D_all = np.concatenate([x.reshape(-1) for x in D_all])
    logJ_true = np.log(J_all + 1e-30)
    logAOA_true = np.log(aoa_true + 1e-30)
    metrics_aoa = regression_metrics(aoa_true, aoa_pred)
    metrics_disp = regression_metrics(disp_true, disp_pred)
    metrics_proxy = regression_metrics(disp_true * delta_all**2, disp_pred * delta_all**2)
    return {
        "R2_AOA_val": metrics_aoa["r2"],
        "R2_disp_val": metrics_disp["r2"],
        "R2_AOA_proxy_val": metrics_proxy["r2"],
        "MAE_AOA_val": metrics_aoa["mae"],
        "MAE_disp_val": metrics_disp["mae"],
        "corr_zJ_logJ": pearson_corr(zJ_all, logJ_true),
        "corr_zAOA_logAOA": pearson_corr(zAOA_all, logAOA_true),
        "corr_logJlatent_logJ": pearson_corr(logJ_latent_all, logJ_true),
        "corr_logAOAlatent_logAOA": pearson_corr(logAOA_latent_all, logAOA_true),
        "corr_J_residual_logD": pearson_corr(logJ_latent_all - logJ_true, np.log(D_all + 1e-30)),
    }
