from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.data import FormulaVideoDataset, load_formula_table, load_or_create_split, save_split, split_frame
from src.losses import AOAFormulaLoss, DispFormulaLoss, total_loss
from src.models import build_model
from src.train_utils import append_log, evaluate_model, load_config, resolve_device, save_config


def compute_log_stats(train_df):
    import numpy as np

    eps = 1e-30
    logJ = np.log(train_df["J"].to_numpy(dtype="float64") + eps)
    logAOA = np.log(train_df["AOA_var"].to_numpy(dtype="float64") + eps)
    logDisp = np.log(train_df["disp_var"].to_numpy(dtype="float64") + eps)
    logDelta = np.log(train_df["Delta_theta"].to_numpy(dtype="float64") + eps)
    logD = np.log(train_df["D_aperture"].to_numpy(dtype="float64") + eps)
    residual = logAOA - logJ + (1.0 / 3.0) * logD
    stats = {
        "mu_logJ": float(logJ.mean()),
        "std_logJ": float(logJ.std()),
        "mu_logAOA": float(logAOA.mean()),
        "std_logAOA": float(logAOA.std()),
        "mu_logDisp": float(logDisp.mean()),
        "std_logDisp": float(logDisp.std()),
        "mu_logDelta": float(logDelta.mean()),
        "std_logDelta": float(logDelta.std()),
        "c0_init": float(residual.mean()),
        "oracle_aoa_residual_mean": float(residual.mean()),
        "oracle_aoa_residual_std": float(residual.std()),
        "oracle_aoa_residual_min": float(residual.min()),
        "oracle_aoa_residual_max": float(residual.max()),
    }
    return stats


def epoch_weights(base_weights, schedule, epoch):
    weights = dict(base_weights)
    if not schedule:
        return weights
    warmup_epochs = int(schedule.get("warmup_epochs", 0))
    formula_start = int(schedule.get("formula_start_epoch", warmup_epochs + 1))
    formula_ramp = max(1, int(schedule.get("formula_ramp_epochs", 1)))
    corr_start = int(schedule.get("corr_start_epoch", formula_start + formula_ramp))
    target_formula = float(base_weights.get("aoa_formula", 0.0))
    target_corr = float(base_weights.get("corr", 0.0))
    aux_start = int(schedule.get("aux_start_epoch", warmup_epochs + 1))
    aux_ramp = max(1, int(schedule.get("aux_ramp_epochs", 1)))
    if epoch < formula_start:
        weights["aoa_formula"] = 0.0
    else:
        ramp = min(1.0, max(0.0, (epoch - formula_start + 1) / formula_ramp))
        weights["aoa_formula"] = target_formula * ramp
    if epoch < corr_start:
        weights["corr"] = 0.0
    else:
        weights["corr"] = target_corr
    for key in ("disp_formula", "j_residual_d"):
        target = float(base_weights.get(key, 0.0))
        if epoch < aux_start:
            weights[key] = 0.0
        else:
            ramp = min(1.0, max(0.0, (epoch - aux_start + 1) / aux_ramp))
            weights[key] = target * ramp
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train latent Deflex model with SwinMAE encoder.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="outputs_latent_codex")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--init-ckpt", default=None, help="Optional checkpoint to initialize model weights from.")
    parser.add_argument("--max-train-video-views", type=int, default=None)
    parser.add_argument("--max-val-video-views", type=int, default=None)
    return parser.parse_args()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def checkpoint_payload(model, aoa_formula_loss, disp_formula_loss, config, epoch, score, stats):
    return {
        "model": unwrap_model(model).state_dict(),
        "aoa_formula_loss": aoa_formula_loss.state_dict(),
        "disp_formula_loss": disp_formula_loss.state_dict(),
        "config": config,
        "epoch": epoch,
        "score": score,
        "mu_logJ": stats["mu_logJ"],
        "std_logJ": stats["std_logJ"],
        "mu_logAOA": stats["mu_logAOA"],
        "std_logAOA": stats["std_logAOA"],
        "c0_init": stats["c0_init"],
        "c0_current": float(aoa_formula_loss.c0.detach().cpu()),
        "disp_formula_C_current": float(disp_formula_loss.log_c.detach().exp().cpu()),
    }


def selection_scores(row):
    def finite(value, default=0.0):
        value = float(value)
        return value if math.isfinite(value) else default

    corr_zj = abs(finite(row.get("corr_zJ_logJ", 0.0)))
    corr_zaoa = abs(finite(row.get("corr_zAOA_logAOA", 0.0)))
    loss_align = finite(row.get("loss_align", 0.0))
    r2_proxy = finite(row.get("R2_AOA_proxy_val", row.get("R2_disp_val", float("-inf"))), default=float("-inf"))
    residual_d_corr = abs(finite(row.get("corr_J_residual_logD", 0.0)))
    latent_corr = 0.5 * (corr_zj + corr_zaoa)
    return {
        "aoa_proxy": r2_proxy,
        "latent_corr": latent_corr,
        "disentangled": latent_corr - residual_d_corr,
        "composite": r2_proxy + 0.5 * corr_zj + 0.5 * corr_zaoa - 0.25 * residual_d_corr - 0.05 * loss_align,
    }


def optimizer_param_groups(model, aoa_formula_loss, disp_formula_loss, train_cfg):
    base_lr = float(train_cfg["lr"])
    encoder_lr = train_cfg.get("encoder_lr")
    if encoder_lr is None:
        params = (
            list(filter(lambda p: p.requires_grad, model.parameters()))
            + list(aoa_formula_loss.parameters())
            + list(disp_formula_loss.parameters())
        )
        return params

    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.") or name.startswith("module.encoder."):
            encoder_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr})
    groups.append({"params": list(aoa_formula_loss.parameters()), "lr": base_lr})
    groups.append({"params": list(disp_formula_loss.parameters()), "lr": base_lr})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": float(encoder_lr)})
    return groups


def optimizer_trainable_params(param_groups):
    if not param_groups:
        return []
    if isinstance(param_groups[0], dict):
        params = []
        for group in param_groups:
            params.extend(group["params"])
        return params
    return param_groups


def load_compatible_model_state(model, state_dict):
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"Skipped incompatible checkpoint tensors: {skipped}")
    if missing:
        print(f"Missing tensors initialized from current model: {missing}")
    if unexpected:
        print(f"Unexpected checkpoint tensors ignored: {unexpected}")


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Training requires PyTorch, OpenCV, and tqdm. Install requirements.txt or run in the "
            f"project environment with those packages available. Missing: {exc.name}"
        ) from exc

    args = parse_args()
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    train_cfg = config["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.max_train_video_views is not None:
        train_cfg["max_train_video_views"] = args.max_train_video_views
    if args.max_val_video_views is not None:
        train_cfg["max_val_video_views"] = args.max_val_video_views

    df = load_formula_table(data_cfg["csv"])
    split = load_or_create_split(
        df,
        out / "split_info.json",
        data_cfg["train_ratio"],
        data_cfg["val_ratio"],
        data_cfg["test_ratio"],
        data_cfg["random_state"],
    )
    save_split(split, out / "split_info.json")
    train_df = split_frame(df, split, "train")
    stats = compute_log_stats(train_df)
    print(
        "oracle AOA residual:",
        {
            key: stats[key]
            for key in [
                "oracle_aoa_residual_mean",
                "oracle_aoa_residual_std",
                "oracle_aoa_residual_min",
                "oracle_aoa_residual_max",
            ]
        },
    )
    config.setdefault("model", {}).update(
        {
            "mu_logJ": stats["mu_logJ"],
            "std_logJ": stats["std_logJ"],
            "mu_logAOA": stats["mu_logAOA"],
            "std_logAOA": stats["std_logAOA"],
            "c0_init": stats["c0_init"],
        }
    )
    if config["model"].get("disp_head_type") == "powerlaw":
        init_aoa_exp = float(config["model"].get("disp_powerlaw_init_aoa_exp", 1.0))
        init_delta_exp = float(config["model"].get("disp_powerlaw_init_delta_exp", -2.0))
        config["model"]["disp_powerlaw_init_log_const"] = float(
            stats["mu_logDisp"] - init_aoa_exp * stats["mu_logAOA"] - init_delta_exp * stats["mu_logDelta"]
        )
    config["oracle_aoa_residual"] = {
        key: stats[key]
        for key in [
            "oracle_aoa_residual_mean",
            "oracle_aoa_residual_std",
            "oracle_aoa_residual_min",
            "oracle_aoa_residual_max",
        ]
    }
    save_config(config, out / "config_used.yaml")

    train_ds = FormulaVideoDataset(
        df,
        split,
        "train",
        data_cfg["data_root"],
        frame_count=data_cfg["frame_count"],
        image_size=data_cfg["image_size"],
        max_video_views=train_cfg.get("max_train_video_views"),
    )
    val_ds = FormulaVideoDataset(
        df,
        split,
        "val",
        data_cfg["data_root"],
        frame_count=data_cfg["frame_count"],
        image_size=data_cfg["image_size"],
        max_video_views=train_cfg.get("max_val_video_views"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    device = resolve_device(train_cfg["device"])
    model = build_model(config).to(device)
    aoa_formula_loss = AOAFormulaLoss(
        mu_logJ=stats["mu_logJ"],
        std_logJ=stats["std_logJ"],
        c0_init=stats["c0_init"],
    ).to(device)
    disp_formula_loss = DispFormulaLoss(
        c_init=float(config["loss_weights"].get("disp_formula_c_init", 1.0))
    ).to(device)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        load_compatible_model_state(model, ckpt["model"])
        if "aoa_formula_loss" in ckpt:
            aoa_formula_loss.load_state_dict(ckpt["aoa_formula_loss"], strict=True)
        if "disp_formula_loss" in ckpt:
            disp_formula_loss.load_state_dict(ckpt["disp_formula_loss"], strict=True)
        print(f"Initialized from checkpoint: {args.init_ckpt}")
    if device.type == "cuda" and train_cfg.get("data_parallel", True) and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    param_groups = optimizer_param_groups(model, aoa_formula_loss, disp_formula_loss, train_cfg)
    trainable_params = optimizer_trainable_params(param_groups)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg["lr"],
    )
    grad_clip_norm = train_cfg.get("grad_clip_norm")

    best_scores = {
        "aoa_proxy": float("-inf"),
        "latent_corr": float("-inf"),
        "disentangled": float("-inf"),
        "composite": float("-inf"),
    }
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(train_cfg.get("save_every_epochs", 0) or 0)
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        current_weights = epoch_weights(config["loss_weights"], config.get("loss_schedule"), epoch)
        loss_sums = {}
        steps = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            frames = batch["frames"].to(device)
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if hasattr(value, "to") and key != "frames"
            }
            out_batch = model(frames, tensors["D_aperture"], tensors["alpha"], tensors["Delta_theta"])
            loss, parts = total_loss(out_batch, tensors, aoa_formula_loss, disp_formula_loss, current_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            optimizer.step()
            for key, value in parts.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
            steps += 1

        row = {"epoch": epoch}
        row.update({key: value / max(steps, 1) for key, value in loss_sums.items()})
        row.update(evaluate_model(model, val_loader, device))
        row["c0_current"] = float(aoa_formula_loss.c0.detach().cpu())
        row["disp_formula_C_current"] = float(disp_formula_loss.log_c.detach().exp().cpu())
        disp_head = unwrap_model(model).disp_head
        if hasattr(disp_head, "log_const"):
            row["disp_log_const_current"] = float(disp_head.log_const.detach().cpu())
            row["disp_aoa_exp_current"] = float(disp_head.aoa_exp.detach().cpu())
            row["disp_delta_exp_current"] = float(disp_head.delta_exp.detach().cpu())
        row["lambda_aoa_formula_current"] = float(current_weights.get("aoa_formula", 0.0))
        row["lambda_disp_formula_current"] = float(current_weights.get("disp_formula", 0.0))
        row["lambda_j_residual_d_current"] = float(current_weights.get("j_residual_d", 0.0))
        row["lambda_corr_current"] = float(current_weights.get("corr", 0.0))
        row.update({f"score_{key}": value for key, value in selection_scores(row).items()})
        append_log(out / "train_log.csv", row)
        scores = selection_scores(row)
        payload = None
        if save_every > 0 and epoch % save_every == 0:
            payload = checkpoint_payload(model, aoa_formula_loss, disp_formula_loss, config, epoch, scores["composite"], stats)
            torch.save(payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
        for name, score in scores.items():
            if score > best_scores[name]:
                best_scores[name] = score
                if payload is None:
                    payload = checkpoint_payload(model, aoa_formula_loss, disp_formula_loss, config, epoch, score, stats)
                else:
                    payload = dict(payload)
                    payload["score"] = score
                torch.save(payload, ckpt_dir / f"best_{name}.pt")
                if name == "composite":
                    torch.save(payload, ckpt_dir / "best.pt")
        print(row)


if __name__ == "__main__":
    main()
