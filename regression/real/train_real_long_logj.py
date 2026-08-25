from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

import train_real_joint as joint


SOURCES = joint.LONG_SOURCES
RLRAT = SOURCES[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two long-range datasets with a calibrated J target."
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config_real_long_logj.yaml"))
    )
    parser.add_argument(
        "--stage", choices=("all", "prepare", "train", "evaluate"), default="all"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    temporary.replace(path)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def discover_scene_map(root: Path) -> dict[str, str]:
    result = {}
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        for path in directory.glob("*.avi"):
            if path.stem in result:
                raise RuntimeError(f"Duplicate grouped video: {path.stem}")
            result[path.stem] = directory.name
    if not result:
        raise RuntimeError(f"No grouped RLRAT scenes found under {root}")
    return result


def load_base_data(config: dict, max_videos: int | None):
    base = Path(config["data"]["base_experiment"])
    records = joint.discover_long_records(config, max_videos)
    split = json.loads((base / "splits.json").read_text(encoding="utf-8"))
    split = {source: split[source] for source in SOURCES}
    caches = {
        source: joint.cache_to_dict(base / "proxy_cache" / f"{source}.npz")
        for source in SOURCES
    }
    return records, split, caches


def transformed_stats(
    keys: list[str], cache: dict, j_floor: float, target_mode: str
) -> dict:
    entries = [cache[key] for key in keys if key in cache]
    aoa = joint.weighted_stats(
        (entry["aoa_log"] for entry in entries),
        (entry["confidence"] for entry in entries),
    )
    if target_mode == "direct_proxy":
        j_proxy = joint.weighted_stats(
            (entry["j_log"] for entry in entries),
            (
                entry["confidence"] * np.isfinite(entry["j_log"])
                for entry in entries
            ),
        )
        clip_logs = []
        for entry in entries:
            value = entry["j_log"]
            weight = entry["confidence"] * np.isfinite(value)
            weight_sum = float(weight.sum())
            if weight_sum <= 0:
                continue
            clip_value = float((np.nan_to_num(value) * weight).sum() / weight_sum)
            if clip_value > j_floor:
                clip_logs.append(math.log(clip_value))
        if len(clip_logs) < 3:
            raise RuntimeError("Not enough positive clip-level J proxies")
        clip_logs_array = np.asarray(clip_logs, dtype=np.float64)
        return {
            "aoa_log": aoa,
            "j_proxy": j_proxy,
            "j_clip_log": {
                "mean": float(clip_logs_array.mean()),
                "std": float(max(clip_logs_array.std(), 1e-6)),
                "weight_sum": float(len(clip_logs_array)),
            },
            "j_positive_floor": j_floor,
            "j_target_mode": target_mode,
            "train_records": len(entries),
        }
    if target_mode != "log_positive_proxy":
        raise ValueError(f"Unsupported J target mode: {target_mode}")
    j_values = []
    j_weights = []
    for entry in entries:
        strength = entry["j_log"]
        valid = np.isfinite(strength) & (strength > 0.0)
        j_values.append(np.log(np.maximum(strength, j_floor)))
        j_weights.append(entry["confidence"] * valid)
    j_log_strength = joint.weighted_stats(j_values, j_weights)
    return {
        "aoa_log": aoa,
        "j_log_strength": j_log_strength,
        "j_positive_floor": j_floor,
        "j_target_mode": target_mode,
        "train_records": len(entries),
    }


def prepare(config: dict, out: Path, max_videos: int | None) -> None:
    records, split, caches = load_base_data(config, max_videos)
    floor = float(config["proxy"]["j_positive_floor"])
    target_mode = str(config["proxy"].get("j_target_mode", "log_positive_proxy"))
    stats = {}
    counts = {}
    for source in SOURCES:
        train_keys = [key for key in split[source]["train"] if key in caches[source]]
        stats[source] = transformed_stats(
            train_keys, caches[source], floor, target_mode
        )
        counts[source] = {
            part: sum(key in caches[source] for key in split[source][part])
            for part in ("train", "val", "test")
        }
    save_json(split, out / "splits.json")
    save_json(stats, out / "proxy_stats.json")
    save_json(counts, out / "dataset_counts.json")
    save_json(
        {
            "cache_source": str(Path(config["data"]["base_experiment"]).resolve()),
            "sources": list(SOURCES),
            "heat_chamber_included": False,
            "j_target_mode": target_mode,
            "j_target": (
                "raw log(lucky-gradient / mean-gradient) proxy"
                if target_mode == "direct_proxy"
                else "log(max(positive J proxy, floor))"
            ),
        },
        out / "proxy_cache_source.json",
    )
    print("[prepare] counts:", json.dumps(counts, indent=2), flush=True)
    print("[prepare] stats:", json.dumps(stats, indent=2), flush=True)


class LongLogJDataset(Dataset):
    def __init__(
        self,
        records: list[joint.Record],
        keys: list[str],
        cache: dict,
        stats: dict,
        config: dict,
        scene_map: dict[str, str],
        training: bool,
    ):
        record_map = {record.key: record for record in records}
        self.records = [
            record_map[key] for key in keys if key in record_map and key in cache
        ]
        self.cache = cache
        self.stats = stats
        self.image_size = int(config["data"]["image_size"])
        self.frame_count = int(config["data"]["model_frames"])
        self.j_floor = float(config["proxy"]["j_positive_floor"])
        self.j_target_mode = str(
            config["proxy"].get("j_target_mode", "log_positive_proxy")
        )
        self.scene_map = scene_map
        self.training = training
        if not self.records:
            raise RuntimeError("LongLogJDataset has no records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        frames = joint.read_avi_rgb(record.path, self.frame_count, self.training)
        resized = np.stack(
            [
                joint.cv2.resize(
                    frame,
                    (self.image_size, self.image_size),
                    interpolation=joint.cv2.INTER_AREA,
                )
                for frame in frames
            ]
        ).astype(np.float32)
        target = self.cache[record.key]
        aoa = target["aoa_log"].copy()
        j_proxy = target["j_log"].copy()
        confidence = target["confidence"].copy()
        if self.j_target_mode == "direct_proxy":
            j_valid = np.isfinite(j_proxy)
            j_value = np.where(j_valid, j_proxy, 0.0)
            j_stats = self.stats["j_proxy"]
        else:
            j_valid = np.isfinite(j_proxy) & (j_proxy > 0.0)
            j_value = np.log(np.maximum(j_proxy, self.j_floor))
            j_stats = self.stats["j_log_strength"]

        if self.training and random.random() < 0.5:
            resized = resized[:, :, ::-1, :].copy()
            aoa = aoa[:, ::-1].copy()
            j_value = j_value[:, ::-1].copy()
            confidence = confidence[:, ::-1].copy()
            j_valid = j_valid[:, ::-1].copy()
        if self.training:
            gain = random.uniform(0.90, 1.10)
            bias = random.uniform(-5.0, 5.0)
            resized = np.clip(resized * gain + bias, 0.0, 255.0)

        aoa_stats = self.stats["aoa_log"]
        aoa_z = (aoa - aoa_stats["mean"]) / aoa_stats["std"]
        j_z = (j_value - j_stats["mean"]) / j_stats["std"]
        frames_chw = resized.transpose(0, 3, 1, 2) / 255.0
        if record.source == RLRAT:
            scene_group = self.scene_map.get(
                Path(record.path).stem, f"unassigned_{Path(record.path).stem}"
            )
        else:
            scene_group = record.source
        return {
            "frames": torch.from_numpy(frames_chw.astype(np.float32)),
            "aoa_target": torch.from_numpy(aoa_z.astype(np.float32)).reshape(-1),
            "j_target": torch.from_numpy(j_z.astype(np.float32)).reshape(-1),
            "confidence": torch.from_numpy(confidence.astype(np.float32)).reshape(-1),
            "j_valid": torch.from_numpy(j_valid.astype(np.float32)).reshape(-1),
            "j_log_mean": torch.tensor(
                float(j_stats["mean"]), dtype=torch.float32
            ),
            "j_log_std": torch.tensor(
                float(j_stats["std"]), dtype=torch.float32
            ),
            "j_clip_log_mean": torch.tensor(
                float(
                    self.stats.get("j_clip_log", j_stats)["mean"]
                ),
                dtype=torch.float32,
            ),
            "j_clip_log_std": torch.tensor(
                float(
                    self.stats.get("j_clip_log", j_stats)["std"]
                ),
                dtype=torch.float32,
            ),
            "j_direct": torch.tensor(
                self.j_target_mode == "direct_proxy", dtype=torch.bool
            ),
            "key": record.key,
            "source": record.source,
            "scene_group": scene_group,
        }


class GroupedLongBatchSampler(Sampler):
    def __init__(
        self,
        sequence_dataset: LongLogJDataset,
        rlrat_dataset: LongLogJDataset,
        batch_size: int,
        steps: int,
        rlrat_groups_per_batch: int,
        seed: int,
    ):
        if batch_size % 2:
            raise ValueError("batch_size must be even")
        self.sequence_count = len(sequence_dataset)
        self.rlrat_offset = self.sequence_count
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed
        self.epoch = 0
        self.rlrat_count = batch_size // 2
        self.sequence_batch_count = batch_size - self.rlrat_count
        self.groups_per_batch = rlrat_groups_per_batch
        if self.rlrat_count % self.groups_per_batch:
            raise ValueError("RLRAT half-batch must divide by rlrat_groups_per_batch")
        self.clips_per_group = self.rlrat_count // self.groups_per_batch
        grouped = defaultdict(list)
        for index, record in enumerate(rlrat_dataset.records):
            group = rlrat_dataset.scene_map.get(
                Path(record.path).stem, f"unassigned_{Path(record.path).stem}"
            )
            grouped[group].append(index)
        self.group_names = np.asarray(sorted(grouped))
        self.group_indices = [np.asarray(grouped[name]) for name in self.group_names]
        sizes = np.asarray([len(indices) for indices in self.group_indices], np.float64)
        self.group_probabilities = sizes / sizes.sum()

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps):
            sequence = rng.integers(
                0, self.sequence_count, self.sequence_batch_count
            ).tolist()
            selected_groups = rng.choice(
                len(self.group_indices),
                self.groups_per_batch,
                replace=False,
                p=self.group_probabilities,
            )
            rlrat = []
            for group_index in selected_groups:
                candidates = self.group_indices[int(group_index)]
                chosen = rng.choice(
                    candidates,
                    self.clips_per_group,
                    replace=len(candidates) < self.clips_per_group,
                )
                rlrat.extend((chosen + self.rlrat_offset).tolist())
            batch = sequence + rlrat
            rng.shuffle(batch)
            yield batch


def build_datasets(
    config: dict,
    stats: dict,
    records: dict,
    split: dict,
    caches: dict,
    scene_map: dict,
    part: str,
    training: bool,
):
    datasets = {}
    for source in SOURCES:
        keys = [key for key in split[source][part] if key in caches[source]]
        datasets[source] = LongLogJDataset(
            records[source],
            keys,
            caches[source],
            stats[source],
            config,
            scene_map,
            training,
        )
    return datasets


def make_datasets(config: dict, out: Path, max_videos: int | None, part: str, training: bool):
    records, split, caches = load_base_data(config, max_videos)
    stats = json.loads((out / "proxy_stats.json").read_text(encoding="utf-8"))
    scene_map = discover_scene_map(Path(config["data"]["grouped_scenes"]))
    return (
        build_datasets(
            config,
            stats,
            records,
            split,
            caches,
            scene_map,
            part,
            training,
        ),
        stats,
    )


def weighted_slope_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    weight = weight.reshape(-1)
    valid = (
        torch.isfinite(prediction)
        & torch.isfinite(target)
        & torch.isfinite(weight)
        & (weight > 0)
    )
    if int(valid.sum()) < 3:
        return None
    prediction, target, weight = (
        prediction[valid],
        target[valid],
        weight[valid],
    )
    weight_sum = weight.sum().clamp_min(1e-6)
    target_mean = (target * weight).sum() / weight_sum
    prediction_mean = (prediction * weight).sum() / weight_sum
    target_center = target - target_mean
    prediction_center = prediction - prediction_mean
    target_var = (weight * target_center.square()).sum() / weight_sum
    if float(target_var.detach()) < 1e-5:
        return None
    covariance = (weight * target_center * prediction_center).sum() / weight_sum
    slope = (covariance / target_var.clamp_min(1e-6)).clamp(-2.0, 4.0)
    return F.smooth_l1_loss(slope, torch.ones_like(slope))


def weighted_clip_mean(
    value: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)


def j_strength_clip_z(
    value_z: torch.Tensor, batch: dict
) -> torch.Tensor:
    weight = batch["confidence"] * batch["j_valid"]
    mean = batch["j_log_mean"].unsqueeze(1)
    std = batch["j_log_std"].unsqueeze(1)
    if bool(batch["j_direct"][0]):
        patch_proxy = value_z * std + mean
        clip_proxy = weighted_clip_mean(patch_proxy, weight)
        clip_log = torch.log(clip_proxy.clamp_min(1e-6))
        return (
            clip_log - batch["j_clip_log_mean"]
        ) / batch["j_clip_log_std"]
    patch_log_strength = value_z * std + mean
    log_weight = torch.log(weight.clamp_min(1e-30))
    clip_log_strength = torch.logsumexp(
        patch_log_strength + log_weight, dim=1
    ) - torch.log(weight.sum(dim=1).clamp_min(1e-30))
    return (clip_log_strength - batch["j_log_mean"]) / batch["j_log_std"]


def positive_j_clip_mask(batch: dict) -> torch.Tensor:
    if not bool(batch["j_direct"][0]):
        return torch.ones(
            batch["j_target"].shape[0],
            dtype=torch.bool,
            device=batch["j_target"].device,
        )
    weight = batch["confidence"] * batch["j_valid"]
    mean = batch["j_log_mean"].unsqueeze(1)
    std = batch["j_log_std"].unsqueeze(1)
    target_proxy = batch["j_target"] * std + mean
    return weighted_clip_mean(target_proxy, weight) > 1e-6


def grouped_scale_loss(
    output: dict, batch: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    confidence = batch["confidence"]
    j_weight = confidence * batch["j_valid"]
    patch_losses = []
    for source in SOURCES:
        indices = [i for i, value in enumerate(batch["source"]) if value == source]
        if not indices:
            continue
        index = torch.as_tensor(indices, device=confidence.device)
        for prediction, target, weight in (
            (
                output["aoa_z"].index_select(0, index),
                batch["aoa_target"].index_select(0, index),
                confidence.index_select(0, index),
            ),
            (
                output["j_z"].index_select(0, index),
                batch["j_target"].index_select(0, index),
                j_weight.index_select(0, index),
            ),
        ):
            value = weighted_slope_loss(prediction, target, weight)
            if value is not None:
                patch_losses.append(value)

    aoa_clip_prediction = weighted_clip_mean(output["aoa_z"], confidence)
    aoa_clip_target = weighted_clip_mean(batch["aoa_target"], confidence)
    j_clip_prediction = j_strength_clip_z(output["j_z"], batch)
    j_clip_target = j_strength_clip_z(batch["j_target"], batch)
    j_clip_valid = positive_j_clip_mask(batch)
    clip_losses = []
    sequence_indices = [
        i for i, source in enumerate(batch["source"]) if source == SOURCES[0]
    ]
    groups = [sequence_indices]
    rlrat_groups = defaultdict(list)
    for index, source in enumerate(batch["source"]):
        if source == RLRAT:
            rlrat_groups[batch["scene_group"][index]].append(index)
    groups.extend(rlrat_groups.values())
    for indices in groups:
        if len(indices) < 3:
            continue
        index = torch.as_tensor(indices, device=confidence.device)
        unit_weight = torch.ones(len(indices), device=confidence.device)
        for prediction, target, valid_weight in (
            (
                aoa_clip_prediction.index_select(0, index),
                aoa_clip_target.index_select(0, index),
                unit_weight,
            ),
            (
                j_clip_prediction.index_select(0, index),
                j_clip_target.index_select(0, index),
                j_clip_valid.index_select(0, index).to(unit_weight.dtype),
            ),
        ):
            value = weighted_slope_loss(prediction, target, valid_weight)
            if value is not None:
                clip_losses.append(value)
    zero = output["H"].sum() * 0.0
    patch = torch.stack(patch_losses).mean() if patch_losses else zero
    clip = torch.stack(clip_losses).mean() if clip_losses else zero
    return patch, clip


def loss_for_batch(output: dict, batch: dict, config: dict):
    cfg = config["loss"]
    confidence = batch["confidence"]
    j_weight = confidence * batch["j_valid"]
    aoa_loss = joint.weighted_smooth_l1(
        output["aoa_z"], batch["aoa_target"], confidence
    )
    j_loss = joint.weighted_smooth_l1(
        output["j_z"], batch["j_target"], j_weight
    )
    j_clip_prediction = j_strength_clip_z(output["j_z"], batch)
    j_clip_target = j_strength_clip_z(batch["j_target"], batch)
    j_clip_valid = positive_j_clip_mask(batch)
    if bool(j_clip_valid.any()):
        j_clip_loss = F.smooth_l1_loss(
            j_clip_prediction[j_clip_valid], j_clip_target[j_clip_valid]
        )
    else:
        j_clip_loss = output["H"].sum() * 0.0
    corr_loss = 0.5 * (
        joint.weighted_corr_loss(
            output["aoa_z"], batch["aoa_target"], confidence
        )
        + joint.weighted_corr_loss(
            output["j_z"], batch["j_target"], j_weight
        )
    )
    patch_scale, clip_scale = grouped_scale_loss(output, batch)
    nuisance = output["H"][..., 2:].square().mean()
    total = (
        float(cfg["aoa"]) * aoa_loss
        + float(cfg["j"]) * j_loss
        + float(cfg["j_clip"]) * j_clip_loss
        + float(cfg["correlation"]) * corr_loss
        + float(cfg["patch_scale"]) * patch_scale
        + float(cfg["clip_scale"]) * clip_scale
        + float(cfg["nuisance_compact"]) * nuisance
    )
    return total, {
        "loss": float(total.detach()),
        "loss_aoa": float(aoa_loss.detach()),
        "loss_j": float(j_loss.detach()),
        "loss_j_clip": float(j_clip_loss.detach()),
        "loss_corr": float(corr_loss.detach()),
        "loss_patch_scale": float(patch_scale.detach()),
        "loss_clip_scale": float(clip_scale.detach()),
        "loss_nuisance": float(nuisance.detach()),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    valid = np.isfinite(target) & np.isfinite(prediction)
    target, prediction = target[valid], prediction[valid]
    if len(target) < 3 or np.var(target) < 1e-12:
        return {"n": int(len(target)), "log_r2": float("nan"), "log_corr": float("nan")}
    return {
        "n": int(len(target)),
        "log_r2": float(r2_score(target, prediction)),
        "log_corr": float(np.corrcoef(target, prediction)[0, 1]),
        "log_mae": float(np.mean(np.abs(target - prediction))),
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: dict,
    amp: bool,
    prediction_csv: Path | None = None,
) -> dict:
    model.eval()
    patch_targets = {"aoa": [], "j": []}
    patch_predictions = {"aoa": [], "j": []}
    rows = []
    aoa_mean = float(stats["aoa_log"]["mean"])
    aoa_std = float(stats["aoa_log"]["std"])
    direct_j = "j_proxy" in stats
    j_stats = stats["j_proxy"] if direct_j else stats["j_log_strength"]
    j_mean = float(j_stats["mean"])
    j_std = float(j_stats["std"])
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {loader.dataset.records[0].source}"):
            frames = batch["frames"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                output = model(frames)
            aoa_prediction = output["aoa_z"].detach().cpu().numpy()
            j_prediction = output["j_z"].detach().cpu().numpy()
            aoa_target = batch["aoa_target"].numpy()
            j_target = batch["j_target"].numpy()
            confidence = batch["confidence"].numpy()
            j_valid = batch["j_valid"].numpy()
            aoa_patch_mask = confidence >= 0.20
            j_patch_mask = (confidence >= 0.20) & (j_valid > 0)
            patch_targets["aoa"].append(aoa_target[aoa_patch_mask])
            patch_predictions["aoa"].append(aoa_prediction[aoa_patch_mask])
            patch_targets["j"].append(j_target[j_patch_mask])
            patch_predictions["j"].append(j_prediction[j_patch_mask])
            for index, key in enumerate(batch["key"]):
                aoa_weight = confidence[index]
                j_weight = confidence[index] * j_valid[index]
                aoa_weight_sum = max(float(aoa_weight.sum()), 1e-8)
                j_weight_sum = max(float(j_weight.sum()), 1e-8)
                aoa_target_log = (
                    float((aoa_target[index] * aoa_weight).sum() / aoa_weight_sum)
                    * aoa_std
                    + aoa_mean
                )
                aoa_prediction_log = (
                    float(
                        (aoa_prediction[index] * aoa_weight).sum()
                        / aoa_weight_sum
                    )
                    * aoa_std
                    + aoa_mean
                )
                j_target_patch = j_target[index] * j_std + j_mean
                j_prediction_patch = j_prediction[index] * j_std + j_mean
                if direct_j:
                    j_target_strength = float(
                        (j_target_patch * j_weight).sum() / j_weight_sum
                    )
                    j_prediction_strength = float(
                        (j_prediction_patch * j_weight).sum() / j_weight_sum
                    )
                else:
                    j_target_strength = float(
                        (np.exp(j_target_patch) * j_weight).sum() / j_weight_sum
                    )
                    j_prediction_strength = float(
                        (np.exp(j_prediction_patch) * j_weight).sum()
                        / j_weight_sum
                    )
                rows.append(
                    {
                        "key": key,
                        "scene_group": batch["scene_group"][index],
                        "aoa_target_log_proxy": aoa_target_log,
                        "aoa_prediction_log_proxy": aoa_prediction_log,
                        "j_target_strength_proxy": j_target_strength,
                        "j_prediction_strength_proxy": j_prediction_strength,
                        "j_target_log_strength": (
                            math.log(j_target_strength)
                            if j_target_strength > 0
                            else float("nan")
                        ),
                        "j_prediction_log_strength": (
                            math.log(j_prediction_strength)
                            if j_prediction_strength > 0
                            else float("nan")
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    metrics = {
        "patch": {
            name: regression_metrics(
                np.concatenate(patch_targets[name]),
                np.concatenate(patch_predictions[name]),
            )
            for name in ("aoa", "j")
        },
        "clip": {
            "aoa": regression_metrics(
                frame["aoa_target_log_proxy"].to_numpy(),
                frame["aoa_prediction_log_proxy"].to_numpy(),
            ),
            "j": regression_metrics(
                frame["j_target_log_strength"].to_numpy(),
                frame["j_prediction_log_strength"].to_numpy(),
            ),
        },
    }
    if prediction_csv is not None:
        prediction_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_csv.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(prediction_csv)
    return metrics


def make_val_loaders(
    config: dict,
    stats: dict,
    records: dict,
    split: dict,
    caches: dict,
    scene_map: dict,
    device,
):
    datasets = build_datasets(
        config, stats, records, split, caches, scene_map, "val", False
    )
    maximum = int(config["train"].get("validation_max_videos_per_source", 600))
    loaders = {}
    for source, dataset in datasets.items():
        if len(dataset) > maximum:
            order = sorted(
                range(len(dataset)),
                key=lambda index: joint.stable_fraction(
                    dataset.records[index].key, 99173
                ),
            )[:maximum]
            keys = [dataset.records[index].key for index in order]
            dataset = LongLogJDataset(
                records[source],
                keys,
                caches[source],
                stats[source],
                config,
                scene_map,
                False,
            )
        loaders[source] = DataLoader(
            dataset,
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )
    return loaders, stats


def validation_score(metrics: dict) -> float:
    values = [
        metrics[source]["clip"][target]["log_r2"]
        for source in SOURCES
        for target in ("aoa", "j")
    ]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("-inf")


def train(config: dict, out: Path, max_videos: int | None) -> None:
    started = time.perf_counter()
    print("[train] loading proxy caches", flush=True)
    records, split, caches = load_base_data(config, max_videos)
    print(
        f"[train] proxy caches loaded in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    stats = json.loads((out / "proxy_stats.json").read_text(encoding="utf-8"))
    scene_map = discover_scene_map(Path(config["data"]["grouped_scenes"]))
    print(
        f"[train] grouped-scene map loaded; scenes={len(set(scene_map.values()))}",
        flush=True,
    )
    datasets = build_datasets(
        config, stats, records, split, caches, scene_map, "train", True
    )
    print(
        f"[train] train datasets ready: "
        f"{SOURCES[0]}={len(datasets[SOURCES[0]])}, {RLRAT}={len(datasets[RLRAT])}",
        flush=True,
    )
    batch_sampler = GroupedLongBatchSampler(
        datasets[SOURCES[0]],
        datasets[RLRAT],
        int(config["train"]["batch_size"]),
        int(config["train"]["steps_per_epoch"]),
        int(config["train"]["rlrat_groups_per_batch"]),
        int(config["data"]["split_seed"]),
    )
    combined = torch.utils.data.ConcatDataset([datasets[source] for source in SOURCES])
    device = joint.resolve_device(str(config["train"]["device"]))
    loader = DataLoader(
        combined,
        batch_sampler=batch_sampler,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loaders, _ = make_val_loaders(
        config, stats, records, split, caches, scene_map, device
    )
    print("[train] validation loaders ready; initializing 83 encoder", flush=True)

    model = joint.RealProxyModel(config).to(device)
    train_cfg = config["train"]
    print(
        f"[train] device={device}, cuda_devices={torch.cuda.device_count()}, "
        "initialization=83/83-checkpoint-499.pth, heat_chamber=False",
        flush=True,
    )
    if (
        device.type == "cuda"
        and bool(train_cfg.get("data_parallel", True))
        and torch.cuda.device_count() > 1
    ):
        model = nn.DataParallel(model)
        print(f"[train] DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
    optimizer = torch.optim.AdamW(
        joint.optimizer_groups(model, config),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    warmup = max(1, int(train_cfg["warmup_epochs"]))
    base_lrs = [
        float(train_cfg["learning_rate"]),
        float(train_cfg["encoder_learning_rate"]),
    ]

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < warmup:
            return (epoch_index + 1) / warmup
        progress = (epoch_index - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train_cfg["amp"]) and device.type == "cuda"
    )
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.csv"
    if log_path.exists():
        log_path.unlink()
    best_score = float("-inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        factor = lr_factor(epoch - 1)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * factor
        model.train()
        sums = defaultdict(float)
        steps = 0
        progress = tqdm(loader, desc=f"epoch {epoch:03d}/{epochs}")
        for batch in progress:
            frames = batch["frames"].to(device, non_blocking=True)
            for key in (
                "aoa_target",
                "j_target",
                "confidence",
                "j_valid",
                "j_log_mean",
                "j_log_std",
                "j_clip_log_mean",
                "j_clip_log_std",
                "j_direct",
            ):
                batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(frames)
                loss, parts = loss_for_batch(output, batch, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float(train_cfg["grad_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                sums[key] += value
            steps += 1
            if steps % int(train_cfg["log_every"]) == 0:
                progress.set_postfix(loss=f"{sums['loss'] / steps:.4f}")

        val_metrics = {
            source: evaluate_loader(
                model,
                val_loaders[source],
                device,
                stats[source],
                bool(train_cfg["amp"]),
            )
            for source in SOURCES
        }
        score = validation_score(val_metrics)
        row = {
            "epoch": epoch,
            **{key: value / max(steps, 1) for key, value in sums.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_learning_rate": optimizer.param_groups[1]["lr"],
            "val_mean_clip_log_r2": score,
        }
        for source in SOURCES:
            short = "seq" if source == SOURCES[0] else "rlrat"
            for target in ("aoa", "j"):
                metric = val_metrics[source]["clip"][target]
                row[f"val_{short}_{target}_clip_log_r2"] = metric["log_r2"]
                row[f"val_{short}_{target}_clip_log_corr"] = metric["log_corr"]
        append_csv(log_path, row)
        payload = {
            "model": joint.unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
            "config": config,
            "proxy_stats": stats,
            "val_metrics": val_metrics,
            "initialization": str(
                Path(config["model"]["external_encoder_dir"])
                / "83-checkpoint-499.pth"
            ),
            "heat_chamber_included": False,
            "j_target_mode": str(
                config["proxy"].get("j_target_mode", "log_positive_proxy")
            ),
        }
        torch.save(payload, checkpoints / "last.pt")
        if score > best_score:
            best_score = score
            stale = 0
            torch.save(payload, checkpoints / "best.pt")
        else:
            stale += 1
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale >= int(train_cfg["early_stopping_patience"]):
            print(f"[train] early stopping at epoch {epoch}", flush=True)
            break


def plot_test(frame: pd.DataFrame, source: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    columns = (
        ("aoa_target_log_proxy", "aoa_prediction_log_proxy", "AOA log proxy"),
        ("j_target_log_strength", "j_prediction_log_strength", "log J strength"),
    )
    for axis, (x_column, y_column, label) in zip(axes, columns):
        x = frame[x_column].to_numpy()
        y = frame[y_column].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        score = r2_score(x, y)
        corr = np.corrcoef(x, y)[0, 1]
        axis.scatter(x, y, s=8, alpha=0.25, color="#26738d", edgecolors="none")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([lo, hi], [lo, hi], color="#c4473a", linewidth=1.4)
        axis.set_xlabel(f"Measured {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.set_title(f"log R2={score:.3f}, log r={corr:.3f}")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(source)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"test_scatter_{source}.png", dpi=350)
    plt.close(fig)


def evaluate(
    config: dict,
    out: Path,
    max_videos: int | None,
    checkpoint_arg: str | None,
) -> None:
    datasets, stats = make_datasets(config, out, max_videos, "test", False)
    device = joint.resolve_device(str(config["train"]["device"]))
    checkpoint_path = (
        Path(checkpoint_arg)
        if checkpoint_arg
        else out / "checkpoints" / "best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = joint.RealProxyModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    if (
        device.type == "cuda"
        and bool(config["train"].get("data_parallel", True))
        and torch.cuda.device_count() > 1
    ):
        model = nn.DataParallel(model)
    metrics = {}
    for source, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        prediction_path = out / f"test_predictions_{source}.csv"
        metrics[source] = evaluate_loader(
            model,
            loader,
            device,
            stats[source],
            bool(config["train"]["amp"]),
            prediction_path,
        )
        plot_test(pd.read_csv(prediction_path), source, out / "figures")
        print(f"[test] {source}: {json.dumps(metrics[source])}", flush=True)
    save_json(metrics, out / "test_metrics.json")
    lines = [
        "# Long-Range Log-J Test Results",
        "",
        "- Heat Chamber excluded from training and evaluation.",
        (
            "- J head directly predicts the original J proxy; clip supervision "
            "is applied after positive log transformation."
            if str(config["proxy"].get("j_target_mode")) == "direct_proxy"
            else "- J head directly predicts standardized log positive J strength."
        ),
        f"- Checkpoint: `{checkpoint_path}`",
        "",
        "| Dataset | AOA log R2 | AOA log corr | J log R2 | J log corr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source in SOURCES:
        aoa, j_value = metrics[source]["clip"]["aoa"], metrics[source]["clip"]["j"]
        lines.append(
            f"| {source} | {aoa['log_r2']:.6f} | {aoa['log_corr']:.6f} | "
            f"{j_value['log_r2']:.6f} | {j_value['log_corr']:.6f} |"
        )
    (out / "TEST_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    for argument, key in (
        (args.epochs, "epochs"),
        (args.steps_per_epoch, "steps_per_epoch"),
        (args.batch_size, "batch_size"),
        (args.num_workers, "num_workers"),
    ):
        if argument is not None:
            config["train"][key] = argument


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    seed = int(config["data"]["split_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    out = Path(args.out or config["output"]["root"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(config["output"]["tmp_dir"]).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    save_yaml(config, out / "config_used.yaml")
    if args.max_videos is not None:
        save_json(
            {"max_videos_per_source": args.max_videos},
            out / "smoke_limit.json",
        )
    if args.stage in ("all", "prepare"):
        prepare(config, out, args.max_videos)
    if args.stage in ("all", "train"):
        train(config, out, args.max_videos)
    if args.stage in ("all", "evaluate"):
        evaluate(config, out, args.max_videos, args.checkpoint)


if __name__ == "__main__":
    main()
