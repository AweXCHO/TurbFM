from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


LONG_SOURCES = ("turbulence_sequences", "turbulence_sequences_RLRAT")
HEAT_SOURCE = "heat_chamber"
ALL_SOURCES = (*LONG_SOURCES, HEAT_SOURCE)


@dataclass(frozen=True)
class Record:
    key: str
    source: str
    path: str
    scene_id: str = ""
    clip_id: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a real-only Swin-MAE + Deflex AOA/J proxy model."
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config_real_joint.yaml"))
    )
    parser.add_argument(
        "--stage", choices=("all", "prepare", "train", "evaluate"), default="all"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit each long-range source for smoke testing.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--overwrite-proxies", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def save_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    tmp.replace(path)


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_name(value: float, train_ratio: float, val_ratio: float) -> str:
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def discover_long_records(config: dict, max_videos: int | None) -> dict[str, list[Record]]:
    result: dict[str, list[Record]] = {}
    for source in LONG_SOURCES:
        root = Path(config["data"][source])
        paths = sorted(root.glob("*.avi"))
        if max_videos is not None:
            paths = paths[:max_videos]
        if not paths:
            raise FileNotFoundError(f"No AVI files found under {root}")
        result[source] = [
            Record(
                key=f"{source}/{path.stem}",
                source=source,
                path=str(path),
            )
            for path in paths
        ]
    return result


def discover_heat_records(config: dict) -> list[Record]:
    root = Path(config["data"]["heat_chamber"])
    proxy = pd.read_csv(config["data"]["heat_proxy_csv"], usecols=["scene_id", "clip_id"])
    pairs = proxy.drop_duplicates().sort_values(["scene_id", "clip_id"])
    records = []
    for row in pairs.itertuples(index=False):
        scene_id = str(row.scene_id)
        clip_id = int(row.clip_id)
        scene = root / scene_id
        if not (scene / "gt.png").exists():
            raise FileNotFoundError(scene / "gt.png")
        records.append(
            Record(
                key=f"{HEAT_SOURCE}/{scene_id}/clip_{clip_id:02d}",
                source=HEAT_SOURCE,
                path=str(scene),
                scene_id=scene_id,
                clip_id=clip_id,
            )
        )
    if not records:
        raise RuntimeError("No Heat Chamber records were discovered")
    return records


def make_splits(
    config: dict,
    records_by_source: dict[str, list[Record]],
    heat_records: list[Record],
) -> dict:
    data = config["data"]
    seed = int(data["split_seed"])
    train_ratio = float(data["train_ratio"])
    val_ratio = float(data["val_ratio"])
    split: dict[str, dict[str, list[str]]] = {}
    for source, records in records_by_source.items():
        split[source] = {"train": [], "val": [], "test": []}
        for record in records:
            name = split_name(
                stable_fraction(record.key, seed), train_ratio, val_ratio
            )
            split[source][name].append(record.key)

    # Heat Chamber clips from the same scene must stay in one split.
    split[HEAT_SOURCE] = {"train": [], "val": [], "test": []}
    scene_split = {}
    for scene_id in sorted({record.scene_id for record in heat_records}):
        value = stable_fraction(f"{HEAT_SOURCE}/{scene_id}", seed)
        scene_split[scene_id] = split_name(value, train_ratio, val_ratio)
    for record in heat_records:
        split[HEAT_SOURCE][scene_split[record.scene_id]].append(record.key)
    split["_heat_scene_split"] = scene_split
    return split


def sample_video_frames(path: str, count: int, size: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count: {path}")
    wanted = set(np.linspace(0, frame_total - 1, count).round().astype(int).tolist())
    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        index += 1
    cap.release()
    if len(frames) != len(wanted):
        raise RuntimeError(
            f"Expected {len(wanted)} sampled frames, decoded {len(frames)}: {path}"
        )
    return np.stack(frames)


def patch_mean(array: np.ndarray, grid: int) -> np.ndarray:
    h, w = array.shape
    if h % grid or w % grid:
        raise ValueError(f"Shape {(h, w)} is not divisible by grid={grid}")
    ph, pw = h // grid, w // grid
    return array.reshape(grid, ph, grid, pw).mean(axis=(1, 3))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def long_video_proxy(task: tuple) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, str]:
    key, path, frame_count, size, grid, eps, flow_method = task
    try:
        frames_u8 = sample_video_frames(path, frame_count, size)
        reference = np.median(frames_u8, axis=0).astype(np.uint8)
        if flow_method != "dis_fast":
            raise ValueError(f"Unsupported flow method: {flow_method}")
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        local_flows = []
        aligned_frames = []
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        for frame in frames_u8:
            flow = estimator.calc(reference, frame, None).astype(np.float32)
            translation = np.median(flow.reshape(-1, 2), axis=0)
            local_flows.append(flow - translation[None, None, :])
            # DIS returns reference-to-frame coordinates. Full local-flow
            # registration removes geometric AOA motion before measuring the
            # residual high-frequency loss used as the J proxy.
            map_x = xx + flow[..., 0]
            map_y = yy + flow[..., 1]
            aligned = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            aligned_frames.append(aligned.astype(np.float32) / 255.0)

        local = np.stack(local_flows)
        center = np.median(local, axis=0, keepdims=True)
        robust_variance = (
            1.4826**2 * np.median((local - center) ** 2, axis=0).sum(axis=-1)
        )
        aoa_log = np.log(patch_mean(robust_variance, grid) + eps)

        aligned = np.stack(aligned_frames)
        sharpness = np.asarray(
            [gradient_magnitude(frame).mean() for frame in aligned]
        )
        lucky = aligned[int(np.argmax(sharpness))]
        temporal_mean = aligned.mean(axis=0)
        lucky_gradient = gradient_magnitude(lucky)
        mean_gradient = gradient_magnitude(temporal_mean)
        j_attenuation = np.log(
            (patch_mean(lucky_gradient, grid) + eps)
            / (patch_mean(mean_gradient, grid) + eps)
        )
        j_attenuation = np.clip(j_attenuation, -2.0, 5.0)

        texture = patch_mean(gradient_magnitude(reference / 255.0), grid)
        lo, hi = np.quantile(texture, [0.20, 0.80])
        confidence = np.clip((texture - lo) / (hi - lo + eps), 0.0, 1.0)
        confidence = 0.10 + 0.90 * confidence
        return (
            key,
            aoa_log.astype(np.float32),
            j_attenuation.astype(np.float32),
            confidence.astype(np.float32),
            "",
        )
    except Exception as exc:
        shape = (grid, grid)
        empty = np.full(shape, np.nan, dtype=np.float32)
        return key, empty, empty.copy(), empty.copy(), f"{type(exc).__name__}: {exc}"


def prepare_long_proxy_cache(
    config: dict,
    records: list[Record],
    cache_path: Path,
    overwrite: bool,
) -> None:
    if cache_path.exists() and not overwrite:
        print(f"[prepare] reuse {cache_path}", flush=True)
        return
    proxy_cfg = config["proxy"]
    data_cfg = config["data"]
    tasks = [
        (
            record.key,
            record.path,
            int(data_cfg["proxy_frames"]),
            int(data_cfg["proxy_size"]),
            int(data_cfg["patch_grid"]),
            float(proxy_cfg["epsilon"]),
            str(proxy_cfg["flow_method"]),
        )
        for record in records
    ]
    keys, aoa, j_value, confidence, errors = [], [], [], [], {}
    workers = int(proxy_cfg.get("workers", 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(long_video_proxy, tasks, chunksize=32)
        for key, aoa_map, j_map, conf_map, error in tqdm(
            iterator, total=len(tasks), desc=f"proxy {records[0].source}"
        ):
            keys.append(key)
            aoa.append(aoa_map)
            j_value.append(j_map)
            confidence.append(conf_map)
            if error:
                errors[key] = error
    atomic_npz(
        cache_path,
        keys=np.asarray(keys),
        aoa_log=np.stack(aoa),
        j_log=np.stack(j_value),
        confidence=np.stack(confidence),
    )
    if errors:
        atomic_json(errors, cache_path.with_suffix(".errors.json"))
    print(
        f"[prepare] wrote {cache_path}; valid={len(keys)-len(errors)}, errors={len(errors)}",
        flush=True,
    )


def heat_clip_proxy(task: tuple) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, str]:
    key, scene_path, clip_id, frame_count, size, grid, eps = task
    try:
        scene = Path(scene_path)
        gt = cv2.imread(str(scene / "gt.png"), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(scene / "gt.png")
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_AREA)
        paths = sorted((scene / "turb").glob("*.png"))
        start = clip_id * frame_count
        paths = paths[start : start + frame_count]
        if len(paths) != frame_count:
            raise RuntimeError(f"{key}: incomplete Heat Chamber clip")

        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        local_flows = []
        aligned_frames = []
        for path in paths:
            frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise FileNotFoundError(path)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            flow = estimator.calc(gt, frame, None).astype(np.float32)
            translation = np.median(flow.reshape(-1, 2), axis=0)
            local_flows.append(flow - translation[None, None, :])
            aligned = cv2.remap(
                frame,
                xx + flow[..., 0],
                yy + flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            aligned_frames.append(aligned.astype(np.float32) / 255.0)

        local = np.stack(local_flows)
        center = np.median(local, axis=0, keepdims=True)
        robust_variance = (
            1.4826**2 * np.median((local - center) ** 2, axis=0).sum(axis=-1)
        )
        aoa_log = np.log(patch_mean(robust_variance, grid) + eps)

        gt_float = gt.astype(np.float32) / 255.0
        temporal_mean = np.stack(aligned_frames).mean(axis=0)
        gt_gradient = patch_mean(gradient_magnitude(gt_float), grid)
        degraded_gradient = patch_mean(gradient_magnitude(temporal_mean), grid)
        j_attenuation = np.log(
            (gt_gradient + eps) / (degraded_gradient + eps)
        )
        j_attenuation = np.clip(j_attenuation, -2.0, 5.0)

        texture = gt_gradient
        lo, hi = np.quantile(texture, [0.20, 0.80])
        confidence = np.clip((texture - lo) / (hi - lo + eps), 0.0, 1.0)
        confidence = 0.10 + 0.90 * confidence
        return (
            key,
            aoa_log.astype(np.float32),
            j_attenuation.astype(np.float32),
            confidence.astype(np.float32),
            "",
        )
    except Exception as exc:
        shape = (grid, grid)
        empty = np.full(shape, np.nan, dtype=np.float32)
        return key, empty, empty.copy(), empty.copy(), f"{type(exc).__name__}: {exc}"


def prepare_heat_proxy_cache(
    config: dict,
    records: list[Record],
    cache_path: Path,
    overwrite: bool,
) -> None:
    if cache_path.exists() and not overwrite:
        print(f"[prepare] reuse {cache_path}", flush=True)
        return
    data_cfg = config["data"]
    proxy_cfg = config["proxy"]
    tasks = [
        (
            record.key,
            record.path,
            record.clip_id,
            int(data_cfg["model_frames"]),
            int(data_cfg["proxy_size"]),
            int(data_cfg["patch_grid"]),
            float(proxy_cfg["epsilon"]),
        )
        for record in records
    ]
    keys, aoa, j_value, confidence, errors = [], [], [], [], {}
    with ProcessPoolExecutor(max_workers=int(proxy_cfg.get("workers", 1))) as pool:
        iterator = pool.map(heat_clip_proxy, tasks, chunksize=12)
        for key, aoa_map, j_map, conf_map, error in tqdm(
            iterator, total=len(tasks), desc="proxy heat_chamber"
        ):
            keys.append(key)
            aoa.append(aoa_map)
            j_value.append(j_map)
            confidence.append(conf_map)
            if error:
                errors[key] = error
    atomic_npz(
        cache_path,
        keys=np.asarray(keys),
        aoa_log=np.stack(aoa),
        j_log=np.stack(j_value),
        confidence=np.stack(confidence),
    )
    if errors:
        atomic_json(errors, cache_path.with_suffix(".errors.json"))
    print(f"[prepare] wrote {cache_path}", flush=True)


def cache_to_dict(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path) as data:
        keys = data["keys"].astype(str)
        aoa = data["aoa_log"].astype(np.float32)
        j_value = data["j_log"].astype(np.float32)
        confidence = data["confidence"].astype(np.float32)
    result = {}
    for index, key in enumerate(keys):
        if not (
            np.isfinite(aoa[index]).all()
            and np.isfinite(j_value[index]).all()
            and np.isfinite(confidence[index]).all()
        ):
            continue
        result[key] = {
            "aoa_log": aoa[index],
            "j_log": j_value[index],
            "confidence": confidence[index],
        }
    return result


def weighted_stats(values: Iterable[np.ndarray], weights: Iterable[np.ndarray]) -> dict:
    total_w = 0.0
    total_x = 0.0
    total_x2 = 0.0
    for value, weight in zip(values, weights):
        mask = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
        x = value[mask].astype(np.float64)
        w = weight[mask].astype(np.float64)
        total_w += float(w.sum())
        total_x += float(np.dot(x, w))
        total_x2 += float(np.dot(x * x, w))
    mean = total_x / max(total_w, 1e-12)
    variance = max(total_x2 / max(total_w, 1e-12) - mean * mean, 1e-8)
    return {"mean": mean, "std": math.sqrt(variance), "weight_sum": total_w}


def compute_proxy_stats(
    split: dict,
    caches: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict:
    result = {}
    for source in ALL_SOURCES:
        keys = [key for key in split[source]["train"] if key in caches[source]]
        values = [caches[source][key] for key in keys]
        result[source] = {
            "aoa_log": weighted_stats(
                (item["aoa_log"] for item in values),
                (item["confidence"] for item in values),
            ),
            "j_log": weighted_stats(
                (item["j_log"] for item in values),
                (item["confidence"] for item in values),
            ),
            "train_records": len(keys),
        }
    return result


def prepare(config: dict, out: Path, max_videos: int | None, overwrite: bool) -> None:
    records_by_source = discover_long_records(config, max_videos)
    heat_records = discover_heat_records(config)
    split = make_splits(config, records_by_source, heat_records)
    cache_dir = out / "proxy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in LONG_SOURCES:
        prepare_long_proxy_cache(
            config,
            records_by_source[source],
            cache_dir / f"{source}.npz",
            overwrite,
        )
    prepare_heat_proxy_cache(
        config, heat_records, cache_dir / f"{HEAT_SOURCE}.npz", overwrite
    )
    atomic_json(split, out / "splits.json")
    caches = {
        source: cache_to_dict(cache_dir / f"{source}.npz")
        for source in ALL_SOURCES
    }
    stats = compute_proxy_stats(split, caches)
    atomic_json(stats, out / "proxy_stats.json")
    counts = {
        source: {
            part: sum(key in caches[source] for key in split[source][part])
            for part in ("train", "val", "test")
        }
        for source in ALL_SOURCES
    }
    atomic_json(counts, out / "dataset_counts.json")
    print("[prepare] dataset counts:", json.dumps(counts, indent=2), flush=True)


def read_avi_rgb(path: str, frame_count: int, training: bool) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < frame_count:
        raise RuntimeError(f"{path}: decoded only {len(frames)} frames")
    if training:
        indices = sorted(random.sample(range(len(frames)), frame_count))
    else:
        indices = np.linspace(0, len(frames) - 1, frame_count).round().astype(int)
    return np.stack([frames[int(index)] for index in indices])


def read_heat_rgb(record: Record, frame_count: int, training: bool) -> np.ndarray:
    paths = sorted((Path(record.path) / "turb").glob("*.png"))
    start = record.clip_id * frame_count
    clip_paths = paths[start : start + frame_count]
    if len(clip_paths) != frame_count:
        raise RuntimeError(f"{record.key}: incomplete Heat Chamber clip")
    if training:
        order = list(range(frame_count))
        random.shuffle(order)
        clip_paths = [clip_paths[index] for index in order]
    frames = []
    for path in clip_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


class ProxyDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        keys: list[str],
        cache: dict[str, dict[str, np.ndarray]],
        stats: dict,
        config: dict,
        training: bool,
    ):
        record_map = {record.key: record for record in records}
        self.records = [record_map[key] for key in keys if key in record_map and key in cache]
        self.cache = cache
        self.stats = stats
        self.image_size = int(config["data"]["image_size"])
        self.frame_count = int(config["data"]["model_frames"])
        self.training = training
        if not self.records:
            raise RuntimeError("ProxyDataset received no valid records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        if record.source == HEAT_SOURCE:
            frames = read_heat_rgb(record, self.frame_count, self.training)
        else:
            frames = read_avi_rgb(record.path, self.frame_count, self.training)
        resized = np.stack(
            [
                cv2.resize(
                    frame,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in frames
            ]
        ).astype(np.float32)
        target = self.cache[record.key]
        aoa = target["aoa_log"].copy()
        j_value = target["j_log"].copy()
        confidence = target["confidence"].copy()

        if self.training and random.random() < 0.5:
            resized = resized[:, :, ::-1, :].copy()
            aoa = aoa[:, ::-1].copy()
            j_value = j_value[:, ::-1].copy()
            confidence = confidence[:, ::-1].copy()
        if self.training:
            gain = random.uniform(0.90, 1.10)
            bias = random.uniform(-5.0, 5.0)
            resized = np.clip(resized * gain + bias, 0.0, 255.0)

        aoa_stats = self.stats["aoa_log"]
        j_stats = self.stats["j_log"]
        aoa_z = (aoa - float(aoa_stats["mean"])) / float(aoa_stats["std"])
        j_z = (j_value - float(j_stats["mean"])) / float(j_stats["std"])
        frames_chw = resized.transpose(0, 3, 1, 2) / 255.0
        return {
            "frames": torch.from_numpy(frames_chw.astype(np.float32)),
            "aoa_target": torch.from_numpy(aoa_z.astype(np.float32)).reshape(-1),
            "j_target": torch.from_numpy(j_z.astype(np.float32)).reshape(-1),
            "confidence": torch.from_numpy(confidence.astype(np.float32)).reshape(-1),
            "key": record.key,
            "source": record.source,
        }


class RealProxyModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        project_root = Path(config["model"]["project_root"])
        sys.path.insert(0, str(project_root))
        try:
            from src.deflex_blocks import TurbDeflexBlock
            from src.models import Bottleneck, build_encoder
        finally:
            sys.path.pop(0)
        self.encoder = build_encoder(config)
        model_cfg = config["model"]
        embedding_dim = int(model_cfg["embedding_dim"])
        self.blocks = nn.ModuleList(
            [
                TurbDeflexBlock(
                    embedding_dim,
                    n_heads=int(model_cfg["attention_heads"]),
                    dropout=float(model_cfg["dropout"]),
                )
                for _ in range(int(model_cfg["deflex_blocks"]))
            ]
        )
        self.bottleneck = Bottleneck(
            embedding_dim=embedding_dim,
            latent_dim=int(model_cfg["latent_dim"]),
        )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.encoder(frames)
        for block in self.blocks:
            representation = block(representation)
        patch_repr = representation.mean(dim=1)
        latent = self.bottleneck(patch_repr)
        return {
            "H": latent,
            "aoa_z": latent[..., 0],
            "j_z": latent[..., 1],
        }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_records_and_caches(config: dict, out: Path, max_videos: int | None):
    records_by_source = discover_long_records(config, max_videos)
    records_by_source[HEAT_SOURCE] = discover_heat_records(config)
    split = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "proxy_stats.json").read_text(encoding="utf-8"))
    caches = {
        source: cache_to_dict(out / "proxy_cache" / f"{source}.npz")
        for source in ALL_SOURCES
    }
    return records_by_source, split, stats, caches


def make_dataset(
    source: str,
    part: str,
    records_by_source: dict,
    split: dict,
    stats: dict,
    caches: dict,
    config: dict,
    training: bool,
    max_records: int | None = None,
) -> ProxyDataset:
    keys = [key for key in split[source][part] if key in caches[source]]
    if max_records is not None and len(keys) > max_records:
        keys = sorted(keys, key=lambda key: stable_fraction(key, 99173))[:max_records]
    return ProxyDataset(
        records_by_source[source],
        keys,
        caches[source],
        stats[source],
        config,
        training,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def optimizer_groups(model: nn.Module, config: dict) -> list[dict]:
    train_cfg = config["train"]
    encoder_params, other_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder.") or name.startswith("module.encoder."):
            encoder_params.append(parameter)
        else:
            other_params.append(parameter)
    return [
        {"params": other_params, "lr": float(train_cfg["learning_rate"])},
        {
            "params": encoder_params,
            "lr": float(train_cfg["encoder_learning_rate"]),
        },
    ]


def weighted_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def weighted_corr_loss(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    x = prediction.reshape(-1)
    y = target.reshape(-1)
    w = weight.reshape(-1)
    w_sum = w.sum().clamp_min(1e-6)
    x_mean = (x * w).sum() / w_sum
    y_mean = (y * w).sum() / w_sum
    x_center = x - x_mean
    y_center = y - y_mean
    covariance = (w * x_center * y_center).sum() / w_sum
    x_var = (w * x_center.square()).sum() / w_sum
    y_var = (w * y_center.square()).sum() / w_sum
    corr = covariance / torch.sqrt(x_var * y_var + 1e-8)
    return 1.0 - corr


def loss_for_batch(output: dict, batch: dict, config: dict) -> tuple[torch.Tensor, dict]:
    loss_cfg = config["loss"]
    confidence = batch["confidence"]
    aoa_source_weight = torch.ones(
        confidence.shape[0], 1, dtype=confidence.dtype, device=confidence.device
    )
    j_source_weight = torch.ones_like(aoa_source_weight)
    for index, source in enumerate(batch["source"]):
        if source == HEAT_SOURCE:
            aoa_source_weight[index] = float(loss_cfg["heat_gt_label_weight"])
            j_source_weight[index] = float(loss_cfg["heat_gt_label_weight"])
        else:
            aoa_source_weight[index] = float(
                loss_cfg["long_aoa_pseudo_label_weight"]
            )
            j_source_weight[index] = float(loss_cfg["long_j_pseudo_label_weight"])
    aoa_weight = confidence * aoa_source_weight
    j_weight = confidence * j_source_weight
    aoa_loss = weighted_smooth_l1(
        output["aoa_z"], batch["aoa_target"], aoa_weight
    )
    j_loss = weighted_smooth_l1(output["j_z"], batch["j_target"], j_weight)
    corr_loss = 0.5 * (
        weighted_corr_loss(output["aoa_z"], batch["aoa_target"], aoa_weight)
        + weighted_corr_loss(output["j_z"], batch["j_target"], j_weight)
    )
    nuisance = output["H"][..., 2:].square().mean()
    total = (
        float(loss_cfg["aoa"]) * aoa_loss
        + float(loss_cfg["j"]) * j_loss
        + float(loss_cfg["correlation"]) * corr_loss
        + float(loss_cfg["nuisance_compact"]) * nuisance
    )
    return total, {
        "loss": float(total.detach()),
        "loss_aoa": float(aoa_loss.detach()),
        "loss_j": float(j_loss.detach()),
        "loss_corr": float(corr_loss.detach()),
        "loss_nuisance": float(nuisance.detach()),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    mask = np.isfinite(target) & np.isfinite(prediction)
    target = target[mask]
    prediction = prediction[mask]
    if len(target) < 3 or np.var(target) < 1e-12:
        return {"n": int(len(target)), "log_r2": float("nan"), "log_corr": float("nan")}
    return {
        "n": int(len(target)),
        "log_r2": float(r2_score(target, prediction)),
        "log_corr": safe_corr(target, prediction),
        "mae_z": float(np.mean(np.abs(target - prediction))),
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
    patch_target = {"aoa": [], "j": []}
    patch_prediction = {"aoa": [], "j": []}
    clip_rows = []
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
            for name, target, prediction in (
                ("aoa", aoa_target, aoa_prediction),
                ("j", j_target, j_prediction),
            ):
                mask = confidence >= 0.20
                patch_target[name].append(target[mask])
                patch_prediction[name].append(prediction[mask])
            for index, key in enumerate(batch["key"]):
                weight = confidence[index]
                weight_sum = max(float(weight.sum()), 1e-8)
                aoa_t = float((aoa_target[index] * weight).sum() / weight_sum)
                aoa_p = float((aoa_prediction[index] * weight).sum() / weight_sum)
                j_t = float((j_target[index] * weight).sum() / weight_sum)
                j_p = float((j_prediction[index] * weight).sum() / weight_sum)
                clip_rows.append(
                    {
                        "key": key,
                        "aoa_target_z": aoa_t,
                        "aoa_prediction_z": aoa_p,
                        "j_target_z": j_t,
                        "j_prediction_z": j_p,
                        "aoa_target_log_proxy": aoa_t * stats["aoa_log"]["std"]
                        + stats["aoa_log"]["mean"],
                        "aoa_prediction_log_proxy": aoa_p * stats["aoa_log"]["std"]
                        + stats["aoa_log"]["mean"],
                        "j_target_log_proxy": j_t * stats["j_log"]["std"]
                        + stats["j_log"]["mean"],
                        "j_prediction_log_proxy": j_p * stats["j_log"]["std"]
                        + stats["j_log"]["mean"],
                    }
                )
    clip = pd.DataFrame(clip_rows)
    metrics = {
        "patch": {
            name: regression_metrics(
                np.concatenate(patch_target[name]),
                np.concatenate(patch_prediction[name]),
            )
            for name in ("aoa", "j")
        },
        "clip": {
            "aoa": regression_metrics(
                clip["aoa_target_z"].to_numpy(),
                clip["aoa_prediction_z"].to_numpy(),
            ),
            "j": regression_metrics(
                clip["j_target_z"].to_numpy(),
                clip["j_prediction_z"].to_numpy(),
            ),
        },
    }
    if prediction_csv is not None:
        prediction_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = prediction_csv.with_suffix(".csv.tmp")
        clip.to_csv(tmp, index=False)
        tmp.replace(prediction_csv)
    return metrics


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validation_score(metrics: dict) -> float:
    values = []
    weights = []
    for source in ALL_SOURCES:
        for target in ("aoa", "j"):
            value = float(metrics[source]["clip"][target]["log_r2"])
            if math.isfinite(value):
                values.append(value)
                # Long-range J is a weak no-GT proxy and must not dominate
                # checkpoint selection. Heat J and all AOA targets are primary.
                weights.append(
                    0.25 if source in LONG_SOURCES and target == "j" else 1.0
                )
    return (
        float(np.average(values, weights=weights))
        if values
        else float("-inf")
    )


def train(config: dict, out: Path, max_videos: int | None) -> None:
    records, split, stats, caches = load_records_and_caches(config, out, max_videos)
    train_datasets = {
        source: make_dataset(
            source, "train", records, split, stats, caches, config, training=True
        )
        for source in ALL_SOURCES
    }
    combined = ConcatDataset([train_datasets[source] for source in ALL_SOURCES])
    probabilities = config["train"]["source_probabilities"]
    weights = []
    for source in ALL_SOURCES:
        dataset = train_datasets[source]
        sample_weight = float(probabilities[source]) / len(dataset)
        weights.extend([sample_weight] * len(dataset))
    train_cfg = config["train"]
    sample_count = int(train_cfg["steps_per_epoch"]) * int(train_cfg["batch_size"])
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=sample_count,
        replacement=True,
    )
    device = resolve_device(str(train_cfg["device"]))
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        combined,
        batch_size=int(train_cfg["batch_size"]),
        sampler=sampler,
        num_workers=int(train_cfg["num_workers"]),
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    max_val = int(train_cfg.get("validation_max_videos_per_source", 300))
    val_loaders = {}
    for source in ALL_SOURCES:
        dataset = make_dataset(
            source,
            "val",
            records,
            split,
            stats,
            caches,
            config,
            training=False,
            max_records=max_val,
        )
        val_loaders[source] = DataLoader(
            dataset,
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=pin_memory,
        )

    model = RealProxyModel(config).to(device)
    print(
        f"[train] device={device}, cuda_devices={torch.cuda.device_count()}, "
        f"initialization=83/83-checkpoint-499.pth",
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
        optimizer_groups(model, config),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    total_epochs = int(train_cfg["epochs"])
    warmup_epochs = max(1, int(train_cfg["warmup_epochs"]))

    def lr_factor(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / warmup_epochs
        progress = (epoch_index - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    base_lrs = [
        float(train_cfg["learning_rate"]),
        float(train_cfg["encoder_learning_rate"]),
    ]
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg["amp"]) and pin_memory)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    stale_epochs = 0
    train_log = out / "train_log.csv"
    if train_log.exists():
        train_log.unlink()

    for epoch in range(1, total_epochs + 1):
        factor = lr_factor(epoch - 1)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * factor
        model.train()
        sums: dict[str, float] = {}
        step_count = 0
        progress = tqdm(loader, desc=f"epoch {epoch:03d}/{total_epochs}")
        for batch in progress:
            frames = batch["frames"].to(device, non_blocking=True)
            for key in ("aoa_target", "j_target", "confidence"):
                batch[key] = batch[key].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                output = model(frames)
                loss, parts = loss_for_batch(output, batch, config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                float(train_cfg["grad_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            step_count += 1
            if step_count % int(train_cfg["log_every"]) == 0:
                progress.set_postfix(loss=f"{sums['loss']/step_count:.4f}")
        val_metrics = {
            source: evaluate_loader(
                model,
                val_loaders[source],
                device,
                stats[source],
                bool(train_cfg["amp"]),
            )
            for source in ALL_SOURCES
        }
        score = validation_score(val_metrics)
        row = {
            "epoch": epoch,
            **{key: value / max(step_count, 1) for key, value in sums.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_learning_rate": optimizer.param_groups[1]["lr"],
            "val_mean_clip_log_r2": score,
        }
        for source in ALL_SOURCES:
            short = "seq" if source == LONG_SOURCES[0] else (
                "rlrat" if source == LONG_SOURCES[1] else "heat"
            )
            for target in ("aoa", "j"):
                row[f"val_{short}_{target}_clip_log_r2"] = val_metrics[source]["clip"][
                    target
                ]["log_r2"]
                row[f"val_{short}_{target}_clip_log_corr"] = val_metrics[source][
                    "clip"
                ][target]["log_corr"]
        append_csv(train_log, row)
        payload = {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
            "config": config,
            "proxy_stats": stats,
            "val_metrics": val_metrics,
            "initialization": str(
                Path(config["model"]["external_encoder_dir"]) / "83-checkpoint-499.pth"
            ),
        }
        torch.save(payload, checkpoint_dir / "last.pt")
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(payload, checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stale_epochs >= int(train_cfg["early_stopping_patience"]):
            print(f"[train] early stopping at epoch {epoch}", flush=True)
            break


def plot_test_scatter(
    prediction_csv: Path, source: str, figures: Path, max_points: int
) -> None:
    frame = pd.read_csv(prediction_csv)
    rng = np.random.default_rng(20260724)
    if len(frame) > max_points:
        frame = frame.iloc[rng.choice(len(frame), max_points, replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    for axis, name, label in (
        (axes[0], "aoa", "AOA proxy"),
        (axes[1], "j", "J proxy"),
    ):
        x = frame[f"{name}_target_log_proxy"].to_numpy()
        y = frame[f"{name}_prediction_log_proxy"].to_numpy()
        score = r2_score(x, y)
        corr = safe_corr(x, y)
        axis.scatter(x, y, s=8, alpha=0.25, color="#26738d", edgecolors="none")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        axis.plot([lo, hi], [lo, hi], color="#c4473a", linewidth=1.4)
        axis.set_xlabel(f"Measured {label} (log scale)")
        axis.set_ylabel(f"Predicted {label} (log scale)")
        axis.set_title(f"{label}: log R2={score:.3f}, log r={corr:.3f}")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(source)
    fig.tight_layout()
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"test_scatter_{source}.png", dpi=300)
    plt.close(fig)


def write_test_report(out: Path, metrics: dict, checkpoint: Path) -> None:
    lines = [
        "# Joint Real-Data AOA/J Proxy Test Results",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Initialization: `83/83-checkpoint-499.pth`",
        "- Training input: both long-range datasets plus paired Heat Chamber supervision",
        "- Evaluation: each dataset uses its own held-out test split",
        "",
        "| Dataset | AOA clip log R2 | AOA log corr | J clip log R2 | J log corr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source in ALL_SOURCES:
        aoa = metrics[source]["clip"]["aoa"]
        j_value = metrics[source]["clip"]["j"]
        lines.append(
            f"| {source} | {aoa['log_r2']:.6f} | {aoa['log_corr']:.6f} | "
            f"{j_value['log_r2']:.6f} | {j_value['log_corr']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The two long-range test sets have no physical AOA or J ground truth. Their",
            "scores measure agreement with independently precomputed temporal-flow and",
            "high-frequency attenuation proxies. Heat Chamber uses paired clear images",
            "and is therefore the stronger real-data check.",
            "",
            "Because aperture, propagation distance, and angular pixel scale are absent,",
            "these results do not validate an absolute physical J value or a D exponent.",
        ]
    )
    (out / "TEST_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    config: dict,
    out: Path,
    max_videos: int | None,
    checkpoint_arg: str | None,
) -> None:
    records, split, stats, caches = load_records_and_caches(config, out, max_videos)
    device = resolve_device(str(config["train"]["device"]))
    checkpoint_path = (
        Path(checkpoint_arg)
        if checkpoint_arg
        else out / "checkpoints" / "best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RealProxyModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    if (
        device.type == "cuda"
        and bool(config["train"].get("data_parallel", True))
        and torch.cuda.device_count() > 1
    ):
        model = nn.DataParallel(model)
    metrics = {}
    for source in ALL_SOURCES:
        dataset = make_dataset(
            source,
            "test",
            records,
            split,
            stats,
            caches,
            config,
            training=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["evaluation"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["evaluation"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        prediction_csv = out / f"test_predictions_{source}.csv"
        metrics[source] = evaluate_loader(
            model,
            loader,
            device,
            stats[source],
            bool(config["train"]["amp"]),
            prediction_csv,
        )
        plot_test_scatter(
            prediction_csv,
            source,
            out / "figures",
            int(config["evaluation"]["max_scatter_points"]),
        )
        print(f"[test] {source}: {json.dumps(metrics[source])}", flush=True)
    atomic_json(metrics, out / "test_metrics.json")
    write_test_report(out, metrics, checkpoint_path)


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    train_cfg = config["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        train_cfg["steps_per_epoch"] = args.steps_per_epoch
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers


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
    out = Path(args.out or config["output"]["root"])
    out.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(config["output"]["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    save_yaml(config, out / "config_used.yaml")
    if args.max_videos is not None:
        atomic_json({"max_videos_per_long_source": args.max_videos}, out / "smoke_limit.json")

    if args.stage in ("all", "prepare"):
        prepare(config, out, args.max_videos, args.overwrite_proxies)
    if args.stage in ("all", "train"):
        train(config, out, args.max_videos)
    if args.stage in ("all", "evaluate"):
        evaluate(config, out, args.max_videos, args.checkpoint)


if __name__ == "__main__":
    main()
