from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


KEY_COLUMNS = [
    "sample_id",
    "sample_index",
    "view_id",
    "patch_i",
    "patch_j",
]

PHYSICS_COLUMNS = [
    "J",
    "AOA_var",
    "AOA_proxy",
    "disp_var",
    "Delta_theta",
    "D_aperture",
    "alpha",
    "L",
    "delta",
    "lambda",
]


def load_formula_table(csv_path: str | Path, usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=usecols)
    for col in PHYSICS_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "AOA_proxy" not in df.columns and {"disp_var", "Delta_theta"}.issubset(df.columns):
        df["AOA_proxy"] = df["disp_var"] * (df["Delta_theta"] ** 2)
    if "AOA_proxy_pred" not in df.columns and {"disp_hat", "Delta_theta"}.issubset(df.columns):
        df["AOA_proxy_pred"] = df["disp_hat"] * (df["Delta_theta"] ** 2)
    for col in ["view_id", "patch_i", "patch_j", "sample_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def make_group_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 0,
) -> Dict[str, List[str]]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    groups = df["sample_id"].astype(str).to_numpy()
    indices = np.arange(len(df))

    first = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, tmp_idx = next(first.split(indices, groups=groups))
    tmp_df = df.iloc[tmp_idx].reset_index(drop=True)
    tmp_groups = tmp_df["sample_id"].astype(str).to_numpy()
    tmp_indices = np.arange(len(tmp_df))
    val_fraction = val_ratio / (val_ratio + test_ratio)
    second = GroupShuffleSplit(n_splits=1, train_size=val_fraction, random_state=random_state + 1)
    val_rel, test_rel = next(second.split(tmp_indices, groups=tmp_groups))

    split = {
        "train": sorted(df.iloc[train_idx]["sample_id"].astype(str).unique().tolist()),
        "val": sorted(tmp_df.iloc[val_rel]["sample_id"].astype(str).unique().tolist()),
        "test": sorted(tmp_df.iloc[test_rel]["sample_id"].astype(str).unique().tolist()),
    }
    return split


def save_split(split: Dict[str, List[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in split.items()}
    payload["counts"] = {k: len(v) for k, v in split.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_or_create_split(
    df: pd.DataFrame,
    path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 0,
) -> Dict[str, List[str]]:
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {k: payload[k] for k in ["train", "val", "test"]}
    split = make_group_split(df, train_ratio, val_ratio, test_ratio, random_state)
    save_split(split, path)
    return split


def split_frame(df: pd.DataFrame, split: Dict[str, List[str]], part: str) -> pd.DataFrame:
    wanted = set(split[part])
    return df[df["sample_id"].astype(str).isin(wanted)].copy()


def video_view_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in {"patch_i", "patch_j"}]
    return (
        df[keep]
        .sort_values(["sample_id", "view_id"])
        .drop_duplicates(["sample_id", "view_id"])
        .reset_index(drop=True)
    )


def normalize_sim_path(path_value: str, data_root: str | Path) -> Path:
    raw = str(path_value).replace("\\", "/")
    parts = raw.split("/")
    sim_index = next((i for i, part in enumerate(parts) if part.lower().startswith("sim")), None)
    if sim_index is not None:
        rel = parts[sim_index + 1 :]
        return Path(data_root).joinpath(*rel)
    return Path(raw)


def patch_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["sample_id", "view_id", "patch_i", "patch_j"]).reset_index(drop=True)


def describe_dataset(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "sample_ids": int(df["sample_id"].nunique()),
        "video_views": int(df[["sample_id", "view_id"]].drop_duplicates().shape[0]),
        "patch_rows": int(len(df)),
    }


class FormulaVideoDataset:
    """Torch dataset over video-view samples, returning per-patch supervision."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: Dict[str, List[str]],
        part: str,
        data_root: str | Path,
        frame_count: int = 15,
        image_size: int = 224,
        patch_grid_size: int = 14,
        max_video_views: int | None = None,
    ):
        try:
            import torch
            from torch.utils.data import Dataset
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"PyTorch is required for FormulaVideoDataset: {exc}") from exc
        from .video_io import read_video_frames

        self.torch = torch
        self._dataset_base = Dataset
        self.read_video_frames = read_video_frames
        self.data_root = Path(data_root)
        self.frame_count = frame_count
        self.image_size = image_size
        self.patch_grid_size = int(patch_grid_size)
        part_df = split_frame(patch_order(df), split, part)
        groups = []
        for (sample_id, view_id), group in part_df.groupby(["sample_id", "view_id"], sort=True):
            group = group.sort_values(["patch_i", "patch_j"]).reset_index(drop=True)
            groups.append((sample_id, int(view_id), group))
        if max_video_views is not None:
            groups = groups[: int(max_video_views)]
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        sample_id, view_id, group = self.groups[idx]
        row0 = group.iloc[0]
        video_path = normalize_sim_path(row0["video_path"], self.data_root)
        frames = self.read_video_frames(video_path, self.frame_count, self.image_size)
        item = {
            "frames": self.torch.from_numpy(frames).float(),
            "sample_id": str(sample_id),
            "view_id": int(view_id),
        }
        grid_size = self.patch_grid_size
        patch_count = grid_size * grid_size
        patch_i = group["patch_i"].to_numpy(dtype="int64")
        patch_j = group["patch_j"].to_numpy(dtype="int64")
        # Older tables use 0-based coordinates; the new AOA_10_80_10 table uses 1-based coordinates.
        if patch_i.min() >= 1 and patch_i.max() <= grid_size and patch_j.min() >= 1 and patch_j.max() <= grid_size:
            patch_i = patch_i - 1
            patch_j = patch_j - 1
        flat_index = patch_i * grid_size + patch_j
        valid = (patch_i >= 0) & (patch_i < grid_size) & (patch_j >= 0) & (patch_j < grid_size)
        if "valid_patch" in group.columns:
            valid &= group["valid_patch"].to_numpy(dtype="float32") > 0
        patch_mask = np.zeros(patch_count, dtype=bool)
        patch_mask[flat_index[valid]] = True
        item["patch_mask"] = self.torch.from_numpy(patch_mask)
        for col in PHYSICS_COLUMNS:
            if col in group.columns:
                values = np.zeros(patch_count, dtype="float32")
                values[flat_index[valid]] = group.loc[valid, col].to_numpy(dtype="float32")
                item[col] = self.torch.from_numpy(values)
        for col in KEY_COLUMNS:
            if col in group.columns and col not in {"sample_id", "view_id"}:
                values = np.full(patch_count, -1, dtype="int64")
                values[flat_index[valid]] = group.loc[valid, col].astype("int64").to_numpy()
                item[col] = self.torch.from_numpy(values)
        item["video_path"] = str(video_path)
        return item
