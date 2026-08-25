from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import discover_grouped_scene_formula as grouped
import train_real_joint as joint
import train_real_long_logj as training


SOURCES = training.SOURCES
RLRAT = training.RLRAT
RELATIONS = {
    "measured_proxy": {
        "x": "j_target_strength_proxy",
        "y": "aoa_target_log_proxy",
        "label": "Measured proxies",
    },
    "model_output": {
        "x": "j_prediction_strength_proxy",
        "y": "aoa_prediction_log_proxy",
        "label": "Model outputs",
    },
    "predicted_j_to_measured_aoa": {
        "x": "j_prediction_strength_proxy",
        "y": "aoa_target_log_proxy",
        "label": "Predicted J to measured AOA",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover long-range AOA=C*J^p after calibrated J training."
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config_real_long_logj.yaml"))
    )
    parser.add_argument("--experiment-out", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model(config: dict, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    model = joint.RealProxyModel(config)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device)
    if (
        device.type == "cuda"
        and bool(config["train"].get("data_parallel", True))
        and torch.cuda.device_count() > 1
    ):
        model = torch.nn.DataParallel(model)
    model.eval()
    return model


def infer_partition(
    model,
    config: dict,
    experiment_out: Path,
    formula_out: Path,
    source: str,
    part: str,
    device: torch.device,
    overwrite: bool,
) -> Path:
    destination = formula_out / f"{part}_predictions_{source}.csv"
    if destination.exists() and not overwrite:
        return destination
    if part == "test":
        existing = experiment_out / f"test_predictions_{source}.csv"
        if existing.exists() and not overwrite:
            pd.read_csv(existing).to_csv(destination, index=False)
            return destination
    datasets, stats = training.make_datasets(
        config, experiment_out, None, part, False
    )
    loader = DataLoader(
        datasets[source],
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["evaluation"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    training.evaluate_loader(
        model,
        loader,
        device,
        stats[source],
        bool(config["train"]["amp"]),
        destination,
    )
    return destination


def add_scene_group(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["video"] = result["key"].str.rsplit("/", n=1).str[-1]
    mapping = manifest.set_index("video")["scene_group"]
    result["scene_group"] = result["video"].map(mapping)
    return result.loc[result["scene_group"].notna()].reset_index(drop=True)


def calibration_slope(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    within_scene: bool,
) -> dict:
    table = frame[["scene_group", target_column, prediction_column]].dropna().copy()
    if within_scene:
        table[target_column] -= table.groupby("scene_group")[
            target_column
        ].transform("mean")
        table[prediction_column] -= table.groupby("scene_group")[
            prediction_column
        ].transform("mean")
    x = table[target_column].to_numpy(np.float64)
    y = table[prediction_column].to_numpy(np.float64)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    slope = (
        float(np.dot(centered, y - y.mean()) / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    correlation = (
        float(np.corrcoef(x, y)[0, 1])
        if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12
        else float("nan")
    )
    return {"n": len(table), "slope": slope, "correlation": correlation}


def scale_diagnostics(frame: pd.DataFrame) -> dict:
    positive = frame.loc[
        (frame["j_target_strength_proxy"] > 0)
        & (frame["j_prediction_strength_proxy"] > 0)
    ].copy()
    positive["log_j_target"] = np.log(positive["j_target_strength_proxy"])
    positive["log_j_prediction"] = np.log(
        positive["j_prediction_strength_proxy"]
    )
    result = {}
    for name, target, prediction in (
        (
            "aoa_log_prediction_vs_target",
            "aoa_target_log_proxy",
            "aoa_prediction_log_proxy",
        ),
        (
            "log_j_prediction_vs_target",
            "log_j_target",
            "log_j_prediction",
        ),
    ):
        result[name] = {
            "pooled": calibration_slope(
                positive, target, prediction, within_scene=False
            ),
            "within_scene": calibration_slope(
                positive, target, prediction, within_scene=True
            ),
        }
    return result


def draw_primary(
    out: Path,
    tables: dict,
    results: dict,
) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    rng = np.random.default_rng(20260724)
    for axis, source in zip(axes, SOURCES):
        test = tables[source][1]
        relation = RELATIONS["measured_proxy"]
        xy = grouped.prepare_xy(test, relation["x"], relation["y"])
        if source == RLRAT:
            result = results[source]["measured_proxy"]["grouped"]
            constants = pd.read_csv(
                out / "rlrat_scene_constants_measured_proxy.csv"
            ).set_index("scene_group")["constant"]
            constant = xy["scene_group"].map(constants).fillna(
                result["global_fallback_constant"]
            )
            prediction = np.log(constant.to_numpy()) + result["p"] * xy["x"]
            title = "RLRAT: grouped scene constants"
        else:
            result = results[source]["measured_proxy"]["pooled"]
            prediction = math.log(result["constant"]) + result["p"] * xy["x"]
            title = "Long-range: dataset constant"
        indices = np.arange(len(xy))
        if len(indices) > 6000:
            indices = rng.choice(indices, 6000, replace=False)
        target = xy["y"].to_numpy()
        prediction = np.asarray(prediction)
        lo = min(target[indices].min(), prediction[indices].min())
        hi = max(target[indices].max(), prediction[indices].max())
        axis.scatter(
            target[indices],
            prediction[indices],
            s=9,
            alpha=0.25,
            color="#26738d",
            edgecolors="none",
        )
        axis.plot([lo, hi], [lo, hi], color="#c4473a", linewidth=1.5)
        axis.set_xlabel("Measured log(AOA proxy)")
        axis.set_ylabel("Formula-predicted log(AOA proxy)")
        axis.set_title(
            f"{title}\np={result['p']:.3f}, "
            f"log R2={result['test']['log_r2']:.3f}"
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(figure_dir / "measured_proxy_formula.png", dpi=400)
    plt.close(fig)


def write_report(out: Path, results: dict, diagnostics: dict, checkpoint: Path) -> None:
    lines = [
        "# Long-Range Direct-log-J Formula Discovery",
        "",
        "## Protocol",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Heat Chamber excluded from training and evaluation.",
        "- J latent directly supervised by log positive J strength.",
        "- No AOA-J formula or exponent used during neural-network training.",
        "- Formula coefficients fit on validation and evaluated on untouched test.",
        "- RLRAT uses scene-specific constants and one shared exponent.",
        "",
        "## Formula Results",
        "",
        "| Dataset | Relation | Formula | Test log R2 | Log corr | Log MAE |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    labels = {
        SOURCES[0]: "Long-range",
        RLRAT: "RLRAT grouped",
    }
    for source in SOURCES:
        for relation_name in RELATIONS:
            payload = results[source][relation_name]
            selected = payload["grouped"] if source == RLRAT else payload["pooled"]
            test = selected["test"]
            lines.append(
                f"| {labels[source]} | {RELATIONS[relation_name]['label']} | "
                f"`{selected['formula']}` | {test['log_r2']:.6f} | "
                f"{test['log_corr']:.6f} | {test['log_mae']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## RLRAT Output-Scale Diagnostics",
            "",
            "| Quantity | Pooled slope | Within-scene slope |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, payload in diagnostics.items():
        lines.append(
            f"| {name} | {payload['pooled']['slope']:.6f} | "
            f"{payload['within_scene']['slope']:.6f} |"
        )
    lines.extend(
        [
            "",
            "A slope close to one means the model preserves the target dynamic range.",
            "The model-output exponent should be interpreted only after checking these",
            "calibration slopes. The measured-proxy exponent remains the primary",
            "empirical formula result.",
            "",
            "## Figure",
            "",
            "![Measured proxy formulas](figures/measured_proxy_formula.png)",
        ]
    )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_out = Path(
        args.experiment_out or config["output"]["root"]
    ).resolve()
    out = Path(
        args.out or experiment_out / "aoa_j_formula_discovery_grouped"
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grouped.configure_font()
    checkpoint = Path(
        args.checkpoint or experiment_out / "checkpoints" / "best.pt"
    ).resolve()
    device = joint.resolve_device(str(config["train"]["device"]))
    prediction_files_exist = all(
        (out / f"{part}_predictions_{source}.csv").exists()
        for source in SOURCES
        for part in ("val", "test")
    )
    model = (
        load_model(config, checkpoint, device)
        if args.recompute or not prediction_files_exist
        else None
    )

    manifest = grouped.discover_group_manifest(
        Path(config["data"]["grouped_scenes"])
    )
    manifest.to_csv(out / "group_manifest.csv", index=False)
    tables = {}
    for source in SOURCES:
        partitions = []
        for part in ("val", "test"):
            path = infer_partition(
                model,
                config,
                experiment_out,
                out,
                source,
                part,
                device,
                args.recompute,
            )
            frame = pd.read_csv(path)
            if source == RLRAT:
                frame = add_scene_group(frame, manifest)
            else:
                frame["scene_group"] = source
            partitions.append(frame)
        tables[source] = tuple(partitions)

    results = {}
    constant_tables = {}
    cv_tables = []
    for source_index, source in enumerate(SOURCES):
        calibration, test = tables[source]
        results[source] = {}
        for relation_index, (name, relation) in enumerate(RELATIONS.items()):
            cal_xy = grouped.prepare_xy(
                calibration, relation["x"], relation["y"]
            )
            test_xy = grouped.prepare_xy(test, relation["x"], relation["y"])
            pooled = grouped.evaluate_pooled_relation(cal_xy, test_xy)
            payload = {"pooled": pooled}
            if source == RLRAT:
                result, cv, constants = grouped.evaluate_grouped_relation(
                    cal_xy,
                    test_xy,
                    args.cv_folds,
                    args.bootstrap_repeats,
                    20260724 + source_index * 100 + relation_index,
                )
                payload["grouped"] = result
                cv["relation"] = name
                cv_tables.append(cv)
                constants["relation"] = name
                constant_tables[name] = constants
            results[source][name] = payload

    (out / "formula_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for relation, constants in constant_tables.items():
        constants.to_csv(
            out / f"rlrat_scene_constants_{relation}.csv", index=False
        )
    pd.concat(cv_tables, ignore_index=True).to_csv(
        out / "rlrat_regularization_cv.csv", index=False
    )
    diagnostics = scale_diagnostics(tables[RLRAT][1])
    (out / "scale_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    draw_primary(out, tables, results)
    write_report(out, results, diagnostics, checkpoint)
    config_used = {
        "experiment_out": str(experiment_out),
        "checkpoint": str(checkpoint),
        "heat_chamber_included": False,
        "formula": "AOA_proxy = C_g * J_strength_proxy^p",
        "calibration_split": "validation",
        "evaluation_split": "test",
        "bootstrap_repeats": args.bootstrap_repeats,
        "cv_folds": args.cv_folds,
    }
    with (out / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_used, handle, sort_keys=False)
    print(json.dumps(results[RLRAT], indent=2))
    print(f"Saved formula discovery to {out}")


if __name__ == "__main__":
    main()
