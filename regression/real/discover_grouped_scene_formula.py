from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score


RELATIONS = {
    "measured_proxy": (
        "j_target_log_proxy",
        "aoa_target_log_proxy",
        "Measured proxies",
    ),
    "model_output": (
        "j_prediction_log_proxy",
        "aoa_prediction_log_proxy",
        "Model outputs",
    ),
    "predicted_j_to_measured_aoa": (
        "j_prediction_log_proxy",
        "aoa_target_log_proxy",
        "Predicted J to measured AOA",
    ),
}
SOURCES = (
    "turbulence_sequences",
    "turbulence_sequences_RLRAT",
    "heat_chamber",
)
GROUPED_SOURCE = "turbulence_sequences_RLRAT"
LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover AOA=C_g*J^p with grouped long-range scene constants."
    )
    parser.add_argument(
        "--experiment-out",
        default=str(Path(__file__).with_name("outputs_real_joint_83init")),
    )
    parser.add_argument(
        "--grouped-root", default="data/grouped_scenes"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def configure_font() -> None:
    font_dir = Path(os.environ.get("MSYH_FONT_DIR", "fonts"))
    for path in (
        font_dir / "MSYH.TTC",
        font_dir / "MSYHBD.TTC",
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.unicode_minus": False,
        }
    )


def stable_fold(key: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def prepare_xy(
    frame: pd.DataFrame, x_column: str, y_column: str
) -> pd.DataFrame:
    x_raw = frame[x_column].to_numpy(np.float64)
    y = frame[y_column].to_numpy(np.float64)
    mask = np.isfinite(x_raw) & np.isfinite(y) & (x_raw > 1e-8)
    result = frame.loc[mask, ["key", "scene_group"]].copy()
    result["x"] = np.log(x_raw[mask])
    result["y"] = y[mask]
    return result.reset_index(drop=True)


def fit_pooled(data: pd.DataFrame, fixed_p: float | None = None) -> dict:
    x = data["x"].to_numpy(np.float64)
    y = data["y"].to_numpy(np.float64)
    if fixed_p is None:
        centered = x - x.mean()
        p = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
    else:
        p = float(fixed_p)
    intercept = float(np.mean(y - p * x))
    return {"intercept": intercept, "p": p}


def predict_pooled(model: dict, data: pd.DataFrame) -> np.ndarray:
    return model["intercept"] + model["p"] * data["x"].to_numpy(np.float64)


def fit_grouped(
    data: pd.DataFrame, regularization: float, fixed_p: float | None = None
) -> dict:
    x = data["x"].to_numpy(np.float64)
    y = data["y"].to_numpy(np.float64)
    groups = data["scene_group"].astype(str).to_numpy()
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    sx = np.bincount(inverse, weights=x)
    sy = np.bincount(inverse, weights=y)
    denominator = count + regularization

    if fixed_p is None:
        design = np.column_stack([np.ones(len(x)), x])
        xtx = design.T @ design
        xty = design.T @ y
        aggregate_x = np.column_stack([count, sx])
        correction_x = np.einsum(
            "gi,gj,g->ij", aggregate_x, aggregate_x, 1.0 / denominator
        )
        correction_y = np.einsum(
            "gi,g,g->i", aggregate_x, sy, 1.0 / denominator
        )
        beta = np.linalg.solve(xtx - correction_x, xty - correction_y)
        intercept, p = float(beta[0]), float(beta[1])
    else:
        p = float(fixed_p)
        residual = y - p * x
        sr = np.bincount(inverse, weights=residual)
        intercept = float(
            (residual.sum() - np.sum(count * sr / denominator))
            / (len(residual) - np.sum(count * count / denominator))
        )

    offsets = (sy - count * intercept - p * sx) / denominator
    return {
        "intercept": intercept,
        "p": p,
        "regularization": float(regularization),
        "offsets": {group: float(value) for group, value in zip(unique, offsets)},
        "counts": {group: int(value) for group, value in zip(unique, count)},
    }


def predict_grouped(model: dict, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = data["scene_group"].astype(str)
    offsets = groups.map(model["offsets"])
    known = offsets.notna().to_numpy()
    offset_values = offsets.fillna(0.0).to_numpy(np.float64)
    prediction = (
        model["intercept"]
        + model["p"] * data["x"].to_numpy(np.float64)
        + offset_values
    )
    return prediction, known


def macro_scene_mae(data: pd.DataFrame, prediction: np.ndarray) -> float:
    errors = np.abs(data["y"].to_numpy(np.float64) - prediction)
    table = pd.DataFrame(
        {"scene_group": data["scene_group"].to_numpy(), "error": errors}
    )
    return float(table.groupby("scene_group")["error"].mean().mean())


def metrics(data: pd.DataFrame, prediction: np.ndarray) -> dict:
    target = data["y"].to_numpy(np.float64)
    return {
        "n": int(len(data)),
        "log_r2": float(r2_score(target, prediction)),
        "log_corr": safe_corr(target, prediction),
        "log_mae": float(np.mean(np.abs(target - prediction))),
        "macro_scene_log_mae": macro_scene_mae(data, prediction),
    }


def choose_regularization(
    calibration: pd.DataFrame, folds: int, seed: int, fixed_p: float | None = None
) -> tuple[float, pd.DataFrame]:
    fold_ids = np.asarray(
        [stable_fold(key, seed, folds) for key in calibration["key"]], dtype=int
    )
    rows = []
    for regularization in LAMBDA_GRID:
        predictions = np.full(len(calibration), np.nan, dtype=np.float64)
        known = np.zeros(len(calibration), dtype=bool)
        for fold in range(folds):
            fit_mask = fold_ids != fold
            eval_mask = fold_ids == fold
            model = fit_grouped(
                calibration.loc[fit_mask], regularization, fixed_p=fixed_p
            )
            fold_prediction, fold_known = predict_grouped(
                model, calibration.loc[eval_mask]
            )
            predictions[eval_mask] = fold_prediction
            known[eval_mask] = fold_known
        result = metrics(calibration, predictions)
        rows.append(
            {
                "regularization": regularization,
                "known_scene_fraction": float(known.mean()),
                **result,
            }
        )
    table = pd.DataFrame(rows)
    selected = float(
        table.sort_values(
            ["macro_scene_log_mae", "log_mae", "regularization"]
        ).iloc[0]["regularization"]
    )
    return selected, table


def fit_independent_scene_diagnostic(
    calibration: pd.DataFrame,
    grouped_model: dict,
    minimum_samples: int = 8,
) -> dict:
    models = {}
    for group, subset in calibration.groupby("scene_group"):
        if len(subset) >= minimum_samples and np.std(subset["x"]) > 1e-8:
            models[str(group)] = fit_pooled(subset)
    return {
        "fallback": grouped_model,
        "scene_models": models,
        "minimum_calibration_samples": minimum_samples,
    }


def predict_independent_scene(model: dict, data: pd.DataFrame) -> np.ndarray:
    fallback, _ = predict_grouped(model["fallback"], data)
    prediction = fallback.copy()
    for group, indices in data.groupby("scene_group").groups.items():
        scene_model = model["scene_models"].get(str(group))
        if scene_model is not None:
            prediction[indices] = predict_pooled(scene_model, data.loc[indices])
    return prediction


def cluster_bootstrap_p(
    calibration: pd.DataFrame,
    regularization: float,
    repeats: int,
    seed: int,
) -> list[float]:
    grouped = [part for _, part in calibration.groupby("scene_group")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = []
        for index, selected in enumerate(rng.integers(0, len(grouped), len(grouped))):
            part = grouped[int(selected)].copy()
            part["scene_group"] = f"bootstrap_{index}"
            sampled.append(part)
        try:
            values.append(
                fit_grouped(
                    pd.concat(sampled, ignore_index=True), regularization
                )["p"]
            )
        except np.linalg.LinAlgError:
            continue
    return values


def discover_group_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        for path in directory.glob("*.avi"):
            rows.append(
                {
                    "video": path.stem,
                    "scene_group": directory.name,
                    "grouped_path": str(path),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or result["video"].duplicated().any():
        raise RuntimeError("Grouped-scene manifest is empty or has duplicate videos")
    return result.sort_values(["scene_group", "video"]).reset_index(drop=True)


def add_scene_groups(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["video"] = result["key"].str.rsplit("/", n=1).str[-1]
    mapping = manifest.set_index("video")["scene_group"]
    result["scene_group"] = result["video"].map(mapping)
    return result.loc[result["scene_group"].notna()].reset_index(drop=True)


def evaluate_pooled_relation(
    calibration: pd.DataFrame, test: pd.DataFrame
) -> dict:
    model = fit_pooled(calibration)
    prediction = predict_pooled(model, test)
    fixed = fit_pooled(calibration, fixed_p=1.0)
    return {
        "formula": f"AOA_proxy = {math.exp(model['intercept']):.6g} * J_proxy^{model['p']:.6f}",
        "constant": float(math.exp(model["intercept"])),
        "p": model["p"],
        "test": metrics(test, prediction),
        "fixed_p1_test": metrics(test, predict_pooled(fixed, test)),
    }


def evaluate_grouped_relation(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    folds: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    selected, cv = choose_regularization(calibration, folds, seed)
    model = fit_grouped(calibration, selected)
    prediction, known = predict_grouped(model, test)

    selected_p1, cv_p1 = choose_regularization(
        calibration, folds, seed + 17, fixed_p=1.0
    )
    model_p1 = fit_grouped(calibration, selected_p1, fixed_p=1.0)
    prediction_p1, _ = predict_grouped(model_p1, test)

    independent = fit_independent_scene_diagnostic(calibration, model)
    independent_prediction = predict_independent_scene(independent, test)
    bootstrap = cluster_bootstrap_p(
        calibration, selected, bootstrap_repeats, seed + 31
    )
    ci = (
        [float(v) for v in np.quantile(bootstrap, [0.025, 0.975])]
        if bootstrap
        else [float("nan"), float("nan")]
    )

    constant_rows = []
    test_counts = test.groupby("scene_group").size()
    for group, offset in model["offsets"].items():
        constant_rows.append(
            {
                "scene_group": group,
                "calibration_n": model["counts"][group],
                "test_n": int(test_counts.get(group, 0)),
                "log_constant": model["intercept"] + offset,
                "constant": float(math.exp(model["intercept"] + offset)),
                "offset_from_global": offset,
            }
        )
    constants = pd.DataFrame(constant_rows)
    result = {
        "formula": f"AOA_proxy = C_g * J_proxy^{model['p']:.6f}",
        "global_fallback_constant": float(math.exp(model["intercept"])),
        "p": model["p"],
        "p_scene_bootstrap_ci95": ci,
        "regularization": selected,
        "known_test_scene_fraction": float(known.mean()),
        "known_test_scene_count": int(
            test.loc[known, "scene_group"].nunique()
        ),
        "test_scene_count": int(test["scene_group"].nunique()),
        "test": metrics(test, prediction),
        "fixed_p1_regularization": selected_p1,
        "fixed_p1_test": metrics(test, prediction_p1),
        "independent_scene_exponent_count": len(independent["scene_models"]),
        "independent_scene_exponent_test": metrics(
            test, independent_prediction
        ),
    }
    cv = cv.assign(model="shared_p_grouped_C")
    cv_p1 = cv_p1.assign(model="fixed_p1_grouped_C")
    return result, pd.concat([cv, cv_p1], ignore_index=True), constants


def draw_outputs(
    out: Path,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    result: dict,
    pooled: dict,
    constants: pd.DataFrame,
) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    grouped_model = fit_grouped(calibration, result["regularization"])
    grouped_prediction, _ = predict_grouped(grouped_model, test)
    pooled_model = fit_pooled(calibration)
    pooled_prediction = predict_pooled(pooled_model, test)

    rng = np.random.default_rng(20260724)
    indices = np.arange(len(test))
    if len(indices) > 6000:
        indices = rng.choice(indices, 6000, replace=False)
    target = test["y"].to_numpy()
    lo = float(
        min(target[indices].min(), grouped_prediction[indices].min())
    )
    hi = float(
        max(target[indices].max(), grouped_prediction[indices].max())
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    for axis, prediction, title, score in (
        (
            axes[0],
            pooled_prediction,
            "One constant for all scenes",
            pooled["test"]["log_r2"],
        ),
        (
            axes[1],
            grouped_prediction,
            "Scene constants, shared exponent",
            result["test"]["log_r2"],
        ),
    ):
        axis.scatter(
            target[indices],
            prediction[indices],
            s=9,
            alpha=0.24,
            color="#26738d",
            edgecolors="none",
            rasterized=True,
        )
        axis.plot([lo, hi], [lo, hi], color="#c4473a", linewidth=1.6)
        axis.set_xlabel("Measured log(AOA proxy)")
        axis.set_ylabel("Formula-predicted log(AOA proxy)")
        axis.set_title(f"{title}\nTest log R2 = {score:.3f}")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(figures / "rlrat_pooled_vs_grouped.png", dpi=400)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    axes[0].hist(
        np.log(constants["constant"]),
        bins=45,
        color="#26738d",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].set_xlabel("Scene log constant, log(C_g)")
    axes[0].set_ylabel("Number of scenes")
    axes[0].set_title("Distribution of calibrated scene constants")
    axes[1].scatter(
        constants["calibration_n"],
        constants["offset_from_global"],
        s=13,
        alpha=0.35,
        color="#d08b31",
        edgecolors="none",
    )
    axes[1].set_xscale("log")
    axes[1].axhline(0.0, color="#6f6f6f", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Calibration samples per scene")
    axes[1].set_ylabel("Scene offset from global log(C)")
    axes[1].set_title("Constant estimates and scene sample size")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(figures / "rlrat_scene_constants.png", dpi=400)
    plt.close(fig)


def write_report(
    out: Path,
    results: dict,
    grouped_constants: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    primary = results[GROUPED_SOURCE]["measured_proxy"]
    grouped = primary["grouped"]
    pooled = primary["pooled"]
    ci = grouped["p_scene_bootstrap_ci95"]
    lines = [
        "# Grouped-Scene AOA-J Formula Discovery",
        "",
        "## Main Result",
        "",
        "The primary RLRAT model uses one shared physical exponent and a",
        "scene-specific calibration constant:",
        "",
        f"`AOA_proxy = C_g * J_proxy^{grouped['p']:.4f}`",
        "",
        f"The scene-cluster bootstrap 95% CI for `p` is "
        f"[{ci[0]:.4f}, {ci[1]:.4f}]. The untouched test log R2 improves "
        f"from {pooled['test']['log_r2']:.4f} with one pooled constant to "
        f"{grouped['test']['log_r2']:.4f} with grouped constants.",
        "",
        "## Why There Is Not a Separate Free Formula for Every Scene",
        "",
        "The camera geometry, distance, scene scale, and proxy calibration are",
        "absorbed into `C_g`. The turbulence relation is represented by the shared",
        "exponent `p`. Allowing every scene to estimate both `C_g` and `p_g` would",
        "turn the experiment into hundreds of small regressions; 24 scenes contain",
        "only one clip and many others have too little dynamic range to identify an",
        "exponent. It would improve fit by adding degrees of freedom without",
        "providing stronger evidence for a common physical relationship.",
        "",
            "Scene constants were therefore estimated with ridge shrinkage toward the",
            "global constant. The shrinkage strength was selected by five-fold",
            "cross-validation inside the validation split. Formula coefficients were",
            "then frozen and evaluated on the original test split.",
            "",
            "This is a known-scene calibration protocol: test clips use `C_g` only",
            "when other validation clips from the same acquisition group are",
            "available. A completely new scene must either use the global fallback",
            "constant or provide a small calibration subset before scene-specific",
            "absolute AOA values can be evaluated.",
        "",
        "## Test Results",
        "",
        "| Dataset / relation | Formula model | p | Test log R2 | Log corr | Macro-scene log MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "measured_proxy": "Measured proxies",
        "model_output": "Model outputs",
        "predicted_j_to_measured_aoa": "Predicted J -> measured AOA",
    }
    for source in SOURCES:
        for relation, payload in results[source].items():
            if source == GROUPED_SOURCE:
                for model_name in ("pooled", "grouped"):
                    item = payload[model_name]
                    test = item["test"]
                    lines.append(
                        f"| {source}: {labels[relation]} | {model_name} | "
                        f"{item['p']:.4f} | {test['log_r2']:.4f} | "
                        f"{test['log_corr']:.4f} | "
                        f"{test['macro_scene_log_mae']:.4f} |"
                    )
            else:
                item = payload["pooled"]
                test = item["test"]
                lines.append(
                    f"| {source}: {labels[relation]} | one constant | "
                    f"{item['p']:.4f} | {test['log_r2']:.4f} | "
                    f"{test['log_corr']:.4f} | "
                    f"{test['macro_scene_log_mae']:.4f} |"
                )

    lines.extend(
        [
            "",
            "## Evaluation Rule",
            "",
            "The primary score is test log R2 for the measured-proxy relationship.",
            "Macro-scene log MAE is reported alongside it so that large scenes do not",
            "dominate the conclusion. Improvement over the pooled-constant baseline",
            "tests the stated hypothesis that acquisition groups require different",
            "constants. Log correlation alone is not sufficient because it is",
            "unchanged by constant calibration errors.",
            "",
            f"The grouped manifest contains {len(manifest):,} videos in "
            f"{manifest['scene_group'].nunique():,} scenes. "
            f"{grouped['known_test_scene_fraction'] * 100:.2f}% of usable test clips "
            "belong to a scene observed during formula calibration. Remaining test",
            "clips use the global fallback constant and are retained in all metrics.",
            "",
            "The independent-scene-exponent model is diagnostic only. It estimates",
            f"free exponents for {grouped['independent_scene_exponent_count']} scenes "
            "with at least eight calibration clips and falls back to the shared",
            "model elsewhere. It must not replace the shared-exponent result unless",
            "it yields a substantial, repeatable test improvement and the recovered",
            "scene exponents are statistically consistent.",
            "",
            "## Dataset Handling",
            "",
            "`grouped_scenes` matches the RLRAT filenames except for `real_00001`,",
            "which is absent from the grouped directory and excluded here.",
            "`turbulence_sequences` has no supplied group mapping and is evaluated",
            "with one dataset-level constant. Heat Chamber is evaluated with one",
            "constant for the entire dataset, as specified.",
            "",
            "## Figures",
            "",
            "![Pooled versus grouped](figures/rlrat_pooled_vs_grouped.png)",
            "",
            "![Scene constants](figures/rlrat_scene_constants.png)",
            "",
            "## Artifacts",
            "",
            "- `formula_results.json`: complete pooled and grouped results",
            "- `rlrat_scene_constants_*.csv`: calibrated constants for each relation",
            "- `rlrat_scene_constants.csv`: primary measured-proxy constants",
            "- `rlrat_regularization_cv.csv`: validation-fold model selection",
            "- `group_manifest.csv`: video-to-scene mapping",
            "- `config_used.yaml`: exact experiment inputs and protocol",
        ]
    )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_out = Path(args.experiment_out).resolve()
    grouped_root = Path(args.grouped_root).resolve()
    out = Path(
        args.out or experiment_out / "aoa_j_formula_discovery_grouped_scenes"
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    configure_font()

    manifest = discover_group_manifest(grouped_root)
    manifest.to_csv(out / "group_manifest.csv", index=False)
    config_used = {
        "experiment_out": str(experiment_out),
        "checkpoint": str(experiment_out / "checkpoints" / "best.pt"),
        "grouped_root": str(grouped_root),
        "grouped_source": GROUPED_SOURCE,
        "formula": "AOA_proxy = C_g * J_proxy^p",
        "calibration_split": "validation",
        "evaluation_split": "test",
        "cv_folds": args.cv_folds,
        "regularization_grid": list(LAMBDA_GRID),
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
    }
    with (out / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_used, handle, sort_keys=False)

    previous = experiment_out / "aoa_j_formula_discovery"
    tables = {}
    for source in SOURCES:
        calibration = pd.read_csv(previous / f"val_predictions_{source}.csv")
        test = pd.read_csv(previous / f"test_predictions_{source}.csv")
        if source == GROUPED_SOURCE:
            calibration = add_scene_groups(calibration, manifest)
            test = add_scene_groups(test, manifest)
        else:
            calibration["scene_group"] = source
            test["scene_group"] = source
        tables[source] = (calibration, test)

    results = {}
    cv_tables = []
    constants_tables = {}
    constants_primary = None
    primary_xy = None
    for source_index, source in enumerate(SOURCES):
        calibration, test = tables[source]
        results[source] = {}
        for relation_index, (name, columns) in enumerate(RELATIONS.items()):
            x_column, y_column, _ = columns
            cal_xy = prepare_xy(calibration, x_column, y_column)
            test_xy = prepare_xy(test, x_column, y_column)
            pooled = evaluate_pooled_relation(cal_xy, test_xy)
            payload = {"pooled": pooled}
            if source == GROUPED_SOURCE:
                grouped, cv, constants = evaluate_grouped_relation(
                    cal_xy,
                    test_xy,
                    args.cv_folds,
                    args.bootstrap_repeats,
                    args.seed + source_index * 100 + relation_index,
                )
                payload["grouped"] = grouped
                cv["relation"] = name
                cv_tables.append(cv)
                constants["relation"] = name
                constants_tables[name] = constants
                if name == "measured_proxy":
                    constants_primary = constants
                    primary_xy = (cal_xy, test_xy)
            results[source][name] = payload

    (out / "formula_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.concat(cv_tables, ignore_index=True).to_csv(
        out / "rlrat_regularization_cv.csv", index=False
    )
    for relation, constants in constants_tables.items():
        constants.to_csv(
            out / f"rlrat_scene_constants_{relation}.csv", index=False
        )
    constants_primary.to_csv(out / "rlrat_scene_constants.csv", index=False)
    draw_outputs(
        out,
        primary_xy[0],
        primary_xy[1],
        results[GROUPED_SOURCE]["measured_proxy"]["grouped"],
        results[GROUPED_SOURCE]["measured_proxy"]["pooled"],
        constants_primary,
    )
    write_report(out, results, constants_primary, manifest)
    print(json.dumps(results[GROUPED_SOURCE]["measured_proxy"], indent=2))
    print(f"Saved grouped-scene experiment to {out}")


if __name__ == "__main__":
    main()
