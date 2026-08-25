from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREDICTION_SUFFIXES = (
    "mixed_loglaw_test_predictions.csv",
    "powerlaw_test_predictions.csv",
    "pysr_test_predictions.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reusable visual reports for latent symbolic regression outputs."
    )
    parser.add_argument(
        "--run-dir",
        default="outputs_powerlaw_disp_head_from_scratch",
        help="Training output directory that contains checkpoint_symbolic_eval/.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_composite",
        help="Checkpoint tag under the symbolic evaluation directory, e.g. best_composite or epoch_0040.",
    )
    parser.add_argument(
        "--eval-dir",
        default="checkpoint_symbolic_eval",
        help="Symbolic evaluation directory below --run-dir."
        " Use checkpoint_symbolic_eval_corrected_root for the corrected dataset-root evaluation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["latent_disp_AOAlatent", "latent_disp_AOA_extra", "latent_disp_full"],
        help="Symbolic result names to visualize.",
    )
    parser.add_argument(
        "--oracle",
        default="oracle_disp",
        help="Oracle/real-formula result name used for formula-level comparison. Use '' to disable.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to <run-dir>/visualizations/<checkpoint>.",
    )
    parser.add_argument("--max-points", type=int, default=6000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def find_prediction_file(symbolic_dir: Path, name: str) -> Path | None:
    model_dir = symbolic_dir / name
    if not model_dir.exists():
        return None
    for suffix in PREDICTION_SUFFIXES:
        matches = sorted(model_dir.glob(suffix))
        if matches:
            return matches[0]
    matches = sorted(model_dir.glob("*_test_predictions.csv"))
    return matches[0] if matches else None


def detect_target_and_pred(df: pd.DataFrame) -> tuple[str, str]:
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    if len(pred_cols) != 1:
        raise ValueError(f"Expected exactly one *_pred column, found {pred_cols}")
    pred_col = pred_cols[0]
    target = pred_col[: -len("_pred")]
    if target not in df.columns:
        numeric = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c != pred_col and not c.endswith("_pred")
        ]
        if not numeric:
            raise ValueError(f"Could not infer target column for {pred_col}")
        target = numeric[-1]
    return target, pred_col


def read_results(symbolic_dir: Path) -> pd.DataFrame:
    result_csv = symbolic_dir / "symbolic_results.csv"
    if not result_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(result_csv)


def read_metrics(symbolic_dir: Path, names: Iterable[str]) -> pd.DataFrame:
    results = read_results(symbolic_dir)
    if results.empty:
        return pd.DataFrame()
    rows = []
    for name in names:
        row = results[results["name"] == name]
        if not row.empty:
            rows.append(row.iloc[0].to_dict())
    return pd.DataFrame(rows)


def safe_sample(df: pd.DataFrame, max_points: int, random_state: int = 0) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    return df.sample(max_points, random_state=random_state).sort_index()


def positive_limits(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1e-12, 1.0
    lo = float(np.nanpercentile(values, 0.5))
    hi = float(np.nanpercentile(values, 99.5))
    if not math.isfinite(lo) or lo <= 0:
        lo = float(values.min())
    if not math.isfinite(hi) or hi <= lo:
        hi = float(values.max())
    pad = (hi / lo) ** 0.04 if lo > 0 and hi > lo else 1.2
    return lo / pad, hi * pad


def annotate_equation(ax: plt.Axes, equation: str, metrics: dict) -> None:
    parts = []
    if equation:
        parts.append(equation)
    for key in ["r2", "rmse", "mae"]:
        value = metrics.get(key)
        if value is not None and pd.notna(value):
            parts.append(f"{key.upper()}={float(value):.4g}")
    text = "\n".join(parts)
    if text:
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "#d0d0d0"},
        )


def plot_true_vs_pred(
    df: pd.DataFrame,
    target: str,
    pred_col: str,
    name: str,
    metrics: dict,
    out_path: Path,
    max_points: int,
    dpi: int,
) -> None:
    sampled = safe_sample(df[[target, pred_col]].dropna(), max_points)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(sampled[target], sampled[pred_col], s=9, alpha=0.42, color="#2563eb", linewidths=0)
    lo, hi = positive_limits(np.concatenate([sampled[target].to_numpy(), sampled[pred_col].to_numpy()]))
    ax.plot([lo, hi], [lo, hi], color="#111827", lw=1.2, label="ideal: y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"true {target}")
    ax.set_ylabel(f"predicted {target}")
    ax.set_title(f"{name}: true value vs fitted value")
    ax.grid(True, which="both", color="#e5e7eb", lw=0.6)
    ax.legend(loc="lower right", frameon=False)
    annotate_equation(ax, str(metrics.get("equation", "")), metrics)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_residuals(
    df: pd.DataFrame,
    target: str,
    pred_col: str,
    name: str,
    out_path: Path,
    max_points: int,
    dpi: int,
) -> None:
    work = df[[target, pred_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = work[(work[target] != 0) & np.isfinite(work[target]) & np.isfinite(work[pred_col])]
    work["residual"] = work[pred_col] - work[target]
    work["relative_error"] = work["residual"] / work[target]
    sampled = safe_sample(work, max_points)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    axes[0].scatter(sampled[target], sampled["relative_error"], s=8, alpha=0.35, color="#0891b2", linewidths=0)
    axes[0].axhline(0, color="#111827", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(f"true {target}")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("relative residuals")
    axes[0].grid(True, which="both", color="#e5e7eb", lw=0.6)

    clipped = sampled["relative_error"].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = np.nanpercentile(clipped, [1, 99]) if len(clipped) else (-1, 1)
    clipped = clipped.clip(lo, hi)
    axes[1].hist(clipped, bins=60, color="#f97316", alpha=0.82)
    axes[1].axvline(0, color="#111827", lw=1)
    axes[1].set_xlabel("relative error")
    axes[1].set_ylabel("count")
    axes[1].set_title("relative error distribution")
    axes[1].grid(True, axis="y", color="#e5e7eb", lw=0.6)

    fig.suptitle(f"{name}: residual diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def choose_x_column(df: pd.DataFrame, target: str, pred_col: str) -> str | None:
    preferred = ["AOA_latent", "AOA_var", "hAOA", "J_latent", "hJ"]
    for col in preferred:
        if col in df.columns:
            return col
    numeric = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {target, pred_col, "Delta_theta", "h3", "h4"}
    ]
    return numeric[0] if numeric else None


def plot_delta_slices(
    df: pd.DataFrame,
    target: str,
    pred_col: str,
    name: str,
    out_path: Path,
    dpi: int,
) -> bool:
    if "Delta_theta" not in df.columns:
        return False
    x_col = choose_x_column(df, target, pred_col)
    if x_col is None:
        return False
    work = df[[x_col, "Delta_theta", target, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work[x_col] > 0) & (work[target] > 0) & (work[pred_col] > 0)]
    if work.empty:
        return False

    unique_delta = np.sort(work["Delta_theta"].unique())
    if len(unique_delta) > 4:
        idx = np.linspace(0, len(unique_delta) - 1, 4).round().astype(int)
        selected = unique_delta[idx]
    else:
        selected = unique_delta

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]
    for color, delta in zip(colors, selected):
        part = work[np.isclose(work["Delta_theta"], delta)].sort_values(x_col)
        if part.empty:
            continue
        if len(part) > 450:
            part = part.iloc[np.linspace(0, len(part) - 1, 450).round().astype(int)]
        ax.scatter(part[x_col], part[target], s=8, alpha=0.25, color=color, linewidths=0)
        ax.plot(part[x_col], part[pred_col], color=color, lw=1.6, label=f"fit Delta={delta:.3g}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel(target)
    ax.set_title(f"{name}: formula curves by Delta_theta")
    ax.grid(True, which="both", color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_formula_comparison(
    model_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    target: str,
    model_pred_col: str,
    oracle_pred_col: str,
    name: str,
    oracle_name: str,
    out_path: Path,
    max_points: int,
    dpi: int,
) -> bool:
    n = min(len(model_df), len(oracle_df))
    if n == 0:
        return False
    model = model_df.iloc[:n].reset_index(drop=True)
    oracle = oracle_df.iloc[:n].reset_index(drop=True)
    if target not in model.columns or target not in oracle.columns:
        return False
    work = pd.DataFrame(
        {
            "true": model[target],
            "fit": model[model_pred_col],
            "oracle": oracle[oracle_pred_col],
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    work = work[(work["true"] > 0) & (work["fit"] > 0) & (work["oracle"] > 0)]
    if work.empty:
        return False
    work = safe_sample(work, max_points).sort_values("true").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    x = np.arange(len(work))
    axes[0].plot(x, work["true"], color="#111827", lw=1.2, label="measured target")
    axes[0].plot(x, work["oracle"], color="#059669", lw=1.2, label=f"{oracle_name} prediction")
    axes[0].plot(x, work["fit"], color="#dc2626", lw=1.2, label=f"{name} prediction")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("test samples sorted by true target")
    axes[0].set_ylabel(target)
    axes[0].set_title("real/oracle formula vs fitted latent formula")
    axes[0].grid(True, which="both", color="#e5e7eb", lw=0.6)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].scatter(work["oracle"], work["fit"], s=9, alpha=0.38, color="#7c3aed", linewidths=0)
    lo, hi = positive_limits(np.concatenate([work["oracle"].to_numpy(), work["fit"].to_numpy()]))
    axes[1].plot([lo, hi], [lo, hi], color="#111827", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(lo, hi)
    axes[1].set_ylim(lo, hi)
    axes[1].set_xlabel(f"{oracle_name} prediction")
    axes[1].set_ylabel(f"{name} prediction")
    axes[1].set_title("formula prediction agreement")
    axes[1].grid(True, which="both", color="#e5e7eb", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_metrics(metrics_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if metrics_df.empty:
        return False
    cols = [c for c in ["r2", "rmse", "mae"] if c in metrics_df.columns]
    if not cols:
        return False
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.3))
    if len(cols) == 1:
        axes = [axes]
    labels = metrics_df["name"].tolist()
    for ax, col in zip(axes, cols):
        values = pd.to_numeric(metrics_df[col], errors="coerce")
        ax.bar(labels, values, color="#2563eb" if col == "r2" else "#f97316")
        ax.set_title(col.upper())
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.6)
    fig.suptitle("Symbolic formula metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def semantic_aoa_exponent(row: pd.Series) -> float:
    for col in ["exp_AOA_var", "exp_AOA_latent", "exp_hAOA"]:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return np.nan


def semantic_j_exponent(row: pd.Series) -> float:
    for col in ["exp_J", "exp_J_latent", "exp_hJ"]:
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return np.nan


def plot_formula_parameters(metrics_df: pd.DataFrame, out_path: Path, dpi: int) -> bool:
    if metrics_df.empty:
        return False
    rows = []
    for _, row in metrics_df.iterrows():
        rows.append(
            {
                "name": row.get("name", ""),
                "constant": float(row["constant"]) if "constant" in row and pd.notna(row["constant"]) else np.nan,
                "J exponent": semantic_j_exponent(row),
                "D aperture exponent": float(row["exp_D_aperture"])
                if "exp_D_aperture" in row and pd.notna(row["exp_D_aperture"])
                else np.nan,
                "AOA exponent": semantic_aoa_exponent(row),
                "Delta exponent": float(row["exp_Delta_theta"])
                if "exp_Delta_theta" in row and pd.notna(row["exp_Delta_theta"])
                else np.nan,
            }
        )
    param_df = pd.DataFrame(rows)
    params = ["constant", "J exponent", "D aperture exponent", "AOA exponent", "Delta exponent"]
    params = [param for param in params if not param_df[param].isna().all()]
    if param_df[params].isna().all().all():
        return False

    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 4.3))
    if len(params) == 1:
        axes = [axes]
    for ax, param in zip(axes, params):
        values = param_df[param]
        ax.bar(param_df["name"], values, color="#059669" if param != "constant" else "#2563eb")
        ax.set_title(param)
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", color="#e5e7eb", lw=0.6)
        if param == "constant":
            positive = values[np.isfinite(values) & (values > 0)]
            if len(positive) and positive.max() / positive.min() > 20:
                ax.set_yscale("log")
    fig.suptitle("Oracle/real formula parameters vs latent fitted formula parameters")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_training_curves(run_dir: Path, out_path: Path, dpi: int) -> bool:
    train_log = run_dir / "train_log.csv"
    if not train_log.exists():
        return False
    df = pd.read_csv(train_log)
    if "epoch" not in df.columns:
        return False
    panels = [
        ("R2_disp_val", "disp validation R2"),
        ("score_composite", "composite score"),
        ("corr_zAOA_logAOA", "latent AOA correlation"),
        ("loss_aoa_formula", "AOA formula loss"),
    ]
    panels = [(c, title) for c, title in panels if c in df.columns]
    if not panels:
        return False
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.8, 2.4 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, title) in zip(axes, panels):
        ax.plot(df["epoch"], df[col], color="#2563eb", lw=1.8)
        best_idx = df[col].idxmin() if col.startswith("loss") else df[col].idxmax()
        ax.scatter(df.loc[best_idx, "epoch"], df.loc[best_idx, col], color="#dc2626", s=28, zorder=3)
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.grid(True, color="#e5e7eb", lw=0.6)
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def write_text_report(out_dir: Path, checkpoint: str, metrics_df: pd.DataFrame) -> Path:
    path = out_dir / "equation_report.txt"
    lines = [f"Checkpoint: {checkpoint}", ""]
    if metrics_df.empty:
        lines.append("No symbolic_results.csv metrics were found.")
    else:
        for _, row in metrics_df.iterrows():
            lines.append(f"[{row.get('name', '')}]")
            lines.append(f"backend: {row.get('backend', '')}")
            lines.append(f"equation: {row.get('equation', '')}")
            for key in ["r2", "rmse", "mae", "mape"]:
                if key in row and pd.notna(row[key]):
                    lines.append(f"{key}: {row[key]}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_html_index(out_dir: Path, checkpoint: str, metrics_df: pd.DataFrame, images: list[Path]) -> Path:
    rows = []
    if not metrics_df.empty:
        for _, row in metrics_df.iterrows():
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('name', '')))}</td>"
                f"<td>{html.escape(str(row.get('backend', '')))}</td>"
                f"<td>{float(row.get('r2')):.4g}</td>"
                f"<td><code>{html.escape(str(row.get('equation', '')))}</code></td>"
                "</tr>"
            )
    image_blocks = []
    for image in images:
        rel = image.name
        image_blocks.append(f"<section><h2>{html.escape(image.stem)}</h2><img src=\"{html.escape(rel)}\" /></section>")
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Symbolic Visualization - {html.escape(checkpoint)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    code {{ white-space: pre-wrap; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
    section {{ margin: 28px 0; }}
  </style>
</head>
<body>
  <h1>Symbolic Visualization: {html.escape(checkpoint)}</h1>
  <table>
    <thead><tr><th>Name</th><th>Backend</th><th>R2</th><th>Equation</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {''.join(image_blocks)}
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / args.eval_dir / args.checkpoint
    symbolic_dir = checkpoint_dir / "symbolic"
    if not symbolic_dir.exists():
        raise FileNotFoundError(f"Missing symbolic directory: {symbolic_dir}")

    out_dir = Path(args.out) if args.out else run_dir / "visualizations" / args.checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(dict.fromkeys([m for m in args.models if m]))
    oracle_name = args.oracle.strip()
    all_metric_names = ([oracle_name] if oracle_name else []) + model_names
    metrics_df = read_metrics(symbolic_dir, all_metric_names)
    metrics_by_name = {
        row["name"]: row.to_dict() for _, row in metrics_df.iterrows() if "name" in row and pd.notna(row["name"])
    }

    images: list[Path] = []
    metrics_path = out_dir / "formula_metrics.png"
    if plot_metrics(metrics_df, metrics_path, args.dpi):
        images.append(metrics_path)

    params_path = out_dir / "formula_parameters_comparison.png"
    if plot_formula_parameters(metrics_df, params_path, args.dpi):
        images.append(params_path)

    train_path = out_dir / "training_curves.png"
    if plot_training_curves(run_dir, train_path, args.dpi):
        images.append(train_path)

    oracle_df = None
    oracle_target = None
    oracle_pred_col = None
    if oracle_name:
        oracle_pred_file = find_prediction_file(symbolic_dir, oracle_name)
        if oracle_pred_file is not None:
            oracle_df = pd.read_csv(oracle_pred_file)
            oracle_target, oracle_pred_col = detect_target_and_pred(oracle_df)

    generated = {}
    for name in model_names:
        pred_file = find_prediction_file(symbolic_dir, name)
        if pred_file is None:
            print(f"skip {name}: no prediction CSV found")
            continue
        df = pd.read_csv(pred_file)
        target, pred_col = detect_target_and_pred(df)
        metrics = metrics_by_name.get(name, {})

        true_pred_path = out_dir / f"{name}_true_vs_pred.png"
        plot_true_vs_pred(df, target, pred_col, name, metrics, true_pred_path, args.max_points, args.dpi)
        images.append(true_pred_path)

        residual_path = out_dir / f"{name}_residuals.png"
        plot_residuals(df, target, pred_col, name, residual_path, args.max_points, args.dpi)
        images.append(residual_path)

        delta_path = out_dir / f"{name}_delta_slices.png"
        if plot_delta_slices(df, target, pred_col, name, delta_path, args.dpi):
            images.append(delta_path)

        if oracle_df is not None and oracle_target == target and oracle_pred_col is not None:
            compare_path = out_dir / f"{name}_vs_{oracle_name}_formula_comparison.png"
            if plot_formula_comparison(
                df,
                oracle_df,
                target,
                pred_col,
                oracle_pred_col,
                name,
                oracle_name,
                compare_path,
                args.max_points,
                args.dpi,
            ):
                images.append(compare_path)

        generated[name] = {
            "prediction_file": str(pred_file),
            "target": target,
            "prediction_column": pred_col,
            "metrics": metrics,
        }

    report_path = write_text_report(out_dir, args.checkpoint, metrics_df)
    html_path = write_html_index(out_dir, args.checkpoint, metrics_df, images)
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "symbolic_dir": str(symbolic_dir),
        "output_dir": str(out_dir),
        "models": generated,
        "images": [str(p) for p in images],
        "equation_report": str(report_path),
        "html_index": str(html_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(images)} images")
    print(f"HTML report: {html_path}")
    print(f"Equation report: {report_path}")


if __name__ == "__main__":
    main()
