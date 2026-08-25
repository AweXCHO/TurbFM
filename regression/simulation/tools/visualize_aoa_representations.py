from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize AOA_var, AOA_latent, aoa_hat, and AOA_proxy.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best_composite")
    parser.add_argument("--eval-root", default="checkpoint_symbolic_eval")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-points", type=int, default=6000)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - residual / total) if total > 0 else float("nan")


def log_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    return float(np.corrcoef(np.log(x[valid]), np.log(y[valid]))[0, 1])


def positive_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=float).reshape(-1) for a in arrays])
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = np.percentile(values, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    pad = (hi / lo) ** 0.04 if hi > lo > 0 else 1.2
    return float(lo / pad), float(hi * pad)


def sample_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=0)


def plot_pair(ax, true: np.ndarray, pred: np.ndarray, title: str, true_name: str, pred_name: str) -> None:
    lo, hi = positive_limits(true, pred)
    ax.scatter(true, pred, s=8, alpha=0.32, linewidths=0, color="#2563eb")
    ax.plot([lo, hi], [lo, hi], color="#111827", lw=1.1, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel(true_name)
    ax.set_ylabel(pred_name)
    ax.grid(True, which="both", color="#e5e7eb", lw=0.6)
    ax.legend(frameon=False, fontsize=8)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    table = run_dir / args.eval_root / args.checkpoint / "latent_tables" / f"latent_table_{args.split}.csv"
    if not table.exists():
        raise FileNotFoundError(f"Missing latent table: {table}")
    out_dir = Path(args.out) if args.out else run_dir / "visualizations_aoa" / args.eval_root / args.checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table)
    required = ["AOA_var", "AOA_latent", "aoa_hat", "disp_var", "disp_hat", "Delta_theta"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Latent table is missing columns: {missing}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    df["AOA_proxy"] = df["disp_var"] * df["Delta_theta"] ** 2
    df["AOA_proxy_pred"] = df["disp_hat"] * df["Delta_theta"] ** 2
    df = df[(df["AOA_var"] > 0) & (df["AOA_latent"] > 0) & (df["aoa_hat"] > 0)]
    df = df[(df["AOA_proxy"] > 0) & (df["AOA_proxy_pred"] > 0)]
    plot_df = sample_frame(df, args.max_points)

    metrics = {
        "rows": int(len(df)),
        "AOA_var_vs_AOA_latent_r2": r2_score(df["AOA_var"].to_numpy(), df["AOA_latent"].to_numpy()),
        "AOA_var_vs_AOA_latent_log_corr": log_corr(df["AOA_var"].to_numpy(), df["AOA_latent"].to_numpy()),
        "AOA_var_vs_aoa_hat_r2": r2_score(df["AOA_var"].to_numpy(), df["aoa_hat"].to_numpy()),
        "AOA_var_vs_aoa_hat_log_corr": log_corr(df["AOA_var"].to_numpy(), df["aoa_hat"].to_numpy()),
        "AOA_proxy_vs_proxy_pred_r2": r2_score(df["AOA_proxy"].to_numpy(), df["AOA_proxy_pred"].to_numpy()),
        "AOA_proxy_vs_AOA_var_log_corr": log_corr(df["AOA_proxy"].to_numpy(), df["AOA_var"].to_numpy()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    plot_pair(axes[0], plot_df["AOA_var"].to_numpy(), plot_df["AOA_latent"].to_numpy(), "AOA_var vs AOA_latent", "true AOA_var", "AOA_latent")
    plot_pair(axes[1], plot_df["AOA_var"].to_numpy(), plot_df["aoa_hat"].to_numpy(), "AOA_var vs network prediction", "true AOA_var", "aoa_hat")
    plot_pair(axes[2], plot_df["AOA_proxy"].to_numpy(), plot_df["AOA_proxy_pred"].to_numpy(), "AOA_proxy from displacement", "true AOA_proxy", "predicted AOA_proxy")
    fig.suptitle(f"AOA representations: {args.checkpoint} ({args.split})")
    fig.tight_layout()
    fig.savefig(out_dir / "aoa_three_representations.png", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
    lo, hi = positive_limits(plot_df["AOA_var"].to_numpy(), plot_df["AOA_proxy"].to_numpy())
    axes[0].scatter(plot_df["AOA_var"], plot_df["AOA_proxy"], s=8, alpha=0.3, linewidths=0, color="#059669")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel("AOA_var")
    axes[0].set_ylabel("AOA_proxy = disp_var * Delta_theta^2")
    axes[0].set_title("Physical AOA proxy relation")
    axes[0].grid(True, which="both", color="#e5e7eb", lw=0.6)

    for column, label, color in [
        ("AOA_var", "AOA_var", "#2563eb"),
        ("AOA_latent", "AOA_latent", "#059669"),
        ("aoa_hat", "aoa_hat", "#d97706"),
    ]:
        axes[1].hist(np.log10(plot_df[column]), bins=60, density=True, alpha=0.42, label=label, color=color)
    axes[1].set_xlabel("log10(value)")
    axes[1].set_ylabel("density")
    axes[1].set_title("AOA representation distributions")
    axes[1].grid(True, axis="y", color="#e5e7eb", lw=0.6)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "aoa_proxy_and_distributions.png", dpi=args.dpi)
    plt.close(fig)

    (out_dir / "aoa_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [f"# AOA Visualization Metrics: {args.checkpoint}", ""]
    lines.extend(f"- {name}: {value}" for name, value in metrics.items())
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote AOA visualizations to: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
