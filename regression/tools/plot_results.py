from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "results"
FIGURE_DIR = ROOT / "figure"
PROCESS_DIR = FIGURE_DIR / "use" / "process"
FONT_DIR = Path(os.environ.get("MSYH_FONT_DIR", PACKAGE_ROOT / "fonts"))
FONT_REGULAR = FONT_DIR / "MSYH.TTC"
FONT_BOLD = FONT_DIR / "MSYHBD.TTC"
PROCESS_MODE = False
OUTPUT_DIR = FIGURE_DIR
SAVE_PDF = True

COLORS = {
    "formula": "#1f77b4",
    "secondary": "#ff7f0e",
    "auxiliary": "#2ca02c",
    "model": "#d62728",
    "truth": "#b276b2",
}

SCIENCE_COLORS = {
    "navy": "#3C5488",
    "red": "#E64B35",
    "green": "#00A087",
    "cyan": "#4DBBD5",
}


def configure_style() -> tuple[font_manager.FontProperties, font_manager.FontProperties]:
    if FONT_REGULAR.exists() and FONT_BOLD.exists():
        for path in (FONT_REGULAR, FONT_BOLD):
            font_manager.fontManager.addfont(str(path))
        regular = font_manager.FontProperties(fname=str(FONT_REGULAR))
        bold = font_manager.FontProperties(fname=str(FONT_BOLD))
    else:
        warnings.warn(
            "Microsoft YaHei was not found. Set MSYH_FONT_DIR or place "
            "MSYH.TTC/MSYHBD.TTC in ./fonts; using the system sans-serif font."
        )
        regular = font_manager.FontProperties(family="sans-serif")
        bold = font_manager.FontProperties(family="sans-serif", weight="bold")
    regular_name = regular.get_name()
    bold_name = bold.get_name()
    plt.rcParams.update(
        {
            "font.family": regular_name,
            "font.sans-serif": [regular_name],
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.linewidth": 1.2,
            "axes.unicode_minus": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return (
        regular,
        bold,
    )


REGULAR_FONT, BOLD_FONT = configure_style()


def finite_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(target) & np.isfinite(prediction)
    return float(r2_score(target[valid], prediction[valid]))


def positive_log(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = np.log(values[valid])
    return result


def ordered_bin_median(
    order_value: np.ndarray,
    series: dict[str, np.ndarray],
    bins: int = 120,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    valid = np.isfinite(order_value)
    for values in series.values():
        valid &= np.isfinite(values)
    order_value = order_value[valid]
    filtered = {name: values[valid] for name, values in series.items()}
    order = np.argsort(order_value, kind="mergesort")
    chunks = [chunk for chunk in np.array_split(order, bins) if len(chunk)]
    x = np.linspace(0.0, 100.0, len(chunks))
    reduced = {
        name: np.asarray([np.median(values[chunk]) for chunk in chunks])
        for name, values in filtered.items()
    }
    return x, reduced


def style_axis(axis: plt.Axes) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_color("#4a4a4a")
        spine.set_linewidth(1.2)
    axis.tick_params(direction="out", width=1.1, length=4.5, color="#4a4a4a")
    if PROCESS_MODE:
        for axis_dimension in (axis.xaxis, axis.yaxis):
            if not isinstance(axis_dimension.get_major_locator(), FixedLocator):
                axis_dimension.set_major_locator(
                    MaxNLocator(nbins=6, integer=True, min_n_ticks=3)
                )
                axis_dimension.set_major_formatter(FormatStrFormatter("%.0f"))
        axis.set_xlabel("")
        axis.set_ylabel("")
    tick_font = REGULAR_FONT
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontproperties(tick_font)
        if PROCESS_MODE:
            label.set_fontsize(18)


def style_scatter_axis(axis: plt.Axes) -> None:
    style_axis(axis)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_legend(
    axis: plt.Axes,
    *,
    loc: str = "upper left",
    ncol: int = 1,
) -> None:
    if PROCESS_MODE:
        return
    legend = axis.legend(
        loc=loc,
        ncol=ncol,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#d0d0d0",
        handlelength=2.7,
        borderpad=0.55,
    )
    for text in legend.get_texts():
        text.set_fontproperties(REGULAR_FONT)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_identity_scatter(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    point_label: str,
    color: str,
    annotation: str | None = None,
    max_points: int = 12000,
) -> None:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    if not len(target):
        raise ValueError(f"No finite values available for {stem}")

    rng = np.random.default_rng(20260730)
    if len(target) > max_points:
        keep = rng.choice(len(target), size=max_points, replace=False)
        scatter_target = target[keep]
        scatter_prediction = prediction[keep]
    else:
        scatter_target = target
        scatter_prediction = prediction

    low = float(min(np.min(target), np.min(prediction)))
    high = float(max(np.max(target), np.max(prediction)))
    padding = max((high - low) * 0.045, 0.08)
    limits = (low - padding, high + padding)
    score = finite_r2(target, prediction)

    fig, axis = plt.subplots(figsize=(7.2, 6.45))
    axis.scatter(
        scatter_target,
        scatter_prediction,
        s=18,
        alpha=0.28,
        linewidths=0,
        color=color,
        label=point_label,
        rasterized=True,
    )
    axis.plot(
        limits,
        limits,
        color="#2b2b2b",
        linewidth=2.8,
        linestyle="--",
        label="理想一致线",
        zorder=4,
    )
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(xlabel, fontproperties=REGULAR_FONT)
    axis.set_ylabel(ylabel, fontproperties=REGULAR_FONT)
    axis.set_title(title, fontproperties=BOLD_FONT, pad=12)
    score_text = f"log-R² = {score:.3f}"
    if annotation:
        score_text = f"{annotation}\n{score_text}"
    axis.text(
        0.96,
        0.045,
        score_text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=BOLD_FONT,
        fontsize=13,
        color="#222222",
    )
    style_scatter_axis(axis)
    add_legend(axis)
    fig.tight_layout()
    save_figure(fig, stem)


def plot_simulation() -> None:
    base = (
        ROOT
        / "outputs_aoa10_80_baseline"
        / "checkpoint_symbolic_eval_corrected_root"
        / "best_composite"
    )
    latent = pd.read_csv(base / "latent_tables" / "latent_table_test.csv")
    symbolic = pd.read_csv(
        base
        / "symbolic"
        / "latent_aoa_check"
        / "powerlaw_test_predictions.csv"
    )
    if len(latent) != len(symbolic):
        raise RuntimeError("Simulation latent and symbolic tables have different lengths")
    target = np.log(symbolic["AOA_var"].to_numpy(np.float64))
    model = np.log(latent["aoa_hat"].to_numpy(np.float64))
    formula = np.log(symbolic["AOA_var_pred"].to_numpy(np.float64))
    x, curves = ordered_bin_median(
        target,
        {"truth": target, "model": model, "formula": formula},
    )

    fig, axis = plt.subplots(figsize=(7.6, 6.37))
    axis.plot(
        x,
        curves["formula"],
        color=COLORS["formula"],
        linewidth=5.2,
        label="发现公式",
    )
    axis.plot(
        x,
        curves["model"],
        color=COLORS["model"],
        linewidth=5.2,
        label="模型输出",
    )
    axis.plot(
        x,
        curves["truth"],
        color=COLORS["truth"],
        linewidth=5.2,
        label="真实值",
    )
    axis.set_xlabel("按真实 AOA 排序的测试样本 (%)", fontproperties=REGULAR_FONT)
    axis.set_ylabel("ln(AOA 方差)", fontproperties=REGULAR_FONT)
    axis.set_title("仿真数据：AOA 公式发现", fontproperties=BOLD_FONT, pad=12)
    formula_r2 = finite_r2(target, formula)
    model_r2 = finite_r2(target, model)
    axis.text(
        0.97,
        0.055,
        f"公式 log-R² = {formula_r2:.3f}\n模型 log-R² = {model_r2:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=BOLD_FONT,
        fontsize=13,
        color="#222222",
    )
    style_axis(axis)
    add_legend(axis)
    fig.tight_layout()
    save_figure(fig, "simulation_aoa_formula")


def plot_simulation_aoa_proxy() -> None:
    base, symbolic_root = simulation_paths()
    latent = pd.read_csv(base / "latent_tables" / "latent_table_test.csv")
    symbolic = pd.read_csv(
        symbolic_root / "latent_aoa_proxy" / "powerlaw_test_predictions.csv"
    )
    if len(latent) != len(symbolic):
        raise RuntimeError(
            "Simulation latent and AOA proxy tables have different lengths"
        )

    target = positive_log(symbolic["AOA_proxy"].to_numpy(np.float64))
    model_proxy = (
        latent["disp_hat"].to_numpy(np.float64)
        * latent["Delta_theta"].to_numpy(np.float64) ** 2
    )
    model = positive_log(model_proxy)
    formula = positive_log(symbolic["AOA_proxy_pred"].to_numpy(np.float64))
    x, curves = ordered_bin_median(
        target,
        {"truth": target, "model": model, "formula": formula},
    )

    fig, axis = plt.subplots(figsize=(7.6, 6.37))
    axis.plot(
        x,
        curves["formula"],
        color=COLORS["formula"],
        linewidth=5.2,
        label="发现公式",
    )
    axis.plot(
        x,
        curves["model"],
        color=COLORS["model"],
        linewidth=5.2,
        label="模型输出",
    )
    axis.plot(
        x,
        curves["truth"],
        color=COLORS["truth"],
        linewidth=5.2,
        label="真实值",
    )
    axis.set_xlabel(
        "按真实 AOA proxy 排序的测试样本 (%)", fontproperties=REGULAR_FONT
    )
    axis.set_ylabel("ln(AOA proxy)", fontproperties=REGULAR_FONT)
    axis.set_title(
        "仿真数据：AOA proxy 公式发现", fontproperties=BOLD_FONT, pad=12
    )
    formula_r2 = finite_r2(target, formula)
    model_r2 = finite_r2(target, model)
    axis.text(
        0.97,
        0.055,
        f"公式 log-R² = {formula_r2:.3f}\n模型 log-R² = {model_r2:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=BOLD_FONT,
        fontsize=13,
        color="#222222",
    )
    style_axis(axis)
    add_legend(axis)
    fig.tight_layout()
    save_figure(fig, "simulation_aoa_proxy_formula")


def simulation_paths() -> tuple[Path, Path]:
    base = (
        ROOT
        / "outputs_aoa10_80_baseline"
        / "checkpoint_symbolic_eval_corrected_root"
        / "best_composite"
    )
    return base, base / "symbolic"


def plot_simulation_scatters() -> None:
    base, symbolic_root = simulation_paths()
    latent = pd.read_csv(base / "latent_tables" / "latent_table_test.csv")
    aoa_formula = pd.read_csv(
        symbolic_root / "latent_aoa_check" / "powerlaw_test_predictions.csv"
    )
    proxy_formula = pd.read_csv(
        symbolic_root / "latent_aoa_proxy" / "powerlaw_test_predictions.csv"
    )
    disp_formula = pd.read_csv(
        symbolic_root / "latent_disp_AOAlatent" / "powerlaw_test_predictions.csv"
    )

    plot_identity_scatter(
        positive_log(latent["AOA_var"].to_numpy()),
        positive_log(latent["aoa_hat"].to_numpy()),
        title="仿真数据：模型 AOA 预测",
        xlabel="真实 ln(AOA 方差)",
        ylabel="模型预测 ln(AOA 方差)",
        stem="simulation_aoa_model_scatter",
        point_label="测试样本",
        color=COLORS["model"],
    )
    plot_identity_scatter(
        positive_log(aoa_formula["AOA_var"].to_numpy()),
        positive_log(aoa_formula["AOA_var_pred"].to_numpy()),
        title="仿真数据：AOA 发现公式",
        xlabel="真实 ln(AOA 方差)",
        ylabel="公式计算 ln(AOA 方差)",
        stem="simulation_aoa_formula_scatter",
        point_label="发现公式",
        color=COLORS["formula"],
        annotation="AOA = 2.838 · hJ^1.003 · D^-0.334",
    )
    plot_identity_scatter(
        positive_log(proxy_formula["AOA_proxy"].to_numpy()),
        positive_log(proxy_formula["AOA_proxy_pred"].to_numpy()),
        title="仿真数据：AOA proxy 发现公式",
        xlabel="真实 ln(AOA proxy)",
        ylabel="公式计算 ln(AOA proxy)",
        stem="simulation_aoa_proxy_formula_scatter",
        point_label="发现公式",
        color=COLORS["secondary"],
        annotation="AOA proxy = 5.286 · hJ^1.003 · D^-0.335",
    )
    plot_identity_scatter(
        positive_log(disp_formula["disp_var"].to_numpy()),
        positive_log(disp_formula["disp_var_pred"].to_numpy()),
        title="仿真数据：位移方差公式验证",
        xlabel="真实 ln(位移方差)",
        ylabel="公式计算 ln(位移方差)",
        stem="simulation_disp_formula_scatter",
        point_label="发现公式",
        color=COLORS["auxiliary"],
        annotation="disp = 2.581 · AOA latent^1.002 · Δθ^-1.946",
    )


def plot_simulation_exponents() -> None:
    _, symbolic_root = simulation_paths()
    results = pd.read_csv(symbolic_root / "symbolic_results.csv").set_index("name")
    labels = ["J\n(AOA)", "D\n(AOA)", "J\n(AOA proxy)", "D\n(AOA proxy)"]
    theory = np.array([1.0, -1.0 / 3.0, 1.0, -1.0 / 3.0])
    oracle = np.array(
        [
            results.loc["oracle_aoa", "exp_J"],
            results.loc["oracle_aoa", "exp_D_aperture"],
            results.loc["oracle_aoa_proxy", "exp_J"],
            results.loc["oracle_aoa_proxy", "exp_D_aperture"],
        ]
    )
    latent = np.array(
        [
            results.loc["latent_aoa_check", "exp_hJ"],
            results.loc["latent_aoa_check", "exp_D_aperture"],
            results.loc["latent_aoa_proxy", "exp_hJ"],
            results.loc["latent_aoa_proxy", "exp_D_aperture"],
        ]
    )
    x = np.arange(len(labels), dtype=np.float64)

    fig, axis = plt.subplots(figsize=(7.6, 6.2))
    axis.plot(
        x,
        theory,
        color=COLORS["truth"],
        linewidth=4.8,
        marker="o",
        markersize=9,
        label="理论指数",
    )
    axis.plot(
        x,
        oracle,
        color=COLORS["secondary"],
        linewidth=4.2,
        marker="s",
        markersize=8,
        label="真实物理量拟合",
    )
    axis.plot(
        x,
        latent,
        color=COLORS["formula"],
        linewidth=4.2,
        marker="D",
        markersize=8,
        label="latent 公式",
    )
    for index, value in enumerate(latent):
        axis.text(
            index,
            value + (0.085 if value >= 0 else -0.115),
            f"{value:.3f}",
            ha="center",
            va="center",
            fontproperties=BOLD_FONT,
            fontsize=12,
            color=COLORS["formula"],
        )
    axis.axhline(0, color="#4a4a4a", linewidth=1.2)
    axis.set_xticks(x, labels)
    axis.set_ylim(-0.58, 1.40)
    axis.set_ylabel("幂律指数", fontproperties=REGULAR_FONT)
    axis.set_title("仿真数据：物理指数恢复", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    add_legend(axis, loc="upper center", ncol=3)
    fig.tight_layout()
    save_figure(fig, "simulation_exponent_recovery")


def plot_simulation_ablation() -> None:
    experiments = [
        ("Baseline", ROOT / "outputs_aoa10_80_baseline"),
        ("Disp + J-D", ROOT / "outputs_aoa10_80_disp_jd"),
        ("J-D only", ROOT / "outputs_aoa10_80_jd_only"),
    ]
    scores = []
    for _, experiment in experiments:
        prediction = pd.read_csv(
            experiment
            / "checkpoint_symbolic_eval_corrected_root"
            / "best_composite"
            / "symbolic"
            / "latent_aoa_proxy"
            / "powerlaw_test_predictions.csv"
        )
        scores.append(
            finite_r2(
                positive_log(prediction["AOA_proxy"].to_numpy()),
                positive_log(prediction["AOA_proxy_pred"].to_numpy()),
            )
        )

    labels = [item[0] for item in experiments]
    colors = [COLORS["formula"], COLORS["secondary"], COLORS["auxiliary"]]
    fig, axis = plt.subplots(figsize=(7.25, 6.1))
    bars = axis.bar(
        labels,
        scores,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, value in zip(bars, scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=BOLD_FONT,
            fontsize=13,
        )
    axis.set_ylim(0.78, 0.92)
    axis.set_ylabel("AOA proxy 公式 log-R²", fontproperties=REGULAR_FONT)
    axis.set_title("仿真数据：物理约束消融", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "simulation_constraint_ablation")


def plot_simulation_training() -> None:
    history = pd.read_csv(ROOT / "outputs_aoa10_80_baseline" / "train_log.csv")
    fig, axis = plt.subplots(figsize=(7.6, 6.2))
    axis.plot(
        history["epoch"],
        history["corr_logJlatent_logJ"],
        color=COLORS["formula"],
        linewidth=4.2,
        label="corr(hJ, J)",
    )
    axis.plot(
        history["epoch"],
        history["corr_logAOAlatent_logAOA"],
        color=COLORS["model"],
        linewidth=4.2,
        label="corr(hAOA, AOA)",
    )
    axis.plot(
        history["epoch"],
        history["score_composite"],
        color=COLORS["truth"],
        linewidth=4.2,
        label="综合得分",
    )
    axis.set_xlabel("训练轮次", fontproperties=REGULAR_FONT)
    axis.set_ylabel("验证集指标", fontproperties=REGULAR_FONT)
    axis.set_title("仿真数据：训练过程", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    add_legend(axis)
    fig.tight_layout()
    save_figure(fig, "simulation_training_curves")


def load_real_results() -> tuple[Path, dict]:
    base = ROOT / "outputs_real_long_jcal_83init" / "aoa_j_formula_discovery_grouped"
    results = json.loads((base / "formula_results.json").read_text(encoding="utf-8"))
    return base, results


def plot_long_range() -> None:
    base, results = load_real_results()
    table = pd.read_csv(base / "test_predictions_turbulence_sequences.csv")
    result = results["turbulence_sequences"]["measured_proxy"]["pooled"]
    j_value = table["j_target_strength_proxy"].to_numpy(np.float64)
    truth = table["aoa_target_log_proxy"].to_numpy(np.float64)
    model = table["aoa_prediction_log_proxy"].to_numpy(np.float64)
    formula = math.log(result["constant"]) + result["p"] * np.log(j_value)
    order_value = np.log(j_value)
    x, curves = ordered_bin_median(
        order_value,
        {"truth": truth, "model": model, "formula": formula},
    )

    fig, axis = plt.subplots(figsize=(7.6, 6.37))
    axis.plot(
        x,
        curves["formula"],
        color=COLORS["formula"],
        linewidth=5.2,
        label="发现公式",
    )
    axis.plot(
        x,
        curves["model"],
        color=COLORS["model"],
        linewidth=5.2,
        label="模型输出",
    )
    axis.plot(
        x,
        curves["truth"],
        color=COLORS["truth"],
        linewidth=5.2,
        label="代理真值",
    )
    axis.set_xlabel("按 J 代理排序的测试样本 (%)", fontproperties=REGULAR_FONT)
    axis.set_ylabel("ln(AOA 代理)", fontproperties=REGULAR_FONT)
    axis.set_title("真实长焦数据：AOA–J 关系", fontproperties=BOLD_FONT, pad=12)
    axis.text(
        0.97,
        0.055,
        f"AOA = {result['constant']:.3f} · J^{result['p']:.3f}\n"
        f"公式 log-R² = {result['test']['log_r2']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=BOLD_FONT,
        fontsize=13,
        color="#222222",
    )
    style_axis(axis)
    add_legend(axis)
    fig.tight_layout()
    save_figure(fig, "real_long_range_aoa_j_formula")


def plot_rlrat() -> None:
    base, results = load_real_results()
    table = pd.read_csv(base / "test_predictions_turbulence_sequences_RLRAT.csv")
    constants = pd.read_csv(base / "rlrat_scene_constants_measured_proxy.csv")
    constants = constants.set_index("scene_group")["constant"]
    result = results["turbulence_sequences_RLRAT"]["measured_proxy"]["grouped"]
    table["constant"] = table["scene_group"].map(constants)
    table = table.loc[
        table["constant"].notna() & (table["j_target_strength_proxy"] > 0)
    ].copy()
    j_value = table["j_target_strength_proxy"].to_numpy(np.float64)
    truth = table["aoa_target_log_proxy"].to_numpy(np.float64)
    model = table["aoa_prediction_log_proxy"].to_numpy(np.float64)
    formula = (
        np.log(table["constant"].to_numpy(np.float64))
        + result["p"] * np.log(j_value)
    )
    x, curves = ordered_bin_median(
        formula,
        {"truth": truth, "model": model, "formula": formula},
    )

    fig, axis = plt.subplots(figsize=(7.6, 6.37))
    axis.plot(
        x,
        curves["formula"],
        color=COLORS["formula"],
        linewidth=5.2,
        label="分场景公式",
    )
    axis.plot(
        x,
        curves["model"],
        color=COLORS["model"],
        linewidth=5.2,
        label="模型输出",
    )
    axis.plot(
        x,
        curves["truth"],
        color=COLORS["truth"],
        linewidth=5.2,
        label="代理真值",
    )
    axis.set_xlabel("按分场景公式排序的测试样本 (%)", fontproperties=REGULAR_FONT)
    axis.set_ylabel("ln(AOA 代理)", fontproperties=REGULAR_FONT)
    axis.set_title("RLRAT 数据：分场景 AOA–J 关系", fontproperties=BOLD_FONT, pad=12)
    axis.text(
        0.97,
        0.055,
        f"AOA = C_g · J^{result['p']:.3f}\n"
        f"公式 log-R² = {result['test']['log_r2']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontproperties=BOLD_FONT,
        fontsize=13,
        color="#222222",
    )
    style_axis(axis)
    add_legend(axis)
    fig.tight_layout()
    save_figure(fig, "real_rlrat_grouped_aoa_j_formula")


def real_formula_prediction(
    table: pd.DataFrame,
    result: dict,
    constants: pd.Series | None = None,
) -> np.ndarray:
    j_value = table["j_target_strength_proxy"].to_numpy(np.float64)
    if constants is None:
        return math.log(result["constant"]) + result["p"] * np.log(j_value)
    mapped = table["scene_group"].map(constants).to_numpy(np.float64)
    return np.log(mapped) + result["p"] * np.log(j_value)


def bootstrap_pooled_exponent(
    table: pd.DataFrame,
    *,
    repeats: int = 300,
    seed: int = 20260724,
) -> tuple[float, list[float], np.ndarray]:
    j_value = table["j_target_strength_proxy"].to_numpy(np.float64)
    y_value = table["aoa_target_log_proxy"].to_numpy(np.float64)
    valid = np.isfinite(j_value) & np.isfinite(y_value) & (j_value > 1e-8)
    x_value = np.log(j_value[valid])
    y_value = y_value[valid]

    def fit_exponent(x_sample: np.ndarray, y_sample: np.ndarray) -> float:
        centered_x = x_sample - x_sample.mean()
        denominator = float(np.dot(centered_x, centered_x))
        if denominator < 1e-12:
            return float("nan")
        return float(
            np.dot(centered_x, y_sample - y_sample.mean()) / denominator
        )

    exponent = fit_exponent(x_value, y_value)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            fit_exponent(x_value[index], y_value[index])
            for index in (
                rng.integers(0, len(x_value), len(x_value))
                for _ in range(repeats)
            )
        ],
        dtype=np.float64,
    )
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    ci = [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])]
    return exponent, ci, bootstrap


def plot_real_scatters() -> None:
    base, results = load_real_results()
    long_table = pd.read_csv(base / "test_predictions_turbulence_sequences.csv")
    long_result = results["turbulence_sequences"]["measured_proxy"]["pooled"]
    long_formula = real_formula_prediction(long_table, long_result)
    plot_identity_scatter(
        long_table["aoa_target_log_proxy"].to_numpy(),
        long_table["aoa_prediction_log_proxy"].to_numpy(),
        title="真实长焦数据：模型 AOA 预测",
        xlabel="代理真值 ln(AOA)",
        ylabel="模型输出 ln(AOA)",
        stem="real_long_range_model_scatter",
        point_label="测试序列",
        color=COLORS["model"],
    )
    plot_identity_scatter(
        long_table["aoa_target_log_proxy"].to_numpy(),
        long_formula,
        title="真实长焦数据：AOA–J 公式验证",
        xlabel="代理真值 ln(AOA)",
        ylabel="公式计算 ln(AOA)",
        stem="real_long_range_formula_scatter",
        point_label="发现公式",
        color=COLORS["formula"],
        annotation=(
            f"AOA proxy = {long_result['constant']:.3f} · "
            f"J proxy^{long_result['p']:.3f}"
        ),
    )

    rlrat_table = pd.read_csv(
        base / "test_predictions_turbulence_sequences_RLRAT.csv"
    )
    constants = pd.read_csv(base / "rlrat_scene_constants_measured_proxy.csv")
    constants = constants.set_index("scene_group")["constant"]
    rlrat_result = results["turbulence_sequences_RLRAT"]["measured_proxy"]["grouped"]
    valid = (
        rlrat_table["scene_group"].isin(constants.index)
        & (rlrat_table["j_target_strength_proxy"] > 0)
    )
    rlrat_known = rlrat_table.loc[valid].copy()
    rlrat_formula = real_formula_prediction(rlrat_known, rlrat_result, constants)
    plot_identity_scatter(
        rlrat_table["aoa_target_log_proxy"].to_numpy(),
        rlrat_table["aoa_prediction_log_proxy"].to_numpy(),
        title="RLRAT 数据：模型 AOA 预测",
        xlabel="代理真值 ln(AOA)",
        ylabel="模型输出 ln(AOA)",
        stem="real_rlrat_model_scatter",
        point_label="测试序列",
        color=COLORS["model"],
    )
    plot_identity_scatter(
        rlrat_known["aoa_target_log_proxy"].to_numpy(),
        rlrat_formula,
        title="RLRAT 数据：分场景公式验证",
        xlabel="代理真值 ln(AOA)",
        ylabel="分场景公式 ln(AOA)",
        stem="real_rlrat_grouped_formula_scatter",
        point_label="分场景公式",
        color=COLORS["formula"],
        annotation=f"AOA proxy = Cg · J proxy^{rlrat_result['p']:.3f}",
    )


def plot_real_exponents() -> None:
    base, results = load_real_results()
    long_result = results["turbulence_sequences"]["measured_proxy"]["pooled"]
    rlrat_result = results["turbulence_sequences_RLRAT"]["measured_proxy"]["grouped"]
    long_calibration = pd.read_csv(
        base / "val_predictions_turbulence_sequences.csv"
    )
    _, long_ci, _ = bootstrap_pooled_exponent(long_calibration)
    labels = ["长焦数据", "RLRAT 分场景"]
    values = np.array([long_result["p"], rlrat_result["p"]])
    lower = np.array([long_ci[0], rlrat_result["p_scene_bootstrap_ci95"][0]])
    upper = np.array([long_ci[1], rlrat_result["p_scene_bootstrap_ci95"][1]])
    yerr = np.vstack([values - lower, upper - values])
    x = np.arange(2)

    fig, axis = plt.subplots(figsize=(7.2, 6.1))
    point_colors = [SCIENCE_COLORS["navy"], SCIENCE_COLORS["green"]]
    for index, (value, color) in enumerate(zip(values, point_colors)):
        axis.errorbar(
            index,
            value,
            yerr=yerr[:, index : index + 1],
            fmt="o",
            markersize=13,
            linewidth=3.2,
            capsize=8,
            capthick=2.5,
            color=color,
            zorder=3,
        )
    for index, value in enumerate(values):
        axis.text(
            index,
            value + 0.055,
            f"p = {value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=BOLD_FONT,
            fontsize=13,
        )
    axis.set_xticks(x, labels)
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(1.15, 1.48)
    axis.set_ylabel("AOA–J 幂律指数 p", fontproperties=REGULAR_FONT)
    axis.set_title("真实数据：AOA–J 指数发现", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "real_aoa_j_exponent_comparison")


def plot_rlrat_grouping_ablation() -> None:
    _, results = load_real_results()
    result = results["turbulence_sequences_RLRAT"]["measured_proxy"]
    labels = ["整体常数", "分场景常数", "分场景指数"]
    scores = [
        result["pooled"]["test"]["log_r2"],
        result["grouped"]["test"]["log_r2"],
        result["grouped"]["independent_scene_exponent_test"]["log_r2"],
    ]
    colors = [COLORS["secondary"], COLORS["formula"], COLORS["auxiliary"]]
    fig, axis = plt.subplots(figsize=(7.25, 6.1))
    bars = axis.bar(
        labels,
        scores,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, value in zip(bars, scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=BOLD_FONT,
            fontsize=13,
        )
    axis.set_ylim(0.5, 0.76)
    axis.set_ylabel("测试集 log-R²", fontproperties=REGULAR_FONT)
    axis.set_title("RLRAT 数据：场景建模消融", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "real_rlrat_scene_grouping_ablation")


def plot_real_prediction_summary() -> None:
    metrics_path = ROOT / "outputs_real_long_jcal_83init" / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = ["长焦 AOA", "长焦 J", "RLRAT AOA", "RLRAT J"]
    values = [
        metrics["turbulence_sequences"]["clip"]["aoa"]["log_r2"],
        metrics["turbulence_sequences"]["clip"]["j"]["log_r2"],
        metrics["turbulence_sequences_RLRAT"]["clip"]["aoa"]["log_r2"],
        metrics["turbulence_sequences_RLRAT"]["clip"]["j"]["log_r2"],
    ]
    colors = [
        SCIENCE_COLORS["navy"],
        SCIENCE_COLORS["cyan"],
        SCIENCE_COLORS["green"],
        SCIENCE_COLORS["red"],
    ]
    fig, axis = plt.subplots(figsize=(7.6, 6.1))
    bars = axis.bar(
        labels,
        values,
        width=0.64,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=BOLD_FONT,
            fontsize=12.5,
        )
    axis.set_ylim(0.7, 1.0)
    axis.set_ylabel("序列级 log-R²", fontproperties=REGULAR_FONT)
    axis.set_title("真实数据：物理代理预测性能", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "real_model_prediction_summary")


def plot_real_combined_summary() -> None:
    base, results = load_real_results()
    metrics_path = ROOT / "outputs_real_long_jcal_83init" / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    group_labels = ["长焦", "RLRAT"]
    aoa_values = [
        metrics["turbulence_sequences"]["clip"]["aoa"]["log_r2"],
        metrics["turbulence_sequences_RLRAT"]["clip"]["aoa"]["log_r2"],
    ]
    j_values = [
        metrics["turbulence_sequences"]["clip"]["j"]["log_r2"],
        metrics["turbulence_sequences_RLRAT"]["clip"]["j"]["log_r2"],
    ]

    long_result = results["turbulence_sequences"]["measured_proxy"]["pooled"]
    rlrat_result = results["turbulence_sequences_RLRAT"]["measured_proxy"]["grouped"]
    long_calibration = pd.read_csv(
        base / "val_predictions_turbulence_sequences.csv"
    )
    long_bootstrap_p, long_ci, long_bootstrap = bootstrap_pooled_exponent(
        long_calibration
    )
    exponent_values = np.array([long_result["p"], rlrat_result["p"]])
    exponent_errors = np.array(
        [
            [
                long_result["p"] - long_ci[0],
                rlrat_result["p"] - rlrat_result["p_scene_bootstrap_ci95"][0],
            ],
            [
                long_ci[1] - long_result["p"],
                rlrat_result["p_scene_bootstrap_ci95"][1] - rlrat_result["p"],
            ],
        ]
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bootstrap_exponent_p": long_bootstrap}).to_csv(
        FIGURE_DIR / "real_long_range_p_bootstrap.csv",
        index=False,
    )
    bootstrap_summary = {
        "method": "sequence-level nonparametric bootstrap on validation clips",
        "repeats": 300,
        "seed": 20260724,
        "n_sequences": int(len(long_calibration)),
        "fitted_p": float(long_bootstrap_p),
        "p_ci95": long_ci,
    }
    (FIGURE_DIR / "real_long_range_p_bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    x = np.arange(len(group_labels), dtype=np.float64)
    bar_width = 0.34
    fig, bar_axis = plt.subplots(figsize=(8.4, 6.4))
    exponent_axis = bar_axis.twinx()

    aoa_bars = bar_axis.bar(
        x - bar_width / 2,
        aoa_values,
        width=bar_width,
        color=SCIENCE_COLORS["navy"],
        edgecolor="white",
        linewidth=1.2,
        label="AOA",
        zorder=2,
    )
    j_bars = bar_axis.bar(
        x + bar_width / 2,
        j_values,
        width=bar_width,
        color=SCIENCE_COLORS["cyan"],
        edgecolor="white",
        linewidth=1.2,
        label="J",
        zorder=2,
    )
    for bar, value in zip(
        list(aoa_bars) + list(j_bars),
        aoa_values + j_values,
    ):
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.007,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=BOLD_FONT,
            fontsize=12,
            zorder=6,
        )

    exponent_line = exponent_axis.plot(
        x,
        exponent_values,
        color=SCIENCE_COLORS["red"],
        linestyle="none",
        marker="D",
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2.5,
        label="幂律指数 p",
        zorder=7,
    )[0]
    exponent_axis.errorbar(
        x,
        exponent_values,
        yerr=exponent_errors,
        fmt="none",
        color=SCIENCE_COLORS["red"],
        linewidth=3.0,
        capsize=8,
        capthick=2.4,
        zorder=6,
    )
    for index, value in enumerate(exponent_values):
        exponent_axis.annotate(
            f"p = {value:.3f}",
            (x[index], value),
            xytext=(0, 13) if index == 0 else (18, -20),
            textcoords="offset points",
            ha="center" if index == 0 else "left",
            va="bottom" if index == 0 else "top",
            fontproperties=BOLD_FONT,
            fontsize=12,
            color=SCIENCE_COLORS["red"],
            zorder=8,
        )

    bar_axis.set_xticks(x, group_labels)
    bar_axis.set_xlim(-0.58, 1.58)
    bar_axis.set_ylim(0.70, 1.00)
    bar_axis.set_yticks([0.70, 0.80, 0.90, 1.00])
    bar_axis.set_ylabel("序列级 log-R²", fontproperties=REGULAR_FONT)
    bar_axis.set_xlabel("真实数据集", fontproperties=REGULAR_FONT)
    bar_axis.set_title(
        "真实数据：物理代理预测与公式发现",
        fontproperties=BOLD_FONT,
        fontsize=20,
        pad=12 if PROCESS_MODE else 58,
    )
    style_axis(bar_axis)

    exponent_axis.set_ylim(1.15, 1.48)
    exponent_axis.set_yticks([1.15, 1.25, 1.35, 1.45])
    exponent_axis.set_ylabel("AOA–J 幂律指数 p", fontproperties=REGULAR_FONT)
    exponent_axis.patch.set_visible(False)
    exponent_axis.spines["left"].set_visible(False)
    exponent_axis.spines["bottom"].set_visible(False)
    exponent_axis.spines["top"].set_visible(False)
    exponent_axis.spines["right"].set_color("#222222")
    exponent_axis.spines["right"].set_linewidth(1.4)
    exponent_axis.tick_params(
        axis="y",
        direction="out",
        width=1.1,
        length=4.5,
        colors="#222222",
    )
    exponent_axis.yaxis.label.set_color("#222222")
    for label in exponent_axis.get_yticklabels():
        label.set_fontproperties(REGULAR_FONT)

    if not PROCESS_MODE:
        legend = bar_axis.legend(
            [aoa_bars[0], j_bars[0], exponent_line],
            ["AOA log-R²", "J log-R²", "幂律指数 p"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=3,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#d0d0d0",
            borderpad=0.5,
            handlelength=2.2,
        )
        for text_item in legend.get_texts():
            text_item.set_fontproperties(REGULAR_FONT)

    fig.subplots_adjust(
        top=0.88 if PROCESS_MODE else 0.80,
        bottom=0.14,
        left=0.13,
        right=0.87,
    )
    save_figure(fig, "real_prediction_exponent_combined")


def plot_sequence_level_log_r2() -> None:
    metrics_path = ROOT / "outputs_real_long_jcal_83init" / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    labels = ["长焦", "RLRAT"]
    aoa_values = [
        metrics["turbulence_sequences"]["clip"]["aoa"]["log_r2"],
        metrics["turbulence_sequences_RLRAT"]["clip"]["aoa"]["log_r2"],
    ]
    j_values = [
        metrics["turbulence_sequences"]["clip"]["j"]["log_r2"],
        metrics["turbulence_sequences_RLRAT"]["clip"]["j"]["log_r2"],
    ]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.34

    fig, axis = plt.subplots(figsize=(7.6, 6.1))
    aoa_bars = axis.bar(
        x - width / 2,
        aoa_values,
        width=width,
        color=SCIENCE_COLORS["navy"],
        edgecolor="white",
        linewidth=1.2,
    )
    j_bars = axis.bar(
        x + width / 2,
        j_values,
        width=width,
        color=SCIENCE_COLORS["cyan"],
        edgecolor="white",
        linewidth=1.2,
    )
    for name, bars, values in (
        ("AOA", aoa_bars, aoa_values),
        ("J", j_bars, j_values),
    ):
        for bar, value in zip(bars, values):
            center = bar.get_x() + bar.get_width() / 2
            axis.text(
                center,
                value + 0.007,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontproperties=BOLD_FONT,
                fontsize=13,
            )
            axis.text(
                center,
                0.715,
                name,
                ha="center",
                va="bottom",
                fontproperties=BOLD_FONT,
                fontsize=13,
                color="white",
            )

    axis.set_xticks(x, labels)
    axis.set_xlim(-0.58, 1.58)
    axis.set_ylim(0.70, 1.00)
    axis.set_yticks([0.70, 0.80, 0.90, 1.00])
    axis.set_title("Sequence-level log-R²", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "sequence_level_log_r2")


def plot_power_law_exponent_p() -> None:
    base, results = load_real_results()
    long_result = results["turbulence_sequences"]["measured_proxy"]["pooled"]
    rlrat_result = results["turbulence_sequences_RLRAT"]["measured_proxy"][
        "grouped"
    ]
    long_calibration = pd.read_csv(
        base / "val_predictions_turbulence_sequences.csv"
    )
    _, long_ci, _ = bootstrap_pooled_exponent(long_calibration)

    labels = ["长焦", "RLRAT"]
    values = np.array([long_result["p"], rlrat_result["p"]])
    lower = np.array(
        [long_ci[0], rlrat_result["p_scene_bootstrap_ci95"][0]]
    )
    upper = np.array(
        [long_ci[1], rlrat_result["p_scene_bootstrap_ci95"][1]]
    )
    errors = np.vstack([values - lower, upper - values])
    x = np.arange(len(labels), dtype=np.float64)

    fig, axis = plt.subplots(figsize=(7.2, 6.1))
    axis.errorbar(
        x,
        values,
        yerr=errors,
        fmt="D",
        markersize=11,
        markerfacecolor="white",
        markeredgewidth=2.7,
        linewidth=3.2,
        capsize=9,
        capthick=2.5,
        color=SCIENCE_COLORS["red"],
        zorder=3,
    )
    for index, value in enumerate(values):
        axis.text(
            index,
            value + 0.015,
            f"p = {value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=BOLD_FONT,
            fontsize=13,
            color=SCIENCE_COLORS["red"],
        )

    axis.set_xticks(x, labels)
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(1.15, 1.48)
    axis.set_yticks([1.15, 1.25, 1.35, 1.45])
    axis.set_title("Power-law exponent p", fontproperties=BOLD_FONT, pad=12)
    style_axis(axis)
    fig.tight_layout()
    save_figure(fig, "power_law_exponent_p")


def plot_process_subset() -> None:
    plot_simulation()
    plot_simulation_aoa_proxy()
    _, symbolic_root = simulation_paths()
    simulation_formula = pd.read_csv(
        symbolic_root / "latent_aoa_check" / "powerlaw_test_predictions.csv"
    )
    plot_identity_scatter(
        positive_log(simulation_formula["AOA_var"].to_numpy()),
        positive_log(simulation_formula["AOA_var_pred"].to_numpy()),
        title="仿真数据：AOA 发现公式",
        xlabel="真实 ln(AOA 方差)",
        ylabel="公式计算 ln(AOA 方差)",
        stem="simulation_aoa_formula_scatter",
        point_label="发现公式",
        color=COLORS["formula"],
        annotation="AOA = 2.838 · hJ^1.003 · D^-0.334",
    )

    plot_long_range()
    plot_rlrat()
    real_base, results = load_real_results()
    long_table = pd.read_csv(
        real_base / "test_predictions_turbulence_sequences.csv"
    )
    long_result = results["turbulence_sequences"]["measured_proxy"]["pooled"]
    plot_identity_scatter(
        long_table["aoa_target_log_proxy"].to_numpy(),
        real_formula_prediction(long_table, long_result),
        title="真实长焦数据：AOA–J 公式验证",
        xlabel="代理真值 ln(AOA)",
        ylabel="公式计算 ln(AOA)",
        stem="real_long_range_formula_scatter",
        point_label="发现公式",
        color=COLORS["formula"],
        annotation=(
            f"AOA proxy = {long_result['constant']:.3f} · "
            f"J proxy^{long_result['p']:.3f}"
        ),
    )

    rlrat_table = pd.read_csv(
        real_base / "test_predictions_turbulence_sequences_RLRAT.csv"
    )
    constants = pd.read_csv(
        real_base / "rlrat_scene_constants_measured_proxy.csv"
    ).set_index("scene_group")["constant"]
    rlrat_result = results["turbulence_sequences_RLRAT"]["measured_proxy"][
        "grouped"
    ]
    valid = (
        rlrat_table["scene_group"].isin(constants.index)
        & (rlrat_table["j_target_strength_proxy"] > 0)
    )
    rlrat_known = rlrat_table.loc[valid].copy()
    plot_identity_scatter(
        rlrat_known["aoa_target_log_proxy"].to_numpy(),
        real_formula_prediction(rlrat_known, rlrat_result, constants),
        title="RLRAT 数据：分场景公式验证",
        xlabel="代理真值 ln(AOA)",
        ylabel="分场景公式 ln(AOA)",
        stem="real_rlrat_grouped_formula_scatter",
        point_label="分场景公式",
        color=COLORS["formula"],
        annotation=f"AOA proxy = Cg · J proxy^{rlrat_result['p']:.3f}",
    )
    plot_sequence_level_log_r2()
    plot_power_law_exponent_p()


def write_figure_index() -> None:
    entries = [
        ("simulation_aoa_formula", "仿真 AOA 真实值、模型输出与发现公式的排序曲线"),
        (
            "simulation_aoa_proxy_formula",
            "仿真 AOA proxy 真实值、模型输出与发现公式的排序曲线",
        ),
        ("simulation_aoa_model_scatter", "仿真 AOA 模型输出与真实值散点"),
        ("simulation_aoa_formula_scatter", "仿真 AOA 公式计算值与真实值散点"),
        ("simulation_aoa_proxy_formula_scatter", "仿真 AOA proxy 公式验证散点"),
        ("simulation_disp_formula_scatter", "仿真位移方差公式验证散点"),
        ("simulation_exponent_recovery", "J 与孔径 D 的理论、oracle 和 latent 指数对比"),
        ("simulation_constraint_ablation", "Baseline、Disp + J-D、J-D only 消融"),
        ("simulation_training_curves", "仿真训练过程中的 latent 相关性与综合得分"),
        ("real_long_range_aoa_j_formula", "长焦数据 AOA–J 排序曲线"),
        ("real_long_range_model_scatter", "长焦数据模型 AOA 输出散点"),
        ("real_long_range_formula_scatter", "长焦数据 AOA–J 发现公式散点"),
        ("real_rlrat_grouped_aoa_j_formula", "RLRAT 分场景 AOA–J 排序曲线"),
        ("real_rlrat_model_scatter", "RLRAT 模型 AOA 输出散点"),
        ("real_rlrat_grouped_formula_scatter", "RLRAT 分场景公式散点"),
        ("real_aoa_j_exponent_comparison", "两类真实数据的 AOA–J 指数对比"),
        ("real_rlrat_scene_grouping_ablation", "RLRAT 场景常数建模消融"),
        ("real_model_prediction_summary", "两类真实数据的序列级 AOA/J 预测性能"),
        (
            "real_prediction_exponent_combined",
            "真实数据序列级预测性能与 AOA–J 指数的合并图",
        ),
    ]
    lines = [
        "# 图件索引",
        "",
        "全部图件使用微软雅黑、白色背景和与 `use/image.png` 一致的粗线五色体系。",
        "每张图均提供 600 dpi PNG 和矢量 PDF。",
        "",
        "| 文件名 | 内容 |",
        "| --- | --- |",
    ]
    lines.extend(f"| `{stem}.png` | {description} |" for stem, description in entries)
    lines.append("")
    (FIGURE_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR, PROCESS_MODE, SAVE_PDF
    parser = argparse.ArgumentParser(description="Plot simulation and real results")
    parser.add_argument(
        "--process",
        action="store_true",
        help="Render the selected process figures with integer bold ticks and no legends",
    )
    args = parser.parse_args()
    if args.process:
        PROCESS_MODE = True
        OUTPUT_DIR = PROCESS_DIR
        SAVE_PDF = False
        plot_process_subset()
        print(f"Saved process figures to {OUTPUT_DIR}")
        return

    plot_simulation()
    plot_simulation_aoa_proxy()
    plot_simulation_scatters()
    plot_simulation_exponents()
    plot_simulation_ablation()
    plot_simulation_training()
    plot_long_range()
    plot_rlrat()
    plot_real_scatters()
    plot_real_exponents()
    plot_rlrat_grouping_ablation()
    plot_real_prediction_summary()
    plot_real_combined_summary()
    write_figure_index()
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
