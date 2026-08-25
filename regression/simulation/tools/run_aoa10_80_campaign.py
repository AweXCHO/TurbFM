from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TMPDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "aoa_formula_campaign"
CAMPAIGN_DIR = ROOT.parent / "runs" / "outputs_aoa10_80_campaign"

EXPERIMENTS = [
    {
        "name": "baseline",
        "config": "configs/config_aoa10_80_baseline.yaml",
        "out": "../runs/outputs_aoa10_80_baseline",
    },
    {
        "name": "disp_jd",
        "config": "configs/config_aoa10_80_disp_jd.yaml",
        "out": "../runs/outputs_aoa10_80_disp_jd",
    },
    {
        "name": "jd_only",
        "config": "configs/config_aoa10_80_jd_only.yaml",
        "out": "../runs/outputs_aoa10_80_jd_only",
    },
]


def env() -> dict[str, str]:
    values = os.environ.copy()
    TMPDIR.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        values[key] = str(TMPDIR)
    return values


def latest_epoch(out_dir: Path) -> int:
    log = out_dir / "train_log.csv"
    if not log.exists():
        return 0
    with log.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]["epoch"]) if rows else 0


def baseline_is_running() -> bool:
    marker = "train.py --config configs/config_aoa10_80_baseline.yaml --out outputs_aoa10_80_baseline"
    result = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, check=True)
    return any(marker in line for line in result.stdout.splitlines())


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env(), check=True)


def train(experiment: dict[str, str]) -> None:
    out_dir = ROOT / experiment["out"]
    if latest_epoch(out_dir) >= 100:
        print(f"+ reuse completed training: {out_dir}", flush=True)
        return
    run(
        [
            str(PYTHON),
            "train.py",
            "--config",
            experiment["config"],
            "--out",
            experiment["out"],
            "--epochs",
            "100",
            "--batch-size",
            "8",
            "--num-workers",
            "4",
        ]
    )
    if latest_epoch(out_dir) < 100:
        raise RuntimeError(f"Training did not finish: {out_dir}")


def evaluate(experiment: dict[str, str]) -> dict[str, float | str]:
    out_dir = ROOT / experiment["out"]
    ckpt = out_dir / "checkpoints" / "best_composite.pt"
    eval_dir = out_dir / "checkpoint_symbolic_eval" / "best_composite"
    latent_table = eval_dir / "latent_tables" / "latent_table_all.csv"
    if not latent_table.exists():
        run(
            [
                str(PYTHON),
                "extract_latents.py",
                "--csv",
                "../data/sim_AOA_10_80_10/all_formula_table.csv",
                "--data-root",
                "../data/sim_AOA_10_80_10",
                "--ckpt",
                str(ckpt),
                "--out",
                str(eval_dir / "latent_tables"),
                "--split-info",
                str(out_dir / "split_info.json"),
                "--batch-size",
                "2",
                "--num-workers",
                "4",
            ]
        )
    result_csv = eval_dir / "symbolic" / "symbolic_results.csv"
    if not result_csv.exists():
        run([str(PYTHON), "evaluate.py", "--latent-table", str(latent_table), "--out", str(eval_dir)])
    with result_csv.open(newline="", encoding="utf-8") as f:
        rows = {row["name"]: row for row in csv.DictReader(f)}
    row = rows["latent_aoa_proxy_Jlatent"]
    return {
        "equation": row["equation"],
        "r2": float(row["r2"]),
        "exp_J": float(row["exp_J_latent"]),
        "exp_D": float(row["exp_D_aperture"]),
    }


def formula_is_good(metrics: dict[str, float | str]) -> bool:
    return (
        float(metrics["r2"]) >= 0.70
        and abs(float(metrics["exp_J"]) - 1.0) <= 0.15
        and abs(float(metrics["exp_D"]) + 1.0 / 3.0) <= 0.15
    )


def write_summary(rows: list[dict[str, float | str]]) -> None:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN_DIR / "campaign_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# AOA 10-80-10 Campaign", ""]
    for row in rows:
        lines.append(f"## {row['name']}")
        lines.append(f"- equation: `{row['equation']}`")
        lines.append(f"- R2: {float(row['r2']):.6f}")
        lines.append(f"- J exponent: {float(row['exp_J']):.6f}")
        lines.append(f"- D exponent: {float(row['exp_D']):.6f}")
        lines.append("")
    (CAMPAIGN_DIR / "campaign_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    CAMPAIGN_DIR.mkdir(parents=True, exist_ok=True)
    baseline = EXPERIMENTS[0]
    baseline_out = ROOT / baseline["out"]
    while latest_epoch(baseline_out) < 100 and baseline_is_running():
        print(f"+ waiting for baseline, epoch={latest_epoch(baseline_out)}", flush=True)
        time.sleep(60)
    train(baseline)

    results: list[dict[str, float | str]] = []
    baseline_metrics = {"name": baseline["name"], **evaluate(baseline)}
    results.append(baseline_metrics)
    write_summary(results)
    if formula_is_good(baseline_metrics):
        print("+ baseline meets campaign criterion", flush=True)
        return

    constrained = EXPERIMENTS[1]
    train(constrained)
    constrained_metrics = {"name": constrained["name"], **evaluate(constrained)}
    results.append(constrained_metrics)
    write_summary(results)
    if formula_is_good(constrained_metrics):
        print("+ disp_jd meets campaign criterion", flush=True)
        return

    ablation = EXPERIMENTS[2]
    train(ablation)
    ablation_metrics = {"name": ablation["name"], **evaluate(ablation)}
    results.append(ablation_metrics)
    write_summary(results)


if __name__ == "__main__":
    main()
