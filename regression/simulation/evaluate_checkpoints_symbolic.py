from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints by latent symbolic discovery.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--csv", default="../data/sim_AOA_10_80_10/all_formula_table.csv")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--python", default="python")
    parser.add_argument("--pattern", default="epoch_*.pt")
    return parser.parse_args()


def checkpoint_sort_key(path: Path):
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return (0, int(stem.split("_", 1)[1]))
        except ValueError:
            pass
    return (1, stem)


def run_cmd(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob(args.pattern), key=checkpoint_sort_key)
    for extra in ["best_latent_corr.pt", "best_composite.pt", "best_disp.pt", "best_formula.pt"]:
        path = ckpt_dir / extra
        if path.exists() and path not in checkpoints:
            checkpoints.append(path)

    rows = []
    eval_root = run_dir / "checkpoint_symbolic_eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    for ckpt in checkpoints:
        tag = ckpt.stem
        out_dir = eval_root / tag
        latent_dir = out_dir / "latent_tables"
        latent_table = latent_dir / "latent_table_all.csv"
        if latent_table.exists():
            print(f"+ reuse {latent_table}", flush=True)
        else:
            run_cmd(
                [
                    args.python,
                    "extract_latents.py",
                    "--csv",
                    args.csv,
                    "--ckpt",
                    str(ckpt),
                    "--out",
                    str(latent_dir),
                    "--split-info",
                    str(run_dir / "split_info.json"),
                    "--batch-size",
                    str(args.batch_size),
                    "--num-workers",
                    str(args.num_workers),
                ]
            )
        run_cmd(
            [
                args.python,
                "evaluate.py",
                "--latent-table",
                str(latent_table),
                "--out",
                str(out_dir),
            ]
        )
        result_csv = out_dir / "symbolic" / "symbolic_results.csv"
        result_df = pd.read_csv(result_csv)
        row = {"checkpoint": ckpt.name}
        for name in [
            "oracle_aoa_proxy",
            "oracle_aoa_proxy_from_AOA",
            "latent_aoa_proxy",
            "latent_aoa_proxy_Jlatent",
            "latent_aoa_proxy_AOAlatent",
            "latent_aoa_proxy_pred",
            "latent_aoa_proxy_pred_Jlatent",
            "latent_aoa_proxy_pred_AOAlatent",
            "latent_disp_AOAlatent",
            "latent_disp_AOA_extra",
            "latent_disp_full",
            "latent_aoa_check_Jlatent",
        ]:
            match = result_df[result_df["name"] == name]
            if not match.empty:
                row[f"{name}_r2"] = float(match.iloc[0]["r2"])
                row[f"{name}_equation"] = str(match.iloc[0]["equation"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(eval_root / "checkpoint_symbolic_summary.csv", index=False)
    sort_col = "latent_disp_AOA_extra_r2"
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col, ascending=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
