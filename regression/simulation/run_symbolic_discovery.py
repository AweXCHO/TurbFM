from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import load_formula_table, load_or_create_split, split_frame
from src.report import write_summary
from src.symbolic import run_symbolic_suite


def parse_args():
    parser = argparse.ArgumentParser(description="Run power-law and optional PySR symbolic discovery.")
    parser.add_argument("--csv", default="../data/sim_AOA_10_80_10/all_formula_table.csv")
    parser.add_argument("--latent-table", default=None, help="Optional latent_table_all.csv to use instead of raw CSV.")
    parser.add_argument("--out", default="outputs_latent_codex")
    parser.add_argument("--run-pysr", action="store_true")
    parser.add_argument("--pysr-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    source = args.latent_table or args.csv
    df = load_formula_table(source)
    split = load_or_create_split(df, root / "split_info.json", random_state=args.random_state)
    train_df = split_frame(df, split, "train")
    test_df = split_frame(df, split, "test")
    results = run_symbolic_suite(
        train_df,
        test_df,
        root / "symbolic",
        run_pysr=args.run_pysr,
        pysr_iterations=args.pysr_iterations,
        random_state=args.random_state,
    )
    write_summary(root / "summary.md", df, split, results, root / "train_log.csv")
    print(pd.read_csv(root / "symbolic" / "symbolic_results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
