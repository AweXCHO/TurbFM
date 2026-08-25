from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_formula_table, load_or_create_split, split_frame
from src.report import write_summary
from src.symbolic import run_symbolic_suite


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate exported latent tables with symbolic regression.")
    parser.add_argument("--latent-table", default="outputs_latent_codex/latent_tables/latent_table_all.csv")
    parser.add_argument("--out", default="outputs_latent_codex")
    parser.add_argument("--run-pysr", action="store_true")
    parser.add_argument("--pysr-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    df = load_formula_table(args.latent_table)
    split = load_or_create_split(df, out / "split_info.json", random_state=args.random_state)
    results = run_symbolic_suite(
        split_frame(df, split, "train"),
        split_frame(df, split, "test"),
        out / "symbolic",
        run_pysr=args.run_pysr,
        pysr_iterations=args.pysr_iterations,
        random_state=args.random_state,
    )
    write_summary(out / "summary.md", df, split, results, out / "train_log.csv")


if __name__ == "__main__":
    main()
