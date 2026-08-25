from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove optimizer state and export a publication checkpoint."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--kind", choices=("encoder", "simulation", "real"), required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config used to replace machine-specific paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = torch.load(args.input, map_location="cpu")
    config = None
    if args.config:
        with args.config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    if args.kind == "encoder":
        release = {
            "model": source["model"],
            "epoch": source.get("epoch"),
        }
    elif args.kind == "simulation":
        keys = (
            "model",
            "aoa_formula_loss",
            "disp_formula_loss",
            "epoch",
            "score",
            "mu_logJ",
            "std_logJ",
            "mu_logAOA",
            "std_logAOA",
            "c0_init",
            "c0_current",
            "disp_formula_C_current",
        )
        release = {key: source[key] for key in keys if key in source}
        release["config"] = config or source["config"]
    else:
        keys = (
            "model",
            "epoch",
            "score",
            "proxy_stats",
            "val_metrics",
            "initialization",
            "heat_chamber_included",
            "j_target_mode",
        )
        release = {key: source[key] for key in keys if key in source}
        release["config"] = config or source["config"]
        release["initialization"] = "encoder/83-checkpoint-499.pth"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release, args.output)
    print(f"Saved {args.kind} checkpoint to {args.output}")


if __name__ == "__main__":
    main()
