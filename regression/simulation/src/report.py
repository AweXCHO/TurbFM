from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .data import describe_dataset


def write_summary(
    out_path: str | Path,
    df: pd.DataFrame,
    split: Dict[str, list],
    symbolic_results: Iterable,
    train_log_path: str | Path | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    desc = describe_dataset(df)
    lines = [
        "# Latent Deflex AOA Constraint and Displacement Discovery",
        "",
        "## Data",
        f"- sample_id count: {desc['sample_ids']}",
        f"- video-view samples: {desc['video_views']}",
        f"- patch rows: {desc['patch_rows']}",
        f"- train/val/test sample_id split: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
        "",
        "## Notes",
        "- AOA formula is physics-guided, not discovery evidence.",
        "- Displacement formulas are evaluated as discovery targets.",
        "",
        "## Training",
    ]
    if train_log_path and Path(train_log_path).exists():
        log = pd.read_csv(train_log_path)
        if not log.empty:
            last = log.iloc[-1].to_dict()
            for key in [
                "R2_AOA_val",
                "R2_disp_val",
                "corr_hJ_J",
                "corr_hAOA_AOA",
                "loss_aoa_formula",
            ]:
                if key in last:
                    lines.append(f"- {key}: {last[key]}")
    else:
        lines.append("- No train_log.csv found; symbolic oracle results may still be available.")

    lines.extend(["", "## Symbolic Regression"])
    for result in symbolic_results:
        metrics = ", ".join(f"{k}={v:.6g}" for k, v in result.metrics.items())
        lines.append(f"- {result.name} ({result.backend}): `{result.equation}`; {metrics}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
