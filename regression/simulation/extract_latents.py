from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from src.data import FormulaVideoDataset, load_formula_table, load_or_create_split
from src.models import build_model
from src.train_utils import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Export patch-level latent tables from a trained checkpoint.")
    parser.add_argument("--csv", default="../data/sim_AOA_10_80_10/all_formula_table.csv")
    parser.add_argument("--data-root", default=None, help="Video root; defaults to data_root saved in the checkpoint config.")
    parser.add_argument("--ckpt", default="outputs_latent_codex/checkpoints/best.pt")
    parser.add_argument("--out", default="outputs_latent_codex/latent_tables")
    parser.add_argument("--split-info", default="outputs_latent_codex/split_info.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def _tensor_np(batch, key):
    return batch[key].detach().cpu().numpy()


def export_part(model, df, split, part, args, config, device):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    data_cfg = config["data"]
    dataset = FormulaVideoDataset(
        df,
        split,
        part,
        args.data_root,
        frame_count=data_cfg.get("frame_count", 15),
        image_size=data_cfg.get("image_size", 224),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"extract {part}", disable=not sys.stderr.isatty()):
            frames = batch["frames"].to(device)
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if hasattr(value, "to") and key != "frames"
            }
            out = model(frames, tensors["D_aperture"], tensors["alpha"], tensors["Delta_theta"])
            H = out["H"].detach().cpu().numpy()
            aoa_hat = out["aoa_hat"].detach().cpu().numpy()
            disp_hat = out["disp_hat"].detach().cpu().numpy()
            aoa_log_hat = out["aoa_log_hat"].detach().cpu().numpy()
            disp_log_hat = out["disp_log_hat"].detach().cpu().numpy()
            logJ_latent = out["logJ_latent"].detach().cpu().numpy()
            logAOA_latent = out["logAOA_latent"].detach().cpu().numpy()
            J_latent = out["J_latent"].detach().cpu().numpy()
            AOA_latent = out["AOA_latent"].detach().cpu().numpy()
            batch_size, n_patches, latent_dim = H.shape
            tensor_cols = [
                "sample_index",
                "patch_i",
                "patch_j",
                "J",
                "AOA_var",
                "disp_var",
                "Delta_theta",
                "D_aperture",
                "alpha",
                "L",
                "delta",
                "lambda",
            ]
            values = {col: _tensor_np(batch, col) for col in tensor_cols if col in batch}
            patch_mask = _tensor_np(batch, "patch_mask").astype(bool) if "patch_mask" in batch else None
            for b in range(batch_size):
                for n in range(n_patches):
                    if patch_mask is not None and not patch_mask[b, n]:
                        continue
                    row = {
                        "sample_id": batch["sample_id"][b],
                        "view_id": int(batch["view_id"][b]),
                        "zJ": float(H[b, n, 0]) if latent_dim >= 1 else None,
                        "zAOA": float(H[b, n, 1]) if latent_dim >= 2 else None,
                        "logJ_latent": float(logJ_latent[b, n]),
                        "logAOA_latent": float(logAOA_latent[b, n]),
                        "J_latent": float(J_latent[b, n]),
                        "AOA_latent": float(AOA_latent[b, n]),
                        "aoa_log_hat": float(aoa_log_hat[b, n]),
                        "disp_log_hat": float(disp_log_hat[b, n]),
                        "aoa_hat": float(aoa_hat[b, n]),
                        "disp_hat": float(disp_hat[b, n]),
                        "video_path": batch["video_path"][b],
                    }
                    for k in range(latent_dim):
                        row[f"h{k + 1}"] = float(H[b, n, k])
                    if latent_dim >= 1:
                        row["hJ"] = float(J_latent[b, n])
                    if latent_dim >= 2:
                        row["hAOA"] = float(AOA_latent[b, n])
                    for col, arr in values.items():
                        row[col] = arr[b, n].item()
                    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Latent extraction requires PyTorch. Install requirements.txt or run in the project "
            f"environment with PyTorch available. Missing: {exc.name}"
        ) from exc

    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    if args.data_root is None:
        args.data_root = config.get("data", {}).get("data_root", "../data/sim_AOA_10_80_10")
    device = resolve_device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    df = load_formula_table(args.csv)
    split = load_or_create_split(df, args.split_info, random_state=config["data"].get("random_state", 0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_parts = []
    for part in ["train", "val", "test"]:
        part_df = export_part(model, df, split, part, args, config, device)
        part_df.to_csv(out / f"latent_table_{part}.csv", index=False)
        all_parts.append(part_df)
    pd.concat(all_parts, ignore_index=True).to_csv(out / "latent_table_all.csv", index=False)


if __name__ == "__main__":
    main()
