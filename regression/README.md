# 🌊 AOA Formula Discovery

Discover interpretable AOA-J power laws from turbulence-degraded videos with a
**TurbFM encoder + Deflex latent bottleneck + log-space symbolic regression**.

This repository contains the simulation, long-range, and RLRAT experiments used
to generate the figures in `results/figure/`.

## ✨ What is discovered?

The main target is a compact power law:

```text
AOA_proxy = C * J_proxy^p
```

## 📦 Repository layout

```text
encoder/          TurbFM code and pretrained initialization
simulation/       Simulation training and symbolic regression
real/             Long-range and RLRAT training/formula discovery
weights/          Released best checkpoints
data_artifacts/   Fixed real-data splits and proxy caches
results/          CSV/JSON results, reports, and figures
tools/            Publication plotting tools
```

## 🚀 Quick start

### Reproduce all figures

The committed CSV/JSON results are enough. Raw videos and GPUs are not required.

```bash
python tools/plot_results.py
python tools/plot_results.py --process
```

The figures are written to `results/figure/`.

## 🗂️ Prepare the datasets

Raw videos are not redistributed. Create the following layout from the repository
root, or update the paths in the YAML configs:

```text
data/
├── sim_AOA_10_80_10/
│   ├── all_formula_table.csv
│   └── ... simulation videos ...
├── turbulence_sequences/
├── turbulence_sequences_RLRAT/
└── grouped_scenes/
    └── <scene-group>/*.avi
```

Videos in the same `grouped_scenes/<scene-group>/` folder share one RLRAT
formula constant `C_g`.

## 🧪 Simulation workflow

Run simulation commands from `simulation/` so the relative config paths resolve.

### Use the released model

```bash
cd simulation

python extract_latents.py \
  --csv ../data/sim_AOA_10_80_10/all_formula_table.csv \
  --data-root ../data/sim_AOA_10_80_10 \
  --ckpt ../weights/simulation_baseline_best_composite.pt \
  --split-info ../results/outputs_aoa10_80_baseline/split_info.json \
  --out ../runs/simulation_latents

python evaluate.py \
  --latent-table ../runs/simulation_latents/latent_table_all.csv \
  --out ../runs/simulation_symbolic
```

### Train from scratch

```bash
python train.py \
  --config configs/config_aoa10_80_baseline.yaml \
  --out ../runs/outputs_aoa10_80_baseline
```

The other configs reproduce the `Disp + J-D` and `J-D only` ablations:

```text
configs/config_aoa10_80_disp_jd.yaml
configs/config_aoa10_80_jd_only.yaml
```

## 📷 Real-data workflow

Run these commands from the repository root.

### Use the released model

First prepare the fixed splits and proxy statistics, then evaluate the released
checkpoint and discover the grouped formula:

```bash
python real/train_real_long_logj.py \
  --config real/config_real_long_jcal.yaml \
  --stage prepare

python real/train_real_long_logj.py \
  --config real/config_real_long_jcal.yaml \
  --stage evaluate \
  --checkpoint weights/real_long_jcal_best.pt

python real/discover_long_logj_formula.py \
  --config real/config_real_long_jcal.yaml \
  --experiment-out runs/outputs_real_long_jcal_83init \
  --checkpoint weights/real_long_jcal_best.pt \
  --recompute
```

### Train from scratch

```bash
python real/train_real_long_logj.py \
  --config real/config_real_long_jcal.yaml \
  --stage all
```

This final real-data experiment uses only `turbulence_sequences` and
`turbulence_sequences_RLRAT`. 

## ⚖️ Notes

- Released checkpoints keep FP32 model weights but remove optimizer state
- The raw videos and the weights are in the same link in the front page.
- The Deflex MIT notice is retained in `THIRD_PARTY_LICENSES/`.
