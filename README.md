# Pyree

**Fire memory enables global wildfire forecasts that beat climatology up to four months ahead**

Pyree is a global data-driven model that forecasts where wildfires will burn at 0.25° resolution, from one to sixteen 8-day periods ahead (about one week to four months). It combines weather and land-surface drivers, pixel-level fire memory and a recency-weighted fire climatology in a slow–fast transformer (StepsNet backbone from [Sonny](https://arxiv.org/abs/2603.21284)) that decodes every output pixel from its own token. A single model serves all lead times.

On the [SeasFire](https://doi.org/10.5281/zenodo.8055879) benchmark (2019 test year), Pyree reaches:

| Lead (× 8 days) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| **Pyree** (AUPRC) | **0.724** | **0.688** | **0.663** | **0.649** | **0.642** |
| TeleViT<sub>i,g</sub>, reproduced | 0.624 | 0.620 | 0.617 | 0.619 | 0.607 |
| Recency-weighted climatology (no training) | 0.630 | 0.630 | 0.630 | 0.630 | 0.630 |

Results reproduce on 2020 (a second held-out year) and when every land patch is scored.

> Paper: M. Cheon, *Fire memory enables global wildfire forecasts that beat climatology up to four months ahead* (submitted to *npj Natural Hazards*).

## Repository layout

```
pyree/                 model, data pipeline, training and analyses
  model.py             Pyree (slow–fast StepsNet transformer with AdaLN-Zero conditioning)
  model_televit.py     re-implementation of TeleViT_i,g (Prapas et al., 2023) used as baseline
  dataset_v2.py        SeasFire patch dataset: fire memory, anomalies, climatology prior
  fire_prior.py        smoothed, recency-weighted fire climatology (leave-one-year-out for training)
  train_v2.py          training (Pyree, TeleViT, ablations)
  eval_ckpt.py / eval_multi.py   evaluation (benchmark and all-land protocols, 2019 / 2020)
  bootstrap.py         paired block bootstrap over target dates
  baselines.py, sweep_prior*.py  climatology baselines
  attribution.py       integrated gradients (discrimination target)
  pop_sweep.py         population-density counterfactual
  latency.py           inference time
  sae/                 sparse autoencoders (TopK, JumpReLU, KAN-SAE) and causal steering
figures/               scripts that produce every figure and table of the paper
```

## Installation

Tested with Python 3.10 and PyTorch 2.5 (CUDA 12.1) on NVIDIA A40 GPUs.

```bash
pip install -r requirements.txt
```

## Data

Download the SeasFire datacube v0.3 from Zenodo ([doi:10.5281/zenodo.8055879](https://doi.org/10.5281/zenodo.8055879)), unzip it, and point the code to the Zarr store:

```bash
export SEASFIRE_ZARR=/path/to/SeasFireCube_v3.zarr   # default: data/SeasFireCube_v3.zarr
export PYREE_RUNS=runs                                 # where checkpoints and predictions are written
```

## Reproducing the paper

All commands are run from the repository root.

**Pyree (main model)**
```bash
python -m pyree.train_v2 --prior_w 3 --recency_tau 4 --no_anchor \
    --anom_windows 1 4 12 24 --long_fire --tslf --ema 0.999 --out_dir runs/v4_noanchor
```

**Reproduced TeleViT<sub>i,g</sub>** (one model per lead, published hyperparameters)
```bash
for h in 1 2 4 8 16; do
  python -m pyree.train_v2 --arch televit --basic_inputs --leads $h --prior_w 3 --recency_tau 4 --no_anchor \
      --epochs 50 --patience 8 --lr 1e-4 --weight_decay 1e-6 --out_dir runs/televit_orig_h$h
done
```

**Ablations** (the main-model command with exactly one change)

| Run | Change |
|---|---|
| `abl_nofire` | `--drop fire_history` |
| `abl_noclim` | `--drop clim` |
| `abl_nooci` | `--drop oci` |
| `abl_noanom` | remove `--anom_windows` |
| `abl_notau` | remove `--recency_tau 4` |
| `v4_anchor` | remove `--no_anchor` (logit anchoring) |
| `televit_info` | `--arch televit --lr 1e-4` (TeleViT backbone, Pyree inputs) |

**Evaluation**
```bash
# 2020 held-out year (benchmark protocol) and all-land protocol for 2019 / 2020
python -m pyree.eval_multi --split test2020 --runs v4_noanchor televit_orig_h1 televit_orig_h2 televit_orig_h4 televit_orig_h8 televit_orig_h16
python -m pyree.eval_multi --split test --tile_mode land --runs v4_noanchor televit_orig_h1 televit_orig_h2 televit_orig_h4 televit_orig_h8 televit_orig_h16
# paired block bootstrap
python -m pyree.bootstrap --a runs/v4_noanchor/test_preds.npz --b runs/televit_orig_h1/test_preds.npz --leads 1
```

**Analyses**
```bash
python -m pyree.attribution --run_dir runs/v4_noanchor --n_per_lead 384
python -m pyree.pop_sweep --run_dir runs/v4_noanchor --n_per_lead 1000
python -m pyree.latency
python -m pyree.sae.collect_step1 --run_dir runs/v4_noanchor
for v in topk jumprelu kan; do python -m pyree.sae.train_sae --run_dir runs/v4_noanchor --variant $v; done
python -m pyree.sae.interpret --run_dir runs/v4_noanchor
python -m pyree.sae.steer_eval --run_dir runs/v4_noanchor --n_random 20
```

**Figures and tables** (written to `outputs/figures/`)
```bash
cd figures
python score_land.py && python case_studies.py && python fig4_regional.py
python fig2_lead_curve.py   # etc.; one script per figure, see file headers
python make_tables.py
```

The run directory name `v4_noanchor` is the main Pyree model throughout the code.

## Citation

Related work by the author: [Sonny](https://arxiv.org/abs/2603.21284) (StepsNet backbone), [Rescene](https://arxiv.org/abs/2608.09971) (dual-clock design), [KAN-SAE](https://arxiv.org/abs/2605.17493) (sparse-autoencoder interpretation).

## Acknowledgements

The SeasFire datacube is provided by the SeasFire project. The TeleViT baseline follows the original implementation at [Orion-AI-Lab/televit](https://github.com/Orion-AI-Lab/televit).

## License

Released under the [MIT License](LICENSE).

## Contact

Minjong Cheon, Department of Computer Science, Korea National Open University — jmj2316@mail.knou.ac.kr
