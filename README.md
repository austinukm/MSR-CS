# What Drives Mamba-Based Sequential Recommenders?

Code, data splits, and results for the paper

> H. Sun, W. Kaur, and S. A. Mohd Noah, "What Drives Mamba-Based Sequential Recommenders? A Controlled Study of Selectivity, Recency, and the Last Item," submitted to *IEEE Access*, 2026.

All models share one wrapper (item embedding, layer normalization, feed-forward layers, tied output layer, full-softmax cross-entropy) and differ only in the sequence mixer:

| Mixer | What it keeps |
|---|---|
| `mamba` | selective Mamba block (input-dependent Δ, B, C), as in Mamba4Rec |
| `lti` | the same block with Δ, B, C learned constants (linear time-invariant) |
| `ema` | per-channel exponential recency filter |
| `mean` | causal running mean; with the residual path: last item + history mean |
| `last` | no mixing; only the most recent item |
| `sasrec`, `gru` | causal self-attention and GRU references |

Main findings: on three short-history Amazon datasets, the recency filter is more accurate than the Mamba mixer and `mean` is non-inferior to it. On MovieLens-1M, the state space mixers are the most accurate. Input-dependent selectivity gives no reliable gain in either setting.

## Repository layout

```
src/        s1_lib.py      data loading, wrapper, mixers, training, evaluation, placebos
            s1_run.py      tuning / final / robustness runs (one JSON per run; existing runs are skipped)
            s1_runall.py   robustness and scale-check steps in one process
analysis/   s1_analyze.py  all statistics of the paper -> results/RESULTS.md, results/results.json
            extra_strata.py  popularity and history-length strata (Sections V-D, V-E)
            make_figures.py  Figs. 2-4
data/       amazon_splits.zip  leave-one-out splits of Baby, Sports, Clothing, Electronics
            prep_ml1m.py       recreates the MovieLens-1M split from the official archive
            verify_splits.py   checks every split and prints Table I and SHA-256 hashes
            SHA256SUMS.txt
results/    runs/*.json    361 result files (configuration, metrics, per-user test ranks, strata, placebos, timing)
            RESULTS.md, results.json
protocol/   protocol written before the confirmatory runs (original in Chinese, English translation)
```

## Setup

Tested with Python 3.13, PyTorch 2.11 (CUDA 13.0), and `mamba_ssm` 2.3.2 on an NVIDIA RTX 4060 Laptop GPU.

```bash
pip install -r requirements.txt
```

`mamba_ssm` is only needed for training the Mamba mixers on a GPU; the analysis scripts need only NumPy, pandas, SciPy, and Matplotlib.

## Data

```bash
unzip data/amazon_splits.zip -d data/            # -> data/splits/{baby,sports,clothing,elec}/
# download ml-1m.zip from https://grouplens.org/datasets/movielens/1m/
python -I data/prep_ml1m.py ml-1m.zip data/splits/ml1m
python data/verify_splits.py                     # hashes must match data/SHA256SUMS.txt
```

The Amazon splits are derived from the 5-core review data of McAuley et al. (SIGIR 2015) and He and McAuley (WWW 2016), as distributed with the MMRec benchmark. They contain re-indexed user and item identifiers, ratings, and timestamps only, and are provided for research use. MovieLens-1M is not redistributed; `prep_ml1m.py` recreates the exact split.

## Reproduce the paper's numbers from the released runs (no GPU)

```bash
python analysis/s1_analyze.py      # result tables and hypothesis tests -> results/RESULTS.md
python analysis/extra_strata.py    # popularity and history-length strata
python analysis/make_figures.py    # Figs. 2-4 -> results/figures/
```

## Retrain

Set `S1_RUNS` to an empty folder so that the released runs are not skipped.

```bash
export S1_RUNS=$PWD/results/runs_new
for ds in baby sports clothing ml1m; do
  python src/s1_run.py tune  $ds        # grid: lr {1e-3, 3e-3} x dropout {0.2, 0.5}, seed 0
  python src/s1_run.py final $ds        # seeds 0-4 with the selected configuration
done
python src/s1_run.py shuftrain sports   # training on permuted histories (also baby, clothing)
python src/s1_run.py dim128 ml1m        # embedding size 128 (also sports)
# Electronics reuses the Sports configurations, seed 0
for f in $S1_RUNS/tune_sports_*.json; do cp "$f" "${f/tune_sports_/tune_elec_}"; done
S1_SEEDS=0 S1_BS=128 python src/s1_run.py final elec
```

The paper used batch size 128 for MovieLens-1M and Electronics (`S1_BS=128`) and 256 otherwise. One full pass took about 9 GPU-hours. GPU kernels are not bit-wise deterministic, so retrained runs can differ slightly from the released ones.

Relative to the code that produced the released runs, only file paths were made relative, and the wall-clock pause options in `s1_runall.py` default to off. No computation was changed.

## Citation

```bibtex
@article{sun2026drives,
  author  = {Sun, Haojun and Kaur, Wandeep and Mohd Noah, Shahrul Azman},
  title   = {What Drives {Mamba}-Based Sequential Recommenders? {A} Controlled Study of Selectivity, Recency, and the Last Item},
  journal = {IEEE Access},
  note    = {Submitted},
  year    = {2026}
}
```

## License

Code: MIT License (see `LICENSE`). The data splits remain subject to the terms of the original Amazon review and MovieLens datasets.

Contact: Haojun Sun, Universiti Kebangsaan Malaysia (p154528@siswa.ukm.edu.my).
