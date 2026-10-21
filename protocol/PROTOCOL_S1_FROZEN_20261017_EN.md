# S1 confirmatory study: frozen protocol (written 2026-10-17, before the confirmatory runs)

*English translation of `PROTOCOL_S1_FROZEN_20261017.md`, which is the original (Chinese) file. Section 7 lists every change made after the runs started, with date and reason; nothing else was changed.*

**Working title**: What Actually Drives Mamba-Based Sequential Recommenders? Selectivity, Recency, and the Last Item
**Code**: `s1_lib.py` (models, training, evaluation), `s1_run.py` (tuning / final runs), queue scripts. The pilot code (`../seq_common.py`) is no longer modified.
**Provenance**: hypothesis H2 was suggested by observations in the pilot phase (Baby and Sports, two seeds each). The paper must state that the hypothesis was inspired by pilot data and that the confirmatory study tests it with new tuning, new seeds, and additional datasets.

## 1. Research questions
- RQ1 How much does selectivity (Δ, B, C depending on the input) add over the time-invariant version?
- RQ2 How much accuracy is explained by a fixed recency filter (EMA), by the last item, and by the last item plus the history mean?
- RQ3 How much do the models rely on order and on the most recent item (two test-time placebos), and for which users (strata by history length and target popularity)?
- RQ4 Under which conditions do the conclusions fail? MovieLens-1M (long, strongly sequential histories) is used to test the boundary.
- RQ5 Efficiency: parameters, training time per epoch, inference latency.

## 2. Data
| Dataset | Source | Split | Max. length |
|---|---|---|---|
| Amazon Baby / Sports / Clothing | local 5-core copies | leave-one-out by time (last item test, second last validation) | 50 |
| MovieLens-1M | GroupLens | same leave-one-out split, all ratings treated as implicit interactions | 200 |
| Amazon Electronics (scale check) | local | same | 50 |

## 3. Models
Identical wrapper; only the sequence mixer changes (see the header of `s1_lib.py`): item embedding → dropout → LayerNorm → 2 blocks × [mixer → residual → LN → FFN → residual → LN] → last position; output layer tied to the input embedding; full-softmax cross-entropy with every position as a training target.

Mixers: `last` (no mixing), `mean` (causal mean), `ema` (fixed-form recency filter), `lti` (time-invariant Mamba block), `mamba` (selective Mamba block, the mixer of Mamba4Rec), `sasrec` (causal self-attention), `gru` (GRU4Rec, packed sequences, one layer). All models d = 64, state size 16, 2 blocks (one recurrent layer for the GRU). On the GPU the SSM uses the official mamba_ssm CUDA scan kernel (checked element-wise against a pure PyTorch recurrence, maximum difference 1e-5).

## 4. Tuning and seeds
- Grid: learning rate {1e-3, 3e-3} × dropout {0.2, 0.5}, seed 0; for each (dataset, model) the best configuration by validation NDCG@10. Same budget for all models.
- Final: seeds 0–4 with the selected configuration; test metrics are read only in the final runs.
- Electronics: no separate tuning; the configurations selected on Sports are reused; seeds 0–2.
- Training: Adam, batch 256, at most 200 epochs, validation every 2 epochs, early stopping after 5 evaluations without improvement, best validation weights.

## 5. Metrics and analyses
- Primary metric: test NDCG@10 (HR@10/20 and NDCG@20 also reported); full ranking; history items masked; ties ranked pessimistically.
- Per-user ranks are stored, so any later stratification can be recomputed offline.
- Strata: training popularity of the target item (≤5 / 6–20 / >20); history-length tertiles.
- Placebos (test time, no retraining): (a) history order permuted; (b) most recent item removed.
- Efficiency: parameters, training seconds per epoch, inference latency of full-catalogue scoring (GPU, ms per user).

## 6. Pre-registered hypotheses and decision rules
Statistics: paired over (dataset, seed); 95% bootstrap interval of the paired relative difference (20,000 resamples) + exact Wilcoxon test; Holm correction within the family.

| ID | Hypothesis | Decision rule |
|---|---|---|
| H1 | Selectivity adds little on short e-commerce histories | Three Amazon datasets pooled: upper bound of the 95% interval of mamba vs. lti < 3% |
| H2 | A simple mixer is non-inferior to Mamba (short e-commerce histories) | Three Amazon datasets pooled: lower bound of the 95% interval of mean vs. mamba > −3% (non-inferiority margin 3%) |
| H3 | Reliance on the most recent item | Without the most recent item, mamba's NDCG@10 drops by ≥ 20% on every Amazon dataset |
| H4 | Boundary: sequence modelling helps on long, sequential histories | On ML-1M, mamba and sasrec both exceed mean by ≥ 5%, with interval lower bound > 0 |
| H5 | Boundary: selectivity helps on long histories | On ML-1M, lower bound of the interval of mamba vs. lti > 0 |

Results for each dataset are also reported. Any hypothesis that fails is reported as such; the grid, seeds, and decision rules are not changed in response to results.

## 7. Deviations (appended after the runs started)
Every change after the start (code fixes, grid changes, data problems) is appended here with time, reason, and scope.

**Deviation 1 (2026-10-18 07:55).** The machine crashed twice within minutes of starting MovieLens-1M training (Windows stop code 0x113, around 23:54 on 10-17 and 07:49 on 10-18), whereas the three Amazon datasets had trained for more than an hour without problems. To reduce sustained full GPU load, the batch size for MovieLens-1M and Electronics was changed from 256 to 128, and a pause was inserted after every batch (duration = 0.6 × the batch's compute time). Pauses change no computation, only wall-clock time; the reported seconds per epoch exclude them. The batch-size change applies to all models on these datasets, so comparisons between models are unaffected; the Amazon datasets keep batch size 256. The only completed MovieLens-1M tuning run (batch 256) was deleted and repeated with the new setting. (Addendum 2026-10-19: between 13:15 and 13:31 the pauses were briefly removed, during which repeated crashes occurred; from 13:55 the remaining runs, including the d = 128 Sports runs, used the same 0.6 pause. Pauses change no computation.)

**Deviation 2 (2026-10-18 15:45).** Electronics runs took about one hour each and slowed down about tenfold late in a run when GPU memory was full (7.7/8 GB); the original plan (7 models × 3 seeds) would have needed more than two days. Changed to: only `last` (already done), `mean`, `ema`, `lti`, `mamba`, `sasrec`, seed 0; evaluation batch 1024 → 256; `expandable_segments` memory allocation enabled. Electronics enters no pre-registered hypothesis and is reported only as a scale check.

**Deviation 3 (2026-10-19 18:29).** Resumed runs with pause factor 0.6 still reproduced the same 0x113 crash at GPU temperatures of 85–89 °C. To raise the completion rate of the remaining runs, the pause factor was set to 1.0, and before each unfinished run the queue waited until the GPU was at or below 75 °C (stopping safely if this was not reached within 30 minutes). Both measures change only wall-clock time, not computation, random seeds, hyper-parameters, or evaluation.

**Deviation 4 (after 2026-10-19 18:49).** After these resumed runs had finished all d = 128 Sports runs and MovieLens-1M `mean` seed 0, the same 0x113 crash occurred again while the GPU was at 60 °C and 0% utilization, just before the next run started. At the user's request the pause factor was set back to 0 (full speed) for the remaining runs; the ≤ 75 °C start gate was kept. Scheduling only; no change to computation, seeds, hyper-parameters, or evaluation.

**Deviation 5 (2026-10-19 19:57).** At the user's request the start-temperature gate was removed; the pause factor stayed at 0. The remaining runs (the other d = 128 MovieLens-1M seeds and Electronics) were completed with this setting. Scheduling only; no change to computation, seeds, hyper-parameters, or evaluation.

**Additional analyses (outside the protocol; reported as robustness / exploratory in the paper):** (1) embedding size 128: Sports and ML-1M, `mean`/`ema`/`lti`/`mamba`/`sasrec`, configurations from the d = 64 tuning, seeds 0–2; (2) training on permuted histories: the three Amazon datasets, the same five models, seeds 0–2, with test results on the ordered and on a permuted history.
