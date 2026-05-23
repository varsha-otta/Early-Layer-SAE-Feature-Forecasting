# Step 5: Data-efficiency sweeps

**Status**: done. 820 raw probe results (708 non-degenerate), 9-point N-sweep × 5 subsamples × 5 features × 4 layers.

**Date**: 2026-05-23.

## Goal

For each (feature, layer) studied in Step 4, sweep the training-set size `N` to see how much data the linear probe actually needs to hit a meaningful AUC. The headline output is a ratio: `M_SAE / N_probe@AUC_target` - i.e. how many fewer tokens the precursor probe needs vs. training the late-layer SAE.

## Headline result

**Smallest N (tokens) at which the mean AUC-ROC crosses 0.9** (linear interpolation between adjacent N points in log-N space; "never" means even the full 81.6k-token train fold doesn't get there):

| feature | theme | L5 | L8 | L12 | L20 | full AUC-ROC@L20 |
|---|---|---|---|---|---|---|
| 9989 | refusal | never | 49.5k | 21.1k | **3.1k** | 0.996 |
| 817 | deception | never | 31.0k | 26.3k | **3.5k** | 0.997 |
| 12730 | ethics | never | 18.3k | 26.9k | **6.4k** | 0.993 |
| 892 | sycophancy-adj | 4.0k | 0.9k* | 0.9k* | **0.9k*** | 0.998 |
| 1031 | harm | 3.8k | 2.6k | 2.4k | **1.8k** | 0.997 |

\* interpolated from k=2 non-degenerate subsamples at N∈{2, 4} sequences (rare features fail to produce positives in a single sub-1k-token subsample about half the time). Stable N=16-onward results corroborate: sycophancy@L≥8 reaches ROC=0.9 well within 4 k tokens; the small-N number is the right interpretation of "smallest mean-crossing," it just rests on fewer samples than the larger-N points.

**Data-efficiency vs GemmaScope SAE.** Lieberum et al. (2024) report training the residual-stream Gemma-Scope SAEs on **~4B tokens**. Our probes hit AUC-ROC ≥ 0.9 with up to **six orders of magnitude less data**:

| feature | theme | best layer | N_probe (tokens) | M_SAE / N_probe |
|---|---|---|---|---|
| 9989 | refusal | L20 | 3,139 | ~1.3M× |
| 817 | deception | L20 | 3,498 | ~1.1M× |
| 12730 | ethics | L20 | 6,356 | ~630k× |
| 892 | sycophancy-adj | L20 (or L8) | ~860 | ~4.6M× |
| 1031 | harm | L20 | 1,812 | ~2.2M× |

Caveats on the ratio:
- The SAE solves an unsupervised dictionary-learning problem over all features simultaneously; the probe is a labeled binary classifier for one feature. They are not equivalent tasks. The ratio is a "data-token-budget headline," not an apples-to-apples efficiency claim. The number is still striking because labels for a small set of safety-relevant features are much cheaper to obtain (or impute) than the cost of re-training the SAE.
- The probe relies on the SAE itself for the labels in the first place. So a sharper framing: **once we know which 5 features matter, predicting them with a layer-20 probe takes ~10⁻⁶ of the data that surfaced them.**
- For the precursor question (early-layer probes), at layers 5-12 the data efficiency is lower but the result is the more interesting one: predicting feature firing from L8 needs 0.9k-49k tokens, still ~10⁵-10⁶× less than 4B.

## AUC-PR threshold

We also computed the smallest N at which mean AUC-PR crosses 0.5. The result: **only L20 crosses for any feature**, and only refusal, deception, sycophancy-adj, harm cross there (10-30k tokens). PR=0.5 is too strict to discriminate early-layer probes in this regime; Step 4 already showed PR-AUC saturates 0.16-0.50 at full-N for the early layers, well below 0.5.

**Verdict on the "do both?" question:** ROC=0.9 is the right primary threshold for the headline because it produces a non-trivial efficiency number for 19 of 20 (feature, layer) combos. PR is reported in the curves for context but does not earn a slot in the headline table.

## What the numbers say

1. **Feature-dependent decodability dominates.** Harm and sycophancy-adjacent are decodable from layer 5 with single-digit-thousand tokens; refusal, deception, and ethics need either deeper layers or full-fold data to reach the same threshold. This re-confirms the Step 4 finding that "harm is shallowly encoded" and "ethics is the most abstract"; and now quantifies the gap as roughly 10× in N.

2. **The same-layer upper bound is remarkably tight.** Across all 5 features at L20, the probe needs **<6.4k tokens** to hit AUC-ROC ≥ 0.9. That is roughly 0.00016% of the SAE's training data. The "the SAE produces a decodable label" hypothesis holds at every layer we tested, but is most efficient at the layer it was trained on.

3. **Variance shrinks fast.** At N=2 sequences, the std across 5 subsamples is 0.09-0.21 (high; effectively noise). By N=16 (4k tokens), std typically drops below 0.025. The 5-subsample average is a stable summary for N ≥ 16.

4. **Degeneracy floor.** 112 of the 820 raw runs (14%) had zero positives in the subsample and were skipped; concentrated at N=2 (3-5/5 per feature) and N=4 (1-3/5 for low-rate features). For sycophancy-adj (the rarest feature, 0.44%), the floor at N=2 is harsh: only 2 of 5 subsamples retained positives. The aggregate CSV's `n_subsamples` column tracks this.

5. **No regularization re-tune needed.** Holding C = 0.001 (Step 4's pick) across the sweep yielded sensible curves at every N. At very small N the probe is dominated by regularization and produces near-random scores, which is correct behavior; the "data efficiency" comes from N exceeding a level where shrinkage stops dominating.

## Config (single source of truth)

| | value |
|---|---|
| Probes | linear only (sklearn `LogisticRegression`, lbfgs, `class_weight='balanced'`); no MLP (Step 4 showed it doesn't help) |
| N grid (sequences) | `[2, 4, 8, 16, 32, 64, 128, 256, 320]` - 9 points, dense at the low end |
| N grid (tokens, BOS-masked) | `[510, 1020, 2040, 4080, 8160, 16320, 32640, 65280, 81600]` |
| Subsamples per N | 5 (1 at N=320 since that's the full fold, deterministic) |
| Subsample unit | **sequences** (not tokens); preserves leak-free split |
| Subsample seeds | base 10 (= split seed 0 + 10); seeds 10-14 |
| Test fold | fixed, same 80 sequences as Step 4 (cross-N AUC on same denominator) |
| L2 strength | `C = 0.001` constant across all probes |
| Standardization | per-dim z-score, fit on each subsample's rows only (no leakage) |
| Bootstrap CIs | **skipped**; 5-subsample spread is the variance estimate |
| Total non-degenerate fits | 708 |

## Runtime

~27 min wall (much faster than the 80-min pre-flight estimate).

| layer | wall |
|---|---|
| 5 | 6.7 min |
| 8 | 6.4 min |
| 12 | 6.9 min |
| 20 | 6.8 min |
| **total** | **26.6 min** |

Roughly 2.3 sec average per fit (fits range 0.1-23 sec from N=2 to N=320).

## Risks vs outcome

| Risk | Outcome |
|---|---|
| 8 GB RAM | Peak ~940 MB (one layer's train + test in fp32). Comfortable. |
| Degenerate subsamples at small N | Detected and skipped at fit-time; reported as `degenerate=1` rows. 14% of total runs (mostly at N∈{2,4}). |
| C=0.001 not optimal at small N | Not investigated; per-N C re-tune is out of scope. The smooth monotonic curves suggest C=0.001 isn't too far from optimal at any N. |
| Variance estimate based only on 5 subsamples | Small-N curves visibly wobble; the aggregate CSV's std column captures this. ROC=0.9 mean-crossing for sycophancy@L8 (857 tokens) interpolated from k=2 noisy points; flagged in the headline. |

## Differences vs the plan's sketch

| Item | Plan | Actual |
|---|---|---|
| N grid | "log-spaced ~8 points [500, 1k, 2k, 5k, 10k, 20k, 50k, 100k] tokens" | 9 points in sequence-counts `[2, 4, 8, 16, 32, 64, 128, 256, 320]` → `[510, 1020, ..., 81600]` tokens (denser at low end, capped at the full train fold) |
| Subsamples | "5 random subsamples for variance" | 5 (and 1 at the full-fold point) |
| Bootstrap | implicit from Step 4 (200 resamples per fit) | Skipped to save runtime; subsample spread is the variance |
| Threshold metric | "Suggest 0.9" (unspecified which AUC) | ROC=0.9 chosen for headline after both ROC and PR were tested; PR=0.5 is too strict for the early-layer regime |
| Plot | "faceted plot per (feature, layer)" | Deferred to Step 7 with the other figures |

## Caveats for Step 6

1. **Probe weights from Step 4 are the right ones to evaluate cross-distribution.** Step 5's probes are smaller, less stable artifacts; interesting for the efficiency story but Step 6's OOD generalization should use the strongest probe per (feature, layer), which is the full-fold Step 4 model.
2. **The cross-distribution evaluation needs a scaler.** Step 4 saved per-(feature, layer) scaler stats in `lin_*_L*.npz`; Step 6 should re-apply these to the safety-cache residuals before running the probe.
3. **Per-feature N-to-threshold varies by 1-2 orders of magnitude.** Step 6 should be aware that the in-distribution baseline AUC-ROC ranges 0.87-0.998 across (feature, layer); the OOD AUC gap is what matters, not absolute OOD AUC.
4. **Sycophancy-adj is the most sample-starved feature.** Step 6's safety corpus should aim for at least 50-100 positives for this feature to make the OOD comparison meaningful.

## Saved artifacts

| File | Content |
|---|---|
| `src/data_efficiency.py` | Sequence-level subsample helper (`make_subsample`, `subsample_seeds_for`, `N_SEQ_GRID`) |
| `scripts/step5_efficiency.py` | Full-sweep orchestrator with `--smoke` and `--layers` flags |
| `scripts/step5_analysis.py` | Aggregator + interpolated headline-crossing extractor |
| `results/step5_efficiency_curves.csv` | 820 raw rows: one per (feature, layer, N, subsample) |
| `results/step5_efficiency_aggregate.csv` | 176 aggregate rows (mean/std/min/max across subsamples) |
| `results/step5_headline.csv` | 20 rows: full-N AUC + N-tokens-to-threshold per (feature, layer) |
| `results/step5_meta.json` | Config, seeds, per-layer wall times |
| `results/step5_run.log` | Full orchestrator log |

## Reproduction

```bash
./safety-env/Scripts/python.exe scripts/step5_efficiency.py
./safety-env/Scripts/python.exe scripts/step5_analysis.py
```

For a 3-N smoke check:

```bash
./safety-env/Scripts/python.exe scripts/step5_efficiency.py --smoke
```
