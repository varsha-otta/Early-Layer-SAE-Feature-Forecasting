# Step 4: Probe training + per-feature evaluation

**Status**: done. 40 probes (5 features × 4 layers × 2 probe types) trained and evaluated.

**Date**: 2026-05-23.

## Goal

Train one linear (logistic regression) and one tiny MLP probe per (target feature, residual-stream layer) combination, evaluate on a held-out test fold, and check whether the precursor signal — i.e. the ability to predict a late-layer SAE feature's firing from earlier-layer activations — is decodable, and how it scales with depth.

## Headline result

Every target feature is predictable from layer 5 activations alone — and every late-layer probe is essentially saturated. **Linear AUC-ROC by (feature, layer):**

| feature | theme | L5 | L8 | L12 | L20 |
|---|---|---|---|---|---|
| 9989 | refusal | 0.891 | 0.906 | 0.934 | **0.996** |
| 817 | deception | 0.893 | 0.924 | 0.938 | **0.997** |
| 12730 | ethics | 0.873 | 0.925 | 0.918 | **0.993** |
| 892 | sycophancy-adj | 0.936 | 0.972 | 0.987 | **0.998** |
| 1031 | harm | 0.978 | 0.978 | 0.983 | **0.997** |

The corresponding **AUC-PR table** tells a different and more diagnostic story (positives are 0.5-1.4% of tokens, so PR-AUC is the right headline for rare-event ranking):

| feature | theme | L5 | L8 | L12 | L20 | base rate |
|---|---|---|---|---|---|---|
| 9989 | refusal | 0.162 | 0.175 | 0.282 | **0.817** | 0.93% |
| 817 | deception | 0.196 | 0.278 | 0.332 | **0.853** | 1.31% |
| 12730 | ethics | 0.078 | 0.182 | 0.174 | **0.704** | 0.37% |
| 892 | sycophancy-adj | 0.339 | 0.421 | 0.453 | **0.664** | 0.33% |
| 1031 | harm | 0.498 | 0.466 | 0.570 | **0.787** | 0.48% |

PR-AUC ratios over the test-fold base rate range from ~17× (refusal/L5) to ~200× (deception/L20). Even the weakest early-layer PR-AUC (ethics @ L5 = 0.078) is **21× the 0.37% base rate**.

## What the numbers say

1. **The precursor signal exists shallow.** All 5 features get AUC-ROC ≥ 0.87 from layer 5 activations alone. None require the deep layers to be decodable at all; they just become more decodable.
2. **Layer 20 is essentially saturated.** AUC-ROC ≥ 0.99 and AUC-PR > 0.66 across all features, even though the SAE was trained on this layer's activations. This is the upper-bound probe — anything Step 5/6 reports should be benchmarked against it.
3. **Monotonic-but-non-uniform growth.** Refusal and deception climb steadily across layers; harm is nearly flat (0.978 at L5 already); ethics has a small dip at L12 vs L8 (0.918 vs 0.925), within bootstrap CI overlap.
4. **Linear beats MLP almost everywhere.** Across all 20 pairs the MLP either ties (within CI) or loses to the linear probe by 0.001-0.01 AUC-ROC. This is a clean "linear is enough" sanity check: residual-stream → SAE-feature-fires is effectively a linearly-decodable relation at every layer we measured, and the 128-hidden MLP wasn't able to find a non-linear signal worth its expressivity. Two contributing reasons:
   - At the same layer, the SAE encoder *is* a linear projection through `W_enc` followed by JumpReLU; "fires" is a single threshold on a linear function of residuals. So same-layer L20 is linear by construction.
   - Earlier-layer precursor signal seems to flow through linear subspaces of the residual stream — at least at the granularity of 102k training tokens.
5. **Harm (1031) is shallowly encoded.** AUC-ROC ≈ 0.978 already at layer 5, with AUC-PR ≈ 0.50. The feature is largely tracking lexical / shallow semantic cues that the model has already extracted by L5. This is the most "boring" precursor in the sense that depth contributes little; it's also the highest-confidence prediction available from any early layer.
6. **Ethics (12730) is the weakest precursor.** AUC-PR 0.078 at L5, jumping ~2.3× to L8, then flat through L12 before jumping ~4× to L20. The CIs are also widest for this feature (only 75 test positives). Possible interpretation: ethics is the most "compositional" of the five — built up across the network, not surface-form-detectable.
7. **L2 sweep picked C=0.001** (strongest regularization on the [1e-3, 10] grid). All 3 CV folds agreed. With d=2304 and a few hundred positives per feature, heavy shrinkage helps; sklearn's default C=1 would have left the probe under-regularized for this regime.

## Config (single source of truth)

| | value |
|---|---|
| Probes | linear (sklearn `LogisticRegression`, lbfgs, `class_weight='balanced'`) + 2-layer MLP (`2304→128→1`, ReLU, dropout 0.1, BCEWithLogits, Adam 1e-3, 5 epochs, batch 512) |
| Per-probe inputs | per-dim z-scored residuals; scaler fit on the train fold |
| L2 strength | C = 0.001 (chosen once on feature 9989 / layer 12 via 3-fold stratified CV; reused for all 20 linear probes) |
| Split | 320 train / 80 test sequences (seed 0), shared across all 20 probes |
| BOS mask | yes — position 0 of every sequence dropped → 81,600 train / 20,400 test tokens |
| Labels | `feature_acts > 0` (binary fire; the SAE's JumpReLU defines "fires") |
| Bootstrap | 200 stratified resamples on the test fold; 95% percentile CIs |
| Determinism | seed 0 everywhere splits/subsamples/MLP init appear |
| Compute | CPU only (laptop) |

## Per-probe positive counts

After the 80/20 split and BOS mask:

| feature | theme | train positives | test positives |
|---|---|---|---|
| 9989 | refusal | 845 | 190 |
| 817 | deception | 755 | 267 |
| 12730 | ethics | 506 | 75 |
| 892 | sycophancy-adj | 358 | 68 |
| 1031 | harm | 559 | 97 |

These per-feature positive counts will determine the lower end of the N sweep in Step 5: a meaningful subsample needs to retain enough positives for stable AUC-PR estimation. For sycophancy-adj at the smallest N, that's a real constraint (e.g. at N=500 train tokens we'd expect only ~2 positives at the 0.44% rate).

## Runtime

| stage | wall |
|---|---|
| L2 sweep (15 fits, 3-fold × 5 Cs at layer 12) | 8.5 min |
| Layer 5 (5 linear + 5 MLP) | 5.4 min |
| Layer 8 (5 linear + 5 MLP) | 5.3 min |
| Layer 12 (5 linear + 5 MLP) | 5.4 min |
| Layer 20 (5 linear + 5 MLP) | 7.9 min |
| **total** | **~27 min** |

Layer 20 ran longer because the linear fits took more iterations on average — consistent with the picture that L20 residuals carry stronger signal that the optimizer chases further before convergence.

## Risks vs outcome

| Risk | Outcome |
|---|---|
| 8 GB RAM ceiling | Peak ~940 MB per layer (752 MB train + 188 MB test fp32). Comfortable. |
| Linear probe under-regularized | The L2 sweep moved C from 1 (default) to 0.001 and improved CV AUC by 0.04 → confirmed the sweep was worth doing. |
| MLP under-trained at 5 epochs | MLP train loss reached 0.04-0.20; longer training would over-fit (we observed near-zero loss is achievable). The probe was never the headline. |
| Bootstrap noise on small-positive features | Visible: ethics CI half-width ~0.04 AUC-ROC at L5, vs ~0.008 for refusal. PR-AUC CIs even wider. Reported transparently. |
| sklearn FutureWarnings | Removed `penalty='l2'` and `n_jobs=1` after smoke test; no warnings in the full run. |

## Differences vs the plan's sketch

| Item | Plan | Actual |
|---|---|---|
| L2 sweep scope | "swept on a held-out subset of the train fold" | One representative sweep (feature 9989, layer 12), C reused across all 20 probes — explicitly confirmed before implementation |
| C grid | unspecified | `[1e-3, 1e-2, 1e-1, 1, 10]` |
| MLP scope | "sanity check" | Run for all 20 (feature, layer) pairs (explicitly confirmed) |
| Bootstrap resamples | "100 resamples" | **200** resamples for tighter percentile estimates |
| Metric set | AUC-ROC, AUC-PR, precision@k, calibration | AUC-ROC, AUC-PR, precision@k (with k=n_test_pos). Calibration plot deferred to Step 7 alongside other figures. |

## Caveats for Step 5

1. **Sycophancy-adj has the fewest positives.** At 358 train positives, the smallest sample point in the N sweep needs to retain enough positives for AUC-PR to be meaningful. Suggest stratified subsampling by sequence (preserving the per-sequence positive count) or capping the smallest N higher than 500 for this feature.
2. **Harm @ L5 is already saturated** (AUC-ROC 0.978). The data-efficiency curve for this combo will likely flatten very quickly; consider whether to include it in the headline efficiency story or treat it as the "trivial" case that proves the pipeline works.
3. **The chosen C=0.001 transferred well** across layers/features. Step 5's smaller-N regimes might prefer different regularization, but the cross-N comparison is cleaner if we hold C fixed. Suggest: hold C=0.001 across the sweep; flag any combos where final-N AUC is materially below this report's full-N AUC.
4. **Shared sequence-level split is preserved.** The test fold (80 sequences) is fixed; data-efficiency subsamples the train fold by sequence index only. This keeps the test-fold positive counts constant, which is what we need for a clean N-vs-AUC curve.
5. **MLP is uninformative for Step 5's headline.** The linear probe is the data-efficiency story; the MLP results are kept for completeness but the README/writeup should focus on linear curves.

## Saved artifacts

| File | Content |
|---|---|
| `src/data.py` | mmap loaders, sequence-level split (BOS masked), z-score fit/apply |
| `src/eval.py` | AUC-ROC, AUC-PR, precision@k, stratified bootstrap CIs |
| `src/probe.py` | `fit_linear` (sklearn) and `fit_mlp` (torch) |
| `scripts/step4_train_probes.py` | Orchestrator: `--smoke`, `--skip-sweep`, `--C` flags |
| `results/step4_l2_sweep.csv` | 15-row sweep results (per fold, per C) |
| `results/step4_probe_metrics.csv` | 40-row metrics table (one per (feature, layer, probe_type)) |
| `results/step4_meta.json` | Seed, split (full sequence-id lists), best_C, per-layer wall times |
| `results/step4_probe_weights/lin_*.npz` | 20 linear probes (coef, intercept, scaler stats, C, n_iter) |
| `results/step4_probe_weights/mlp_*.npz` | 20 MLP probes (state_dict as numpy arrays, scaler stats, hyperparams) |
| `results/step4_run.log` | Full run log for reproducibility |

## Reproduction

From the repo root:

```bash
./safety-env/Scripts/python.exe scripts/step4_train_probes.py
```

For a one-(feature, layer) end-to-end smoke check:

```bash
./safety-env/Scripts/python.exe scripts/step4_train_probes.py --smoke --C 1.0
```

To re-run the full grid but reuse the previously chosen `best_C`:

```bash
./safety-env/Scripts/python.exe scripts/step4_train_probes.py --skip-sweep
```
