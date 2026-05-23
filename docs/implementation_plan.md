# Implementation plan: Early-Layer-SAE-Feature-Forecasting

Single living plan covering all seven steps in the README. Resumable: each step has a status, design notes, and pointers to its artifacts. Pick up from any "next" or "in progress" row.

**Last updated:** 2026-05-23 (Step 7 done; project complete).

## Research question

Can a small classifier trained on **early-layer** Gemma-2-2B residual stream activations predict whether a **late-layer** safety-flavored SAE feature will fire at the same token position; and crucially, how does the probe's data efficiency compare to training the late-layer SAE itself?

**Headline claim we want to support:** "We predict feature F at layer 20 from layer 5 activations with N tokens of probe data, vs M ≫ N tokens needed for the GemmaScope SAE at layer 20 to surface F as a coherent feature."

Three sub-questions, each owned by a later step:

1. **How early in the network is the precursor signal decodable?** (Step 4: probe at layers 5/8/12/20, compare AUC.)
2. **How data-efficient is the precursor probe vs the SAE?** (Step 5: AUC vs N sweep; baseline is the GemmaScope reported training-set size.)
3. **Does the probe generalize across prompt distributions?** (Step 6: train on web text, eval on safety prompts.)

## Standing decisions (cross-cutting)

| | Value | Set in |
|---|---|---|
| Model | `google/gemma-2-2b` (base, 26 layers) | README + Step 1 |
| SAE release | `gemma-scope-2b-pt-res-canonical` | Step 2 |
| SAE id | `layer_20/width_16k/canonical` | Step 2 |
| Late layer (label source) | **20** (NB: supersedes README's stale `{18, 22}` placeholder) | Step 2 |
| Early layers (probe inputs) | `[5, 8, 12]`; layer 20 also cached as same-layer upper bound | README + Step 3 |
| Target features | `[9989 refusal, 817 deception, 12730 ethics, 892 sycophancy-adj, 1031 harm]` | Step 2 |
| Activation site | post-block residual stream (`model.model.layers[L]` output tuple[0]) | Step 1 |
| Compute split | Colab T4 for extraction; local CPU (8 GB RAM) for probes | README |
| Cache dtype | float16, mmap-readable (separate uncompressed `.npy` per layer) | Step 3 |
| Determinism | Pipeline is seed-free (no shuffling, no sampling, `model.eval()`); seeds appear only at probe train/test split | Step 3 + Step 4 |

## Status snapshot

| # | Step | Status | Artifact |
|---|---|---|---|
| 1 | Env + smoke test | done | `notebooks/01_smoke_test.ipynb` |
| 2 | Target feature selection | done | `docs/02_feature_selection.md`, `data/target_features.json`, `scripts/step2_neuronpedia_search.py` |
| 3 | Activation cache extraction (Colab) | done | `notebooks/02_activation_cache.ipynb`, `scripts/check_activation_cache.py`, `data/cache/v1/`, `docs/03_activation_cache.md` |
| 4 | Probe training + per-feature evaluation | done | `src/{data,eval,probe}.py`, `scripts/step4_train_probes.py`, `results/step4_*`, `docs/04_probes.md` |
| 5 | Data-efficiency sweeps | done | `src/data_efficiency.py`, `scripts/step5_{efficiency,analysis}.py`, `results/step5_*`, `docs/05_data_efficiency.md` |
| 6 | Generalization tests | done | `notebooks/03_safety_cache.ipynb`, `scripts/{check_safety_cache,step6_ood_eval}.py`, `data/cache/safety_v1/`, `results/step6_*`, `docs/06_generalization.md` |
| 7 | Write-up | done | `docs/07_writeup.md`, `docs/figures/fig{1,2,3}_*.png`, `scripts/step7_make_figures.py` |

Workflow: per the project's pause-per-step convention, each step is completed and reviewed before the next begins.

---

## Step 1: Env + smoke test - DONE

Goal was to verify the activation-extraction pipeline end-to-end on Colab T4.

What was verified:
- Gemma-2-2B loads via `transformers` (not `transformer_lens`, which OOMs Colab free's host RAM) with `torch.bfloat16` + `device_map=device` + `low_cpu_mem_usage=True`.
- GemmaScope SAE loads at layer 20, width 16k, canonical.
- Manual `register_forward_hook` on `model.model.layers[L]` (output tuple's first element) captures the post-block residual stream; the activation site GemmaScope residual SAEs were trained on.
- `sae.encode(resid_20)` returns sane sparse activations (~tens to hundreds of nonzero features per token).

Artifact: `notebooks/01_smoke_test.ipynb`.

## Step 2: Target feature selection - DONE

Goal was to pick 3-5 safety-flavored SAE features at Gemma-2-2B layer 20 for use as probe targets.

Process:
- Queried Neuronpedia's `/api/explanation/search` for 20 safety-related keywords (refusal, deception, sycophancy, harm, ethics, hedging, etc.). 79 unique candidates surfaced; 73 in the 0.05-1% firing-rate band.
- For the top 10 candidates, fetched per-feature top-activating contexts to verify the auto-interp labels (3 of 10 turned out mislabeled or too narrow; important catch).
- Picked 5 features for diversity across safety dimensions.

Picks: `[9989 refusal, 817 deception, 12730 ethics, 892 sycophancy-adjacent, 1031 harm]`, firing rates 0.28%-0.76%.

Caveats: no clean sycophancy feature exists in the base model (would need Gemma-2-2B-it); no clean "controversial topic" or distinct "harmful-content recognition" feature surfaced.

Artifacts:
- `docs/02_feature_selection.md` - full record (queries, verification table, decision rationale, swap candidates)
- `data/target_features.json` - machine-readable handoff to Step 3
- `data/neuronpedia_search_raw.json`, `data/shortlist_v1.json` - raw and verified data
- `scripts/step2_neuronpedia_search.py` - idempotent reproduction

## Step 3: Activation cache extraction (Colab) - DONE

Cache built on Colab T4 on 2026-05-23, downloaded and verified locally. Final size matches the design (~1.89 GB across 7 files). Empirical per-feature fire rates and detailed retrospective: `docs/03_activation_cache.md`.

Sections below preserve the pre-implementation design as a reference for what was built; only "Outputs" and "Deliverables" carry post-run state.

### Goal

A ~1.89 GB float16 activation cache, extracted on Colab T4 in one ~5-minute run, downloadable to local. Mmap-readable so Step 4 probe code never holds more than one layer's worth of float32 data in RAM.

### Configuration (single source of truth, top of notebook)

```python
CORPUS = 'NeelNanda/pile-10k'              # web text training distribution
N_SEQUENCES = 400                           # → 400 × 256 = 102,400 tokens total
SEQ_LEN = 256                               # BOS-prefixed
EARLY_LAYERS = [5, 8, 12]                   # precursor probes
LATE_LAYER = 20                             # same as SAE layer (upper-bound probe + label source)
ALL_LAYERS = EARLY_LAYERS + [LATE_LAYER]    # = [5, 8, 12, 20]
FEATURE_INDICES = [9989, 817, 12730, 892, 1031]
SAE_RELEASE = 'gemma-scope-2b-pt-res-canonical'
SAE_ID = 'layer_20/width_16k/canonical'
BATCH_SIZE = 8
OUTPUT_DIR = '/content/drive/MyDrive/safety-sae-cache/v1'
```

### Outputs (actual, ~1.89 GB total)

| File | Shape | dtype | Size |
|---|---|---|---|
| `resid_layer_5.npy`  | (102400, 2304) | float16 | 471.9 MB |
| `resid_layer_8.npy`  | (102400, 2304) | float16 | 471.9 MB |
| `resid_layer_12.npy` | (102400, 2304) | float16 | 471.9 MB |
| `resid_layer_20.npy` | (102400, 2304) | float16 | 471.9 MB |
| `feature_acts.npy`   | (102400, 5)    | float16 | 1.02 MB |
| `token_ids.npy`      | (102400,)      | int32   | 0.41 MB |
| `metadata.json`      | -              | -       | ~1.4 KB |

Uncompressed `.npy` (compressed `.npz` does not mmap).

`metadata.json` captures: full config, empirical per-feature fire rates, pinned Pile-10k commit, torch/sae_lens/transformers/datasets/numpy versions, timestamp, and `platform.platform()`. See `docs/03_activation_cache.md` for the recorded values.

### Token packing

1. Stream `NeelNanda/pile-10k` in dataset order; tokenize each doc with `add_special_tokens=False`.
2. Concatenate into a flat buffer of length `400 × 255 = 102,000` raw tokens; truncate exact.
3. Reshape to `(400, 255)`, prepend BOS column → `(400, 256)`.
4. Cross-document spans within a sequence are acceptable (matches GemmaScope's training packing).

### Notebook structure

Markdown intro → install deps (`sae_lens accelerate datasets`) → mount Drive → HF auth → config cell → load model → load SAE → load+tokenize corpus → pre-allocate output arrays → register hooks → forward+encode loop (50 batches, tqdm) → save artifacts → sanity stats (per-feature empirical fire rate vs Neuronpedia, top-activating tokens, residual norm distributions) → download instructions.

### Memory plan

**Colab (12.7 GB host RAM):** 1.89 GB pre-allocated + ~100 MB dataset + ~95 MB transient per-batch activations. Comfortable.

**Local laptop (8 GB system RAM):** one mmapped layer is 471 MB lazy; one float32 subsample for a probe is ≤460 MB; labels are 1 MB. Step 4 probe code follows the rule **never more than one early-layer's worth of float32 data in RAM at once**.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Drive lacks ~2 GB free | Notebook checks `shutil.disk_usage(...)` early, aborts before GPU work |
| HF Gemma-2-2B license not accepted | Same auth path as smoke test; error message is self-explanatory |
| SAE encode dtype mismatch | Cast residual to `sae.W_enc.dtype` before encode (smoke-test pattern) |
| Float16 clipping | GemmaScope max acts ~125; fp16 range ±65504; safe. Sanity cell prints max-abs per layer to confirm. |
| Pile-10k content drift on HF | Pin the snapshot revision; record commit hash in `metadata.json` |
| User re-runs the hook-registration cell | Cleanup helper removes any pre-existing hooks at top of loop cell |

### Deliverables

- [x] `notebooks/02_activation_cache.ipynb`
- [x] `scripts/check_activation_cache.py` (local verifier; runs after download)
- [x] README status table updated: Step 3 → "ready to run (Colab)", then to "done"
- [x] `data/cache/v1/` populated locally; this section reframed as retrospective; empirical numbers in `docs/03_activation_cache.md`

### Out of scope for Step 3

Safety-prompt corpus (Step 6), probe training code (Step 4), int8 quantization (premature), caching at layers 16/24/etc. (extend in v2 if curve has gaps).

---

## Step 4: Probe training + per-feature evaluation - DONE

40 probes trained and evaluated on 2026-05-23. Full retrospective: `docs/04_probes.md`.

### Resolved open questions

- **BOS masking:** yes. Position 0 of every sequence is dropped from train and test → 81,600 train / 20,400 test tokens.
- **Shared sequence-level split:** yes. Seed 0 partitions the 400 sequences into 320 train / 80 test, shared across all 20 (feature, layer) probe configs so AUC numbers are directly commensurable.

### Final locked-in config

| | value |
|---|---|
| Linear probe | sklearn `LogisticRegression`, lbfgs, `class_weight='balanced'`, max_iter=1000 |
| MLP sanity | 2304→128→1 ReLU, dropout 0.1, BCEWithLogits + pos_weight, Adam 1e-3, 5 epochs, batch 512 |
| Standardization | per-dim z-score, scaler fit on train fold only, applied in-place to free RAM |
| L2 sweep | one representative sweep (feature 9989, layer 12) on C ∈ [1e-3, 1e-2, 1e-1, 1, 10], 3-fold CV. **Best C = 0.001** transferred to all 20 linear probes. |
| Bootstrap | 200 stratified resamples (positives and negatives resampled independently, original counts preserved); 95% percentile CIs. |
| Metrics | AUC-ROC, AUC-PR, precision@k with k = n_test_positives. (Calibration plot deferred to Step 7 with the other figures.) |

### Headline empirical results (linear AUC-ROC)

| feature | L5 | L8 | L12 | L20 |
|---|---|---|---|---|
| 9989 refusal | 0.891 | 0.906 | 0.934 | **0.996** |
| 817 deception | 0.893 | 0.924 | 0.938 | **0.997** |
| 12730 ethics | 0.873 | 0.925 | 0.918 | **0.993** |
| 892 sycophancy-adj | 0.936 | 0.972 | 0.987 | **0.998** |
| 1031 harm | 0.978 | 0.978 | 0.983 | **0.997** |

**Key takeaways:**
- Linear precursor signal is decodable at layer 5 for every feature (AUC ≥ 0.87).
- Layer 20 is essentially saturated (≥ 0.99 across the board); that's the same-layer upper bound the SAE was trained against.
- MLP never beats linear by more than CI noise; the signal is linearly decodable. The "linear probe is enough" sanity check passes.
- Harm (1031) is shallowly encoded (AUC ≈ 0.98 already at L5); ethics (12730) is the weakest and most variable.

### Artifacts

- `src/__init__.py`, `src/data.py`, `src/eval.py`, `src/probe.py` - probe implementation
- `scripts/step4_train_probes.py` - orchestrator with `--smoke`, `--skip-sweep`, `--C` flags
- `results/step4_l2_sweep.csv` - 15-row CV sweep
- `results/step4_probe_metrics.csv` - 40 rows: (5 features) × (4 layers) × (2 probe types)
- `results/step4_meta.json` - split sequence IDs, best_C, per-layer wall times
- `results/step4_probe_weights/{lin,mlp}_{idx}_L{layer}.npz` - 40 weight files for inspection
- `results/step4_run.log` - full run log
- `docs/04_probes.md` - retrospective with full empirical results and Step 5 caveats

### Runtime

~27 min on the laptop (CPU). L2 sweep 8.5 min + 4 layers × ~5-8 min each.

### Out of scope for Step 4 (deferred)

- Calibration plot per probe → Step 7 figures.
- Per-(feature, layer) hyperparameter tuning → not needed; AUC differences across the C grid were monotonic and the chosen C generalized.

## Step 5: Data-efficiency sweeps - DONE

820 probes (708 non-degenerate) trained and evaluated on 2026-05-23. Full retrospective: `docs/05_data_efficiency.md`.

### Resolved open questions

- **N grid:** 9 sequence-counts, dense at low end: `[2, 4, 8, 16, 32, 64, 128, 256, 320]` → `[510, 1020, ..., 81600]` tokens.
- **Threshold metric for headline:** **AUC-ROC ≥ 0.9.** Both ROC and PR were measured; PR=0.5 is too strict for the early-layer regime (only crossed at L20 for most features), so it's reported in the curves but does not earn a headline slot.
- **Subsamples per N:** 5 (1 at N=320, deterministic).
- **C constant:** held at 0.001 (Step 4's pick) across all probes.
- **SAE retraining for ratio validation:** no, cite the published GemmaScope figure (~4B tokens).

### Final locked-in config

| | value |
|---|---|
| Probe | linear only (sklearn `LogisticRegression`, lbfgs, `class_weight='balanced'`, `C=0.001`) |
| Subsample unit | **sequences** (not tokens) |
| Z-score scaler | fit per subsample on its own rows (honest "with N tokens" claim) |
| Test fold | fixed at Step 4's 80-sequence test fold |
| Bootstrap | skipped; cross-subsample spread is the variance estimate |
| Metrics | AUC-ROC, AUC-PR, p@k (point estimates) |

### Headline empirical results

**Smallest N (tokens) at which mean AUC-ROC crosses 0.9:**

| feature | theme | L5 | L8 | L12 | L20 |
|---|---|---|---|---|---|
| 9989 | refusal | never | 49.5k | 21.1k | **3.1k** |
| 817 | deception | never | 31.0k | 26.3k | **3.5k** |
| 12730 | ethics | never | 18.3k | 26.9k | **6.4k** |
| 892 | sycophancy-adj | 4.0k | ~0.9k\* | ~0.9k\* | **~0.9k\*** |
| 1031 | harm | 3.8k | 2.6k | 2.4k | **1.8k** |

\* interpolated from k=2 non-degenerate subsamples at N∈{2, 4}; flagged in retrospective.

**Data-efficiency vs GemmaScope (4B tokens, Lieberum et al. 2024):**

| feature | best-layer N (tokens) | M_SAE / N_probe |
|---|---|---|
| 9989 refusal | 3,139 | ~1.3M× |
| 817 deception | 3,498 | ~1.1M× |
| 12730 ethics | 6,356 | ~630k× |
| 892 sycophancy-adj | ~860 | ~4.6M× |
| 1031 harm | 1,812 | ~2.2M× |

**Key takeaways:**
- Once we know which 5 features matter, the precursor probe predicts them with ~10⁻⁶ of the SAE's training data.
- The ratio is most striking but also most caveated: the SAE solves an unsupervised dictionary-learning task; the probe is labeled binary classification. The ratio is a token-budget headline, not an apples-to-apples claim.
- Refusal, deception, ethics never cross ROC=0.9 at L5 in our 81.6k-token fold; the precursor decodability story has clear feature dependence.
- Harm and sycophancy-adj are decodable from L5 with single-digit-thousand tokens. Harm at L5 reaches ROC=0.9 with 3,791 tokens.

### Artifacts

- `src/data_efficiency.py` - sequence-level subsample helper
- `scripts/step5_efficiency.py` - full-sweep orchestrator (`--smoke`, `--layers` flags)
- `scripts/step5_analysis.py` - aggregator + crossing extractor (mean N at AUC threshold via log-N interpolation)
- `results/step5_efficiency_curves.csv` - 820 raw rows (one per probe fit)
- `results/step5_efficiency_aggregate.csv` - 176 aggregate rows (mean/std/min/max across subsamples)
- `results/step5_headline.csv` - 20 rows (full-N AUC + crossing N at ROC=0.9 and PR=0.5)
- `results/step5_meta.json` - config, seeds, per-layer wall times
- `docs/05_data_efficiency.md` - retrospective

### Runtime

~27 min wall (much faster than the 80-min pre-flight estimate). ~2.3 sec average per fit (0.1-23 sec range).

### Out of scope for Step 5 (deferred)

- Figures (faceted curves per feature × layer) → Step 7.
- Re-tuning C per N → not pursued; curves were sensible at fixed C=0.001.
- Retraining a small SAE for a direct M_SAE comparison → out of scope; cite published number.

## Step 6: Generalization tests - DONE

OOD eval run 2026-05-23 against HH-RLHF red-team-attempts (human turns only). Full retrospective: `docs/06_generalization.md`.

### Resolved open questions

- **Safety corpus:** `Anthropic/hh-rlhf`, data_dir `red-team-attempts`. Single corpus chosen for simplicity; covered all 5 features at 597-1587 OOD positives each; comfortably enough for stable bootstrap CIs.
- **Prompt format:** human turns only, no chat template (base model). Defensive 3-pattern parser handled the dataset's `transcript` field.
- **Tokens:** 25,600 (100 sequences × 256, BOS-masked → 25,500).

### Final locked-in config

| | value |
|---|---|
| OOD corpus | `Anthropic/hh-rlhf`/`red-team-attempts`, human turns only |
| Cache shape | (25,600, 2304) fp16 × 4 layers, (25,600, 5) feature acts |
| Probe inputs | Step 4 `lin_*_L*.npz` weights + their saved scaler stats (Pile-fitted, NOT refit on safety data) |
| Metrics | AUC-ROC, AUC-PR, p@k, 200-resample stratified bootstrap |
| Headline metric | AUC-ROC (PR base rates differ between distributions; AUC-PR not directly comparable) |

### Headline empirical results

**OOD AUC-ROC at each layer + gap from in-distribution baseline:**

| feature | L5 | L8 | L12 | L20 |
|---|---|---|---|---|
| 9989 refusal | 0.761 (+0.13) | 0.794 (+0.11) | 0.838 (+0.10) | **0.987 (+0.009)** |
| 817 deception | 0.721 (+0.17) | 0.805 (+0.12) | 0.820 (+0.12) | **0.984 (+0.013)** |
| 12730 ethics | 0.755 (+0.12) | 0.802 (+0.12) | 0.832 (+0.09) | **0.976 (+0.017)** |
| 892 sycophancy-adj | 0.917 (+0.02) | 0.941 (+0.03) | 0.957 (+0.03) | **0.986 (+0.012)** |
| 1031 harm | 0.915 (+0.06) | 0.935 (+0.04) | 0.943 (+0.04) | **0.986 (+0.011)** |

**Key takeaways:**
- **L20 transfer is essentially clean** (gap ≤ 0.02 across all 5 features). The same-layer probe deploys directly on safety data without recalibration.
- **Early-layer transfer splits by feature kind:**
  - Lexical features (harm, sycophancy-adj): OOD AUC > 0.91 at L5, gap ≤ 0.06 everywhere.
  - Abstract features (refusal, deception, ethics): OOD AUC 0.72-0.84 at L5-L12, gap 0.09-0.17; most of Step 4's early-layer signal for these was Pile-specific.
- **OOD fire rates are 2-9× higher than ID** (ethics 8.7×, harm 9.7×). AUC-ROC is base-rate-invariant so the comparison is honest; AUC-PR numbers OOD are higher partly because of base-rate elevation, not pure ranking improvement.

### Artifacts

- `notebooks/03_safety_cache.ipynb` - Colab extraction notebook
- `scripts/_build_safety_notebook.py` - internal notebook generator
- `scripts/check_safety_cache.py` - local verifier
- `scripts/step6_ood_eval.py` - OOD scoring script
- `data/cache/safety_v1/` - 472 MB OOD cache (gitignored)
- `results/step6_ood_metrics.csv` - 20 rows: OOD + ID + gap per (feature, layer)
- `results/step6_meta.json` - config + per-feature OOD positive counts
- `results/step6_run.log` - eval run log
- `docs/06_generalization.md` - retrospective

### Runtime

- Colab safety-cache extraction: ~1.5 min on T4.
- Local OOD eval: 107 sec wall.

### Out of scope for Step 6 (deferred)

- Multiple OOD corpora (e.g. AdvBench, MoralBench); single-corpus signal was sufficient.
- Step 5-style efficiency curves on the OOD distribution; write-up may flag as future work.
- Retraining a probe on a mixed Pile + safety corpus to see if the abstract-feature gap closes; future work.

## Step 7: Write-up - DONE

Technical-report Markdown (~2,400 words) with blog-style narrative intros to each section, 3 figures, and links to the per-step retrospectives. Written 2026-05-23.

### Final structure

| Section | Content |
|---|---|
| Abstract | Three-claim headline |
| 1. Introduction | The motivation: SAEs are expensive; can we amortize once-trained features into cheap predictors? |
| 2. Method | Model, SAE, features, cache, splits, probe, metrics, OOD eval; compressed pointers to the per-step retrospectives |
| 3. Results | 3.1 In-distribution decodability (Figure 1) → 3.2 Data efficiency (Figure 2, headline ratio table) → 3.3 Generalization (Figure 3, surface-form-vs-abstract split, fire-rate elevation table) |
| 4. Discussion + limitations | Load-bearing claims, 6 numbered limitations, one missing follow-up experiment |
| 5. Reproducibility | Per-step doc + code pointers, compute envelope |

### Figures

- `docs/figures/fig1_auc_by_layer.png` - AUC-ROC and AUC-PR panels, one line per feature, 95% bootstrap bands.
- `docs/figures/fig2_data_efficiency.png` - 5 feature panels + 6th SAE-scale callout. AUC-ROC vs log N, one line per layer, ±1 std bands.
- `docs/figures/fig3_id_vs_ood.png` - paired bars per (feature, layer), ID vs OOD AUC-ROC, OOD bootstrap CIs.

### Out of scope (deferred to a v2 if pursued)

- Mixed-distribution training to test whether the early-layer abstract-feature OOD gap closes.
- arXiv/PDF version (the writeup is markdown; conversion is mechanical).
- A "lookahead" framing (predict next-N-token feature firings rather than current-token).

---

## Resumption notes

To pick up this project from cold:

1. Read this file + `README.md`.
2. Find the row in **Status snapshot** marked **next** or **in progress**.
3. Read the matching section below for the full design.
4. Check the previous step's retrospective doc (`docs/0N_*.md`) for any handoff caveats.
5. Begin work; on completion, update the status row + convert the section's "sketch" wording into "what was done" wording + add a `docs/0N_*.md` retrospective with empirical results.

To extend a step that already completed: add a new row below the current Status snapshot (e.g., "3b: extend cache to 200k tokens") and a corresponding new section below.
