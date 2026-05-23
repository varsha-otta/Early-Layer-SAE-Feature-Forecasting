# Implementation plan: safety-sae-feature-forecasting

Single living plan covering all seven steps in the README. Resumable: each step has a status, design notes, and pointers to its artifacts. Pick up from any "next" or "in progress" row.

**Last updated:** 2026-05-23 (Step 3 done; cache verified locally; Step 4 next).

## Research question

Can a small classifier trained on **early-layer** Gemma-2-2B residual stream activations predict whether a **late-layer** safety-flavored SAE feature will fire at the same token position — and crucially, how does the probe's data efficiency compare to training the late-layer SAE itself?

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
| 4 | Probe training + per-feature evaluation | **next** | (planned) `src/probe.py`, `results/step4_probe_metrics.csv` |
| 5 | Data-efficiency sweeps | pending | (planned) `src/data_efficiency.py`, `results/step5_efficiency_curves.csv` |
| 6 | Generalization tests | pending | (planned) safety-prompts cache + `results/step6_generalization.csv` |
| 7 | Write-up | pending | (planned) `docs/07_writeup.md` |

Workflow: per the project's pause-per-step convention, each step is completed and reviewed before the next begins.

---

## Step 1: Env + smoke test — DONE

Goal was to verify the activation-extraction pipeline end-to-end on Colab T4.

What was verified:
- Gemma-2-2B loads via `transformers` (not `transformer_lens`, which OOMs Colab free's host RAM) with `torch.bfloat16` + `device_map=device` + `low_cpu_mem_usage=True`.
- GemmaScope SAE loads at layer 20, width 16k, canonical.
- Manual `register_forward_hook` on `model.model.layers[L]` (output tuple's first element) captures the post-block residual stream — the activation site GemmaScope residual SAEs were trained on.
- `sae.encode(resid_20)` returns sane sparse activations (~tens to hundreds of nonzero features per token).

Artifact: `notebooks/01_smoke_test.ipynb`.

## Step 2: Target feature selection — DONE

Goal was to pick 3-5 safety-flavored SAE features at Gemma-2-2B layer 20 for use as probe targets.

Process:
- Queried Neuronpedia's `/api/explanation/search` for 20 safety-related keywords (refusal, deception, sycophancy, harm, ethics, hedging, etc.). 79 unique candidates surfaced; 73 in the 0.05-1% firing-rate band.
- For the top 10 candidates, fetched per-feature top-activating contexts to verify the auto-interp labels (3 of 10 turned out mislabeled or too narrow — important catch).
- Picked 5 features for diversity across safety dimensions.

Picks: `[9989 refusal, 817 deception, 12730 ethics, 892 sycophancy-adjacent, 1031 harm]`, firing rates 0.28%-0.76%.

Caveats: no clean sycophancy feature exists in the base model (would need Gemma-2-2B-it); no clean "controversial topic" or distinct "harmful-content recognition" feature surfaced.

Artifacts:
- `docs/02_feature_selection.md` — full record (queries, verification table, decision rationale, swap candidates)
- `data/target_features.json` — machine-readable handoff to Step 3
- `data/neuronpedia_search_raw.json`, `data/shortlist_v1.json` — raw and verified data
- `scripts/step2_neuronpedia_search.py` — idempotent reproduction

## Step 3: Activation cache extraction (Colab) — DONE

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

## Step 4: Probe training + per-feature evaluation — PENDING

### Sketch (design will be locked in when Step 3 is done)

**Probe:** linear (logistic regression) primary, 2-layer MLP as sanity check (per README). One probe per (feature, layer) → 5 features × 4 layers = **20 probes**.

**Label:** binary fire = `feature_acts > 0`. The SAE's own ReLU defines "fires"; no extra threshold.

**Split:** 80/20 by **sequence index** (not token index) to avoid leaking adjacent-position information. Fixed seed.

**Metrics:** AUC-ROC (primary), AUC-PR (because positives are rare: 0.28-0.76%), precision@k, calibration plot. Bootstrap CIs (100 resamples) on the test fold.

**Regularization:** L2 strength swept on a held-out subset of the train fold.

**Implementation:** scikit-learn `LogisticRegression` for the linear probe; tiny PyTorch MLP for sanity. CPU-only; should be seconds per probe.

**Outputs:** `results/step4_probe_metrics.csv` (one row per (feature, layer, probe_type) with point + CI), `results/step4_probe_weights/` (one `.npz` per probe for later inspection).

### Open questions for Step 4 planning

- Should BOS tokens be masked from train/eval? Leaning yes (cleaner).
- Single shared train/test split across all probes, or per-feature splits? Leaning shared (simpler interpretation).

## Step 5: Data-efficiency sweeps — PENDING

### Sketch

**N sweep:** log-spaced, ~8 points: `[500, 1k, 2k, 5k, 10k, 20k, 50k, 100k]`.

**For each (feature, layer, N):** subsample N from the train fold, train linear probe, evaluate on the (fixed) test fold. Repeat with 5 random subsamples for variance.

**Baseline:** GemmaScope's published SAE training data size (from the GemmaScope paper — to be looked up in Step 5; expected ~4B tokens). We report the ratio: `M_SAE / N_probe@AUC_target`.

**Output:** `results/step5_efficiency_curves.csv`, faceted plot per (feature, layer).

### Open questions

- Do we re-train SAEs ourselves on subsets to validate the M_SAE number? Default: no (out of scope, too expensive). Cite published number.
- What AUC threshold defines "the probe works"? Suggest 0.9 (refine after Step 4 results).

## Step 6: Generalization tests — PENDING

### Sketch

Train probe on Pile-10k cache (Step 3), evaluate on a held-out safety-prompt distribution.

**Safety-prompt corpus:** to decide. Candidates: AdvBench, HarmBench, a slice of Anthropic HH-RLHF (red-team subset), or a hand-curated set of refusal-flavored prompts. Will need ~10k-50k tokens. Built as a second activation cache (`data/cache/safety_v1/`) following the Step 3 pipeline.

**Comparison:** in-distribution AUC (Pile test fold) vs out-of-distribution AUC (safety cache). Gap is the generalization story.

### Open questions

- Which safety corpus best matches our 5 features? Refusal/harm picks point toward red-team / AdvBench-like; ethics points toward MoralBench or similar. May want multiple eval sets.
- Tokenize at prompt-only or prompt+response? Probably prompt-only (matches the precursor framing — we want to detect intent before the response).

## Step 7: Write-up — PENDING

### Sketch

**Format:** Markdown technical report, ~2-3 pages, in `docs/07_writeup.md`. Could later port to a blog post or arXiv-style PDF.

**Figures:**
1. Headline: probe AUC vs early layer (5/8/12/20), faceted by feature.
2. Data-efficiency curves: AUC vs N, one panel per feature, with SAE training-set marker for scale.
3. Generalization: in-distribution vs OOD AUC per feature.

**Sections:** intro/motivation → method → results → limitations → reproducibility (links to notebooks + scripts).

---

## Resumption notes

To pick up this project from cold:

1. Read this file + `README.md`.
2. Find the row in **Status snapshot** marked **next** or **in progress**.
3. Read the matching section below for the full design.
4. Check the previous step's retrospective doc (`docs/0N_*.md`) for any handoff caveats.
5. Begin work; on completion, update the status row + convert the section's "sketch" wording into "what was done" wording + add a `docs/0N_*.md` retrospective with empirical results.

To extend a step that already completed: add a new row below the current Status snapshot (e.g., "3b: extend cache to 200k tokens") and a corresponding new section below.
