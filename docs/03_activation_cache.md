# Step 3: Activation cache extraction

**Status**: done. 1.89 GB cache at `data/cache/v1/`, verified by `scripts/check_activation_cache.py`.

**Date**: 2026-05-23.

## Goal

Build a float16 activation cache from Gemma-2-2B on Pile-10k — early-layer residual streams plus the 5 target SAE feature activations at layer 20 — sized to fit the laptop memory budget for Step 4 probe training.

## What was extracted

| File | Shape | dtype | Size |
|---|---|---|---|
| `resid_layer_5.npy`  | (102400, 2304) | float16 | 471.9 MB |
| `resid_layer_8.npy`  | (102400, 2304) | float16 | 471.9 MB |
| `resid_layer_12.npy` | (102400, 2304) | float16 | 471.9 MB |
| `resid_layer_20.npy` | (102400, 2304) | float16 | 471.9 MB |
| `feature_acts.npy`   | (102400, 5)    | float16 | 1.02 MB |
| `token_ids.npy`      | (102400,)      | int32   | 0.41 MB |
| `metadata.json`      | -              | -       | ~1.4 KB |
| **Total**            |                |         | **~1.89 GB** |

Layout: flat row-major, with `seq_idx = token_idx // 256`. So a `(N, D)` array slices into 400 sequences of length 256 each; sequence-level splits in Step 4 work via `np.arange(400)` permutations on the sequence axis.

## Config snapshot

Per `metadata.json`:

- **Model**: `google/gemma-2-2b`, bf16
- **Corpus**: `NeelNanda/pile-10k` pinned to commit `127bfedc...` (400 documents consumed)
- **Tokens**: 400 sequences × 256 = 102,400; sequences are BOS-prefixed with 255 packed tokens from Pile (cross-document spans permitted, matching GemmaScope's training distribution)
- **Layers cached**: `[5, 8, 12, 20]` (early × 3, late × 1 as same-layer upper bound)
- **Target features at layer 20**: `[9989, 817, 12730, 892, 1031]`
- **SAE**: `gemma-scope-2b-pt-res-canonical`, `layer_20/width_16k/canonical`
- **Activation site**: `model.model.layers[L]` output tuple's `[0]` (post-block residual stream)
- **Batch size**: 8 (50 batches)
- **Runtime**: Colab T4 (Linux 6.6.122 / glibc 2.35)
- **Library versions**: torch 2.10.0+cu128, transformers 5.0.0, sae_lens 6.44.0, datasets 4.0.0, numpy 2.0.2

## Empirical fire rates

Fraction of the 102,400 tokens where each target feature's encoded activation is strictly positive (JumpReLU SAE → post-threshold activation):

| idx | theme | empirical (Pile-10k) | Neuronpedia reported | ratio |
|---|---|---|---|---|
| 9989  | refusal           | 1.011% | 0.746% | 1.35x |
| 817   | deception         | 1.389% | 0.763% | 1.82x |
| 12730 | ethics            | 0.567% | 0.485% | 1.17x |
| 892   | sycophancy-adj    | 0.807% | 0.279% | **2.89x** |
| 1031  | harm              | 0.641% | 0.481% | 1.33x |

All five fire more often on our Pile-10k sample than Neuronpedia's reported rate, mostly within 1.2–1.8×. The outlier is **892 (sycophancy-adj) at 2.89×** — interesting but not alarming. Two plausible explanations:

1. Neuronpedia computes `frac_nonzero` over a different corpus (likely the GemmaScope training distribution, sampled differently), so absolute comparison is noisy by construction.
2. Pile-10k is web/news/discussion-heavy text, which oversamples constructions matching "insincere or exaggerated language" relative to Neuronpedia's sample.

The ratio is well within the verifier's [0.25, 4.0] warning band (no warnings fired locally). Sample-size note: at 102,400 tokens, the standard error on a rate of 0.005 is ~0.0002, so ~5% relative — much smaller than the gap.

For Step 4 the practical consequence is positive label counts in the **580–1420 range** across the 5 features in the full cache (after the 80/20 sequence split: ~470–1135 train positives, ~115–285 test positives). Plenty for logistic regression; PR-AUC will matter more than ROC-AUC at these positive rates.

## Differences vs the pre-flight design

None worth flagging. The plan locked in the config in advance and the run matched it byte-for-byte: same 7 files, same shapes, same dtypes, sizes match the plan's table exactly (471 MB per resid layer, 1 MB features, 400 KB tokens).

## Risks the plan called out — outcome

| Risk | Outcome |
|---|---|
| Drive lacks ~2 GB free | Cell 5 passed the disk check; cache fits comfortably. |
| HF Gemma-2-2B license not accepted | Same auth path as smoke test; worked. |
| SAE encode dtype mismatch | Explicit `.to(sae.W_enc.dtype)` cast in extraction cell; no errors. |
| Float16 clipping | Verifier confirmed all four residual arrays have `|x|_max` well below the fp16 ceiling (per local run; metadata didn't persist this stat — could be added in v2). |
| Pile-10k content drift on HF | Pinned to commit `127bfedc...`; recorded in `metadata.json`. |
| User re-runs hook-registration cell | Hooks cleanup at top of extraction cell handles it. |

## Caveats for Step 4

- **Sequence-level split, not token-level.** The plan calls for an 80/20 split by sequence index to prevent positional leakage. The cache layout (flat row-major, 256 tokens per sequence) is set up for this.
- **BOS positions.** Position `k * 256` (k ∈ {0, ..., 399}) is always BOS — verified by the uniform-BOS-column check. The leaning is to mask these from train/eval since they carry no document-specific signal; this remains a Step 4 open question.
- **Feature column order matches `data/target_features.json`.** Column `i` of `feature_acts.npy` is feature index `FEATURE_INDICES[i]` from the notebook config: `[9989, 817, 12730, 892, 1031]`.
- **All layers cached, not just one at a time.** Step 4 should mmap each layer separately (per the local-RAM rule) and never hold more than one fp32 conversion in memory at once.

## Reproduction

To re-run from scratch:

1. `notebooks/02_activation_cache.ipynb` on Colab (T4 GPU), accepting the Gemma license.
2. ~5 min after warm-up; writes to `safety-sae-cache/v1/` in Drive.
3. Download to `data/cache/v1/` locally.
4. `python scripts/check_activation_cache.py`.

Cache files are gitignored under the `data/` rule; only `metadata.json` is small enough to consider committing if we want a record of what the cache contained, but currently it's also ignored.

## Saved artifacts

| File | Content |
|---|---|
| `data/cache/v1/resid_layer_{5,8,12,20}.npy` | Post-block residual streams, float16 |
| `data/cache/v1/feature_acts.npy` | Layer-20 SAE activations for the 5 target features, float16 |
| `data/cache/v1/token_ids.npy` | Token ids (BOS at positions 0, 256, 512, ...), int32 |
| `data/cache/v1/metadata.json` | Config + empirical fire rates + library versions |
| `notebooks/02_activation_cache.ipynb` | Colab extraction pipeline (run this to reproduce) |
| `scripts/check_activation_cache.py` | Local verifier (run after download) |
