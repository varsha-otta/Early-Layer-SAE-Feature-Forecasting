# Step 6: Generalization tests (web text → safety prompts)

**Status**: done. 20 OOD evaluations (5 features × 4 layers) of Step 4's full-fold linear probes on a 25,500-token safety cache.

**Date**: 2026-05-23.

## Goal

Step 4 showed that linear probes trained on Pile-10k achieve high in-distribution AUC at every layer. Step 6 asks the natural follow-up: **do those probes work on out-of-distribution data?** Specifically; when we feed Gemma-2-2B safety-flavored prompts (Anthropic HH-RLHF `red-team-attempts` subset, human turns only; Ganguli et al. 2022; Bai et al. 2022) and score the same Step 4 probes against the SAE's actual feature firings on those prompts, how much AUC do we lose?

## Headline result

**OOD AUC-ROC, ID baseline (Step 4 test fold), and gap (ID − OOD), linear probes:**

| feature | theme | L5 | L8 | L12 | L20 |
|---|---|---|---|---|---|
| 9989 | refusal | 0.761 (gap +0.13) | 0.794 (+0.11) | 0.838 (+0.10) | **0.987 (+0.009)** |
| 817 | deception | 0.721 (+0.17) | 0.805 (+0.12) | 0.820 (+0.12) | **0.984 (+0.013)** |
| 12730 | ethics | 0.755 (+0.12) | 0.802 (+0.12) | 0.832 (+0.09) | **0.976 (+0.017)** |
| 892 | sycophancy-adj | 0.917 (+0.02) | 0.941 (+0.03) | 0.957 (+0.03) | **0.986 (+0.012)** |
| 1031 | harm | 0.915 (+0.06) | 0.935 (+0.04) | 0.943 (+0.04) | **0.986 (+0.011)** |

**Headline:** Layer-20 probes transfer near-perfectly; OOD AUC-ROC ≥ 0.976 across all 5 features, with gap ≤ 0.017 in every case. **Early-layer transfer is feature-dependent**: the abstract features assembled across many layers (refusal, deception, ethics) lose 0.10-0.17 AUC at L5-L12, while the surface-form features tied to specific words/tokens (harm, sycophancy-adj) keep AUC > 0.9 at every layer with gap ≤ 0.06.

**AUC-PR is actually higher OOD than ID for several features**, because positives are 2-10× more common on safety prompts (see "fire-rate differences" below). The probe ranks roughly as well, and there are more positives among the top-ranked tokens, so the precision/recall curve area is more favorable. The right interpretation: AUC-ROC is the metric that lets us claim transfer; AUC-PR is best read alongside the OOD fire rate.

## What the numbers say

1. **L20 transfer is the clean result.** With gaps under 0.02, the same-layer probe is essentially distribution-invariant for our 5 features. This is what you'd hope for if the SAE-encoded direction is genuinely the "this token causes feature F" direction rather than a distribution-specific shortcut. It's also the case that L20 fire patterns on safety data are dominated by the same surface-form and semantic content Pile data is; the SAE encoder's `W_enc[F]` projection just doesn't change behavior across distributions.

2. **Early-layer transfer splits by feature kind.** Two clusters:
   - **Surface-form (harm, sycophancy-adj):** OOD AUC-ROC > 0.91 even at L5. Gaps 0.02-0.06. The early-layer representation that the probe latches onto for these features is something stable across distributions; likely shallow token-level cues.
   - **Abstract (refusal, deception, ethics):** OOD AUC-ROC 0.72-0.84 at L5-L12. Gaps 0.09-0.17. Most of the early-layer signal Step 4 measured for these features is **distribution-specific.** The Pile-trained probe latches onto something correlated with the SAE's "refusal precursor" on web text but not on actual safety prompts.

3. **Refusal/deception/ethics need depth for transfer.** L5 → L8 → L12 closes the gap monotonically but slowly. Only by L20 does the gap become negligible. This is consistent with the picture that abstract concepts assemble across many layers and the early-layer "signal" is partly an emergent correlate that doesn't survive distribution shift.

4. **Fire-rate elevation on safety prompts is real:**

   | feature | Pile-10k | HH-RLHF red-team | ratio |
   |---|---|---|---|
   | 9989 refusal | 1.01% | 2.33% | 2.3× |
   | 817 deception | 1.39% | 4.78% | 3.4× |
   | 12730 ethics | 0.57% | 4.96% | **8.7×** |
   | 892 sycophancy-adj | 0.81% | 3.41% | 4.2× |
   | 1031 harm | 0.64% | 6.20% | **9.7×** |

   Ethics and harm fire ~9× more often on red-team prompts; exactly the direction you'd expect. This means **AUC-PR comparisons across distributions are not directly meaningful** (different base rates), but AUC-ROC remains comparable.

5. **Probe weights from Step 4 are genuinely "what we trained"; no leakage.** The OOD eval reuses the saved `lin_*_L*.npz` files from Step 4 verbatim (coefficients, intercepts, and the per-(feature, layer) scaler stats fit on the Pile train fold). No retraining, no recalibration, no scaler refit on safety data. Any AUC reduction is honest OOD degradation.

## Interpretation: the precursor story now reads differently

The Step 4 + Step 5 picture was: "we can predict feature firing from early-layer activations with surprisingly little data." Step 6 sharpens this:

- **For surface-form features (harm, sycophancy-adj):** the early-layer precursor signal is robust and transfers. The probe is picking up on something general.
- **For abstract features (refusal, deception, ethics):** the early-layer precursor signal is *Pile-specific*. The probe works in-distribution but partially fails to transfer. The "precursor" is real for the trained distribution but not a distribution-invariant property of the network.
- **At layer 20:** transfer is excellent for every feature. If the goal is a deployable feature predictor that generalizes, layer-20 same-layer probes are the answer; if the goal is a true "early warning" detector, the early-layer probe needs to be retrained or pooled across distributions before deployment.

## Config (single source of truth)

| | value |
|---|---|
| OOD corpus | `Anthropic/hh-rlhf` (data_dir: `red-team-attempts`) |
| Turns used | human only (red-teamer prompts; no model responses) |
| Tokenization | concatenated human turns, no chat template, BOS-prefixed 256-token sequences |
| Tokens cached | 100 sequences × 256 = 25,600 (25,500 after BOS mask) |
| Layers cached | `[5, 8, 12, 20]` (same as Step 3 cache) |
| Feature labels | SAE-encoded at L20 for the 5 target features (same as Step 3 cache) |
| Probes evaluated | Step 4 linear probes from `results/step4_probe_weights/lin_*_L*.npz` (saved coefs + intercepts + scaler) |
| Scaler | Pile train-fold mean/std (Step 4's saved stats; **not refit on safety data**) |
| Metrics | AUC-ROC, AUC-PR, p@k, with 200-resample stratified bootstrap 95% CIs |

## Runtime

- Colab safety-cache extraction: ~1.5 min on T4 (user-run, separate notebook).
- Local OOD eval: **107 sec wall** (one layer at a time, ~5 sec per probe, including bootstrap).

## Risks vs outcome

| Risk | Outcome |
|---|---|
| HH-RLHF transcript format mismatch | Defensive parser with 3-pattern fallback; Cell 14 in `notebooks/03_safety_cache.ipynb` reports per-transcript turn counts. Worked first try. |
| Too few positives for OOD AUC-PR | Fire rates on safety prompts are *higher* than Pile (2-10×), giving 597-1587 positives per feature in 25.5k tokens. Plenty for stable bootstrap. |
| Scaler mismatch (Pile vs safety) | Intentional: we use the Pile scaler so the OOD eval matches "deploy this Step 4 probe on new data without recalibration." Residual stats differ between distributions but the linear projection is unchanged. |
| Float16 clipping on safety residuals | Verified: max abs ≤ 2720 at L20 (well below the fp16 ceiling of ~65504). Notable that safety-prompt residuals are larger than Pile residuals; red-team prompts may produce more "extreme" activations; but not a problem. |
| Step 4 weights file format drift | Schema verified: `coef`, `intercept`, `scaler_mean`, `scaler_scale`, `C`, `n_iter`. Matches `step6_ood_eval.py` expectations. |

## Differences vs the plan's sketch

| Item | Plan | Actual |
|---|---|---|
| Safety corpus | "to decide; AdvBench / HarmBench / HH-RLHF red-team / hand-curated" | `Anthropic/hh-rlhf` red-team-attempts (single corpus, human turns only) |
| Tokens | "~10k-50k" | 25,600 (in the middle of the range; chose during planning) |
| Comparison | "in-distribution AUC vs out-of-distribution AUC" | Both reported per (feature, layer) with explicit gap column |
| MoralBench / ethics-specific corpus | "may want multiple eval sets" | Not pursued; ethics happens to fire 8.7× more on red-team data than Pile, so we got a strong signal anyway |
| Prompt-only vs prompt+response | "probably prompt-only" | Confirmed: human-turns only |

## Caveats for Step 7 (write-up)

1. **The "early-layer precursor" headline needs updating.** Step 4 read as "all features decodable from L5." Step 6 says "decodable in-distribution, but **abstract features don't transfer**." The writeup should reflect both findings honestly.

2. **The data-efficiency headline (Step 5) is unchanged**: it's an in-distribution result. M_SAE/N_probe at ID still stands. But OOD efficiency would be a different (likely worse) curve; flag this as out of scope or "next steps."

3. **The OOD AUC-PR numbers above the ID PR-AUC are not an "improvement"**; they reflect the higher safety-data base rate. The writeup should report them with the OOD positive count to avoid misleading the reader.

4. **L20 is the deployment story.** If we wanted to ship a feature-firing predictor today, the layer-20 probe with the Step 4 weights would just work on red-team data with gap ≤ 0.02. For early warning, more work is needed.

5. **Only one OOD corpus tested.** Other safety distributions (AdvBench-style adversarial, model-generated responses, in-the-wild user data) may show different gaps. The single-corpus result is suggestive but not definitive.

## Saved artifacts

| File | Content |
|---|---|
| `notebooks/03_safety_cache.ipynb` | Colab safety-corpus extraction (parallel to 02_activation_cache.ipynb) |
| `scripts/_build_safety_notebook.py` | Internal: regenerate the notebook from source |
| `scripts/check_safety_cache.py` | Local verifier for the safety cache (shapes, fire rates vs Pile) |
| `scripts/step6_ood_eval.py` | OOD scoring: load Step 4 weights, apply to safety residuals, write metrics |
| `data/cache/safety_v1/` | OOD cache (residuals × 4 layers + feature_acts + token_ids + metadata) |
| `results/step6_ood_metrics.csv` | 20 rows: per-(feature, layer) OOD + ID + gap |
| `results/step6_meta.json` | Run config + OOD positive counts |
| `results/step6_run.log` | Eval run log |

## Reproduction

```bash
# 1) Build the cache on Colab (T4 GPU)
# Upload notebooks/03_safety_cache.ipynb to Colab and run all cells.

# 2) Download safety-sae-cache/safety_v1/ from Drive to data/cache/safety_v1/

# 3) Verify locally
./safety-env/Scripts/python.exe scripts/check_safety_cache.py

# 4) Run OOD eval
./safety-env/Scripts/python.exe scripts/step6_ood_eval.py
```
