# safety-sae-feature-forecasting

Forecasting safety-relevant sparse autoencoder (SAE) features in Gemma-2-2B from earlier-layer activations.

## Research question

Can a small classifier trained on early-layer residual stream activations predict whether a *late-layer* SAE feature - chosen to be safety-flavored (refusal, sycophancy, deception, harm-recognition, hedging) - will fire at the same token position? And: how data-efficient is this precursor probe compared to training the late-layer SAE itself?

Headline target claim:

> With N tokens of probe-training data, we predict feature F at layer L_late from layer L_early activations at AUC X - versus M ≫ N tokens needed for the GemmaScope SAE at L_late to surface F as a coherent feature.

Empirical result (Step 5, Pile-10k, AUC-ROC ≥ 0.9 threshold):

> Once GemmaScope's ~4B-token SAE has surfaced a layer-20 safety feature, predicting whether that feature fires from **layer-20 activations** takes ~0.9k-6.4k tokens — **about 10⁵-10⁶× less data** than the SAE itself needed. From mid-network (layer 8) the same threshold is reached in 1-50k tokens with strong feature-by-feature variance; from early layers (5) only the most lexically-salient features (harm, sycophancy-adj) cross 0.9 within our 81.6k-token test budget.

## Approach

| | |
|---|---|
| Model | `google/gemma-2-2b` (base, 26 layers) |
| SAEs | GemmaScope residual-stream, width 16k, canonical |
| Early layers | L_early ∈ {5, 8, 12} |
| Late layer | L_late = 20 (also cached as a same-layer upper-bound probe) |
| Target features | 5 safety-flavored SAE features at layer 20, picked via Neuronpedia + manual verification (`data/target_features.json`) |
| Probes | Linear (logistic regression) primary; 2-layer MLP as sanity check |

## Compute split

- **Activation extraction** (Step 3): Google Colab free tier (T4 GPU). Notebooks in `notebooks/`.
- **Probe training + analysis** (Steps 4–6): local CPU. Code in `src/`.

Rationale: Gemma-2-2B forward passes don't fit comfortably in 8 GB system RAM, but a cached activation tensor (~hundreds of MB) is fine to download and probe locally.

## Status

**Step 7 of 7: write-up: next.** See `docs/implementation_plan.md` for the full per-step design.

Step 5 headline (tokens to hit AUC-ROC ≥ 0.9 vs ~4B-token GemmaScope SAE training corpus):

| feature | best layer | N_probe (tokens) | M_SAE / N_probe |
|---|---|---|---|
| 9989 refusal | L20 | 3.1k | ~1.3M× |
| 817 deception | L20 | 3.5k | ~1.1M× |
| 12730 ethics | L20 | 6.4k | ~630k× |
| 892 sycophancy-adj | L8/L20 | ~0.9k | ~4.6M× |
| 1031 harm | L20 | 1.8k | ~2.2M× |

Step 6 generalization (Pile-trained probes → HH-RLHF red-team prompts, OOD AUC-ROC and gap from in-distribution):

| feature | L5 | L8 | L12 | L20 |
|---|---|---|---|---|
| refusal | 0.761 (+0.13) | 0.794 (+0.11) | 0.838 (+0.10) | **0.987 (+0.009)** |
| deception | 0.721 (+0.17) | 0.805 (+0.12) | 0.820 (+0.12) | **0.984 (+0.013)** |
| ethics | 0.755 (+0.12) | 0.802 (+0.12) | 0.832 (+0.09) | **0.976 (+0.017)** |
| sycophancy-adj | 0.917 (+0.02) | 0.941 (+0.03) | 0.957 (+0.03) | **0.986 (+0.012)** |
| harm | 0.915 (+0.06) | 0.935 (+0.04) | 0.943 (+0.04) | **0.986 (+0.011)** |

**L20 probes transfer near-perfectly (gap ≤ 0.02 across all features); early-layer transfer is feature-dependent — abstract features (refusal, deception, ethics) lose 0.10-0.17 AUC at L5-L12, while lexical features (harm, sycophancy-adj) hold AUC > 0.9 everywhere.** See `docs/04_probes.md`, `docs/05_data_efficiency.md`, `docs/06_generalization.md`.

| # | Step | Status |
|---|---|---|
| 1 | Env + smoke test | done |
| 2 | Target feature selection (Neuronpedia browse) | done. Picks: `[9989, 817, 12730, 892, 1031]`; see `docs/02_feature_selection.md` |
| 3 | Activation cache extraction (Colab) | done. 102,400 tokens × 4 layers cached at `data/cache/v1/`; retrospective: `docs/03_activation_cache.md` |
| 4 | Probe training + per-feature evaluation | done. 40 probes; AUC tables in `docs/04_probes.md`, metrics in `results/step4_probe_metrics.csv` |
| 5 | Data-efficiency sweeps | done. 820 probes; headline in `docs/05_data_efficiency.md`, curves in `results/step5_efficiency_curves.csv` |
| 6 | Generalization tests (web → safety prompts) | done. OOD eval on HH-RLHF red-team; full table in `docs/06_generalization.md`, metrics in `results/step6_ood_metrics.csv` |
| 7 | Write-up | next |

## Layout

```
.
├── README.md
├── requirements.txt                  # local probing deps (CPU-only)
├── .gitignore
├── docs/
│   ├── implementation_plan.md        # living plan: per-step design + status
│   ├── 02_feature_selection.md       # Step 2 research record (queries, verification, decision)
│   ├── 03_activation_cache.md        # Step 3 retrospective (cache empirics, fire rates)
│   ├── 04_probes.md                  # Step 4 retrospective (AUC tables, MLP vs linear)
│   ├── 05_data_efficiency.md         # Step 5 retrospective (N-sweep curves, M_SAE/N_probe ratios)
│   └── 06_generalization.md          # Step 6 retrospective (OOD AUC, ID/OOD gap by layer)
├── notebooks/
│   ├── 01_smoke_test.ipynb           # Colab; Step 1: model + SAE + hooks pipeline
│   ├── 02_activation_cache.ipynb     # Colab; Step 3: ~1.89 GB Pile-10k activation cache
│   └── 03_safety_cache.ipynb         # Colab; Step 6: ~470 MB HH-RLHF red-team activation cache
├── scripts/
│   ├── check_local_env.py            # verifies local env imports
│   ├── check_activation_cache.py     # Step 3: verify downloaded Pile cache locally
│   ├── check_safety_cache.py         # Step 6: verify downloaded safety cache locally
│   ├── step2_neuronpedia_search.py   # Step 2: reproducible Neuronpedia search + verification
│   ├── step4_train_probes.py         # Step 4: trains all 40 probes, writes results/step4_*
│   ├── step5_efficiency.py           # Step 5: N-sweep across (feature, layer, N, subsample)
│   ├── step5_analysis.py             # Step 5: aggregate + headline N-at-threshold extractor
│   ├── step6_ood_eval.py             # Step 6: score Step 4 probes on safety cache, write OOD CSV
│   └── _build_safety_notebook.py     # Step 6: internal regenerator for 03_safety_cache.ipynb
├── src/                              # probe training + analysis (Step 4 onward)
│   ├── data.py                       # cache loaders, sequence split, BOS mask, z-scoring
│   ├── eval.py                       # AUC-ROC, AUC-PR, precision@k, bootstrap CIs
│   ├── probe.py                      # linear + MLP probes
│   └── data_efficiency.py            # Step 5: sequence-level subsample helper, N grid
├── data/                             # cached activations + search results (gitignored)
│   ├── target_features.json          # committed picks → handoff to Step 3
│   ├── neuronpedia_search_raw.json
│   ├── shortlist_v1.json
│   └── cache/v1/                     # Step 3 output, downloaded from Colab/Drive
└── results/                          # tables, plots, writeup
```

## Getting started

### 1. Local environment (for probe training)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python scripts/check_local_env.py
```

### 2. Colab smoke test (for activation extraction pipeline)

1. Upload `notebooks/01_smoke_test.ipynb` to https://colab.research.google.com
2. Runtime → Change runtime type → **T4 GPU**
3. Accept the Gemma-2-2B license at https://huggingface.co/google/gemma-2-2b
4. Generate a Hugging Face read token at https://huggingface.co/settings/tokens
5. Run cells top to bottom

If every cell completes and the final cell prints non-zero SAE feature activations per token, Step 1 is complete.

**Backup if Colab's free GPU is unavailable**: same notebook runs on Kaggle (30 GPU-hours/week, T4 or P100).

### 3. Colab activation cache (Step 3)

1. Upload `notebooks/02_activation_cache.ipynb` to Colab (T4 GPU runtime).
2. Run all cells. ~5 min after warm-up; writes ~1.89 GB to `safety-sae-cache/v1/` in your Drive.
3. Download `safety-sae-cache/v1/` from Drive to `data/cache/v1/` in this repo.
4. Verify locally: `python scripts/check_activation_cache.py`.

### 4. Colab safety-corpus cache (Step 6)

1. Upload `notebooks/03_safety_cache.ipynb` to Colab (T4 GPU runtime).
2. Run all cells. ~1.5 min after warm-up; writes ~470 MB to `safety-sae-cache/safety_v1/` in your Drive (Anthropic/hh-rlhf red-team-attempts, human turns only, 25,600 tokens).
3. Download `safety-sae-cache/safety_v1/` from Drive to `data/cache/safety_v1/` in this repo.
4. Verify locally: `python scripts/check_safety_cache.py`.
5. Run the OOD eval: `python scripts/step6_ood_eval.py` → writes `results/step6_ood_metrics.csv`.

Full step design (config, memory plan, risks): `docs/implementation_plan.md`.
