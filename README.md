# safety-sae-feature-forecasting

Forecasting safety-relevant sparse autoencoder (SAE) features in Gemma-2-2B from earlier-layer activations.

## Research question

Can a small classifier trained on early-layer residual stream activations predict whether a *late-layer* SAE feature - chosen to be safety-flavored (refusal, sycophancy, deception, harm-recognition, hedging) - will fire at the same token position? And: how data-efficient is this precursor probe compared to training the late-layer SAE itself?

Headline target claim:

> With N tokens of probe-training data, we predict feature F at layer L_late from layer L_early activations at AUC X - versus M ≫ N tokens needed for the GemmaScope SAE at L_late to surface F as a coherent feature.

## Approach

| | |
|---|---|
| Model | `google/gemma-2-2b` (base, 26 layers) |
| SAEs | GemmaScope residual-stream, width 16k |
| Early layers | L_early ∈ {5, 8, 12} |
| Late layers | L_late ∈ {18, 22} |
| Target features | 3–5 safety-flavored SAE features, picked via Neuronpedia + manual verification |
| Probes | Linear (logistic regression) primary; 2-layer MLP as sanity check |

## Compute split

- **Activation extraction** (Step 3): Google Colab free tier (T4 GPU). Notebooks in `notebooks/`.
- **Probe training + analysis** (Steps 4–6): local CPU. Code in `src/`.

Rationale: Gemma-2-2B forward passes don't fit comfortably in 8 GB system RAM, but a cached activation tensor (~hundreds of MB) is fine to download and probe locally.

## Status

**Step 1 of 7: environment setup + smoke test (current).**

| # | Step |
|---|---|
| 1 | Env + smoke test |
| 2 | Target feature selection (Neuronpedia browse) |
| 3 | Activation cache extraction (Colab) |
| 4 | Probe training + per-feature evaluation |
| 5 | Data-efficiency sweeps |
| 6 | Generalization tests (web → safety prompts) |
| 7 | Write-up |

## Layout

```
.
├── README.md
├── requirements.txt           # local probing deps (CPU-only)
├── .gitignore
├── notebooks/
│   └── 01_smoke_test.ipynb    # Colab; verifies model + SAE + hooks pipeline
├── scripts/
│   └── check_local_env.py     # verifies local env imports
├── src/                       # probe training + analysis (Steps 4–6, created later)
├── data/                      # cached activation tensors (gitignored)
└── results/                   # tables, plots, writeup
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
