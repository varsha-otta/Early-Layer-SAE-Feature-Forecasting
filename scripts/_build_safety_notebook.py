"""Internal: emit notebooks/03_safety_cache.ipynb (Step 6 safety-corpus extraction).

Mirrors notebooks/02_activation_cache.ipynb but pointed at Anthropic/hh-rlhf
red-team-attempts (human turns only, prompt-only). Run this script once when
the notebook needs to be rebuilt; do NOT run it on Colab.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "03_safety_cache.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md-{abs(hash(text)) % 10_000_000:07d}",
        "metadata": {},
        "source": [line + "\n" for line in dedent(text).strip("\n").split("\n")],
    }


def code(text: str) -> dict:
    src = dedent(text).strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"code-{abs(hash(text)) % 10_000_000:07d}",
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src],
    }


cells = []

# 0
cells.append(md("""
# Early-Layer-SAE-Feature-Forecasting: Step 6 - safety-corpus activation cache

Builds an **out-of-distribution** activation cache from Anthropic's HH-RLHF red-team-attempts subset (human turns only, prompt-only). Step 6's local evaluator scores the Step 4 probes against this cache to measure how the precursor probe generalizes from web text to safety prompts.

**Inputs:** Gemma-2-2B (gated, license required), GemmaScope layer-20 residual SAE, `Anthropic/hh-rlhf` (`red-team-attempts/`).

**Outputs (Google Drive at `safety-sae-cache/safety_v1/`):** same schema as `safety-sae-cache/v1/` but with 100 sequences × 256 tokens = 25,600 tokens. ~470 MB total.

**Runtime:** T4 GPU. ~1.5 min after warm-up.

**After running:** download `safety-sae-cache/safety_v1/` to local `data/cache/safety_v1/`, then `python scripts/check_safety_cache.py`, then `python scripts/step6_ood_eval.py`.
"""))

# 1, 2
cells.append(md("## 1. Install dependencies (~1-2 min)"))
cells.append(code("!pip install -q sae_lens accelerate datasets"))

# 3, 4
cells.append(md("""
## 2. Mount Google Drive

Needs ~0.5 GB free in Drive for the safety cache.
"""))
cells.append(code("""
from google.colab import drive
import shutil, os

drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive'
free_gb = shutil.disk_usage(DRIVE_ROOT).free / 1e9
print(f'Drive free space: {free_gb:.2f} GB')
if free_gb < 1.0:
    raise RuntimeError(f'Need ~1 GB free; only {free_gb:.2f} GB available. Free space and re-run.')
"""))

# 5, 6
cells.append(md("""
## 3. Authenticate with Hugging Face

Gemma-2-2B is gated. Before continuing:
1. Accept the license at https://huggingface.co/google/gemma-2-2b
2. Generate a read token at https://huggingface.co/settings/tokens

Anthropic/hh-rlhf is public; no extra license required.
"""))
cells.append(code("""
from huggingface_hub import notebook_login
notebook_login()
"""))

# 7, 8
cells.append(md("""
## 4. Config (single source of truth)

Same layer/feature/SAE config as Step 3 (so the cache is directly comparable).
The differences are the corpus, the output dir, and N_SEQUENCES (100 instead of 400).
"""))
cells.append(code("""
CORPUS = 'Anthropic/hh-rlhf'
CORPUS_SUBSET = 'red-team-attempts'         # data_dir inside the dataset repo
N_SEQUENCES = 100                            # → 100 × 256 = 25,600 tokens
SEQ_LEN = 256                                # BOS-prefixed
EARLY_LAYERS = [5, 8, 12]
LATE_LAYER = 20
ALL_LAYERS = EARLY_LAYERS + [LATE_LAYER]
FEATURE_INDICES = [9989, 817, 12730, 892, 1031]
FEATURE_THEMES = {9989: 'refusal', 817: 'deception', 12730: 'ethics', 892: 'sycophancy-adj', 1031: 'harm'}
SAE_RELEASE = 'gemma-scope-2b-pt-res-canonical'
SAE_ID = 'layer_20/width_16k/canonical'
BATCH_SIZE = 4                                # smaller; total batches = 25
OUTPUT_DIR = '/content/drive/MyDrive/safety-sae-cache/safety_v1'
MODEL_NAME = 'google/gemma-2-2b'

N_TOKENS = N_SEQUENCES * SEQ_LEN
assert N_SEQUENCES % BATCH_SIZE == 0, 'N_SEQUENCES must be divisible by BATCH_SIZE'
N_BATCHES = N_SEQUENCES // BATCH_SIZE

print(f'Will cache {N_TOKENS:,} tokens × {len(ALL_LAYERS)} layers × 2304 dim (fp16) = '
      f'{N_TOKENS * len(ALL_LAYERS) * 2304 * 2 / 1e6:.1f} MB residuals + '
      f'{N_TOKENS * len(FEATURE_INDICES) * 2 / 1e6:.2f} MB features over {N_BATCHES} batches')
"""))

# 9, 10
cells.append(md("## 5. Load Gemma-2-2B"))
cells.append(code("""
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    raise RuntimeError('No GPU detected. Runtime → Change runtime type → T4 GPU.')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=device,
    low_cpu_mem_usage=True,
)
model.eval()
gc.collect(); torch.cuda.empty_cache()

d_model = model.config.hidden_size
n_layers = model.config.num_hidden_layers
assert max(ALL_LAYERS) < n_layers, f'requested layer {max(ALL_LAYERS)} >= n_layers={n_layers}'
print(f'Loaded {MODEL_NAME}: n_layers={n_layers}, d_model={d_model}, '
      f'GPU mem={torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))

# 11, 12
cells.append(md("## 6. Load the GemmaScope SAE (layer 20, width 16k, canonical)"))
cells.append(code("""
from sae_lens import SAE

sae, cfg_dict, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
sae.eval()
hook_layer = cfg_dict.get('hook_layer')
if hook_layer is not None and hook_layer != LATE_LAYER:
    raise ValueError(f'SAE hook_layer={hook_layer} != LATE_LAYER={LATE_LAYER}')
assert sae.cfg.d_in == d_model, f'SAE d_in={sae.cfg.d_in} != model d_model={d_model}'
print(f'SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, hook_layer={hook_layer}, '
      f'W_enc.dtype={sae.W_enc.dtype}')
"""))

# 13, 14 -- DIFFERENT FROM STEP 3: parse hh-rlhf transcripts, human turns only
cells.append(md("""
## 7. Load HH-RLHF red-team-attempts and extract human turns

The red-team subset's `transcript` field is a multi-turn conversation between a human red-teamer and a model assistant. We extract only the **human turns** because:

- The probe is a *precursor* detector; what fires on the user's input, before the model has decided how to respond.
- The Step 4 probe was trained on Pile-10k (web text, no chat markup); evaluating it on human-side prompts keeps the input distribution closer to "natural text" than on assistant outputs which often carry refusal templates that would be a different distribution shift.

Parsing: split each transcript on `\\n\\nHuman:` / `\\n\\nAssistant:` markers (the canonical Anthropic format), keep only the parts after `Human:` until the next marker, and tokenize without special tokens. Concatenate human turns across the dataset into a flat buffer of `100 × 255 = 25,500` raw tokens, reshape, BOS-prefix → `(100, 256)`.

Cross-transcript spans inside a sequence are acceptable (matches the Pile-10k packing convention).
"""))
cells.append(code("""
import numpy as np
import re
from datasets import load_dataset
from huggingface_hub import HfApi

try:
    DATASET_COMMIT = HfApi().dataset_info(CORPUS).sha
    print(f'Pinning {CORPUS} to commit {DATASET_COMMIT[:12]}')
except Exception as e:
    DATASET_COMMIT = None
    print(f'(could not resolve commit for {CORPUS}: {e}; using default revision)')

ds = load_dataset(CORPUS, data_dir=CORPUS_SUBSET, split='train', revision=DATASET_COMMIT)
print(f'{len(ds):,} transcripts in {CORPUS}/{CORPUS_SUBSET}')

# Defensive parsing: try the canonical \\n\\nHuman: / \\n\\nAssistant: markers, then
# fall back to looser whitespace if a transcript uses a different format.
HUMAN_PATTERNS = [
    re.compile(r'\\n\\nHuman:(.*?)(?=\\n\\nAssistant:|\\n\\nHuman:|$)', re.DOTALL),
    re.compile(r'\\nHuman:(.*?)(?=\\nAssistant:|\\nHuman:|$)', re.DOTALL),
    re.compile(r'Human:(.*?)(?=Assistant:|Human:|$)', re.DOTALL),
]


def extract_human_turns(transcript):
    if not isinstance(transcript, str):
        return []
    for pat in HUMAN_PATTERNS:
        turns = [t.strip() for t in pat.findall(transcript) if t.strip()]
        if turns:
            return turns
    # Fallback: use the whole transcript as one "turn" so we don't lose data.
    return [transcript.strip()] if transcript.strip() else []


raw_tokens_needed = N_SEQUENCES * (SEQ_LEN - 1)  # 25,500
buffer, transcripts_used, turns_used = [], 0, 0
for row in ds:
    turns = extract_human_turns(row.get('transcript', ''))
    if not turns:
        continue
    transcripts_used += 1
    for turn in turns:
        ids = tokenizer(turn, add_special_tokens=False)['input_ids']
        if not ids:
            continue
        buffer.extend(ids)
        turns_used += 1
        if len(buffer) >= raw_tokens_needed:
            break
    if len(buffer) >= raw_tokens_needed:
        break

assert len(buffer) >= raw_tokens_needed, (
    f'Corpus exhausted: only {len(buffer):,} tokens from {transcripts_used} transcripts '
    f'({turns_used} human turns), need {raw_tokens_needed:,}'
)
buffer = buffer[:raw_tokens_needed]

arr = np.array(buffer, dtype=np.int32).reshape(N_SEQUENCES, SEQ_LEN - 1)
bos_col = np.full((N_SEQUENCES, 1), tokenizer.bos_token_id, dtype=np.int32)
input_ids_all = np.concatenate([bos_col, arr], axis=1)
assert input_ids_all.shape == (N_SEQUENCES, SEQ_LEN)
print(f'Used {transcripts_used} transcripts, {turns_used} human turns → '
      f'buffer of {len(buffer):,} tokens → input_ids {input_ids_all.shape}')
"""))

# 15, 16
cells.append(md("""
## 8. Pre-allocate output arrays in host RAM

~470 MB total (much smaller than Step 3's 1.89 GB since we're caching 4× fewer tokens).
"""))
cells.append(code("""
os.makedirs(OUTPUT_DIR, exist_ok=True)

resid_arrays = {L: np.zeros((N_TOKENS, d_model), dtype=np.float16) for L in ALL_LAYERS}
feature_arr = np.zeros((N_TOKENS, len(FEATURE_INDICES)), dtype=np.float16)
token_ids_arr = input_ids_all.reshape(-1).astype(np.int32)

total_gb = (sum(a.nbytes for a in resid_arrays.values()) + feature_arr.nbytes + token_ids_arr.nbytes) / 1e9
print(f'Pre-allocated {total_gb:.3f} GB '
      f'({len(ALL_LAYERS)} resid layers × {resid_arrays[ALL_LAYERS[0]].nbytes/1e6:.0f} MB '
      f'+ features {feature_arr.nbytes/1e6:.2f} MB '
      f'+ token_ids {token_ids_arr.nbytes/1e6:.2f} MB)')
"""))

# 17, 18
cells.append(md("""
## 9. Register hooks and run the forward+encode loop

Same pattern as Step 3 (cleanup, hooks at `[5, 8, 12, 20]`, encode L20 through the SAE, write to pre-allocated buffers). 25 batches at `BATCH_SIZE=4`. ~1-1.5 min on T4.
"""))
cells.append(code("""
import torch
from tqdm.auto import tqdm

for layer_mod in model.model.layers:
    layer_mod._forward_hooks.clear()

_batch_cache = {}
def _make_hook(L):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _batch_cache[L] = h.detach()
    return hook

hooks = [model.model.layers[L].register_forward_hook(_make_hook(L)) for L in ALL_LAYERS]
feature_idx_t = torch.tensor(FEATURE_INDICES, device=device, dtype=torch.long)

try:
    input_ids_t = torch.from_numpy(input_ids_all).to(device)
    for b in tqdm(range(N_BATCHES), desc='extract'):
        s, e = b * BATCH_SIZE, (b + 1) * BATCH_SIZE
        batch = input_ids_t[s:e]
        _batch_cache.clear()
        with torch.no_grad():
            _ = model(batch)

        tok_s, tok_e = s * SEQ_LEN, e * SEQ_LEN

        resid_late = _batch_cache[LATE_LAYER].to(sae.W_enc.dtype)
        with torch.no_grad():
            feats = sae.encode(resid_late)[:, :, feature_idx_t]
        feature_arr[tok_s:tok_e] = feats.reshape(-1, len(FEATURE_INDICES)).to(torch.float16).cpu().numpy()

        for L in ALL_LAYERS:
            h = _batch_cache[L].reshape(-1, d_model).to(torch.float16).cpu().numpy()
            resid_arrays[L][tok_s:tok_e] = h
finally:
    for h in hooks:
        h.remove()

print(f'Extracted {N_TOKENS:,} tokens × {len(ALL_LAYERS)} layers; '
      f'GPU mem={torch.cuda.memory_allocated()/1e9:.2f} GB')
"""))

# 19, 20
cells.append(md("""
## 10. Save artifacts + metadata to Drive
"""))
cells.append(code("""
import json, datetime, platform
from pathlib import Path
import sae_lens, transformers, datasets as _datasets

out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)

for L in ALL_LAYERS:
    np.save(out / f'resid_layer_{L}.npy', resid_arrays[L])
np.save(out / 'feature_acts.npy', feature_arr)
np.save(out / 'token_ids.npy', token_ids_arr)

fire_rates = {str(idx): float((feature_arr[:, i] > 0).mean())
              for i, idx in enumerate(FEATURE_INDICES)}

metadata = {
    'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
    'platform': platform.platform(),
    'config': {
        'model': MODEL_NAME,
        'corpus': CORPUS,
        'corpus_subset': CORPUS_SUBSET,
        'turns': 'human_only',
        'dataset_commit': DATASET_COMMIT,
        'n_sequences': N_SEQUENCES,
        'seq_len': SEQ_LEN,
        'n_tokens': N_TOKENS,
        'early_layers': EARLY_LAYERS,
        'late_layer': LATE_LAYER,
        'all_layers': ALL_LAYERS,
        'feature_indices': FEATURE_INDICES,
        'feature_themes': {str(k): v for k, v in FEATURE_THEMES.items()},
        'sae_release': SAE_RELEASE,
        'sae_id': SAE_ID,
        'batch_size': BATCH_SIZE,
        'd_model': d_model,
        'dtype_resid': 'float16',
        'dtype_features': 'float16',
        'dtype_token_ids': 'int32',
        'layout': 'flat row-major; seq_idx = token_idx // seq_len',
    },
    'empirical': {
        'fire_rates': fire_rates,
    },
    'versions': {
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'sae_lens': sae_lens.__version__,
        'datasets': _datasets.__version__,
        'numpy': np.__version__,
    },
}

(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

print(f'Saved to {out}/')
for p in sorted(out.iterdir()):
    print(f'  {p.name:<22} {p.stat().st_size / 1e6:>10.2f} MB')
"""))

# 21, 22 -- Step 6 specific: compare safety vs Pile fire rates
cells.append(md("""
## 11. Sanity statistics - safety vs Pile-10k

Three checks:
1. **Safety-corpus fire rates vs the Pile-10k cache's** (from Step 3). Big differences are *interesting*, not failures; they tell us how the OOD distribution differs from the in-distribution one.
2. **Top-activating contexts on safety prompts**; should look semantically aligned with each feature's theme (refusal cues should appear in refusal contexts, harm cues in harmful contexts, etc.).
3. **Per-layer residual stats**; the same float16-clipping check as Step 3.
"""))
cells.append(code("""
# Pile-10k fire rates from the Step 3 metadata (committed to the repo).
PILE_FRAC_NONZERO = {9989: 0.010107, 817: 0.013887, 12730: 0.005674, 892: 0.008066, 1031: 0.006406}

print('1. Per-feature empirical fire rate (HH-RLHF red-team human turns, N={:,} tokens):'.format(N_TOKENS))
print(f'   {"idx":>6}  {"theme":<16}  {"empirical":>10}  {"pile-10k":>10}  {"ratio":>8}')
for i, idx in enumerate(FEATURE_INDICES):
    emp = float((feature_arr[:, i] > 0).mean())
    pile = PILE_FRAC_NONZERO[idx]
    ratio = emp / pile if pile > 0 else float('inf')
    print(f'   {idx:>6}  {FEATURE_THEMES[idx]:<16}  {emp:>10.5f}  {pile:>10.5f}  {ratio:>7.2f}x')

print()
print('2. Top-3 activating contexts per feature (5 tokens before, 1 after):')
for i, idx in enumerate(FEATURE_INDICES):
    acts = feature_arr[:, i].astype(np.float32)
    print(f'\\n   Feature {idx} ({FEATURE_THEMES[idx]}):')
    top_pos = np.argsort(acts)[::-1][:3]
    if acts[top_pos[0]] == 0:
        print('     [no firings on this corpus]')
        continue
    for p in top_pos:
        if acts[p] == 0:
            break
        ctx_s = max(0, int(p) - 5)
        ctx_e = min(N_TOKENS, int(p) + 2)
        ctx = tokenizer.decode(token_ids_arr[ctx_s:ctx_e].tolist())
        target = tokenizer.decode([int(token_ids_arr[p])])
        print(f'     act={acts[p]:>6.2f}  pos={int(p):>6}  target={target!r:<14}  ctx={ctx!r}')

print()
print('3. Per-layer residual stats:')
print(f'   {"layer":>5}  {"|x|_max (full)":>16}  {"|x|_mean (samp)":>17}  {"L2 (samp)":>11}')
rng = np.random.default_rng(0)
sample_idx = rng.choice(N_TOKENS, size=min(8192, N_TOKENS), replace=False)
for L in ALL_LAYERS:
    x = resid_arrays[L]
    max_abs = float(np.abs(x).max())
    samp = x[sample_idx].astype(np.float32)
    mean_abs = float(np.abs(samp).mean())
    l2_mean = float(np.linalg.norm(samp, axis=1).mean())
    flag = '  WARN: near fp16 ceiling' if max_abs > 6e4 else ''
    print(f'   {L:>5}  {max_abs:>16.2f}  {mean_abs:>17.4f}  {l2_mean:>11.2f}{flag}')
"""))

# 23
cells.append(md("""
## 12. Download to local

Cache lives in your Drive at `safety-sae-cache/safety_v1/`. To use it for Step 6:

1. Open https://drive.google.com
2. Right-click `safety-sae-cache/safety_v1` → **Download** (Drive zips and serves it).
3. Unzip into the repo so the layout is:
   ```
   data/cache/safety_v1/resid_layer_5.npy
   data/cache/safety_v1/resid_layer_8.npy
   data/cache/safety_v1/resid_layer_12.npy
   data/cache/safety_v1/resid_layer_20.npy
   data/cache/safety_v1/feature_acts.npy
   data/cache/safety_v1/token_ids.npy
   data/cache/safety_v1/metadata.json
   ```
4. Verify: `python scripts/check_safety_cache.py`
5. Run OOD eval: `python scripts/step6_ood_eval.py`

The download is ~470 MB; on a typical home connection, ~1-3 min.
"""))

# 24
cells.append(md("""
## Done

Safety cache is in Drive. After downloading and verifying locally, you're ready for the OOD eval.
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"Wrote {NB_PATH} ({NB_PATH.stat().st_size / 1024:.1f} KB, {len(cells)} cells)")
