"""Verify the Step 3 activation cache downloaded from Colab.

Checks file presence, shapes, dtypes, NaN/inf, fp16 headroom, BOS column layout,
per-feature fire rates (vs metadata and vs Neuronpedia), and metadata schema.

Run after downloading `safety-sae-cache/v1/` from Drive to `data/cache/v1/`:

    python scripts/check_activation_cache.py

Exits 0 on success, 1 on hard failures, 2 on missing files. Warnings (e.g. a
fire-rate ratio that differs from Neuronpedia) do not cause failure.
"""
import json
import sys
from pathlib import Path

import numpy as np


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "v1"

# Expectations from the Step 3 design (must match the notebook's config).
EXPECTED_LAYERS = [5, 8, 12, 20]
EXPECTED_FEATURES = [9989, 817, 12730, 892, 1031]
EXPECTED_N_SEQUENCES = 400
EXPECTED_SEQ_LEN = 256
EXPECTED_N_TOKENS = EXPECTED_N_SEQUENCES * EXPECTED_SEQ_LEN  # 102,400
EXPECTED_D_MODEL = 2304
GEMMA_VOCAB_SIZE = 256_000

# From data/target_features.json (Neuronpedia-reported fire rates).
NEURONPEDIA_FRAC_NONZERO = {
    9989: 0.00746, 817: 0.00763, 12730: 0.00485, 892: 0.00279, 1031: 0.00481,
}


class CheckLog:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def ok(self, msg):
        print(f"  PASS  {msg}")

    def fail(self, msg):
        print(f"  FAIL  {msg}")
        self.fails += 1

    def warn(self, msg):
        print(f"  WARN  {msg}")
        self.warns += 1


def check_files_exist(log):
    print("[1] File presence")
    expected = (
        [f"resid_layer_{L}.npy" for L in EXPECTED_LAYERS]
        + ["feature_acts.npy", "token_ids.npy", "metadata.json"]
    )
    missing = [n for n in expected if not (CACHE_DIR / n).exists()]
    if missing:
        log.fail(f"missing files: {missing}")
        return False
    log.ok(f"all {len(expected)} expected files present")
    return True


def check_metadata(log):
    print("\n[2] metadata.json schema + config")
    try:
        meta = json.loads((CACHE_DIR / "metadata.json").read_text())
    except Exception as e:
        log.fail(f"cannot parse metadata.json: {e}")
        return None
    cfg = meta.get("config", {})
    issues = []
    if cfg.get("n_tokens") != EXPECTED_N_TOKENS:
        issues.append(f"n_tokens={cfg.get('n_tokens')} (expected {EXPECTED_N_TOKENS})")
    if cfg.get("d_model") != EXPECTED_D_MODEL:
        issues.append(f"d_model={cfg.get('d_model')} (expected {EXPECTED_D_MODEL})")
    if cfg.get("all_layers") != EXPECTED_LAYERS:
        issues.append(f"all_layers={cfg.get('all_layers')} (expected {EXPECTED_LAYERS})")
    if cfg.get("feature_indices") != EXPECTED_FEATURES:
        issues.append(f"feature_indices={cfg.get('feature_indices')} (expected {EXPECTED_FEATURES})")
    for x in issues:
        log.fail(x)
    if not issues:
        log.ok(
            f"config matches expected (model={cfg.get('model')}, "
            f"corpus={cfg.get('corpus')}, created={meta.get('created_at')})"
        )
    return meta


def check_resid(log, layer):
    name = f"resid_layer_{layer}.npy"
    print(f"\n[3.{layer}] {name}")
    arr = np.load(CACHE_DIR / name, mmap_mode="r")
    if arr.shape != (EXPECTED_N_TOKENS, EXPECTED_D_MODEL):
        log.fail(f"shape {arr.shape} (expected ({EXPECTED_N_TOKENS}, {EXPECTED_D_MODEL}))")
        return
    if arr.dtype != np.float16:
        log.fail(f"dtype {arr.dtype} (expected float16)")
        return
    log.ok(f"shape={arr.shape} dtype={arr.dtype} (mmap loaded)")

    if np.isnan(arr).any():
        log.fail("contains NaN")
    elif np.isinf(arr).any():
        log.fail("contains inf")
    else:
        log.ok("no NaN/inf")

    max_abs = float(np.abs(arr).max())
    if max_abs > 6e4:
        log.fail(f"|x|_max={max_abs:.2f} too close to fp16 ceiling (65504)")
    else:
        log.ok(f"|x|_max={max_abs:.2f} (safely below fp16 ceiling)")


def check_features(log, meta):
    name = "feature_acts.npy"
    print(f"\n[4] {name}")
    arr = np.load(CACHE_DIR / name, mmap_mode="r")
    if arr.shape != (EXPECTED_N_TOKENS, len(EXPECTED_FEATURES)):
        log.fail(f"shape {arr.shape} (expected ({EXPECTED_N_TOKENS}, {len(EXPECTED_FEATURES)}))")
        return
    if arr.dtype != np.float16:
        log.fail(f"dtype {arr.dtype} (expected float16)")
        return
    log.ok(f"shape={arr.shape} dtype={arr.dtype} (mmap loaded)")

    if (arr < 0).any():
        log.fail("contains negative activations (SAE encode output should be >= 0)")
    else:
        log.ok("all activations >= 0")

    emp = {idx: float((arr[:, i] > 0).mean()) for i, idx in enumerate(EXPECTED_FEATURES)}

    if meta is not None:
        recorded = (meta.get("empirical", {}) or {}).get("fire_rates", {})
        mismatches = []
        for idx in EXPECTED_FEATURES:
            r = float(recorded.get(str(idx), -1))
            if abs(emp[idx] - r) > 1e-6:
                mismatches.append((idx, emp[idx], r))
        if mismatches:
            for idx, e, r in mismatches:
                log.fail(f"feature {idx}: empirical {e:.5f} != metadata {r:.5f}")
        else:
            log.ok("empirical fire rates match metadata.empirical.fire_rates")

    print("\n  empirical vs Neuronpedia (warning if ratio outside [0.25, 4.0]):")
    print(f"    {'idx':>6}  {'empirical':>10}  {'neuronpedia':>12}  {'ratio':>7}")
    out_of_band = []
    for idx in EXPECTED_FEATURES:
        e, n = emp[idx], NEURONPEDIA_FRAC_NONZERO[idx]
        ratio = e / n if n > 0 else float("inf")
        print(f"    {idx:>6}  {e:>10.5f}  {n:>12.5f}  {ratio:>6.2f}x")
        if not (0.25 <= ratio <= 4.0):
            out_of_band.append((idx, ratio))
    for idx, r in out_of_band:
        log.warn(f"feature {idx} fire-rate ratio {r:.2f}x outside [0.25, 4.0]")


def check_tokens(log):
    name = "token_ids.npy"
    print(f"\n[5] {name}")
    tok = np.load(CACHE_DIR / name, mmap_mode="r")
    if tok.shape != (EXPECTED_N_TOKENS,):
        log.fail(f"shape {tok.shape} (expected ({EXPECTED_N_TOKENS},))")
        return
    if tok.dtype != np.int32:
        log.fail(f"dtype {tok.dtype} (expected int32)")
        return
    log.ok(f"shape={tok.shape} dtype={tok.dtype} (mmap loaded)")

    mn, mx = int(tok.min()), int(tok.max())
    if mn < 0 or mx >= GEMMA_VOCAB_SIZE:
        log.fail(f"token range [{mn}, {mx}] outside [0, {GEMMA_VOCAB_SIZE})")
    else:
        log.ok(f"token range [{mn}, {mx}] within Gemma-2 vocab")

    bos_col = tok[::EXPECTED_SEQ_LEN]
    unique_bos = set(int(b) for b in bos_col)
    if len(unique_bos) != 1:
        log.fail(f"BOS column has {len(unique_bos)} distinct values (expected 1)")
    else:
        bos_id = next(iter(unique_bos))
        log.ok(
            f"BOS column uniform (id={bos_id}) across {EXPECTED_N_SEQUENCES} sequences "
            f"of length {EXPECTED_SEQ_LEN}"
        )


def print_sizes():
    print("\n[6] On-disk sizes")
    total = 0
    for name in sorted(p.name for p in CACHE_DIR.iterdir()):
        size = (CACHE_DIR / name).stat().st_size
        total += size
        print(f"  {name:<22} {size / 1e6:>10.2f} MB")
    print(f"  {'total':<22} {total / 1e9:>10.2f} GB")


def main():
    if not CACHE_DIR.exists():
        print(f"Cache directory not found: {CACHE_DIR}")
        print("Run notebooks/02_activation_cache.ipynb on Colab, then download the output here.")
        sys.exit(2)
    print(f"Verifying cache at {CACHE_DIR}\n")
    log = CheckLog()
    if not check_files_exist(log):
        sys.exit(2)
    meta = check_metadata(log)
    for L in EXPECTED_LAYERS:
        check_resid(log, L)
    check_features(log, meta)
    check_tokens(log)
    print_sizes()

    print()
    if log.fails:
        print(f"FAIL: {log.fails} hard failures, {log.warns} warnings")
        sys.exit(1)
    print(f"OK: 0 failures, {log.warns} warnings. Cache is usable for Step 4.")


if __name__ == "__main__":
    main()
