"""Verify the Step 6 safety-corpus activation cache downloaded from Colab.

Same checks as `check_activation_cache.py` but for the smaller OOD cache:
file presence, shapes, dtypes, NaN/inf, fp16 headroom, BOS column layout,
per-feature fire rates, metadata schema. Also compares the safety-corpus
fire rates against the Pile-10k baseline (warning, not failure, if they
diverge by more than 5× either way).

Run after downloading `safety-sae-cache/safety_v1/` from Drive to
`data/cache/safety_v1/`:

    python scripts/check_safety_cache.py

Exits 0 on success, 1 on hard failures, 2 on missing files.
"""
import json
import sys
from pathlib import Path

import numpy as np


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "safety_v1"
PILE_META = Path(__file__).resolve().parent.parent / "data" / "cache" / "v1" / "metadata.json"

EXPECTED_LAYERS = [5, 8, 12, 20]
EXPECTED_FEATURES = [9989, 817, 12730, 892, 1031]
EXPECTED_N_SEQUENCES = 100
EXPECTED_SEQ_LEN = 256
EXPECTED_N_TOKENS = EXPECTED_N_SEQUENCES * EXPECTED_SEQ_LEN  # 25,600
EXPECTED_D_MODEL = 2304
GEMMA_VOCAB_SIZE = 256_000


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
    if cfg.get("corpus") != "Anthropic/hh-rlhf":
        issues.append(f"corpus={cfg.get('corpus')} (expected Anthropic/hh-rlhf)")
    if cfg.get("turns") != "human_only":
        issues.append(f"turns={cfg.get('turns')} (expected human_only)")
    for x in issues:
        log.fail(x)
    if not issues:
        log.ok(
            f"config matches expected (corpus={cfg.get('corpus')}/{cfg.get('corpus_subset')}, "
            f"turns={cfg.get('turns')}, created={meta.get('created_at')})"
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


def load_pile_fire_rates() -> dict[int, float]:
    if not PILE_META.exists():
        return {}
    try:
        meta = json.loads(PILE_META.read_text())
        rates = (meta.get("empirical") or {}).get("fire_rates") or {}
        return {int(k): float(v) for k, v in rates.items()}
    except Exception:
        return {}


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

    pile = load_pile_fire_rates()
    if pile:
        print("\n  empirical vs Pile-10k (warning if ratio outside [0.2, 5.0]):")
        print(f"    {'idx':>6}  {'safety':>10}  {'pile-10k':>10}  {'ratio':>8}")
        for idx in EXPECTED_FEATURES:
            e = emp[idx]
            p = pile.get(idx, float("nan"))
            ratio = (e / p) if p > 0 else float("inf")
            print(f"    {idx:>6}  {e:>10.5f}  {p:>10.5f}  {ratio:>7.2f}x")
            if not (0.2 <= ratio <= 5.0):
                log.warn(f"feature {idx} safety/pile fire-rate ratio {ratio:.2f}x outside [0.2, 5.0]")
    else:
        log.warn("no Pile-10k metadata found at data/cache/v1/metadata.json; skipping fire-rate comparison")

    n_pos_per_feat = (arr > 0).sum(axis=0).tolist()
    print(f"\n  Positive token counts per feature on safety cache: "
          f"{dict(zip(EXPECTED_FEATURES, n_pos_per_feat))}")
    low_count = [(idx, n_pos_per_feat[i]) for i, idx in enumerate(EXPECTED_FEATURES) if n_pos_per_feat[i] < 20]
    if low_count:
        for idx, n in low_count:
            log.warn(f"feature {idx}: only {n} positives in 25,600 tokens — OOD AUC-PR will be wide")


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
        log.ok(f"BOS column uniform (id={bos_id}) across {EXPECTED_N_SEQUENCES} sequences "
               f"of length {EXPECTED_SEQ_LEN}")


def print_sizes():
    print("\n[6] On-disk sizes")
    total = 0
    for name in sorted(p.name for p in CACHE_DIR.iterdir()):
        size = (CACHE_DIR / name).stat().st_size
        total += size
        print(f"  {name:<22} {size / 1e6:>10.2f} MB")
    print(f"  {'total':<22} {total / 1e6:>10.2f} MB")


def main():
    if not CACHE_DIR.exists():
        print(f"Cache directory not found: {CACHE_DIR}")
        print("Run notebooks/03_safety_cache.ipynb on Colab, then download the output here.")
        sys.exit(2)
    print(f"Verifying safety cache at {CACHE_DIR}\n")
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
    print(f"OK: 0 failures, {log.warns} warnings. Cache is usable for Step 6 OOD eval.")


if __name__ == "__main__":
    main()
