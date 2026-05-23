"""Step 6 OOD eval: score Step 4 probes on the Step 6 safety cache.

For each (feature, layer):
  1. Load the saved Step 4 linear probe (`results/step4_probe_weights/lin_*_L*.npz`).
     Each .npz contains `coef`, `intercept`, `scaler_mean`, `scaler_scale`, `C`.
  2. mmap the corresponding safety-cache residuals (`resid_layer_L.npy`).
  3. Mask out BOS positions (offset 0 of each sequence).
  4. Apply the saved scaler (mean and scale fit on the Pile train fold).
  5. Compute decision-function scores via `scaler(X) @ coef + intercept`.
  6. Evaluate against the safety cache's `feature_acts.npy` labels.
  7. Compute AUC-ROC / AUC-PR / p@k with 200 stratified bootstrap CIs.

Output: `results/step6_ood_metrics.csv`, with the Step 4 in-distribution test
AUC joined alongside as the reference baseline. `results/step6_meta.json`
captures config + counts.

Memory: 25,600 × 2304 × fp32 = 235 MB per layer. We hold one layer at a time.

Usage:

    python scripts/step6_ood_eval.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    FEATURE_INDICES,
    FEATURE_THEMES,
    LAYERS,
    SEQ_LEN,
)
from src.eval import evaluate

SAFETY_CACHE_DIR = REPO_ROOT / "data" / "cache" / "safety_v1"
PILE_META_PATH = REPO_ROOT / "data" / "cache" / "v1" / "metadata.json"
WEIGHTS_DIR = REPO_ROOT / "results" / "step4_probe_weights"
STEP4_METRICS_PATH = REPO_ROOT / "results" / "step4_probe_metrics.csv"

RESULTS_DIR = REPO_ROOT / "results"
OOD_CSV = RESULTS_DIR / "step6_ood_metrics.csv"
META_JSON = RESULTS_DIR / "step6_meta.json"

N_BOOTSTRAP = 200
SEED = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bos_masked_rows(n_tokens: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """All non-BOS row indices in a flat row-major cache."""
    n_seq = n_tokens // seq_len
    offsets = np.arange(1, seq_len, dtype=np.int64)
    base = (np.arange(n_seq, dtype=np.int64) * seq_len)[:, None]
    return (base + offsets[None, :]).reshape(-1)


def load_step4_test_baseline() -> dict[tuple[int, int], dict[str, float]]:
    """Map (feature_idx, layer) -> Step 4 linear in-distribution metrics."""
    out: dict[tuple[int, int], dict[str, float]] = {}
    with STEP4_METRICS_PATH.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["probe_type"] != "linear":
                continue
            key = (int(r["feature_idx"]), int(r["layer"]))
            out[key] = {
                "id_auc_roc": float(r["auc_roc"]),
                "id_auc_pr": float(r["auc_pr"]),
                "id_p_at_k": float(r["p_at_k"]),
                "id_n_test": int(r["n_test"]),
                "id_n_test_pos": int(r["n_test_pos"]),
            }
    return out


def load_safety_metadata() -> dict:
    return json.loads((SAFETY_CACHE_DIR / "metadata.json").read_text())


def load_probe(feature_idx: int, layer: int) -> dict:
    path = WEIGHTS_DIR / f"lin_{feature_idx}_L{layer}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing probe weights at {path}")
    payload = np.load(path)
    return {
        "coef": payload["coef"].astype(np.float32),
        "intercept": float(payload["intercept"]),
        "scaler_mean": payload["scaler_mean"].astype(np.float32),
        "scaler_scale": payload["scaler_scale"].astype(np.float32),
        "C": float(payload["C"]),
    }


def score_with_probe(X_raw: np.ndarray, probe: dict) -> np.ndarray:
    """Apply scaler then linear probe to fp32 residuals.

    Returns decision-function scores (logits). Operates in place on X_raw.
    """
    np.subtract(X_raw, probe["scaler_mean"], out=X_raw)
    np.divide(X_raw, probe["scaler_scale"], out=X_raw)
    return X_raw @ probe["coef"] + probe["intercept"]


CSV_FIELDS = [
    "feature_idx", "feature_theme", "layer", "C",
    "ood_n_test", "ood_n_test_pos",
    "ood_auc_roc", "ood_auc_roc_lo", "ood_auc_roc_hi",
    "ood_auc_pr", "ood_auc_pr_lo", "ood_auc_pr_hi",
    "ood_p_at_k", "ood_p_at_k_lo", "ood_p_at_k_hi",
    "id_auc_roc", "id_auc_pr", "id_p_at_k", "id_n_test_pos",
    "gap_auc_roc", "gap_auc_pr",
]


def main():
    if not SAFETY_CACHE_DIR.exists():
        print(f"Safety cache not found: {SAFETY_CACHE_DIR}")
        print("Run notebooks/03_safety_cache.ipynb on Colab, then download the output here.")
        sys.exit(2)
    if not WEIGHTS_DIR.exists():
        print(f"Step 4 weights not found: {WEIGHTS_DIR}")
        print("Run scripts/step4_train_probes.py first.")
        sys.exit(2)

    overall_t0 = time.time()
    safety_meta = load_safety_metadata()
    n_tokens = int(safety_meta["config"]["n_tokens"])
    log(f"Safety cache: {n_tokens:,} tokens, corpus={safety_meta['config']['corpus']}")

    rows_keep = bos_masked_rows(n_tokens)
    log(f"BOS-masked rows: {rows_keep.shape[0]:,}")

    # Labels (binary fire) for all 5 features, BOS removed
    feature_acts = np.load(SAFETY_CACHE_DIR / "feature_acts.npy", mmap_mode="r")
    y_all = (np.asarray(feature_acts[rows_keep], dtype=np.float32) > 0).astype(np.uint8)
    log(f"OOD positives per feature: "
        f"{dict(zip(FEATURE_INDICES, y_all.sum(axis=0).tolist()))}")

    id_baseline = load_step4_test_baseline()
    rows: list[dict] = []

    for layer in LAYERS:
        log(f"--- Layer {layer}: loading residuals ---")
        resid_mmap = np.load(SAFETY_CACHE_DIR / f"resid_layer_{layer}.npy", mmap_mode="r")
        # Materialize the BOS-masked tokens as fp32 once per layer; per-feature
        # scaling mutates it, so we keep a clean copy.
        X_clean = np.asarray(resid_mmap[rows_keep], dtype=np.float32)
        log(f"  X_layer={X_clean.shape} ({X_clean.nbytes/1e6:.0f} MB)")

        for fi, feature_idx in enumerate(FEATURE_INDICES):
            probe = load_probe(feature_idx, layer)
            X_work = X_clean.copy()  # 235 MB working buffer
            t0 = time.time()
            scores = score_with_probe(X_work, probe)
            y_true = y_all[:, fi]
            metrics = evaluate(y_true, scores, n_bootstrap=N_BOOTSTRAP, seed=SEED)
            eval_s = time.time() - t0

            id_row = id_baseline.get((feature_idx, layer), {})
            gap_roc = (id_row.get("id_auc_roc", float("nan")) - metrics.auc_roc.point)
            gap_pr = (id_row.get("id_auc_pr", float("nan")) - metrics.auc_pr.point)

            log(f"  feat={feature_idx} ({FEATURE_THEMES[feature_idx]:<14}) L{layer:>2}: "
                f"OOD ROC={metrics.auc_roc.point:.4f} [{metrics.auc_roc.lo:.4f}, {metrics.auc_roc.hi:.4f}]  "
                f"OOD PR={metrics.auc_pr.point:.4f}  "
                f"ID ROC={id_row.get('id_auc_roc', float('nan')):.4f}  "
                f"gap_ROC={gap_roc:+.4f}  ({eval_s:.1f}s, n_pos={metrics.n_test_pos})")

            rows.append({
                "feature_idx": feature_idx,
                "feature_theme": FEATURE_THEMES[feature_idx],
                "layer": layer,
                "C": probe["C"],
                "ood_n_test": metrics.n_test,
                "ood_n_test_pos": metrics.n_test_pos,
                "ood_auc_roc": metrics.auc_roc.point,
                "ood_auc_roc_lo": metrics.auc_roc.lo,
                "ood_auc_roc_hi": metrics.auc_roc.hi,
                "ood_auc_pr": metrics.auc_pr.point,
                "ood_auc_pr_lo": metrics.auc_pr.lo,
                "ood_auc_pr_hi": metrics.auc_pr.hi,
                "ood_p_at_k": metrics.precision_at_k.point,
                "ood_p_at_k_lo": metrics.precision_at_k.lo,
                "ood_p_at_k_hi": metrics.precision_at_k.hi,
                "id_auc_roc": id_row.get("id_auc_roc", ""),
                "id_auc_pr": id_row.get("id_auc_pr", ""),
                "id_p_at_k": id_row.get("id_p_at_k", ""),
                "id_n_test_pos": id_row.get("id_n_test_pos", ""),
                "gap_auc_roc": gap_roc,
                "gap_auc_pr": gap_pr,
            })
            del X_work
        del X_clean

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with OOD_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"Wrote {OOD_CSV} ({len(rows)} rows)")

    meta = {
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "ood_corpus": safety_meta["config"]["corpus"],
        "ood_corpus_subset": safety_meta["config"].get("corpus_subset"),
        "ood_turns": safety_meta["config"].get("turns"),
        "ood_n_tokens_total": n_tokens,
        "ood_n_tokens_after_bos_mask": int(rows_keep.shape[0]),
        "ood_positives_per_feature": dict(zip(
            [str(f) for f in FEATURE_INDICES], y_all.sum(axis=0).astype(int).tolist()
        )),
        "step4_weights_dir": str(WEIGHTS_DIR),
        "total_seconds": round(time.time() - overall_t0, 1),
    }
    META_JSON.write_text(json.dumps(meta, indent=2))
    log(f"Wrote {META_JSON}")


if __name__ == "__main__":
    main()
