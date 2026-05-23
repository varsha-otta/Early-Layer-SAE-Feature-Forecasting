"""Step 5 orchestrator: data-efficiency sweeps for all (feature, layer, N) probes.

Design (locked in 2026-05-23):
  - N_seq grid: [2, 4, 8, 16, 32, 64, 128, 256, 320] (9 points, log-spaced, dense low end)
  - 5 random subsamples per N for variance estimation (1 subsample at N=320, deterministic)
  - C = 0.001 held constant across all probes (Step 4 picked it; we don't re-tune per N)
  - Z-score scaler fit per subsample on the subsample's rows only (no leakage from
    unused train data; keeps the "with N tokens" claim honest)
  - Test fold fixed at Step 4's 80 test sequences (cross-N AUC is on the same denominator)
  - Linear probes only (Step 4 showed MLP doesn't beat linear)
  - Skip per-probe bootstrap CIs; cross-subsample spread is our variance estimate

Memory: peak ~940 MB per layer (full unscaled train+test fp32) plus per-subsample
copies (small for low N, ~940 MB for N=320). Well under 8 GB.

Usage:

    python scripts/step5_efficiency.py             # full run (~80 min)
    python scripts/step5_efficiency.py --smoke     # 1 layer x 1 feature x 3 Ns
"""
from __future__ import annotations

import argparse
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
    apply_standardizer,
    fit_standardizer,
    load_labels,
    load_layer_split,
    make_split,
)
from src.data_efficiency import (
    N_SEQ_GRID,
    SUBSAMPLE_BASE_SEED,
    SUBSAMPLES_PER_N,
    make_subsample,
    subsample_seeds_for,
)
from src.eval import evaluate
from src.probe import fit_linear


RESULTS_DIR = REPO_ROOT / "results"
CURVES_CSV = RESULTS_DIR / "step5_efficiency_curves.csv"
META_JSON = RESULTS_DIR / "step5_meta.json"

C_FIXED = 0.001  # from Step 4 L2 sweep
SEED = 0  # split seed (same as Step 4)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


CSV_FIELDS = [
    "feature_idx", "feature_theme", "layer", "n_seq", "n_tokens",
    "subsample_seed", "n_train_pos", "n_test", "n_test_pos",
    "auc_roc", "auc_pr", "p_at_k",
    "C", "n_iter", "converged", "fit_seconds",
    "degenerate",  # 1 if subsample had 0 positives, 0 otherwise
]


def make_metric_row(
    feature_idx: int, layer: int, n_seq: int, n_tokens: int,
    subsample_seed: int, n_train_pos: int, metrics, lin,
    fit_seconds: float, degenerate: bool,
) -> dict:
    if metrics is None:
        return {
            "feature_idx": feature_idx,
            "feature_theme": FEATURE_THEMES[feature_idx],
            "layer": layer,
            "n_seq": n_seq,
            "n_tokens": n_tokens,
            "subsample_seed": subsample_seed,
            "n_train_pos": n_train_pos,
            "n_test": "",
            "n_test_pos": "",
            "auc_roc": "",
            "auc_pr": "",
            "p_at_k": "",
            "C": C_FIXED,
            "n_iter": "",
            "converged": "",
            "fit_seconds": round(fit_seconds, 3),
            "degenerate": 1 if degenerate else 0,
        }
    return {
        "feature_idx": feature_idx,
        "feature_theme": FEATURE_THEMES[feature_idx],
        "layer": layer,
        "n_seq": n_seq,
        "n_tokens": n_tokens,
        "subsample_seed": subsample_seed,
        "n_train_pos": n_train_pos,
        "n_test": metrics.n_test,
        "n_test_pos": metrics.n_test_pos,
        "auc_roc": metrics.auc_roc.point,
        "auc_pr": metrics.auc_pr.point,
        "p_at_k": metrics.precision_at_k.point,
        "C": C_FIXED,
        "n_iter": int(lin.n_iter),
        "converged": 1 if lin.n_iter < 1000 else 0,
        "fit_seconds": round(fit_seconds, 3),
        "degenerate": 1 if degenerate else 0,
    }


def write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with CURVES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"Wrote {CURVES_CSV}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true",
                   help="run 1 layer x 1 feature x 3 Ns; do not write CSV")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="override layer list (e.g. --layers 12 20)")
    return p.parse_args()


def main():
    args = parse_args()
    overall_t0 = time.time()

    split = make_split(seed=SEED)
    y_train_all, y_test_all = load_labels(split)
    log(f"Train labels: positives per feature = {y_train_all.sum(axis=0).tolist()}")
    log(f"Test labels:  positives per feature = {y_test_all.sum(axis=0).tolist()}")

    layers = args.layers if args.layers is not None else LAYERS
    if args.smoke:
        layers = [layers[2]]  # default smoke: layer 12 (mid-network)
    features = FEATURE_INDICES
    n_grid = N_SEQ_GRID
    if args.smoke:
        features = [FEATURE_INDICES[0]]  # refusal
        n_grid = [4, 32, 320]  # 3 widely spaced points

    log(
        f"Plan: layers={layers}, features={features}, N_grid={n_grid}, "
        f"subsamples_per_N={SUBSAMPLES_PER_N} (1 at N=full)"
    )

    rows: list[dict] = []
    total_fits = 0
    layer_seconds: dict[int, float] = {}

    for layer in layers:
        layer_t0 = time.time()
        log(f"--- Layer {layer}: loading unscaled fp32 residuals ---")
        X_train, X_test = load_layer_split(layer, split)
        log(f"  X_train={X_train.shape} ({X_train.nbytes/1e6:.0f} MB), "
            f"X_test={X_test.shape} ({X_test.nbytes/1e6:.0f} MB)")

        for n_seq in n_grid:
            seeds = subsample_seeds_for(n_seq, split.train_seq_ids.shape[0], SUBSAMPLES_PER_N)
            for seed in seeds:
                sub = make_subsample(split, n_seq, seed)

                X_sub = X_train[sub.local_rows].astype(np.float32, copy=True)
                mean, scale = fit_standardizer(X_sub)
                apply_standardizer(X_sub, mean, scale)
                X_test_scaled = X_test.copy()
                apply_standardizer(X_test_scaled, mean, scale)

                y_sub_all = y_train_all[sub.local_rows]  # (n_tokens, 5)

                for fi, feature_idx in enumerate(features):
                    col = FEATURE_INDICES.index(feature_idx)
                    y_sub = y_sub_all[:, col]
                    y_test = y_test_all[:, col]
                    n_train_pos = int(y_sub.sum())

                    if n_train_pos < 1 or (y_sub == 0).sum() < 1:
                        # Degenerate: can't fit a logistic regression.
                        rows.append(make_metric_row(
                            feature_idx, layer, n_seq, sub.n_tokens, seed,
                            n_train_pos, None, None, 0.0, degenerate=True,
                        ))
                        continue

                    t0 = time.time()
                    lin = fit_linear(X_sub, y_sub, C=C_FIXED)
                    fit_s = time.time() - t0
                    scores = lin.score(X_test_scaled)
                    metrics = evaluate(y_test, scores, n_bootstrap=0, seed=SEED)
                    rows.append(make_metric_row(
                        feature_idx, layer, n_seq, sub.n_tokens, seed,
                        n_train_pos, metrics, lin, fit_s, degenerate=False,
                    ))
                    total_fits += 1

                # Log the FIRST feature at each (N, seed) to keep noise down at scale.
                first = next(
                    r for r in reversed(rows)
                    if r["feature_idx"] == features[0]
                    and r["layer"] == layer
                    and r["n_seq"] == n_seq
                    and r["subsample_seed"] == seed
                )
                if first.get("auc_roc") == "":
                    log(f"  L{layer} N={n_seq:>3} seed={seed} "
                        f"[{features[0]} {FEATURE_THEMES[features[0]]}]: degenerate")
                else:
                    log(f"  L{layer} N={n_seq:>3} ({sub.n_tokens:>5}t) seed={seed} "
                        f"[{features[0]} {FEATURE_THEMES[features[0]]:<14}]: "
                        f"AUC-ROC={first['auc_roc']:.4f} AUC-PR={first['auc_pr']:.4f} "
                        f"({first['fit_seconds']:.2f}s, n_pos_tr={first['n_train_pos']})")

                del X_sub, X_test_scaled, y_sub_all

        del X_train, X_test
        layer_seconds[layer] = round(time.time() - layer_t0, 1)
        log(f"--- Layer {layer} done ({layer_seconds[layer]:.0f}s wall) ---")

    log(f"Completed {total_fits} non-degenerate probe fits in {time.time() - overall_t0:.0f}s")

    if args.smoke:
        log("Smoke run; skipping CSV writes")
        for r in rows[:18]:
            log(f"  smoke row: {r}")
    else:
        write_csv(rows)
        meta = {
            "seed": SEED,
            "split_seed": SEED,
            "subsample_base_seed": SUBSAMPLE_BASE_SEED,
            "subsamples_per_N": SUBSAMPLES_PER_N,
            "C_fixed": C_FIXED,
            "n_seq_grid": N_SEQ_GRID,
            "tokens_grid": [n * (256 - 1) for n in N_SEQ_GRID],
            "layers": list(layers),
            "features": list(features),
            "total_fits": total_fits,
            "layer_seconds": layer_seconds,
            "total_seconds": round(time.time() - overall_t0, 1),
        }
        META_JSON.write_text(json.dumps(meta, indent=2))
        log(f"Wrote {META_JSON}")


if __name__ == "__main__":
    main()
