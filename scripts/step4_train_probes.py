"""Step 4 orchestrator: train and evaluate all 20 (feature, layer) probes.

Pipeline:
  1) Build the shared 320/80 sequence-level split (BOS masked).
  2) Run a 3-fold L2 sweep on a representative config (feature 9989, layer 12)
     to pick a single C; reuse for all 20 linear probes. Writes
     `results/step4_l2_sweep.csv`.
  3) For each layer in {5, 8, 12, 20}: materialize the layer's train+test fp32
     residuals once, z-score with train stats, then fit one linear + one MLP
     probe per feature, evaluate on the test fold, save weights, and free the
     layer's tensors before moving on.
  4) Write `results/step4_probe_metrics.csv` with 40 rows
     (5 features × 4 layers × 2 probe types).

Memory: peak per layer is roughly train (752 MB fp32) + test (188 MB fp32)
plus model state, well under the laptop's 8 GB ceiling.

Usage (from repo root):

    python scripts/step4_train_probes.py

For a quick end-to-end smoke test on one (feature, layer) instead of the full
20+20 grid:

    python scripts/step4_train_probes.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

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
from src.eval import evaluate
from src.probe import fit_linear, fit_mlp


RESULTS_DIR = REPO_ROOT / "results"
WEIGHTS_DIR = RESULTS_DIR / "step4_probe_weights"
SWEEP_CSV = RESULTS_DIR / "step4_l2_sweep.csv"
METRICS_CSV = RESULTS_DIR / "step4_probe_metrics.csv"
META_JSON = RESULTS_DIR / "step4_meta.json"

# L2 sweep configuration. We sweep on (feature=9989, layer=12) - refusal at the
# midpoint of the early-layer range, the canonical "is the precursor signal
# there?" question. The chosen C is reused for all 20 linear probes.
SWEEP_FEATURE = 9989
SWEEP_LAYER = 12
SWEEP_C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
SWEEP_FOLDS = 3
SEED = 0
N_BOOTSTRAP = 200


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_l2_sweep(X_train: np.ndarray, y_train: np.ndarray) -> tuple[float, list[dict]]:
    """3-fold stratified CV on the (already standardized) train fold."""
    log(
        f"L2 sweep on feature {SWEEP_FEATURE} / layer {SWEEP_LAYER}: "
        f"{len(SWEEP_C_GRID)} Cs x {SWEEP_FOLDS} folds = {len(SWEEP_C_GRID) * SWEEP_FOLDS} fits"
    )
    skf = StratifiedKFold(n_splits=SWEEP_FOLDS, shuffle=True, random_state=SEED)

    rows: list[dict] = []
    fold_scores: dict[float, list[float]] = {C: [] for C in SWEEP_C_GRID}
    for fi, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        for C in SWEEP_C_GRID:
            t0 = time.time()
            probe = fit_linear(X_tr, y_tr, C=C)
            scores = probe.score(X_va)
            metrics = evaluate(y_va, scores, n_bootstrap=0, seed=SEED)
            dt = time.time() - t0
            fold_scores[C].append(metrics.auc_roc.point)
            log(
                f"  fold {fi+1}/{SWEEP_FOLDS} C={C:>6g}: "
                f"AUC-ROC={metrics.auc_roc.point:.4f} "
                f"AUC-PR={metrics.auc_pr.point:.4f} "
                f"iters={probe.n_iter} ({dt:.1f}s)"
            )
            rows.append({
                "fold": fi,
                "C": C,
                "auc_roc": metrics.auc_roc.point,
                "auc_pr": metrics.auc_pr.point,
                "n_iter": probe.n_iter,
                "seconds": dt,
            })

    mean_aucs = {C: float(np.mean(s)) for C, s in fold_scores.items()}
    best_C = max(mean_aucs.items(), key=lambda kv: kv[1])[0]
    log(f"L2 sweep complete. Mean AUC by C: {mean_aucs}. Best C={best_C}")
    return best_C, rows


def write_sweep_csv(rows: list[dict]) -> None:
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "C", "auc_roc", "auc_pr", "n_iter", "seconds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"Wrote {SWEEP_CSV}")


METRIC_FIELDS = [
    "feature_idx", "feature_theme", "layer", "probe_type",
    "n_train", "n_train_pos", "n_test", "n_test_pos",
    "auc_roc", "auc_roc_lo", "auc_roc_hi",
    "auc_pr", "auc_pr_lo", "auc_pr_hi",
    "p_at_k", "p_at_k_lo", "p_at_k_hi",
    "C", "mlp_hidden", "mlp_epochs", "mlp_final_train_loss",
    "fit_seconds", "eval_seconds",
]


def metric_row(
    feature_idx: int,
    layer: int,
    probe_type: str,
    n_train: int,
    n_train_pos: int,
    metrics,
    fit_seconds: float,
    eval_seconds: float,
    C: float | None = None,
    mlp_hidden: int | None = None,
    mlp_epochs: int | None = None,
    mlp_final_train_loss: float | None = None,
) -> dict:
    return {
        "feature_idx": feature_idx,
        "feature_theme": FEATURE_THEMES[feature_idx],
        "layer": layer,
        "probe_type": probe_type,
        "n_train": n_train,
        "n_train_pos": n_train_pos,
        "n_test": metrics.n_test,
        "n_test_pos": metrics.n_test_pos,
        "auc_roc": metrics.auc_roc.point,
        "auc_roc_lo": metrics.auc_roc.lo,
        "auc_roc_hi": metrics.auc_roc.hi,
        "auc_pr": metrics.auc_pr.point,
        "auc_pr_lo": metrics.auc_pr.lo,
        "auc_pr_hi": metrics.auc_pr.hi,
        "p_at_k": metrics.precision_at_k.point,
        "p_at_k_lo": metrics.precision_at_k.lo,
        "p_at_k_hi": metrics.precision_at_k.hi,
        "C": C if C is not None else "",
        "mlp_hidden": mlp_hidden if mlp_hidden is not None else "",
        "mlp_epochs": mlp_epochs if mlp_epochs is not None else "",
        "mlp_final_train_loss": mlp_final_train_loss if mlp_final_train_loss is not None else "",
        "fit_seconds": round(fit_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
    }


def write_metrics_csv(rows: list[dict]) -> None:
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"Wrote {METRICS_CSV}")


def save_linear_weights(feature_idx: int, layer: int, probe, scaler_mean, scaler_scale, C: float) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        WEIGHTS_DIR / f"lin_{feature_idx}_L{layer}.npz",
        coef=probe.coef.astype(np.float32),
        intercept=np.float32(probe.intercept),
        scaler_mean=scaler_mean.astype(np.float32),
        scaler_scale=scaler_scale.astype(np.float32),
        C=np.float32(C),
        n_iter=np.int32(probe.n_iter),
    )


def save_mlp_weights(feature_idx: int, layer: int, probe, scaler_mean, scaler_scale) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "hidden": np.int32(probe.hidden),
        "dropout": np.float32(probe.dropout),
        "epochs": np.int32(probe.epochs),
        "final_train_loss": np.float32(probe.final_train_loss),
    }
    for k, v in probe.state_dict.items():
        # keys: fc1.weight, fc1.bias, fc2.weight, fc2.bias
        payload[k.replace(".", "_")] = v
    np.savez(WEIGHTS_DIR / f"mlp_{feature_idx}_L{layer}.npz", **payload)


def run_full_grid(best_C: float, smoke: bool = False) -> tuple[list[dict], dict]:
    """Train + eval all 20 linear and 20 MLP probes. Returns (metric rows, meta)."""
    split = make_split(seed=SEED)
    y_train_all, y_test_all = load_labels(split)
    log(
        f"Labels: train={y_train_all.shape} (positives per feature: "
        f"{y_train_all.sum(axis=0).tolist()}), "
        f"test={y_test_all.shape} (positives: {y_test_all.sum(axis=0).tolist()})"
    )

    layers = LAYERS if not smoke else [LAYERS[0]]
    features = FEATURE_INDICES if not smoke else [FEATURE_INDICES[0]]

    rows: list[dict] = []
    per_layer_seconds: dict[int, float] = {}
    for layer in layers:
        layer_t0 = time.time()
        log(f"--- Layer {layer}: loading + standardizing fp32 residuals ---")
        X_train, X_test = load_layer_split(layer, split)
        log(f"  X_train={X_train.shape} ({X_train.nbytes/1e6:.0f} MB) "
            f"X_test={X_test.shape} ({X_test.nbytes/1e6:.0f} MB)")
        mean, scale = fit_standardizer(X_train)
        apply_standardizer(X_train, mean, scale)
        apply_standardizer(X_test, mean, scale)

        for fi, feature_idx in enumerate(features):
            col = FEATURE_INDICES.index(feature_idx)
            y_train = y_train_all[:, col]
            y_test = y_test_all[:, col]
            n_train_pos = int(y_train.sum())

            # Linear probe
            t0 = time.time()
            lin = fit_linear(X_train, y_train, C=best_C)
            fit_s = time.time() - t0
            t0 = time.time()
            scores = lin.score(X_test)
            metrics = evaluate(y_test, scores, n_bootstrap=N_BOOTSTRAP, seed=SEED)
            eval_s = time.time() - t0
            log(
                f"  [linear] feat={feature_idx} ({FEATURE_THEMES[feature_idx]:<14}) layer={layer:>2}: "
                f"AUC-ROC={metrics.auc_roc.point:.4f} [{metrics.auc_roc.lo:.4f}, {metrics.auc_roc.hi:.4f}]  "
                f"AUC-PR={metrics.auc_pr.point:.4f} [{metrics.auc_pr.lo:.4f}, {metrics.auc_pr.hi:.4f}]  "
                f"p@k={metrics.precision_at_k.point:.4f}  "
                f"({fit_s:.1f}s fit, {eval_s:.1f}s eval, n_iter={lin.n_iter})"
            )
            save_linear_weights(feature_idx, layer, lin, mean, scale, best_C)
            rows.append(metric_row(
                feature_idx, layer, "linear", X_train.shape[0], n_train_pos,
                metrics, fit_s, eval_s, C=best_C,
            ))

            # MLP sanity check
            t0 = time.time()
            mlp = fit_mlp(X_train, y_train, seed=SEED + fi)
            fit_s = time.time() - t0
            t0 = time.time()
            scores = mlp.score(X_test)
            metrics = evaluate(y_test, scores, n_bootstrap=N_BOOTSTRAP, seed=SEED)
            eval_s = time.time() - t0
            log(
                f"  [mlp   ] feat={feature_idx} ({FEATURE_THEMES[feature_idx]:<14}) layer={layer:>2}: "
                f"AUC-ROC={metrics.auc_roc.point:.4f} [{metrics.auc_roc.lo:.4f}, {metrics.auc_roc.hi:.4f}]  "
                f"AUC-PR={metrics.auc_pr.point:.4f} [{metrics.auc_pr.lo:.4f}, {metrics.auc_pr.hi:.4f}]  "
                f"p@k={metrics.precision_at_k.point:.4f}  "
                f"({fit_s:.1f}s fit, {eval_s:.1f}s eval, loss={mlp.final_train_loss:.4f})"
            )
            save_mlp_weights(feature_idx, layer, mlp, mean, scale)
            rows.append(metric_row(
                feature_idx, layer, "mlp", X_train.shape[0], n_train_pos,
                metrics, fit_s, eval_s,
                mlp_hidden=mlp.hidden, mlp_epochs=mlp.epochs,
                mlp_final_train_loss=mlp.final_train_loss,
            ))

        per_layer_seconds[layer] = round(time.time() - layer_t0, 1)
        del X_train, X_test, mean, scale  # free before next layer
        log(f"--- Layer {layer} done ({per_layer_seconds[layer]:.0f}s wall) ---")

    meta = {
        "seed": SEED,
        "split": {
            "train_seq_ids": split.train_seq_ids.tolist(),
            "test_seq_ids": split.test_seq_ids.tolist(),
            "n_train": split.n_train,
            "n_test": split.n_test,
            "bos_masked": True,
        },
        "best_C": best_C,
        "n_bootstrap": N_BOOTSTRAP,
        "per_layer_seconds": per_layer_seconds,
    }
    return rows, meta


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="single (feature, layer) end-to-end test")
    p.add_argument("--skip-sweep", action="store_true", help="reuse cached best_C from META_JSON")
    p.add_argument("--C", type=float, default=None, help="skip sweep and force this C")
    return p.parse_args()


def main():
    args = parse_args()
    overall_t0 = time.time()

    # --- L2 sweep stage ---
    if args.C is not None:
        log(f"Using forced C={args.C}; skipping L2 sweep")
        best_C = args.C
    elif args.skip_sweep and META_JSON.exists():
        prev = json.loads(META_JSON.read_text())
        best_C = float(prev["best_C"])
        log(f"Reusing best_C={best_C} from {META_JSON}")
    else:
        split = make_split(seed=SEED)
        y_train_all, _ = load_labels(split)
        col = FEATURE_INDICES.index(SWEEP_FEATURE)
        y_train = y_train_all[:, col]
        log(f"Loading layer {SWEEP_LAYER} for sweep (will be re-loaded for the full grid)")
        X_train, _ = load_layer_split(SWEEP_LAYER, split)
        mean, scale = fit_standardizer(X_train)
        apply_standardizer(X_train, mean, scale)
        best_C, sweep_rows = run_l2_sweep(X_train, y_train)
        write_sweep_csv(sweep_rows)
        del X_train, mean, scale

    if args.smoke:
        log("Smoke mode: running 1 feature x 1 layer end-to-end")

    rows, meta = run_full_grid(best_C, smoke=args.smoke)

    if not args.smoke:
        write_metrics_csv(rows)
        META_JSON.write_text(json.dumps(meta, indent=2))
        log(f"Wrote {META_JSON}")
    else:
        log("Smoke run complete. Skipping CSV write.")
        for r in rows:
            log(f"  smoke result: {r}")

    log(f"Total wall: {time.time() - overall_t0:.0f}s")


if __name__ == "__main__":
    main()
