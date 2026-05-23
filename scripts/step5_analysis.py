"""Aggregate the Step 5 raw CSV into headline data-efficiency numbers.

Two views are produced:

  1. **Per-(feature, layer, N) aggregate**; mean and std of AUC-ROC, AUC-PR,
     p@k across the (up to 5) subsamples. Degenerate rows are dropped before
     aggregation. Written to `results/step5_efficiency_aggregate.csv`.

  2. **Headline table**; for each (feature, layer), find the smallest N at
     which the *mean* AUC crosses two thresholds: AUC-ROC = 0.9 and
     AUC-PR = 0.5. Linear interpolation between adjacent N points in log-N
     space; reports n_tokens at the crossing. Written to
     `results/step5_headline.csv`.

The headline table is the input to the writeup's "M_SAE / N_probe" ratio.

Usage:

    python scripts/step5_analysis.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_INDICES, FEATURE_THEMES, LAYERS

RESULTS_DIR = REPO_ROOT / "results"
RAW_CSV = RESULTS_DIR / "step5_efficiency_curves.csv"
AGG_CSV = RESULTS_DIR / "step5_efficiency_aggregate.csv"
HEADLINE_CSV = RESULTS_DIR / "step5_headline.csv"

ROC_THRESHOLD = 0.9
PR_THRESHOLD = 0.5


def load_rows() -> list[dict]:
    with RAW_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(s: str) -> float:
    return float(s) if s not in ("", None) else float("nan")


def aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate (feature, layer, N) → mean/std/min/max across subsamples.

    Degenerate rows (where the probe couldn't be fit) are dropped before
    aggregation; the resulting `n_subsamples` field shows how many were used.
    """
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        if int(r["degenerate"]):
            continue
        key = (int(r["feature_idx"]), int(r["layer"]), int(r["n_seq"]))
        by_key.setdefault(key, []).append(r)

    out = []
    for (feat, layer, n_seq), bucket in sorted(by_key.items()):
        roc = np.array([parse_float(r["auc_roc"]) for r in bucket])
        pr = np.array([parse_float(r["auc_pr"]) for r in bucket])
        pk = np.array([parse_float(r["p_at_k"]) for r in bucket])
        n_pos = np.array([int(r["n_train_pos"]) for r in bucket])
        n_tokens = int(bucket[0]["n_tokens"])

        out.append({
            "feature_idx": feat,
            "feature_theme": FEATURE_THEMES[feat],
            "layer": layer,
            "n_seq": n_seq,
            "n_tokens": n_tokens,
            "n_subsamples": len(bucket),
            "n_train_pos_mean": float(n_pos.mean()),
            "auc_roc_mean": float(roc.mean()),
            "auc_roc_std": float(roc.std(ddof=1)) if len(bucket) > 1 else 0.0,
            "auc_roc_min": float(roc.min()),
            "auc_roc_max": float(roc.max()),
            "auc_pr_mean": float(pr.mean()),
            "auc_pr_std": float(pr.std(ddof=1)) if len(bucket) > 1 else 0.0,
            "auc_pr_min": float(pr.min()),
            "auc_pr_max": float(pr.max()),
            "p_at_k_mean": float(pk.mean()),
            "p_at_k_std": float(pk.std(ddof=1)) if len(bucket) > 1 else 0.0,
        })
    return out


def find_crossing(aggs: list[dict], feature_idx: int, layer: int,
                  metric: str, threshold: float) -> tuple[float | None, float | None]:
    """Smallest N at which the mean metric first reaches `threshold`.

    Linear interpolation between adjacent N points in log-N space. Returns
    (n_tokens_crossed, n_seq_crossed) or (None, None) if the curve never
    crosses (we return the saturation value separately as a flag).
    """
    points = [
        a for a in aggs
        if a["feature_idx"] == feature_idx and a["layer"] == layer
    ]
    points.sort(key=lambda a: a["n_seq"])
    if not points:
        return None, None

    vals = [(a["n_seq"], a["n_tokens"], a[metric]) for a in points]
    # If the smallest N already exceeds threshold, return the smallest N.
    if vals[0][2] >= threshold:
        return float(vals[0][1]), float(vals[0][0])
    # If even the largest N doesn't cross, return None.
    if vals[-1][2] < threshold:
        return None, None

    # Find the bracket and interpolate in log-N space.
    for (n_seq_lo, n_tok_lo, v_lo), (n_seq_hi, n_tok_hi, v_hi) in zip(vals, vals[1:]):
        if v_lo < threshold <= v_hi:
            if v_hi == v_lo:
                return float(n_tok_hi), float(n_seq_hi)
            t = (threshold - v_lo) / (v_hi - v_lo)
            log_lo, log_hi = np.log(n_tok_lo), np.log(n_tok_hi)
            n_tok = float(np.exp(log_lo + t * (log_hi - log_lo)))
            log_seq_lo, log_seq_hi = np.log(n_seq_lo), np.log(n_seq_hi)
            n_seq = float(np.exp(log_seq_lo + t * (log_seq_hi - log_seq_lo)))
            return n_tok, n_seq
    return None, None


def headline(aggs: list[dict]) -> list[dict]:
    rows = []
    for feat in FEATURE_INDICES:
        for layer in LAYERS:
            # Full-N reference numbers.
            full = next(
                (a for a in aggs if a["feature_idx"] == feat
                 and a["layer"] == layer and a["n_seq"] == 320),
                None,
            )
            if full is None:
                continue
            n_roc, n_seq_roc = find_crossing(aggs, feat, layer, "auc_roc_mean", ROC_THRESHOLD)
            n_pr, n_seq_pr = find_crossing(aggs, feat, layer, "auc_pr_mean", PR_THRESHOLD)
            rows.append({
                "feature_idx": feat,
                "feature_theme": FEATURE_THEMES[feat],
                "layer": layer,
                "full_auc_roc": full["auc_roc_mean"],
                "full_auc_pr": full["auc_pr_mean"],
                "n_tokens_at_roc_0.9": n_roc if n_roc is not None else "",
                "n_seq_at_roc_0.9": n_seq_roc if n_seq_roc is not None else "",
                "n_tokens_at_pr_0.5": n_pr if n_pr is not None else "",
                "n_seq_at_pr_0.5": n_seq_pr if n_seq_pr is not None else "",
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  (no rows; skipping {path.name})")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def print_headline_table(headline_rows: list[dict]) -> None:
    print("\nHeadline: smallest N (tokens) at which mean AUC crosses threshold")
    print("=" * 90)
    print(f"{'feature':<22} {'L':>3}  {'full_ROC':>9}  {'full_PR':>8}  "
          f"{'N@ROC>=0.9':>11}  {'N@PR>=0.5':>11}")
    print("-" * 90)
    for r in headline_rows:
        n_roc = r["n_tokens_at_roc_0.9"]
        n_pr = r["n_tokens_at_pr_0.5"]
        print(
            f"{r['feature_idx']:>5} {r['feature_theme']:<16} "
            f"{r['layer']:>3}  {r['full_auc_roc']:>9.3f}  {r['full_auc_pr']:>8.3f}  "
            f"{(f'{int(n_roc)}' if n_roc != '' else '  never'):>10}  "
            f"{(f'{int(n_pr)}' if n_pr != '' else '  never'):>10}"
        )


def main():
    if not RAW_CSV.exists():
        print(f"missing {RAW_CSV}; run scripts/step5_efficiency.py first")
        sys.exit(2)
    rows = load_rows()
    n_degen = sum(1 for r in rows if int(r["degenerate"]))
    print(f"loaded {len(rows)} raw rows ({n_degen} degenerate)")

    aggs = aggregate(rows)
    print(f"aggregated to {len(aggs)} (feature, layer, N) groups")
    write_csv(AGG_CSV, aggs)

    headline_rows = headline(aggs)
    write_csv(HEADLINE_CSV, headline_rows)
    print_headline_table(headline_rows)


if __name__ == "__main__":
    main()
