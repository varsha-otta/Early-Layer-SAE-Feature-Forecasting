"""Probe evaluation: AUC-ROC, AUC-PR, precision@k, and stratified bootstrap CIs.

Positives are rare (0.5-1.4% of tokens), so PR-AUC is the more informative
headline and ROC-AUC supplements it. Bootstrap is stratified by label so each
resample has the same positive/negative counts as the original test fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class MetricCI:
    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class ProbeMetrics:
    auc_roc: MetricCI
    auc_pr: MetricCI
    precision_at_k: MetricCI  # k = number of test positives
    n_test: int
    n_test_pos: int


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top-k scored tokens that are actual positives.

    With k = n_test_pos, this is a balanced precision/recall measure: if the
    probe ranks perfectly, p@k = 1.0; if it ranks randomly, p@k ≈ base rate.
    """
    if k <= 0 or k > scores.shape[0]:
        return float("nan")
    # argpartition is O(n) vs argsort's O(n log n); we don't need ties broken.
    top_idx = np.argpartition(-scores, k - 1)[:k]
    return float(y_true[top_idx].mean())


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC that returns NaN on degenerate resamples (all same label)."""
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> ProbeMetrics:
    """Compute metrics with stratified bootstrap 95% CIs.

    Stratified resampling: positive indices and negative indices are resampled
    independently (each with replacement to its original count), then merged.
    This preserves the test fold's base rate exactly in every resample and
    matches how the linear-probe baseline literature typically reports CIs.
    """
    y_true = np.asarray(y_true, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape:
        raise ValueError(f"shape mismatch: y={y_true.shape} scores={scores.shape}")

    n_test = int(y_true.shape[0])
    n_pos = int(y_true.sum())

    point_roc = _safe_roc_auc(y_true, scores)
    point_pr = _safe_pr_auc(y_true, scores)
    point_pk = precision_at_k(y_true, scores, n_pos)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)
    roc_samples = np.empty(n_bootstrap)
    pr_samples = np.empty(n_bootstrap)
    pk_samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        bp = rng.choice(pos_idx, size=pos_idx.shape[0], replace=True)
        bn = rng.choice(neg_idx, size=neg_idx.shape[0], replace=True)
        idx = np.concatenate([bp, bn])
        yb = y_true[idx]
        sb = scores[idx]
        roc_samples[b] = _safe_roc_auc(yb, sb)
        pr_samples[b] = _safe_pr_auc(yb, sb)
        pk_samples[b] = precision_at_k(yb, sb, n_pos)

    def ci(point: float, samples: np.ndarray) -> MetricCI:
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return MetricCI(point, float("nan"), float("nan"))
        lo, hi = np.percentile(finite, [2.5, 97.5])
        return MetricCI(point=float(point), lo=float(lo), hi=float(hi))

    return ProbeMetrics(
        auc_roc=ci(point_roc, roc_samples),
        auc_pr=ci(point_pr, pr_samples),
        precision_at_k=ci(point_pk, pk_samples),
        n_test=n_test,
        n_test_pos=n_pos,
    )
