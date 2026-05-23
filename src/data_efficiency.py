"""Sequence-level subsampling for the Step 5 data-efficiency sweeps.

Given the Step 4 shared train fold (320 sequences, 81,600 BOS-masked tokens),
draw a subsample of `n_seq` train sequences and return the corresponding row
positions in the already-materialized X_train tensor.

The subsample unit is **sequences** (not tokens). Step 3's cache layout puts
sequences sequence-major after BOS masking; train sequence position `p` in
X_train occupies rows [p*255, (p+1)*255). We sample WITHOUT replacement from
the 320 train sequence positions.

A new scaler is fit on each subsample (rather than reusing the full-fold
scaler) so the data-efficiency claim is honest: "with N tokens of probe
data" means N tokens were used end-to-end, including statistics for
standardization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import SEQ_LEN, Split


# 9 points, log-spaced and dense at the low end (where the curve is steep).
# Tokens = N_seq * 255 (BOS removed). Top value 320 is the full train fold.
N_SEQ_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 320]

SUBSAMPLES_PER_N = 5
SUBSAMPLE_BASE_SEED = 10  # distinct from the split seed (0) used in Step 4


@dataclass(frozen=True)
class Subsample:
    n_seq: int
    seed: int
    seq_positions: np.ndarray  # (n_seq,) positions in [0, 320)
    local_rows: np.ndarray     # (n_seq * 255,) positions in [0, 81600)

    @property
    def n_tokens(self) -> int:
        return int(self.local_rows.shape[0])


def make_subsample(split: Split, n_seq: int, seed: int) -> Subsample:
    """Sample n_seq distinct train-sequence positions and expand to row indices.

    Positions are drawn from [0, len(split.train_seq_ids)); i.e. positions
    within the already-built train fold (320 sequences). The returned
    `local_rows` indexes directly into a materialized X_train of shape
    (81600, D_MODEL).
    """
    n_train_seqs = int(split.train_seq_ids.shape[0])
    if not (1 <= n_seq <= n_train_seqs):
        raise ValueError(f"n_seq={n_seq} outside [1, {n_train_seqs}]")
    rng = np.random.default_rng(seed)
    positions = rng.choice(n_train_seqs, size=n_seq, replace=False)
    positions.sort()

    rows_per_seq = SEQ_LEN - 1  # 255, after BOS mask
    offsets = np.arange(rows_per_seq, dtype=np.int64)  # 0..254
    local_rows = (positions.astype(np.int64)[:, None] * rows_per_seq + offsets).reshape(-1)
    return Subsample(
        n_seq=int(n_seq),
        seed=int(seed),
        seq_positions=positions,
        local_rows=local_rows,
    )


def subsample_seeds_for(n_seq: int, n_train_seqs: int, n_subsamples: int = SUBSAMPLES_PER_N) -> list[int]:
    """Pick subsample seeds; when N == n_train_seqs there's only one possible
    subsample (the full fold), so we return a single seed.
    """
    if n_seq >= n_train_seqs:
        return [SUBSAMPLE_BASE_SEED]
    return [SUBSAMPLE_BASE_SEED + i for i in range(n_subsamples)]
