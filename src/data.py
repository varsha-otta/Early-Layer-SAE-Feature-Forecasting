"""Cache loading and train/test split helpers for Step 4.

The Step 3 cache is laid out flat row-major with sequence-major ordering:
row i belongs to sequence `i // SEQ_LEN`, position `i % SEQ_LEN`. Position 0
of every sequence is BOS. We split 80/20 by sequence index (not token index)
to avoid leaking positional context; BOS positions are masked because they
carry no document-specific signal.

Memory rule (laptop has 8 GB system RAM): the residual cache is fp16 on disk
(~470 MB per layer). We mmap it lazily and only materialize one layer's worth
of fp32 train + test data at a time (~940 MB peak per layer).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "v1"

N_SEQUENCES = 400
SEQ_LEN = 256
N_TOKENS = N_SEQUENCES * SEQ_LEN  # 102,400
D_MODEL = 2304
LAYERS = [5, 8, 12, 20]
FEATURE_INDICES = [9989, 817, 12730, 892, 1031]
FEATURE_THEMES = {
    9989: "refusal",
    817: "deception",
    12730: "ethics",
    892: "sycophancy-adj",
    1031: "harm",
}

TRAIN_FRAC = 0.8
SPLIT_SEED = 0


def load_metadata(cache_dir: Path = CACHE_DIR) -> dict:
    return json.loads((cache_dir / "metadata.json").read_text())


@dataclass(frozen=True)
class Split:
    """Token-row indices for the train/test split, already BOS-masked.

    `train_rows` and `test_rows` are int64 arrays of row indices into the
    flat cache arrays. Lengths: 320*255 = 81,600 train, 80*255 = 20,400 test.
    """
    train_seq_ids: np.ndarray  # (320,)
    test_seq_ids: np.ndarray  # (80,)
    train_rows: np.ndarray  # (81_600,)
    test_rows: np.ndarray  # (20_400,)

    @property
    def n_train(self) -> int:
        return int(self.train_rows.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test_rows.shape[0])


def make_split(seed: int = SPLIT_SEED) -> Split:
    """Sequence-level 80/20 split with BOS positions removed.

    Same seed → same partition across all 20 probes (the Step 4 design calls
    for one shared split so cross-(feature, layer) AUC numbers are directly
    commensurable).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_SEQUENCES)
    n_train_seq = int(round(TRAIN_FRAC * N_SEQUENCES))
    train_seq = np.sort(perm[:n_train_seq])
    test_seq = np.sort(perm[n_train_seq:])

    def rows_for(seq_ids: np.ndarray) -> np.ndarray:
        # For each seq id s, the rows are [s*256 + 1, ..., s*256 + 255]
        # (BOS at offset 0 is dropped).
        offsets = np.arange(1, SEQ_LEN, dtype=np.int64)  # (255,)
        base = (seq_ids.astype(np.int64) * SEQ_LEN)[:, None]  # (k, 1)
        return (base + offsets[None, :]).reshape(-1)

    return Split(
        train_seq_ids=train_seq,
        test_seq_ids=test_seq,
        train_rows=rows_for(train_seq),
        test_rows=rows_for(test_seq),
    )


def mmap_residual(layer: int, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Memory-map a single residual-stream layer (fp16, shape (N_TOKENS, D_MODEL))."""
    return np.load(cache_dir / f"resid_layer_{layer}.npy", mmap_mode="r")


def mmap_feature_acts(cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Memory-map the 5-feature SAE activations (fp16, shape (N_TOKENS, 5))."""
    return np.load(cache_dir / "feature_acts.npy", mmap_mode="r")


def load_layer_split(layer: int, split: Split, cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one layer's train+test residuals as fp32 ndarrays.

    Returns (X_train, X_test) with shapes (n_train, D_MODEL) and (n_test, D_MODEL).
    Peak transient memory: ~940 MB.
    """
    src = mmap_residual(layer, cache_dir)
    X_train = np.asarray(src[split.train_rows], dtype=np.float32)
    X_test = np.asarray(src[split.test_rows], dtype=np.float32)
    return X_train, X_test


def load_labels(split: Split, cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Binary fire labels for all 5 target features, train and test.

    Returns (y_train, y_test) with shapes (n_train, 5) and (n_test, 5), dtype uint8.
    A token "fires" for feature f iff feature_acts[token, col(f)] > 0.
    """
    src = mmap_feature_acts(cache_dir)
    y_train = (np.asarray(src[split.train_rows], dtype=np.float32) > 0).astype(np.uint8)
    y_test = (np.asarray(src[split.test_rows], dtype=np.float32) > 0).astype(np.uint8)
    return y_train, y_test


def fit_standardizer(X_train: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature z-score statistics fitted on the training fold.

    Returns (mean, scale) both shaped (D_MODEL,), float32. `scale` floors small
    stds at `eps` to avoid div-by-zero on residual dims that are constant on the
    train fold (none expected with these sizes, but defensive).
    """
    mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(std < eps, 1.0, std).astype(np.float32)
    return mean, scale


def apply_standardizer(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply (X - mean) / scale in place when possible. Returns the same buffer."""
    np.subtract(X, mean, out=X)
    np.divide(X, scale, out=X)
    return X
