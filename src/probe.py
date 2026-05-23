"""Single-probe trainers for Step 4.

Linear probe = sklearn LogisticRegression with class_weight='balanced'.
MLP sanity = tiny PyTorch 2-layer net trained with BCEWithLogitsLoss.

Both probes expect the residuals to already be standardized (mean=0, std=1
per dim); done by the orchestrator so the same scaler is shared across all
5 features for a given layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------------------
# Linear probe
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LinearProbeFit:
    coef: np.ndarray  # (D,)
    intercept: float
    C: float
    n_iter: int

    def score(self, X: np.ndarray) -> np.ndarray:
        """Decision function (logit). Higher = more positive."""
        return X @ self.coef + self.intercept


def fit_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> LinearProbeFit:
    """Fit an L2-regularized logistic regression with balanced class weights."""
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=max_iter,
        tol=tol,
    )
    clf.fit(X_train, y_train.astype(np.int8))
    return LinearProbeFit(
        coef=clf.coef_[0].astype(np.float32),
        intercept=float(clf.intercept_[0]),
        C=C,
        n_iter=int(clf.n_iter_[0]),
    )


# ----------------------------------------------------------------------
# MLP probe
# ----------------------------------------------------------------------


class TinyMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


@dataclass(frozen=True)
class MLPProbeFit:
    state_dict: dict  # tensor weights as numpy arrays for portable serialization
    hidden: int
    dropout: float
    epochs: int
    final_train_loss: float

    def score(self, X: np.ndarray) -> np.ndarray:
        """Forward pass to logits using the saved weights (no torch grad)."""
        W1 = self.state_dict["fc1.weight"]  # (H, D)
        b1 = self.state_dict["fc1.bias"]    # (H,)
        W2 = self.state_dict["fc2.weight"]  # (1, H)
        b2 = self.state_dict["fc2.bias"]    # (1,)
        h = X @ W1.T + b1
        np.maximum(h, 0, out=h)  # ReLU (dropout is eval-time identity)
        return (h @ W2.T + b2).squeeze(-1)


def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden: int = 128,
    dropout: float = 0.1,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
) -> MLPProbeFit:
    """Train the sanity-check MLP on CPU."""
    torch.manual_seed(seed)

    d_in = int(X_train.shape[1])
    n_pos = int(y_train.sum())
    n_neg = int(y_train.shape[0] - n_pos)
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 1.0)], dtype=torch.float32)

    model = TinyMLP(d_in=d_in, hidden=hidden, dropout=dropout)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Keep data as a single CPU float32 tensor; mini-batches are index slices.
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train.astype(np.float32))
    n = X_t.shape[0]
    rng = np.random.default_rng(seed)

    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
        order = rng.permutation(n)
        running = 0.0
        nbatch = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            running += float(loss.item())
            nbatch += 1
        final_loss = running / max(nbatch, 1)

    model.eval()
    state = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    return MLPProbeFit(
        state_dict=state,
        hidden=hidden,
        dropout=dropout,
        epochs=epochs,
        final_train_loss=final_loss,
    )
