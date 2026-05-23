"""Step 7: generate the 3 writeup figures from existing CSVs.

  Figure 1 (`fig1_auc_by_layer.png`): linear-probe AUC-ROC and AUC-PR vs layer,
    one line per feature, bootstrap 95% CI bands. Source: Step 4 metrics CSV.

  Figure 2 (`fig2_data_efficiency.png`): AUC-ROC vs N tokens, one panel per
    feature, one line per layer, std band across subsamples, horizontal
    threshold at AUC=0.9. Source: Step 5 aggregate CSV.

  Figure 3 (`fig3_id_vs_ood.png`): in-distribution vs out-of-distribution
    AUC-ROC at each (feature, layer); paired bars with bootstrap CIs.
    Sources: Step 4 metrics CSV (ID baseline) + Step 6 OOD CSV.

All PNGs saved at 150 dpi to `docs/figures/`. Idempotent; re-run any time.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_INDICES, FEATURE_THEMES, LAYERS

RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

STEP4_CSV = RESULTS / "step4_probe_metrics.csv"
STEP5_AGG_CSV = RESULTS / "step5_efficiency_aggregate.csv"
STEP6_CSV = RESULTS / "step6_ood_metrics.csv"

# Consistent feature color across all three figures.
FEATURE_COLORS = {
    9989: "#1f77b4",   # refusal; blue
    817: "#ff7f0e",    # deception; orange
    12730: "#2ca02c",  # ethics; green
    892: "#d62728",    # sycophancy-adj; red
    1031: "#9467bd",   # harm; purple
}

# Sequential colormap for layers (depth → darker).
LAYER_COLORS = {5: "#fdae61", 8: "#f46d43", 12: "#d73027", 20: "#a50026"}


def label(idx: int) -> str:
    return f"{FEATURE_THEMES[idx]} ({idx})"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(x):
    return float(x) if x not in ("", None) else float("nan")


# -----------------------------------------------------------------------
# Figure 1: AUC-ROC and AUC-PR vs layer, one line per feature
# -----------------------------------------------------------------------


def figure1():
    rows = [r for r in load_csv(STEP4_CSV) if r["probe_type"] == "linear"]
    by_feature = {
        idx: sorted(
            [r for r in rows if int(r["feature_idx"]) == idx],
            key=lambda r: int(r["layer"]),
        )
        for idx in FEATURE_INDICES
    }

    fig, (axR, axP) = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx in FEATURE_INDICES:
        pts = by_feature[idx]
        layers = [int(p["layer"]) for p in pts]
        roc = np.array([f(p["auc_roc"]) for p in pts])
        roc_lo = np.array([f(p["auc_roc_lo"]) for p in pts])
        roc_hi = np.array([f(p["auc_roc_hi"]) for p in pts])
        pr = np.array([f(p["auc_pr"]) for p in pts])
        pr_lo = np.array([f(p["auc_pr_lo"]) for p in pts])
        pr_hi = np.array([f(p["auc_pr_hi"]) for p in pts])

        c = FEATURE_COLORS[idx]
        axR.plot(layers, roc, marker="o", color=c, label=label(idx), lw=1.8)
        axR.fill_between(layers, roc_lo, roc_hi, color=c, alpha=0.12)
        axP.plot(layers, pr, marker="o", color=c, label=label(idx), lw=1.8)
        axP.fill_between(layers, pr_lo, pr_hi, color=c, alpha=0.12)

    for ax, title, ylim in [
        (axR, "AUC-ROC", (0.83, 1.005)),
        (axP, "AUC-PR", (0, 1.0)),
    ]:
        ax.set_xlabel("Layer (Gemma-2-2B)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(LAYERS)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylim)

    axR.axhline(0.9, color="grey", ls="--", lw=0.8, alpha=0.6, label="0.9 threshold")
    axR.legend(loc="lower right", fontsize=8, framealpha=0.95)
    axP.legend(loc="upper left", fontsize=8, framealpha=0.95)

    fig.suptitle(
        "Figure 1. Linear-probe AUC vs layer, by feature (Pile-10k test fold). "
        "Bands are 95% bootstrap CIs.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = FIGS / "fig1_auc_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------
# Figure 2: data-efficiency curves
# -----------------------------------------------------------------------


SAE_TRAIN_TOKENS = 4_000_000_000  # GemmaScope: ~4B tokens (Lieberum et al. 2024)


def figure2():
    rows = load_csv(STEP5_AGG_CSV)
    by_fl: dict[tuple[int, int], list[dict]] = {}
    for r in rows:
        key = (int(r["feature_idx"]), int(r["layer"]))
        by_fl.setdefault(key, []).append(r)

    # Layout: 2 rows × 3 cols; the 6th panel is the SAE-reference callout
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    flat = axes.flatten()

    for ax_i, idx in enumerate(FEATURE_INDICES):
        ax = flat[ax_i]
        for layer in LAYERS:
            pts = sorted(by_fl.get((idx, layer), []), key=lambda r: int(r["n_seq"]))
            if not pts:
                continue
            x = np.array([int(p["n_tokens"]) for p in pts])
            mean = np.array([f(p["auc_roc_mean"]) for p in pts])
            std = np.array([f(p["auc_roc_std"]) for p in pts])
            c = LAYER_COLORS[layer]
            ax.plot(x, mean, marker="o", color=c, lw=1.5, label=f"L{layer}", ms=4)
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)

        ax.axhline(0.9, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlim(400, 200_000)
        ax.set_ylim(0.45, 1.02)
        ax.set_title(label(idx), fontsize=10)
        ax.set_xlabel("N tokens (probe training)")
        if ax_i % 3 == 0:
            ax.set_ylabel("AUC-ROC")
        ax.grid(True, alpha=0.3)
        if ax_i == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # 6th panel: SAE-scale reference
    ref = flat[5]
    ref.set_xscale("log")
    ref.set_xlim(400, 1e10)
    ref.set_ylim(0.45, 1.02)
    # Place markers showing the orders-of-magnitude gap
    ref.axvspan(500, 100_000, alpha=0.15, color="#1f77b4", label="probe range")
    ref.axvline(SAE_TRAIN_TOKENS, color="black", ls="-", lw=2,
                label=f"GemmaScope SAE\n(~{SAE_TRAIN_TOKENS/1e9:.0f}B tokens)")
    ref.set_xlabel("Tokens (log scale)")
    ref.set_title("Scale reference: probe vs SAE training corpus", fontsize=10)
    ref.set_yticks([])
    ref.legend(loc="center left", fontsize=9)
    ref.grid(True, alpha=0.3, which="both", axis="x")

    fig.suptitle(
        "Figure 2. Data efficiency by feature and layer. "
        "Bands are ±1 std across 5 subsamples; dashed line marks AUC-ROC = 0.9.",
        fontsize=10, y=1.00,
    )
    fig.tight_layout()
    out = FIGS / "fig2_data_efficiency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


# -----------------------------------------------------------------------
# Figure 3: ID vs OOD AUC-ROC, paired bars
# -----------------------------------------------------------------------


def figure3():
    ood_rows = load_csv(STEP6_CSV)
    # Index: (feature, layer) → row
    by_key = {(int(r["feature_idx"]), int(r["layer"])): r for r in ood_rows}

    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2), sharey=True)

    x_layer = np.arange(len(LAYERS))
    bar_w = 0.38

    for ax_i, idx in enumerate(FEATURE_INDICES):
        ax = axes[ax_i]
        id_vals, ood_vals = [], []
        id_err_lo, id_err_hi = [], []
        ood_err_lo, ood_err_hi = [], []
        for layer in LAYERS:
            r = by_key.get((idx, layer))
            id_vals.append(f(r["id_auc_roc"]))
            ood_vals.append(f(r["ood_auc_roc"]))
            ood_err_lo.append(f(r["ood_auc_roc"]) - f(r["ood_auc_roc_lo"]))
            ood_err_hi.append(f(r["ood_auc_roc_hi"]) - f(r["ood_auc_roc"]))
            # ID CIs from Step 4 - not joined into Step 6 CSV; omit ID error bars
            id_err_lo.append(0.0)
            id_err_hi.append(0.0)

        ax.bar(x_layer - bar_w / 2, id_vals, bar_w,
               color="#4C72B0", label="ID (Pile)")
        ax.bar(x_layer + bar_w / 2, ood_vals, bar_w,
               yerr=[ood_err_lo, ood_err_hi],
               color="#DD8452", label="OOD (HH-RLHF red-team)",
               error_kw={"elinewidth": 1, "ecolor": "black"})

        ax.axhline(0.5, color="grey", ls=":", lw=0.8, alpha=0.6)
        ax.set_xticks(x_layer)
        ax.set_xticklabels([f"L{L}" for L in LAYERS])
        ax.set_ylim(0.5, 1.02)
        ax.set_title(label(idx), fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if ax_i == 0:
            ax.set_ylabel("AUC-ROC")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.suptitle(
        "Figure 3. Generalization: probes trained on Pile-10k, evaluated on Pile (ID) vs "
        "HH-RLHF red-team (OOD). Error bars: 95% bootstrap on OOD.",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = FIGS / "fig3_id_vs_ood.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main():
    print("Generating Step 7 figures…")
    figure1()
    figure2()
    figure3()
    print("Done.")


if __name__ == "__main__":
    main()
