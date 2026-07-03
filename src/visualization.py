"""
visualization.py
================
Professional, self-contained plotting. Every function writes a PNG to `outputs/` and
returns its path. All plots are theme-neutral, labelled, and safe on large data
(subsampling where needed). No plot ever calls plt.show() so the module is headless-safe.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")               # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
_PALETTE = "viridis"


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def correlation_heatmap(corr: pd.DataFrame, outdir: str, name="correlation_heatmap.png") -> str:
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax,
                cbar_kws={"shrink": 0.6}, annot=False)
    ax.set_title("Feature correlation (Spearman)")
    return _save(fig, outdir, name)


def distributions(df: pd.DataFrame, cols: list[str], outdir: str,
                  name="feature_distributions.png", sample=20000) -> str:
    cols = cols[:16]
    d = df.sample(min(sample, len(df)), random_state=0) if len(df) > sample else df
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    for ax, c in zip(np.array(axes).ravel(), cols):
        x = pd.to_numeric(d[c], errors="coerce").dropna()
        # Log axis for heavy-tailed positive money columns.
        if x.min() >= 0 and x.max() > 0 and (x.max() / (x.median() + 1e-9)) > 50:
            sns.histplot(np.log1p(x), bins=40, ax=ax, color=sns.color_palette(_PALETTE)[3])
            ax.set_xlabel(f"log1p({c})")
        else:
            sns.histplot(x, bins=40, ax=ax, color=sns.color_palette(_PALETTE)[2])
            ax.set_xlabel(c)
    for ax in np.array(axes).ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Feature distributions", y=1.02)
    return _save(fig, outdir, name)


def pca_scatter(pc_scores: pd.DataFrame, color: pd.Series, outdir: str,
                name="pca_scatter.png", sample=20000, label="profitability rank") -> str:
    idx = pc_scores.index
    if len(idx) > sample:
        idx = pd.Index(np.random.default_rng(0).choice(idx.values, sample, replace=False))
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(pc_scores.loc[idx, "PC1"], pc_scores.loc[idx, "PC2"],
                    c=color.loc[idx], cmap=_PALETTE, s=6, alpha=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA projection")
    fig.colorbar(sc, ax=ax, label=label, shrink=0.7)
    return _save(fig, outdir, name)


def cluster_scatter(pc_scores: pd.DataFrame, labels: pd.Series, outdir: str,
                    name="cluster_scatter.png", sample=20000) -> str:
    idx = pc_scores.index
    if len(idx) > sample:
        idx = pd.Index(np.random.default_rng(0).choice(idx.values, sample, replace=False))
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(pc_scores.loc[idx, "PC1"], pc_scores.loc[idx, "PC2"],
                    c=labels.loc[idx], cmap="tab10", s=6, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Clusters in PCA space")
    fig.colorbar(sc, ax=ax, label="cluster", shrink=0.7)
    return _save(fig, outdir, name)


def outlier_plot(pc_scores: pd.DataFrame, anomaly: pd.Series, outdir: str,
                 name="outlier_plot.png", sample=20000) -> str:
    idx = pc_scores.index
    if len(idx) > sample:
        idx = pd.Index(np.random.default_rng(0).choice(idx.values, sample, replace=False))
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(pc_scores.loc[idx, "PC1"], pc_scores.loc[idx, "PC2"],
                    c=anomaly.loc[idx], cmap="magma", s=6, alpha=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Anomaly score (Isolation Forest)")
    fig.colorbar(sc, ax=ax, label="anomaly", shrink=0.7)
    return _save(fig, outdir, name)


def score_distribution(scores: pd.DataFrame, outdir: str,
                       name="profitability_distributions.png") -> str:
    fw = [c for c in scores.columns if c.startswith("score__")]
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in fw:
        x = scores[c]
        xr = (x - x.min()) / (x.max() - x.min() + 1e-9)
        sns.kdeplot(xr, ax=ax, label=c.replace("score__", ""), fill=False, lw=2)
    ax.set_title("Profitability score distributions (min-max scaled)")
    ax.set_xlabel("scaled score"); ax.legend(fontsize=10)
    return _save(fig, outdir, name)


def framework_overlap_heatmap(overlap_df: pd.DataFrame, outdir: str,
                              name="framework_overlap.png") -> str:
    fw = sorted(set(overlap_df["a"]) | set(overlap_df["b"]))
    m = pd.DataFrame(np.eye(len(fw)), index=fw, columns=fw)
    for _, r in overlap_df.iterrows():
        m.loc[r["a"], r["b"]] = r["top_overlap"]
        m.loc[r["b"], r["a"]] = r["top_overlap"]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(m, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax, square=True)
    ax.set_title("Top-20% set overlap between frameworks")
    return _save(fig, outdir, name)


def cluster_profile_bar(profile: pd.DataFrame, outdir: str, name="cluster_profitability.png") -> str:
    p = profile[profile.index >= 0] if profile.index.min() < 0 else profile
    fig, ax = plt.subplots(figsize=(10, 6))
    order = p.sort_values("profitability_rank", ascending=False)
    ax.bar(order.index.astype(str), order["profitability_rank"],
           color=sns.color_palette(_PALETTE, len(order)))
    ax.set_xlabel("cluster"); ax.set_ylabel("mean profitability rank")
    ax.set_title("Cluster mean profitability (KMeans)")
    return _save(fig, outdir, name)
