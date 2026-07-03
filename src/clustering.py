"""
clustering.py
=============
Unsupervised segmentation + per-cluster profitability profiling.

We fit KMeans, Gaussian Mixture and (on a subsample) Agglomerative clustering and DBSCAN
on the robust group-scores, then profile each cluster on the interpretable sub-scores and
the primary profitability rank. This tells us WHICH segments concentrate the profitable
tail — a sanity check on the framework and a source of cluster-relative features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, load_config


def _matrix(gs: pd.DataFrame) -> np.ndarray:
    return gs.fillna(gs.median()).values


def fit_kmeans(gs: pd.DataFrame, k: int, seed: int = 42) -> pd.Series:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return pd.Series(km.fit_predict(_matrix(gs)), index=gs.index, name="kmeans")


def fit_gmm(gs: pd.DataFrame, k: int, seed: int = 42) -> pd.Series:
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full",
                         reg_covar=1e-4, max_iter=200)
    return pd.Series(gm.fit_predict(_matrix(gs)), index=gs.index, name="gmm")


def fit_hierarchical(gs: pd.DataFrame, k: int, sample: int, seed: int = 42) -> pd.Series:
    """Agglomerative on a subsample (O(n^2)); unassigned rows -> -1."""
    from sklearn.cluster import AgglomerativeClustering
    rng = np.random.default_rng(seed)
    idx = gs.index.values
    if len(idx) > sample:
        idx = rng.choice(idx, size=sample, replace=False)
    sub = gs.loc[idx]
    lab = AgglomerativeClustering(n_clusters=k).fit_predict(_matrix(sub))
    out = pd.Series(-1, index=gs.index, name="hierarchical")
    out.loc[idx] = lab
    return out


def fit_dbscan(gs: pd.DataFrame, eps: float, min_samples: int, sample: int, seed: int = 42) -> pd.Series:
    from sklearn.cluster import DBSCAN
    rng = np.random.default_rng(seed)
    idx = gs.index.values
    if len(idx) > sample:
        idx = rng.choice(idx, size=sample, replace=False)
    sub = gs.loc[idx]
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(_matrix(sub))
    out = pd.Series(-2, index=gs.index, name="dbscan")     # -2 = not evaluated
    out.loc[idx] = lab                                     # -1 = DBSCAN noise
    return out


def fit_all(gs: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    cc = cfg.raw.get("clustering", {})
    sample = int(cc.get("sample_for_heavy", 20000))
    hier_sample = int(cc.get("hierarchical_sample", 5000))
    labels = pd.DataFrame(index=gs.index)
    labels["kmeans"] = fit_kmeans(gs, cc.get("kmeans_k", 6), cfg.seed)
    labels["gmm"] = fit_gmm(gs, cc.get("gmm_k", 6), cfg.seed)
    labels["hierarchical"] = fit_hierarchical(gs, cc.get("kmeans_k", 6), hier_sample, cfg.seed)
    labels["dbscan"] = fit_dbscan(gs, cc.get("dbscan_eps", 1.5),
                                  cc.get("dbscan_min_samples", 50), sample, cfg.seed)
    return labels


def profile_clusters(labels: pd.Series, gs: pd.DataFrame, subscores: pd.DataFrame,
                     primary_rank: pd.Series) -> pd.DataFrame:
    """Mean group-scores + sub-scores + mean profitability rank per cluster, sorted by profitability."""
    df = gs.join(subscores).copy()
    df["profitability_rank"] = primary_rank
    df["cluster"] = labels.values
    agg = df.groupby("cluster").mean(numeric_only=True)
    agg["n"] = df.groupby("cluster").size()
    agg["share"] = agg["n"] / len(df)
    return agg.sort_values("profitability_rank", ascending=False)
