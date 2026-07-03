"""
dimensionality.py
=================
Latent-factor exploration: PCA, Factor Analysis and ICA on the robust group-scores.

Goal: check whether a small set of latent factors (e.g. a dominant "spend/affluence"
factor vs a "risk/revolve" factor vs an "engagement" factor) explains the data, and
whether the leading factor is a cleaner profitability axis than any single feature. The
first PC, sign-aligned to the spend group, is offered as an auxiliary ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config


def _matrix(gs: pd.DataFrame) -> pd.DataFrame:
    return gs.fillna(gs.median())


def run_pca(gs: pd.DataFrame, n_components: int = 5, seed: int = 42):
    from sklearn.decomposition import PCA
    X = _matrix(gs)
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=seed)
    comps = pca.fit_transform(X.values)
    loadings = pd.DataFrame(pca.components_.T, index=gs.columns,
                            columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    scores = pd.DataFrame(comps, index=gs.index, columns=loadings.columns)
    return {"scores": scores, "loadings": loadings,
            "explained_variance_ratio": pca.explained_variance_ratio_}


def run_factor_analysis(gs: pd.DataFrame, n_components: int = 4, seed: int = 42):
    from sklearn.decomposition import FactorAnalysis
    X = _matrix(gs)
    fa = FactorAnalysis(n_components=min(n_components, X.shape[1]), random_state=seed, max_iter=1000)
    scores = pd.DataFrame(fa.fit_transform(X.values), index=gs.index,
                          columns=[f"F{i+1}" for i in range(fa.n_components)])
    loadings = pd.DataFrame(fa.components_.T, index=gs.columns, columns=scores.columns)
    return {"scores": scores, "loadings": loadings}


def run_ica(gs: pd.DataFrame, n_components: int = 4, seed: int = 42):
    from sklearn.decomposition import FastICA
    X = _matrix(gs)
    ica = FastICA(n_components=min(n_components, X.shape[1]), random_state=seed,
                  max_iter=500, whiten="unit-variance")
    scores = pd.DataFrame(ica.fit_transform(X.values), index=gs.index,
                          columns=[f"IC{i+1}" for i in range(n_components)])
    loadings = pd.DataFrame(ica.mixing_, index=gs.columns, columns=scores.columns)
    return {"scores": scores, "loadings": loadings}


def spend_aligned_pc1(gs: pd.DataFrame, seed: int = 42) -> pd.Series:
    """First PC, sign-flipped so it correlates positively with the spend group — an
    unsupervised, model-free profitability axis to compare against the frameworks."""
    res = run_pca(gs, n_components=1, seed=seed)
    pc1 = res["scores"]["PC1"]
    if "spend" in gs.columns and pc1.corr(gs["spend"]) < 0:
        pc1 = -pc1
    from .feature_engineering import percentile_rank
    return percentile_rank(pc1).rename("pca_rank")
