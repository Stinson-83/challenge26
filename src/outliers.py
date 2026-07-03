"""
outliers.py
===========
Unusual-customer detection with Isolation Forest and Local Outlier Factor, plus a
business read on WHICH tail an outlier sits in.

Anomalies here are not noise to discard — they are the ultra-high-spend whales (extremely
profitable) and the high-risk/high-benefit-extraction customers (extremely unprofitable).
We tag each outlier with the sign of its profitability rank so the reader can see which
extreme it belongs to.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config


def _matrix(gs: pd.DataFrame) -> np.ndarray:
    return gs.fillna(gs.median()).values


def isolation_forest_scores(gs: pd.DataFrame, seed: int = 42, contamination: float = 0.02) -> pd.Series:
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(random_state=seed, contamination=contamination, n_estimators=200)
    iso.fit(_matrix(gs))
    # Higher = more anomalous.
    return pd.Series(-iso.score_samples(_matrix(gs)), index=gs.index, name="iforest_anomaly")


def lof_scores(gs: pd.DataFrame, sample: int = 50000, seed: int = 42, n_neighbors: int = 20) -> pd.Series:
    from sklearn.neighbors import LocalOutlierFactor
    rng = np.random.default_rng(seed)
    idx = gs.index.values
    if len(idx) > sample:
        idx = rng.choice(idx, size=sample, replace=False)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit_predict(gs.loc[idx].fillna(gs.median()).values)
    out = pd.Series(np.nan, index=gs.index, name="lof_anomaly")
    out.loc[idx] = -lof.negative_outlier_factor_
    return out


def analyze_outliers(gs: pd.DataFrame, primary_rank: pd.Series, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    cc = cfg.raw.get("clustering", {})
    iso = isolation_forest_scores(gs, cfg.seed)
    lof = lof_scores(gs, sample=int(cc.get("sample_for_heavy", 50000)), seed=cfg.seed)

    k = max(1, int(0.02 * len(gs)))
    top_iso = iso.nlargest(k).index
    # Split anomalies by profitability tail.
    hi = primary_rank.loc[top_iso] >= 0.5
    summary = {
        "n_flagged_iforest": int(k),
        "anomalies_profitable_tail": int(hi.sum()),
        "anomalies_unprofitable_tail": int((~hi).sum()),
        "mean_rank_of_anomalies": float(primary_rank.loc[top_iso].mean()),
    }
    scores = pd.DataFrame({"iforest_anomaly": iso, "lof_anomaly": lof,
                           "profitability_rank": primary_rank})
    return {"summary": summary, "scores": scores}
