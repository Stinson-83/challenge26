"""
scoring.py
==========
The multi-score "scorecard" system: build several INDEPENDENT scores, normalise them onto
a common scale, and combine them intelligently into an overall profitability score.

This complements profitability.py: where that module encodes full equations, this module
exposes the individual Revenue / Cost / Risk / Engagement / Loyalty / Relationship scores
and several transparent combination strategies so the reader can switch philosophies and
see how the overall score moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config
from .feature_engineering import build_group_scores, percentile_rank
from .profitability import compute_subscores


def normalize_scores(sub: pd.DataFrame) -> pd.DataFrame:
    """Percentile-normalise each sub-score to [0,1] (already ~[0,1]; this re-ranks robustly)."""
    return sub.apply(percentile_rank)


def combine(sub: pd.DataFrame, strategy: str = "weighted", weights: dict | None = None) -> pd.Series:
    """
    Combine sub-scores into ONE overall profitability score in [0,1].

    strategies:
      'weighted'  : revenue + engagement + loyalty + relationship  MINUS cost + risk + attrition.
      'rank_mean' : mean of good ranks minus mean of bad ranks (unit-free, very robust).
      'multiplicative': revenue * (1-risk) * (1-attrition) - cost   (risk gates revenue).
    """
    w = weights or {
        "revenue_score": 1.00, "engagement_score": 0.10, "loyalty_score": 0.10,
        "relationship_score": 0.10, "cost_score": -0.15, "risk_score": -0.20,
        "attrition_score": -0.10,
    }
    good = ["revenue_score", "engagement_score", "loyalty_score", "relationship_score"]
    bad = ["cost_score", "risk_score", "attrition_score"]

    if strategy == "weighted":
        s = sum(w.get(c, 0.0) * sub[c] for c in sub.columns)
    elif strategy == "rank_mean":
        s = sub[good].mean(axis=1) - sub[bad].mean(axis=1)
    elif strategy == "multiplicative":
        s = (sub["revenue_score"] * (1 - 0.5 * sub["risk_score"])
             * (1 - 0.5 * sub["attrition_score"]) - 0.3 * sub["cost_score"])
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return percentile_rank(s).rename(f"overall_{strategy}")


def build_scorecard(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """
    Full scorecard: id + seven sub-scores + overall score under all three strategies.
    Convenient single call for reporting and dashboards.
    """
    cfg = cfg or load_config()
    gs = build_group_scores(df, cfg)
    sub = compute_subscores(gs)
    card = pd.DataFrame(index=df.index)
    if cfg.id_column in df.columns:
        card[cfg.id_column] = df[cfg.id_column].values
    for c in sub.columns:
        card[c] = sub[c].values
    for strat in ("weighted", "rank_mean", "multiplicative"):
        card[f"overall_{strat}"] = combine(sub, strat).values
    return card
