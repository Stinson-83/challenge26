"""
feature_engineering.py
======================
Business-driven feature construction + the robust group-scores that every profitability
framework consumes.

Design principle
----------------
Because the exact column meanings are uncertain, we never feed raw dollar magnitudes into
a weighted sum (a single 500K-dollar outlier would dominate). Instead every signal is
converted to a **robust percentile score in [0,1]** (rank-based, immune to scale/skew/
outliers) and sign-corrected so that "higher = more profitable". Weighted sums of these
comparable [0,1] scores are stable exactly where the leaderboard is decided — the 80th
percentile boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import Config, load_config
from .preprocessing import ColumnRole, feature_columns, infer_column_roles


# =======================================================================================
# Robust primitives
# =======================================================================================
def percentile_rank(s: pd.Series) -> pd.Series:
    """Average-rank percentile in [0,1]. Ties share the mean rank. NaNs -> median rank."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    filled = x.fillna(x.median())
    r = rankdata(filled.values, method="average")
    return pd.Series((r - 1) / (len(r) - 1 if len(r) > 1 else 1), index=s.index)


def winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    ql, qh = x.quantile(lo), x.quantile(hi)
    return x.clip(ql, qh)


def log1p_signed(s: pd.Series) -> pd.Series:
    """log1p that tolerates the negative values in net-spend columns."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return np.sign(x) * np.log1p(np.abs(x))


def robust_z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else (x.std() or 1.0)
    return (x - med) / scale


# =======================================================================================
# Group scores — the atoms of every framework
# =======================================================================================
def money_score(df: pd.DataFrame, cols: list[str], hi_pct: float = 0.995) -> pd.Series:
    """
    Robust [0,1] score for a MONETARY bucket: sum the RAW (lightly upper-winsorised,
    non-negative) dollars across columns, then percentile-rank.

    Summing raw dollars — not per-column logs — preserves total-spend ordering, which is
    what interchange revenue actually is (proportional to total charge volume). Negatives
    (returns in net-spend columns) are floored at 0 (== max(x,0)); a light upper winsor at
    the 99.5th pct only tames absurd single-column outliers without demoting genuine whales
    (ranking is invariant to any monotone transform, so the log is unnecessary here).
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.5, index=df.index)
    total = pd.Series(0.0, index=df.index)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
        cap = x.quantile(hi_pct)
        if cap > 0:
            x = x.clip(upper=cap)
        total = total + x
    return percentile_rank(total)


def rankmean_score(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Robust [0,1] score for a heterogeneous COUNT/SCORE bucket: mean of per-column
    percentile ranks (unit-free; right when columns are on different, bounded scales)."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.5, index=df.index)
    return pd.concat([percentile_rank(df[c]) for c in cols], axis=1).mean(axis=1)


def build_group_scores(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """
    One robust [0,1] score per semantic group, oriented so higher raw = higher score.
    Frameworks apply the (signed) weights. These are the direct inputs to profitability.py.
    Groups tagged `excluded_ambiguous` in config are intentionally NOT built (sign-uncertain
    or redundant features — robustness beats precision near the 80th-percentile boundary).
    """
    cfg = cfg or load_config()
    sem = cfg.semantics

    def cols_of(*groups):
        gset = set(groups)
        return [c for c, m in sem.items() if m.get("group") in gset and c in df.columns]

    gs = pd.DataFrame(index=df.index)
    # Monetary buckets (sum raw dollars -> rank).
    gs["spend"] = money_score(df, cols_of("spend"))                       # f5 Total Spend (PRIMARY)
    gs["spend_cat"] = money_score(df, cols_of("spend_cat_5x", "spend_cat_1x"))  # category-sum corroborator
    gs["revolve"] = money_score(df, cols_of("revolve"))                  # f1 revolve balance
    gs["credit_line"] = money_score(df, cols_of("credit_line"))          # f17/f18 lend lines
    gs["points"] = money_score(df, cols_of("points"))                    # f4 balance + f21 redeemed
    gs["benefit_cost"] = rankmean_score(df, cols_of("benefit_cost"))     # mixed $ + counts -> rankmean
    # Count/score buckets (mean of per-column ranks).
    gs["engagement"] = rankmean_score(df, cols_of("engagement"))         # f12 logins + f22/f23 emails
    gs["relationship"] = rankmean_score(df, cols_of("relationship"))     # f19/f20 accounts
    gs["risk"] = rankmean_score(df, cols_of("risk"))                     # f11 risk score
    gs["attrition"] = rankmean_score(df, cols_of("attrition"))           # f2 cancel calls
    gs["collection"] = rankmean_score(df, cols_of("collection"))         # f3 collection cancel calls

    # Coupon-clipper signal: benefit utilisation NOT justified by spend (high = extractor).
    benefit_level = rankmean_score(df, cols_of("benefit_cost"))
    gs["benefit_efficiency"] = (benefit_level - gs["spend"]).clip(-1, 1)
    return gs


# =======================================================================================
# Rich engineered feature table (for EDA, clustering, and interpretability)
# =======================================================================================
def engineer_features(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """
    Extensive business features. Returns a NEW dataframe (id preserved) with:
      transforms, percentile ranks, and composite business indicators. Never alters
      or drops input rows.
    """
    cfg = cfg or load_config()
    sem = cfg.semantics
    out = pd.DataFrame(index=df.index)
    if cfg.id_column in df.columns:
        out[cfg.id_column] = df[cfg.id_column].values

    feats = feature_columns(df, cfg)

    # --- per-feature robust transforms + ranks -------------------------------------
    for c in feats:
        out[f"{c}__pr"] = percentile_rank(df[c])            # percentile rank
        out[f"{c}__logw"] = log1p_signed(winsorize(df[c]))  # winsorised log
        out[f"{c}__rz"] = robust_z(df[c])                   # robust z

    def cols_of(*groups):
        gset = set(groups)
        return [c for c, m in sem.items() if m.get("group") in gset and c in df.columns]

    total_col = cols_of("spend")                       # f5 Total Spend
    cat_cols = cols_of("spend_cat_5x", "spend_cat_1x")  # f6..f10 categories
    # --- composite business indicators ---------------------------------------------
    total_spend = df[total_col[0]].clip(lower=0) if total_col else (
        df[cat_cols].clip(lower=0).sum(axis=1) if cat_cols else pd.Series(0.0, index=df.index))
    out["total_spend"] = total_spend
    out["log_total_spend"] = np.log1p(total_spend)
    # Spend diversity: how many categories are active (breadth of wallet).
    out["spend_breadth"] = (df[cat_cols] > 0).sum(axis=1) if cat_cols else 0
    # 5x-category share (Airlines+Lodging / total) — the rewards-margin drag proxy.
    cat5 = cols_of("spend_cat_5x")
    spend_5x = df[cat5].clip(lower=0).sum(axis=1) if cat5 else pd.Series(0.0, index=df.index)
    out["premium_5x_share"] = (spend_5x / (total_spend + 1.0)).clip(0, 1)
    # Spend concentration (HHI): 1 => all in one category, low => diversified.
    if cat_cols:
        denom = df[cat_cols].clip(lower=0).sum(axis=1).replace(0, np.nan)
        shares = df[cat_cols].clip(lower=0).div(denom, axis=0).fillna(0)
        out["spend_hhi"] = (shares ** 2).sum(axis=1)
    else:
        out["spend_hhi"] = 0.0

    gs = build_group_scores(df, cfg)
    for g in gs.columns:
        out[f"grp_{g}"] = gs[g]

    # Benefit utilisation ratio (cost per $1k of spend) — coupon-clipper detector.
    benefit_cols = cols_of("benefit_cost")
    benefit_level = df[benefit_cols].clip(lower=0).sum(axis=1) if benefit_cols else pd.Series(0.0, index=df.index)
    out["benefit_utilisation"] = benefit_level
    out["benefit_per_1k_spend"] = benefit_level / (total_spend / 1000.0 + 1.0)

    # Spend efficiency: revenue-proxy net of benefit consumption.
    out["spend_efficiency"] = gs["spend"] - gs["benefit_cost"]

    # Risk-adjusted spend: down-weight spend by risk & collection distress.
    out["risk_adjusted_spend"] = gs["spend"] * (1.0 - 0.5 * gs["risk"] - 0.5 * gs["collection"]).clip(0, 1)

    # Engagement / loyalty / relationship indices.
    out["engagement_index"] = gs["engagement"]
    out["loyalty_index"] = 0.5 * gs["points"] + 0.5 * gs["relationship"]
    out["relationship_depth"] = gs["relationship"]

    # Revolve intensity (interest revenue proxy, penalised by risk).
    out["revolve_intensity"] = gs["revolve"] * (1.0 - 0.5 * gs["risk"])

    # Points redemption ratio (redeemed / balance) — engagement + liability signal.
    pe = cols_of("points")                             # [f4 balance, f21 redeemed]
    if len(pe) >= 2:
        bal = df[pe[0]].clip(lower=0)
        redeemed = df[pe[1]].clip(lower=0)
        out["redemption_ratio"] = (redeemed / (bal + 1.0)).clip(0, 2)
    else:
        out["redemption_ratio"] = 0.0

    # Customer-lifetime indicator: spend x relationship, net of attrition/distress.
    out["clv_proxy"] = (gs["spend"] * (0.5 + 0.5 * gs["relationship"])
                        * (1.0 - gs["attrition"]) * (1.0 - gs["collection"]))

    # A first-pass composite (frameworks refine this).
    out["composite_v0"] = (
        1.00 * gs["spend"] + 0.15 * gs["spend_cat"] + 0.15 * gs["revolve"]
        + 0.05 * gs["credit_line"] + 0.06 * gs["engagement"] + 0.08 * gs["relationship"]
        + 0.05 * gs["points"] - 0.15 * gs["risk"] - 0.12 * gs["benefit_cost"]
        - 0.10 * gs["attrition"] - 0.15 * gs["collection"]
    )
    return out
