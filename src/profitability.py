"""
profitability.py
================
The heart of the solution: multiple, independent profitability frameworks that each
rank-order cardmembers by estimated Profit = Revenue - Cost - Expected Risk Loss.

Every framework consumes the robust [0,1] group-scores from feature_engineering and
emits a per-cardmember score plus a percentile rank. Because the true label is unknown
and the metric is top-20% overlap, we also build a RANK ENSEMBLE that averages the ranks
of several plausible frameworks — this stabilises exactly the 80th-percentile boundary
where the leaderboard is decided.

Business justification lives beside each framework so every equation is defensible from
Amex's unit economics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, load_config
from .feature_engineering import build_group_scores, percentile_rank


# =======================================================================================
# Interpretable sub-scores (the "scorecard")
# =======================================================================================
def compute_subscores(gs: pd.DataFrame) -> pd.DataFrame:
    """
    Seven independent, human-readable [0,1] scores. Higher = better for the customer's
    contribution to that dimension (risk/cost scores are oriented so higher = *worse*,
    matching their natural meaning; the overall score subtracts them).
    """
    sub = pd.DataFrame(index=gs.index)
    sub["revenue_score"] = (0.70 * gs["spend"] + 0.15 * gs["revolve"]
                            + 0.10 * gs["spend_cat"] + 0.05 * gs["credit_line"]).clip(0, 1)
    sub["cost_score"] = gs["benefit_cost"].clip(0, 1)
    sub["risk_score"] = (0.6 * gs["risk"] + 0.4 * gs["collection"]).clip(0, 1)
    sub["engagement_score"] = gs["engagement"].clip(0, 1)
    sub["loyalty_score"] = (0.5 * gs["points"] + 0.5 * gs["relationship"]).clip(0, 1)
    sub["relationship_score"] = gs["relationship"].clip(0, 1)
    sub["attrition_score"] = (0.5 * gs["attrition"] + 0.5 * gs["collection"]).clip(0, 1)
    return sub


# =======================================================================================
# Framework implementations
# =======================================================================================
def _linear_framework(gs: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted sum of group scores. Positive weights = revenue/engagement; negative = cost/risk."""
    score = pd.Series(0.0, index=gs.index)
    for group, w in weights.items():
        if group in gs.columns:
            score = score + float(w) * gs[group]
    return score


# NOTE: every framework has signature (gs, df, cfg). Group-based frameworks ignore `df`;
# the dollar-P&L framework needs the raw columns, so it uses `df`.
def framework_revenue_first(gs, df, cfg: Config) -> pd.Series:
    """
    REVENUE-FIRST. Thesis: for a charge card, issuer profit is dominated by interchange
    (= net margin x total spend). A $500 fee or a few hundred in credits are second-order
    versus interchange on tens of thousands of spend. So spend leads; everything else is a
    light touch. This is the safest high-overlap ranking.
    """
    return _linear_framework(gs, cfg.framework_weights("revenue_first"))


def framework_full_pnl(gs, df, cfg: Config) -> pd.Series:
    """
    FULL P&L (group-based). Revenue (spend + revolve interest + points/relationship) minus
    Cost (benefit redemptions) minus Risk (default) minus Attrition, all on robust group
    scores. Captures second-order re-ranking of the top tail that pure spend misses.
    """
    return _linear_framework(gs, cfg.framework_weights("full_pnl"))


def framework_risk_adjusted(gs, df, cfg: Config) -> pd.Series:
    """
    RISK-ADJUSTED. Revenue is discounted MULTIPLICATIVELY by risk and attrition, because a
    high spender likely to default or churn can be a net LOSS (tail losses dominate the
    downside). Multiplicative form preserves spend ordering among low-risk members while
    convexly hammering the profitable-looking-but-risky tail:
        base = spend + w_st*spend_total + w_rev*revolve
        adj  = base * (1 - a*risk) * (1 - b*attrition) - c*benefit_cost
    """
    w = cfg.framework_weights("risk_adjusted")
    base = (gs["spend"] + w.get("spend_cat", 0.15) * gs["spend_cat"]
            + w.get("revolve", 0.2) * gs["revolve"] + w.get("credit_line", 0.05) * gs["credit_line"])
    a = abs(w.get("risk", -0.35))
    b = abs(w.get("attrition", -0.15))
    d = abs(w.get("collection", -0.25))
    c = abs(w.get("benefit_cost", -0.08))
    adj = (base * (1.0 - a * gs["risk"]) * (1.0 - b * gs["attrition"])
           * (1.0 - d * gs["collection"]) - c * gs["benefit_cost"])
    return adj


def framework_relationship(gs, df, cfg: Config) -> pd.Series:
    """
    RELATIONSHIP / coupon-clipper-aware. Rewards engaged power-users and PENALISES
    perk-harvesters (high benefit utilisation not justified by spend), who cost more in
    redemptions than they generate. Uses benefit_efficiency (benefit level minus spend
    level): positive = extractor -> penalty.
    """
    w = cfg.framework_weights("relationship")
    score = _linear_framework(gs, {k: v for k, v in w.items() if k != "benefit_efficiency"})
    score = score + w.get("benefit_efficiency", -0.15) * gs.get("benefit_efficiency", 0.0)
    return score


def framework_dollar_pnl(gs, df, cfg: Config) -> pd.Series:
    """
    DOLLAR-GROUNDED P&L. Computes Profit = Revenue - Cost - Expected Risk Loss in RAW
    DOLLARS from the (now known) columns, so it can encode literal cost sizes and the
    5x-rewards drag on Airlines(f6)+Lodging(f9) — the mechanism by which true profit rank
    diverges from naive gross-spend rank. The raw profit is percentile-ranked downstream.

        Revenue  = discount*TotalSpend(f5) + interest*Revolve(f1)
        Rewards  = rewards*TotalSpend(f5)  + drag5x*(Airlines f6 + Lodging f9)
        Benefit  = airline_credit(f14) + entertainment_credit(f16)
                   + lounge_cost*LoungeCount(f13) + cab_cost*CabUse(f15)
        RiskLoss = RiskScore(f11) * Revolve(f1) * LGD
        Servicing= call*CancelCalls(f2) + collection*CollectionCalls(f3)
    Coefficients come from config.frameworks.dollar_pnl.
    """
    p = cfg.frameworks.get("dollar_pnl", {})

    def col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    def one(group, prefer=None):
        cols = _cols_in(cfg, group)
        name = prefer if (prefer and prefer in cols) else (cols[0] if cols else None)
        return col(name) if name else pd.Series(0.0, index=df.index)

    def summ(group):
        return sum((col(c).clip(lower=0) for c in _cols_in(cfg, group)),
                   start=pd.Series(0.0, index=df.index))

    total_spend = one("spend", "f5").clip(lower=0)
    spend_5x = summ("spend_cat_5x")                             # Airlines f6 + Lodging f9
    f1 = one("revolve", "f1").clip(lower=0)
    f11 = one("risk", "f11").clip(lower=0)
    # Benefit costs: f14 & f16 are $ redeemed; f13 (lounge) & f15 (cab) are counts.
    bc = _cols_in(cfg, "benefit_cost")                          # [f13,f14,f15,f16]
    airline_credit = col("f14"); entertainment_credit = col("f16")
    lounge = col("f13"); cab = col("f15")
    calls = one("attrition", "f2")
    collection = one("collection", "f3")

    revenue = p.get("discount_rate", 0.022) * total_spend + p.get("interest_rate", 0.15) * f1
    rewards = p.get("rewards_rate", 0.011) * total_spend + p.get("rewards_drag_5x", 0.008) * spend_5x
    benefit = (p.get("airline_credit_coef", 1.0) * airline_credit
               + p.get("entertainment_credit_coef", 1.0) * entertainment_credit
               + p.get("lounge_cost", 35.0) * lounge + p.get("cab_cost", 15.0) * cab)
    risk_loss = f11 * f1 * p.get("lgd", 1.0)
    servicing = p.get("servicing_call", 30.0) * calls + p.get("collection_cost", 250.0) * collection

    profit = revenue - rewards - benefit - risk_loss - servicing
    return profit


def _cols_in(cfg: Config, group: str) -> list[str]:
    return [c for c, m in cfg.semantics.items() if m.get("group") == group]


FRAMEWORKS = {
    "revenue_first": framework_revenue_first,
    "full_pnl": framework_full_pnl,
    "risk_adjusted": framework_risk_adjusted,
    "relationship": framework_relationship,
    "dollar_pnl": framework_dollar_pnl,
}


def framework_rank_ensemble(gs, df, cfg: Config) -> pd.Series:
    """
    RANK ENSEMBLE. Convert each member framework to a percentile rank, then take a weighted
    average of RANKS (not raw scores — ranks are unit-free and robust). This is the
    recommended primary: it hedges the unknown "true" weighting and stabilises the top-20%
    set against any single framework's mis-specification.
    """
    spec = cfg.frameworks.get("rank_ensemble", {})
    members = spec.get("members", list(FRAMEWORKS))
    mweights = spec.get("member_weights", [1.0] * len(members))
    mweights = np.array(mweights, dtype=float)
    mweights = mweights / mweights.sum()

    ranks = pd.DataFrame(index=gs.index)
    for m, w in zip(members, mweights):
        ranks[m] = percentile_rank(FRAMEWORKS[m](gs, df, cfg)) * w
    return ranks.sum(axis=1)


ALL_FRAMEWORKS = {**FRAMEWORKS, "rank_ensemble": framework_rank_ensemble}


# =======================================================================================
# Top-level API
# =======================================================================================
@dataclass
class ProfitabilityResult:
    scores: pd.DataFrame          # id + every framework score + its percentile rank
    subscores: pd.DataFrame       # the seven interpretable sub-scores
    group_scores: pd.DataFrame    # the robust [0,1] group atoms
    active: str                   # name of the active/primary framework
    primary_rank: pd.Series       # percentile rank under the active framework (the ranking)


def score_dataframe(df: pd.DataFrame, cfg: Config | None = None,
                    active: str | None = None) -> ProfitabilityResult:
    """
    Compute every framework's score + rank for `df`. `df` must be CLEANED (see
    preprocessing.clean). The active framework's percentile rank is the submission ranking.
    """
    cfg = cfg or load_config()
    active = active or cfg.active_framework
    gs = build_group_scores(df, cfg)
    sub = compute_subscores(gs)

    scores = pd.DataFrame(index=df.index)
    if cfg.id_column in df.columns:
        scores[cfg.id_column] = df[cfg.id_column].values
    for name, fn in ALL_FRAMEWORKS.items():
        raw = fn(gs, df, cfg)
        scores[f"score__{name}"] = raw.values
        scores[f"rank__{name}"] = percentile_rank(raw).values

    primary_rank = scores[f"rank__{active}"].copy()

    # Optional narrow boundary guards (precision protection at the 80th-pct cut).
    if cfg.raw.get("guards", {}).get("enable", False):
        primary_rank = apply_guards(primary_rank, df, gs, cfg)

    scores["profitability_score"] = primary_rank.values      # canonical output column
    return ProfitabilityResult(scores=scores, subscores=sub, group_scores=gs,
                               active=active, primary_rank=primary_rank)


def apply_guards(primary_rank: pd.Series, df: pd.DataFrame, gs: pd.DataFrame,
                 cfg: Config, top_pct: float | None = None) -> pd.Series:
    """
    Narrow, reversible boundary guards. We only DEMOTE a genuinely dangerous handful out of
    the predicted top set and let clean survivors backfill. Deliberately conservative — an
    over-broad guard destroys top-20% overlap (see synthesis risks).

    loss_screen : high-PD (f11) AND high-balance (f1) revolvers, or cancellers, are pushed
                  just below the top-20% cut.
    coupon_clipper : below-median-spend members whose benefit utilisation exceeds their
                  spend level (benefit_efficiency > 0) are pushed below the cut.
    """
    g = cfg.raw.get("guards", {})
    top_pct = top_pct or cfg.top_pct
    sem = cfg.semantics
    rank = primary_rank.copy()
    n = len(rank)
    cut = rank.quantile(1 - top_pct)
    demote_to = cut - 1e-6                      # place just below the cut

    def col(group, prefer):
        cols = [c for c, m in sem.items() if m.get("group") == group]
        name = prefer if prefer in cols else (cols[0] if cols else None)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in (df.columns if name else []) else pd.Series(0.0, index=df.index)

    mask = pd.Series(False, index=rank.index)
    if g.get("loss_screen", True):
        f11 = col("risk", "f11")          # risk score
        f1 = col("revolve", "f1")         # revolve balance (exposure)
        collection = col("collection", "f3")   # collection cancel calls (severe)
        calls = col("attrition", "f2")    # cancellation calls
        pd_hi = f11 > f11.quantile(g.get("pd_pct", 0.90))
        bal_hi = f1 > f1.quantile(g.get("balance_pct", 0.75))
        mask = mask | (pd_hi & bal_hi) | (collection >= 1) | (calls >= 2)
    if g.get("coupon_clipper", True):
        below_med_spend = gs["spend"] < 0.5
        extractor = gs["benefit_efficiency"] > 0.0
        mask = mask | (below_med_spend & extractor)

    # Only demote members currently INSIDE the predicted top set.
    inside_top = rank >= cut
    demote = mask & inside_top
    rank.loc[demote] = demote_to * (1 - 1e-9) - 1e-3 * gs["risk"].loc[demote]
    return rank


def framework_agreement(scores: pd.DataFrame, top_pct: float = 0.20) -> pd.DataFrame:
    """
    Pairwise overlap of the top-`top_pct` sets across frameworks + Spearman of full ranks.
    High agreement in the top set = a confident, robust ranking; low agreement flags the
    contested boundary customers that most affect the leaderboard.
    """
    fw = [c.replace("rank__", "") for c in scores.columns if c.startswith("rank__")]
    n = len(scores)
    k = max(1, int(round(top_pct * n)))
    top_sets = {f: set(scores.nlargest(k, f"rank__{f}").index) for f in fw}
    rows = []
    for i, a in enumerate(fw):
        for b in fw[i + 1:]:
            overlap = len(top_sets[a] & top_sets[b]) / k
            rho = scores[f"rank__{a}"].corr(scores[f"rank__{b}"], method="spearman")
            rows.append({"a": a, "b": b, "top_overlap": round(overlap, 4),
                         "spearman": round(float(rho), 4)})
    return pd.DataFrame(rows).sort_values("top_overlap")
