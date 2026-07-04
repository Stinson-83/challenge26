"""
rankers.py — self-contained candidate ranking functions + a registry.
=====================================================================
Each ranker maps an (already NaN-imputed) feature DataFrame to a per-member score where
HIGHER = more profitable. The submission's top-20% is the 100k highest scores.

Design: a ranker is expressed as `build(df) -> {component: [0,1]-or-raw vector}` plus a
`weights` dict and a `combine` rule. This makes two robustness operations GENERIC:
  * weight-perturbation  : jitter `weights`, re-combine.
  * leave-one-feature-out: set a raw column to a constant, rebuild components, re-score.

We keep this decoupled from src/feature_engineering.py / src/profitability.py on purpose:
this module is the neutral judge, so it must not import the code it will grade.

NaN handling is done UPSTREAM by `impute_df` (so imputation policy is a first-class,
swappable axis of the harness), leaving `R()` as a pure percentile rank.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FCOLS = [f"f{i}" for i in range(1, 24)]


# ---------------------------------------------------------------------------------------
# Imputation (the swappable NaN policy) and the percentile-rank primitive
# ---------------------------------------------------------------------------------------
def impute_df(df: pd.DataFrame, policy: str = "median") -> pd.DataFrame:
    """
    Return a copy of df with every f1..f23 NaN filled per `policy`:
      'median'    -> column median               (mid-rank; the SPEC baseline)
      'zero'      -> 0.0                          (bottom-of-support; matches shipped 0.768 file)
      'indicator' -> (column min - 1)            (a distinct 'missing' floor below all observed)
    Only columns present in df are touched; non-feature columns (e.g. id) are left as-is.
    """
    out = df.copy()
    for c in FCOLS:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce").astype(float)
        if policy == "zero":
            fill = 0.0
        elif policy == "indicator":
            mn = s.min()
            fill = (mn - 1.0) if pd.notna(mn) else 0.0
        else:  # median
            med = s.median()
            fill = med if pd.notna(med) else 0.0
        out[c] = s.fillna(fill)
    return out


def R(x) -> np.ndarray:
    """Percentile rank in [0,1] (average ties). Assumes NaN already imputed upstream."""
    v = pd.to_numeric(pd.Series(x), errors="coerce").astype(float).values
    if np.isnan(v).any():  # defensive: rank NaN at the median
        v = np.where(np.isnan(v), np.nanmedian(v), v)
    r = rankdata(v, method="average")
    n = len(r)
    return (r - 1.0) / max(n - 1, 1)


def _c(df: pd.DataFrame, name: str) -> np.ndarray:
    """Raw numeric column as float array (0.0 if absent)."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float).values
    return np.zeros(len(df), dtype=float)


# ---------------------------------------------------------------------------------------
# Ranker abstraction
# ---------------------------------------------------------------------------------------
@dataclass
class Ranker:
    name: str
    build: Callable[[pd.DataFrame], dict[str, np.ndarray]]
    weights: dict[str, float]
    features: list[str]                    # raw f-columns this ranker depends on (for LOFO)
    combine: str = "linear"                # 'linear' | 'risk_multiplicative'
    note: str = ""

    def combine_components(self, comps: dict[str, np.ndarray],
                           weights: dict[str, float] | None = None) -> np.ndarray:
        """Combine precomputed components with `weights` (defaults to self.weights)."""
        w = self.weights if weights is None else weights
        if self.combine == "risk_multiplicative":
            # base = the (perturbable) revenue mix; the rest are risk discounts / cost.
            discount = {"risk", "attrition", "collection", "benefit"}
            adj = sum(w[k] * comps[k] for k in w if k not in discount)
            adj = adj * (1.0 - abs(w.get("risk", 0.0)) * comps["risk"])
            adj = adj * (1.0 - abs(w.get("attrition", 0.0)) * comps["attrition"])
            adj = adj * (1.0 - abs(w.get("collection", 0.0)) * comps["collection"])
            adj = adj - abs(w.get("benefit", 0.0)) * comps["benefit"]
            return adj
        return sum(w[k] * comps[k] for k in w)

    def score(self, df: pd.DataFrame, weights: dict[str, float] | None = None) -> np.ndarray:
        return self.combine_components(self.build(df), weights)

    def perturbed_weights(self, rng: np.random.Generator, sigma: float = 0.30) -> dict[str, float]:
        """Each weight multiplied by exp(N(0, sigma)) (sign-preserving magnitude jitter)."""
        return {k: v * float(np.exp(rng.normal(0.0, sigma))) for k, v in self.weights.items()}


# ---------------------------------------------------------------------------------------
# Derived signals (all on an ALREADY-IMPUTED df)
# ---------------------------------------------------------------------------------------
def _cats(df):   # summed category spend (dominated by f7 on real data)
    return _c(df, "f6") + _c(df, "f7") + _c(df, "f8") + _c(df, "f9") + _c(df, "f10")

def _catavg_rank(df):  # SCALE-FAIR category spend: mean of per-column ranks (the R2 fix)
    return (R(_c(df, "f6")) + R(_c(df, "f7")) + R(_c(df, "f8"))
            + R(_c(df, "f9")) + R(_c(df, "f10"))) / 5.0

def _eng(df):    return _c(df, "f12") + _c(df, "f22") + _c(df, "f23")
def _rel(df):    return _c(df, "f19") + _c(df, "f20")
def _pts(df):    return _c(df, "f4") + _c(df, "f21")
def _benefit(df):return _c(df, "f13") + _c(df, "f14") + _c(df, "f15") + _c(df, "f16")
def _lend(df):   return _c(df, "f17") + _c(df, "f18")
def _EL(df):     return _c(df, "f11") * _c(df, "f1")            # PD x EAD expected-loss interaction


# ---------------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------------
def build_registry() -> dict[str, Ranker]:
    reg: dict[str, Ranker] = {}

    def add(r: Ranker):
        reg[r.name] = r

    # --- single-signal baselines ---------------------------------------------------------
    add(Ranker("f5_only", lambda d: {"f5": R(_c(d, "f5"))}, {"f5": 1.0}, ["f5"],
               note="pure Total-Spend label; ~random on public (0.337) — the null to beat"))
    add(Ranker("cats_only", lambda d: {"cats": R(_cats(d))}, {"cats": 1.0},
               ["f6", "f7", "f8", "f9", "f10"], note="summed category spend (~f7); public 0.668"))
    add(Ranker("catavg_only", lambda d: {"cats": _catavg_rank(d)}, {"cats": 1.0},
               ["f6", "f7", "f8", "f9", "f10"], note="scale-fair category spend (rank-mean)"))
    add(Ranker("revolve_only", lambda d: {"f1": R(_c(d, "f1"))}, {"f1": 1.0}, ["f1"],
               note="revolve balance / interest axis; public 0.549"))

    # --- the honest 0.768 model (bit-exact vs shipped file under zero-fill) --------------
    add(Ranker(
        "honest_0768",
        lambda d: {"cats": R(_cats(d)), "f1": R(_c(d, "f1")), "EL": R(_EL(d)),
                   "f3": R(_c(d, "f3")), "f21": R(_c(d, "f21"))},
        {"cats": 0.573, "f1": 0.427, "EL": -0.18, "f3": -0.15, "f21": -0.06},
        ["f6", "f7", "f8", "f9", "f10", "f1", "f11", "f3", "f21"],
        note="rank(0.573*cats + 0.427*f1 - 0.18*EL - 0.15*f3 - 0.06*f21) — current honest best"))

    # --- a scale-fair, data-motivated variant of the honest model (candidate rebuild seed)
    add(Ranker(
        "honest_catavg",
        lambda d: {"cats": _catavg_rank(d), "f1": R(_c(d, "f1")), "EL": R(_EL(d)),
                   "f3": R(_c(d, "f3"))},
        {"cats": 0.55, "f1": 0.45, "EL": -0.18, "f3": -0.15},
        ["f6", "f7", "f8", "f9", "f10", "f1", "f11", "f3"],
        note="honest model with SCALE-FAIR category spend + disjoint f3; no f21 (incoherent bundle)"))

    # --- config frameworks (f5-primary family; expected to be robustly WRONG) ------------
    add(Ranker(
        "revenue_first_cfg",
        lambda d: {"f5": R(_c(d, "f5")), "cats": R(_cats(d)), "f1": R(_c(d, "f1")),
                   "lend": R(_lend(d)), "pts": R(_pts(d)), "eng": R(_eng(d)), "f11": R(_c(d, "f11"))},
        {"f5": 1.0, "cats": 0.15, "f1": 0.12, "lend": 0.05, "pts": 0.05, "eng": 0.03, "f11": -0.05},
        ["f5", "f6", "f7", "f8", "f9", "f10", "f1", "f17", "f18", "f4", "f21", "f12", "f22", "f23", "f11"],
        note="config revenue_first — dominated by R(f5)"))
    add(Ranker(
        "full_pnl_cfg",
        lambda d: {"f5": R(_c(d, "f5")), "cats": R(_cats(d)), "f1": R(_c(d, "f1")),
                   "lend": R(_lend(d)), "eng": R(_eng(d)), "rel": R(_rel(d)), "pts": R(_pts(d)),
                   "benefit": R(_benefit(d)), "f11": R(_c(d, "f11")), "f2": R(_c(d, "f2")), "f3": R(_c(d, "f3"))},
        {"f5": 1.0, "cats": 0.15, "f1": 0.20, "lend": 0.05, "eng": 0.06, "rel": 0.08, "pts": 0.05,
         "benefit": -0.12, "f11": -0.18, "f2": -0.10, "f3": -0.15},
        ["f5", "f6", "f7", "f8", "f9", "f10", "f1", "f17", "f18", "f12", "f22", "f23",
         "f19", "f20", "f4", "f21", "f13", "f14", "f15", "f16", "f11", "f2", "f3"],
        note="config full_pnl — f5 primary minus costs/risk"))
    add(Ranker(
        "risk_adjusted_cfg",
        lambda d: {"f5": R(_c(d, "f5")), "cats": R(_cats(d)), "f1": R(_c(d, "f1")), "lend": R(_lend(d)),
                   "risk": R(_c(d, "f11")), "attrition": R(_c(d, "f2")),
                   "collection": R(_c(d, "f3")), "benefit": R(_benefit(d))},
        {"f5": 1.0, "cats": 0.15, "f1": 0.20, "lend": 0.05,
         "risk": -0.35, "attrition": -0.15, "collection": -0.25, "benefit": -0.08},
        ["f5", "f6", "f7", "f8", "f9", "f10", "f1", "f17", "f18", "f11", "f2", "f3",
         "f13", "f14", "f15", "f16"],
        combine="risk_multiplicative",
        note="config risk_adjusted — multiplicative risk discount on an f5-primary base"))

    # --- raw-dollar models (score is raw $; used for ranking only) -----------------------
    add(Ranker(
        "dollar_pnl",
        lambda d: {"interchange": 0.022 * _c(d, "f5"), "interest": 0.15 * _c(d, "f1"),
                   "rewards": -0.011 * _c(d, "f5"), "drag5x": -0.008 * (_c(d, "f6") + _c(d, "f9")),
                   "benefit": -(_c(d, "f14") + _c(d, "f16") + 35.0 * _c(d, "f13") + 15.0 * _c(d, "f15")),
                   "risk_loss": -_EL(d), "servicing": -(30.0 * _c(d, "f2") + 250.0 * _c(d, "f3"))},
        {k: 1.0 for k in ["interchange", "interest", "rewards", "drag5x", "benefit", "risk_loss", "servicing"]},
        ["f5", "f1", "f6", "f9", "f14", "f16", "f13", "f15", "f11", "f2", "f3"],
        note="config dollar_pnl (raw $); structurally ranks by revolve f1, not spend"))

    return reg


REGISTRY = build_registry()


def register(ranker: Ranker) -> None:
    """Register a new candidate model (e.g. a rebuilt framework) so the harness scores it."""
    REGISTRY[ranker.name] = ranker


def top_set(score: np.ndarray, top_pct: float = 0.20) -> np.ndarray:
    """Indices of the top-`top_pct` by score (largest). Returns a sorted index array."""
    n = len(score)
    k = max(1, int(round(top_pct * n)))
    idx = np.argpartition(score, n - k)[n - k:]
    return np.sort(idx)
