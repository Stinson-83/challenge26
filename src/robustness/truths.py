"""
truths.py — a library of STRUCTURALLY-DIFFERENT synthetic hidden-profit generators.
==================================================================================
WHY THIS EXISTS. The real profitability label is hidden, and the repo's existing synthetic
validation is CIRCULAR: its hidden truth shares the exact functional form of the dollar_pnl
scorer, so dollar_pnl "wins" by construction (0.87) yet is near-orthogonal (24-29%) to every
framework on real data. Selecting on that harness would ship the worst model.

WHAT THIS IS. Not a claim about the real label. It is an ECONOMIC-UNCERTAINTY STRESS TEST:
a panel of plausible-but-different profit economics, each built from the REAL features (so the
marginals, skew, missingness and censoring are realistic), each with a DIFFERENT functional
form (linear / multiplicative / threshold-tier / relationship-led / adversarially f5-led).

HOW TO READ IT. A ranker that recovers the top-20% of a BROAD RANGE of these truths well —
especially its WORST-CASE recovery across the panel — is robust to our uncertainty about which
economics Amex actually scores. That worst-case-over-truths number is the harness's headline
generalization proxy (a ranker tuned to one economics but fragile on the rest is a red flag).

Each generator: (filled_df, seed) -> per-member profit array (higher = more profitable).
Multiplicative + additive lognormal noise (seeded) makes the top-20% shift across seeds so the
harness can report a recovery confidence interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FCOLS = [f"f{i}" for i in range(1, 24)]


def _c(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float).values
    return np.zeros(len(df), dtype=float)


def _z(x: np.ndarray) -> np.ndarray:
    """Standardize log1p(|x|)*sign(x) so heavy-tailed dollars become comparable, sign-safe."""
    lx = np.sign(x) * np.log1p(np.abs(x))
    sd = lx.std()
    return (lx - lx.mean()) / (sd if sd > 1e-9 else 1.0)


def _thr(x: np.ndarray, q: float) -> np.ndarray:
    """1.0 where x is above its q-quantile, else 0.0."""
    return (x >= np.quantile(x, q)).astype(float)


def _noise(n: int, seed: int, mult_sigma: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(0.0, mult_sigma, size=n))


# ---------------------------------------------------------------------------------------
# The panel. Each is a DIFFERENT economics; none replicates a ranker's exact formula.
# ---------------------------------------------------------------------------------------
def t_spend_led(df, seed):
    """Interchange-dominated: category spend leads, mild revolve, mild expected-loss drag."""
    prof = (1.00 * _z(_c(df, "f6") + _c(df, "f7") + _c(df, "f8") + _c(df, "f9") + _c(df, "f10"))
            + 0.25 * _z(_c(df, "f1"))
            - 0.20 * _z(_c(df, "f11") * _c(df, "f1")))
    return prof * _noise(len(df), seed)


def t_revolve_led(df, seed):
    """Lending economics: interest on revolve net of PD*EAD dominates; collection is costly."""
    prof = (1.00 * _z(_c(df, "f1"))
            - 0.55 * _z(_c(df, "f11") * _c(df, "f1"))
            - 0.35 * _c(df, "f3")                       # collection flag, hard penalty
            + 0.20 * _z(_c(df, "f6") + _c(df, "f7") + _c(df, "f8") + _c(df, "f9") + _c(df, "f10")))
    return prof * _noise(len(df), seed)


def t_relationship_led(df, seed):
    """Depth economics: spend + household breadth + engagement, penalise perk extraction."""
    spend = _z(_c(df, "f6") + _c(df, "f7") + _c(df, "f8") + _c(df, "f9") + _c(df, "f10"))
    benefit = _z(_c(df, "f13") + _c(df, "f14") + _c(df, "f15") + _c(df, "f16"))
    prof = (0.70 * spend
            + 0.35 * _z(_c(df, "f19") + _c(df, "f20"))
            + 0.20 * _z(_c(df, "f12") + _c(df, "f22"))
            - 0.30 * np.maximum(benefit - spend, 0.0))   # benefit not justified by spend
    return prof * _noise(len(df), seed)


def t_threshold_tier(df, seed):
    """Non-linear tiers: real profit accrues only to high-spend AND low-risk members."""
    spend = _c(df, "f6") + _c(df, "f7") + _c(df, "f8") + _c(df, "f9") + _c(df, "f10")
    hi_spend = _thr(spend, 0.70)
    lo_risk = 1.0 - _thr(_c(df, "f11"), 0.80)
    hi_revolve = _thr(_c(df, "f1"), 0.80)
    prof = (2.0 * hi_spend * lo_risk               # the profitable tier
            + 0.8 * hi_revolve * lo_risk
            + 0.3 * _z(spend)                       # mild continuous tilt inside tiers
            - 0.5 * _c(df, "f3"))
    return prof * _noise(len(df), seed)


def t_interaction(df, seed):
    """Multiplicative: spend scaled by (1-risk) and lifted by engagement, minus costs."""
    spend = _z(_c(df, "f6") + _c(df, "f7") + _c(df, "f8") + _c(df, "f9") + _c(df, "f10"))
    risk01 = _c(df, "f11") / (np.quantile(_c(df, "f11"), 0.99) + 1e-9)
    risk01 = np.clip(risk01, 0.0, 1.0)
    eng = _z(_c(df, "f12") + _c(df, "f22"))
    base = spend - spend.min() + 0.1
    prof = (base * (1.0 - 0.6 * risk01) * (1.0 + 0.15 * (eng - eng.min()))
            - 0.4 * _z(_c(df, "f13") + _c(df, "f14") + _c(df, "f15") + _c(df, "f16"))
            - 0.6 * _c(df, "f3"))
    return prof * _noise(len(df), seed)


def t_f5_led(df, seed):
    """Adversarial: what if f5 REALLY is the driver? Ensures we notice if our model would
    catastrophically fail on an f5-shaped label (keeps us honest, not just anti-f5)."""
    prof = (1.00 * _z(_c(df, "f5"))
            + 0.20 * _z(_c(df, "f1"))
            - 0.15 * _c(df, "f3"))
    return prof * _noise(len(df), seed)


TRUTHS = {
    "spend_led": t_spend_led,
    "revolve_led": t_revolve_led,
    "relationship_led": t_relationship_led,
    "threshold_tier": t_threshold_tier,
    "interaction": t_interaction,
    "f5_led": t_f5_led,
}

TRUTH_NOTES = {
    "spend_led": "interchange economics (category spend leads)",
    "revolve_led": "lending economics (revolve interest net of PD*EAD; collection costly)",
    "relationship_led": "depth economics (household breadth + engagement; anti-extraction)",
    "threshold_tier": "non-linear: profit only for high-spend AND low-risk tiers",
    "interaction": "multiplicative spend*(1-risk)*engagement minus costs",
    "f5_led": "ADVERSARIAL: f5 really is the driver (guards against anti-f5 overfit)",
}
