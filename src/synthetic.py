"""
synthetic.py
============
Faithful synthetic data generator used ONLY when the real competition file is absent.

The real 500K-row file must be downloaded from Unstop into `data/train.csv`. Until then we
generate data whose columns mirror the KNOWN schema (data/feature_description.csv) and bake
in a HIDDEN ground-truth profit, so we can exercise the pipeline end-to-end and actually
MEASURE the top-20% accuracy of every framework.

True column meanings honoured here:
  f1 revolve balance | f2 cancel calls | f3 cancel calls (collection) | f4 points balance
  f5 TOTAL spend | f6 airlines | f7 other | f8 entertainment | f9 lodging | f10 dining
  f11 risk score | f12 logins | f13 lounge count | f14 airline credit $ | f15 cab use
  f16 entertainment credit $ | f17 lend line | f18 consumer lend line | f19 supp accts
  f20 active charge cards | f21 points redeemed | f22 emails opened | f23 emails clicked

Anti-circularity: the hidden P&L uses category-specific margins, a 5x-rewards drag, tail
credit losses, servicing/collection costs and multiplicative noise, with coefficients that
are DIFFERENT from any scorer's — so recovering the top-20% is a genuine reverse-engineering
test, and customers span affluent power-users, mass-affluent revolvers and loss-making
coupon-clippers so the profit tail and the raw-spend tail intentionally differ.
Deterministic given the seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config


def _std01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def make_synthetic_dataset(n: int = 60_000, cfg: Config | None = None,
                           return_truth: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Return (df[id,f1..f23], hidden true_profit). Truth is for validation only."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(cfg.seed)

    # -- latent customer state (never exposed) ---------------------------------------
    # segments: 0 affluent power-user, 1 mass-affluent revolver, 2 coupon-clipper, 3 dormant
    seg = rng.choice(4, size=n, p=[0.20, 0.35, 0.15, 0.30])
    aff_shift = np.select([seg == 0, seg == 1, seg == 2, seg == 3], [1.4, 0.4, -0.3, -0.7])
    affluence = np.clip(rng.normal(0, 0.8, n) + aff_shift, -3, 4)
    engagement = np.clip(0.6 * affluence + rng.normal(0, 0.8, n)
                         + np.select([seg == 0, seg == 2], [0.5, 0.7], 0.0), -3, 4)
    risk_latent = np.clip(-0.4 * affluence + rng.normal(0, 0.9, n)
                          + np.select([seg == 1, seg == 3], [0.7, 0.4], 0.0), -3, 4)
    tenure_years = np.clip(rng.gamma(2.0, 2.0, n) + 0.5 * (affluence + 1), 0.1, 30)
    aff01, eng01, risk01 = _std01(affluence), _std01(engagement), _std01(risk_latent)

    # -- spend categories (f6..f10) then TOTAL (f5 = their sum) -----------------------
    def cat(mu, sig, drive, sparsity=0.0):
        base = rng.lognormal(mu + 1.6 * drive, sig, n)
        if sparsity:
            base = base * (rng.random(n) > sparsity)
        return np.round(base, 2)

    f6 = cat(6.8, 1.2, affluence, sparsity=0.45)          # airlines (5x, lumpy, often 0)
    f9 = cat(6.6, 1.1, affluence, sparsity=0.40)          # lodging (5x)
    f7 = cat(8.2, 0.9, affluence, sparsity=0.05)          # other (catch-all, always present)
    f8 = cat(6.9, 1.0, 0.7 * affluence + 0.3 * engagement, sparsity=0.30)  # entertainment
    f10 = cat(7.3, 0.8, affluence, sparsity=0.05)         # dining
    f5 = np.round(f6 + f7 + f8 + f9 + f10, 2)             # TOTAL spend = sum of categories

    # -- revolve balance (f1): mass-affluent revolvers carry balances -----------------
    revolve_prop = _std01(risk_latent + 0.3 * (seg == 1) - 0.6 * (seg == 0))
    f1 = np.round(rng.lognormal(7.0, 1.0, n) * (rng.random(n) < 0.30 + 0.45 * revolve_prop), 2)

    # -- rewards points: balance (f4) accrues with spend+tenure; redeemed (f21) --------
    pts_earn = f5 * (1 + 3 * _std01(f6 + f9)) * (0.9 + 0.4 * rng.random(n))    # 5x on travel
    f4 = np.round(pts_earn * (0.5 + 0.15 * tenure_years) * (0.7 + 0.6 * rng.random(n)))  # balance
    redeem_rate = np.clip(0.25 + 0.5 * eng01 + rng.normal(0, 0.1, n), 0, 1)
    f21 = np.round(f4 * redeem_rate * 0.4)                                     # redeemed 12m

    # -- risk score (f11) in [0,1]; cancel calls (f2); collection calls (f3) -----------
    f11 = np.round(np.clip(rng.beta(1.2, 20, n) + 0.18 * risk01, 0, 1), 6)
    f2 = rng.poisson(np.clip(0.15 + 1.2 * risk01 + 0.6 * (seg == 3), 0, None)).clip(0, 12)
    f3 = (rng.random(n) < (0.01 + 0.06 * risk01 + 0.05 * (f11 > 0.2))).astype(int) \
        * rng.poisson(1.0, n).clip(0, 5)                                       # rare, severe

    # -- engagement: logins (f12), emails opened (f22), clicked (f23) ------------------
    f12 = rng.poisson(np.clip(3 + 60 * eng01, 0, None)).clip(0, 200)
    f22 = rng.poisson(np.clip(1 + 25 * eng01, 0, None)).clip(0, 120)
    f23 = np.minimum(f22, rng.poisson(np.clip(0.5 + 10 * eng01, 0, None))).clip(0, 120)

    # -- benefit usage / cost: lounge (f13), airline credit $ (f14), cab (f15),
    #    entertainment credit $ (f16). Coupon-clippers (seg 2) max these vs low spend. -----
    lounge_lam = np.clip(1 + 6 * eng01 + 3 * (seg == 2) + 4 * (seg == 0), 0, None)
    f13 = rng.poisson(lounge_lam).clip(0, 40)
    f14 = np.round(np.clip(240 * _std01(0.5 * eng01 + 0.6 * (seg == 2) + 0.3 * (seg == 0)
                                        + rng.normal(0, 0.35, n)), 0, 240), 2)
    f15 = rng.poisson(np.clip(1 + 5 * eng01 + 3 * (seg == 2), 0, None)).clip(0, 15)
    f16 = np.round(np.clip(260 * _std01(0.5 * eng01 + 0.5 * (seg == 2)
                                        + rng.normal(0, 0.35, n)), 0, 260), 2)

    # -- credit lines (f17 total, f18 consumer subset) ~ affluence --------------------
    f17 = np.round(rng.lognormal(10.5, 0.6, n) * (0.6 + 0.8 * aff01), 2)
    f18 = np.round(f17 * np.clip(rng.beta(6, 3, n), 0, 1), 2)                 # subset of f17

    # -- relationship depth: supplementary accounts (f19), active charge cards (f20) ---
    f19 = rng.poisson(np.clip(0.4 + 1.5 * aff01, 0, None)).clip(0, 6)
    f20 = (1 + rng.poisson(np.clip(0.3 + 1.2 * aff01, 0, None))).clip(1, 5)

    df = pd.DataFrame({
        "id": np.arange(n),
        "f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6, "f7": f7, "f8": f8,
        "f9": f9, "f10": f10, "f11": f11, "f12": f12, "f13": f13, "f14": f14, "f15": f15,
        "f16": f16, "f17": f17, "f18": f18, "f19": f19, "f20": f20, "f21": f21, "f22": f22,
        "f23": f23,
    })

    # Inject realistic missingness (sparse-activity columns arrive as blanks in the file).
    for c, frac in {"f6": 0.10, "f8": 0.10, "f9": 0.12, "f1": 0.15, "f14": 0.20,
                    "f16": 0.20, "f21": 0.12, "f3": 0.70, "f13": 0.15}.items():
        mask = rng.random(n) < frac
        df.loc[mask & (df[c] == 0), c] = np.nan

    truth = pd.Series(
        _hidden_true_profit(rng=rng, f5=f5, f6=f6, f9=f9, f1=f1, f11=f11, f13=f13, f14=f14,
                            f15=f15, f16=f16, f2=f2, f3=f3, f20=f20),
        index=df.index, name="true_profit")
    return df, truth


def _hidden_true_profit(*, rng, f5, f6, f9, f1, f11, f13, f14, f15, f16, f2, f3, f20) -> np.ndarray:
    """
    Hidden Amex-style annual P&L (coefficients DIFFERENT from any scorer):

        Profit = Revenue - Cost - Expected Risk Loss

    Revenue : net interchange (~1.1% of TOTAL spend f5) with an extra 5x-rewards drag on
              Airlines(f6)+Lodging(f9); plus thin net interest on the revolve balance f1
              (suppressed by risk); plus a roughly-constant net annual fee.
    Cost    : benefit redemptions — airline credit $ (f14) + entertainment credit $ (f16)
              + lounge visits (f13) + cab uses (f15).
    Risk    : expected credit loss = RiskScore(f11) * Revolve(f1) * LGD  (tail story);
              servicing on cancellation calls (f2); heavy loss on collection calls (f3).
    """
    spend_5x = f6 + f9
    net_interchange = 0.022 * f5 - (0.012 * f5 + 0.007 * spend_5x)     # ~1.0% net, thin on 5x
    interest = 0.12 * f1 * (1 - np.clip(3 * f11, 0, 0.9))
    fee_net = 150.0 * (0.8 + 0.2 * (f20 / 3.0))                        # ~constant -> drops out of rank

    benefit_cost = f14 + f16 + 28.0 * f13 + 14.0 * f15
    risk_loss = f11 * f1 * 2.0                                          # PD * exposure * LGD (tail)
    servicing = 22.0 * f2 + 260.0 * f3                                  # collection is severe

    profit = net_interchange + interest + fee_net - benefit_cost - risk_loss - servicing
    profit = profit * rng.normal(1.0, 0.10, size=len(profit)) + rng.normal(0, 25, size=len(profit))
    return profit
