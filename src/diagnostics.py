"""
diagnostics.py
==============
Data-dependent checks that resolve the few remaining modelling ambiguities ON THE REAL
DATA (they cannot be settled from priors alone). Run these once the real train.csv is in
place; each returns a machine-readable verdict the pipeline logs and can act on.

Checks
------
1. total_spend_consistency : is f5 ("Total Spend") ~= sum of the category columns
   (f6..f10)? If yes, f5 is the clean aggregate and the category-sum corroborator is
   redundant (harmless). If not, f5 measures something broader — we then trust f5 as the
   primary but keep the corroborator meaningful.
2. credit_line_redundancy : is f18 (consumer lend line) ~ a subset of f17 (total lend
   line)? Confirms they are near-duplicates so the credit_line group isn't double-weighted.
3. risk_sign_sanity : does higher f11 (risk score) co-move with credit-distress signals
   (f3 collection calls, f2 cancel calls, f1 revolve balance)? If yes, higher f11 = riskier
   (our assumed sign). If it instead co-moves with spend/affluence, the sign may be flipped
   and risk should be down-weighted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, load_config


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else None


def total_spend_consistency(df: pd.DataFrame, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    sem = cfg.semantics
    total = [c for c, m in sem.items() if m.get("group") == "spend"]
    cats = [c for c, m in sem.items() if m.get("group") in ("spend_cat_5x", "spend_cat_1x")]
    if not total or not cats:
        return {"status": "skipped", "reason": "missing spend columns"}
    f5 = _num(df, total[0]).fillna(0)
    catsum = df[cats].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    ratio = (f5 / catsum.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    frac_close = float((((f5 - catsum).abs()) <= (0.02 * catsum + 1)).mean())
    return {
        "check": "total_spend_consistency",
        "median_f5_over_catsum": round(float(ratio.median()), 4),
        "frac_rows_f5_equals_catsum": round(frac_close, 4),
        "verdict": ("f5 == sum(categories): use f5 as clean total (corroborator redundant)"
                    if frac_close > 0.9 else
                    "f5 differs from category sum: f5 is a broader/independent total — keep both"),
    }


def credit_line_redundancy(df: pd.DataFrame, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    cl = [c for c, m in cfg.semantics.items() if m.get("group") == "credit_line"]
    if len(cl) < 2:
        return {"status": "skipped"}
    a, b = _num(df, cl[0]), _num(df, cl[1])
    corr = float(a.corr(b, method="spearman"))
    frac_subset = float((b.fillna(0) <= a.fillna(0) + 1).mean())
    return {"check": "credit_line_redundancy", "spearman": round(corr, 4),
            "frac_f18_le_f17": round(frac_subset, 4),
            "verdict": "near-duplicate credit lines (expected)" if corr > 0.8 else "distinct lines"}


def risk_sign_sanity(df: pd.DataFrame, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    sem = cfg.semantics
    def one(group):
        cols = [c for c, m in sem.items() if m.get("group") == group]
        return _num(df, cols[0]) if cols else None
    f11 = one("risk")
    if f11 is None:
        return {"status": "skipped"}
    distress = {}
    for name, grp in [("collection_f3", "collection"), ("cancel_f2", "attrition"),
                      ("revolve_f1", "revolve"), ("spend_f5", "spend")]:
        s = one(grp)
        if s is not None:
            distress[name] = round(float(f11.corr(s, method="spearman")), 4)
    # Higher risk should correlate POSITIVELY with distress signals, and NOT strongly with spend.
    distress_signal = np.nanmean([distress.get("collection_f3", np.nan),
                                  distress.get("cancel_f2", np.nan),
                                  distress.get("revolve_f1", np.nan)])
    verdict = ("risk sign OK (co-moves with distress)" if distress_signal > 0.02
               else "WARNING: f11 does not co-move with distress — consider down-weighting/ flipping risk")
    return {"check": "risk_sign_sanity", "corr_f11_with": distress, "verdict": verdict}


def run_all(df: pd.DataFrame, cfg: Config | None = None) -> dict:
    cfg = cfg or load_config()
    return {
        "total_spend_consistency": total_spend_consistency(df, cfg),
        "credit_line_redundancy": credit_line_redundancy(df, cfg),
        "risk_sign_sanity": risk_sign_sanity(df, cfg),
    }
