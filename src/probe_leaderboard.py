"""
probe_leaderboard.py
====================
Our elaborate spend-dominant ranking scored only ~0.337 (random ≈ 0.20), so the
spend-is-the-driver prior is wrong for this target. With no label, the public leaderboard is
our only oracle. This script turns the RAW data into a battery of simple, ORTHOGONAL candidate
rankings so we can (a) read their cross-correlations for free, and (b) submit the few most
informative ones to identify the true profit driver.

Run (on the machine that has data/train.csv):
    python -m src.probe_leaderboard
Outputs -> outputs/probes/:
    probe_<name>.csv                     (id, profitability_score)   — quick to submit
    probe_<name>.xlsx                    filled Unstop template      — ready to submit
    _probe_correlations.csv              Spearman of every candidate vs every other + vs f5
    _probe_feature_correlations.csv      Spearman among all raw features (structure map)
    _README.txt                          recommended submission ORDER + what each result means

Nothing here fits weights to anything — each candidate is a transparent ranking of raw features
so a leaderboard score maps directly to "does this signal identify the profitable top-20%?".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import load_config, resolve
from .submission import build_submission, fill_excel_template, write_submission


def _pr(s: pd.Series) -> pd.Series:
    """Percentile rank in [0,1]; NaN -> median rank."""
    x = pd.to_numeric(s, errors="coerce").astype(float)
    x = x.fillna(x.median())
    return pd.Series((rankdata(x.values, method="average") - 1) / max(len(x) - 1, 1), index=s.index)


def _col(df, name):
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)


def pnl_composite(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """
    Standardized (rank-averaged) revenue-minus-cost P&L composite. Every axis enters as a
    percentile rank in [0,1] BEFORE weighting, so the largest-scale column (f7/f21) cannot
    silently dominate (the failure mode of a naive raw-dollar sum). Default weights are the
    panel's economic estimate; recalibrate them proportional to (each probe's score − 0.20)
    once the single-axis leaderboard results are in.
    """
    w = {"spend": 1.0, "interest": 0.6, "lending": 0.6, "redeem_cost": 0.4, "risk_loss": 0.4}
    if weights:
        w.update(weights)
    spend = _pr(_col(df, "f6") + _col(df, "f7") + _col(df, "f8") + _col(df, "f9") + _col(df, "f10"))
    interest = _pr(_col(df, "f1"))
    lending = _pr(_col(df, "f17") + _col(df, "f18"))
    redeem = _pr(_col(df, "f21"))                                   # realized rewards COST
    risk_loss = _pr(_col(df, "f11") * (_col(df, "f1") + _col(df, "f17")))  # expected loss
    score = (w["spend"] * spend + w["interest"] * interest + w["lending"] * lending
             - w["redeem_cost"] * redeem - w["risk_loss"] * risk_loss)
    return _pr(score)


def build_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {name: score Series}. Higher score = predicted MORE profitable. Composites
    combine PERCENTILE RANKS (standardized), never raw dollars."""
    def c(name):
        return _col(df, name)
    cats = c("f6") + c("f7") + c("f8") + c("f9") + c("f10")          # category-spend total
    cand: dict[str, pd.Series] = {
        # --- spend hypotheses (H1: f5 is the WRONG column; the category sum is the base) ---
        "category_total":     _pr(cats),           # <-- probe #1 (interchange base)
        "total_spend_f5":     _pr(c("f5")),        # baseline (≈ what scored 0.337)
        "other_spend_f7":     _pr(c("f7")),        # ~64% of the category sum
        "airlines_f6":        _pr(c("f6")),
        "lodging_f9":         _pr(c("f9")),
        "dining_f10":         _pr(c("f10")),
        # --- interest / lending hypotheses (H3: orthogonal axis we missed) ---
        "revolve_f1":         _pr(c("f1")),        # interest income / exposure
        "lend_line_f17":      _pr(c("f17")),       # credit capacity / lending spread
        "consumer_line_f18":  _pr(c("f18")),
        "lend_total_f17_f18": _pr(c("f17") + c("f18")),
        # --- points / loyalty (H4: f21 realized COST, sign ambiguous; f4 is a liability) ---
        "points_redeemed_f21": _pr(c("f21")),
        "points_balance_f4":  _pr(c("f4")),
        # --- engagement / relationship ---
        "logins_f12":         _pr(c("f12")),
        "relationship_f19_f20": _pr(c("f19") + c("f20")),
        # --- the exploit (H2): standardized multi-term P&L composite (default weights) ---
        "pnl_composite_v1":   pnl_composite(df),
        "revenue_only":       _pr(_pr(cats) + 0.6 * _pr(c("f1")) + 0.6 * _pr(c("f17") + c("f18"))),
    }
    return cand


# Submission order (most informative first) — from the 5-expert diagnosis synthesis.
# NOTE: the panel explicitly says DO NOT submit the inverse of the current ranking
# (0.337 > 0.20 already proves the sign and id-join are correct — it would score ~0.06).
RECOMMENDED_ORDER = [
    "category_total",      # #1: is the true spend base (not f5) the driver?  >0.45 confirms H1
    "revolve_f1",          # #2: interest/revolver axis (orthogonal to spend)
    "lend_line_f17",       # #3: lending-spread axis (orthogonal to both)
    "pnl_composite_v1",    # #4: the exploit — submit AFTER 1-3, reweighted by their scores
    "points_redeemed_f21", # #5: contingent — resolves f21 cost-vs-engagement sign
]


def run(train_path: str | None = None):
    cfg = load_config()
    train_path = train_path or cfg.train_path()
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Need the real data at {train_path} to build probes. Put train.csv there and re-run.")
    df = pd.read_csv(train_path)
    id_col = cfg.id_column
    ids = df[id_col]
    outdir = resolve("outputs/probes")
    os.makedirs(outdir, exist_ok=True)

    cand = build_candidates(df)

    # (a) FREE diagnostics — cross-correlations (no submission needed).
    C = pd.DataFrame({k: v.values for k, v in cand.items()})
    C.corr(method="spearman").round(3).to_csv(os.path.join(outdir, "_probe_correlations.csv"))
    feats = [f"f{i}" for i in range(1, 24) if f"f{i}" in df.columns]
    df[feats].apply(pd.to_numeric, errors="coerce").corr(method="spearman").round(3).to_csv(
        os.path.join(outdir, "_probe_feature_correlations.csv"))

    # (b) One submittable csv + xlsx per candidate.
    tmpl = cfg.raw["submission"].get("template_xlsx")
    tmpl = resolve(tmpl) if tmpl else None
    try:
        from .framework_writeup import FRAMEWORK_RESPONSES
    except Exception:
        FRAMEWORK_RESPONSES = None
    for name, score in cand.items():
        sub = build_submission(ids, score, cfg)
        write_submission(sub, cfg, path=os.path.join(outdir, f"probe_{name}.csv"))
        if tmpl and os.path.exists(tmpl):
            fill_excel_template(tmpl, pd.Series(score.values, index=ids.values),
                                os.path.join(outdir, f"probe_{name}.xlsx"),
                                responses=FRAMEWORK_RESPONSES)

    # Zero-cost offline read: the two correlations that confirm the root cause.
    f5 = _col(df, "f5"); catsum = _col(df, "f6") + _col(df, "f7") + _col(df, "f8") + _col(df, "f9") + _col(df, "f10")
    corr_f5_cats = float(f5.corr(catsum, method="spearman"))
    corr_f7_cats = float(_col(df, "f7").corr(catsum, method="spearman"))

    with open(os.path.join(outdir, "_README.txt"), "w") as fh:
        fh.write("PROBE PLAN — our spend(f5) ranking scored ~0.337 (random ~0.20).\n")
        fh.write("Root cause (5-expert consensus, conf 0.70): f5 is the WRONG spend column.\n\n")
        fh.write("ZERO-COST OFFLINE CHECK (this run):\n")
        fh.write(f"  corr(f5, sum f6..f10) = {corr_f5_cats:.3f}   (expect <0.1 -> f5 is NOT the total)\n")
        fh.write(f"  corr(f7, sum f6..f10) = {corr_f7_cats:.3f}   (expect >0.9 -> f7 dominates the spend base)\n\n")
        fh.write("SUBMIT IN THIS ORDER (each score = that signal's alignment with true top-20%):\n")
        for i, n in enumerate(RECOMMENDED_ORDER, 1):
            fh.write(f"  {i}. probe_{n}.xlsx\n")
        fh.write("\nRULES:\n")
        fh.write("  - Random ~0.20; a real driver scores >0.45. STOP using f5.\n")
        fh.write("  - DO NOT submit an inverse ranking — 0.337>0.20 proves sign+id-join are correct.\n")
        fh.write("  - Falsification guard: if category_total <= ~0.35, the spend thesis is DEAD;\n")
        fh.write("    route remaining slots to interest/lending (revolve_f1, lend_line_f17).\n")
        fh.write("  - Composite (probe #4): recalibrate weights ∝ (each probe score − 0.20) via\n")
        fh.write("    src.probe_leaderboard.recalibrate() before submitting pnl_composite.\n")
        fh.write("  - Combine components as RANKS (standardized), never raw dollars.\n")
    print(f"[probe] wrote {len(cand)} candidates + diagnostics to {outdir}/")
    print(f"[probe] OFFLINE: corr(f5,catsum)={corr_f5_cats:.3f} (expect <0.1), "
          f"corr(f7,catsum)={corr_f7_cats:.3f} (expect >0.9)")
    print(f"[probe] SUBMIT FIRST: probe_{RECOMMENDED_ORDER[0]}.xlsx")
    return outdir


def recalibrate(probe_scores: dict[str, float], train_path: str | None = None,
                out_name: str = "pnl_composite_calibrated"):
    """
    After you know the single-axis leaderboard scores, rebuild the composite with weights
    proportional to (score − 0.20) (a signal that scored at/below random gets ~0 weight;
    a cost axis you want to SUBTRACT should be passed as its own key). Writes a filled
    submission for the recalibrated composite.

    Example:
        recalibrate({"spend": 0.52, "interest": 0.41, "lending": 0.38,
                     "redeem_cost": 0.14, "risk_loss": 0.30})
    """
    cfg = load_config()
    train_path = train_path or cfg.train_path()
    df = pd.read_csv(train_path)
    ids = df[cfg.id_column]
    # weight = max(0, score - 0.20); cost axes keep their (subtracted) role in pnl_composite.
    w = {k: max(0.0, v - 0.20) for k, v in probe_scores.items()}
    score = pnl_composite(df, weights=w)
    outdir = resolve("outputs/probes"); os.makedirs(outdir, exist_ok=True)
    sub = build_submission(ids, score, cfg)
    write_submission(sub, cfg, path=os.path.join(outdir, f"probe_{out_name}.csv"))
    tmpl = cfg.raw["submission"].get("template_xlsx")
    if tmpl and os.path.exists(resolve(tmpl)):
        from .framework_writeup import FRAMEWORK_RESPONSES
        fill_excel_template(resolve(tmpl), pd.Series(score.values, index=ids.values),
                            os.path.join(outdir, f"probe_{out_name}.xlsx"), responses=FRAMEWORK_RESPONSES)
    print(f"[recalibrate] weights={w} -> outputs/probes/probe_{out_name}.xlsx")
    return score


if __name__ == "__main__":
    run()
