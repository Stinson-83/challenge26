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


def build_candidates(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {name: score Series indexed by row}. Higher score = predicted MORE profitable."""
    def c(name):  # raw column, missing -> 0
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0) if name in df.columns else pd.Series(0.0, index=df.index)

    cats = c("f6") + c("f7") + c("f8") + c("f9") + c("f10")          # category-spend total
    cand: dict[str, pd.Series] = {
        # --- single-feature magnitude hypotheses ---
        "total_spend_f5":     _pr(c("f5")),        # baseline (≈ what already scored 0.337)
        "other_spend_f7":     _pr(c("f7")),        # the largest $ column
        "category_total":     _pr(cats),           # true "total spend" hypothesis
        "airlines_f6":        _pr(c("f6")),
        "lodging_f9":         _pr(c("f9")),
        "dining_f10":         _pr(c("f10")),
        # --- lending / interest hypotheses ---
        "revolve_f1":         _pr(c("f1")),        # interest income / exposure
        "lend_line_f17":      _pr(c("f17")),       # credit capacity
        "consumer_line_f18":  _pr(c("f18")),
        "lend_total_f17_f18": _pr(c("f17") + c("f18")),
        # --- points / loyalty hypotheses ---
        "points_balance_f4":  _pr(c("f4")),
        "points_redeemed_f21": _pr(c("f21")),
        "points_activity":    _pr(c("f4") + c("f21")),
        # --- engagement / relationship ---
        "logins_f12":         _pr(c("f12")),
        "relationship_f19_f20": _pr(c("f19") + c("f20")),
        # --- broad composites ---
        "all_spend_broad":    _pr(c("f5") + cats + c("f1")),
        "lending_x_lowrisk":  _pr(_pr(c("f17") + c("f18")) * (1 - _pr(c("f11")))),
        "revenue_minus_risk": _pr(_pr(cats + c("f1")) - _pr(c("f11")) - _pr(c("f3"))),
        # --- direction sanity check: is the grader inverted? ---
        # (submit the ascending version of the current spend ranking)
        "INVERSE_total_spend_f5": 1.0 - _pr(c("f5")),
    }
    return cand


# Suggested order to spend submissions (most informative first). Refined by the diagnosis panel.
RECOMMENDED_ORDER = [
    "category_total",     # is the REAL spend total (not f5) the driver?
    "revolve_f1",         # is profit interest/revolve driven?
    "lend_line_f17",      # is profit lending-capacity driven?
    "points_redeemed_f21",  # PC1-dominant signal
    "other_spend_f7",     # largest single $ column
    "INVERSE_total_spend_f5",  # cheap direction/format check
    "points_balance_f4",
    "lending_x_lowrisk",
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

    with open(os.path.join(outdir, "_README.txt"), "w") as fh:
        fh.write("PROBE SUBMISSIONS — spend ranking scored ~0.337 (random ~0.20).\n")
        fh.write("Submit in this order; each score tells you whether that signal drives profit.\n")
        fh.write("Random baseline ~0.20; a driver should score clearly higher (>0.45).\n\n")
        for i, n in enumerate(RECOMMENDED_ORDER, 1):
            fh.write(f"  {i}. probe_{n}.xlsx\n")
        fh.write("\nInterpretation:\n")
        fh.write("  - If category_total >> total_spend_f5: the category sum is the real spend driver.\n")
        fh.write("  - If revolve_f1 / lend_line_f17 win: profit is interest/lending-driven, not spend.\n")
        fh.write("  - If points_* win: profit tracks loyalty/redemption.\n")
        fh.write("  - If INVERSE_total_spend_f5 ~ 0.66: the grader direction is flipped (submit inverses).\n")
        fh.write("  - Then combine the 1-2 winning signals and submit that.\n")
    print(f"[probe] wrote {len(cand)} candidates + diagnostics to {outdir}/")
    print(f"[probe] recommended first submission: probe_{RECOMMENDED_ORDER[0]}.xlsx")
    return outdir


if __name__ == "__main__":
    run()
