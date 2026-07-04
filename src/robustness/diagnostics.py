"""
diagnostics.py — label-free construct-validity oracles.
======================================================
These check feature IDENTITY / QUALITY / SIGN using the data alone — no leaderboard, no label.
They exist so the rebuilt model can JUSTIFY its feature choices from first principles (the
"data-motivated, LB-corroborated" mandate), and so a wrong masked-feature guess is caught
before it silently mis-ranks both splits.

They operationalize the five signals that independently reject f5-as-"Total Spend":
  1. censoring/cap detector          (f5 and every continuous feature are winsorized)
  2. total-vs-parts sanity           (a true total cannot be below the sum of its parts)
  3. pairwise redundancy             (f17/f18 are duplicates -> collapse)
  4. block co-missingness            (structural missingness = stable segments)
  5. rewards-accrual consistency     (real spend co-moves with points earned/redeemed; noise doesn't)

Run on the RAW dataframe (NaN preserved).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FCOLS = [f"f{i}" for i in range(1, 24)]


def _num(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(dtype=float)


def censoring_report(df: pd.DataFrame, cont_cols=None) -> dict:
    """Flag columns whose max is a hard cap (winsorized): mass piled at max & max==p99."""
    if cont_cols is None:
        cont_cols = [c for c in FCOLS if c in df.columns and _num(df, c).nunique() > 50]
    rows = {}
    for c in cont_cols:
        s = _num(df, c).dropna()
        if len(s) == 0:
            continue
        mx = s.max()
        p99 = s.quantile(0.99)
        frac_at_max = float((s == mx).mean())
        rows[c] = {
            "max": float(mx), "p99": float(p99),
            "max_eq_p99": bool(np.isclose(mx, p99)),
            "frac_at_max_pct": round(100 * frac_at_max, 3),
            "censored": bool(np.isclose(mx, p99) and frac_at_max > 0.01),
        }
    n_censored = sum(v["censored"] for v in rows.values())
    return {"per_column": rows, "n_censored": n_censored,
            "verdict": ("global upper-winsorization detected" if n_censored >= 3
                        else "no systematic censoring")}


def total_vs_parts(df: pd.DataFrame, total="f5", parts=("f6", "f7", "f8", "f9", "f10")) -> dict:
    """A claimed TOTAL must be >= the sum of its parts and correlate with it. Tests f5 label."""
    t = _num(df, total)
    p = sum(_num(df, c).fillna(0.0) for c in parts)
    both = t.notna() & (sum(_num(df, c).notna() for c in parts) > 0)
    tt, pp = t[both], p[both]
    frac_total_lt_parts = float((tt < pp).mean()) if len(tt) else np.nan
    sp = float(tt.corr(pp, method="spearman")) if len(tt) else np.nan
    ratio_med = float((tt / pp.replace(0, np.nan)).median()) if len(tt) else np.nan
    plausible = (frac_total_lt_parts < 0.05) and (sp > 0.5)
    return {
        "total": total, "parts": list(parts), "n": int(both.sum()),
        "frac_total_below_sum_of_parts": round(frac_total_lt_parts, 4),
        "spearman_total_vs_partsum": round(sp, 4),
        "median_total_over_partsum": round(ratio_med, 4),
        "label_plausible": bool(plausible),
        "verdict": ("consistent with a true total" if plausible
                    else f"REFUTES '{total} = sum of parts' (below-parts in "
                          f"{round(100*frac_total_lt_parts,1)}% of rows, corr {round(sp,3)})"),
    }


def redundancy(df: pd.DataFrame, thresh: float = 0.90) -> dict:
    """Pairwise near-duplicate features (|Spearman| > thresh), pairwise-complete."""
    cols = [c for c in FCOLS if c in df.columns]
    dups = []
    for i, a in enumerate(cols):
        sa = _num(df, a)
        for b in cols[i + 1:]:
            sb = _num(df, b)
            m = sa.notna() & sb.notna()
            if m.sum() < 1000:
                continue
            rho = float(sa[m].corr(sb[m], method="spearman"))
            if abs(rho) > thresh:
                dups.append({"a": a, "b": b, "spearman": round(rho, 4), "n": int(m.sum())})
    return {"threshold": thresh, "near_duplicate_pairs": dups,
            "verdict": (f"{len(dups)} redundant pair(s) — collapse each to one feature"
                        if dups else "no near-duplicates")}


def block_comissingness(df: pd.DataFrame) -> dict:
    """Identify columns that go missing together (structural segments)."""
    miss = pd.DataFrame({c: df[c].isna().astype(float) for c in FCOLS if c in df.columns})
    frac = miss.mean().round(4)
    varying = [c for c in miss.columns if 0.001 < frac[c] < 0.999]
    blocks, seen = [], set()
    for c in varying:
        if c in seen:
            continue
        block = [c]
        for d in varying:
            if d == c or d in seen:
                continue
            # identical (or near-identical) missingness mask?
            if float((miss[c] == miss[d]).mean()) > 0.995:
                block.append(d)
        if len(block) > 1:
            for d in block:
                seen.add(d)
            blocks.append({"cols": block, "missing_frac": float(frac[block[0]])})
    return {"missing_frac": {c: float(frac[c]) for c in miss.columns},
            "co_missing_blocks": blocks,
            "verdict": f"{len(blocks)} structural co-missing block(s) -> candidate segment indicators"}


def rewards_accrual(df: pd.DataFrame) -> dict:
    """
    Rewards points accrue from spend. A REAL spend signal correlates with points balance (f4)
    and points redeemed (f21); a noise column labelled 'spend' does not. This is the clinching
    label-free discriminator between f5 and the category-sum.
    """
    cats = sum(_num(df, c).fillna(0.0) for c in ("f6", "f7", "f8", "f9", "f10"))
    f5 = _num(df, "f5")
    out = {}
    for pts_name in ("f4", "f21"):
        pts = _num(df, pts_name)
        out[f"cats_vs_{pts_name}"] = round(float(cats.corr(pts, method="spearman")), 4)
        out[f"f5_vs_{pts_name}"] = round(float(f5.corr(pts, method="spearman")), 4)
    cats_wins = (out["cats_vs_f4"] > out["f5_vs_f4"]) and (out["cats_vs_f21"] > out["f5_vs_f21"])
    out["verdict"] = ("category-sum behaves like real spend vs rewards; f5 does not "
                      "-> prefer category-sum as the spend axis"
                      if cats_wins else "inconclusive")
    return out


def run_all(df: pd.DataFrame) -> dict:
    """Full label-free construct-validity report on the raw dataframe."""
    return {
        "censoring": censoring_report(df),
        "total_vs_parts_f5": total_vs_parts(df),
        "redundancy": redundancy(df),
        "block_comissingness": block_comissingness(df),
        "rewards_accrual": rewards_accrual(df),
    }
