"""
preprocessing.py
================
Cleaning + automated schema understanding.

The single most important function here is `infer_column_roles`: because Amex masks the
column names, we recover each column's *statistical role* (identifier / binary / rare-flag
/ ratio / count / dollar / score) from its signature. These auto-detected roles are then
reconciled with the human-inferred `semantics` map in config.yaml, so a wrong hand guess
degrades gracefully and disagreements are surfaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, load_config


# ---------------------------------------------------------------------------------------
# Column-role inference
# ---------------------------------------------------------------------------------------
@dataclass
class ColumnRole:
    name: str
    role: str                       # identifier|binary|rare_flag|ratio|count|dollar|score|constant|other
    n_missing: int
    frac_missing: float
    n_unique: int
    frac_zero: float
    min: float
    max: float
    mean: float
    std: float
    skew: float
    has_negative: bool
    notes: str = ""


def infer_column_roles(df: pd.DataFrame, cfg: Config | None = None) -> dict[str, ColumnRole]:
    """Infer a statistical role for every column from its numeric signature."""
    cfg = cfg or load_config()
    ad = cfg.autodetect
    n = len(df)
    roles: dict[str, ColumnRole] = {}

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        nonnull = s.dropna()
        n_missing = int(s.isna().sum())
        n_unique = int(nonnull.nunique())
        frac_zero = float((nonnull == 0).mean()) if len(nonnull) else 0.0
        vmin = float(nonnull.min()) if len(nonnull) else np.nan
        vmax = float(nonnull.max()) if len(nonnull) else np.nan
        mean = float(nonnull.mean()) if len(nonnull) else np.nan
        std = float(nonnull.std()) if len(nonnull) else np.nan
        skew = float(nonnull.skew()) if len(nonnull) > 2 else 0.0
        has_neg = bool((nonnull < 0).any())

        role, notes = _classify(
            col=col, s=nonnull, n=n, n_unique=n_unique, frac_zero=frac_zero,
            vmin=vmin, vmax=vmax, skew=skew, has_neg=has_neg, ad=ad, id_col=cfg.id_column,
        )

        roles[col] = ColumnRole(
            name=col, role=role, n_missing=n_missing,
            frac_missing=n_missing / n if n else 0.0, n_unique=n_unique,
            frac_zero=frac_zero, min=vmin, max=vmax, mean=mean, std=std,
            skew=skew, has_negative=has_neg, notes=notes,
        )
    return roles


def _classify(*, col, s, n, n_unique, frac_zero, vmin, vmax, skew, has_neg, ad, id_col) -> tuple[str, str]:
    """Rule-based classifier for a single column's statistical role."""
    if n_unique <= 1:
        return "constant", "single value"
    if len(s) == 0:
        return "other", "all missing"

    is_intlike = bool(np.allclose(s.values, np.round(s.values), equal_nan=False))

    # Identifier: name match OR near-unique monotonic integer sequence.
    if col == id_col:
        return "identifier", "declared id"
    if is_intlike and (n_unique / max(n, 1)) >= ad.get("id_unique_frac", 0.98) and vmin >= 0:
        return "identifier", "near-unique integer"

    # Binary / rare flag.
    if n_unique == 2 and set(np.unique(np.round(s.values))).issubset({0, 1}):
        pos = float((s == s.max()).mean())
        if pos <= ad.get("rare_flag_frac", 0.15) or (1 - pos) <= ad.get("rare_flag_frac", 0.15):
            return "rare_flag", f"binary, positive rate {min(pos, 1 - pos):.3f}"
        return "binary", "binary 0/1"

    # Ratio / probability in [0,1].
    if vmin >= -1e-9 and vmax <= ad.get("ratio_max", 1.05) and not is_intlike:
        return "ratio", "bounded in [0,1]"

    # Count: few small non-negative integers.
    if (is_intlike and vmin >= 0 and n_unique <= ad.get("count_max_unique", 40)
            and vmax <= ad.get("count_max_value", 400)):
        return "count", f"{n_unique} small integer levels"

    # Dollar amount: wide continuous range, right skewed, or has negatives (returns).
    rng = (vmax - vmin) if np.isfinite(vmax) and np.isfinite(vmin) else 0.0
    if rng >= ad.get("dollar_min_range", 500) and (skew >= ad.get("skew_dollar_min", 1.5) or has_neg):
        return "dollar", f"wide range {rng:,.0f}, skew {skew:.2f}"

    # Bounded score: continuous, moderate range, not obviously money.
    if not is_intlike and rng < ad.get("dollar_min_range", 500):
        return "score", f"bounded continuous, range {rng:,.2f}"

    # Fallback bucket for medium-range integers / everything else.
    return "dollar" if rng >= ad.get("dollar_min_range", 500) else "score", "fallback"


# ---------------------------------------------------------------------------------------
# Data-quality primitives
# ---------------------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum()
    out = pd.DataFrame({"n_missing": m, "frac_missing": m / len(df)})
    return out.sort_values("frac_missing", ascending=False)


def duplicate_report(df: pd.DataFrame, id_column: str | None = None) -> dict:
    feat = df.drop(columns=[id_column]) if id_column and id_column in df.columns else df
    dup_full = int(df.duplicated().sum())
    dup_feat = int(feat.duplicated().sum())
    id_dups = 0
    if id_column and id_column in df.columns:
        id_dups = int(df[id_column].duplicated().sum())
    return {
        "duplicate_full_rows": dup_full,
        "duplicate_feature_rows": dup_feat,
        "duplicate_ids": id_dups,
    }


def constant_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


def identifier_columns(roles: dict[str, ColumnRole]) -> list[str]:
    return [c for c, r in roles.items() if r.role == "identifier"]


# ---------------------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------------------
def clean(
    df: pd.DataFrame,
    cfg: Config | None = None,
    roles: dict[str, ColumnRole] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Non-destructive cleaning that RESPECTS the competition rule of not adding/removing rows:
      - numeric coercion
      - missing-value imputation that is *role aware* (0 for sparse dollar/count where NaN
        means "no activity"; median for scores/ratios where NaN means "unobserved")
      - never drops or duplicates rows; never touches the id column values.

    Returns (clean_df, log).
    """
    cfg = cfg or load_config()
    roles = roles or infer_column_roles(df, cfg)
    out = df.copy()
    log: dict = {"imputed": {}, "coerced": []}

    id_col = cfg.id_column
    for col in out.columns:
        if col == id_col:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if not s.equals(out[col]):
            log["coerced"].append(col)
        role = roles[col].role if col in roles else "other"
        if s.isna().any():
            if role in ("dollar", "count", "rare_flag", "binary"):
                fill = 0.0                    # absence of activity -> zero
                strat = "zero(no-activity)"
            elif role in ("ratio", "score"):
                fill = float(s.median())      # unobserved score -> median
                strat = "median"
            else:
                fill = float(s.median()) if s.notna().any() else 0.0
                strat = "median"
            s = s.fillna(fill)
            log["imputed"][col] = {"strategy": strat, "value": fill,
                                   "n": int(df[col].isna().sum())}
        out[col] = s
    return out, log


def feature_columns(df: pd.DataFrame, cfg: Config | None = None,
                    roles: dict[str, ColumnRole] | None = None) -> list[str]:
    """Modelling features = everything except identifiers and constants (rule-compliant)."""
    cfg = cfg or load_config()
    roles = roles or infer_column_roles(df, cfg)
    excluded = set(identifier_columns(roles)) | set(constant_columns(df)) | {cfg.id_column}
    return [c for c in df.columns if c not in excluded]
