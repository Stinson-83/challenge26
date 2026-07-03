"""
eda.py
======
Automated exploratory data analysis + a machine-readable data-quality report.

Covers every item requested in the brief: missing values, duplicates, constant columns,
identifier detection, numeric-vs-categorical split, outliers (IQR & z-score), correlation,
distribution & skewness, and unsupervised feature-importance heuristics (variance,
correlation-to-first-PC, and correlation-to-a-provisional spend aggregate).
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import Config, load_config
from .preprocessing import (
    ColumnRole,
    constant_columns,
    duplicate_report,
    feature_columns,
    identifier_columns,
    infer_column_roles,
    missing_report,
)


def outlier_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """IQR-fence and 3-sigma outlier counts per column."""
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        if s.std(ddof=0) > 0:
            z = (s - s.mean()) / s.std(ddof=0)
            n_z = int((z.abs() > 3).sum())
        else:
            n_z = 0
        rows.append({
            "column": c, "n_iqr_outliers": n_iqr, "frac_iqr": n_iqr / len(s),
            "n_z3_outliers": n_z, "frac_z3": n_z / len(s),
            "p99_over_p50": float(s.quantile(0.99) / s.quantile(0.50)) if s.quantile(0.50) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("frac_iqr", ascending=False)


def skewness_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 3:
            continue
        rows.append({
            "column": c, "skew": float(s.skew()), "kurtosis": float(s.kurtosis()),
            "frac_zero": float((s == 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("skew", ascending=False)


def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "spearman") -> pd.DataFrame:
    """Spearman by default — rank correlation is robust to the heavy skew in spend columns."""
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)


def unsupervised_importance(df: pd.DataFrame, cols: list[str], cfg: Config) -> pd.DataFrame:
    """
    Heuristic importance without labels:
      - variance of the robust-scaled column (information content),
      - |corr| with the first principal component (shared-variance leadership),
      - |corr| with a provisional 'total spend' aggregate (proxy for the dominant
        revenue driver, which we believe leads the profit ranking).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xs = pd.DataFrame(RobustScaler().fit_transform(X), columns=cols, index=X.index)

    pca = PCA(n_components=1, random_state=cfg.seed).fit(Xs.values)
    pc1 = Xs.values @ pca.components_[0]

    spend_cols = [c for c in cols if cfg.semantics.get(c, {}).get("group") in ("spend", "revolve")]
    spend_proxy = X[spend_cols].sum(axis=1) if spend_cols else pd.Series(pc1, index=X.index)

    rows = []
    for c in cols:
        col = Xs[c]
        rows.append({
            "column": c,
            "robust_variance": float(np.var(col.values)),
            "abs_corr_pc1": float(abs(np.corrcoef(col.values, pc1)[0, 1])) if col.std() > 0 else 0.0,
            "abs_corr_spend_proxy": float(abs(col.corr(spend_proxy, method="spearman"))),
        })
    imp = pd.DataFrame(rows)
    # Composite heuristic rank.
    for c in ["robust_variance", "abs_corr_pc1", "abs_corr_spend_proxy"]:
        imp[c + "_rank"] = imp[c].rank(pct=True)
    imp["heuristic_importance"] = imp[[c + "_rank" for c in
                                       ["robust_variance", "abs_corr_pc1", "abs_corr_spend_proxy"]]].mean(axis=1)
    return imp.sort_values("heuristic_importance", ascending=False)


def data_quality_report(df: pd.DataFrame, cfg: Config | None = None) -> dict:
    """One call -> a complete, serialisable data-quality + EDA summary."""
    cfg = cfg or load_config()
    roles = infer_column_roles(df, cfg)
    feats = feature_columns(df, cfg, roles)

    role_counts: dict[str, int] = {}
    for r in roles.values():
        role_counts[r.role] = role_counts.get(r.role, 0) + 1

    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "id_column": cfg.id_column,
        "identifier_columns": identifier_columns(roles),
        "constant_columns": constant_columns(df),
        "feature_columns": feats,
        "n_features_used": len(feats),
        "role_counts": role_counts,
        "duplicates": duplicate_report(df, cfg.id_column),
        "missing_top": missing_report(df).head(25).round(4).reset_index()
                        .rename(columns={"index": "column"}).to_dict("records"),
        "column_roles": {c: asdict(r) for c, r in roles.items()},
        "skew_top": skewness_report(df, feats).head(25).round(4).to_dict("records"),
        "outliers_top": outlier_report(df, feats).head(25).round(4).to_dict("records"),
        "importance": unsupervised_importance(df, feats, cfg).round(4).to_dict("records"),
    }
    return report


def save_report(report: dict, path: str) -> str:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
