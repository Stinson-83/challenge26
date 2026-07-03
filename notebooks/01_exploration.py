"""
01_exploration.py — a runnable walk-through of the analysis.

Run:  python notebooks/01_exploration.py
(Structured as ordered cells; convert to .ipynb with jupytext if desired.)

It loads the data (real if present, else synthetic), prints the data-quality report and
real-data diagnostics, shows the robust group scores, compares every profitability framework
(against the hidden truth when synthetic), and profiles the most-profitable cluster.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import clustering, diagnostics, dimensionality, evaluation
from src.config import load_config
from src.data_io import load_dataset
from src.feature_engineering import build_group_scores, engineer_features
from src.preprocessing import clean, infer_column_roles, missing_report
from src.profitability import framework_agreement, score_dataframe

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)

# --- Cell 1: load -----------------------------------------------------------------------
cfg = load_config()
df, is_real = load_dataset(cfg)
truth = None
if not is_real:
    from src.synthetic import make_synthetic_dataset
    df, truth = make_synthetic_dataset(cfg=cfg)
print(f"Loaded {len(df):,} rows x {df.shape[1]} cols (real={is_real})")

# --- Cell 2: schema + roles -------------------------------------------------------------
roles = infer_column_roles(df, cfg)
print("\nAuto-detected statistical roles:")
for c, r in roles.items():
    sem = cfg.semantics.get(c, {}).get("meaning", "")
    print(f"  {c:>4} | {r.role:>10} | miss={r.frac_missing:5.2%} | {sem[:60]}")

# --- Cell 3: clean + missing ------------------------------------------------------------
clean_df, clean_log = clean(df, cfg, roles)
print("\nTop missing before cleaning:")
print(missing_report(df).head(8).round(4))

# --- Cell 4: real-data diagnostics ------------------------------------------------------
print("\nDiagnostics:")
for k, v in diagnostics.run_all(clean_df, cfg).items():
    print(f"  {k}: {v.get('verdict', v)}")

# --- Cell 5: robust group scores --------------------------------------------------------
gs = build_group_scores(clean_df, cfg)
print("\nGroup-score correlations (Spearman):")
print(gs.corr(method="spearman").round(2))

# --- Cell 6: score every framework ------------------------------------------------------
result = score_dataframe(clean_df, cfg)
print(f"\nActive framework: {result.active}")
print("\nFramework top-20% agreement (overlap / spearman):")
print(framework_agreement(result.scores, cfg.top_pct).to_string(index=False))

# --- Cell 7: validate against hidden truth (synthetic only) -----------------------------
if truth is not None:
    print("\nTOP-20% ACCURACY vs hidden truth:")
    print(evaluation.compare_frameworks(result.scores, truth, cfg.top_pct)
          [["framework", "acc_top20", "acc_top10", "spearman"]].to_string(index=False))

# --- Cell 8: cluster profitability profile ----------------------------------------------
labels = clustering.fit_all(gs, cfg)
prof = clustering.profile_clusters(labels["kmeans"], gs, result.subscores, result.primary_rank)
print("\nKMeans clusters, most→least profitable:")
print(prof[["n", "share", "spend", "risk", "benefit_cost", "profitability_rank"]].round(3))

# --- Cell 9: latent factors -------------------------------------------------------------
pca = dimensionality.run_pca(gs, n_components=5, seed=cfg.seed)
print("\nPCA explained variance:", np.round(pca["explained_variance_ratio"], 3).tolist())
print("PC1 loadings (profitability axis):")
print(pca["loadings"]["PC1"].sort_values(ascending=False).round(3))

print("\nDone. See outputs/ for artefacts and submissions after running `python -m src.pipeline`.")
