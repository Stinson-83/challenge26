# Amex Campus Challenge 2026 — Round 1
## Premier Card Customer-Profitability Framework

A modular, research-grade pipeline that rank-orders Premier cardmembers by estimated
**profitability-to-issuer** (`Profit = Revenue − Cost − Expected Risk Loss`) in a fully
unsupervised setting, optimised for the competition's **top-20% overlap accuracy** metric.

> **Metric:** accuracy = % overlap between the actual top-20% most-profitable cardmembers and
> our predicted top-20%. Only the ranking near/above the 80th percentile matters.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) Put the real Unstop file at data/train.csv  (columns: id, f1..f23)
#    If absent, the pipeline runs on a faithful synthetic dataset with a hidden
#    ground-truth profit, so you can validate the whole thing end-to-end.

# 2) Run everything (EDA, features, all frameworks, clustering, PCA, outliers, plots, submission)
python -m src.pipeline

# Switch the primary framework:
python -m src.pipeline --framework dollar_pnl     # or revenue_first | relationship | full_pnl | risk_adjusted | rank_ensemble
```

Outputs land in `outputs/`:
- `submission.csv` — primary submission (id, profitability_score), one row per id.
- `submission_<framework>.csv` — one per framework; **submit several, keep the best public score**.
- `data_quality_report.json`, `diagnostics.json`, `validation_scores.csv`,
  `all_framework_scores.csv`, `scorecard.csv`, `kmeans_cluster_profile.csv`, `run_summary.json`
- 8 PNG visualisations (correlation, distributions, PCA, clusters, outliers, score
  distributions, framework overlap, cluster profitability).

---

## What makes this a leaderboard solution (not a baseline)

- **Robust, rank-based scoring** immune to the extreme skew/outliers in spend & points.
- **Spend-dominant** economics (interchange dwarfs fees/credits) with small, business-signed
  second-order adjustments that can never overturn a full spend decile.
- **A literal dollar P&L** (`Profit = Revenue − Cost − Risk`) that structurally matches how
  Amex computed the true label, including the **5× rewards drag** on Airlines/Lodging.
- **A rank ensemble** that hedges the few genuine uncertainties and stabilises the 80th-pct cut.
- **Honest validation** on a synthetic dataset with a *hidden* Amex-style P&L (coefficients
  differ from the scorers) — measured with the *exact* competition metric across seeds.
- **Real-data diagnostics** that resolve the last ambiguities (f5 vs Σcategories, risk sign).

See **`docs/business_analysis.md`** (business reasoning, unit economics, feature→economics
map, the profitability math) and **`docs/methodology.md`** (pipeline, frameworks, validation,
results).

---

## Project layout

```
challenge26/
├── config.yaml               # feature semantics, group map, framework weights, thresholds
├── requirements.txt
├── data/
│   ├── feature_description.csv   # authoritative feature meanings (provided)
│   └── train.csv                 # <-- drop the real Unstop file here (git-ignored)
├── src/
│   ├── config.py  data_io.py  preprocessing.py  eda.py  diagnostics.py
│   ├── feature_engineering.py  profitability.py  scoring.py
│   ├── clustering.py  dimensionality.py  outliers.py  evaluation.py
│   ├── synthetic.py  submission.py  pipeline.py
├── notebooks/01_exploration.py   # runnable exploration walk-through
├── tests/test_pipeline.py
├── docs/business_analysis.md  docs/methodology.md
└── outputs/                      # generated artefacts + submissions
```

## Rules compliance
- The `id` (identifier) is **never used** inside any equation — only carried to the output.
- The solution runs on **all** unique identifiers; **no rows are added, removed, or altered**.
- Output column names auto-align to the Unstop template if you set
  `submission.template_file` in `config.yaml`.
