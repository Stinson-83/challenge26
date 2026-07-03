# outputs/ — generated artefacts

**These committed files are EXAMPLE artefacts from a synthetic validation run** (no real
`data/train.csv` was present, so the pipeline used the faithful synthetic generator with a
hidden Amex-style ground-truth profit). They exist so the deliverable is viewable without
running anything. Re-running `python -m src.pipeline` overwrites them; on the real Unstop
data the submissions here are replaced by the real ranking.

| File | What it is |
|---|---|
| `submission.csv` | Primary submission (active framework = rank ensemble): `id, profitability_score`, one row per id. |
| `submission_<framework>.csv` | One submission per framework — upload several, keep the best public-leaderboard score. |
| `validation_scores.csv` | Top-20%-overlap accuracy of every framework vs the hidden synthetic truth. |
| `all_framework_scores.csv` | Per-id score + percentile rank for every framework. |
| `scorecard.csv` | Seven interpretable sub-scores + three combined overalls per id. |
| `framework_agreement.csv` | Pairwise top-20% overlap + Spearman between frameworks. |
| `kmeans_cluster_profile.csv` | Cluster means, sorted most→least profitable. |
| `pca_loadings.csv` | PCA loadings (PC1 = the latent profitability axis). |
| `data_quality_report.json` | Missing/dup/constant/outlier/skew/importance report. |
| `diagnostics.json` | Real-data checks: f5 vs Σcategories, credit-line redundancy, risk-sign sanity. |
| `run_summary.json` | One-glance run summary (roles, best framework, paths, timings). |
| `*.png` | Correlation heatmap, distributions, PCA/cluster/outlier scatter, score distributions, framework overlap, cluster profitability. |
