# Methodology — Pipeline, Frameworks, Validation & Results

## 0. Design principles (why this beats a baseline)

1. **The metric is top-20% set overlap** → optimise a *stable ranking at the 80th
   percentile*, not calibrated dollars. Every signal is a **robust percentile score in
   [0,1]** (rank-based), so scale, skew, and outliers (f4 points ~570k, f5 spend) cannot
   distort the cut.
2. **The true label is a Revenue−Cost−Risk function of the same 23 attributes**, and it is
   **interchange-dominated** → `spend` carries the ranking (weight 1.0); every other lever
   is a small, business-signed adjustment that can shift a member only a few percentile
   points.
3. **No label ⇒ robustness beats precision.** We never fit weights to data. We use business
   priors and a **rank ensemble** to hedge the few genuine uncertainties (coefficient sizes,
   risk sign), and we emit **one submission per framework** so the public leaderboard makes
   the final call.

## 1. Pipeline (`src/pipeline.py`)

```
load → clean (role-aware imputation) → EDA + data-quality report + real-data diagnostics
     → feature engineering → robust group scores → 6 profitability frameworks
     → clustering (KMeans/GMM/Agglomerative/DBSCAN) + profiling
     → PCA / Factor Analysis / ICA → Isolation Forest / LOF outliers
     → scorecard (7 sub-scores × 3 combine strategies)
     → validation (synthetic only) → visualisations → submissions
```
Run: `python -m src.pipeline` (uses `config.yaml`; falls back to a synthetic dataset with a
hidden ground-truth profit if `data/train.csv` is absent). Heavy analysis (clustering, PCA,
outliers, plots) auto-subsamples to ≤80k rows; **the submission scores always use all rows**
(vectorised percentile ranks — 500k runs in ~2 min).

## 2. Modules

| Module | Responsibility |
|---|---|
| `config.py` / `config.yaml` | Single source of truth: feature semantics, group map, framework weights, thresholds. |
| `preprocessing.py` | Auto-detects each column's statistical role (identifier/binary/ratio/count/dollar/score), reconciles with the semantic map, role-aware missing-value imputation, duplicate/constant/identifier detection. Never adds/drops rows. |
| `eda.py` | Missing / duplicate / outlier / skew / correlation reports + unsupervised feature-importance heuristics + JSON data-quality report. |
| `feature_engineering.py` | Robust primitives (percentile rank, winsorise, log1p-signed, robust-z), the **[0,1] group scores** that every framework consumes, and ~90 business features (efficiency, risk-adjusted spend, engagement/loyalty/relationship indices, CLV proxy, benefit-per-$1k, composites). |
| `profitability.py` | The 5 base frameworks + rank ensemble, the 7 sub-scores, framework-agreement diagnostics, optional boundary guards. |
| `scoring.py` | Multi-score "scorecard": normalise + combine sub-scores under 3 strategies. |
| `clustering.py` | KMeans, GMM, Agglomerative, DBSCAN + per-cluster profitability profiling. |
| `dimensionality.py` | PCA, Factor Analysis, ICA; spend-aligned PC1 as a model-free ranking. |
| `outliers.py` | Isolation Forest + LOF, tagged by profitability tail. |
| `diagnostics.py` | Real-data checks: total-spend consistency (f5 vs Σcategories), credit-line redundancy, risk-sign sanity. |
| `evaluation.py` | The exact competition metric (top-k overlap) + rank diagnostics; framework comparison vs a known truth. |
| `submission.py` | Rule-compliant submission (all ids, no identifier in equation), template auto-alignment, per-framework files. |
| `synthetic.py` | Faithful synthetic data + hidden Amex-style P&L for honest validation. |

## 3. The six frameworks

| Framework | Philosophy | Form |
|---|---|---|
| **revenue_first** | Interchange dominates; spend leads, rest is a light touch. | Linear on group scores, spend=1.0. |
| **full_pnl** | Balanced Revenue−Cost−Risk on robust group scores. | Linear with cost/risk/collection penalties. |
| **risk_adjusted** | Tail losses dominate downside; discount revenue multiplicatively by risk/attrition/collection. | `base·(1−a·risk)·(1−b·attr)·(1−d·coll) − c·benefit`. |
| **relationship** | Reward engaged power-users; penalise coupon-clippers (benefit ≫ spend). | Linear + `benefit_efficiency` penalty. |
| **dollar_pnl** | Literal Profit = Revenue − Cost − Risk in **dollars**; encodes the 5× rewards drag. | Raw-dollar P&L → percentile rank. |
| **rank_ensemble** (primary) | Rank-average: dollar_pnl (0.40) + revenue_first (0.25) + relationship (0.20) + full_pnl (0.15). | Weighted mean of member ranks. |

`risk_adjusted` is excluded from the ensemble: its multiplicative risk discount is the most
sensitive to the f11 risk-sign uncertainty (the single biggest ranking risk).

## 4. Validation protocol (honest, non-circular)

Because there is no label, we validate the *pipeline and the approach* on a **synthetic
dataset that mirrors the true schema** and carries a **hidden Amex-style dollar P&L** whose
coefficients **differ from every scorer's** (different margins, benefit unit-costs, LGD,
noise). Customers span affluent power-users, mass-affluent revolvers and loss-making
coupon-clippers, so the profit tail and the raw-spend tail genuinely differ. We then measure
the **exact competition metric** (top-20% overlap) of each framework against the hidden truth,
across multiple seeds.

> **We deliberately do NOT tune framework weights to this synthetic** — that would overfit
> our own assumptions. The synthetic confirms the method recovers a realistic
> interchange-dominated truth; the **public leaderboard** selects the final model.

## 5. Results (synthetic, top-20% overlap accuracy; mean over 5 seeds)

| Framework | Mean acc@20 | Stability (std) |
|---|---|---|
| **dollar_pnl** | **~0.87** | very stable |
| **rank_ensemble (primary)** | **~0.85** | very stable |
| pca_spend_pc1 (model-free) | ~0.83 | stable |
| relationship | ~0.81 | stable |
| revenue_first | ~0.80 | stable |
| full_pnl | ~0.78 | stable |
| risk_adjusted | ~0.73 | stable |
| random baseline | 0.20 | — |

Reading: a pure spend ranking already recovers ~0.80 of the true top-20% (interchange
dominance), and the structural **dollar P&L** + the **ensemble** add several points by
correctly netting the 5× rewards drag, benefit redemptions and tail credit loss — exactly
the levers that separate the profit tail from the spend tail.

## 6. How to choose the final submission

1. Run the pipeline on the real `data/train.csv`.
2. Read `outputs/diagnostics.json` — confirm f5=Σcategories, credit-line redundancy, and the
   risk-sign sanity check (if risk does *not* co-move with distress on the real data,
   down-weight or flip `risk` in `config.yaml`).
3. Upload `outputs/submission_dollar_pnl.csv`, `outputs/submission_rank_ensemble.csv`, and
   `outputs/submission_revenue_first.csv` (3 of your 10 submissions) and **keep the best
   public score**; iterate weights only if a variant clearly wins.
