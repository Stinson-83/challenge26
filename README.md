# Amex Campus Challenge 2026 — Round 1
## Premier Card Customer-Profitability Framework

A modular, research-grade pipeline that rank-orders Premier cardmembers by estimated
**profitability-to-issuer** (`Profit = Revenue − Cost − Expected Risk Loss`) in a fully
unsupervised setting, optimised for the competition's **top-20% overlap accuracy** metric.

> **Metric:** accuracy = % overlap between the actual top-20% most-profitable cardmembers and
> our predicted top-20%. Only the ranking near/above the 80th percentile matters.

**Status:** The defensible deliverable is the **CONSENSUS of two data-motivated models**, validated
**leaderboard-free** by the neutral robustness harness in `src/robustness/`. The two members are
`honest_0768` — the LB-validated **0.768** top-20% overlap baseline
(`rank(0.573·R(cats) + 0.427·R(f1) − 0.18·R(f11·f1) − 0.15·R(f3) − 0.06·R(f21))`, verified this
session to reproduce `FINAL_SUBMISSION_0.768.xlsx` bit-exactly) — and `rebuilt_v1`
(`rank(0.60·catavg + 0.40·R(f1) − 0.18·R(f11·f1) − 0.12·f3)`, using scale-fair category spend,
disjoint `f3`, and principled leaderboard-free weights). **Recommended submission:**
**`outputs/submissions_v2/FINAL_REBUILT_CONSENSUS.xlsx`** (the rank-average of the two).

> **Quarantined — do NOT submit.** The inverse-calibration files (`FINAL_INVERSE_FIT` and the
> R2–R5 rounds, `docs/inverse_calibration.md`) are **public-leaderboard overfitting, not a private-LB
> edge**: reconstructing a 100k-of-500k top-20% set from ~10 public scores is under-determined by
> ~4,000×, the fitted coefficients sign-flip across rounds, expected private score is ~0.64–0.68 (at
> or below the honest base, not 0.90+), the optimizer code is absent (unreproducible), and the rounds
> imply 13–14 graded submissions against the 10-cap. This is exactly the "misuse or gaming" the rules
> warn against — see the banner atop `docs/inverse_calibration.md`.

Full record: **`docs/technical_review.md`** and **`docs/rebuild_results.md`**;
harness: **`src/robustness/README.md`**.

---

## Quickstart

```bash
pip install -r requirements.txt          # numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, pyyaml, openpyxl

# 1) Put the real Unstop file at data/train.csv  (columns: id, f1..f23)
#    If absent, the pipeline runs on a faithful synthetic dataset with a hidden
#    ground-truth profit, so you can validate the whole thing end-to-end.

# 2) Run everything: EDA, diagnostics, features, all frameworks, clustering, PCA,
#    outliers, plots, CSV submissions, and the filled Unstop .xlsx.
python -m src.pipeline

# Switch the primary framework (drives submission.csv and the filled .xlsx):
python -m src.pipeline --framework revenue_first   # revenue_first | relationship | full_pnl | risk_adjusted | dollar_pnl | rank_ensemble
```

Outputs land in `outputs/`:
- **`campus_challenge_r1_submission_FILLED.xlsx`** — the official deliverable: `Predictions`
  sheet (all 500k ids) + `Profitability Framework` methodology write-up. *(real-data runs only)*
- `submission.csv` — primary submission (`id, profitability_score`), one row per id.
- `submission_<framework>.csv` — one per framework; **submit several, keep the best public score**.
- `submission_group_consensus.csv` — the recommended robust ensemble (see below).
- `data_quality_report.json`, `diagnostics.json`, `framework_agreement.csv`,
  `kmeans_cluster_profile.csv`, `pca_loadings.csv`, `run_summary.json`.
- 8 PNG visualisations (correlation, distributions, PCA, clusters, outliers, score
  distributions, framework overlap, cluster profitability).

---

## What makes this a leaderboard solution (not a baseline)

- **Robust, rank-based scoring.** Every signal → a percentile rank in `[0,1]`, immune to the
  extreme skew/outliers in the real data (e.g. points balances up to ~700k). Weighted sums of
  comparable ranks are stable exactly at the 80th-percentile cut the metric scores.
- **Spend-dominant economics.** Interchange on `f5` Total Spend dwarfs the ~fixed fee and a
  few hundred dollars of credits, so spend carries the ranking (weight 1.0) and every cost/risk
  term is a small, business-signed adjustment that can't overturn a full spend decile.
- **Six independent frameworks + a rank ensemble**, so you can switch scoring philosophies and
  A/B them on the public leaderboard.
- **Honest validation.** Because the task is unsupervised, the pipeline is validated on a
  synthetic dataset carrying a *hidden* Amex-style dollar P&L (coefficients differ from the
  scorers), measured with the *exact* competition metric across seeds (robust ensemble ≈ 0.85).
- **Real-data diagnostics that changed the model.** On the real 500k the diagnostics revealed
  that the raw-dollar `dollar_pnl` framework is **mis-scaled** — its interest term (`0.15·f1`)
  swamps interchange (`0.022·f5`), so it ranks by revolve balance, not spend, and its top-20%
  overlaps only ~25% with every other framework. The **primary was therefore switched to the
  robust `group_consensus`** (rank-average of the four scale-immune frameworks). See
  **`docs/real_data_findings.md`**.

See also **`docs/business_analysis.md`** (business reasoning, unit economics, feature→economics
map, the profitability math) and **`docs/methodology.md`** (pipeline, frameworks, validation).

---

## Real-data findings (500k run) — headline

| Check (`outputs/diagnostics.json`) | Result | Action |
|---|---|---|
| `f5` (Total Spend) vs Σ(f6..f10) | f5 ≈ 9% of the category sum; near-uncorrelated | keep `f5` as the clean primary spend signal; categories = minor corroborator |
| risk sign (`f11`) | co-moves with collection (0.45) & revolve (0.58) | higher `f11` = riskier confirmed (negative weight) |
| credit lines (`f17`/`f18`) | Spearman 0.93 | near-duplicates, grouped |
| `dollar_pnl` vs other frameworks | top-20% overlap 24–29% (outlier) | **dropped from the primary ensemble** |

**Segments (KMeans):** a profitable core (high spend, engaged, low–moderate risk), a
coupon-clipper cluster (~21%: lowest spend, highest benefit extraction → least profitable),
risky revolvers (high risk + collection), and churners (high cancellation).

---

## Submitting to Unstop

The submission is an `.xlsx` workbook with two sheets — a `Predictions` sheet and a
`Profitability Framework` write-up. The pipeline fills both automatically:

- Predictions ← the active framework's score for every id (order-independent ID matching).
- Framework sheet ← the responses in `src/framework_writeup.py` (edit these if you submit a
  different prediction column so your methodology and predictions stay consistent).

Recommended play with your 10 submissions: submit the **rebuilt consensus**
(`outputs/submissions_v2/FINAL_REBUILT_CONSENSUS.xlsx`), the leaderboard-free deliverable validated
by `src/robustness/`; `honest_0768` and `rebuilt_v1` are available as its individual members. Do
**not** submit the quarantined inverse-calibration files (`FINAL_INVERSE_FIT` / R2–R5).

---

## Project layout

```
challenge26/
├── config.yaml               # feature semantics, group map, framework weights, submission config
├── requirements.txt
├── data/
│   ├── feature_description.csv   # authoritative feature meanings (provided)
│   └── train.csv                 # <-- drop the real Unstop file here (git-ignored)
├── templates/
│   └── submission_template.xlsx  # official Unstop template
├── src/
│   ├── config.py  data_io.py  preprocessing.py  eda.py  diagnostics.py
│   ├── feature_engineering.py  profitability.py  scoring.py
│   ├── clustering.py  dimensionality.py  outliers.py  evaluation.py
│   ├── synthetic.py  submission.py  framework_writeup.py  pipeline.py
├── notebooks/01_exploration.py   # runnable exploration walk-through
├── tests/test_pipeline.py
├── docs/business_analysis.md  docs/methodology.md  docs/real_data_findings.md
└── outputs/                      # generated artefacts + submissions
```

## Frameworks

| Framework | Idea |
|---|---|
| `revenue_first` | Spend (f5) dominates; everything else a light touch. |
| `full_pnl` | Balanced Revenue − Cost − Risk on robust group scores. |
| `risk_adjusted` | Revenue discounted multiplicatively by risk / attrition / collection. |
| `relationship` | Rewards engaged power-users, penalises coupon-clippers. |
| `dollar_pnl` | Literal `$` P&L (kept for A/B; mis-scaled on real data — excluded from the primary). |
| `rank_ensemble` **(primary)** | Rank-average of the four scale-immune frameworks (= `group_consensus`). |

> **Note:** "`f5` = Total Spend" is **refuted by the real data** (Spearman(f5, Σf6..f10)=0.013;
> f5 is ~1/12 the scale of the category sum and fails the rewards-accrual check). The recommended
> `rebuilt_v1`/consensus therefore ranks on **category spend** (`Σ/avg f6..f10`), not `f5`.

## Rules compliance
- The `id` (identifier) is **never used** inside any equation — only carried to the output.
- The solution runs on **all** 500,000 unique identifiers; **no rows are added, removed, or altered**.
- Predictions and the methodology write-up are emitted directly into the official Unstop
  `.xlsx` template (`submission.template_xlsx` in `config.yaml`).
