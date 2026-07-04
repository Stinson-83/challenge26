# Real-Data Findings (500K run) & Submission Strategy

Analysis of the actual competition run (`outputs/run_summary.json`: 500,000 rows,
`is_real_data: true`). This supersedes the synthetic-validation guidance where they differ.

> **Note:** On the *modelling* recommendation, this file is superseded by
> `docs/technical_review.md` + `docs/rebuild_results.md` (the honest LB-validated 0.768 model and
> the rebuilt consensus). The diagnostics below (segments, dollar_pnl mis-scaling) remain valid.

## 1. Run health ✅
- 500,000 rows scored in ~63s; submission is one row per id, correct `id, profitability_score` format.
- Column roles auto-detected correctly (11 dollar, 1 ratio = f11 risk, 1 binary = f2, 1 rare-flag = f3, 3 score, 6 count).
- **Missing values are pervasive (the earlier "no missing values" claim is FALSE).** Per-column missing %:

  | Col | Miss % | Col | Miss % | Col | Miss % |
  |---|---|---|---|---|---|
  | f1 | 0.00 | f9 | 23.14 | f17 | 58.45 |
  | f2 | 0.00 | f10 | 23.14 | f18 | 61.89 |
  | f3 | 0.00 | f11 | 0.50 | f19 | 0.004 |
  | f4 | 51.45 | f12 | 5.00 | f20 | 0.02 |
  | f5 | 1.27 | f13 | 2.74 | f21 | 51.45 |
  | f6 | 23.14 | f14 | 2.74 | f22 | 18.93 |
  | f7 | 23.14 | f15 | 2.74 | f23 | 87.79 |
  | f8 | 23.14 | f16 | 2.74 | | |

  **97.17% of rows have ≥1 missing value; only 2.83% (14,153 rows) are complete.**
  `outputs/data_quality_report.json` reports `frac_missing 0.0` for every column ONLY because the
  pipeline zero-imputes *before* the quality check (e.g. it reports f7 mean 23,689 = the zero-filled
  mean vs the true complete-case 30,822), so that report understates real missingness.

## 2. Diagnostics (real-data reality checks)
| Check | Result | Verdict |
|---|---|---|
| **f5 vs Σ(categories)** | f5 mean **\$3,421** vs category sum **~\$37,000** (f7 "Other" alone \$23,690). Spearman(f5, Σf6…f10)=0.013; f5 < Σ-of-parts in 88% of rows; f5 fails the rewards-accrual check (Σcategories correlates +0.31/+0.24 with f4/f21, f5 only +0.03/+0.02). | **f5 is NOT the category total and is a weak spend signal** (public probe: f5-alone 0.337 ≈ random vs category-sum 0.668). **Use f6–f10 as the interchange base; exclude f5.** *(This corrects the earlier "keep f5 primary" conclusion, which rested on a circular spend-proxy correlation.)* |
| **Risk sign** | corr(f11, collection f3)=0.45, **Spearman** corr(f11, revolve f1)=0.58 (Pearson only ~0.20), corr(f11, spend f5)=−0.05 | Higher f11 = riskier. **Negative sign correct.** |
| **Credit lines** | f17/f18 Spearman 0.93 | Near-duplicate (as expected). |

## 3. 🔴 dollar_pnl mis-scales on the real data (the key fix)
`dollar_pnl` works in **raw dollars**, so it depends on true column magnitudes. At the real means:

| Term | Formula | ≈ \$/customer |
|---|---|---|
| interchange | 0.022·f5 | 75 |
| **interest** | 0.15·f1 | **370** |
| benefit cost | f14+f16+35·f13+15·f15 | 172 |
| rewards / risk / servicing | — | 110 / 49 / 31 |

Interest (revolve balance) is **~5×** interchange, and benefit cost exceeds interchange, so
`dollar_pnl` effectively ranks by **revolve + cost, not spend** — contrary to the intended
charge-card economics. Evidence: its top-20% overlaps only **24–29%** with every other
framework (rank-corr ≈ 0.13–0.16), i.e. near-orthogonal. Because it had **40% weight** in the
old primary ensemble, **~24%** of that submission's top-20% came *only* from this mis-scaled
model (old ensemble overlaps just **64%** with the robust consensus).

**Action taken:** removed `dollar_pnl` from the primary `rank_ensemble`
(`config.yaml`). The scale-robust (percentile-rank) frameworks — revenue_first, relationship,
full_pnl, risk_adjusted — agree **82–97%** and keep spend dominant. `dollar_pnl` stays
available as a standalone framework for A/B testing.

## 4. Segments (KMeans)
- **Profitable core (~32%):** high f5 spend, engaged, high credit line, low–moderate risk.
- **Coupon-clippers (~21%):** lowest f5 spend, highest benefit usage/efficiency → ranked least profitable.
- **Risky revolvers:** high revolve + highest risk (0.88) + collection (0.78).
- **Churners:** highest attrition/cancellation.

## 5. Recommended leaderboard strategy (10 submissions)
The decisive unknown: does Amex's true profit track **spend** or **revolve/lending + cost**?
Test it, don't guess:

1. **`outputs/submission_group_consensus.csv`** — robust spend-dominant ensemble (dollar_pnl
   removed). **Recommended primary.**
2. **`outputs/submission_revenue_first.csv`** — pure spend baseline.
3. **`outputs/submission_rank_ensemble.csv`** — old ensemble (measures whether dollar_pnl helps).
4. **`outputs/submission_dollar_pnl.csv`** — standalone (directly tests the revolve/lending hypothesis).

Keep whichever scores best on the public 70%. If group_consensus ≫ dollar_pnl (expected),
spend economics win; if dollar_pnl wins, rebuild deliberately around f1 + lend-lines rather
than by accidental mis-scaling.
