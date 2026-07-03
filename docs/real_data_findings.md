# Real-Data Findings (500K run) & Submission Strategy

Analysis of the actual competition run (`outputs/run_summary.json`: 500,000 rows,
`is_real_data: true`). This supersedes the synthetic-validation guidance where they differ.

## 1. Run health ✅
- 500,000 rows scored in ~63s; submission is one row per id, correct `id, profitability_score` format.
- Column roles auto-detected correctly (11 dollar, 1 ratio = f11 risk, 1 binary = f2, 1 rare-flag = f3, 3 score, 6 count). **No missing values** in the real data.

## 2. Diagnostics (real-data reality checks)
| Check | Result | Verdict |
|---|---|---|
| **f5 vs Σ(categories)** | f5 mean **\$3,421** vs category sum **~\$37,000** (f7 "Other" alone \$23,690). Near-uncorrelated (corr w/ spend-proxy: f5=0.74, categories ≈0.00–0.03); load oppositely on PC1. | f5 is a **clean, self-consistent spend signal**; f6–f10 are a differently-scaled, weakly-related set. Keep f5 primary, categories a minor corroborator (as configured). |
| **Risk sign** | corr(f11, collection f3)=0.45, corr(f11, revolve f1)=0.58, corr(f11, spend f5)=−0.05 | Higher f11 = riskier. **Negative sign correct.** |
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
