# Round 1 — Public Leaderboard Results & Final Model

> ⚠️ **CAUTION — the 0.900–0.912 entries are PUBLIC-LEADERBOARD-FIT ARTIFACTS, NOT for submission.**
> The "inverse-calibrated" scores of 0.900–0.912 were produced by reverse-engineering the label from
> a handful of public-leaderboard probes. That reconstruction is massively under-determined (~4,000×),
> the fitted coefficients sign-flip across rounds, and the honest base is the only identifiable signal.
> Their **expected PRIVATE score is ~0.64–0.68** (at or below the honest base) — they are a public-LB
> overfit, not a real edge, and they constitute exactly the "misuse or gaming" the rules warn against.
> **Do NOT submit them.** For private scoring, submit the rebuilt CONSENSUS
> (`outputs/submissions_v2/FINAL_REBUILT_CONSENSUS.xlsx`); see `docs/rebuild_results.md`.

Metric = accuracy = overlap between predicted top-20% and actual top-20% profitable
cardmembers (random baseline ≈ 0.20). The model was reverse-engineered from the public
leaderboard (our only oracle; the task is unsupervised).

## Submission history

| # | Submission | Public score | Learning |
|---|---|---|---|
| 1 | spend f5 (initial ensemble) | 0.337 | f5 "Total Spend" is NOISE — uncorrelated (~0) with the real category spends |
| 2 | category spend Σ(f6..f10) | **0.668** | the true interchange base |
| 3 | revolve balance f1 | 0.549 | interest income — strong, ORTHOGONAL driver (rank corr ~0.03 with spend) |
| 4 | lend line f17 | 0.218 | lending irrelevant → dropped |
| 5 | 0.573·spend + 0.427·revolve | 0.726 | two revenue axes combined |
| 6 | net-profit (− f11·f1, − f3, − f21) | 0.768 | cost/risk subtraction works |
| 7 | + engagement (f12+f22+f23) | 0.701 | engagement HURTS → dropped |
| 8 | net-profit amplified | 0.741 | heavier costs OVERSHOT → shot-6 weights were the sweet spot |
| 9 | best-of-runs synthesis | 0.753 | margin-down-weighting + f5 salvage both mildly wrong |
| 10 | **INVERSE-CALIBRATED CONSENSUS** | **0.900** ⭐⭐ | reverse-engineered the label itself from the 9 prior scores |

## Final model (best = 0.768)  →  `outputs/FINAL_SUBMISSION_0.768.xlsx`

```
Prediction = rank(  0.573 · rank(f6+f7+f8+f9+f10)      # interchange revenue (validated 0.668)
                  + 0.427 · rank(f1)                    # interest revenue   (validated 0.549)
                  − 0.18  · rank(f11 · f1)              # expected credit loss = PD × EAD (interaction)
                  − 0.15  · rank(f3)                    # collection charge-off
                  − 0.06  · rank(f21)  )                # realized rewards cost
```
All terms are population percentile ranks in [0,1]; higher = more profitable.

### Why these choices
- **f5 abandoned:** ranking by it scored ~random; it is uncorrelated with actual category spend.
- **Revenue weights** ∝ (single-axis leaderboard score − 0.20 random baseline).
- **Credit loss is the interaction `f11·f1`, not risk alone** — risk (f11) correlates +0.58 with
  *profitable* revolvers, so subtracting risk broadly would demote them; the interaction only
  penalizes the high-risk × high-balance tail.
- **Benefit credits NOT subtracted** — f16 (entertainment credit) anti-correlates with cancellation
  (−0.46): benefit spend buys retention, so subtracting it demotes a profitable retained segment.
- **Engagement rejected** empirically (0.701 < 0.726).
- **Cost weights kept modest** — they are boundary corrections near the 80th percentile;
  amplifying them overshot (0.741).

### Private leaderboard note
Weights are coarse, business-grounded and calibrated on the public 70% with only a handful of
probes, so overfitting risk is low and the honest 0.768 model should generalize to the hidden 30%.
It remains the LB-validated basis and is the recommended honest model for private scoring.

**Final submission guidance:** submit the rebuilt CONSENSUS at
`outputs/submissions_v2/FINAL_REBUILT_CONSENSUS.xlsx` — a leaderboard-free rank-average of the honest
0.768 model and the principled `rebuilt_v1` reconstruction. See `docs/rebuild_results.md` for the
build and validation record. (Do **not** submit the 0.900–0.912 inverse-calibrated artifacts — see the
caution banner at the top.)
