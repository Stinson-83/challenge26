git # Rebuild Results — data-motivated model + principled weights (roadmap R1–R7)

**Scored by the neutral harness (`src/robustness/`), leaderboard-free.** Reproduce with
`python -m src.robustness.rebuild`. Companion to `docs/technical_review.md`.

## What changed vs the honest 0.768 base (all data-justified, no leaderboard fitting)

| Fix | From | To | Data-internal reason |
|---|---|---|---|
| Category spend (R2) | raw-sum `f6..f10` (≈ `f7` alone, ρ 0.93) | **rank-mean of R(f6..f10)** | raw-sum collapses to one heavy-tailed, refund-bearing column; rank-mean uses all five fairly |
| Collection distress (R3) | `R(f3)` (89% zeros pinned at 0.45 → muted) | **raw binary `f3` flag** | `f3` is binary and near-disjoint from `f2`; a clean flag actually bites the 11% flagged |
| Risk (R3) | broad — | **`−rank(f11·f1)`** kept | PD×EAD interaction; `f11` alone correlates +0.58 with profitable revolvers |
| Rewards term | `−0.06·R(f21)` | **dropped** | `pts=f4+f21` is incoherent (ρ −0.036); `f21` is 51% missing → quasi-degenerate rank |
| Engagement | (config had it) | **excluded** | orthogonal noise here; public probe showed engagement *hurts* |

## Weight derivation (R4) — three principled methods + a reconciled choice

Investigated (leaderboard-free, on the diverse economy panel): **A** equal prior, **B** axis-strength
(∝ single-axis recovery − 0.20), **C** minimax over the panel. Each validated with **leave-one-truth-out
CV**. Key finding: pure worst-case maximization (B/C) pushes `cats`→0.79, but that is an **artifact of the
panel leaning spend-forward** — it collapses the revolve-economics hypothesis (`revolve_led` recovery
0.70→0.40) and the `f5_led` guardrail. So the split was chosen by **method D**: maximize *mean* recovery
across the panel subject to revolve/f5 floors — which lands at **cats 0.60 / f1 0.40**, in a flat
weight-robust region, and **corroborated by the real single-axis leaderboard ratio 0.573/0.427**.

**`rebuilt_v1` = `rank( 0.60·catavg + 0.40·R(f1) − 0.18·R(f11·f1) − 0.12·f3 )`**

## Head-to-head (200K, worst-economy recovery across the 6-truth panel)

| ranker | truth_worst | truth_mean | weight_min | noise_robust | note |
|---|---|---|---|---|---|
| honest_0768 | 0.315 | 0.528 | 0.71 | 0.966 | incumbent (raw-sum cats + f21) |
| **consensus_50_50** | **0.319** | **0.569** | — | — | rank-avg(honest_0768, rebuilt_v1) — best worst **and** mean |
| **rebuilt_v1** | 0.309 | 0.511 | 0.62 | 0.981 | cleaner single model (structural fixes) |
| rebuilt_A/B/C | 0.26–0.29 | — | 0.45–0.82 | — | alternative weightings (rejected) |
| cats_only | 0.254 | 0.515 | — | — | single axis |
| risk_adjusted_cfg | 0.246 | 0.360 | — | — | **shipped "primary" config (f5)** |
| full_pnl_cfg | 0.240 | 0.373 | — | — | shipped config (f5) |
| revenue_first_cfg | 0.233 | 0.365 | — | — | shipped config (f5) |
| dollar_pnl | 0.200 | 0.333 | — | — | mis-scaled |

## Honest conclusion

1. **The f5-primary configs the repo ships as "primary" are decisively worse** (worst 0.23–0.25 vs
   ~0.31–0.32 for the honest/rebuilt family) — the headline audit finding, now confirmed leaderboard-free.
2. **`rebuilt_v1`, `honest_0768`, and their consensus are statistically TIED on the synthetic panel**
   (worst 0.31–0.32, within imputation noise). The rebuild did **not** produce a large worst-case jump —
   and it **should not be claimed to**. On real data `rebuilt_v1` moves ~20% of the top-20% vs
   `honest_0768` (overlap 0.80, ρ 0.91), driven by the scale-fair category fix; whether those swaps help
   the *true* label is unknowable without the leaderboard.
3. **What the rebuild genuinely delivers:** a **cleaner, more defensible, equally-robust** model with
   **principled, leaderboard-free weights** — it removes the incoherent `f21` term, the silent `≈f7`
   collapse, and the muted `f3`, at no robustness cost, with better noise robustness (0.981 vs 0.966).
4. **The consensus marginally dominates** (best worst *and* mean) and is the best-justified submission: it
   keeps the ~80% both defensible models agree on and **hedges the unvalidated 20% boundary disagreement** —
   the right posture when the true label is hidden. This is legitimate rank aggregation over two
   *independent, both-validated* models (unlike the config "ensemble" of four `f5` clones).
5. **This tie is itself an important result:** you cannot reliably push a top-20%-overlap model far past
   the honest base **without** leaderboard feedback — which is exactly why the inverse-calibration's
   claimed 0.90+ was a public-fit artifact, not a private edge. The disciplined ceiling here is ~0.31
   synthetic / ~0.77 real, honestly earned.

## Recommendation for the private leaderboard

Rank by expected-private + defensibility:
1. **`consensus_50_50`** (primary) — best panel robustness, hedges unvalidated swaps, inherits ~0.9 of the
   LB-validated set. Fully leaderboard-free and business-interpretable.
2. **`rebuilt_v1`** — the clean single model; submit as an A/B alternative.
3. **`honest_0768`** — the LB-anchored fallback (only model with a real public score, 0.768).

Do **not** submit any inverse-calibration file. Next step: a clean submission producer that emits these
three for all 500K (CSV + Unstop `.xlsx`) with a truthful methodology write-up.
