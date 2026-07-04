# `src.robustness` — neutral, leaderboard-free robustness harness

Built first (roadmap **R6** in `docs/technical_review.md`) so that every later modeling change is
scored by a neutral judge **before** it touches the frameworks. Deliberately **decoupled** from
`src/profitability.py` (the code we intend to rewrite) — it re-implements scoring self-contained and
reuses only the exact competition metric from `src/evaluation.py`.

## Run

```bash
python -m src.robustness.harness            # 200k subsample (~2 min) — default
python -m src.robustness.harness --full     # all 500k rows
python -m src.robustness.harness --sample 100000 --out outputs/robustness
```

Outputs → `outputs/robustness/`:
- `scorecard.csv` — one row per ranker, sorted by **worst-case recovery across the truth panel**.
- `robustness_report.json` — full per-truth / per-feature / per-pair detail + construct-validity oracles.

## What each number means — and the two traps this harness refuses to fall into

| Column | Meaning | How to read it |
|---|---|---|
| `truth_mean` | mean top-20% recovery across the 6 synthetic economies (× seeds) | higher = recovers plausible economics on average |
| **`truth_worst`** | **min recovery over the panel** | **the headline: robustness to *which* economics Amex actually scores** |
| `weight_robust` / `weight_min` | top-20% overlap under lognormal(0,0.3) weight jitter | fragility of the *weight operating point* (a tuned split scores low here) |
| `imput_med_vs_zero` | overlap between median- and zero-imputation | dependence on how missing cells are filled |
| `noise_robust` | overlap under 10% multiplicative feature noise | measurement-noise robustness |
| `top_driver` | feature whose removal most changes the top-20% (LOFO) | the ranker's true load-bearing signal |
| `bootstrap` | 70%-subset top-20% stability | **≈1.0 BY CONSTRUCTION — a sanity check, not a transfer proof** |

**Trap 1 — bootstrap stability is not transfer.** Every ranker here is a deterministic per-row score,
so a 70% row-subsample reproduces the top-20% almost exactly. This says nothing about whether the ranker
matches the *hidden label* on the private 30%. We report it, flagged, and never rank on it.

**Trap 2 — cross-agreement is a decoy.** The shipped `*_cfg` frameworks agree 82–97% only because they
share the `f5`-as-spend misspecification. Maximising agreement would select the *wrong* model. The
`agreement` block in the JSON is reported for transparency, **never optimized**. Robustness is measured on
**validity-screened** signals via worst-case truth recovery.

## The truth panel (`truths.py`)

Not a claim about the real label — an **economic-uncertainty stress test**. Six *structurally different*
profit economics built from the real features (realistic marginals/missingness/censoring): `spend_led`,
`revolve_led`, `relationship_led`, `threshold_tier`, `interaction`, and an adversarial `f5_led` (so we
notice if a model would fail on an f5-shaped label — keeps us honest, not merely anti-f5). A model that
scores well **worst-case** across all six is robust to our uncertainty about Amex's true economics. This
replaces the repo's original **circular** synthetic validation (whose hidden truth shared `dollar_pnl`'s
exact form, so `dollar_pnl` "won" by construction yet was near-orthogonal on real data).

## Construct-validity oracles (`diagnostics.py`) — feature justification without the leaderboard

Label-free checks that let the rebuilt model justify its feature choices from first principles:
censoring/cap detection, total-vs-parts sanity (rejects `f5`-as-total), near-duplicate redundancy
(`f17/f18`), structural co-missing blocks (segment indicators), and the **rewards-accrual** oracle
(real spend co-moves with points earned/redeemed; `f5` does not → prefer the category-sum).

## Extending it

Register a rebuilt framework and it is scored automatically:

```python
from src.robustness.rankers import Ranker, register, R, _c
register(Ranker(
    name="rebuilt_v1",
    build=lambda d: {"cats": ..., "f1": R(_c(d, "f1")), "EL": R(_c(d,"f11")*_c(d,"f1"))},
    weights={"cats": 0.55, "f1": 0.45, "EL": -0.18},
    features=["f6","f7","f8","f9","f10","f1","f11"],
))
```
Then re-run the harness and compare `truth_worst` against `honest_0768` (the incumbent to beat).
