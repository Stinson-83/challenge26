"""
src.robustness — a neutral, leaderboard-free robustness harness.
================================================================
Built as the FIRST step of the generalization-first rebuild (roadmap R6 in
docs/technical_review.md). Its job is to be the *judge* every later modeling change is
scored against, BEFORE any framework is rewritten — so it is deliberately DECOUPLED from
src/profitability.py (which we intend to change).

What it provides
----------------
- `rankers`      : a registry of self-contained candidate ranking functions (raw df -> score),
                   each expressed so weights can be perturbed and features ablated generically.
- `truths`       : a library of STRUCTURALLY-DIFFERENT synthetic hidden-profit generators over
                   the REAL features. Not a claim about the true label — an economic-uncertainty
                   stress test that breaks the circular "truth == one of the scorers" mistake.
- `diagnostics`  : label-free construct-validity oracles (censoring, total-vs-parts, redundancy,
                   block co-missingness, rewards-accrual) that justify feature choices WITHOUT
                   the leaderboard.
- `harness`      : runs everything and emits a scorecard to outputs/robustness/.

Interpretation guardrails (see README.md) — the two traps this harness is designed to avoid:
1. Bootstrap top-20% stability of a deterministic per-row score is ~1.0 BY CONSTRUCTION and is
   NOT evidence of public->private transfer. We report it, flagged, as a sanity check only.
2. Internal cross-framework AGREEMENT is a DECOY: the config frameworks agree 82-97% only because
   they share the f5 misspecification. Robustness must be measured on validity-screened signals,
   and the headline generalization metric here is WORST-CASE recovery across the diverse truth
   library, not agreement.
"""
from __future__ import annotations

from . import diagnostics, rankers, truths  # noqa: F401

__all__ = ["rankers", "truths", "diagnostics", "harness"]
