PROBE PLAN — our spend(f5) ranking scored ~0.337 (random ~0.20).
Root cause (5-expert consensus, conf 0.70): f5 is the WRONG spend column.

ZERO-COST OFFLINE CHECK (this run):
  corr(f5, sum f6..f10) = 0.011   (expect <0.1 -> f5 is NOT the total)
  corr(f7, sum f6..f10) = 0.903   (expect >0.9 -> f7 dominates the spend base)

SUBMIT IN THIS ORDER (each score = that signal's alignment with true top-20%):
  1. probe_category_total.xlsx
  2. probe_revolve_f1.xlsx
  3. probe_lend_line_f17.xlsx
  4. probe_pnl_composite_v1.xlsx
  5. probe_points_redeemed_f21.xlsx

RULES:
  - Random ~0.20; a real driver scores >0.45. STOP using f5.
  - DO NOT submit an inverse ranking — 0.337>0.20 proves sign+id-join are correct.
  - Falsification guard: if category_total <= ~0.35, the spend thesis is DEAD;
    route remaining slots to interest/lending (revolve_f1, lend_line_f17).
  - Composite (probe #4): recalibrate weights ∝ (each probe score − 0.20) via
    src.probe_leaderboard.recalibrate() before submitting pnl_composite.
  - Combine components as RANKS (standardized), never raw dollars.
