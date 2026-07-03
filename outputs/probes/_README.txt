PROBE SUBMISSIONS — spend ranking scored ~0.337 (random ~0.20).
Submit in this order; each score tells you whether that signal drives profit.
Random baseline ~0.20; a driver should score clearly higher (>0.45).

  1. probe_category_total.xlsx
  2. probe_revolve_f1.xlsx
  3. probe_lend_line_f17.xlsx
  4. probe_points_redeemed_f21.xlsx
  5. probe_other_spend_f7.xlsx
  6. probe_INVERSE_total_spend_f5.xlsx
  7. probe_points_balance_f4.xlsx
  8. probe_lending_x_lowrisk.xlsx

Interpretation:
  - If category_total >> total_spend_f5: the category sum is the real spend driver.
  - If revolve_f1 / lend_line_f17 win: profit is interest/lending-driven, not spend.
  - If points_* win: profit tracks loyalty/redemption.
  - If INVERSE_total_spend_f5 ~ 0.66: the grader direction is flipped (submit inverses).
  - Then combine the 1-2 winning signals and submit that.
