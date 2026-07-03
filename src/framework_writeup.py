"""
framework_writeup.py
====================
Responses for the "Profitability Framework" sheet of the Unstop submission template.
Each key matches a Section label in the template; the value is written into the Response
column. Text is kept concise and self-contained so it reads well inside a spreadsheet cell.

Edit these strings to match whichever prediction column you submit — the defaults describe
the recommended ROBUST rank-ensemble of the scale-immune frameworks (revenue_first,
relationship, full_pnl, risk_adjusted), i.e. the `group_consensus` predictions.
"""

FRAMEWORK_RESPONSES = {
    "Variables Used":
        "All 23 attributes are used (id excluded as identifier). "
        "REVENUE: f5 Total Spend (primary), f6 Airlines, f7 Other, f8 Entertainment, "
        "f9 Lodging, f10 Dining (category spends), f1 Avg Revolve Balance (net interest). "
        "RELATIONSHIP/CAPACITY: f17 Total Lend Line, f18 Consumer Lend Line, "
        "f19 Supplementary Accounts, f20 Active Charge Cards. "
        "ENGAGEMENT: f12 Website Logins, f22 Emails Opened, f23 Emails Clicked, "
        "f4 Points Balance, f21 Points Redeemed. "
        "RISK: f11 Risk Score, f2 Cancellation Calls, f3 Cancellation Calls due to Collection. "
        "COST (benefit redemptions): f13 Lounge Access, f14 Airline Credits $, "
        "f15 Cab Benefit Usage, f16 Entertainment Credit $.",

    "Profitability Equation":
        "Profit(to issuer) = Revenue - Cost - Expected Risk Loss, implemented as a robust, "
        "scale-free percentile-rank model. Every signal is converted to a percentile rank in "
        "[0,1] (higher = more profitable) and combined as a signed weighted sum per framework:\n"
        "Score = 1.00*Spend(f5) + 0.15*CategorySpend(f6..f10) + 0.18*Revolve(f1) "
        "+ 0.05*CreditLine(f17,f18) + 0.06*Engagement(f12,f22,f23) "
        "+ 0.08*Relationship(f19,f20) + 0.05*Points(f4,f21) "
        "- 0.12*BenefitCost(f13,f14,f15,f16) - 0.18*Risk(f11) "
        "- 0.10*Attrition(f2) - 0.15*Collection(f3).\n"
        "The final Prediction is the rank-average of four such frameworks "
        "(revenue-first, full P&L, risk-adjusted, relationship/coupon-clipper-aware), "
        "which stabilises the ranking at the top-20% decision boundary.",

    "Prediction Logic":
        "The Prediction is each cardmember's profitability percentile score in [0,1]; higher = "
        "more profitable. Ranking cardmembers in descending order of Prediction and taking the "
        "top 20% identifies the most profitable cardmembers, matching the evaluation metric "
        "(overlap of predicted vs actual top-20%). Only the ordering matters, not the absolute value.",

    "Variable Selection Logic":
        "Each masked variable was mapped to its economic role (Revenue / Cost / Risk / "
        "Relationship / Engagement) using the provided feature descriptions. We included every "
        "variable that plausibly drives issuer profit and excluded only the identifier (id). "
        "Spend (f5) is kept dominant because interchange on spend is the primary profit lever of "
        "a charge card; cost/risk variables enter with small weights so they re-rank the tail "
        "without overturning the spend ordering. Ambiguous or redundant signals (e.g. f18 is a "
        "near-duplicate of f17) are down-weighted or grouped to avoid double-counting.",

    "Coefficient/Weight Derivation":
        "This is an UNSUPERVISED problem (no label), so weights are BUSINESS PRIORS from Amex "
        "unit economics, not fitted to a target. Rationale: ~2% interchange on tens of thousands "
        "of dollars of spend dwarfs the ~fixed ~$600 annual fee and a few hundred dollars of "
        "credits, so Spend gets weight 1.0 and every other term is a small adjustment "
        "(|non-spend weight| kept well below the spend weight). Signs come from economics "
        "(revenue +, cost/risk -). Magnitudes were sanity-checked on (a) a synthetic dataset "
        "with a hidden Amex-style dollar P&L using DIFFERENT coefficients, and (b) real-data "
        "diagnostics (see Validation Approach).",

    "Feature Transformations":
        "1) Role-aware missing-value handling: dollar/count features -> 0 (absence of activity), "
        "ratio/score features -> median. 2) Each signal -> robust percentile rank in [0,1] "
        "(rank-based, immune to scale, skew and outliers such as points balances up to ~700k). "
        "3) Monetary buckets are summed in RAW dollars then ranked (preserving total-volume "
        "ordering = interchange). 4) Light 99.5th-percentile winsorization to tame extreme "
        "single-column outliers. These transforms make the ranking stable exactly at the "
        "80th-percentile cut that the metric scores.",

    "Business Logic":
        "Amex earns mainly interchange (~2% of spend) plus net interest on revolving balances "
        "plus a roughly-fixed annual fee; it pays rewards and benefit redemptions (lounge visits, "
        "airline/cab/entertainment credits) and servicing, and loses money to credit risk and "
        "attrition. The most profitable Premier cardmember therefore spends heavily, is engaged "
        "and has a deep relationship (multiple cards/supplementary accounts), carries low risk, "
        "and does NOT over-consume costly benefits relative to spend. The equation rewards spend, "
        "revolve (net of risk), engagement and relationship, and penalises benefit cost, risk "
        "score, collection distress and cancellation intent - which separates profitable "
        "power-users from unprofitable 'coupon-clippers'.",

    "Assumptions":
        "1) f5 Total Spend is the primary revenue proxy (validated on real data as a clean, "
        "self-consistent signal; category columns are on a different scale and weakly related, so "
        "used only as a minor corroborator). 2) Higher f11 Risk Score = riskier (validated: it "
        "co-moves with collection calls and revolve balance). 3) The annual fee is ~constant "
        "across members and drops out of a ranking. 4) Interchange dominates charge-card profit. "
        "5) Benefit credits/lounge visits are real per-use issuer costs. 6) With no label, "
        "robustness of the ranking is prioritised over point precision.",

    "Validation Approach":
        "Because the task is unsupervised, we validated the pipeline and equation on a SYNTHETIC "
        "dataset that mirrors the real schema and carries a HIDDEN Amex-style dollar P&L (with "
        "coefficients deliberately different from the scorer's). Measured with the EXACT "
        "competition metric (top-20% overlap) across multiple seeds, the robust rank-ensemble "
        "scored ~0.85 and was the most stable. On the real 500k data we ran diagnostics: "
        "total-spend consistency, credit-line redundancy, and a risk-sign sanity check "
        "(f11 co-moves with distress - confirming the sign). Framework-agreement analysis shows "
        "the spend-dominant frameworks agree 82-97% on the top-20%, so the final Prediction is "
        "their rank-average (a stable consensus).",

    "Additional Notes (Optional)":
        "We built and compared multiple independent frameworks (revenue-first, full P&L, "
        "risk-adjusted, relationship/coupon-clipper-aware, and a literal dollar P&L). The "
        "submitted score is the rank-ensemble of the scale-immune frameworks. A raw-dollar P&L "
        "variant was de-emphasised for the final submission because, on the real column scales, "
        "its interest term swamped interchange and it began ranking by revolve balance rather "
        "than spend. The identifier (id) is never used in the equation; all 500,000 unique ids "
        "are scored; no rows were added, removed or altered.",
}
