# Business Analysis — Amex Campus Challenge 2026, Round 1
### Measuring Customer Profitability for the Premier Card

This document is the **pre-code reasoning** behind the solution. It explains the business
problem, Amex's unit economics on a Premier cardmember (CM), what makes a CM profitable,
how each of the 23 attributes maps to that economics, and the mathematical profitability
framework we then implement.

---

## 1. The business problem

Amex wants a **framework/equation that quantifies each Premier cardmember's profitability
to the issuer** (Revenue − Cost − Risk), so the most profitable CMs can be identified and
prioritised. We are given ~23 masked cardmember attributes for **500K** members and **no
label**. We must produce a **rank ordering**.

**Evaluation is decisive and shapes everything:** *Accuracy = the percentage overlap between
the actual top-20% most-profitable CMs and our predicted top-20%.* Formally this is
`recall@20% = precision@20% = |pred_top20 ∩ actual_top20| / |top20|`. Two consequences:

1. **Only the ordering near and above the 80th percentile matters.** Calibrated dollar
   values are irrelevant; a *stable ranking at the cut* is everything. Any monotonic
   transform (log, percentile) is "free".
2. **The ground truth is (almost certainly) a function of these same 23 attributes.** Amex
   would not grade us on something unknowable from the data they shared. So the true profit
   is a Revenue−Cost−Risk function of f1…f23, and our job is to **reverse-engineer a ranking
   that matches their top-20% cut** — using business priors, since there is no label to fit.

The public/private split is 70/30 of the same data; the metric is identical on both. With
10 submissions allowed, the **public leaderboard is the legitimate model-selection signal**.

---

## 2. How Amex earns and loses money on a Premier cardmember

The Premier Card is an ultra-premium **charge** card (Amex-Platinum-like): $500–750 annual
fee, no preset spend limit, 5× points on flights/prepaid hotels, 1× elsewhere, and a large
stack of annual credits (airline, hotel, cab, dining/entertainment, etc.).

### Revenue (issuer)
| Lever | Mechanism | Magnitude intuition |
|---|---|---|
| **Interchange / discount revenue** | ~2.2–2.6% of every dollar spent, paid by merchants | **Dominant.** On $60k–$500k annual spend this is **$1.3k–$13k gross**. |
| **Net interest income** | APR on any **revolving / Pay-Over-Time balance** | Secondary; matters for the minority who revolve. |
| **Annual fee** | $500–750, largely fixed per active CM | **Roughly constant → drops out of a *ranking*.** |
| FX, late fees, etc. | Minor | Negligible for ranking. |

### Cost (issuer)
| Lever | Mechanism |
|---|---|
| **Rewards cost** | Points earned ≈ 1–2¢ each; **5× travel categories cost far more in points than 1× spend**, so *net* interchange margin is thinner on Airlines/Lodging. |
| **Benefit redemptions** | Real cash out: **lounge visits** (~$30–40 each), **airline / cab / entertainment credits** actually used. |
| **Servicing** | Call-centre and operations, esp. **cancellation calls**. |

### Risk / loss
| Lever | Mechanism |
|---|---|
| **Expected credit loss** | `PD (risk score) × exposure (revolving balance) × LGD`. **Small for the mostly-transactor premium book, but a convex tail risk**: a high-risk revolver can be a net loss. |
| **Collection distress** | **Cancellation calls due to collection** = delinquency signal → elevated loss + churn. |
| **Attrition** | Cancellation intent destroys forward value (fee + future interchange). |

### Net
```
Profit ≈ net_interchange(spend, category_mix)
       + net_interest(revolve_balance, risk)
       + (≈constant annual fee)
       − benefit_redemptions(lounge, credits)
       − servicing(cancellation calls)
       − expected_credit_loss(risk, revolve_balance)
       − attrition/collection_loss
```
**Because interchange on tens of thousands of dollars dwarfs a ~$600 fee and a few hundred
dollars of credits, profit is INTERCHANGE-DOMINATED and roughly monotone in total spend.**
Everything else re-ranks the tail second-order.

---

## 3. What makes a Premier cardmember highly profitable

- **High total spend** — the #1 driver (interchange). Especially high *net-margin* spend
  (1× everyday categories out-earn equal-dollar 5× travel after rewards cost).
- **Revolves a balance but is low-risk** — earns interest without defaulting.
- **Deep, engaged, loyal relationship** — multiple cards/supplementary accounts, active
  logins, redeems/uses the card — engagement corroborates and sustains spend.
- **Does NOT over-consume costly benefits relative to spend** — the profitable power-user
  uses lounges/credits *and* spends heavily; the **"coupon-clipper"** maxes the credit stack
  (>$1,000/yr of hard cost) on low spend and is **net-negative**.
- **Low risk, no collection distress, no cancellation intent** — protects the tail.

The **profit tail ≠ the raw-spend tail**: coupon-clippers and high-risk revolvers look good
on one axis and destroy value on another. Separating them is where accuracy at the 80th
percentile is won.

---

## 4. Feature inference → economic role

Feature meanings are **known** (`data/feature_description.csv`). The table below records the
economic role and profit sign we assign each — the modelling judgement, not the identity.

| Feature | Meaning | Role | Sign | Notes |
|---|---|---|---|---|
| f5 | Labelled "Total Spend 12m" | Revenue | **excluded** | REFUTED as the category-sum (Spearman ~0.01, ~1/12 the scale, below Σ-parts in 88% of rows); f6…f10 are the interchange base. |
| f6 | Airlines Spend | Revenue | + (thin) | 5× category → lower net margin (rewards drag). |
| f9 | Lodging Spend | Revenue | + (thin) | 5× category → lower net margin. |
| f7 | Other Spend | Revenue | + | 1× category (catch-all). |
| f8 | Entertainment Spend | Revenue | + | 1× category. |
| f10 | Dining Spend | Revenue | + | 1× category. |
| f1 | Avg Revolve Balance | Revenue / exposure | + / risk | Interest income **and** credit exposure. |
| f17 | Total Lend Line | Relationship | + (weak) | Affluence/capacity proxy. |
| f18 | Total Consumer Lend Line | Relationship | + (weak) | Subset of f17 (near-duplicate). |
| f4 | Rewards Points Balance | Engagement | + (weak) | Accrues with spend; also unredeemed liability. |
| f21 | Points Redeemed 12m | Engagement | + (weak) | Active-use signal. |
| f12 | Website Logins | Engagement | + | Digital engagement. |
| f22 | Emails Opened 6m | Engagement | + (weak) | Marketing engagement. |
| f23 | Emails Clicked 6m | Engagement | + (weak) | Marketing engagement. |
| f19 | # Supplementary Accounts | Relationship | + | Household spend depth. |
| f20 | # Active Charge Cards | Relationship | + | Relationship depth. |
| f11 | **Average Risk Score** | Risk | **−** | Expected credit loss (PD). |
| f2 | Cancellation Calls 12m | Risk/attrition | − | Attrition intent. |
| f3 | **Cancellation Calls — Collection** | Risk | **− (severe)** | Delinquency/credit distress. |
| f13 | Lounge Access Count | Cost | − | ~$30–40 issuer cost per visit. |
| f14 | Airline Credits Used ($) | Cost | − | Cash redemption. |
| f15 | Cab Benefits Usage | Cost | − | Cash redemption. |
| f16 | Entertainment Credit Used ($) | Cost | − | Cash redemption. |
| id | identifier | — | exclude | Never used in the equation (rule). |

**Key structural facts we exploit:**
- **f5 is NOT the category total** — data forensics REFUTE `f5 = Σ(f6…f10)`:
  Spearman(f5, Σf6…f10) ≈ 0.01, f5 is ~1/12 the scale of the category sum, f5 sits *below*
  the sum of its own parts in 88% of rows, and f5 fails the rewards-accrual check (the
  category sum correlates +0.31/+0.24 with points balance / points redeemed, while f5 is only
  +0.03/+0.02). So we treat the **category spends f6…f10 as the interchange base** (summed
  then ranked) and **exclude f5** from the spend total. The category split (f6, f9 = 5×) still
  drives the *margin* adjustment.
- **f4 ≈ 570k values are POINTS, not dollars** — treating them as spend would have been a
  large error. Fixed.
- **We no longer assume a clean spend identity;** the remaining judgement is the *weight* of
  each second-order term, which we keep small and robust.

---

## 5. The mathematical profitability framework

We model, per the brief, **Profit = Revenue − Cost − Expected Risk Loss**, and implement it
in two complementary ways.

### (a) Robust rank form (scale-free, for the metric)
Every signal → a **robust percentile score in [0,1]** (rank-based ⇒ immune to scale, skew,
and the f4/f5 outliers), sign-corrected so higher = more profitable. Monetary buckets are
aggregated as **raw-dollar sums then ranked** (preserving total-volume ordering, which *is*
interchange). Groups: `spend_cat` (f6…f10, the interchange base — summed then ranked; f5 excluded), `revolve` (f1),
`credit_line` (f17/f18), `points` (f4/f21), `engagement` (f12/f22/f23), `relationship`
(f19/f20), `risk` (f11), `attrition` (f2), `collection` (f3), `benefit_cost` (f13–f16).

A framework is a signed weighted sum of these [0,1] group scores (spend weight = 1.0; all
non-spend terms kept small so they shift a member only a few percentile points and can never
overturn a full spend decile).

### (b) Dollar-grounded P&L (structural match to the target)
```
Revenue   = 0.022·TotalSpend(Σf6…f10) + interest·Revolve(f1)
Rewards   = 0.011·TotalSpend(Σf6…f10) + drag·(Airlines f6 + Lodging f9)      # 5× drag, net still +
Benefit   = AirlineCredit(f14) + EntCredit(f16) + louncost·Lounge(f13) + cabcost·CabUse(f15)
RiskLoss  = RiskScore(f11) · Revolve(f1) · LGD
Servicing = call·CancelCalls(f2) + collection·CollectionCalls(f3)
Profit$   = Revenue − Rewards − Benefit − RiskLoss − Servicing      →  rank by percentile
```
This directly encodes Amex's stated P&L and is the **best structural match to how the true
label was constructed**; the 5× drag is the mechanism by which net-profit rank diverges from
gross-spend rank.

### Ensuring robustness (no label to fit)
- **Percentile/rank transforms** everywhere → robust to outliers and to whether the true
  target is linear or concave in dollars.
- **Rank ensemble** of the active members `revenue_first` (0.30) + `relationship` (0.25) +
  `full_pnl` (0.25) + `risk_adjusted` (0.20) — `dollar_pnl` was removed — hedges coefficient
  mis-specification and stabilises the 80th-percentile boundary.
- **We do NOT fit weights to any data** (fitting to our synthetic would overfit our own
  assumptions). Weights are business priors; the **public leaderboard** chooses between the
  ensemble and pure dollar-P&L via the per-framework submission files.

See `docs/methodology.md` for the pipeline, frameworks, validation protocol and results.
