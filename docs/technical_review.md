# Technical Review — Amex Campus Challenge 2026 R1 Profitability Repo
### Principal-Scientist audit for EXPECTED PRIVATE-LB performance and defensibility
**Date:** 2026-07-04 · **Scope:** full repository (`challenge26/`) · **Data:** real `data/train.csv` (500,000 rows)

> This is a **technical review only** — no solution code has been rewritten. It answers the six audit
> questions, backs every claim with numbers computed on the real 500K data, and lays out a
> generalization-first roadmap. Method: an 18-agent audit (8 code/doc reviewers + 6 empirical analyses
> on the real data + 4 adversarial verifiers). All headline numbers were reproduced directly from the CSV
> and the six submission workbooks; analysis scripts are in the session scratchpad.

---

## 0. Verdict in one paragraph

The repository is **two solutions wearing one coat**. **Component A** — the modular first-principles
pipeline in `src/` — is a genuinely sound, robust, business-grounded ranking engine; its best honest
artifact is `FINAL_SUBMISSION_0.768.xlsx`. **Component B** — the "inverse calibration" (`probe_leaderboard.py`
+ `docs/inverse_calibration.md` + the `FINAL_INVERSE_FIT`/`R2`–`R5` workbooks) — is a **public-leaderboard
overfitting engine**: it reverse-engineers the hidden label from ~10 graded scores. Its 0.900–0.912 public
numbers are **not a private-LB edge** and are a **compliance liability**. For expected private performance
and rules-compliance: **select the honest model for private scoring, quarantine Component B, and rebuild
the honest base so its key decisions are justified by data-internal forensics rather than leaderboard
feedback** — which we show is achievable for the single most important decision in the whole solution.

---

## 1. The six audit questions

### 1.1 Which components are statistically sound?

| Component | Sound? | Basis |
|---|---|---|
| **The metric** (`evaluation.py` top-k overlap) | ✅ **Exact** | Reproduces the PDF metric; verified: perfect=1.0, random mean=0.2001. Keep verbatim. |
| **Rank/percentile transform core** (`feature_engineering.percentile_rank`, group scores) | ✅ **Sound** | Distribution-free, global, unsupervised → structurally leakage-free; top-20% persists 98.3% under 2% noise, ~100% under 90% subsample. |
| **Role auto-detection** (`preprocessing.infer_column_roles`) | ✅ **Sound** | Deterministic, lands correctly on all 24 columns; robust to scale/skew. |
| **Rank-ensemble machinery** (`framework_rank_ensemble`) | ✅ **Sound mechanism** | Averaging percentile *ranks* (not raw scores) is the correct scale-invariant blend; stabilises the 80th-pct cut. |
| **Multiplicative risk form** (`risk_adjusted`) | ✅ **Well-behaved** | Multipliers stay in [0.65, 1], never invert ordering. |
| **Submission/pipeline I/O** (`submission.py`, `pipeline.py`) | ✅ **Mechanically correct** | id-join order-independent (verified diff 0.0 / 5.5e-17), exactly 500,000 unique ids, id never used in any equation, no NaN/drop corruption. **Rule-compliant.** |
| **Synthetic *metric* usage** | ✅ | The metric is applied correctly. |

**Bottom line:** the *estimator machinery* is sound. Its problems are **model specification and validation
design**, not statistics.

### 1.2 Which components are likely overfitting?

**Component B (inverse calibration) — HIGH overfitting, quantified and confirmed by four independent
arguments. This is the single most important finding.**

1. **Information-theoretic impossibility.** Specifying one 100k-of-500k top-20% set requires
   `log2 C(500000,100000) = 360,955 bits`. Ten public scores at the ±0.0015 noise floor supply
   **~81–100 bits total** → the reconstruction is **under-determined by ~4,000×**. The ~22,000-member
   deviation from the honest base is *unconstrained guessing*, not identified quantity.
2. **Parameter non-identifiability (measured).** Across the four fitted coefficient files
   (22/32/31/40 free coeffs vs 9–12 scalar constraints), the terms that justify moving members past the
   base are wildly unstable — `risk_f11`: −272 → −1912 → −2347 → **+212** (sign flip, ~11×); `cab_f15`,
   `clicks_f23`: sign flips; `coll_f3`: 33× swing; `cards_f20`: 10× (hits bound). **The only stable
   coefficients are `revolve_f1` (~0.34–0.39) + small category-spend — i.e. the honest base itself.**
   Nothing beyond the base is pinned by the data.
3. **The "ladder toward 0.97" is jitter around two champions.** Measured set overlaps show R3 re-anchors
   on R1 (0.97, 3,000 swaps) — *not* on the preceding R2; R5 re-anchors on R3 (0.98, 2,000 swaps) —
   *not* on R4. **R2 (0.880) and R4 (0.909) are rolled-back dead-ends.** Net R1→R5 drift is **4.6%**.
   The docs self-incriminate: R2 fit to `wRMSE 0.0009` — "far below the ±0.0015 noise floor" = fitting
   split/rounding noise.
4. **Expected private regression (estimated).** The 0.912 set (R3) shares only **0.778** of its top-20%
   with the honest base. Under "only the base signal generalizes, swaps ~random on private," expected
   private = `0.768·0.778 + b·0.222` ≈ **0.64–0.68**, capped at 0.768 and plausibly **~0.13 below the base**.
   **~100% of the +0.144 public lift is public-specific overfit.** A consistency check (forcing the shared
   core to base-rate implies swap public hit-rate 1.42 > 1) proves the public gain is label-fitting.

Two further integrity flags on Component B:
- **The optimizer does not exist in the repo.** `grep` for `differential_evolution|inverse_fit|manifold|
  champion.anchor|constraint.projection` over all `*.py` returns **nothing**. The R1–R5 methodology is
  **unreproducible**; the 0.880/0.912/0.909 scores are unverifiable.
- **Submission-cap conflict.** `leaderboard_results.md` documents exactly **10** graded submissions
  (#10 = 0.900). `inverse_calibration.md` then adds R2–R5 (0.880/0.912/0.909/…) → **13–14 implied graded
  submissions vs the 10-cap.** Either the cap was breached or those rounds were never graded (fabricated).

**Everything else has LOW overfitting risk** (no fitted parameters): Component A's transforms, weights,
and role detection are hand-set business priors, and clustering/PCA/outliers don't touch the ranking.

> **Caveat that matters:** *bootstrap stability is NOT evidence of transfer here.* Every framework is a
> deterministic per-row score, so 70%-subset bootstrap gives top-20% Jaccard 0.999–1.000 **by construction**.
> The private 30% is a random partition of the same population, so any deterministic score reproduces on it.
> Stability measures reproducibility of the *score*, never its correlation with the *hidden label*. The only
> genuine transfer risks are (a) fitting to public labels — Component B — and (b) weight/feature-selection
> tuned on public feedback — a mild risk in the honest 0.768 model.

### 1.3 Which assumptions are unsupported by the data?

Data-forensic findings that **contradict the documentation and the config semantics** (all computed on
the real 500K; the masked labels in `feature_description.csv` must be treated as hypotheses, not truth):

| Assumption (as coded/claimed) | Reality | Impact |
|---|---|---|
| **`f5` = "Total Spend" = sum of category spends** | Spearman(f5, Σf6..f10) = **0.013**; f5 is **~1/12** the scale; f5 < Σparts in **88%** of rows; f5's max ($13.6k) < f7's median ($14.4k). A total cannot be uncorrelated with, 12× smaller than, and below its own parts. | **Load-bearing.** The whole config framework weights `f5` at ~1.0 — it is ranking a near-noise column (public: f5-alone 0.337 ≈ random). |
| **"No missing values in the real data"** (`real_data_findings.md:8`) | **FALSE.** 20/23 features missing; **97.2% of rows** miss ≥1 value (f23 88%, f18 62%, f17 58%, f4/f21 51%, f6–f10 23%). The report shows 0% only because the pipeline **zero-imputes before the check**. | Hides that the ranking depends heavily on imputation policy. |
| **`f2`, `f3` are "Cancellation Calls" counts** | Both are strictly **binary {0,1}**. | Not counts. |
| **`f3` (collection cancels) ⊂ `f2` (cancels)** | **Near mutually exclusive**: f3=1&f2=1 only **99 rows** vs f3=1&f2=0 = **54,205**. | Treat as two **disjoint** negative flags, not nested. |
| **`f16` = "Entertainment Credit Used Amount"** | **0% zeros** (min 8.88), negative skew, **35% piled at the max** — not a zero-inflated usage amount; behaves like a bounded score. | Its meaning and its cost-sign are doubtful — don't subtract it confidently. |
| **`f11` = risk probability in [0,1]** | Capped at **0.326**, 33% exactly zero. | Partial — bounded/probability-like but non-standard range. |
| **`f4`/`f21` = points counts** | **Non-integer** (e.g. 98615.44). | Mild flag on identity. |
| **Category spends are comparable scales (summed raw)** | f7 max 146,700 vs f8 max 9,420; **`cats` ≈ f7 alone (ρ=0.885)**, f7 = 63.5% of category dollars, and f7 has **5.8% negative** values. | `money_score` raw-sum silently reduces "category spend" to "Other Spend f7". |
| **Data is clean/uncensored** | **Global winsorization artifact**: every continuous feature capped at exactly p99 with a uniform ~2.6% mass at the cap. | The exact top tail the top-20% metric scores is **flattened**; raw-dollar models ingest pre-clipped values. |
| **Synthetic validation is independent of the scorers** | The hidden truth and `dollar_pnl` share the **same functional form** → `dollar_pnl` "wins" synthetic (0.87) but is near-orthogonal (24–29%) on real data. | **Circular.** Selecting on synthetic would ship the *worst* framework. |

### 1.4 Which business assumptions are strongest?

Ranked by defensibility **and** data support:

1. **"Issuer profit is revenue-led (interchange on charge volume + revolve interest), with cost/risk as
   second-order corrections."** The economics are correct. The failure is *operationalisation* (mapping
   "charge volume" to the noise column `f5`), not the thesis.
2. **Credit loss is the `f11·f1` interaction (PD × EAD), not `f11` alone.** Strongly data-supported:
   `f11` correlates **+0.58** with revolve balance (profitable revolvers), so a broad risk penalty would
   demote the profitable tail; the interaction penalises only the high-risk × high-balance corner. This is
   the honest model's best idea.
3. **Revolve balance `f1` is a genuine, orthogonal revenue axis** (interest income), rank-independent of
   spend (ρ≈0.03). Convergent evidence: both the honest model *and* the reverse-engineered inverse-fit
   independently land on **`f1` + `f7`** as their top drivers (LOFO/permutation). This `f1`+category-spend
   pair is the **most defensible generalizing basis** in the whole repo.
4. **Multiplicative risk-discount form** — numerically sound, appropriately down-weighted.

Weakest business assumptions: `f5`-as-primary-spend (refuted), engagement helps (public showed it *hurts*,
0.726→0.701), benefit-credit subtraction (can demote a *retained* profitable segment), and the
"6 independent frameworks" diversity claim (they are 4 clones of the same `f5` signal — see §2).

### 1.5 Which modules should be retained?

Retain (with fixes) the **entire honest pipeline**: `evaluation.py` (verbatim), the rank/percentile core,
role detection, submission/pipeline I/O, and the EDA layer (clustering/PCA/outliers) as *interpretability
only*. This is the defensible, generalizing, rule-compliant base the competition intends.

### 1.6 Which modules should be removed or rewritten?

- **REMOVE from the solution:** `probe_leaderboard.py` + the inverse-calibration docs/artifacts, as a
  *submission path*. Keep them **only** as a clearly-labelled cautionary appendix on leaderboard information
  limits — retitled away from "final innovation," 0.97 targets stripped.
- **REWRITE (specification, not machinery):** the framework definitions — swap `f5`→category-spend as the
  ~1.0 primary; make risk the `f11·f1` interaction; fix `dollar_pnl`'s scale (or keep it excluded); repair
  the incoherent `pts`/`eng` bundles.
- **REWRITE (validation):** replace the circular synthetic harness and **add the resampling/robustness
  harness that is currently entirely absent** (see §4).
- **REWRITE (docs):** correct the false "no missing values" and "f5 = Σcategories" claims, the stale
  `rank_ensemble` weights_note (propagated into 3 docs), and the mislabeled "synthetic run" in
  `outputs/README.md`; lead with the 0.768 model, not the 0.900 file.

---

## 2. Master retain / rewrite / remove table

| Module / file | Verdict | Overfit risk | Why (private-LB lens) |
|---|---|---|---|
| `evaluation.py` (metric) | **Retain verbatim** | none | Exact competition metric; the one asset that transfers unconditionally. |
| `feature_engineering.py`, `scoring.py` | **Retain + fix** | low | Robust rank core is the keeper. Fix: `money_score` raw-sum (→ rank-then-average), near-binary penalty muting, dead `feat_df`/config block. |
| `preprocessing.py`, `data_io.py`, `config.py` | **Retain + fix** | low | Role detection sound. Fix: **preserve missingness** (emit block indicators before imputation); stop treating `semantics` as ground truth; add label-free forensic guards. |
| `profitability.py` + `config.yaml` frameworks | **Retain architecture / rewrite spec** | low (bias, not variance) | Keep rank-ensemble blending + multiplicative risk. **Swap `f5`→Σcategory**, use `f11·f1`, fix `dollar_pnl` scale, delete stale weights_note. |
| `probe_leaderboard.py` + `docs/inverse_calibration.md` + `FINAL_INVERSE_FIT`/`R2`–`R5` | **REMOVE from solution** | **HIGH** | Under-determined ~4,000×; coeffs unstable/sign-flipping; unreproducible (no optimizer in repo); implies >10 graded submissions; ≈100% of public lift won't transfer; compliance liability. |
| `synthetic.py` | **Rewrite** | high (as selector) | Circular (shares form with `dollar_pnl`). Add ≥3 structurally-different truths; never let it pick the submission. |
| `clustering/dimensionality/outliers/diagnostics/eda/visualization` | **Retain as EDA** | n/a | Zero submission impact. Fix narrative overreach; disclose 80K subsample + circular cluster-profitability. |
| `submission.py`, `pipeline.py`, `framework_writeup.py` | **Retain + fix** | low | Mechanically correct, rule-compliant. Make id-join index-based; **never attach the honest writeup to a probe-derived submission** (provenance = gaming). |
| `README.md`, `docs/*.md`, `outputs/README.md` | **Rewrite** | — | Fix false claims; lead with the honest model. |

---

## 3. The generalization principle & the "no leaderboard feedback" tension

The instruction is to avoid **any** leaderboard dependence. Two honest observations:

**(a) The most important decision survives a strict no-leaderboard test.** The single biggest score driver
(`f5`→category-sum, public 0.337→0.668) is **re-derivable from data-internal forensics alone** — five
independent, label-free signals converge: (1) `f5` is censored/capped; (2) `f5` < sum-of-its-parts in 88%
of rows (a total can't be below its parts); (3) `f5` ~0 correlation with its components; (4) `f6–f10` form
a coherent cross-category latent (mean ρ 0.60); (5) **rewards cross-check** — `cats` correlates +0.31/+0.24
with points balance/redeemed (rewards accrue from spend) while `f5` correlates +0.03/+0.02. Any one of
(1),(2),(5) alone rejects `f5`. **So we can rebuild the model leaderboard-free and keep the gain**; only the
exact magnitude was ever LB-confirmed. Frame it as *"data-motivated, leaderboard-corroborated,"* not
*"leaderboard-derived."*

**(b) Stability/agreement weighting is a DECOY here — do not blindly apply it.** The four config frameworks
agree 82–97% *only because they all share the `f5` misspecification*. A variance/entropy/agreement-weighted
ensemble of them reproduces the `f5` ranking (top-20% overlap **0.85 with f5, 0.34 with the validated
category-sum model**) → expected **~0.34–0.40**. **Maximising internal agreement would actively select the
wrong model.** Robustness must be applied to **validity-screened** signals, never to raw agreement. This is
the key methodological guardrail for the weight-estimation work.

---

## 4. Forward roadmap (recommendations only — WHY each improves expected generalization)

> Not implemented yet, per the "review first" instruction. Ordered by expected private-LB impact.

**R1 — Fix the primary signal (highest impact).** Replace `f5`-as-spend with a robust category-spend
construct **and** revolve `f1`, the two convergent load-bearing drivers. *Why:* the current ensemble
overlaps the validated model only ~0.34 in the top-20%, so it likely scores ~0.34–0.40 on *both* splits
(bias transfers). This is the biggest single expected-private gain, and it's data-justified (§3a).

**R2 — Represent category spend correctly.** Rank each of `f6–f10` then average (or scale-normalise before
summing) instead of raw-summing. *Why:* raw-sum collapses "category spend" to `f7` alone (ρ 0.93) and lets
one heavy-tailed, refund-bearing column dominate; a rank-average is a true multi-category signal and more
stable across resamples.

**R3 — Risk as interaction, collection as a disjoint flag.** Use `−w·rank(f11·f1)` (PD×EAD) and treat `f3`
as its own binary penalty (not nested in `f2`). *Why:* broad `f11` subtraction demotes profitable revolvers
(ρ(f11,f1)=+0.58); the interaction penalises only the dangerous corner. Data-supported, generalizes.

**R4 — Principled, robust weights (not the 0.573/0.427 LB point).** The honest model's specific split is
its **least weight-robust** aspect (weight-perturbation top-20% overlap 0.805, sd 0.122, **min 0.499** — a
plausible perturbation ejects half the top-20%). *Why & how:* choose an operating point in a **flat region**
of the top-20%-stability surface; derive relative weights from *data-internal* axis strength (e.g. the
rewards-accrual oracle, or equal priors on the two validated revenue axes) rather than the LB-fit point.
**Explicitly avoid** entropy/variance/agreement weighting over the `f5`-family (§3b). Bayesian model
averaging is acceptable *only* across genuinely independent, validity-screened members.

**R5 — Missing-data as signal (modest, honest lift).** Emit structural missing-indicators (f6–f10 block,
f4/f21, f17, f18, f23) before imputation. *Why:* missingness is **MNAR, block-structured, low-entropy**
(5 patterns = 50% of rows) → stable behavioral segments that **transfer to any split**; the f6–f10-missing
block is a de-facto revolver/high-risk flag (collection Cohen's d ≈ 1.9). **Honest caveat:** it's largely
**collinear** with risk features (`f3`, `f11`) already used, so expect *marginal, mostly-redundant* lift,
not a new orthogonal axis. Also **stop zero-imputing before the data-quality check** (the source of the
false "0% missing" report) and prefer a NaN convention chosen for stability (median-vs-zero alone churns
~5,000 of the top 100k).

**R6 — Rebuild validation as a robustness harness (currently absent).** Add: (i) **many random 70/30
splits** and structurally-**different** synthetic truths (break the circularity) to stress *framework
selection*; (ii) **weight-perturbation**, **imputation-policy**, **feature-noise**, and **LOFO/permutation**
sensitivity as the real robustness diagnostics; (iii) **data-internal construct-validity oracles** (rewards
accrual, censoring/total-vs-parts, block coherence). *Why:* for deterministic scores, bootstrap top-20%
stability is ≈1 by construction and proves nothing — these are the diagnostics that actually discriminate
generalizing signals from fragile ones, and they need no leaderboard.

**R7 — Ensemble only over independent, validated signals.** Rank-aggregate (mean-rank / Borda / RRF) over
{category-spend, revolve, risk-interaction, missingness} — not four `f5` clones. *Why:* the current
ensemble's diversity is illusory (primary ρ 0.97 with pure spend); genuine independence is what makes an
ensemble reduce variance without importing shared bias.

**R8 — Code-quality fixes** (low risk, high clarity): consume or delete `engineer_features`/dead config;
make `money_score`/`percentile_rank` NaN handling consistent; index-based id-join; fix `risk_adjusted`'s
ignored config weight and `credit_line` double-count; reconcile the stale weights_note across config + 3
docs. *Why:* removes silent corruption vectors and makes the audited scoring surface unambiguous.

---

## 5. Submission & compliance decision

- **For private scoring, select `FINAL_SUBMISSION_0.768.xlsx`** (the honest model) — it is the largest
  identifiable/stable component and public≈private for a non-fitted global rank model. Better still, submit
  the **R1–R7 rebuilt model**.
- **Do NOT select any `FINAL_INVERSE_FIT`/`R2`–`R5` file.** Expected private ~0.64–0.68 and high gaming
  exposure.
- **Never attach the honest `framework_writeup` to a probe-derived submission** — that misrepresents
  provenance, which is exactly the "misuse or gaming" the rules police.

---

## 6. One open decision for the reviewer

How strictly should "no leaderboard feedback" be enforced on the *rebuilt* model?
- **Recommended:** rebuild so every structural choice is justified by data-internal forensics + business
  priors (we've shown the key decision passes this), and cite past LB scores only as weak *post-hoc*
  corroboration — never as a fitting target. Defensible and near leaderboard-independent.
- **Strict purist:** discard even the coarse feature-selection knowledge from past probes and re-derive
  from forensics only. Achievable, marginally more conservative.

Both lead to essentially the same model; they differ only in how the write-up frames provenance.
