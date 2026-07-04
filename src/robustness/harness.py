"""
harness.py — the robustness runner + CLI.
========================================
Runs the full battery on the real data and emits a scorecard. The METRIC is the exact
competition metric (top-20% overlap); we reuse src.evaluation.top_k_overlap_accuracy and
cross-check the fast array version against it.

Battery (per ranker in src.robustness.rankers.REGISTRY):
  A. TRUTH RECOVERY (headline)  — mean & WORST-CASE top-20% recovery across the diverse truth
                                  panel x seeds. Worst-case = robustness to economic uncertainty.
  B. WEIGHT SENSITIVITY         — top-20% overlap under lognormal weight jitter (fit-point risk).
  C. IMPUTATION SENSITIVITY     — median vs zero vs indicator NaN policy.
  D. FEATURE-NOISE SENSITIVITY  — multiplicative lognormal feature noise.
  E. LOFO / PERMUTATION         — which raw features are load-bearing (true drivers).
  F. CROSS-AGREEMENT            — top-20% overlap + Spearman matrix (reported, NOT optimized:
                                  agreement is a decoy — see README).
  G. BOOTSTRAP STABILITY        — reported with the explicit caveat that it is ~1 by construction
                                  for deterministic scores and is NOT a transfer proof.
Plus the label-free construct-validity diagnostics (src.robustness.diagnostics).

Usage:
    python -m src.robustness.harness                 # default: 200k subsample (fast)
    python -m src.robustness.harness --full          # all 500k rows
    python -m src.robustness.harness --sample 100000 --out outputs/robustness
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ..config import load_config, resolve
from ..data_io import load_dataset
from ..evaluation import top_k_overlap_accuracy
from . import diagnostics as diag
from .rankers import REGISTRY, impute_df, top_set
from .truths import TRUTHS, TRUTH_NOTES

TOP_PCT = 0.20


# ---------------------------------------------------------------------------------------
# fast metrics
# ---------------------------------------------------------------------------------------
def _overlap(idx_a: np.ndarray, idx_b: np.ndarray, k: int) -> float:
    return len(np.intersect1d(idx_a, idx_b, assume_unique=True)) / k


def _acc(pred_score: np.ndarray, true_val: np.ndarray, top_pct: float = TOP_PCT) -> float:
    """Fast top-20% overlap accuracy (== competition metric for equal-size sets)."""
    n = len(pred_score)
    k = max(1, int(round(top_pct * n)))
    return _overlap(top_set(pred_score, top_pct), top_set(true_val, top_pct), k)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------
def run(sample: int | None = 200_000, seed: int = 20260704,
        out_dir: str = "outputs/robustness", n_truth_seeds: int = 5,
        n_weight_draws: int = 100, n_noise_draws: int = 20, k_boot: int = 15,
        verbose: bool = True) -> dict:
    t0 = time.time()
    cfg = load_config()
    raw, is_real = load_dataset(cfg, verbose=verbose)
    if sample is not None and sample < len(raw):
        raw = raw.sample(sample, random_state=seed).reset_index(drop=True)
    n = len(raw)
    k = max(1, int(round(TOP_PCT * n)))
    rng = np.random.default_rng(seed)
    rng_w, rng_n, rng_l, rng_b = rng.spawn(4)   # independent streams so each knob is reproducible
    if verbose:
        print(f"[harness] {'REAL' if is_real else 'SYNTHETIC'} data, n={n:,}, top-20% k={k:,}")

    # base imputed frames. Only median vs zero are informative for the top-20% (an 'indicator'
    # floor is order-identical to zero at the cut); missingness-as-signal is handled by explicit
    # indicator COLUMNS in a candidate ranker, not by the fill policy.
    fdf = {p: impute_df(raw, p) for p in ("median", "zero")}
    base = fdf["median"]

    # ---- precompute nominal components + scores + top-sets --------------------------------
    names = list(REGISTRY)
    comps = {nm: REGISTRY[nm].build(base) for nm in names}
    score = {nm: REGISTRY[nm].combine_components(comps[nm]) for nm in names}
    tset = {nm: top_set(score[nm], TOP_PCT) for nm in names}

    # sanity: fast _acc matches src.evaluation on one ranker (tolerate boundary tie-breaking)
    _p = pd.Series(score["honest_0768"]); _t = pd.Series(TRUTHS["spend_led"](base, 1))
    _delta = abs(_acc(score["honest_0768"], TRUTHS["spend_led"](base, 1))
                 - top_k_overlap_accuracy(_p, _t, TOP_PCT))
    if _delta > 0.005 and verbose:
        print(f"[harness] WARN: fast metric differs from evaluation by {_delta:.4f} (tie-breaking?)")

    # ---- A. TRUTH RECOVERY (headline) ----------------------------------------------------
    truth_recovery = {nm: {} for nm in names}          # nm -> truth -> {mean,std,min}
    for tname, tfn in TRUTHS.items():
        per_seed = {nm: [] for nm in names}
        for s in range(1, n_truth_seeds + 1):
            tv = tfn(base, 1000 * s + 7)
            ttop = top_set(tv, TOP_PCT)
            for nm in names:
                per_seed[nm].append(_overlap(tset[nm], ttop, k))
        for nm in names:
            arr = np.array(per_seed[nm])
            truth_recovery[nm][tname] = {"mean": round(float(arr.mean()), 4),
                                         "std": round(float(arr.std()), 4),
                                         "min": round(float(arr.min()), 4)}
    # per-ranker aggregate: mean-over-truths and WORST-ECONOMY (min over truths of the
    # per-truth mean-over-seeds; the per-seed worst is kept in truth_recovery[...]['min']).
    truth_summary = {}
    for nm in names:
        per_truth_means = np.array([truth_recovery[nm][t]["mean"] for t in TRUTHS])
        truth_summary[nm] = {"mean_over_truths": round(float(per_truth_means.mean()), 4),
                             "worst_truth": round(float(per_truth_means.min()), 4),
                             "worst_truth_name": list(TRUTHS)[int(per_truth_means.argmin())]}

    # L1: is the headline ranking imputation-invariant? Recompute worst-economy under zero-fill.
    z_score = {nm: REGISTRY[nm].score(fdf["zero"]) for nm in names}
    z_tset = {nm: top_set(z_score[nm], TOP_PCT) for nm in names}
    truth_worst_zero = {}
    for nm in names:
        ptm = [np.mean([_overlap(z_tset[nm], top_set(tfn(fdf["zero"], 1000 * s + 7), TOP_PCT), k)
                        for s in range(1, 4)]) for tfn in TRUTHS.values()]
        truth_worst_zero[nm] = round(float(np.min(ptm)), 4)
    order_med = [nm for nm, _ in sorted(truth_summary.items(), key=lambda kv: -kv[1]["worst_truth"])]
    order_zero = sorted(truth_worst_zero, key=lambda nm: -truth_worst_zero[nm])
    top2_invariant = set(order_med[:2]) == set(order_zero[:2])

    # ---- B. WEIGHT SENSITIVITY -----------------------------------------------------------
    weight_sens = {}
    for nm in names:
        r = REGISTRY[nm]
        ov = []
        for _ in range(n_weight_draws):
            w = r.perturbed_weights(rng_w, sigma=0.30)
            sc = r.combine_components(comps[nm], w)
            ov.append(_overlap(tset[nm], top_set(sc, TOP_PCT), k))
        ov = np.array(ov)
        weight_sens[nm] = {"mean": round(float(ov.mean()), 4), "sd": round(float(ov.std()), 4),
                           "min": round(float(ov.min()), 4)}

    # ---- C. IMPUTATION SENSITIVITY (median vs zero — the only informative fill contrast) --
    imput_sens = {}
    for nm in names:
        sc_zero = z_score[nm]
        imput_sens[nm] = {"median_vs_zero": round(_overlap(tset[nm], top_set(sc_zero, TOP_PCT), k), 4)}

    # ---- D. FEATURE-NOISE SENSITIVITY ----------------------------------------------------
    noise_sens = {}
    feat_cols = [c for c in base.columns if c.startswith("f")]
    for nm in names:
        ov = []
        for _ in range(n_noise_draws):
            noisy = base.copy()
            fac = np.exp(rng_n.normal(0.0, 0.10, size=(n, len(feat_cols))))
            noisy[feat_cols] = base[feat_cols].values * fac
            sc = REGISTRY[nm].score(noisy)
            ov.append(_overlap(tset[nm], top_set(sc, TOP_PCT), k))
        noise_sens[nm] = {"mean": round(float(np.mean(ov)), 4), "min": round(float(np.min(ov)), 4)}

    # ---- E. LOFO + PERMUTATION -----------------------------------------------------------
    # NOTE: PERMUTATION is the trusted driver signal — it breaks a feature's alignment through
    # ALL terms including interactions (e.g. f11 in EL=f11*f1). Single-column LOFO understates a
    # feature that also feeds an interaction (setting f11->median leaves EL ~ R(f1)), so it is
    # kept as a secondary diagnostic only; the scorecard `top_driver` reads from permutation.
    drivers = {}
    for nm in names:
        r = REGISTRY[nm]
        lofo, perm = {}, {}
        for c in r.features:
            if c not in base.columns:
                continue
            d2 = base.copy(); d2[c] = float(pd.to_numeric(base[c], errors="coerce").median())
            lofo[c] = round(1.0 - _overlap(tset[nm], top_set(r.score(d2), TOP_PCT), k), 4)
            d3 = base.copy(); d3[c] = rng_l.permutation(base[c].values)
            perm[c] = round(1.0 - _overlap(tset[nm], top_set(r.score(d3), TOP_PCT), k), 4)
        drivers[nm] = {"lofo_top": sorted(lofo.items(), key=lambda kv: -kv[1])[:4],
                       "perm_top": sorted(perm.items(), key=lambda kv: -kv[1])[:4]}

    # ---- F. CROSS-AGREEMENT (reported, not optimized) ------------------------------------
    agree = {"overlap": {}, "spearman": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agree["overlap"][f"{a}|{b}"] = round(_overlap(tset[a], tset[b], k), 4)
            agree["spearman"][f"{a}|{b}"] = round(_spearman(score[a], score[b]), 4)

    # ---- G. BOOTSTRAP STABILITY (caveated) -----------------------------------------------
    boot = {}
    for nm in names:
        r = REGISTRY[nm]
        ov = []
        for _ in range(k_boot):
            sidx = rng_b.choice(n, size=int(0.7 * n), replace=False)
            sub = base.iloc[sidx]
            sub_score = r.score(sub)
            full_sub = score[nm][sidx]
            kk = max(1, int(round(TOP_PCT * len(sub))))
            ov.append(_overlap(top_set(sub_score, TOP_PCT), top_set(full_sub, TOP_PCT), kk))
        boot[nm] = round(float(np.mean(ov)), 4)

    # ---- construct-validity diagnostics (on RAW df) --------------------------------------
    construct = diag.run_all(raw)

    # ---- assemble scorecard --------------------------------------------------------------
    scorecard = []
    for nm in names:
        scorecard.append({
            "ranker": nm,
            "truth_mean": truth_summary[nm]["mean_over_truths"],
            "truth_worst": truth_summary[nm]["worst_truth"],
            "worst_truth": truth_summary[nm]["worst_truth_name"],
            "weight_robust": weight_sens[nm]["mean"],
            "weight_min": weight_sens[nm]["min"],
            "imput_med_vs_zero": imput_sens[nm]["median_vs_zero"],
            "noise_robust": noise_sens[nm]["mean"],
            "top_driver": drivers[nm]["perm_top"][0][0] if drivers[nm]["perm_top"] else "",
            "bootstrap": boot[nm],
            "note": REGISTRY[nm].note,
        })
    sc_df = pd.DataFrame(scorecard).sort_values("truth_worst", ascending=False).reset_index(drop=True)

    out_dir = resolve(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    sc_df.to_csv(os.path.join(out_dir, "scorecard.csv"), index=False)
    full = {"meta": {"n": n, "is_real": is_real, "sample": sample, "seed": seed,
                     "runtime_sec": round(time.time() - t0, 1)},
            "truth_recovery": truth_recovery, "truth_summary": truth_summary,
            "truth_worst_zero_impute": truth_worst_zero,
            "top2_invariant_across_imputation": bool(top2_invariant),
            "weight_sensitivity": weight_sens, "imputation_sensitivity": imput_sens,
            "noise_sensitivity": noise_sens, "drivers": drivers, "agreement": agree,
            "bootstrap": boot, "construct_validity": construct,
            "truth_notes": TRUTH_NOTES}
    with open(os.path.join(out_dir, "robustness_report.json"), "w") as fh:
        json.dump(full, fh, indent=2)

    if verbose:
        _print_summary(sc_df, construct, top2_invariant, out_dir, time.time() - t0)
    return {"scorecard": sc_df, "report": full}


def _print_summary(sc_df, construct, top2_invariant, out_dir, secs):
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print("ROBUSTNESS SCORECARD  (sorted by WORST-ECONOMY recovery = min over the 6-truth panel)")
    print("=" * 100)
    cols = ["ranker", "truth_mean", "truth_worst", "worst_truth", "weight_robust",
            "weight_min", "imput_med_vs_zero", "noise_robust", "top_driver", "bootstrap"]
    print(sc_df[cols].to_string(index=False))
    print(f"\nTop-2 ranking invariant across median/zero imputation: {top2_invariant}")
    print("\nConstruct-validity oracles (label-free):")
    print("  f5-as-total :", construct["total_vs_parts_f5"]["verdict"])
    print("  rewards     :", construct["rewards_accrual"]["verdict"])
    print("  censoring   :", construct["censoring"]["verdict"],
          f"({construct['censoring']['n_censored']} cols)")
    print("  redundancy  :", construct["redundancy"]["verdict"])
    print("  segments    :", construct["block_comissingness"]["verdict"])
    print(f"\n[harness] wrote scorecard.csv + robustness_report.json to {out_dir}  ({secs:.1f}s)")
    print("REMINDER: bootstrap ~1.0 is BY CONSTRUCTION (deterministic score), not a transfer proof;")
    print("         cross-agreement is a DECOY (shared f5 misspecification). Read WORST-CASE recovery.")


def main():
    ap = argparse.ArgumentParser(description="Robustness harness for R1 profitability rankers.")
    ap.add_argument("--full", action="store_true", help="use all 500k rows (default: 200k sample)")
    ap.add_argument("--sample", type=int, default=200_000, help="row subsample size")
    ap.add_argument("--out", type=str, default="outputs/robustness")
    ap.add_argument("--seed", type=int, default=20260704)
    args = ap.parse_args()
    run(sample=None if args.full else args.sample, out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
