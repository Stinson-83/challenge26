"""
pipeline.py
===========
End-to-end orchestration. One command runs the whole research pipeline:

  load -> clean -> EDA/data-quality -> feature engineering -> group scores ->
  all profitability frameworks -> clustering + profiling -> PCA/FA/ICA ->
  outlier analysis -> scorecard -> visualisations -> (validation on synthetic) ->
  submission file.

Run:  python -m src.pipeline            (uses config.yaml; synthetic fallback if no data)
      python -m src.pipeline --framework revenue_first
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import (clustering, diagnostics, dimensionality, eda, evaluation, outliers,
               scoring, visualization)
from .config import load_config
from .data_io import load_dataset, save_df
from .feature_engineering import engineer_features
from .preprocessing import clean, feature_columns, infer_column_roles
from .profitability import framework_agreement, score_dataframe
from .submission import build_submission, write_all_framework_submissions, write_submission


def run(config_path: str | None = None, framework: str | None = None,
        outdir: str | None = None, make_plots: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    cfg = load_config(config_path) if config_path else load_config()
    active = framework or cfg.active_framework
    outdir = outdir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outdir, exist_ok=True)
    log = lambda *a: print(*a) if verbose else None

    # ---- 1. Load ------------------------------------------------------------------
    df, is_real = load_dataset(cfg, verbose=verbose)
    truth = None
    if not is_real:                       # synthetic run carries a hidden truth for validation
        from .synthetic import make_synthetic_dataset
        df, truth = make_synthetic_dataset(cfg=cfg)
    log(f"[1] Loaded {len(df):,} rows x {df.shape[1]} cols (real={is_real})")

    # ---- 2. Clean + schema --------------------------------------------------------
    roles = infer_column_roles(df, cfg)
    clean_df, clean_log = clean(df, cfg, roles)
    feats = feature_columns(clean_df, cfg, roles)
    log(f"[2] Cleaned. {len(feats)} modelling features; "
        f"identifiers excluded: {[c for c,r in roles.items() if r.role=='identifier']}")

    # ---- 3. EDA / data-quality + real-data diagnostics ----------------------------
    report = eda.data_quality_report(clean_df, cfg)
    eda.save_report(report, os.path.join(outdir, "data_quality_report.json"))
    corr = eda.correlation_matrix(clean_df, feats, method="spearman")
    diag = diagnostics.run_all(clean_df, cfg)
    with open(os.path.join(outdir, "diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=2, default=eda._json_default)
    log(f"[3] Data-quality report written. Roles: {report['role_counts']}")
    for k, v in diag.items():
        log(f"       diag[{k}]: {v.get('verdict', v.get('status'))}")

    # ---- 4. Feature engineering ---------------------------------------------------
    feat_df = engineer_features(clean_df, cfg)
    log(f"[4] Engineered {feat_df.shape[1]} features.")

    # ---- 5. Profitability frameworks ---------------------------------------------
    result = score_dataframe(clean_df, cfg, active=active)
    gs = result.group_scores
    agreement = framework_agreement(result.scores, cfg.top_pct)
    log(f"[5] Scored {len(result.scores):,} CMs across "
        f"{sum(c.startswith('score__') for c in result.scores.columns)} frameworks. Active: {active}")

    # Heavy analysis (clustering/PCA/outliers/plots) runs on a representative SUBSAMPLE for
    # scalability; the submission scores always use ALL rows (score_dataframe is vectorised).
    an_cap = int(cfg.raw.get("clustering", {}).get("analysis_sample_max", 80000))
    if len(gs) > an_cap:
        an_idx = gs.sample(an_cap, random_state=cfg.seed).index
    else:
        an_idx = gs.index
    gs_an = gs.loc[an_idx]
    rank_an = result.primary_rank.loc[an_idx]
    sub_an = result.subscores.loc[an_idx]

    # ---- 6. Clustering + profiling ------------------------------------------------
    labels = clustering.fit_all(gs_an, cfg)
    kprofile = clustering.profile_clusters(labels["kmeans"], gs_an, sub_an, rank_an)
    log(f"[6] Clustering done (n={len(gs_an):,}). Most-profitable KMeans cluster: "
        f"{kprofile.index[0]} (mean rank {kprofile['profitability_rank'].iloc[0]:.3f})")

    # ---- 7. Dimensionality --------------------------------------------------------
    pca = dimensionality.run_pca(gs_an, n_components=5, seed=cfg.seed)
    pca_rank = dimensionality.spend_aligned_pc1(gs, seed=cfg.seed)   # full: used in validation
    log(f"[7] PCA explained variance (top5): "
        f"{np.round(pca['explained_variance_ratio'], 3).tolist()}")

    # ---- 8. Outliers --------------------------------------------------------------
    out_res = outliers.analyze_outliers(gs_an, rank_an, cfg)
    log(f"[8] Outliers: {out_res['summary']}")

    # ---- 9. Scorecard -------------------------------------------------------------
    card = scoring.build_scorecard(clean_df, cfg)

    # ---- 10. Validation (synthetic only) -----------------------------------------
    validation = None
    if truth is not None:
        comp = evaluation.compare_frameworks(result.scores, truth, cfg.top_pct)
        # also evaluate PCA baseline and scorecard overalls
        extra = []
        extra.append(("pca_spend_pc1", evaluation.evaluate_ranking(pca_rank, truth)))
        for strat in ("weighted", "rank_mean", "multiplicative"):
            extra.append((f"scorecard_{strat}",
                          evaluation.evaluate_ranking(card[f"overall_{strat}"], truth)))
        extra_df = pd.DataFrame([{"framework": n, **d} for n, d in extra])
        validation = pd.concat([comp, extra_df], ignore_index=True).sort_values(
            "acc_top20", ascending=False)
        validation.to_csv(os.path.join(outdir, "validation_scores.csv"), index=False)
        log("[10] VALIDATION (top-20% overlap accuracy vs hidden truth):")
        for _, r in validation.iterrows():
            log(f"       {r['framework']:>22s}  acc@20={r['acc_top20']:.3f}  "
                f"acc@10={r.get('acc_top10', float('nan')):.3f}  spearman={r['spearman']:.3f}")

    # ---- 11. Visualisations -------------------------------------------------------
    plots = {}
    if make_plots:
        plots["correlation"] = visualization.correlation_heatmap(corr, outdir)
        plots["distributions"] = visualization.distributions(clean_df, feats, outdir)
        plots["pca"] = visualization.pca_scatter(pca["scores"], result.primary_rank, outdir)
        plots["clusters"] = visualization.cluster_scatter(pca["scores"], labels["kmeans"], outdir)
        plots["outliers"] = visualization.outlier_plot(pca["scores"],
                                                       out_res["scores"]["iforest_anomaly"], outdir)
        plots["score_dist"] = visualization.score_distribution(result.scores, outdir)
        plots["overlap"] = visualization.framework_overlap_heatmap(agreement, outdir)
        plots["cluster_profit"] = visualization.cluster_profile_bar(kprofile, outdir)
        log(f"[11] Wrote {len(plots)} plots to {outdir}/")

    # ---- 12. Submission (primary + one per framework for leaderboard A/B) ---------
    ids = clean_df[cfg.id_column]
    template = cfg.raw["submission"].get("template_file")
    sub = build_submission(ids, result.primary_rank, cfg, template_path=template)
    sub_path = write_submission(sub, cfg)
    fw_subs = write_all_framework_submissions(result.scores, cfg, outdir=outdir, template_path=template)
    log(f"[12] Primary submission: {sub_path}  ({len(sub):,} rows). "
        f"Per-framework submissions: {len(fw_subs)} in {outdir}/ (submit each; keep the best public score).")

    # ---- artefacts ----------------------------------------------------------------
    save_df(result.scores, os.path.join(outdir, "all_framework_scores.csv"))
    save_df(card, os.path.join(outdir, "scorecard.csv"))
    save_df(agreement, os.path.join(outdir, "framework_agreement.csv"))
    save_df(kprofile.reset_index(), os.path.join(outdir, "kmeans_cluster_profile.csv"))
    save_df(pca["loadings"].reset_index().rename(columns={"index": "group"}),
            os.path.join(outdir, "pca_loadings.csv"))

    summary = {
        "n_rows": int(len(df)), "is_real_data": bool(is_real), "active_framework": active,
        "runtime_sec": round(time.time() - t0, 1),
        "role_counts": report["role_counts"],
        "framework_agreement_min_overlap": float(agreement["top_overlap"].min()),
        "most_profitable_kmeans_cluster": int(kprofile.index[0]),
        "submission_path": sub_path,
        "plots": plots,
        "validation_best": (validation.iloc[0].to_dict() if validation is not None else None),
    }
    with open(os.path.join(outdir, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=eda._json_default)
    log(f"[done] {summary['runtime_sec']}s. Summary -> {outdir}/run_summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Amex R1 profitability pipeline")
    ap.add_argument("--config", default=None)
    ap.add_argument("--framework", default=None,
                    help="revenue_first|full_pnl|risk_adjusted|relationship|rank_ensemble")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run(config_path=args.config, framework=args.framework, outdir=args.outdir,
        make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
