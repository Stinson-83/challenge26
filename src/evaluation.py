"""
evaluation.py
=============
The competition metric and a validation harness.

Metric (from the PDF): "Accuracy measured as percentage of Actual Top 20% vs Top 20%
profitable CMs identified." With equal set sizes this is the overlap fraction
(= recall@20% = precision@20%). We implement it exactly and add rank-quality diagnostics
(Spearman, top-k lift, decile capture) so frameworks can be compared even though the real
label is hidden — here we score them against the synthetic ground-truth profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def top_k_overlap_accuracy(pred_rank: pd.Series, true_value: pd.Series,
                           top_pct: float = 0.20) -> float:
    """
    THE competition metric. Fraction of the actual top-`top_pct` (by true_value) that also
    appear in the predicted top-`top_pct` (by pred_rank). Both sets have equal size, so
    overlap == accuracy == recall == precision.
    """
    n = len(pred_rank)
    k = max(1, int(round(top_pct * n)))
    pred_top = set(pd.Series(pred_rank).nlargest(k).index)
    true_top = set(pd.Series(true_value).nlargest(k).index)
    return len(pred_top & true_top) / k


def evaluate_ranking(pred_rank: pd.Series, true_value: pd.Series,
                     top_pcts=(0.05, 0.10, 0.20, 0.30)) -> dict:
    """Full diagnostic bundle for one ranking against a known truth."""
    pred_rank = pd.Series(pred_rank).astype(float)
    true_value = pd.Series(true_value).astype(float)
    out = {"spearman": float(spearmanr(pred_rank, true_value).statistic)}
    for p in top_pcts:
        out[f"acc_top{int(p*100)}"] = round(top_k_overlap_accuracy(pred_rank, true_value, p), 4)

    # Decile capture: share of total true profit captured by predicted top decile.
    order = pred_rank.sort_values(ascending=False).index
    tv = true_value.loc[order].values
    d = max(1, len(tv) // 10)
    total = np.nansum(true_value.values)
    out["top_decile_profit_capture"] = float(np.nansum(tv[:d]) / total) if total else np.nan
    return out


def compare_frameworks(scores: pd.DataFrame, true_value: pd.Series,
                       top_pct: float = 0.20) -> pd.DataFrame:
    """
    Rank every framework in a scores table (columns 'rank__*') by the competition metric
    against the known truth. Used on synthetic data to pick / validate the primary.
    """
    fw = [c.replace("rank__", "") for c in scores.columns if c.startswith("rank__")]
    rows = []
    truth = true_value.reindex(scores.index)
    for f in fw:
        diag = evaluate_ranking(scores[f"rank__{f}"], truth, top_pcts=(0.05, 0.10, 0.20))
        rows.append({"framework": f, **diag})
    return pd.DataFrame(rows).sort_values(f"acc_top{int(top_pct*100)}", ascending=False)
