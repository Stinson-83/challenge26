"""
test_pipeline.py — fast, dependency-light tests (run with `pytest -q` or `python -m pytest`).

They lock in the invariants that matter for a valid, competitive submission:
  * rule compliance (all ids, no rows added/dropped, id never scored),
  * robust group scores are well-formed [0,1],
  * every framework produces a valid full ranking,
  * the exact competition metric behaves (perfect ranking = 1.0, random ~ top_pct),
  * on synthetic data the spend-dominant frameworks clear a sensible accuracy bar.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.evaluation import compare_frameworks, top_k_overlap_accuracy
from src.feature_engineering import build_group_scores, percentile_rank
from src.preprocessing import clean, feature_columns, infer_column_roles
from src.profitability import score_dataframe
from src.submission import build_submission
from src.synthetic import make_synthetic_dataset


@pytest.fixture(scope="module")
def data():
    cfg = load_config()
    df, truth = make_synthetic_dataset(n=8000, cfg=cfg)
    roles = infer_column_roles(df, cfg)
    clean_df, _ = clean(df, cfg, roles)
    return cfg, df, clean_df, truth


def test_metric_bounds():
    truth = pd.Series(np.arange(1000))
    assert top_k_overlap_accuracy(truth, truth, 0.20) == 1.0          # perfect
    rng = np.random.default_rng(0)
    rand = pd.Series(rng.permutation(1000))
    acc = top_k_overlap_accuracy(rand, truth, 0.20)
    assert 0.10 < acc < 0.32                                          # random ~ 0.20


def test_clean_preserves_rows(data):
    cfg, df, clean_df, _ = data
    assert len(clean_df) == len(df)                                  # no rows added/dropped
    assert list(clean_df[cfg.id_column]) == list(df[cfg.id_column])  # id order & values intact
    assert clean_df.drop(columns=[cfg.id_column]).isna().sum().sum() == 0


def test_id_excluded_from_features(data):
    cfg, _, clean_df, _ = data
    feats = feature_columns(clean_df, cfg)
    assert cfg.id_column not in feats


def test_group_scores_well_formed(data):
    cfg, _, clean_df, _ = data
    gs = build_group_scores(clean_df, cfg)
    assert not gs.isna().any().any()
    for col in ["spend", "revolve", "risk", "benefit_cost", "collection"]:
        assert gs[col].between(-1, 1).all()
    assert gs["spend"].between(0, 1).all()


def test_all_frameworks_rank(data):
    cfg, _, clean_df, _ = data
    res = score_dataframe(clean_df, cfg)
    rank_cols = [c for c in res.scores.columns if c.startswith("rank__")]
    assert len(rank_cols) >= 6
    for c in rank_cols:
        r = res.scores[c]
        assert r.between(0, 1).all() and r.notna().all()


def test_submission_valid(data):
    cfg, df, clean_df, _ = data
    res = score_dataframe(clean_df, cfg)
    sub = build_submission(clean_df[cfg.id_column], res.primary_rank, cfg)
    assert len(sub) == len(df)
    assert sub.iloc[:, 0].nunique() == len(df)                       # all ids present, unique


def test_spend_dominant_beats_bar(data):
    cfg, _, clean_df, truth = data
    res = score_dataframe(clean_df, cfg)
    comp = compare_frameworks(res.scores, truth, 0.20).set_index("framework")["acc_top20"]
    # On an interchange-dominated truth, the strong frameworks must clear a clear bar.
    assert comp["dollar_pnl"] > 0.60
    assert comp["rank_ensemble"] > 0.60
    assert comp.max() > 0.75
