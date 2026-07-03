"""
submission.py
=============
Build the Round-1 submission file.

Rules honoured (from the PDF guidelines):
  * The solution runs on ALL unique_identifiers (one row per input id, no rows added/dropped).
  * Identifier variables are never used inside the equation (only carried through for output).
  * Output column names are auto-aligned to the exact Unstop template if one is provided,
    so the file matches the required format byte-for-byte.
"""
from __future__ import annotations

import os

import pandas as pd

from .config import Config, load_config


def align_to_template(sub: pd.DataFrame, template_path: str | None,
                      id_col: str, score_col: str) -> pd.DataFrame:
    """
    If an Unstop template CSV is available, rename our (id, score) columns to the template's
    column names and column order. Otherwise return as-is.
    """
    if not template_path or not os.path.exists(template_path):
        return sub
    tmpl = pd.read_csv(template_path, nrows=5)
    cols = list(tmpl.columns)
    if len(cols) >= 2:
        rename = {id_col: cols[0], score_col: cols[1]}
        out = sub.rename(columns=rename)
        return out[[cols[0], cols[1]]]
    if len(cols) == 1:            # some templates want only the ranked id list
        return sub[[id_col]].rename(columns={id_col: cols[0]})
    return sub


def build_submission(
    ids: pd.Series,
    ranking: pd.Series,
    cfg: Config | None = None,
    template_path: str | None = None,
    as_rank: bool = False,
) -> pd.DataFrame:
    """
    Assemble the submission dataframe.

    ranking : per-id profitability score (higher = more profitable). We emit the continuous
              score by default; set as_rank=True to emit an integer rank (1 = most profitable)
              if the template requires ranks.
    """
    cfg = cfg or load_config()
    sub = cfg.raw["submission"]
    id_name = sub.get("id_out_name", "id")
    score_name = sub.get("score_out_name", "profitability_score")

    out = pd.DataFrame({id_name: ids.values, score_name: ranking.values})
    if as_rank:
        out[score_name] = out[score_name].rank(ascending=False, method="first").astype(int)
    out = out.sort_values(id_name).reset_index(drop=True)     # preserve id order

    tmpl = template_path or cfg.raw["submission"].get("template_file")
    out = align_to_template(out, tmpl, id_name, score_name)
    return out


def write_submission(out: pd.DataFrame, cfg: Config | None = None, path: str | None = None) -> str:
    cfg = cfg or load_config()
    path = path or cfg.out_path("out_file")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return path


def write_all_framework_submissions(scores: pd.DataFrame, cfg: Config | None = None,
                                    outdir: str | None = None,
                                    template_path: str | None = None) -> dict[str, str]:
    """
    Emit ONE submission file per framework (using its percentile rank as the score), so the
    user can upload each to the PUBLIC LEADERBOARD and keep whichever scores highest — the
    only legitimate way to choose the final model when the true label is hidden.
    Returns {framework: path}.
    """
    import os
    cfg = cfg or load_config()
    outdir = outdir or os.path.dirname(cfg.out_path("out_file"))
    id_col = cfg.id_column
    ids = scores[id_col] if id_col in scores.columns else pd.Series(scores.index, index=scores.index)
    template = template_path or cfg.raw["submission"].get("template_file")
    paths = {}
    for c in [c for c in scores.columns if c.startswith("rank__")]:
        fw = c.replace("rank__", "")
        sub = build_submission(ids, scores[c], cfg, template_path=template)
        path = os.path.join(outdir, f"submission_{fw}.csv")
        write_submission(sub, cfg, path=path)
        paths[fw] = path
    return paths


def top_flag_submission(ids: pd.Series, ranking: pd.Series, top_pct: float,
                        cfg: Config | None = None) -> pd.DataFrame:
    """Alternative format: a 0/1 flag marking the predicted top-`top_pct` (some templates
    ask for the identified top set rather than a score)."""
    cfg = cfg or load_config()
    n = len(ids)
    k = max(1, int(round(top_pct * n)))
    top_idx = ranking.nlargest(k).index
    flag = pd.Series(0, index=ranking.index)
    flag.loc[top_idx] = 1
    return pd.DataFrame({cfg.raw["submission"].get("id_out_name", "id"): ids.values,
                         "is_top20": flag.values}).sort_values(
        cfg.raw["submission"].get("id_out_name", "id")).reset_index(drop=True)
