"""
data_io.py
==========
Thin, defensive I/O layer. Loads the competition file if present, otherwise falls back
to the synthetic generator so the whole pipeline is runnable end-to-end without the
real data. Also handles saving outputs.
"""
from __future__ import annotations

import os

import pandas as pd

from .config import Config, load_config


def load_dataset(cfg: Config | None = None, verbose: bool = True) -> tuple[pd.DataFrame, bool]:
    """
    Return (dataframe, is_real).

    is_real is True when the actual competition file was found on disk, False when we
    fell back to synthetic data. Callers use this to decide whether to trust absolute
    numbers (real) or treat the run as a pipeline smoke-test (synthetic).
    """
    cfg = cfg or load_config()
    path = cfg.train_path()
    if os.path.exists(path):
        if verbose:
            print(f"[data_io] Loading real competition data: {path}")
        df = _read_any(path)
        return df, True

    if cfg.raw["data"].get("synthetic_fallback", True):
        if verbose:
            print(f"[data_io] Real data not found at {path}. Generating synthetic fallback "
                  f"(structure mirrors the screenshot; a hidden ground-truth profit is baked in "
                  f"for validation only).")
        from .synthetic import make_synthetic_dataset
        df, _truth = make_synthetic_dataset(cfg=cfg)
        return df, False

    raise FileNotFoundError(
        f"Competition data not found at {path} and synthetic_fallback is disabled."
    )


def _read_any(path: str) -> pd.DataFrame:
    """Read csv / parquet / excel by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path)
    # Best effort: assume csv.
    return pd.read_csv(path)


def save_df(df: pd.DataFrame, path: str, index: bool = False) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    return path
