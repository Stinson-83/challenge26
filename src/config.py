"""
config.py
=========
Typed access to `config.yaml`. Centralises paths, the inferred semantic map, the
auto-detection thresholds and every framework's weights so that the rest of the
pipeline never hard-codes a column name or a magic number.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

# Project root = parent of this file's directory (src/..).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


def resolve(path: str) -> str:
    """Resolve a possibly-relative path against the project root."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


@dataclass
class Config:
    """In-memory view of config.yaml with a couple of convenience helpers."""

    raw: dict[str, Any]
    path: str = DEFAULT_CONFIG_PATH

    # ---- frequently used shortcuts -------------------------------------------------
    @property
    def id_column(self) -> str:
        return self.raw["data"]["id_column"]

    @property
    def seed(self) -> int:
        return int(self.raw.get("random_seed", 42))

    @property
    def semantics(self) -> dict[str, dict]:
        return self.raw.get("semantics", {}) or {}

    @property
    def autodetect(self) -> dict[str, Any]:
        return self.raw.get("autodetect", {}) or {}

    @property
    def frameworks(self) -> dict[str, Any]:
        return self.raw.get("frameworks", {}) or {}

    @property
    def active_framework(self) -> str:
        return self.frameworks.get("active_framework", "rank_ensemble")

    @property
    def top_pct(self) -> float:
        return float(self.raw.get("submission", {}).get("top_pct", 0.20))

    def train_path(self) -> str:
        return resolve(self.raw["data"]["train_file"])

    def out_path(self, key: str = "out_file") -> str:
        return resolve(self.raw["submission"][key])

    def group_of(self, col: str) -> str | None:
        """Semantic aggregation group of a column, if we specified one."""
        return self.semantics.get(col, {}).get("group")

    def columns_in_group(self, group: str) -> list[str]:
        return [c for c, m in self.semantics.items() if m.get("group") == group]

    def sign_of(self, col: str) -> str:
        return self.semantics.get(col, {}).get("profit_sign", "ambiguous")

    def framework_weights(self, name: str) -> dict[str, Any]:
        fw = self.frameworks.get(name, {})
        return {k: v for k, v in fw.items() if k not in ("weights_note", "members", "member_weights")}


@lru_cache(maxsize=4)
def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and cache the YAML config."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=path)
