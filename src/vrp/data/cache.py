"""Local parquet cache for data layer outputs.

Each cache entry is a DataFrame keyed by a string. Files live under
data/vrp_cache/ at the repo root. Cache is content-addressed only by
its key — callers are responsible for invalidating when upstream data
changes (e.g. pass a date-suffixed key for daily snapshots).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "vrp_cache"


def _path_for(key: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}.parquet"


def load(key: str) -> Optional[pd.DataFrame]:
    p = _path_for(key)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def save(key: str, df: pd.DataFrame) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path_for(key)
    df.to_parquet(p, index=True)
    return p
