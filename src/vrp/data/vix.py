"""VIX spot loader. Source: Yahoo ^VIX. Use for VIX-regime overlays and
as an IV proxy for the Strategy C VRP calculation (flag as approximation).
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import cache

_KEY_PREFIX = "vix_spot_daily"


def load_vix(start: str = "2005-01-01", end: str | None = None,
             use_cache: bool = True) -> pd.Series:
    key = f"{_KEY_PREFIX}__{start}__{end or 'live'}"
    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached["vix"]

    raw = yf.download("^VIX", start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].rename("vix").dropna()
    s.index.name = "date"
    cache.save(key, s.to_frame())
    return s
