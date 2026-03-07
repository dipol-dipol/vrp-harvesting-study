"""Historical CBOE benchmark-option indices.

- PUT: CBOE S&P 500 PutWrite Index. Published version of monthly ATM cash-secured
  put-writing strategy. Benchmark for Strategy B.
- BXM: CBOE S&P 500 BuyWrite Index. Monthly covered-call benchmark; used as
  a secondary option-strategy benchmark.

Data source: CBOE published historical CSVs. URLs change periodically. Pinned
URL is in _URLS below and must be revalidated if a fetch fails.
"""
from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from . import cache

_URLS = {
    "PUT": "https://cdn.cboe.com/api/global/us_indices/daily_prices/PUT_History.csv",
    "BXM": "https://cdn.cboe.com/api/global/us_indices/daily_prices/BXM_History.csv",
}


def load_cboe_index(name: str, use_cache: bool = True) -> pd.Series:
    name = name.upper()
    if name not in _URLS:
        raise KeyError(f"unknown CBOE index '{name}'. supported: {list(_URLS)}")
    key = f"cboe_{name}_daily"
    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached[name.lower()]
    r = requests.get(_URLS[name], timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = "date" if "date" in df.columns else df.columns[0]
    price_col = name.lower() if name.lower() in df.columns else \
        ("close" if "close" in df.columns else df.columns[1])
    df["date"] = pd.to_datetime(df[date_col])
    s = df.set_index("date")[price_col].astype(float).rename(name.lower())
    s = s.sort_index().dropna()
    cache.save(key, s.to_frame())
    return s
