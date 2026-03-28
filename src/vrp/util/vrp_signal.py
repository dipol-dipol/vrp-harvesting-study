"""Volatility Risk Premium signal.

The VRP at time t is defined as IV_t - RV_t, both expressed in vol points
(percentage points of annualized volatility). IV is the 30-day at-the-
money implied vol; in this study we use VIX as an IV proxy (VIX is a
variance-swap construct rather than strict ATM IV, so the proxy is
directionally correct but biases VRP estimates during skew-heavy
regimes; flagged in the Phase 3 README).

RV is typically a 20-day close-to-close realized vol from
vrp.util.vol.close_to_close_rv. VIX is supplied in vol points (20.0 means
20 vol); RV comes back as a decimal (0.15 means 15%). This function
multiplies RV by 100 before subtracting so the returned VRP is in
vol points.

References:
- Carr, Wu (2009) "Variance Risk Premiums"
- Bondarenko (2014) "Why Are Put Options So Expensive?"
- Dew-Becker et al. (2017) "The Price of Variance Risk"
"""
from __future__ import annotations

import pandas as pd


def compute_vrp(vix_pct: pd.Series, rv_decimal: pd.Series) -> pd.Series:
    """VRP = IV - RV in vol points, aligned on the index intersection."""
    aligned = pd.concat(
        [vix_pct.rename("iv"), (rv_decimal * 100.0).rename("rv_pct")],
        axis=1, join="inner",
    ).dropna()
    return (aligned["iv"] - aligned["rv_pct"]).rename("vrp")


def month_end_signal(series: pd.Series) -> pd.Series:
    """Pick the value on the last trading day of each calendar month.

    Returns a DatetimeIndex-indexed series (not PeriodIndex).
    """
    grouped = series.groupby(series.index.to_period("M"))
    last_dates = grouped.apply(lambda s: s.index[-1])
    last_values = grouped.last()
    last_values.index = pd.DatetimeIndex(last_dates.values)
    return last_values.sort_index()
