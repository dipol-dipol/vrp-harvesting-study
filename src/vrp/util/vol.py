"""Realized volatility on a price series.

Close-to-close realized vol, log-return based, annualized with sqrt(252).
Standard reference in Carr & Wu (2009) and downstream VRP literature.
Alternatives (Parkinson, Garman-Klass) require OHLC data and are deferred
to later phases if needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .annualize import TRADING_DAYS


def _log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


def close_to_close_rv(prices: pd.Series, window: int) -> pd.Series:
    """Rolling close-to-close realized volatility, annualized.

    window is in trading days. Uses population std (ddof=0) per CBOE's
    realized-vol convention; change to ddof=1 only if matching a specific
    paper that uses sample std.
    """
    lr = _log_returns(prices)
    # min_periods is window - 1 because the first log return is always NaN
    # (no prior price), so a W-bar window contains at most W - 1 observations.
    rv = lr.rolling(window=window, min_periods=window - 1).std(ddof=0)
    return rv * np.sqrt(TRADING_DAYS)


def rolling_rv(prices: pd.Series, window: int = 20) -> pd.Series:
    """Convenience wrapper: 20-day realized vol by default (matches spec)."""
    return close_to_close_rv(prices, window=window)
