"""Performance metrics for backtest return series.

Inputs are pandas Series of simple daily returns indexed by trading date.
Drawdown functions operate on the equity curve derived via (1+r).cumprod().
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from vrp.util.annualize import TRADING_DAYS, ann_return, ann_vol


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = _equity_curve(returns)
    peak = eq.cummax()
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def drawdown_duration_days(returns: pd.Series) -> int:
    """Longest drawdown duration: bars from first underwater bar to recovery."""
    dd = drawdown_series(returns)
    longest = current = 0
    in_dd = False
    for v in dd.values:
        if v < 0:
            in_dd = True
            current += 1
            longest = max(longest, current)
        elif in_dd and v >= 0.0:
            current += 1
            longest = max(longest, current)
            in_dd = False
            current = 0
        else:
            in_dd = False
            current = 0
    return int(longest)


def sortino_ratio(returns: pd.Series, target: float = 0.0) -> float:
    """Annualized Sortino. target is the daily MAR (minimum acceptable return)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    downside = r[r < target]
    if len(downside) == 0:
        return float("inf")
    downside_std = float(np.sqrt(((downside - target) ** 2).mean()))
    if downside_std == 0:
        return float("inf")
    mu = r.mean() - target
    return float((mu / downside_std) * np.sqrt(TRADING_DAYS))


def distribution_stats(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    return {
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "p01": float(r.quantile(0.01)),
        "p05": float(r.quantile(0.05)),
        "p95": float(r.quantile(0.95)),
        "p99": float(r.quantile(0.99)),
    }


def summary(returns: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """One-stop summary. Does NOT include benchmark comparisons."""
    mu = ann_return(returns)
    sig = ann_vol(returns)
    sharpe = (mu - rf) / sig if (sig and not np.isnan(sig) and sig > 0) else float("nan")
    return {
        "ann_return": mu,
        "ann_vol": sig,
        "sharpe": sharpe,
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "drawdown_duration_days": drawdown_duration_days(returns),
        **distribution_stats(returns),
    }
