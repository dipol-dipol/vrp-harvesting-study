import numpy as np
import pandas as pd

from vrp.report.metrics import (
    sortino_ratio,
    max_drawdown,
    drawdown_series,
    drawdown_duration_days,
    distribution_stats,
)


def test_max_drawdown_simple():
    returns = pd.Series([0.2, -0.10, 0.3888888888])
    mdd = max_drawdown(returns)
    assert abs(mdd - (-0.10)) < 1e-9


def test_drawdown_series_monotone_to_zero():
    returns = pd.Series([0.1, 0.05, -0.2, 0.3, 0.05])
    dd = drawdown_series(returns)
    assert dd.iloc[0] == 0.0
    assert dd.min() < 0
    assert dd.iloc[-1] <= 0


def test_drawdown_duration_days():
    returns = pd.Series([0.0, 0.10, -0.05, -0.01, 0.07])
    assert drawdown_duration_days(returns) == 3


def test_drawdown_duration_closes_on_overshoot():
    # Recovery bar overshoots the prior peak; duration still closes the
    # drawdown window. Previously buggy when the recovery branch used
    # `v == 0.0` strict equality.
    returns = pd.Series([0.0, 0.10, -0.05, -0.01, 0.16015])
    assert drawdown_duration_days(returns) == 3


def test_sortino_positive_when_upside_dominates():
    rng = np.random.default_rng(0)
    up = rng.normal(0.002, 0.005, 500)
    down = rng.normal(-0.001, 0.005, 500)
    rets = pd.Series(np.concatenate([up, down]))
    s = sortino_ratio(rets, target=0.0)
    assert s > 0


def test_distribution_stats_keys():
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1000))
    d = distribution_stats(rets)
    for k in ("skew", "kurtosis", "p01", "p05", "p95", "p99"):
        assert k in d
    assert d["p01"] < d["p05"] < d["p95"] < d["p99"]
