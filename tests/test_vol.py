import numpy as np
import pandas as pd

from vrp.util.vol import close_to_close_rv, rolling_rv


def _constant_log_return_series(n=1000, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, sigma, n)
    prices = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(prices, index=idx)


def test_close_to_close_recovers_sigma():
    prices = _constant_log_return_series(n=5000, sigma=0.01, seed=42)
    rv = close_to_close_rv(prices, window=5000)
    last = rv.dropna().iloc[-1]
    expected = 0.01 * np.sqrt(252)
    assert abs(last - expected) / expected < 0.05


def test_rolling_rv_shape_and_lookahead():
    prices = _constant_log_return_series(n=200)
    rv20 = rolling_rv(prices, window=20)
    assert rv20.iloc[:19].isna().all()
    assert not np.isnan(rv20.iloc[-1])


def test_rolling_rv_uses_log_returns():
    prices = pd.Series(
        100 * np.exp(np.linspace(0, 0.02 * 100, 101)),
        index=pd.bdate_range("2020-01-02", periods=101),
    )
    rv = rolling_rv(prices, window=20)
    assert abs(rv.iloc[-1]) < 1e-9
