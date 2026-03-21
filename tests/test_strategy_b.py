import numpy as np
import pandas as pd
import pytest

from vrp.strategies.strategy_b import run_strategy_b


def _synth_spx_and_vix(n_days: int = 252, spx_level: float = 100.0,
                       vix_level: float = 20.0):
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    spx = pd.Series(spx_level + np.zeros(n_days), index=idx)
    vix = pd.Series(vix_level + np.zeros(n_days), index=idx)
    return spx, vix


def test_strategy_b_returns_series_shape():
    spx, vix = _synth_spx_and_vix(120)
    out = run_strategy_b(spx, vix, target_delta=-0.30)
    assert set(out.keys()) >= {"daily_return", "positions", "monthly_pnl"}
    assert len(out["daily_return"]) == len(spx)


def test_strategy_b_profitable_on_flat_underlying():
    # Flat SPX: puts expire worthless every month, premium is collected.
    spx, vix = _synth_spx_and_vix(252)
    out = run_strategy_b(spx, vix, target_delta=-0.30,
                         tc_pct_of_premium=0.05)
    assert out["daily_return"].sum() > 0


def test_strategy_b_losses_on_crash():
    # SPX drops 20% in one day mid-month -> short puts go deep ITM.
    n = 60
    idx = pd.bdate_range("2020-01-02", periods=n)
    spx = pd.Series(100.0, index=idx)
    spx.iloc[n // 2:] = 80.0
    vix = pd.Series(25.0, index=idx)
    out = run_strategy_b(spx, vix, target_delta=-0.30, tc_pct_of_premium=0.0)
    assert (1 + out["daily_return"]).cumprod().iloc[-1] < 0.95


def test_strategy_b_invalid_delta_raises():
    spx, vix = _synth_spx_and_vix(60)
    with pytest.raises(ValueError):
        run_strategy_b(spx, vix, target_delta=0.30)
    with pytest.raises(ValueError):
        run_strategy_b(spx, vix, target_delta=-1.5)
