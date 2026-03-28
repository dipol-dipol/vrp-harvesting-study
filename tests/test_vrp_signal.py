import pandas as pd

from vrp.util.vrp_signal import compute_vrp, month_end_signal


def _constant_vix_rv_series(n=252, vix_pct=20.0, rv_pct=15.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    vix = pd.Series(vix_pct, index=idx)
    rv = pd.Series(rv_pct / 100.0, index=idx)
    return vix, rv


def test_compute_vrp_constant_case():
    vix, rv = _constant_vix_rv_series(vix_pct=20.0, rv_pct=15.0)
    vrp = compute_vrp(vix, rv)
    assert (vrp == 5.0).all()


def test_compute_vrp_aligns_by_index():
    idx = pd.bdate_range("2020-01-02", periods=10)
    vix = pd.Series(20.0, index=idx)
    rv = pd.Series(0.15, index=idx[2:])
    vrp = compute_vrp(vix, rv)
    assert vrp.index.equals(rv.index)
    assert (vrp == 5.0).all()


def test_month_end_signal_picks_last_trading_day_per_month():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    vrp = pd.Series(range(len(idx)), index=idx).astype(float)
    mes = month_end_signal(vrp)
    assert len(mes) == 3
    assert mes.index[0] == pd.Timestamp("2020-01-31")
    assert mes.index[1] == pd.Timestamp("2020-02-28")
    assert mes.index[2] == pd.Timestamp("2020-03-31")
