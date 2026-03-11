import numpy as np
import pandas as pd

from vrp.data.vx_futures import vx_expiration, build_roll_calendar
from vrp.strategies.strategy_a import run_strategy_a


def test_vx_expiration_known_dates():
    # 3rd Friday of Feb 2018 is Feb 16; SOQ Wed = 30 days before = Jan 17
    assert vx_expiration(2018, 1) == pd.Timestamp("2018-01-17")
    # 3rd Friday of Jan 2020 is Jan 17; 30 days before = Dec 18 2019
    assert vx_expiration(2019, 12) == pd.Timestamp("2019-12-18")


def test_build_roll_calendar_columns_and_order():
    cal = build_roll_calendar(start="2018-01-01", end="2018-06-30")
    assert list(cal.columns) == ["front_expiry", "second_expiry"]
    # calendar is monotonically non-decreasing in front_expiry
    diffs = cal["front_expiry"].diff().dropna()
    assert (diffs >= pd.Timedelta(0)).all()
    # second expiry always strictly greater than front
    assert (cal["second_expiry"] > cal["front_expiry"]).all()


def _synth_vx_series(front_start: float = 16.0, front_end: float = 14.0,
                     second_const: float = 16.0, n: int = 100):
    """n business days with the front decaying linearly front_start -> front_end
    while second stays at second_const. Models pure contango carry: short-front
    should profit as front drops toward spot.
    """
    idx = pd.bdate_range("2020-01-02", periods=n)
    front = np.linspace(front_start, front_end, n)
    second = np.full(n, second_const)
    # days_to_front_expiry declines from n to 1
    dte = np.arange(n, 0, -1)
    return pd.DataFrame({
        "front_settle": front,
        "second_settle": second,
        "front_expiry": idx[-1] + pd.Timedelta(days=1),
        "second_expiry": idx[-1] + pd.Timedelta(days=30),
        "days_to_front_expiry": dte,
    }, index=idx)


def test_strategy_a_positive_when_front_decays_to_spot():
    df = _synth_vx_series()
    result = run_strategy_a(df, tc_bps_per_roll=0)
    # Short front drops -> positive PnL on the short leg; second flat -> zero
    # on the long leg. Dollar-neutral construction means net PnL is positive.
    assert result["daily_pnl"].sum() > 0


def test_strategy_a_returns_series_shape():
    df = _synth_vx_series()
    result = run_strategy_a(df)
    for k in ("daily_pnl", "daily_return", "positions"):
        assert k in result
    assert len(result["daily_return"]) == len(df)


def test_strategy_a_roll_mask_triggers_near_expiry():
    df = _synth_vx_series(n=30)
    result = run_strategy_a(df, roll_days_before_expiry=5, tc_bps_per_roll=1.0)
    # Last 5 rows should be flagged as roll days (dte <= 5)
    assert result["positions"]["is_roll_day"].iloc[-5:].all()
    # Earlier rows should not be flagged
    assert not result["positions"]["is_roll_day"].iloc[:20].any()
