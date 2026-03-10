import pandas as pd

from vrp.data.vx_futures import vx_expiration, build_roll_calendar


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
