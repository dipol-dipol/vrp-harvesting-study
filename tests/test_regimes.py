import pandas as pd

from vrp.report.regimes import REGIMES, slice_regime, regime_metrics


def test_named_regimes_cover_required_events():
    names = {r.name for r in REGIMES}
    for required in ("gfc_2008", "vol_spike_2015", "volmageddon_2018",
                     "covid_2020", "bear_2022"):
        assert required in names


def test_slice_regime_returns_subset():
    idx = pd.bdate_range("2018-01-01", "2018-03-01")
    s = pd.Series(range(len(idx)), index=idx)
    sliced = slice_regime(s, "volmageddon_2018")
    assert sliced.index.min() >= pd.Timestamp("2018-02-01")
    assert sliced.index.max() <= pd.Timestamp("2018-02-28")


def test_regime_metrics_returns_all_named():
    idx = pd.bdate_range("2006-01-01", "2024-12-31")
    s = pd.Series(0.0001, index=idx)
    out = regime_metrics(s)
    assert set(out.keys()) >= {"gfc_2008", "vol_spike_2015",
                               "volmageddon_2018", "covid_2020", "bear_2022"}
