"""Named regime windows for VRP strategy analysis.

Windows are approximate crisis/vol-spike bands chosen to match the 'regime
performance' block from the spec. Widen or narrow with sensitivity analysis
in the writeup — do not tune them to improve results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .metrics import summary


@dataclass(frozen=True)
class Regime:
    name: str
    start: str  # inclusive
    end: str    # inclusive
    note: str


REGIMES: List[Regime] = [
    Regime("gfc_2008", "2008-09-01", "2009-03-31",
           "Global Financial Crisis — Lehman through March '09 low."),
    Regime("vol_spike_2015", "2015-08-17", "2015-09-30",
           "August 2015 China-driven flash crash."),
    Regime("volmageddon_2018", "2018-02-01", "2018-02-28",
           "Short-vol ETN blowup."),
    Regime("covid_2020", "2020-02-20", "2020-04-30",
           "COVID crash and initial recovery."),
    Regime("bear_2022", "2022-01-01", "2022-10-31",
           "2022 rates-driven bear market."),
]


def slice_regime(series: pd.Series, name: str) -> pd.Series:
    regime = next(r for r in REGIMES if r.name == name)
    return series.loc[regime.start:regime.end]


def regime_metrics(returns: pd.Series) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in REGIMES:
        window = returns.loc[r.start:r.end]
        if len(window) == 0:
            out[r.name] = {"n_days": 0}
            continue
        out[r.name] = {"n_days": len(window), **summary(window)}
    return out
