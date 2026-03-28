"""Strategy C — conditional VRP harvester.

Wraps Strategy B with a VRP-gated position sizing. On each month-open
trading day, consult the VRP signal (IV_t - RV_t in vol points). If the
signal is at or above the threshold, the Strategy B position for that
month is taken; otherwise the month is held in cash (zero daily return).

Hypothesis (from Carr & Wu 2009, Bondarenko 2014): the VRP is time-
varying, so avoiding low-VRP periods should improve risk-adjusted
returns even at the cost of lower total premium.

Important: threshold selection must happen on the training window only.
The test window is evaluated once at the chosen threshold.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from vrp.strategies.strategy_b import run_strategy_b, _month_starts


def run_strategy_c(spx: pd.Series, vix: pd.Series, vrp: pd.Series,
                   threshold: float,
                   target_delta: float = -0.30,
                   long_put_delta: Optional[float] = None,
                   maturity_days: int = 30,
                   tc_pct_of_premium: float = 0.05,
                   r: float = 0.0) -> Dict[str, object]:
    """Conditional put-writer. See run_strategy_b for non-gating parameters.

    Parameters
    ----------
    vrp:
        VRP signal in vol points, indexed by trading date. Must cover
        at least the month-start dates of the backtest window.
    threshold:
        Minimum VRP to take a position for the month. Below threshold,
        the month is held in cash.
    """
    base = run_strategy_b(spx, vix, target_delta=target_delta,
                           long_put_delta=long_put_delta,
                           maturity_days=maturity_days,
                           tc_pct_of_premium=tc_pct_of_premium, r=r)
    daily_return = base["daily_return"].copy()

    aligned = pd.concat([spx.rename("S"), vix.rename("vix")],
                         axis=1).dropna()
    month_start_dates = _month_starts(aligned.index)

    active_count = 0
    gated_positions = []
    for i, open_date in enumerate(month_start_dates):
        close_date = (month_start_dates[i + 1]
                       if i + 1 < len(month_start_dates) else aligned.index[-1])
        cycle_idx = aligned.index[(aligned.index >= open_date)
                                   & (aligned.index < close_date)]
        if len(cycle_idx) < 2:
            continue
        vrp_history = vrp.loc[:open_date].dropna()
        if len(vrp_history) == 0:
            daily_return.loc[cycle_idx] = 0.0
            gated_positions.append({"open_date": open_date, "vrp": None,
                                     "active": False})
            continue
        vrp_value = float(vrp_history.iloc[-1])
        active = vrp_value >= threshold
        if not active:
            daily_return.loc[cycle_idx] = 0.0
        else:
            active_count += 1
        gated_positions.append({"open_date": open_date, "vrp": vrp_value,
                                 "active": active})

    active_fraction = (active_count / len(gated_positions)
                        if gated_positions else 0.0)

    return {
        "daily_return": daily_return,
        "positions": base["positions"],
        "monthly_pnl": base["monthly_pnl"],
        "gating": pd.DataFrame(gated_positions),
        "active_months_fraction": active_fraction,
        "threshold": float(threshold),
    }
