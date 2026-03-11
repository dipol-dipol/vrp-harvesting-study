"""Strategy A — VIX futures term-structure carry (dollar-neutral).

Short front-month VX, long second-month VX, dollar-neutral at close. Roll
5 trading days before front-month expiration. See:
- Whaley (2013) "Trading Volatility: At What Cost?" for VX roll dynamics.
- Alexander, Korovilas (2013) for a critique of naive VX carry.

This implementation targets gross notional = $1 per side (so $2 gross).
PnL is in "dollars" of that unit — interpret as a return series on $1 of
capital per leg. Daily return = daily PnL / (notional gross = 2) for
reporting symmetry with other strategies' return series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


ROLL_DAYS_BEFORE_EXPIRY = 5


@dataclass
class StrategyAConfig:
    roll_days_before_expiry: int = ROLL_DAYS_BEFORE_EXPIRY
    tc_bps_per_roll: float = 1.0
    gross_notional_per_leg: float = 1.0


def _is_roll_day(days_to_front_expiry: pd.Series, roll_days: int) -> pd.Series:
    return days_to_front_expiry <= roll_days


def run_strategy_a(vx: pd.DataFrame,
                   roll_days_before_expiry: int = ROLL_DAYS_BEFORE_EXPIRY,
                   tc_bps_per_roll: float = 1.0,
                   gross_notional_per_leg: float = 1.0) -> Dict[str, pd.Series]:
    """Run Strategy A on a continuous VX front/second DataFrame.

    Inputs:
        vx: index=date, columns include front_settle, second_settle,
            front_expiry, days_to_front_expiry
    Outputs dict with:
        daily_pnl: per-day dollar PnL (on 2*gross_notional gross capital)
        daily_return: daily return in units of 2*gross_notional capital
        positions: DataFrame of short_front / long_second notionals
    """
    cfg = StrategyAConfig(roll_days_before_expiry=roll_days_before_expiry,
                          tc_bps_per_roll=tc_bps_per_roll,
                          gross_notional_per_leg=gross_notional_per_leg)

    df = vx.copy().sort_index()
    short_qty = cfg.gross_notional_per_leg / df["front_settle"]
    long_qty = cfg.gross_notional_per_leg / df["second_settle"]

    # Position taken at close of day t earns PnL from t -> t+1. diff().shift(-1)
    # aligns (settle_{t+1} - settle_t) to t; after PnL is computed at t it is
    # shifted by +1 to land on the date the PnL is recognized (t+1).
    d_front = df["front_settle"].diff().shift(-1)
    d_second = df["second_settle"].diff().shift(-1)
    daily_pnl = (-short_qty * d_front) + (long_qty * d_second)
    daily_pnl = daily_pnl.shift(1).fillna(0.0)

    roll_mask = _is_roll_day(df["days_to_front_expiry"], cfg.roll_days_before_expiry)
    tc = roll_mask.astype(float) * (cfg.tc_bps_per_roll * 1e-4) * (
        2 * cfg.gross_notional_per_leg
    )
    daily_pnl = daily_pnl - tc

    gross = 2.0 * cfg.gross_notional_per_leg
    daily_return = daily_pnl / gross

    positions = pd.DataFrame({
        "short_front_qty": short_qty,
        "long_second_qty": long_qty,
        "is_roll_day": roll_mask.astype(bool),
    }, index=df.index)

    return {
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "positions": positions,
    }
