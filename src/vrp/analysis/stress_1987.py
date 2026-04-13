"""October 1987 stress test — extrapolated single-day scenario.

Black Monday 1987 hard numbers (historical record):
- S&P 500: -20.5% single-day drop
- VIX did not exist in 1987; 30-day realized vol spiked from ~15 to ~60
  in the days around the event. Hull (2017) and academic
  reconstructions estimate a "VIX-equivalent" jump of +25 to +35 vol
  points on Oct 19.

We model the scenario as:
- Front-month VX: +30 vol points (larger response)
- Second-month VX: +20 vol points (smaller, typical term-structure
  compression in a stress event)
- SPX spot: -20.5%
- ATM IV (VIX): from ~20 to ~50

The goal is illustrative, not predictive. Strategy-A-style short-vol
constructions would have been wiped out; hedged put-spread constructions
would have been injured but survived.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional

from vrp.util.bs import bs_price


STRESS_SPX_DROP = -0.205
STRESS_VIX_JUMP = 30.0


def stress_calendar(front_0: float, second_0: float,
                    delta_front: float, delta_second: float,
                    direction: Literal["short_front", "long_front"] = "short_front",
                    gross_notional_per_leg: float = 1.0) -> Dict[str, float]:
    """Single-day PnL of a dollar-neutral VX calendar under the stress."""
    front_qty = gross_notional_per_leg / front_0
    second_qty = gross_notional_per_leg / second_0
    if direction == "short_front":
        front_sign, second_sign = -1.0, +1.0
    else:
        front_sign, second_sign = +1.0, -1.0
    pnl = (front_sign * front_qty * delta_front
            + second_sign * second_qty * delta_second)
    gross = 2.0 * gross_notional_per_leg
    return {
        "direction": direction,
        "daily_pnl": float(pnl),
        "daily_pnl_return": float(pnl / gross),
        "front_change": float(delta_front),
        "second_change": float(delta_second),
    }


def stress_put_writer(S0: float, K_short: float, sigma0_pct: float,
                      S_shock: float, sigma_shock_pct: float,
                      T_remaining_days: int,
                      K_long: Optional[float],
                      premium_collected: float,
                      r: float = 0.0) -> Dict[str, float]:
    """Single-day MTM hit to a cash-secured put writer (with optional long leg)."""
    T = T_remaining_days / 365.0
    sigma0 = sigma0_pct / 100.0
    sigma_shock = sigma_shock_pct / 100.0

    short_pre = bs_price(S0, K_short, T, sigma0, r, "put")
    short_post = bs_price(S_shock, K_short, T, sigma_shock, r, "put")
    short_pnl = -(short_post - short_pre)

    long_pnl = 0.0
    long_pre = long_post = 0.0
    if K_long is not None:
        long_pre = bs_price(S0, K_long, T, sigma0, r, "put")
        long_post = bs_price(S_shock, K_long, T, sigma_shock, r, "put")
        long_pnl = long_post - long_pre

    total_pnl = short_pnl + long_pnl
    return {
        "short_pre": float(short_pre),
        "short_post": float(short_post),
        "long_pre": float(long_pre),
        "long_post": float(long_post),
        "premium_collected": float(premium_collected),
        "pnl": float(total_pnl),
        "pnl_return": float(total_pnl / K_short),
    }
