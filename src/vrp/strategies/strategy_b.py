"""Strategy B — synthetic monthly put-writer on SPX.

Approximate Black-Scholes simulation of the CBOE PUT index methodology:
each calendar month, sell a 30-day SPX put at target delta, mark to
market daily, settle at month-end. This is a *pedagogical* replication.
Meaningful divergence from the PUT index is expected because:
- VIX is used as the ATM IV proxy (a variance-swap construct, not strict ATM IV).
- Monthly option cycles align to calendar-month ends rather than the
  third-Friday SPX monthly expiry used by CBOE.
- Risk-free rate is held at zero.

References:
- CBOE White Paper on PUT Index methodology.
- Bondarenko (2014). Why Are Put Options So Expensive?
- Israelov & Nielsen (2015). Covered Calls Uncovered (AQR). Implementation
  realism critique of naive option-writing backtests.

The correlation-against-PUT sanity gate (see scripts/run_strategy_b_synthetic.py)
requires monthly-return correlation >= 0.6. This engine is documented as
approximate and is NOT the primary Strategy B deliverable — that is the
CBOE PUT index analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import pandas as pd

from vrp.util.bs import bs_price, strike_from_delta


Variant = Literal["naked", "spread"]


@dataclass
class StrategyBConfig:
    target_delta: float = -0.30
    long_put_delta: Optional[float] = None
    maturity_days: int = 30
    tc_pct_of_premium: float = 0.05
    r: float = 0.0


def _month_starts(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    df = pd.Series(index=idx, data=1)
    return list(df.groupby(df.index.to_period("M")).apply(lambda s: s.index[0]))


def _mark_leg(S_series: pd.Series, K: float, sigma_series: pd.Series,
              days_to_expiry: pd.Series, r: float) -> pd.Series:
    """Daily mark of a single long-put leg at quantity +1."""
    marks = []
    for S, sigma_pct, dte in zip(S_series.values, sigma_series.values,
                                  days_to_expiry.values):
        T = max(float(dte), 0.0) / 365.0
        sigma = float(sigma_pct) / 100.0
        if T <= 0 or sigma <= 0:
            marks.append(max(K - float(S), 0.0))
            continue
        marks.append(bs_price(float(S), K, T, sigma, r=r, option_type="put"))
    return pd.Series(marks, index=S_series.index)


def run_strategy_b(spx: pd.Series, vix: pd.Series,
                   target_delta: float = -0.30,
                   long_put_delta: Optional[float] = None,
                   maturity_days: int = 30,
                   tc_pct_of_premium: float = 0.05,
                   r: float = 0.0) -> Dict[str, object]:
    """Simulate monthly short-put writing on SPX with VIX as IV proxy.

    Parameters
    ----------
    spx, vix:
        Aligned daily price and VIX-spot series.
    target_delta:
        Short-put delta target (e.g. -0.30). Must be in (-1, 0).
    long_put_delta:
        If not None, enables spread variant: buy a further-OTM put of
        this delta (e.g. -0.10). Must be between target_delta and 0.
    maturity_days:
        Time-to-expiry in calendar days at option open.
    tc_pct_of_premium:
        Round-trip transaction cost as a fraction of premium collected.
    r:
        Risk-free rate (annualized). Default 0.

    Returns
    -------
    Dict with keys ``daily_return``, ``positions`` (DataFrame),
    ``monthly_pnl`` (DataFrame).
    """
    if not (-1.0 < target_delta < 0.0):
        raise ValueError(f"target_delta must be in (-1, 0), got {target_delta}")
    if long_put_delta is not None and not (target_delta < long_put_delta < 0.0):
        raise ValueError(
            f"long_put_delta ({long_put_delta}) must be between target_delta "
            f"({target_delta}) and 0"
        )

    cfg = StrategyBConfig(target_delta=target_delta,
                          long_put_delta=long_put_delta,
                          maturity_days=maturity_days,
                          tc_pct_of_premium=tc_pct_of_premium, r=r)

    df = pd.concat([spx.rename("S"), vix.rename("vix")], axis=1).dropna()
    idx = df.index
    month_start_dates = _month_starts(idx)

    daily_pnl = pd.Series(0.0, index=idx)
    positions = []
    monthly_pnl = []

    for i, open_date in enumerate(month_start_dates):
        close_date = (month_start_dates[i + 1]
                      if i + 1 < len(month_start_dates) else idx[-1])
        cycle_idx = idx[(idx >= open_date) & (idx < close_date)]
        if len(cycle_idx) < 2:
            continue

        S0 = float(df.loc[open_date, "S"])
        sigma0 = float(df.loc[open_date, "vix"]) / 100.0
        T0 = cfg.maturity_days / 365.0

        K_short = strike_from_delta(S0, T0, sigma0, cfg.r,
                                    option_type="put",
                                    target_delta=cfg.target_delta)
        p_short_open = bs_price(S0, K_short, T0, sigma0, cfg.r, "put")

        K_long = None
        p_long_open = 0.0
        if cfg.long_put_delta is not None:
            K_long = strike_from_delta(S0, T0, sigma0, cfg.r,
                                       option_type="put",
                                       target_delta=cfg.long_put_delta)
            p_long_open = bs_price(S0, K_long, T0, sigma0, cfg.r, "put")

        premium_collected = p_short_open - p_long_open

        dte = pd.Series(
            [max(cfg.maturity_days - (d - open_date).days, 0) for d in cycle_idx],
            index=cycle_idx,
        )

        short_marks = _mark_leg(df.loc[cycle_idx, "S"], K_short,
                                df.loc[cycle_idx, "vix"], dte, cfg.r)
        long_marks = pd.Series(0.0, index=cycle_idx)
        if K_long is not None:
            long_marks = _mark_leg(df.loc[cycle_idx, "S"], K_long,
                                   df.loc[cycle_idx, "vix"], dte, cfg.r)

        # Short position: collected premium, value declines as marks rise.
        # Long position: paid premium, value rises as marks rise.
        # Equity contribution (relative to open) at time t:
        #   (p_short_open - short_marks_t) + (long_marks_t - p_long_open)
        position_value = (p_short_open - short_marks) + (long_marks - p_long_open)

        tc_open = cfg.tc_pct_of_premium * 0.5 * abs(premium_collected)
        tc_close = cfg.tc_pct_of_premium * 0.5 * abs(premium_collected)
        position_value.iloc[0] -= tc_open
        position_value.iloc[-1] -= tc_close

        # PnL is denominated on capital = K_short (cash-secured).
        position_return = position_value / K_short

        cycle_daily_pnl = position_return.diff().fillna(position_return.iloc[0])
        daily_pnl.loc[cycle_idx] = cycle_daily_pnl.values

        positions.append({
            "open_date": open_date,
            "close_date": close_date,
            "S0": S0, "sigma0": sigma0,
            "K_short": K_short, "K_long": K_long,
            "p_short_open": p_short_open, "p_long_open": p_long_open,
            "premium_collected": premium_collected,
            "tc_total": tc_open + tc_close,
        })
        monthly_pnl.append({
            "month_start": open_date,
            "return": float(position_return.iloc[-1]),
        })

    return {
        "daily_return": daily_pnl,
        "positions": pd.DataFrame(positions),
        "monthly_pnl": pd.DataFrame(monthly_pnl),
    }
