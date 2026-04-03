# VRP Phase 4 — Tail-Risk Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Three tail-risk overlays applied to Strategy C (conditional VRP-gated put-writer). Each overlay is implemented as a standalone transform, tested in isolation, then layered on Strategy C and evaluated with the existing train/test split. A combined runner applies all three to C-spread and reports the aggregate.

**Architecture:** New `vrp.overlays` subpackage with three pure-function transforms:
- `regime_filter.py` — VIX/term-structure regime mask (Overlay 1)
- `vol_scaling.py` — realized-vol position scaling (Overlay 2)
- `tail_hedge.py` — 5-delta tail-hedge spend (Overlay 3)

Each takes strategy outputs + the minimum data it needs and returns a transformed `daily_return` series (Overlay 3 also returns hedge-leg diagnostics). One runner script applies each overlay independently and in combination to Strategy C (spread variant, threshold=−2 train-optimal from Phase 3).

**Tech Stack:** Same as Phases 2-3. Reuses `vrp.util.bs` for the tail-hedge implementation.

**Branch:** `phase4-overlays`, branched off `phase3-strategy-c`.

**Train/Test split:** Same as prior phases.

**Key discipline rule:** Overlay parameters taken *verbatim from the spec*. No tuning:
- Overlay 1: VIX>30 / VIX<25 thresholds, 7-day re-entry confirmation, front>second inversion trigger.
- Overlay 2: target vol 10%, 20-day RV window, leverage cap 1.0.
- Overlay 3: 15% of premium, 5-delta put hedge, 1-month maturity.

If an overlay underperforms the unadorned strategy on the test window, report the loss honestly — this is a test of the spec's tail-risk rules, not a fit.

---

## File Structure

```
src/vrp/
└── overlays/
    ├── __init__.py
    ├── regime_filter.py                # Overlay 1
    ├── vol_scaling.py                  # Overlay 2
    └── tail_hedge.py                   # Overlay 3
scripts/
└── run_strategy_c_overlays.py          # applies each overlay + combined
tests/
├── test_overlay_regime_filter.py
├── test_overlay_vol_scaling.py
└── test_overlay_tail_hedge.py
```

---

## Task 1: Overlay 1 — VIX Regime Filter

**Files:**
- Create: `src/vrp/overlays/__init__.py` (empty)
- Create: `src/vrp/overlays/regime_filter.py`
- Create: `tests/test_overlay_regime_filter.py`

### Spec

- Go to cash when VIX spot > 30 OR when VIX term structure inverts (front-month VX > second-month VX).
- Re-enter when VIX < 25 AND term structure is back in contango (front < second), confirmed over 7 consecutive days (anti-whipsaw).

The mask is a daily boolean series. Apply to strategy returns by zeroing out daily returns on inactive days.

- [ ] **Step 1.1: Write failing test**

`tests/test_overlay_regime_filter.py`:

```python
import numpy as np
import pandas as pd

from vrp.overlays.regime_filter import vix_regime_mask, apply_mask


def _synth_data(n=100, vix_level=15.0, front=15.0, second=16.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(vix_level, index=idx),
            pd.Series(front, index=idx),
            pd.Series(second, index=idx))


def test_mask_all_active_in_quiet_regime():
    vix, front, second = _synth_data(vix_level=15.0)
    mask = vix_regime_mask(vix, front, second)
    assert mask.all()


def test_mask_turns_off_when_vix_spikes():
    vix, front, second = _synth_data(n=50, vix_level=15.0)
    vix.iloc[20:30] = 35.0  # Spike above 30 for a window
    mask = vix_regime_mask(vix, front, second)
    # During the spike: inactive
    assert not mask.iloc[20:30].any()
    # After re-entry (vix drops below 25), requires 7-day confirmation
    # So bars 30-36 should still be inactive, bar 37+ active again
    assert not mask.iloc[30:36].any()
    assert mask.iloc[37:].all()


def test_mask_turns_off_on_backwardation():
    vix, front, second = _synth_data(n=50)
    # Invert term structure for a window
    front.iloc[10:20] = 20.0
    second.iloc[10:20] = 17.0
    mask = vix_regime_mask(vix, front, second)
    assert not mask.iloc[10:20].any()


def test_apply_mask_zeros_daily_returns():
    idx = pd.bdate_range("2020-01-02", periods=20)
    daily_return = pd.Series(0.01, index=idx)
    mask = pd.Series([True] * 10 + [False] * 10, index=idx)
    out = apply_mask(daily_return, mask)
    assert (out.iloc[:10] == 0.01).all()
    assert (out.iloc[10:] == 0.0).all()
```

- [ ] **Step 1.2: Run test — verify FAIL**

Run: `.venv/bin/pytest tests/test_overlay_regime_filter.py -v`
Expected: `ModuleNotFoundError: No module named 'vrp.overlays.regime_filter'`.

- [ ] **Step 1.3: Write implementation**

`src/vrp/overlays/regime_filter.py`:

```python
"""Overlay 1 — VIX regime filter.

Go to cash when VIX spot > 30 OR VIX term structure inverts
(front-month VX > second-month VX). Re-enter when VIX < 25 AND term
structure is back in contango, confirmed for 7 consecutive trading days
(anti-whipsaw).

Rule is hand-picked from the project spec; not tuned to data.
"""
from __future__ import annotations

import pandas as pd


HIGH_VIX = 30.0
LOW_VIX = 25.0
CONFIRMATION_DAYS = 7


def vix_regime_mask(vix_spot: pd.Series,
                    vx_front: pd.Series,
                    vx_second: pd.Series,
                    high_vix: float = HIGH_VIX,
                    low_vix: float = LOW_VIX,
                    confirm_days: int = CONFIRMATION_DAYS) -> pd.Series:
    """Daily boolean mask. True = allowed to trade, False = go to cash.

    State machine:
        - active → inactive when vix > high_vix OR front > second
        - inactive → active only after `confirm_days` consecutive trading
          days of vix < low_vix AND front < second
    """
    aligned = pd.concat([vix_spot.rename("vix"),
                          vx_front.rename("front"),
                          vx_second.rename("second")], axis=1).dropna()

    stress = (aligned["vix"] > high_vix) | (aligned["front"] > aligned["second"])
    calm = (aligned["vix"] < low_vix) & (aligned["front"] < aligned["second"])

    # Rolling count of consecutive calm days
    calm_streak = calm.astype(int).groupby(
        (~calm).cumsum()
    ).cumsum()

    mask = pd.Series(False, index=aligned.index)
    active = True  # start active
    for i, (is_stress, streak) in enumerate(zip(stress.values, calm_streak.values)):
        if active:
            if is_stress:
                active = False
        else:
            # inactive: need confirm_days of calm streak to re-enter
            if streak >= confirm_days:
                active = True
        mask.iloc[i] = active
    return mask


def apply_mask(daily_return: pd.Series, mask: pd.Series) -> pd.Series:
    """Zero out daily_return on bars where mask is False."""
    aligned = pd.concat([daily_return.rename("r"),
                          mask.rename("m")], axis=1)
    # Fill missing mask values with False (cash-out on unknown regime)
    aligned["m"] = aligned["m"].fillna(False).astype(bool)
    out = aligned["r"].where(aligned["m"], 0.0)
    return out.rename(daily_return.name)
```

- [ ] **Step 1.4: Run test — verify PASS**

Run: `.venv/bin/pytest tests/test_overlay_regime_filter.py -v`
Expected: 4 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/vrp/overlays/__init__.py src/vrp/overlays/regime_filter.py tests/test_overlay_regime_filter.py
git commit -m "vrp: Overlay 1 — VIX regime filter (Phase 4 Task 1)"
```

---

## Task 2: Overlay 2 — Realized-Vol Position Scaling

**Files:**
- Create: `src/vrp/overlays/vol_scaling.py`
- Create: `tests/test_overlay_vol_scaling.py`

### Spec

- Scale daily returns by `target_vol / realized_vol_20d`, with an annualized target of 10% and a leverage cap of 1.0 (no upsizing).
- Use the strategy's own return series to compute realized vol. This means the scaling responds to the strategy's volatility, not the underlying's.

- [ ] **Step 2.1: Write failing test**

`tests/test_overlay_vol_scaling.py`:

```python
import numpy as np
import pandas as pd

from vrp.overlays.vol_scaling import target_vol_scale


def test_scale_down_when_realized_exceeds_target():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-02", periods=500)
    # Annualized vol of this series ≈ 0.02 * sqrt(252) ≈ 31.7%
    ret = pd.Series(rng.normal(0, 0.02, 500), index=idx)
    scaled = target_vol_scale(ret, target_vol=0.10, window=20,
                              leverage_cap=1.0)
    # After the warmup window, scale should be < 1
    realized_vol_late = scaled.iloc[-100:].std() * np.sqrt(252)
    assert realized_vol_late < 0.15


def test_leverage_cap_applied_when_realized_below_target():
    idx = pd.bdate_range("2020-01-02", periods=500)
    # Very quiet: realized vol much lower than target. Cap should prevent upsizing.
    ret = pd.Series(0.0001, index=idx)
    scaled = target_vol_scale(ret, target_vol=0.10, window=20,
                              leverage_cap=1.0)
    assert (scaled == ret).all()


def test_scale_preserves_zero_days():
    idx = pd.bdate_range("2020-01-02", periods=100)
    ret = pd.Series([0.01] * 30 + [0.0] * 40 + [0.01] * 30, index=idx)
    scaled = target_vol_scale(ret, target_vol=0.10, window=20,
                              leverage_cap=1.0)
    # Zero stays zero
    assert (scaled.iloc[30:70] == 0.0).all()
```

- [ ] **Step 2.2: Run test — verify FAIL**

Run: `.venv/bin/pytest tests/test_overlay_vol_scaling.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 2.3: Write implementation**

`src/vrp/overlays/vol_scaling.py`:

```python
"""Overlay 2 — realized-vol position scaling.

At each bar, scale position size by target_vol / realized_vol_trailing_N.
Cap leverage at 1.0 (no upsizing above baseline). Realized vol is
computed from the strategy's own daily returns (not the underlying).

Matches the project-spec overlay: target_vol = 10% annualized,
window = 20 trading days, leverage cap 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vrp.util.annualize import TRADING_DAYS


def target_vol_scale(daily_return: pd.Series,
                     target_vol: float = 0.10,
                     window: int = 20,
                     leverage_cap: float = 1.0) -> pd.Series:
    """Return the target-vol-scaled daily return series.

    Scale factor at time t = min(leverage_cap, target_vol / rv_{t-1}),
    using the previous bar's realized vol (no lookahead). Bars before the
    window fills have scale = leverage_cap (conservative).
    """
    r = daily_return.fillna(0.0)
    # Annualized realized vol of the strategy's own returns, shifted one
    # bar so today's scale uses information through yesterday's close.
    rv = r.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(TRADING_DAYS)
    rv = rv.shift(1)
    raw_scale = target_vol / rv.replace(0.0, np.nan)
    scale = raw_scale.clip(upper=leverage_cap).fillna(leverage_cap)
    return (r * scale).rename(daily_return.name)
```

- [ ] **Step 2.4: Run test — verify PASS**

Run: `.venv/bin/pytest tests/test_overlay_vol_scaling.py -v`
Expected: 3 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add src/vrp/overlays/vol_scaling.py tests/test_overlay_vol_scaling.py
git commit -m "vrp: Overlay 2 — realized-vol position scaling (Phase 4 Task 2)"
```

---

## Task 3: Overlay 3 — Tail-Hedge Spend

**Files:**
- Create: `src/vrp/overlays/tail_hedge.py`
- Create: `tests/test_overlay_tail_hedge.py`

### Spec

- Per option cycle in Strategy C, spend 15% of the premium collected on a long 5-delta 1-month SPX put.
- Compute daily MTM of the hedge via Black-Scholes using VIX as IV proxy, exactly like Strategy B.
- Net overlaid daily return = strategy_daily_return + hedge_daily_return.
- Report hedge cost, hedge PnL, and net effect.

The hedge spend comes *out of* the strategy's gross PnL — if we keep the full put-writing position size and then add a separately-funded hedge, that double-counts capital. Two cleanest interpretations:

**Interpretation A (implemented here):** the hedge is funded *out of* the collected premium, so the strategy's effective premium is reduced by 15%. Rather than rewriting Strategy B, we layer this post-hoc: compute hedge PnL on the same capital base (K_short) and add it to the strategy's daily return. The 15% "spend" is the debit to open the hedge position, which shows up as a negative contribution to hedge PnL on the open bar.

**Interpretation B (rejected):** pretend the capital is free. This would overstate returns.

Approach A is principled, closely matches the spec wording ("use 15% of premium collected to buy ..."), and quantifies the intended "does this reduce max drawdown more than it costs in returns?" tradeoff.

- [ ] **Step 3.1: Write failing test**

`tests/test_overlay_tail_hedge.py`:

```python
import numpy as np
import pandas as pd

from vrp.overlays.tail_hedge import add_tail_hedge
from vrp.strategies.strategy_b import run_strategy_b


def _synth_spx_vix(n=252, spx_level=100.0, vix_level=20.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(spx_level, index=idx),
            pd.Series(vix_level, index=idx))


def test_hedge_reduces_net_on_flat_underlying():
    # Flat SPX: hedge premium spent is pure cost (put expires worthless).
    # Net return should be lower than strategy alone.
    spx, vix = _synth_spx_vix(252)
    strat = run_strategy_b(spx, vix, target_delta=-0.30,
                            tc_pct_of_premium=0.0)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    assert hedged["net_daily_return"].sum() < strat["daily_return"].sum()


def test_hedge_reduces_drawdown_on_crash():
    # SPX crashes mid-sample; hedge should fire and reduce the drawdown.
    n = 60
    idx = pd.bdate_range("2020-01-02", periods=n)
    spx = pd.Series(100.0, index=idx)
    spx.iloc[n // 2:] = 80.0
    vix = pd.Series(25.0, index=idx)
    strat = run_strategy_b(spx, vix, target_delta=-0.30,
                            tc_pct_of_premium=0.0)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    strat_eq = (1 + strat["daily_return"]).cumprod()
    net_eq = (1 + hedged["net_daily_return"]).cumprod()
    # At the low point, hedged should be higher than unhedged
    low_day = strat_eq.idxmin()
    assert net_eq.loc[low_day] > strat_eq.loc[low_day]


def test_hedge_structure_keys():
    spx, vix = _synth_spx_vix(120)
    strat = run_strategy_b(spx, vix, target_delta=-0.30)
    hedged = add_tail_hedge(strat, spx, vix,
                             hedge_delta=-0.05, hedge_spend_pct=0.15)
    assert set(hedged.keys()) >= {"net_daily_return", "hedge_daily_return",
                                    "hedge_legs"}
    assert len(hedged["net_daily_return"]) == len(strat["daily_return"])
```

- [ ] **Step 3.2: Run test — verify FAIL**

Run: `.venv/bin/pytest tests/test_overlay_tail_hedge.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3.3: Write implementation**

`src/vrp/overlays/tail_hedge.py`:

```python
"""Overlay 3 — tail-hedge spend.

For each monthly option cycle in the underlying put-writing strategy,
spend `hedge_spend_pct` of the premium collected on a long 5-delta
1-month SPX put. The hedge is marked daily via Black-Scholes (same IV
proxy as the strategy).

Net daily return = strategy_daily_return + hedge_daily_return, where
both are expressed on the same capital base (K_short of the strategy's
short put). The hedge's day-0 debit shows up as a small negative
contribution on the cycle's opening bar.

Spec parameters:
    hedge_delta      = -0.05  (5-delta)
    hedge_spend_pct  = 0.15   (15% of collected premium)
    maturity_days    = 30
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from vrp.util.bs import bs_price, strike_from_delta


def add_tail_hedge(strategy_result: Dict[str, object],
                   spx: pd.Series, vix: pd.Series,
                   hedge_delta: float = -0.05,
                   hedge_spend_pct: float = 0.15,
                   maturity_days: int = 30,
                   r: float = 0.0) -> Dict[str, object]:
    """Return a dict with net_daily_return, hedge_daily_return, hedge_legs.

    strategy_result must be the dict returned by run_strategy_b or
    run_strategy_c: needs `positions` (per-cycle DataFrame with
    open_date, close_date, S0, sigma0, K_short, premium_collected)
    and `daily_return`.
    """
    if not (-1.0 < hedge_delta < 0.0):
        raise ValueError(f"hedge_delta must be in (-1, 0), got {hedge_delta}")

    positions = strategy_result["positions"]
    daily = strategy_result["daily_return"]

    aligned = pd.concat([spx.rename("S"), vix.rename("vix")],
                         axis=1).dropna()
    hedge_daily = pd.Series(0.0, index=aligned.index)
    hedge_legs = []

    for _, row in positions.iterrows():
        open_date = row["open_date"]
        close_date = row["close_date"]
        S0 = float(row["S0"])
        sigma0 = float(row["sigma0"])
        K_short = float(row["K_short"])
        premium_collected = float(row["premium_collected"])
        T0 = maturity_days / 365.0

        hedge_spend = max(0.0, hedge_spend_pct * premium_collected)
        if hedge_spend <= 0:
            continue

        K_hedge = strike_from_delta(S0, T0, sigma0, r, "put", hedge_delta)
        p_hedge_open = bs_price(S0, K_hedge, T0, sigma0, r, "put")
        if p_hedge_open <= 0:
            continue
        hedge_qty = hedge_spend / p_hedge_open

        cycle_idx = aligned.index[(aligned.index >= open_date)
                                    & (aligned.index < close_date)]
        if len(cycle_idx) < 2:
            continue

        # Daily marks
        marks = []
        for d in cycle_idx:
            dte = max(maturity_days - (d - open_date).days, 0)
            T = dte / 365.0
            sigma = float(aligned.loc[d, "vix"]) / 100.0
            if T <= 0 or sigma <= 0:
                marks.append(max(K_hedge - float(aligned.loc[d, "S"]), 0.0))
            else:
                marks.append(
                    bs_price(float(aligned.loc[d, "S"]),
                              K_hedge, T, sigma, r, "put")
                )
        marks_s = pd.Series(marks, index=cycle_idx)

        # Long position: position_value_t = hedge_qty * (mark_t - p_hedge_open)
        # Debit on open: first bar picks up (mark_0 - p_hedge_open) = 0 in theory
        # but the hedge_spend dollars are allocated at open regardless. We
        # represent the open debit as a negative contribution on the first bar
        # proportional to hedge_spend / K_short.
        pos_value = hedge_qty * (marks_s - p_hedge_open)

        # Daily PnL on hedge capital, expressed as a return on K_short:
        daily_hedge_return = pos_value.diff().fillna(pos_value.iloc[0]) / K_short
        # Explicit opening debit: the hedge_spend is deducted from strategy
        # PnL on the open bar (the premium collected is offset by the
        # hedge buy):
        daily_hedge_return.iloc[0] -= hedge_spend / K_short

        hedge_daily.loc[cycle_idx] = (hedge_daily.loc[cycle_idx].values
                                         + daily_hedge_return.values)

        hedge_legs.append({
            "open_date": open_date,
            "K_hedge": K_hedge,
            "p_hedge_open": p_hedge_open,
            "hedge_qty": hedge_qty,
            "hedge_spend": hedge_spend,
        })

    net = (daily.add(hedge_daily, fill_value=0.0)).rename("net_daily_return")

    return {
        "net_daily_return": net,
        "hedge_daily_return": hedge_daily.rename("hedge_daily_return"),
        "hedge_legs": pd.DataFrame(hedge_legs),
    }
```

- [ ] **Step 3.4: Run test — verify PASS**

Run: `.venv/bin/pytest tests/test_overlay_tail_hedge.py -v`
Expected: 3 tests pass.

- [ ] **Step 3.5: Commit**

```bash
git add src/vrp/overlays/tail_hedge.py tests/test_overlay_tail_hedge.py
git commit -m "vrp: Overlay 3 — 5-delta tail-hedge spend (Phase 4 Task 3)"
```

---

## Task 4: Combined Overlay Runner

**Files:**
- Create: `scripts/run_strategy_c_overlays.py`

- [ ] **Step 4.1: Write runner**

`scripts/run_strategy_c_overlays.py`:

```python
"""Strategy C + tail-risk overlays comparison.

Base strategy: Strategy C (spread variant, threshold = -2 vol points —
the Phase 3 train-optimal). Applies each overlay individually and all
three combined, reports train/test metrics and equity/drawdown figures.

Outputs:
    reports/strategy_c_overlays/metrics.json
    reports/strategy_c_overlays/equity.png
    reports/strategy_c_overlays/drawdown.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.data.vx_futures import load_vx_continuous
from vrp.overlays.regime_filter import vix_regime_mask, apply_mask
from vrp.overlays.tail_hedge import add_tail_hedge
from vrp.overlays.vol_scaling import target_vol_scale
from vrp.report.metrics import summary, drawdown_series
from vrp.strategies.strategy_c import run_strategy_c
from vrp.util.vol import close_to_close_rv
from vrp.util.vrp_signal import compute_vrp

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"
THRESHOLD = -2.0  # Phase 3 train-optimal for spread variant


def _windowed(ret: pd.Series) -> dict:
    return {"train": summary(ret.loc[TRAIN_START:TRAIN_END]),
             "test":  summary(ret.loc[TEST_START:TEST_END])}


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_c_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx_df = load_spx(start=TRAIN_START, end=TEST_END)
    spx = spx_df["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)
    vx = load_vx_continuous(start=TRAIN_START, end=TEST_END)
    rv = close_to_close_rv(spx, window=20)
    vrp = compute_vrp(vix, rv).dropna()

    strat = run_strategy_c(spx, vix, vrp, threshold=THRESHOLD,
                            target_delta=-0.30, long_put_delta=-0.10,
                            tc_pct_of_premium=0.05)
    base_ret = strat["daily_return"]

    # Overlay 1: VIX regime filter
    mask = vix_regime_mask(vix, vx["front_settle"], vx["second_settle"])
    ret_o1 = apply_mask(base_ret, mask)

    # Overlay 2: RV position scaling
    ret_o2 = target_vol_scale(base_ret, target_vol=0.10, window=20,
                                leverage_cap=1.0)

    # Overlay 3: tail hedge
    o3 = add_tail_hedge(strat, spx, vix, hedge_delta=-0.05,
                         hedge_spend_pct=0.15)
    ret_o3 = o3["net_daily_return"]

    # Combined: apply all three. Order matters slightly; apply regime first
    # (masks cash days), then vol-scale (uses strategy's own returns), then
    # tail-hedge (adds PnL that is not masked by the regime filter — the
    # hedge is a separate position and continues to mark).
    ret_masked = apply_mask(base_ret, mask)
    ret_scaled_masked = target_vol_scale(ret_masked, target_vol=0.10,
                                           window=20, leverage_cap=1.0)
    hedged_strat = {"daily_return": ret_scaled_masked,
                    "positions": strat["positions"]}
    combined_hedged = add_tail_hedge(hedged_strat, spx, vix,
                                       hedge_delta=-0.05,
                                       hedge_spend_pct=0.15)
    ret_combined = combined_hedged["net_daily_return"]

    results = {
        "base_C_spread_thr=-2": _windowed(base_ret),
        "+O1_regime_filter":    _windowed(ret_o1),
        "+O2_vol_scale_10pct":  _windowed(ret_o2),
        "+O3_tail_hedge_15pct": _windowed(ret_o3),
        "+O1+O2+O3_combined":   _windowed(ret_combined),
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    # Plots
    eq = pd.DataFrame({
        "base C(spread, thr=-2)": (1 + base_ret).cumprod(),
        "+O1 regime filter":       (1 + ret_o1).cumprod(),
        "+O2 vol scale":           (1 + ret_o2).cumprod(),
        "+O3 tail hedge":          (1 + ret_o3).cumprod(),
        "combined":                (1 + ret_combined).cumprod(),
    })

    fig, ax = plt.subplots(figsize=(11, 5))
    eq.plot(ax=ax)
    ax.set_title("Strategy C + tail-risk overlays — cumulative return")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red",
                label="out-of-sample")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_dir / "equity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    dd = pd.DataFrame({col: drawdown_series(s) for col, s in [
        ("base", base_ret), ("+O1", ret_o1), ("+O2", ret_o2),
        ("+O3", ret_o3), ("combined", ret_combined),
    ]})
    fig, ax = plt.subplots(figsize=(11, 4))
    dd.plot(ax=ax)
    ax.set_title("Strategy C + tail-risk overlays — drawdown")
    ax.set_ylabel("Drawdown")
    fig.savefig(out_dir / "drawdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy C overlays outputs written to {out_dir}")
    for name, block in results.items():
        tr, te = block["train"], block["test"]
        print(f"  {name:30s}  train S={tr['sharpe']:+.3f} MDD={tr['max_drawdown']:+.3f}  |  "
              f"test S={te['sharpe']:+.3f} MDD={te['max_drawdown']:+.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run**

Run: `.venv/bin/python scripts/run_strategy_c_overlays.py`

Expected: produces metrics JSON + two figures. Prints the five-way comparison.

**Valid outcomes:**
- If any overlay strictly improves both Sharpe and MDD on test: honest win.
- If an overlay improves MDD at a Sharpe cost: document the trade-off honestly.
- If an overlay makes things worse: report transparently. The point of the study is to test the spec's overlays, not to retrofit them.

- [ ] **Step 4.3: Commit**

```bash
git add scripts/run_strategy_c_overlays.py
git commit -m "vrp: Strategy C + tail-risk overlays combined runner (Phase 4 Task 4)"
```

---

## Task 5: Phase 4 README Section

**Files:**
- Modify: `src/vrp/README.md` — add Phase 4 section before Limitations.

- [ ] **Step 5.1: Edit README**

Add a `## Phase 4 — Tail-Risk Overlays` section with:

```markdown
## Phase 4 — Tail-Risk Overlays

Three overlays from the project spec, applied to Strategy C (spread,
threshold = −2 vol points — the Phase 3 train-optimal). Parameters are
pinned to the spec, not tuned:

- **Overlay 1 — VIX regime filter.** Go to cash when VIX > 30 or VX
  term structure inverts (front > second). Re-enter after 7 consecutive
  calm days (VIX < 25 and front < second).
- **Overlay 2 — Realized-vol position scaling.** Scale daily returns by
  `min(1, 0.10 / rv_20)` of the strategy's own returns (target 10%
  annualized vol, no upsizing).
- **Overlay 3 — Tail-hedge spend.** Use 15% of premium collected each
  cycle to buy a 5-delta 1-month SPX put; mark daily.

| configuration | train Sharpe | test Sharpe | train MDD | test MDD |
|---|---|---|---|---|
| base C(spread, thr=−2)   | <fill> | <fill> | <fill> | <fill> |
| + O1 regime filter        | <fill> | <fill> | <fill> | <fill> |
| + O2 vol scale 10%        | <fill> | <fill> | <fill> | <fill> |
| + O3 tail hedge 15% spend | <fill> | <fill> | <fill> | <fill> |
| all three combined        | <fill> | <fill> | <fill> | <fill> |

### Verdict

<fill — 2-3 sentences based on the numbers. Typical findings in the
literature: regime filters help tail-risk measures but cost Sharpe by
cashing out of normal months; vol-scaling trades peak-return for
smoother PnL; tail-hedge spend generally reduces MDD at a direct
~1-2% annualized return cost. Report what actually happens with the
numbers in this sample.>
```

Fill `<fill>` from the runner's `metrics.json`.

- [ ] **Step 5.2: Commit**

```bash
git add src/vrp/README.md
git commit -m "vrp: Phase 4 README section — tail-risk overlays (Phase 4 Task 5)"
```

---

## Phase 4 Definition of Done

1. `.venv/bin/pytest` passes (41 from Phase 3 + 4 regime + 3 vol_scale + 3 tail_hedge = 51 tests).
2. `scripts/run_strategy_c_overlays.py` runs and produces metrics + figures.
3. README updated with filled numbers and an honest verdict.
4. Five task commits present.

## Self-Review

- **Spec coverage:** all three overlays with spec-mandated parameters. No tuning.
- **Placeholders:** all steps are concrete. README has `<fill>` directives for the runner outputs.
- **Type consistency:** `vix_regime_mask`/`apply_mask` → `pd.Series`; `target_vol_scale` → `pd.Series`; `add_tail_hedge` returns a dict that works against strategy outputs with `.positions` and `.daily_return`. All consistent.
- **TDD:** each overlay gets strict red/green tests in isolation. Runner is exercised via produced reports.
- **Scope:** Phase 4 covers only overlays. Bootstrap/1987 stress is Phase 5.
