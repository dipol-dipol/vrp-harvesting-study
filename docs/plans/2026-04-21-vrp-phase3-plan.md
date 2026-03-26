# VRP Phase 3 — Strategy C (Conditional VRP Harvester) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Strategy C — a conditional variant of Strategy B that only writes puts when the realized VRP signal `IV_t − RV_t` exceeds a threshold. Test the hypothesis that the VRP is time-varying and that conditional exposure improves risk-adjusted returns even at the cost of lower total premium.

**Architecture:** New VRP-signal utility under `vrp.util.vrp_signal`. New `strategy_c.py` wrapping `run_strategy_b` with a month-end gate. Two runners: a baseline runner at threshold=2 vol points (spec default), and a threshold-sensitivity analysis that honors the train/test discipline (tune on 2013-2018, apply to 2019-2024 once, report).

**Tech Stack:** Same as Phase 2. Extends the existing `vrp` package.

**Branch:** `phase3-strategy-c`, branched off `phase2-strategy-b`.

**Train/Test split:** Train 2013-01-01 → 2018-12-31; Test 2019-01-01 → 2024-12-31. Same as Phases 1 and 2.

**Key discipline rule:** Threshold selection happens on the training window only. The test window is evaluated **once** at the chosen threshold and reported without subsequent tuning. If the optimal training threshold turns out to be 0 (i.e. conditioning doesn't help), report that honestly.

---

## File Structure

```
src/vrp/
├── util/
│   └── vrp_signal.py                 # NEW — compute_vrp + month-end extraction
└── strategies/
    └── strategy_c.py                 # NEW — gated wrapper around strategy_b
scripts/
├── run_strategy_c.py                 # NEW — baseline (threshold=2)
└── run_strategy_c_sensitivity.py     # NEW — train-window sweep + held-out test
tests/
├── test_vrp_signal.py
└── test_strategy_c.py
```

---

## Task 1: VRP Signal Utility

**Files:**
- Create: `src/vrp/util/vrp_signal.py`
- Create: `tests/test_vrp_signal.py`

- [ ] **Step 1.1: Write failing test**

`tests/test_vrp_signal.py`:

```python
import numpy as np
import pandas as pd

from vrp.util.vrp_signal import compute_vrp, month_end_signal


def _constant_vix_rv_series(n=252, vix_pct=20.0, rv_pct=15.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    vix = pd.Series(vix_pct, index=idx)
    rv = pd.Series(rv_pct / 100.0, index=idx)  # rv as decimal (matches realized_vol)
    return vix, rv


def test_compute_vrp_constant_case():
    vix, rv = _constant_vix_rv_series(vix_pct=20.0, rv_pct=15.0)
    vrp = compute_vrp(vix, rv)
    # VIX - RV in vol points: 20 - 15 = 5
    assert (vrp == 5.0).all()


def test_compute_vrp_aligns_by_index():
    idx = pd.bdate_range("2020-01-02", periods=10)
    vix = pd.Series(20.0, index=idx)
    # RV available for a different subset
    rv = pd.Series(0.15, index=idx[2:])
    vrp = compute_vrp(vix, rv)
    assert vrp.index.equals(rv.index)
    assert (vrp == 5.0).all()


def test_month_end_signal_picks_last_trading_day_per_month():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    vrp = pd.Series(range(len(idx)), index=idx).astype(float)
    mes = month_end_signal(vrp)
    # Expect one value per month, indexed at the last trading date of each
    assert len(mes) == 3
    assert mes.index[0] == pd.Timestamp("2020-01-31")
    assert mes.index[1] == pd.Timestamp("2020-02-28")
    assert mes.index[2] == pd.Timestamp("2020-03-31")
```

- [ ] **Step 1.2: Run test — verify FAIL**

Run: `.venv/bin/pytest tests/test_vrp_signal.py -v`
Expected: `ModuleNotFoundError: No module named 'vrp.util.vrp_signal'`.

- [ ] **Step 1.3: Write implementation**

`src/vrp/util/vrp_signal.py`:

```python
"""Volatility Risk Premium signal.

The VRP at time t is defined as IV_t - RV_t, both expressed in vol points
(percentage points of annualized volatility). IV is the 30-day at-the-
money implied vol. In this study we use VIX as an IV proxy — VIX is a
variance-swap construct rather than strict ATM IV, so the proxy is
directionally correct but biases VRP estimates upward during skew-heavy
regimes. Flagged in the Phase 3 README.

RV is typically a 20-day close-to-close realized vol (matching the
vrp.util.vol.close_to_close_rv default). Both series should be supplied
in decimal form internally: VIX is scaled percent (e.g. 20.0 means 20
vol points), RV from close_to_close_rv is decimal (e.g. 0.15 for 15%).
This function multiplies RV by 100 before subtracting, so the returned
VRP is in vol points.

References:
- Carr, Wu (2009) "Variance Risk Premiums"
- Bondarenko (2014) "Why Are Put Options So Expensive?"
- Dew-Becker et al. (2017) "The Price of Variance Risk"
"""
from __future__ import annotations

import pandas as pd


def compute_vrp(vix_pct: pd.Series, rv_decimal: pd.Series) -> pd.Series:
    """Compute VRP = IV - RV in vol points.

    Parameters
    ----------
    vix_pct:
        VIX spot series, in vol points (20.0 == 20 vol).
    rv_decimal:
        Realized vol series, in decimal form (0.15 == 15%).

    Returns the index-aligned intersection.
    """
    aligned = pd.concat([vix_pct.rename("iv"), (rv_decimal * 100.0).rename("rv_pct")],
                         axis=1, join="inner").dropna()
    return (aligned["iv"] - aligned["rv_pct"]).rename("vrp")


def month_end_signal(series: pd.Series) -> pd.Series:
    """Pick the value on the last trading day of each calendar month."""
    return series.groupby(series.index.to_period("M")).last()
```

Note on the month-end grouping: `groupby(.to_period("M")).last()` returns a Series whose index is `PeriodIndex("2020-01", ...)`. For downstream use (joining to daily VX or SPX series) we want DatetimeIndex. Fix that inside `month_end_signal`:

Replace the implementation with:

```python
def month_end_signal(series: pd.Series) -> pd.Series:
    """Pick the value on the last trading day of each calendar month."""
    grouped = series.groupby(series.index.to_period("M"))
    last_dates = grouped.apply(lambda s: s.index[-1])
    last_values = grouped.last()
    last_values.index = pd.DatetimeIndex(last_dates.values)
    return last_values.sort_index()
```

- [ ] **Step 1.4: Run test — verify PASS**

Run: `.venv/bin/pytest tests/test_vrp_signal.py -v`
Expected: 3 tests pass.

- [ ] **Step 1.5: Commit**

```bash
git add src/vrp/util/vrp_signal.py tests/test_vrp_signal.py
git commit -m "vrp: VRP signal (IV - RV) with month-end extraction (Phase 3 Task 1)"
```

---

## Task 2: Strategy C Engine

**Files:**
- Create: `src/vrp/strategies/strategy_c.py`
- Create: `tests/test_strategy_c.py`

### Design

Strategy C wraps `run_strategy_b`. For each monthly cycle, it checks the VRP signal on the cycle's open date (the month-start trading day): if `vrp >= threshold`, the Strategy B position for that month is kept; else it is replaced with zero daily returns (cash).

This is implemented post-hoc: run Strategy B once, then mask daily returns to zero for cycles whose open-date VRP is below threshold. This is much simpler than rewriting the cycle loop inside Strategy B.

- [ ] **Step 2.1: Write failing test**

`tests/test_strategy_c.py`:

```python
import numpy as np
import pandas as pd
import pytest

from vrp.strategies.strategy_b import run_strategy_b
from vrp.strategies.strategy_c import run_strategy_c


def _synth_spx_vix(n=252, spx=100.0, vix=20.0):
    idx = pd.bdate_range("2020-01-02", periods=n)
    return (pd.Series(spx, index=idx),
            pd.Series(vix, index=idx))


def _synth_vrp_alternating(idx):
    """VRP alternates month by month between 5.0 and -2.0 in vol points."""
    vrp = pd.Series(0.0, index=idx)
    for i, d in enumerate(idx):
        vrp.iloc[i] = 5.0 if (d.month % 2 == 0) else -2.0
    return vrp


def test_strategy_c_matches_b_when_threshold_low():
    spx, vix = _synth_spx_vix()
    vrp = pd.Series(10.0, index=spx.index)  # always above any low threshold
    b = run_strategy_b(spx, vix, target_delta=-0.30)
    c = run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=-0.30)
    pd.testing.assert_series_equal(b["daily_return"], c["daily_return"],
                                   check_names=False)


def test_strategy_c_all_cash_when_threshold_high():
    spx, vix = _synth_spx_vix()
    vrp = pd.Series(0.0, index=spx.index)  # never above 100
    c = run_strategy_c(spx, vix, vrp, threshold=100.0, target_delta=-0.30)
    assert (c["daily_return"] == 0.0).all()
    assert c["active_months_fraction"] == 0.0


def test_strategy_c_active_fraction():
    # Synth: VRP alternates monthly; half the months are active at threshold=0
    spx, vix = _synth_spx_vix(n=252)
    vrp = _synth_vrp_alternating(spx.index)
    c = run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=-0.30)
    # Roughly half the months should be active (VRP > 0 on odd months)
    assert 0.3 < c["active_months_fraction"] < 0.7


def test_strategy_c_invalid_delta_raises():
    spx, vix = _synth_spx_vix()
    vrp = pd.Series(5.0, index=spx.index)
    with pytest.raises(ValueError):
        run_strategy_c(spx, vix, vrp, threshold=0.0, target_delta=0.30)
```

- [ ] **Step 2.2: Run test — verify FAIL**

Run: `.venv/bin/pytest tests/test_strategy_c.py -v`
Expected: `ModuleNotFoundError: No module named 'vrp.strategies.strategy_c'`.

- [ ] **Step 2.3: Write implementation**

`src/vrp/strategies/strategy_c.py`:

```python
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
    """Conditional put-writer. See Strategy B for non-gating parameters.

    Parameters
    ----------
    vrp:
        VRP signal in vol points, indexed by trading date. Must cover
        at least the month-start dates of the requested backtest window.
    threshold:
        Minimum VRP to take a position that month. Below threshold, the
        month is held in cash.
    """
    # Delegate all parameter validation to Strategy B; then gate.
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
        # Look up VRP as of the LAST trading day in or before open_date that
        # has a VRP value (guards against a short RV-warmup gap at start).
        vrp_history = vrp.loc[:open_date].dropna()
        if len(vrp_history) == 0:
            # No VRP signal yet; stay in cash
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
```

- [ ] **Step 2.4: Run test — verify PASS**

Run: `.venv/bin/pytest tests/test_strategy_c.py -v`
Expected: 4 tests pass.

- [ ] **Step 2.5: Commit**

```bash
git add src/vrp/strategies/strategy_c.py tests/test_strategy_c.py
git commit -m "vrp: Strategy C — conditional VRP-gated put-writer (Phase 3 Task 2)"
```

---

## Task 3: Strategy C Baseline Runner

**Files:**
- Create: `scripts/run_strategy_c.py`

- [ ] **Step 3.1: Write runner**

`scripts/run_strategy_c.py`:

```python
"""Strategy C baseline — conditional put-writer at threshold = 2 vol points.

Runs two variants at the spec-default threshold:
1. Naked put (short -0.30Δ), gated.
2. Spread (short -0.30Δ, long -0.10Δ), gated.

Baseline threshold of 2.0 is a judgment call, not tuned on the data.
The sensitivity script (run_strategy_c_sensitivity.py) picks the train-
optimal threshold and evaluates held-out test performance separately.

Outputs:
    reports/strategy_c/metrics_train.json
    reports/strategy_c/metrics_test.json
    reports/strategy_c/active_months.json
    reports/strategy_c/equity.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.report.metrics import summary
from vrp.strategies.strategy_b import run_strategy_b
from vrp.strategies.strategy_c import run_strategy_c
from vrp.util.vol import close_to_close_rv
from vrp.util.vrp_signal import compute_vrp

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"
THRESHOLD = 2.0


def _windowed(ret: pd.Series) -> dict:
    return {
        "train": summary(ret.loc[TRAIN_START:TRAIN_END]),
        "test":  summary(ret.loc[TEST_START:TEST_END]),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start=TRAIN_START, end=TEST_END)["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)
    rv = close_to_close_rv(spx, window=20)
    vrp = compute_vrp(vix, rv).dropna()

    naked_b = run_strategy_b(spx, vix, target_delta=-0.30,
                              tc_pct_of_premium=0.05)
    spread_b = run_strategy_b(spx, vix, target_delta=-0.30,
                               long_put_delta=-0.10, tc_pct_of_premium=0.05)

    naked_c = run_strategy_c(spx, vix, vrp, threshold=THRESHOLD,
                              target_delta=-0.30, tc_pct_of_premium=0.05)
    spread_c = run_strategy_c(spx, vix, vrp, threshold=THRESHOLD,
                               target_delta=-0.30, long_put_delta=-0.10,
                               tc_pct_of_premium=0.05)

    results = {
        "threshold_vol_points": THRESHOLD,
        "naked_strategy_b":  _windowed(naked_b["daily_return"]),
        "naked_strategy_c":  _windowed(naked_c["daily_return"]),
        "spread_strategy_b": _windowed(spread_b["daily_return"]),
        "spread_strategy_c": _windowed(spread_c["daily_return"]),
        "active_months_fraction_naked":  naked_c["active_months_fraction"],
        "active_months_fraction_spread": spread_c["active_months_fraction"],
    }
    (out_dir / "metrics_train.json").write_text(json.dumps({
        k: (v["train"] if isinstance(v, dict) and "train" in v else v)
        for k, v in results.items()
    }, indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps({
        k: (v["test"] if isinstance(v, dict) and "test" in v else v)
        for k, v in results.items()
    }, indent=2))
    (out_dir / "active_months.json").write_text(json.dumps({
        "naked": naked_c["active_months_fraction"],
        "spread": spread_c["active_months_fraction"],
        "threshold_vol_points": THRESHOLD,
    }, indent=2))

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame({
        "B naked (unconditional)": (1 + naked_b["daily_return"]).cumprod(),
        "C naked (gated)":         (1 + naked_c["daily_return"]).cumprod(),
        "B spread (unconditional)":(1 + spread_b["daily_return"]).cumprod(),
        "C spread (gated)":        (1 + spread_c["daily_return"]).cumprod(),
    }).plot(ax=ax)
    ax.set_title(f"Strategy C vs B — threshold = {THRESHOLD} vol points")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red",
               label="out-of-sample")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_dir / "equity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy C baseline outputs written to {out_dir}")
    print(f"Naked active months fraction:  {naked_c['active_months_fraction']:.2%}")
    print(f"Spread active months fraction: {spread_c['active_months_fraction']:.2%}")

    def _brief(r: dict) -> dict:
        return {"sharpe": r["sharpe"], "ann_return": r["ann_return"],
                 "max_drawdown": r["max_drawdown"]}

    print("\nComparison:")
    for name, block in [("naked_b", _windowed(naked_b["daily_return"])),
                         ("naked_c", _windowed(naked_c["daily_return"])),
                         ("spread_b", _windowed(spread_b["daily_return"])),
                         ("spread_c", _windowed(spread_c["daily_return"]))]:
        print(f"  {name}: train={_brief(block['train'])} "
              f"test={_brief(block['test'])}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run**

Run: `.venv/bin/python scripts/run_strategy_c.py`
Expected: runs to completion; prints active-months fraction (typically 60-80% at threshold=2) and the four-way comparison.

- [ ] **Step 3.3: Commit**

```bash
git add scripts/run_strategy_c.py
git commit -m "vrp: Strategy C baseline runner at threshold = 2 vol points (Phase 3 Task 3)"
```

---

## Task 4: Threshold Sensitivity (train-then-test)

**Files:**
- Create: `scripts/run_strategy_c_sensitivity.py`

### Protocol

1. Sweep thresholds in `[-2, -1, 0, 1, 2, 3, 4, 5, 6]` vol points on the training window only.
2. Record train Sharpe, ann return, max DD, and active months fraction for each threshold and each variant (naked, spread).
3. Pick the **train-optimal threshold** (max Sharpe on train) per variant.
4. Evaluate the chosen threshold on the test window **once**. Report.
5. Also report threshold=0 (no gating, equivalent to Strategy B) as a sanity baseline.

- [ ] **Step 4.1: Write runner**

`scripts/run_strategy_c_sensitivity.py`:

```python
"""Strategy C threshold sensitivity — tune on train, report on test.

Train-then-test protocol:
1. Sweep thresholds across [-2, -1, 0, 1, 2, 3, 4, 5, 6] vol points on the
   training window (2013-2018) only.
2. Pick the threshold that maximizes train Sharpe for each variant.
3. Evaluate that threshold on the test window (2019-2024) ONCE and report.

This is the honest out-of-sample protocol. No peeking at test data.

Outputs:
    reports/strategy_c_sensitivity/train_sweep.json
    reports/strategy_c_sensitivity/chosen_thresholds_test.json
    reports/strategy_c_sensitivity/train_sharpe_curves.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.report.metrics import summary
from vrp.strategies.strategy_c import run_strategy_c
from vrp.util.vol import close_to_close_rv
from vrp.util.vrp_signal import compute_vrp

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"
THRESHOLDS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _run(spx, vix, vrp, threshold, long_put_delta):
    return run_strategy_c(spx, vix, vrp, threshold=threshold,
                          target_delta=-0.30,
                          long_put_delta=long_put_delta,
                          tc_pct_of_premium=0.05)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_c_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start=TRAIN_START, end=TEST_END)["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)
    rv = close_to_close_rv(spx, window=20)
    vrp = compute_vrp(vix, rv).dropna()

    variants = {"naked": None, "spread": -0.10}

    # ---- Train sweep -------------------------------------------------------
    sweep = {v: [] for v in variants}
    for variant, long_put in variants.items():
        for t in THRESHOLDS:
            result = _run(spx, vix, vrp, t, long_put)
            train = result["daily_return"].loc[TRAIN_START:TRAIN_END]
            s = summary(train)
            sweep[variant].append({
                "threshold": t,
                "train_sharpe": s["sharpe"],
                "train_ann_return": s["ann_return"],
                "train_max_drawdown": s["max_drawdown"],
                "active_fraction": result["active_months_fraction"],
            })
    (out_dir / "train_sweep.json").write_text(json.dumps(sweep, indent=2))

    # ---- Choose train-optimal thresholds ----------------------------------
    chosen = {}
    for variant, rows in sweep.items():
        best = max(rows, key=lambda r: r["train_sharpe"]
                   if r["train_sharpe"] == r["train_sharpe"] else float("-inf"))
        chosen[variant] = best["threshold"]

    # ---- Evaluate chosen thresholds on test window once -------------------
    test_reports = {}
    for variant, t in chosen.items():
        result = _run(spx, vix, vrp, t, variants[variant])
        train = result["daily_return"].loc[TRAIN_START:TRAIN_END]
        test = result["daily_return"].loc[TEST_START:TEST_END]
        test_reports[variant] = {
            "chosen_threshold": t,
            "train_summary": summary(train),
            "test_summary": summary(test),
            "active_fraction": result["active_months_fraction"],
        }
    (out_dir / "chosen_thresholds_test.json").write_text(
        json.dumps(test_reports, indent=2)
    )

    # ---- Plot train Sharpe curves -----------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    for variant, rows in sweep.items():
        xs = [r["threshold"] for r in rows]
        ys = [r["train_sharpe"] for r in rows]
        ax.plot(xs, ys, marker="o", label=variant)
    for variant, t in chosen.items():
        ax.axvline(t, linestyle="--", alpha=0.3,
                    label=f"{variant} train-optimal: {t}")
    ax.set_xlabel("Threshold (vol points)")
    ax.set_ylabel("Train Sharpe")
    ax.set_title("Strategy C — train Sharpe as a function of VRP threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "train_sharpe_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy C sensitivity outputs written to {out_dir}")
    print("Chosen thresholds (train-optimal):", chosen)
    for variant, block in test_reports.items():
        print(f"\n{variant} at threshold={block['chosen_threshold']}:")
        print(f"  active months: {block['active_fraction']:.2%}")
        print(f"  train Sharpe: {block['train_summary']['sharpe']:+.3f}  "
              f"MDD={block['train_summary']['max_drawdown']:+.3f}")
        print(f"  test  Sharpe: {block['test_summary']['sharpe']:+.3f}  "
              f"MDD={block['test_summary']['max_drawdown']:+.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run**

Run: `.venv/bin/python scripts/run_strategy_c_sensitivity.py`

Expected: runs; prints the chosen thresholds and test-window performance for each variant.

**Valid outcomes:**
- If conditioning helps: train-optimal threshold > 0, test Sharpe improves over Strategy B at threshold=0.
- If conditioning doesn't help: train-optimal threshold ≤ 0 (never gate off), which degenerates to Strategy B. Report this honestly — it is a valid negative finding.
- If conditioning helps train but hurts test: train-optimal threshold > 0 but test Sharpe is worse than Strategy B. This is classic overfitting to the training window. Report transparently.

- [ ] **Step 4.3: Commit**

```bash
git add scripts/run_strategy_c_sensitivity.py
git commit -m "vrp: Strategy C threshold sensitivity — train-then-test (Phase 3 Task 4)"
```

---

## Task 5: Phase 3 README Section

**Files:**
- Modify: `src/vrp/README.md` — add Phase 3 results section before Limitations.

- [ ] **Step 5.1: Edit README**

Add a `## Phase 3 — Strategy C Results` section with:
- The spec-default baseline (threshold=2 vol points) results for naked and spread variants, side-by-side with Strategy B.
- Train Sharpe vs threshold curve reference.
- The chosen train-optimal thresholds and their out-of-sample test-window performance.
- An honest verdict paragraph: does VRP-gating add value out of sample? (The numbers from the sensitivity JSON tell the story; write the paragraph based on whichever outcome actually occurred.)

Template layout (fill `<fill>` from JSON outputs):

```markdown
## Phase 3 — Strategy C Results

Strategy C gates Strategy B on a VRP signal `VIX_t − RV20_t` (vol points),
taking a position in month `N+1` only when the month-end VRP of month `N`
exceeds a threshold.

### Spec-default baseline (threshold = 2 vol points)

| variant | gating | train Sharpe | test Sharpe | train MDD | test MDD | active months |
|---|---|---|---|---|---|---|
| naked   | off (B) | <fill> | <fill> | <fill> | <fill> | 100% |
| naked   | on  (C) | <fill> | <fill> | <fill> | <fill> | <fill> |
| spread  | off (B) | <fill> | <fill> | <fill> | <fill> | 100% |
| spread  | on  (C) | <fill> | <fill> | <fill> | <fill> | <fill> |

### Threshold sensitivity (train-then-test)

Swept thresholds in `[-2, -1, 0, 1, 2, 3, 4, 5, 6]` vol points on
2013-2018 only, picked the train-maximizing Sharpe per variant, then
evaluated the test window (2019-2024) once.

| variant | train-optimal threshold | train Sharpe @ chosen | test Sharpe @ chosen | test MDD |
|---|---|---|---|---|
| naked  | <fill> | <fill> | <fill> | <fill> |
| spread | <fill> | <fill> | <fill> | <fill> |

### Verdict

<fill — write 2-3 sentences based on the sensitivity JSON: does
out-of-sample performance improve at the train-optimal threshold vs at
threshold=0 (ungated)? If yes, by how much? If no, does the data
support that the VRP is not sufficiently time-varying at monthly
frequency in this sample?>
```

- [ ] **Step 5.2: Commit**

```bash
git add src/vrp/README.md
git commit -m "vrp: Phase 3 README section — Strategy C results (Phase 3 Task 5)"
```

---

## Phase 3 Definition of Done

1. `.venv/bin/pytest` passes (34 from Phase 2 + 3 vrp_signal + 4 strategy_c = 41 tests).
2. `scripts/run_strategy_c.py` runs and produces outputs at threshold=2.
3. `scripts/run_strategy_c_sensitivity.py` runs and produces the train sweep, chosen thresholds, and held-out test performance.
4. README is updated with the filled-in results and honest verdict.
5. 5 task commits present with task-numbered messages.

## Self-Review

- **Spec coverage:** Strategy C spec bullets (VRP = IV − RV, monthly sell-vol gate on threshold, cash when below, threshold sensitivity) are all addressed. Two thresholds (baseline + train-optimal) honor the "no test-tuning" rule by design.
- **Placeholders:** All code and commands are concrete. README placeholders are fill-from-JSON directives for the Phase 3 runner outputs, not deferrals.
- **Type consistency:** `run_strategy_c` returns a superset of `run_strategy_b`'s dict (adds `gating`, `active_months_fraction`, `threshold`). Callers of `daily_return`, `positions`, `monthly_pnl` continue to work unchanged.
- **TDD:** T1 (vrp_signal) and T2 (strategy_c) are strict red/green. T3/T4 are runner scripts exercised via reports.
- **Scope:** Phase 3 covers only Strategy C per the spec. Tail-risk overlays (Phase 4) and bootstrap/1987 stress (Phase 5) are explicitly deferred.
