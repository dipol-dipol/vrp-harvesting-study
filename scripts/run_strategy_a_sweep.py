"""Strategy A robustness sweep.

Two comparisons run from the same cached VX dataset:

1. **Transaction-cost sensitivity** — sweep `tc_bps_per_roll` across
   {1, 5, 10, 20, 30}. The spec notes VIX futures slippage is 1-2 bps in
   practice; 5-30 bps is the stress range. We want to confirm the
   Sharpe sign and rough magnitude are insensitive to cost realism.

2. **Direction-flip comparison** — the baseline is short-front /
   long-second (per spec). The opposite variant, long-front /
   short-second, is logically `-daily_return` of the baseline. We run
   both and compare equity, Sharpe, and drawdown.

Outputs:
    reports/strategy_a_sweep/tc_sweep.json
    reports/strategy_a_sweep/tc_sweep.png
    reports/strategy_a_sweep/variant_comparison.json
    reports/strategy_a_sweep/variant_comparison.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.vx_futures import load_vx_continuous
from vrp.report.metrics import summary
from vrp.strategies.strategy_a import run_strategy_a

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"
TC_BPS_VALUES = [1, 5, 10, 20, 30]


def _train_test_sharpe_mdd(ret: pd.Series) -> dict:
    tr = ret.loc[TRAIN_START:TRAIN_END]
    te = ret.loc[TEST_START:TEST_END]
    tr_sum = summary(tr)
    te_sum = summary(te)
    return {
        "train_sharpe": tr_sum["sharpe"],
        "train_ann_return": tr_sum["ann_return"],
        "train_max_drawdown": tr_sum["max_drawdown"],
        "test_sharpe": te_sum["sharpe"],
        "test_ann_return": te_sum["ann_return"],
        "test_max_drawdown": te_sum["max_drawdown"],
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_a_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    vx = load_vx_continuous(start=TRAIN_START, end=TEST_END)

    # ---- Transaction-cost sensitivity --------------------------------------
    tc_results = {}
    tc_equity = {}
    for tc in TC_BPS_VALUES:
        result = run_strategy_a(vx, tc_bps_per_roll=float(tc))
        ret = result["daily_return"]
        tc_results[f"{tc}_bps"] = _train_test_sharpe_mdd(ret)
        tc_equity[f"{tc} bps"] = (1.0 + ret).cumprod()

    (out_dir / "tc_sweep.json").write_text(json.dumps(tc_results, indent=2))

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame(tc_equity).plot(ax=ax)
    ax.set_title("Strategy A — Equity under varying roll-day transaction costs")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red",
               label="out-of-sample")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out_dir / "tc_sweep.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Direction-flip comparison -----------------------------------------
    baseline = run_strategy_a(vx, tc_bps_per_roll=1.0)
    base_ret = baseline["daily_return"]
    flipped_ret = -base_ret  # long-front / short-second is -daily_return

    variant_results = {
        "short_front_long_second_baseline": _train_test_sharpe_mdd(base_ret),
        "long_front_short_second_flipped": _train_test_sharpe_mdd(flipped_ret),
    }
    (out_dir / "variant_comparison.json").write_text(
        json.dumps(variant_results, indent=2)
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame({
        "short front / long second (baseline)": (1 + base_ret).cumprod(),
        "long front / short second (flipped)":  (1 + flipped_ret).cumprod(),
    }).plot(ax=ax)
    ax.set_title("Strategy A — Direction-flip comparison")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red",
               label="out-of-sample")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "variant_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Print compact summary ---------------------------------------------
    print(f"Strategy A sweep outputs written to {out_dir}")
    print("\nTransaction-cost sweep (Sharpe, MDD):")
    for tc, vals in tc_results.items():
        print(f"  {tc:>8}: train Sharpe={vals['train_sharpe']:+.3f} "
              f"MDD={vals['train_max_drawdown']:+.3f}  |  "
              f"test Sharpe={vals['test_sharpe']:+.3f} "
              f"MDD={vals['test_max_drawdown']:+.3f}")

    print("\nDirection comparison (Sharpe, MDD):")
    for name, vals in variant_results.items():
        print(f"  {name}")
        print(f"    train Sharpe={vals['train_sharpe']:+.3f} "
              f"ann_return={vals['train_ann_return']:+.3f} "
              f"MDD={vals['train_max_drawdown']:+.3f}")
        print(f"    test  Sharpe={vals['test_sharpe']:+.3f} "
              f"ann_return={vals['test_ann_return']:+.3f} "
              f"MDD={vals['test_max_drawdown']:+.3f}")


if __name__ == "__main__":
    main()
