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
THRESHOLD = -2.0


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

    mask = vix_regime_mask(vix, vx["front_settle"], vx["second_settle"])
    ret_o1 = apply_mask(base_ret, mask)

    ret_o2 = target_vol_scale(base_ret, target_vol=0.10, window=20,
                                leverage_cap=1.0)

    o3 = add_tail_hedge(strat, spx, vix, hedge_delta=-0.05,
                         hedge_spend_pct=0.15)
    ret_o3 = o3["net_daily_return"]

    # Combined: mask, then vol-scale, then tail-hedge
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
