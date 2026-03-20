"""Strategy B — canonical CBOE PUT index analysis.

PUT is the published CBOE S&P 500 PutWrite Index: monthly cash-secured
sales of at-the-money SPX puts. This script is the primary Strategy B
backtest — the PUT series already bakes in realistic execution and
transaction costs from CBOE's methodology.

Outputs:
    reports/strategy_b_putindex/metrics_train.json
    reports/strategy_b_putindex/metrics_test.json
    reports/strategy_b_putindex/regime_metrics_test.json
    reports/strategy_b_putindex/alpha_beta_vs_spx.json
    reports/strategy_b_putindex/equity_vs_spx.png
    reports/strategy_b_putindex/drawdown.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.cboe_indices import load_cboe_index
from vrp.data.spx import load_spx
from vrp.report.metrics import summary, drawdown_series
from vrp.report.regimes import regime_metrics

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"


def _daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def _alpha_beta(strategy_ret: pd.Series, bench_ret: pd.Series) -> dict:
    aligned = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["y", "x"]
    x = aligned["x"].values
    y = aligned["y"].values
    x_mean, y_mean = x.mean(), y.mean()
    var_x = ((x - x_mean) ** 2).mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    beta = float(cov_xy / var_x) if var_x > 0 else float("nan")
    alpha_daily = float(y_mean - beta * x_mean)
    alpha_ann = alpha_daily * 252
    return {"alpha_annual": alpha_ann, "beta": beta,
            "n_days": int(len(aligned))}


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_b_putindex"
    out_dir.mkdir(parents=True, exist_ok=True)

    put = load_cboe_index("PUT")
    spx = load_spx(start="2012-01-01")["close"]

    put_ret = _daily_returns(put)
    spx_ret = _daily_returns(spx)

    train_put = put_ret.loc[TRAIN_START:TRAIN_END]
    test_put  = put_ret.loc[TEST_START:TEST_END]

    (out_dir / "metrics_train.json").write_text(json.dumps(summary(train_put), indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(summary(test_put), indent=2))
    (out_dir / "regime_metrics_test.json").write_text(
        json.dumps(regime_metrics(test_put), indent=2, default=str)
    )

    ab_train = _alpha_beta(train_put, spx_ret.loc[TRAIN_START:TRAIN_END])
    ab_test  = _alpha_beta(test_put,  spx_ret.loc[TEST_START:TEST_END])
    (out_dir / "alpha_beta_vs_spx.json").write_text(
        json.dumps({"train": ab_train, "test": ab_test}, indent=2)
    )

    common_start = max(put_ret.index.min(), spx_ret.index.min(),
                       pd.Timestamp(TRAIN_START))
    norm_put = (1 + put_ret).cumprod()
    norm_spx = (1 + spx_ret).cumprod()
    norm_put = norm_put / norm_put.loc[common_start:].iloc[0]
    norm_spx = norm_spx / norm_spx.loc[common_start:].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame({"PUT index": norm_put, "SPX": norm_spx}).loc[common_start:].plot(ax=ax)
    ax.set_title("Strategy B — CBOE PUT index vs SPX (normalized)")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red", label="out-of-sample")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "equity_vs_spx.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    drawdown_series(put_ret).loc[common_start:].plot(ax=ax, color="red")
    ax.set_title("Strategy B — PUT index drawdown")
    ax.set_ylabel("Drawdown")
    fig.savefig(out_dir / "drawdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy B (PUT index) outputs written to {out_dir}")
    print("Train:", json.dumps(summary(train_put), indent=2))
    print("Test: ", json.dumps(summary(test_put),  indent=2))
    print("alpha/beta train:", ab_train)
    print("alpha/beta test: ", ab_test)


if __name__ == "__main__":
    main()
