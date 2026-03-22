"""Synthetic Strategy B (BS put-writer) run and PUT-index correlation gate.

Produces:
    reports/strategy_b_synthetic/metrics_train.json
    reports/strategy_b_synthetic/metrics_test.json
    reports/strategy_b_synthetic/put_index_correlation.json
    reports/strategy_b_synthetic/equity_vs_put_index.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.cboe_indices import load_cboe_index
from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.report.metrics import summary
from vrp.strategies.strategy_b import run_strategy_b

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_b_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start=TRAIN_START, end=TEST_END)["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)
    put = load_cboe_index("PUT")

    out = run_strategy_b(spx, vix, target_delta=-0.30,
                         tc_pct_of_premium=0.05)
    ret = out["daily_return"]

    train_ret = ret.loc[TRAIN_START:TRAIN_END]
    test_ret  = ret.loc[TEST_START:TEST_END]

    (out_dir / "metrics_train.json").write_text(json.dumps(summary(train_ret), indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(summary(test_ret), indent=2))

    put_ret = put.pct_change().dropna()
    combined = pd.concat([
        ret.rename("synth"),
        put_ret.rename("put"),
    ], axis=1).dropna()
    combined_monthly = combined.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_corr = float(combined_monthly["synth"].corr(combined_monthly["put"]))
    daily_corr = float(combined["synth"].corr(combined["put"]))

    (out_dir / "put_index_correlation.json").write_text(
        json.dumps({"monthly_correlation": monthly_corr,
                    "daily_correlation": daily_corr,
                    "note": "Sanity gate requires monthly_correlation >= 0.6"},
                   indent=2)
    )

    synth_equity = (1 + ret).cumprod()
    put_equity = (1 + put_ret).cumprod()
    common = synth_equity.index.intersection(put_equity.index)
    norm = pd.DataFrame({
        "synthetic BS writer": synth_equity.loc[common],
        "PUT index":           put_equity.loc[common],
    })
    norm = norm / norm.iloc[0]

    fig, ax = plt.subplots(figsize=(11, 4))
    norm.plot(ax=ax)
    ax.set_title(
        f"Strategy B — synthetic vs CBOE PUT (monthly corr = {monthly_corr:.2f})"
    )
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red", label="out-of-sample")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "equity_vs_put_index.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Synthetic Strategy B outputs written to {out_dir}")
    print("Train:", json.dumps(summary(train_ret), indent=2))
    print("Test: ", json.dumps(summary(test_ret),  indent=2))
    print(f"Monthly corr vs PUT: {monthly_corr:.3f}  (expected >= 0.6)")
    print(f"Daily corr vs PUT:   {daily_corr:.3f}")


if __name__ == "__main__":
    main()
