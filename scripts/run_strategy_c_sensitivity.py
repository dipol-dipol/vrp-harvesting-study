"""Strategy C threshold sensitivity — tune on train, report on test.

Train-then-test protocol:
1. Sweep thresholds across [-2, -1, 0, 1, 2, 3, 4, 5, 6] vol points on
   the training window (2013-2018) only.
2. Pick the threshold that maximizes train Sharpe for each variant.
3. Evaluate that threshold on the test window (2019-2024) ONCE.

No peeking at test data during threshold selection.

Outputs:
    reports/strategy_c_sensitivity/train_sweep.json
    reports/strategy_c_sensitivity/chosen_thresholds_test.json
    reports/strategy_c_sensitivity/train_sharpe_curves.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

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
        best = max(rows, key=lambda r: (r["train_sharpe"]
                    if r["train_sharpe"] == r["train_sharpe"] else float("-inf")))
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
