# Volatility Risk Premium Harvesting — A Comparative Study

A research project replicating and comparing three implementations of VRP
harvesting on SPX, with explicit tail-risk accounting. Framed as a
comparative study of known strategies, not a novel-alpha claim. Primary
audience: readers evaluating quantitative research work.

## Abstract (Phase 1)

Phase 1 implements the naive baseline: a dollar-neutral short-front /
long-second VIX futures calendar spread with a 5-day pre-expiry roll, 1 bp
per-leg transaction cost on roll days, trained on 2013–2018 and evaluated
out-of-sample on 2019–2024. Both windows contain a major short-vol
blow-up event (Feb 2018 Volmageddon in train; COVID 2020 and the 2022
bear in test). The naive baseline **loses 12–15% annualized with ~60%
maximum drawdown** in both windows. The cross-correlation of daily
Strategy A returns with negative-VXX returns is 0.61 (as expected for a
VIX-shape trade), confirming the engine is directionally sound — the
negative cumulative PnL is a structural property of the naive
construction, not an implementation bug. This result motivates
Strategies B and C in subsequent phases.

## Why the naive baseline fails

Dollar-neutral calendar P&L per day reduces algebraically to
`r_second − r_front`, where `r` is the daily return of each leg. This is
a *shape* trade, not a *carry* trade: it profits when the term structure
steepens and loses when it flattens or inverts. The positive roll drift
that makes outright short-VIX trades (short VXX, short-front calendar,
etc.) profitable over time is cancelled out by the dollar-neutral
construction. What remains is:

- ≈zero expected daily return during quiet contango (both legs decay at
  similar rates per $ notional)
- large negative realizations during vol spikes (front responds more
  than second → `r_second − r_front` strongly negative)

Accumulated over 11 years including 2015, 2018, 2020, and 2022, the
tails dominate and the strategy bleeds.

This is the same result academics have documented for naive calendar
short-vol trades; see Alexander & Korovilas (2013) for a direct
treatment. The finding motivates two subsequent phases: (i) Strategy B,
a cash-secured put-writing variant that captures the VRP directly rather
than via a calendar shape, and (ii) Strategy C, a conditional variant
that gates exposure on a VRP-threshold signal.

## Strategies

- **Strategy A — VIX Term-Structure Carry (naive baseline).**
  Dollar-neutral short front-month VX, long second-month VX. Rolled 5
  trading days before expiry. **(Implemented in Phase 1.)**
- **Strategy B — Systematic Put-Writing.** Monthly −0.30Δ SPX puts,
  cash-secured; benchmarked against CBOE PUT index. *(Phase 2.)*
- **Strategy C — Conditional VRP Harvester.** Strategy B, gated on
  `VRP_t = IV_t − RV_t > threshold`. *(Phase 3.)*

Overlays (applied to C in Phase 4): VIX regime filter, realized-vol
position scaling, tail-hedge spend.

## Literature

- Bakshi & Kapadia (2003). Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium.
- Carr & Wu (2009). Variance Risk Premiums.
- Alexander, Korovilas (2013). Understanding ETNs on VIX Futures.
- Dew-Becker, Giglio, Le, Rodriguez (2017). The Price of Variance Risk.
- Israelov & Nielsen (2015). Covered Calls Uncovered (AQR).
- Bondarenko (2014). Why Are Put Options So Expensive?

## Data

- SPX, VIX spot: Yahoo Finance (`^GSPC`, `^VIX`).
- VIX futures (VX) settlements: CBOE CDN per-contract historical CSVs
  at `cdn.cboe.com/data/us/futures/market_statistics/historical_data/`.
  Coverage is 2013-present; pre-2013 contracts return HTTP 403.
- CBOE benchmark indices (PUT, BXM): CBOE daily CSVs.

## Methodology notes

- Train 2013-01-01 → 2018-12-31. Test 2019-01-01 → 2024-12-31.
- The original spec targeted 2006 onward. CBOE per-contract VX history
  begins in 2013, which forced the reduction. The 2013+ window still
  contains sufficient regime variety (2015 flash crash, 2018
  Volmageddon, 2020 COVID, 2022 bear) for a meaningful test.
- No test-window parameter tuning.
- Transaction costs: 1 bp per leg per VX roll (baseline). Sensitivity
  sweep 5-30 bps is deferred to the Phase 1 robustness pass.
- Annualization: 252 trading days throughout.

## Reproduce Strategy A

```bash
pip install -e '.[dev]'
python scripts/run_strategy_a.py
python scripts/sanity_vxx.py
```

Outputs land in `reports/strategy_a/` (gitignored — regenerate from
source). `notebooks/01_strategy_a.ipynb` regenerates the same figures
interactively.

## Sanity-check gate

The daily-return correlation between Strategy A and `−VXX` is required
to fall in `[0.3, 0.7]`. Correlation outside this band indicates either
a sign bug in the VX PnL computation or a data-ingestion issue. At
Phase 1 completion: **0.609** (in-band).

## Phase 1 results

### Spec-direction baseline (short front / long second)

| window | Sharpe | ann. return | ann. vol | max DD | DD duration |
|---|---|---|---|---|---|
| train (2013-2018) | −0.74 | −14.7% | 19.9% | −59.6% | 1415 days |
| test (2019-2024)  | −0.60 | −11.7% | 19.5% | −58.2% | 1438 days |

Both Sharpe numbers are negative. The VXX correlation gate (0.609)
confirms the engine is directionally consistent with a short-VIX
construction; the negative cumulative PnL is the structural dollar-
neutral-calendar property described above, not an implementation bug.

### Direction-flip comparison

The spec's "short front / long second" direction is the same calendar
rotated; flipping it produces a genuinely different PnL profile. The
baseline captures `r_second − r_front`; the flipped variant captures
`r_front − r_second`. On the full sample, **the flipped direction is
the profitable one**:

| variant | train Sharpe | test Sharpe | train ret | test ret | train MDD | test MDD |
|---|---|---|---|---|---|---|
| short front / long second (spec) | −0.74 | −0.60 | −14.7% | −11.7% | −60% | −58% |
| **long front / short second** | **+0.60** | **+0.43** | **+11.9%** | **+8.3%** | **−15%** | **−17%** |

Mechanically, the flipped variant captures a small daily positive drift
because front-month VX decays *proportionally faster* than second-month
VX in quiet contango (`r_front` is more negative per $ than `r_second`),
while the vol-spike losses that savage the baseline are much smaller on
the long-front side because the short-second leg absorbs the majority
of the spike. This means the commonly-taught "short front / long second
calendar carries the VIX term structure" framing — which is the spec's
framing — is wrong in sign.

This is the most useful Phase 1 finding: naïve quant-retail intuition
about the VIX calendar has the sign backwards, and a simple replication
reveals it out of sample. Phase 2 (Strategy B put-writing) will sit
alongside the flipped direction as the other candidate workable
construction.

### Transaction-cost sensitivity

Sweep of `tc_bps_per_roll` ∈ {1, 5, 10, 20, 30} for the spec baseline:

| tc (bps) | train Sharpe | test Sharpe | train MDD | test MDD |
|---|---|---|---|---|
| 1  | −0.74 | −0.60 | −60% | −58% |
| 5  | −0.80 | −0.66 | −63% | −61% |
| 10 | −0.87 | −0.73 | −66% | −65% |
| 20 | −1.01 | −0.88 | −72% | −71% |
| 30 | −1.14 | −1.02 | −77% | −76% |

The result is monotone in costs as expected. The delta between 1 bp and
30 bps is ~0.4 Sharpe points — real, but smaller than the ~1.3 Sharpe
gap between the spec direction and the flipped direction at 1 bp. Costs
are not the primary driver of the negative result; the construction is.

## Phase 2 — Strategy B Results

Phase 2 implements the put-writing leg of the study along three tracks:
(i) the published CBOE PUT index as the canonical backtest, (ii) a
Black-Scholes-based synthetic put-writer for pedagogical replication and
pipeline validation, and (iii) a put-spread variant layered on the
synthetic engine.

### CBOE PUT index (canonical, published)

Monthly at-the-money cash-secured puts on SPX, executed per CBOE's PUT
index methodology. The PUT series already bakes in realistic execution
and transaction costs.

| window | Sharpe | ann. return | ann. vol | max DD | α vs SPX (ann.) | β vs SPX |
|---|---|---|---|---|---|---|
| train (2013-2018) | **+0.68** | 6.1% | 9.1% | −15.5% | −0.2% | 0.64 |
| test  (2019-2024) | **+0.70** | 9.9% | 14.1% | −28.9% | +0.4% | 0.62 |

Beta to SPX sits around 0.6 in both windows — the characteristic
put-write risk profile (captures most of equity upside, absorbs most of
equity downside). Annualized alpha is at the noise level, consistent
with the VRP literature: put-writing delivers a *different risk profile*
than SPX, not systematic alpha.

### Synthetic Black-Scholes put-writer (replication)

Monthly −0.30Δ short put using VIX as a 30-day ATM IV proxy, 5 bp
round-trip transaction cost as a fraction of premium.

| window | Sharpe | ann. return | max DD | monthly corr vs PUT |
|---|---|---|---|---|
| train | +0.65 | 4.7% | −15.7% | **0.825** ✓ |
| test  | +0.28 | 3.2% | −26.5% | ″ |

**Sanity gate passes** (monthly correlation 0.825 ≥ 0.6). The synthetic
engine tracks the PUT index in shape and timing; the 2013-2018 numbers
are close to PUT's. In 2019-2024 the synthetic underperforms — this is
consistent with the documented approximations (VIX as ATM IV proxy over-
estimates during crashes, calendar-month cycles vs third-Friday expiries
misalign event timing around COVID). The engine is a replication
artifact, not the primary deliverable; the PUT index is.

### Put-spread variant (short −0.30Δ / long −0.10Δ)

Spread truncates the left tail at the bought-put strike, at the cost of
a smaller net premium:

| variant | train Sharpe | test Sharpe | train ret | test ret | train MDD | test MDD |
|---|---|---|---|---|---|---|
| naked  | +0.65 | +0.28 | 4.7% | 3.2% | −15.7% | −26.5% |
| **spread** | **+1.26** | **+0.60** | 4.1% | 2.9% | **−4.0%** | **−9.0%** |

Spread is the unambiguous winner on the synthetic engine. For ~0.3-0.4%
less annualized return, it cuts max drawdown by a factor of ~3–4 and
roughly doubles Sharpe in both windows. A portfolio allocator would
trivially prefer the spread. This is the Phase 2 headline: the put-
spread construction is the strongest single-trade variant identified
so far in the study, and it is now the natural input for the Phase 4
meta-allocation layer.

### Strategy A vs Strategy B side-by-side (test window, 2019-2024)

| strategy | Sharpe | ann. return | max DD |
|---|---|---|---|
| Strategy A — short front / long second (spec) | −0.60 | −11.7% | −58% |
| Strategy A — long front / short second (flipped) | +0.43 | 8.3% | −17% |
| Strategy B — PUT index (canonical) | +0.70 | 9.9% | −29% |
| Strategy B — synthetic spread (−0.30 / −0.10) | +0.60 | 2.9% | −9% |

Both "working" constructions — flipped VX calendar and put-spread writer
— land in the same Sharpe band (0.4–0.7) with different drawdown
profiles. The VX calendar's 17% MDD comes from term-structure inversion
events; the put-spread's 9% MDD comes from the long-put capping tail
risk. They are different hedges for different failure modes, which is
exactly what makes them good candidates to combine in Phase 4.

## Reproduce Strategy B

```bash
python scripts/run_strategy_b_putindex.py
python scripts/run_strategy_b_synthetic.py
python scripts/run_strategy_b_spread.py
```

## Phase 3 — Strategy C Results

Strategy C gates Strategy B on a VRP signal `VIX_t − RV20_t` (vol
points), taking a position in month `N+1` only when the month-end VRP
of month `N` is at or above a threshold. The hypothesis (Carr-Wu,
Bondarenko, Dew-Becker): the VRP is time-varying, so avoiding low-VRP
months should improve risk-adjusted returns even at the cost of lower
total premium.

### Spec-default baseline (threshold = 2 vol points)

Gates roughly 26% of months.

| variant | gating | train Sharpe | test Sharpe | train MDD | test MDD | active % |
|---|---|---|---|---|---|---|
| naked  | off (B) | +0.65 | +0.28 | −15.7% | −26.5% | 100% |
| naked  | on  (C) | +0.75 | +0.30 | −7.4%  | −26.5% | 74%  |
| spread | off (B) | +1.26 | +0.60 | −4.1%  | −9.0%  | 100% |
| spread | on  (C) | +1.11 | **+0.72** | −2.7% | −9.1%  | 74%  |

Spread + gating at the spec-default threshold jumps test Sharpe from
0.60 to 0.72 while leaving max drawdown roughly unchanged — a
meaningful out-of-sample improvement at a small in-sample Sharpe cost.

### Threshold sensitivity (train-then-test)

Swept thresholds in `[-2, -1, 0, 1, 2, 3, 4, 5, 6]` vol points on
2013-2018 only; picked the train-maximizing Sharpe per variant; then
evaluated the test window (2019-2024) **once** at that threshold.

| variant | train-optimal threshold | train Sharpe @ chosen | test Sharpe @ chosen | test MDD |
|---|---|---|---|---|
| naked  | +1.0 vp | +0.78 | +0.31 | −26.5% |
| spread | −2.0 vp | +1.24 | **+0.81** | −8.5%  |

For the **spread variant**, training Sharpe is monotonically decreasing
in threshold. The train-optimal is at the bottom of the sweep
(threshold = −2 vol points, active ~90% of months — effectively a light
tail filter that removes only the most deeply-negative-VRP months).
This makes structural sense: the spread already has a long-put hedge
absorbing the left tail, so aggressive VRP-gating loses more premium
than it saves. Even so, the held-out test Sharpe improves from 0.60
(ungated) to **0.81** at the train-optimal threshold — a clean
out-of-sample gain from a minimal filter.

For the **naked variant**, training Sharpe is hump-shaped and peaks
around threshold = +1 vol point (active ~81% of months). The test-
window improvement over ungated Strategy B is small (+0.31 vs +0.28).
Naked puts benefit from some gating but the signal is weaker
out-of-sample than for the spread.

### Verdict

The VRP signal adds value, and it adds more value when combined with
the spread's structural tail hedge. For the spread construction, the
lightest filter wins: even filtering only the bottom ~10% of months by
VRP bumps out-of-sample Sharpe from 0.60 to 0.81 with a 10% drop in
active exposure. For the naked construction, a modest filter
(~20% of months gated) helps slightly. Neither variant benefits from
heavy gating — the premium lost dominates the risk saved.

The strongest single construction in the study so far is
**spread + light VRP gate**: test Sharpe +0.81, max DD −8.5%, 90%
capital utilization. That is the natural input for the Phase 4 meta-
allocation layer alongside the flipped VX calendar.

## Reproduce Strategy C

```bash
python scripts/run_strategy_c.py                 # threshold=2 baseline
python scripts/run_strategy_c_sensitivity.py     # train-then-test sweep
```

## Limitations (Phase 1)

- Dollar-neutral continuous rolling is an approximation of how a real
  fund would trade this exposure; intraday slippage, contract-size
  rounding, and margin dynamics make the live path worse than the
  backtest.
- VIX is used as an IV proxy in later phases (variance-swap construct,
  not strictly ATM IV). Flagged in those phases' code.
- Sample contains ≤4 major vol events in 11 years. Backtests of
  short-vol strategies systematically underestimate tail risk; the
  bootstrap and stress-test analyses in Phase 5 are designed to correct
  for this.
- CBOE 2013-start truncation removes GFC 2008 from the sample. 2008
  is the canonical short-vol catastrophe and its absence is a known
  limitation of this replication.
