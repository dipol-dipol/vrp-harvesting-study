"""VIX futures (VX) continuous front/second-month series.

Sources CBOE CFE daily settlements by contract, rolls into continuous
series. The roll schedule follows CBOE's SOQ rule: expiration is the
Wednesday 30 days prior to the 3rd Friday of the month FOLLOWING the
contract's designated expiry month.

Data source discovery (tried in order):
  1. https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{year}_VX.csv
     — the canonical per-year bulk file. Confirmed reachable as of 2025.
  2. If (1) is unreachable, set VX_CSV_OVERRIDE to a directory holding
     CFE_{year}_VX.csv files (one per year) with columns:
         Trade Date, Futures, Open, High, Low, Close, Settle, Change, Total Volume
     The fetcher will read those files directly without any HTTP call.
"""
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from . import cache

_KEY_PREFIX = "vx_futures"

_CONTRACT_CODE_MAP = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}


# ---------------------------------------------------------------------------
# Roll calendar — pure logic, no network
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.weekday()) % 7  # Mon=0 … Fri=4
    first_friday = first + pd.Timedelta(days=offset)
    return first_friday + pd.Timedelta(days=14)


def vx_expiration(contract_year: int, contract_month: int) -> pd.Timestamp:
    """Expiration date for a VX contract with designated expiry (year, month).

    CBOE rule: settle on the Wednesday 30 days prior to the 3rd Friday
    of the MONTH AFTER the contract's designated month.
    """
    next_month = (
        pd.Timestamp(year=contract_year, month=contract_month, day=1)
        + pd.offsets.MonthBegin(1)
    )
    tf = _third_friday(next_month.year, next_month.month)
    return (tf - pd.Timedelta(days=30)).normalize()


def build_roll_calendar(start: str, end: str) -> pd.DataFrame:
    """Per business day in [start, end]: front and second contract expiries.

    Returns a DataFrame indexed by business date with columns
    ``front_expiry`` and ``second_expiry``.
    """
    idx = pd.bdate_range(start, end)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    # Generate enough expiration dates to cover the range.
    expirations = []
    y, m = s.year, s.month
    while True:
        exp = vx_expiration(y, m)
        expirations.append(exp)
        if exp > e + pd.Timedelta(days=90):
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    expirations = sorted(set(expirations))

    front_exp = []
    second_exp = []
    for d in idx:
        future = [x for x in expirations if x > d][:2]
        if len(future) < 2:
            front_exp.append(pd.NaT)
            second_exp.append(pd.NaT)
        else:
            front_exp.append(future[0])
            second_exp.append(future[1])

    return pd.DataFrame(
        {"front_expiry": front_exp, "second_expiry": second_exp},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Data fetch — CBOE per-year bulk CSVs
# ---------------------------------------------------------------------------

def _candidate_urls(year: int) -> list[str]:
    return [
        f"https://cdn.cboe.com/resources/futures/archive/volume-and-price/CFE_{year}_VX.csv",
    ]


def _fetch_year(year: int) -> pd.DataFrame:
    """Fetch raw per-contract daily VX data for *year*.

    Falls back to local CSV if ``VX_CSV_OVERRIDE`` env var is set.
    """
    override = os.environ.get("VX_CSV_OVERRIDE")
    if override:
        path = Path(override) / f"CFE_{year}_VX.csv"
        if path.exists():
            return pd.read_csv(path)

    last_exc: Optional[Exception] = None
    for url in _candidate_urls(year):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 100:
                return pd.read_csv(StringIO(r.text))
            last_exc = RuntimeError(f"HTTP {r.status_code} at {url}")
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Could not fetch CFE VX data for {year}. Last error: {last_exc}. "
        f"Place a CSV at ${{VX_CSV_OVERRIDE}}/CFE_{year}_VX.csv and retry."
    )


def _parse_contract_ym(futures_code: str) -> tuple[int, int]:
    """Parse contract code to (year, month).

    Handles formats seen in CBOE historical files:
      'VX/H20', 'VX H20', 'VXH20', 'H20', 'H2020'

    Two-digit year suffix is interpreted as 20xx.
    """
    inv = {v: k for k, v in _CONTRACT_CODE_MAP.items()}
    code = (
        futures_code.strip()
        .upper()
        .replace("VX/", "")
        .replace("VX ", "")
        .replace("VX", "")
        .strip()
    )
    letter = code[0]
    year_suffix = code[1:]
    if len(year_suffix) == 2:
        year = 2000 + int(year_suffix)
    elif len(year_suffix) == 4:
        year = int(year_suffix)
    else:
        raise ValueError(f"Cannot parse contract year from '{futures_code}'")
    month = inv[letter]
    return year, month


# ---------------------------------------------------------------------------
# Continuous series builder
# ---------------------------------------------------------------------------

def load_vx_continuous(
    start: str = "2006-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Continuous front/second VX settlement series.

    Returns a DataFrame indexed by trade date with columns:
        front_settle, second_settle, front_expiry, second_expiry,
        days_to_front_expiry

    The roll is mechanical: on each trade date the "front" contract is
    whichever listed VX contract has the nearest expiry strictly *after*
    that date, and "second" is the next one.

    Parameters
    ----------
    start:
        First date to include (inclusive).
    end:
        Last date to include (inclusive). Defaults to today.
    use_cache:
        Read/write a local parquet cache keyed by (start, end).
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    key = f"{_KEY_PREFIX}_continuous__{start}__{end}"

    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached

    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    frames = []
    for y in years:
        try:
            frames.append(_fetch_year(y))
        except Exception as exc:
            print(f"VX fetch failed for {y}: {exc}")

    if not frames:
        raise RuntimeError(
            "No VX data fetched. Set VX_CSV_OVERRIDE to a directory with "
            "CFE_{year}_VX.csv files."
        )

    raw = pd.concat(frames, ignore_index=True)
    # Normalise column names
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    date_col = "trade_date" if "trade_date" in raw.columns else "date"
    raw["trade_date"] = pd.to_datetime(raw[date_col])

    # Settle column varies by year/source
    settle_col = next(
        (c for c in ("settle", "settlement_price", "settle_price", "close")
         if c in raw.columns),
        None,
    )
    if settle_col is None:
        raise RuntimeError(
            f"No settle column found. Available columns: {raw.columns.tolist()}"
        )

    raw = raw[["trade_date", "futures", settle_col]].dropna()
    raw = raw.rename(columns={settle_col: "settle"})

    raw["expiry"] = raw["futures"].map(
        lambda c: vx_expiration(*_parse_contract_ym(c))
    )

    raw = raw.sort_values(["trade_date", "expiry"])

    out_rows = []
    for d, grp in raw.groupby("trade_date"):
        future = grp[grp["expiry"] > d].sort_values("expiry")
        if len(future) < 2:
            continue
        front = future.iloc[0]
        second = future.iloc[1]
        out_rows.append({
            "date": d,
            "front_settle": float(front["settle"]),
            "second_settle": float(second["settle"]),
            "front_expiry": front["expiry"],
            "second_expiry": second["expiry"],
            "days_to_front_expiry": (front["expiry"] - d).days,
        })

    out = pd.DataFrame(out_rows).set_index("date").sort_index()
    out = out.loc[start:end]

    cache.save(key, out)
    return out
