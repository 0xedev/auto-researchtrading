"""
Deriv Synthetic Index Data Generator
=====================================
Generates 4-year OHLCV parquet files in ~/.cache/autotrader/data/ for 11
Deriv-style synthetic instruments the V2 model has never trained on.
Both _1h and _15m files are produced so bundle_intraday_core works.

Spike/jump probabilities are expressed as events-per-calendar-day so that
both timeframes produce equivalent regimes (avoids 15m compounding explosion).

Instrument families
-------------------
  DERIV_V10/V25/V50/V75/V100   Pure GBM, σ_ann = index_number / 100
  DERIV_CRASH300 / BOOM300     GBM + Poisson spikes, 0.05 events/day
  DERIV_JUMP25 / JUMP100       GBM + bidir jumps, 0.25 / 0.1 events/day
  DERIV_STEP                   Fixed ±step discrete random walk
  DERIV_RB100                  Mean-reverting range-break process

Usage
-----
  uv run prepare_deriv_synth.py [--list] [--symbol SYM ...]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────
START_DATE = "2022-04-01"
END_DATE   = "2026-04-01"       # exclusive; last bar = 2026-03-31 23:00 / 23:45
BASE_PRICE = 1_000.0

DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "data")

TIMEFRAMES = {
    "1h":  (pd.date_range(START_DATE, END_DATE, freq="1h",    tz="UTC", inclusive="left"), 1.0  / 8_760.0,   24.0),
    "15m": (pd.date_range(START_DATE, END_DATE, freq="15min", tz="UTC", inclusive="left"), 0.25 / 8_760.0,   96.0),
}
# dt_ann: bar duration in years; bars_per_day: calendar bars per day


# ─── OHLCV helpers ────────────────────────────────────────────────────────────

def _make_ohlcv(timestamps: pd.DatetimeIndex, closes: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]

    spread = np.abs(closes - opens) * 0.6 + closes * 0.0003
    high = np.maximum(opens, closes) + rng.exponential(np.maximum(spread * 0.5, 1e-9))
    low  = np.minimum(opens, closes) - rng.exponential(np.maximum(spread * 0.5, 1e-9))
    volume = rng.lognormal(mean=6.0, sigma=0.8, size=n)
    ts_ms = (timestamps.astype("int64") // 1_000_000).values

    return pd.DataFrame({
        "timestamp":    ts_ms,
        "open":         opens,
        "high":         high,
        "low":          low,
        "close":        np.maximum(closes, 1e-3),
        "volume":       volume,
        "funding_rate": np.zeros(n, dtype="float64"),
    })


def _gbm(n: int, sigma_ann: float, dt_ann: float, rng: np.random.Generator) -> np.ndarray:
    drift     = (-0.5 * sigma_ann ** 2) * dt_ann
    diffusion = sigma_ann * np.sqrt(dt_ann) * rng.standard_normal(n)
    return BASE_PRICE * np.exp(np.cumsum(drift + diffusion))


# ─── Instrument generators ────────────────────────────────────────────────────

def gen_volatility(
    sigma_ann: float,
    seed: int,
    *,
    timestamps: pd.DatetimeIndex,
    dt_ann: float,
    bars_per_day: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = _gbm(len(timestamps), sigma_ann, dt_ann, rng)
    return _make_ohlcv(timestamps, closes, rng)


def gen_crash_boom(
    spikes_per_day: float,
    direction: float,
    seed: int,
    *,
    timestamps: pd.DatetimeIndex,
    dt_ann: float,
    bars_per_day: float,
) -> pd.DataFrame:
    """direction: -1 = crash (downward spikes), +1 = boom (upward spikes)."""
    rng = np.random.default_rng(seed)
    sigma_ann = 0.50
    n = len(timestamps)
    p_spike = spikes_per_day / bars_per_day

    log_rets = (-0.5 * sigma_ann ** 2) * dt_ann + sigma_ann * np.sqrt(dt_ann) * rng.standard_normal(n)
    spike_mask = rng.random(n) < p_spike
    log_rets  += direction * spike_mask * rng.exponential(0.03, n)   # avg 3% spike

    closes = BASE_PRICE * np.exp(np.cumsum(log_rets))
    return _make_ohlcv(timestamps, closes, rng)


def gen_jump(
    jumps_per_day: float,
    jump_sigma: float,
    seed: int,
    *,
    timestamps: pd.DatetimeIndex,
    dt_ann: float,
    bars_per_day: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma_ann = 0.35
    n = len(timestamps)
    p_jump = jumps_per_day / bars_per_day

    log_rets = (-0.5 * sigma_ann ** 2) * dt_ann + sigma_ann * np.sqrt(dt_ann) * rng.standard_normal(n)
    jump_mask  = rng.random(n) < p_jump
    jump_signs = rng.choice([-1.0, 1.0], n)
    log_rets  += jump_mask * jump_signs * rng.exponential(jump_sigma, n)

    closes = BASE_PRICE * np.exp(np.cumsum(log_rets))
    return _make_ohlcv(timestamps, closes, rng)


def gen_step(
    step_size: float,
    seed: int,
    *,
    timestamps: pd.DatetimeIndex,
    dt_ann: float,
    bars_per_day: float,
) -> pd.DataFrame:
    """Fixed ±step discrete random walk (Deriv Step Index)."""
    rng = np.random.default_rng(seed)
    n   = len(timestamps)
    steps  = rng.choice([-step_size, step_size], n).astype("float64")
    closes = BASE_PRICE + np.cumsum(steps)
    closes = np.maximum(closes, step_size)
    return _make_ohlcv(timestamps, closes, rng)


def gen_range_break(
    range_half_width: float,
    break_prob_per_day: float,
    sigma_inner: float,
    seed: int,
    *,
    timestamps: pd.DatetimeIndex,
    dt_ann: float,
    bars_per_day: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = len(timestamps)
    closes = np.empty(n)
    closes[0] = BASE_PRICE
    center    = BASE_PRICE
    p_break   = break_prob_per_day / bars_per_day

    for i in range(1, n):
        cur = closes[i - 1]
        if rng.random() < p_break:
            center = cur
        dev = (cur - center) / max(center, 1e-9)
        lr  = -0.12 * dev + rng.normal(0.0, sigma_inner * np.sqrt(dt_ann))
        closes[i] = max(cur * np.exp(lr), 1e-3)

    return _make_ohlcv(timestamps, closes, rng)


# ─── Instrument specs ─────────────────────────────────────────────────────────
# (generator_fn, *positional_args_without_kw_params)
# Kwargs timestamps / dt_ann / bars_per_day are injected at call time.

INSTRUMENTS: dict[str, tuple] = {
    "DERIV_V10":      (gen_volatility,   0.10, 101),
    "DERIV_V25":      (gen_volatility,   0.25, 102),
    "DERIV_V50":      (gen_volatility,   0.50, 103),
    "DERIV_V75":      (gen_volatility,   0.75, 104),
    "DERIV_V100":     (gen_volatility,   1.00, 105),
    # 0.05 spike events/day ≈ 1 spike every 20 days
    "DERIV_CRASH300": (gen_crash_boom,   0.05, -1.0, 201),
    "DERIV_BOOM300":  (gen_crash_boom,   0.05, +1.0, 202),
    # 0.25 jumps/day ≈ 1 jump per 4 days; 0.10/day ≈ 1 per 10 days
    "DERIV_JUMP25":   (gen_jump,   0.25, 0.035, 301),
    "DERIV_JUMP100":  (gen_jump,   0.10, 0.025, 302),
    "DERIV_STEP":     (gen_step,   0.10, 401),
    "DERIV_RB100":    (gen_range_break,  0.02, 1.0 / 100, 0.20, 501),
}


def generate(symbol: str, timeframe: str) -> pd.DataFrame:
    timestamps, dt_ann, bars_per_day = TIMEFRAMES[timeframe]
    fn, *args = INSTRUMENTS[symbol]
    return fn(*args, timestamps=timestamps, dt_ann=dt_ann, bars_per_day=bars_per_day)


def save(symbol: str, timeframe: str, df: pd.DataFrame) -> Path:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = Path(DATA_DIR) / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Deriv synthetic OHLCV parquet files")
    parser.add_argument("--list",      action="store_true",  help="Print available symbols and exit")
    parser.add_argument("--symbol",    action="append",      help="Generate specific symbol(s); default = all")
    parser.add_argument("--timeframe", choices=["1h", "15m", "both"], default="both")
    args = parser.parse_args()

    if args.list:
        for sym in sorted(INSTRUMENTS):
            print(sym)
        return

    targets = args.symbol if args.symbol else sorted(INSTRUMENTS)
    tfs     = ["1h", "15m"] if args.timeframe == "both" else [args.timeframe]

    for tf in tfs:
        ts, _, _ = TIMEFRAMES[tf]
        print(f"Generating {len(targets)} symbol(s) @ {tf}  ({len(ts):,} bars each)")

    print(f"Output: {DATA_DIR}\n")

    for sym in targets:
        if sym not in INSTRUMENTS:
            print(f"  SKIP  {sym}  (unknown)")
            continue
        for tf in tfs:
            df   = generate(sym, tf)
            path = save(sym, tf, df)
            cs, ce = df["close"].iloc[0], df["close"].iloc[-1]
            print(f"  OK    {sym:<20}  {tf}   rows={len(df):>7,}  close: {cs:>10.3f} → {ce:>10.3f}  [{path.name}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
