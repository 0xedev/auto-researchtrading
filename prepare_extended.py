"""
prepare_extended.py — Multi-asset data pipeline for the Karpathy-style ML system.

Downloads:
  • Daily OHLCV (2008+) for 7 macro symbols via Twelve Data API
  • Hourly OHLCV for 4 extended symbols via Twelve Data API
  • BTC/ETH/SOL hourly data reuses the existing CryptoCompare + HL pipeline

Usage:
    python prepare_extended.py                    # download all
    python prepare_extended.py --daily-only       # skip hourly extended
    python prepare_extended.py --symbols GOLD DXY # specific symbols

API key: set TWELVE_DATA_API_KEY env var (free tier: 800 calls/day, 8/min).
Free tier is sufficient — ~27 total calls to download everything.
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Training window for daily data
DAILY_TRAIN_START = "2008-01-01"
DAILY_TRAIN_END = "2024-06-30"

# Match prepare.py splits
TRAIN_START = "2023-06-01"
TRAIN_END = "2024-06-30"
VAL_START = "2024-07-01"
VAL_END = "2025-03-31"
TEST_START = "2025-04-01"

# Twelve Data symbol map → (td_symbol, asset_class, description)
DAILY_SYMBOLS = {
    "GOLD":     ("XAU/USD",  "commodity",  "Gold spot price"),
    "SILVER":   ("XAG/USD",  "commodity",  "Silver spot price"),
    "OIL":      ("WTI/USD",  "commodity",  "WTI crude oil"),
    "SPX":      ("SPX",      "equity",     "S&P 500 Index"),
    "DXY":      ("DXY",      "currency",   "US Dollar Index"),
    "BTC_DAILY":("BTC/USD",  "crypto",     "Bitcoin daily (context)"),
    "TLT":      ("TLT",      "bond",       "iShares 20Y Treasury ETF"),
}

HOURLY_SYMBOLS = {
    "GOLD_1H":   ("XAU/USD",  "commodity"),
    "SILVER_1H": ("XAG/USD",  "commodity"),
    "OIL_1H":    ("WTI/USD",  "commodity"),
    "SPX_1H":    ("SPX",      "equity"),
}

HOURS_PER_YEAR = 8760

# ---------------------------------------------------------------------------
# Twelve Data downloader
# ---------------------------------------------------------------------------

def _td_download(td_symbol: str, interval: str, start_date: str, end_date: str,
                 retries: int = 3) -> pd.DataFrame:
    """
    Download time series from Twelve Data API.
    Returns DataFrame with columns: [timestamp(ms), open, high, low, close, volume]
    """
    if not TWELVE_DATA_KEY:
        print("  [WARN] TWELVE_DATA_API_KEY not set — skipping Twelve Data download")
        return pd.DataFrame()

    all_rows = []
    current_end = end_date
    max_per_call = 5000
    rate_limit_pause = 60.0 / 8  # 8 calls/min on free tier

    while True:
        params = {
            "symbol": td_symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": current_end,
            "outputsize": max_per_call,
            "order": "ASC",
            "apikey": TWELVE_DATA_KEY,
        }

        for attempt in range(retries):
            try:
                resp = requests.get(TWELVE_DATA_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"  [ERROR] {td_symbol} {interval}: {e}")
                    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
                time.sleep(5)

        if data.get("status") == "error":
            msg = data.get("message", "unknown")
            if "rate limit" in msg.lower():
                print(f"  [WARN] Rate limited — sleeping 60s")
                time.sleep(60)
                continue
            print(f"  [ERROR] {td_symbol}: {msg}")
            break

        values = data.get("values", [])
        if not values:
            break

        for bar in values:
            try:
                # Twelve Data returns datetime string like "2024-01-15 10:00:00"
                dt_str = bar["datetime"]
                ts_ms = int(pd.Timestamp(dt_str, tz="UTC").timestamp() * 1000)
                all_rows.append({
                    "timestamp": ts_ms,
                    "open": float(bar.get("open", bar.get("close", 0))),
                    "high": float(bar.get("high", bar.get("close", 0))),
                    "low": float(bar.get("low", bar.get("close", 0))),
                    "close": float(bar["close"]),
                    "volume": float(bar.get("volume", 0)),
                })
            except (KeyError, ValueError):
                continue

        # If we got fewer rows than requested, we're done
        if len(values) < max_per_call:
            break

        # Move window: next call starts after last returned bar
        last_dt = pd.Timestamp(values[-1]["datetime"], tz="UTC")
        if interval == "1day":
            next_start = (last_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            next_start = (last_dt + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        if next_start >= end_date:
            break
        start_date = next_start

        time.sleep(rate_limit_pause)

    if not all_rows:
        return pd.DataFrame()

    df = (pd.DataFrame(all_rows)
          .drop_duplicates("timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True))
    return df


# ---------------------------------------------------------------------------
# Download pipeline
# ---------------------------------------------------------------------------

def download_daily_data(symbols=None, force=False):
    """Download daily OHLCV for macro symbols. Skips cached files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if symbols is None:
        symbols = list(DAILY_SYMBOLS.keys())

    for sym in symbols:
        if sym not in DAILY_SYMBOLS:
            print(f"  [SKIP] Unknown symbol {sym}")
            continue
        td_sym, _, desc = DAILY_SYMBOLS[sym]
        filepath = os.path.join(DATA_DIR, f"{sym}_1d.parquet")
        if os.path.exists(filepath) and not force:
            existing = pd.read_parquet(filepath)
            print(f"  {sym}: cached {len(existing)} daily bars")
            continue

        print(f"  {sym} ({desc}): downloading daily from Twelve Data...")
        df = _td_download(td_sym, "1day", DAILY_TRAIN_START, DAILY_TRAIN_END)
        if df.empty:
            print(f"  {sym}: no data returned — skipping")
            continue

        df["funding_rate"] = 0.0
        df.to_parquet(filepath, index=False)
        print(f"  {sym}: saved {len(df)} daily bars → {filepath}")
        time.sleep(8)  # respect 8 calls/min rate limit


def download_hourly_extended(symbols=None, force=False):
    """Download 1-2yr hourly data for macro symbols (for backtest_extended)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if symbols is None:
        symbols = list(HOURLY_SYMBOLS.keys())

    hourly_start = "2023-01-01"
    hourly_end = "2025-12-31"

    for sym in symbols:
        if sym not in HOURLY_SYMBOLS:
            print(f"  [SKIP] Unknown symbol {sym}")
            continue
        td_sym, _ = HOURLY_SYMBOLS[sym]
        filepath = os.path.join(DATA_DIR, f"{sym}_1h.parquet")
        if os.path.exists(filepath) and not force:
            existing = pd.read_parquet(filepath)
            print(f"  {sym}: cached {len(existing)} hourly bars")
            continue

        print(f"  {sym}: downloading hourly from Twelve Data...")
        df = _td_download(td_sym, "1h", hourly_start, hourly_end)
        if df.empty:
            print(f"  {sym}: no data returned — skipping")
            continue

        df["funding_rate"] = 0.0
        df.to_parquet(filepath, index=False)
        print(f"  {sym}: saved {len(df)} hourly bars → {filepath}")
        time.sleep(8)


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def load_daily_data() -> dict:
    """
    Load daily macro closes for all available symbols.
    Returns: {symbol: np.array of daily close prices (oldest first)}
    """
    result = {}
    for sym in DAILY_SYMBOLS:
        filepath = os.path.join(DATA_DIR, f"{sym}_1d.parquet")
        if not os.path.exists(filepath):
            continue
        df = pd.read_parquet(filepath).sort_values("timestamp")
        if len(df) > 0:
            result[sym] = df["close"].values.astype(float)
    return result


def load_daily_data_up_to(timestamp_ms: int) -> dict:
    """
    Load daily macro closes filtered to <= timestamp_ms.
    Used during training to prevent future leakage.
    """
    result = {}
    for sym in DAILY_SYMBOLS:
        filepath = os.path.join(DATA_DIR, f"{sym}_1d.parquet")
        if not os.path.exists(filepath):
            continue
        df = pd.read_parquet(filepath)
        df = df[df["timestamp"] <= timestamp_ms].sort_values("timestamp")
        if len(df) > 0:
            result[sym] = df["close"].values.astype(float)
    return result


def load_extended_data(split: str = "val") -> dict:
    """
    Load all available hourly OHLCV data for a given split.

    Returns: {symbol: DataFrame}  — same schema as prepare.load_data()
    Columns: timestamp, open, high, low, close, volume, funding_rate
    """
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (VAL_START, VAL_END),
        "test": (TEST_START, "2025-12-31"),
    }
    assert split in splits, f"split must be one of {list(splits.keys())}"
    start_str, end_str = splits[split]
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)

    result = {}

    # Primary: BTC/ETH/SOL from existing prepare.py pipeline
    for sym in ["BTC", "ETH", "SOL"]:
        fp = os.path.join(DATA_DIR, f"{sym}_1h.parquet")
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp)
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
        split_df = df[mask].reset_index(drop=True)
        if len(split_df) > 0:
            result[sym] = split_df

    # Extended: macro hourly symbols
    for sym in HOURLY_SYMBOLS:
        fp = os.path.join(DATA_DIR, f"{sym}_1h.parquet")
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp)
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
        split_df = df[mask].reset_index(drop=True)
        if len(split_df) > 0:
            result[sym] = split_df

    return result


def build_training_dataset(seq_len: int = 168, min_history: int = 200) -> dict:
    """
    Build training dataset from daily data (2008-2024).

    Returns dict with keys:
      'features': np.array (N_samples, seq_len, N_FEATURES)
      'labels': np.array (N_samples, 3)  — next-bar returns for BTC/ETH/SOL
      'timestamps': np.array (N_samples,)
    """
    try:
        from features import compute_feature_matrix, compute_macro_features, N_FEATURES
    except ImportError:
        raise ImportError("features.py not found — create it first")

    print("Building training dataset from daily data...")

    # Load daily data for BTC/ETH/SOL as proxy training symbols
    trading_symbols = ["BTC", "ETH", "SOL"]
    all_hourly = load_extended_data("train")

    if not all_hourly:
        print("[WARN] No hourly training data found. Run download first.")
        return {}

    # Load macro daily closes for feature computation
    macro_daily = load_daily_data()
    print(f"  Macro symbols available: {list(macro_daily.keys())}")

    all_features = []
    all_labels = []
    all_timestamps = []

    for symbol in trading_symbols:
        if symbol not in all_hourly:
            print(f"  [SKIP] {symbol}: no hourly data")
            continue

        df = all_hourly[symbol].copy()
        if len(df) < min_history + seq_len:
            print(f"  [SKIP] {symbol}: insufficient data ({len(df)} bars)")
            continue

        print(f"  {symbol}: computing features for {len(df)} bars...")

        # Build per-bar macro features (use daily data up to bar timestamp)
        # For efficiency, recompute macro features only every 24 bars
        macro_cache = {}
        macro_feats_arr = np.zeros(40, dtype=float)

        feature_matrix, timestamps = compute_feature_matrix(df, macro_daily, seq_len=500)
        n = len(feature_matrix)
        if n < seq_len + 1:
            continue

        closes = df["close"].values.astype(float)

        for i in range(seq_len, n):
            # Target: 1-bar forward return (clipped)
            close_cur = closes[i + 1] if i + 1 < len(closes) else closes[i]
            close_prev = closes[i]
            if close_prev < 1e-10:
                continue
            ret = np.clip((close_cur - close_prev) / close_prev, -0.1, 0.1)

            # Feature sequence: last seq_len bars
            feat_seq = feature_matrix[i - seq_len:i]

            all_features.append(feat_seq)
            all_labels.append(ret)
            all_timestamps.append(timestamps[i])

    if not all_features:
        print("[WARN] No training samples generated.")
        return {}

    features = np.array(all_features, dtype=float)
    labels = np.array(all_labels, dtype=float)
    timestamps_arr = np.array(all_timestamps)

    print(f"  Dataset: {features.shape[0]} samples, seq_len={seq_len}, n_features={features.shape[2]}")
    return {
        "features": features,
        "labels": labels,
        "timestamps": timestamps_arr,
    }


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def print_data_status():
    """Print summary of available cached data."""
    print(f"\nCache directory: {DATA_DIR}")
    print("─" * 60)

    total_rows = 0
    for sym in list(DAILY_SYMBOLS.keys()) + list(HOURLY_SYMBOLS.keys()) + ["BTC", "ETH", "SOL"]:
        for interval in ["1d", "1h"]:
            fp = os.path.join(DATA_DIR, f"{sym}_{interval}.parquet")
            if os.path.exists(fp):
                df = pd.read_parquet(fp)
                ts_min = pd.to_datetime(df["timestamp"].min(), unit="ms", utc=True)
                ts_max = pd.to_datetime(df["timestamp"].max(), unit="ms", utc=True)
                print(f"  {sym:12s} [{interval}]: {len(df):6d} bars  "
                      f"{ts_min.strftime('%Y-%m-%d')} → {ts_max.strftime('%Y-%m-%d')}")
                total_rows += len(df)

    print(f"─" * 60)
    print(f"  Total rows: {total_rows:,}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download extended multi-asset data")
    parser.add_argument("--daily-only", action="store_true", help="Skip hourly extended download")
    parser.add_argument("--hourly-only", action="store_true", help="Skip daily download")
    parser.add_argument("--symbols", nargs="+", default=None, help="Specific symbols to download")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--status", action="store_true", help="Show data status and exit")
    args = parser.parse_args()

    if args.status:
        print_data_status()
        sys.exit(0)

    os.makedirs(DATA_DIR, exist_ok=True)

    if not TWELVE_DATA_KEY:
        print("[WARN] TWELVE_DATA_API_KEY not set.")
        print("  Set it with: export TWELVE_DATA_API_KEY=your_key_here")
        print("  Free tier: https://twelvedata.com (800 calls/day)")
        print()
        print("  BTC/ETH/SOL data still downloads via CryptoCompare (no key needed).")
        print("  Run: python prepare.py  for BTC/ETH/SOL")
        print()

    # Download BTC/ETH/SOL via existing prepare.py pipeline
    print("Step 1: BTC/ETH/SOL hourly data (CryptoCompare + Hyperliquid)")
    try:
        from prepare import download_data
        download_data()
    except Exception as e:
        print(f"  [WARN] prepare.py download failed: {e}")
    print()

    if not args.hourly_only and TWELVE_DATA_KEY:
        print("Step 2: Daily macro data (2008+) via Twelve Data")
        syms = [s for s in (args.symbols or []) if s in DAILY_SYMBOLS] or None
        download_daily_data(symbols=syms, force=args.force)
        print()

    if not args.daily_only and TWELVE_DATA_KEY:
        print("Step 3: Hourly macro data for backtest_extended")
        syms = [s for s in (args.symbols or []) if s in HOURLY_SYMBOLS] or None
        download_hourly_extended(symbols=syms, force=args.force)
        print()

    print("Step 4: Data status")
    print_data_status()
    print("Done. Ready for training.")
