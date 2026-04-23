from __future__ import annotations

import numpy as np
import pandas as pd

from prepare import ASSET_CLASS, CONTEXT_COLUMNS, calculate_features


TIMEFRAME_HOURS = {
    "15m": 0.25,
    "1h": 1.0,
    "4h": 4.0,
    "1d": 24.0,
}


def timeframe_to_hours(timeframe: str) -> float:
    if timeframe not in TIMEFRAME_HOURS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return TIMEFRAME_HOURS[timeframe]


def bars_for_hours(hours: float, bar_interval_hours: float) -> int:
    return max(1, int(round(hours / max(bar_interval_hours, 1e-6))))


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0).rolling(window).mean()
    losses = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gains / losses.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_clock_features(
    df: pd.DataFrame,
    timeframe: str,
    symbol: str | None = None,
    feature_profile: str = "price_context_plus",
) -> pd.DataFrame:
    base = calculate_features(df, timeframe=timeframe, symbol=symbol, feature_profile=feature_profile).copy()
    bar_hours = float(base.get("bar_interval_hours", pd.Series([timeframe_to_hours(timeframe)])).iloc[0] or timeframe_to_hours(timeframe))
    close = base["close"]
    high = base["high"]
    low = base["low"]
    volume = base["volume"]
    funding_rate = base["funding_rate"] if "funding_rate" in base.columns else pd.Series(0.0, index=base.index)

    horizons = {
        "1h": 1.0,
        "6h": 6.0,
        "24h": 24.0,
        "72h": 72.0,
        "168h": 168.0,
    }
    for label, hours in horizons.items():
        bars = bars_for_hours(hours, bar_hours)
        base[f"clock_ret_{label}"] = close.pct_change(bars).fillna(0.0)

    for label, hours in {"6h": 6.0, "24h": 24.0}.items():
        bars = bars_for_hours(hours, bar_hours)
        base[f"clock_rsi_{label}"] = _rsi(close, bars).fillna(50.0)

    bb_bars = bars_for_hours(24.0, bar_hours)
    sma = close.rolling(bb_bars).mean()
    std = close.rolling(bb_bars).std()
    base["clock_bb_width_24h"] = ((4.0 * std) / sma.replace(0, 1e-10)).fillna(0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_bars = bars_for_hours(24.0, bar_hours)
    atr = tr.rolling(atr_bars).mean()
    base["clock_atr_pct_24h"] = (atr / close.replace(0, 1e-10)).fillna(0.0)

    range_bars = bars_for_hours(24.0, bar_hours)
    rolling_high = high.rolling(range_bars).max()
    rolling_low = low.rolling(range_bars).min()
    base["clock_dist_to_high_24h"] = ((rolling_high - close) / close.replace(0, 1e-10)).fillna(0.0)
    base["clock_dist_to_low_24h"] = ((close - rolling_low) / close.replace(0, 1e-10)).fillna(0.0)
    base["clock_range_position_24h"] = (
        base["clock_dist_to_low_24h"] / (base["clock_dist_to_low_24h"] + base["clock_dist_to_high_24h"] + 1e-8)
    ).fillna(0.5)

    ema_bars = bars_for_hours(200.0, bar_hours)
    ema = close.ewm(span=max(ema_bars, 2), adjust=False).mean()
    base["clock_ema_200h_dist"] = ((close - ema) / close.replace(0, 1e-10)).fillna(0.0)

    vol_bars = bars_for_hours(24.0, bar_hours)
    vol_mean = volume.rolling(vol_bars).mean().replace(0, 1e-10)
    base["clock_vol_24h"] = volume.rolling(vol_bars).mean().fillna(0.0)
    base["clock_vol_ratio_24h"] = (volume / vol_mean).fillna(0.0)
    base["clock_volume_trend_24h"] = (
        base["clock_vol_ratio_24h"] - base["clock_vol_ratio_24h"].ewm(span=max(vol_bars, 2), adjust=False).mean()
    ).fillna(0.0)

    tp = (high + low + close) / 3.0
    vwap_num = (tp * volume).rolling(vol_bars).sum()
    vwap_den = volume.rolling(vol_bars).sum().replace(0, 1e-10)
    vwap = vwap_num / vwap_den
    base["clock_vwap_dist_24h"] = ((close - vwap) / close.replace(0, 1e-10)).fillna(0.0)
    base["clock_volatility_24h"] = close.pct_change().rolling(vol_bars).std().fillna(0.0)

    base["funding_rate"] = funding_rate.fillna(0.0)
    base["asset_class"] = ASSET_CLASS.get(symbol, 0) if symbol is not None else base.get("asset_class", 0)
    base["bar_interval_hours"] = bar_hours
    for col in CONTEXT_COLUMNS:
        if col not in base.columns:
            base[col] = 0.0

    return base.replace([np.inf, -np.inf], 0.0).fillna(0.0)
