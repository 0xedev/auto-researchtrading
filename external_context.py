from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


CONTEXT_CACHE_DIR = Path.home() / ".cache" / "autotrader" / "context"
FRED_SERIES_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
ALT_FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"

CONTEXT_COLUMNS = [
    "macro_event_flag",
    "context_sentiment",
    "major_market_event_flag",
    "macro_event_intensity",
    "context_sentiment_shock",
    "major_market_event_intensity",
    "cross_asset_stress",
]


FOMC_DECISION_DATES = {
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
}


@dataclass(frozen=True)
class MajorEvent:
    start: str
    end: str
    scope: str
    value: float
    label: str


MAJOR_EVENTS = [
    MajorEvent("2020-03-09", "2020-03-18", "all", -1.0, "covid_crash"),
    MajorEvent("2022-05-09", "2022-05-13", "crypto", -1.0, "terra_implosion"),
    MajorEvent("2022-06-13", "2022-06-15", "all", -0.5, "inflation_shock"),
    MajorEvent("2022-11-08", "2022-11-12", "crypto", -1.0, "ftx_collapse"),
    MajorEvent("2023-03-10", "2023-03-15", "all", -0.5, "svb_liquidity_shock"),
    MajorEvent("2024-01-10", "2024-01-12", "crypto", 1.0, "btc_etf_approval"),
    MajorEvent("2024-04-19", "2024-04-21", "crypto", 0.75, "btc_halving"),
    MajorEvent("2024-08-05", "2024-08-06", "all", -0.5, "global_risk_off"),
]


def _ensure_cache_dir() -> None:
    CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_timestamps(timestamps) -> pd.DatetimeIndex:
    raw = pd.to_numeric(pd.Series(timestamps), errors="coerce").astype("Int64")
    non_na = raw.dropna()
    unit = "ms" if (not non_na.empty and non_na.gt(10**11).any()) else "s"
    return pd.DatetimeIndex(pd.to_datetime(raw, unit=unit, utc=True))


def _cached_text(path: Path, url: str, ttl_hours: int = 24) -> str | None:
    _ensure_cache_dir()
    stale = True
    if path.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds() / 3600.0
        stale = age_hours > ttl_hours
    if stale:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            path.write_text(resp.text)
        except Exception:
            if not path.exists():
                return None
    try:
        return path.read_text()
    except Exception:
        return None


def _load_fred_series(series_id: str) -> pd.Series:
    cache_path = CONTEXT_CACHE_DIR / f"{series_id}.csv"
    raw_text = _cached_text(cache_path, FRED_SERIES_URL.format(series_id=series_id), ttl_hours=24)
    if not raw_text:
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(cache_path)
    except Exception:
        return pd.Series(dtype=float)
    if "observation_date" not in df.columns or series_id not in df.columns:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(df["observation_date"], utc=True, errors="coerce").dt.floor("1D")
    values = pd.to_numeric(df[series_id], errors="coerce")
    return pd.Series(values.values, index=ts).dropna().sort_index()


def _load_fng_series() -> pd.Series:
    cache_path = CONTEXT_CACHE_DIR / "alternative_me_fng.json"
    raw_text = _cached_text(cache_path, ALT_FNG_URL, ttl_hours=24)
    if not raw_text:
        return pd.Series(dtype=float)
    try:
        payload = json.loads(raw_text)
        rows = payload.get("data", [])
    except Exception:
        return pd.Series(dtype=float)
    if not rows:
        return pd.Series(dtype=float)
    ts = pd.to_datetime([int(row["timestamp"]) for row in rows], unit="s", utc=True).floor("1D")
    values = pd.to_numeric([row.get("value", np.nan) for row in rows], errors="coerce")
    normalized = ((values / 50.0) - 1.0).clip(-1.0, 1.0)
    return pd.Series(normalized, index=ts).dropna().sort_index()


def _first_friday(year: int, month: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != 4:
        current += timedelta(days=1)
    return current


def _nfp_dates(start_year: int = 2017, end_year: int = 2026) -> set[str]:
    dates = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.add(_first_friday(year, month).isoformat())
    return dates


NFP_RELEASE_DATES = _nfp_dates()


def _build_macro_event_series(day_index: pd.DatetimeIndex) -> pd.Series:
    values = pd.Series(0.0, index=day_index)
    day_strings = day_index.strftime("%Y-%m-%d")
    fomc_mask = day_strings.isin(FOMC_DECISION_DATES)
    nfp_mask = day_strings.isin(NFP_RELEASE_DATES)
    values.loc[fomc_mask] += 1.0
    values.loc[nfp_mask] += 0.5
    return values.clip(0.0, 1.5)


def _event_scope_matches(scope: str, symbol: str | None, asset_class: int | None) -> bool:
    if scope == "all":
        return True
    if scope == "crypto":
        return asset_class == 0
    if scope == "macro":
        return asset_class in {1, 2}
    if scope == "commodity":
        return asset_class == 1
    if scope == "equity":
        return asset_class == 2
    return scope == (symbol or "")


def _build_major_event_series(day_index: pd.DatetimeIndex, symbol: str | None, asset_class: int | None) -> pd.Series:
    values = pd.Series(0.0, index=day_index)
    for event in MAJOR_EVENTS:
        if not _event_scope_matches(event.scope, symbol, asset_class):
            continue
        start = pd.Timestamp(event.start, tz="UTC")
        end = pd.Timestamp(event.end, tz="UTC")
        mask = (day_index >= start) & (day_index <= end)
        values.loc[mask] += event.value
    return values.clip(-1.5, 1.5)


def _build_sentiment_series(day_index: pd.DatetimeIndex, asset_class: int | None) -> pd.Series:
    if asset_class == 0:
        sentiment = _load_fng_series()
    else:
        vix = _load_fred_series("VIXCLS")
        sentiment = ((20.0 - vix) / 20.0).clip(-1.0, 1.0)
    if sentiment.empty:
        return pd.Series(0.0, index=day_index)
    return sentiment.reindex(day_index, method="ffill").fillna(0.0)


def _build_cross_asset_stress_series(day_index: pd.DatetimeIndex, asset_class: int | None) -> pd.Series:
    vix = _load_fred_series("VIXCLS")
    if vix.empty:
        vix_component = pd.Series(0.0, index=day_index)
    else:
        vix_component = ((vix - 20.0) / 10.0).clip(-1.5, 1.5)
        vix_component = vix_component.reindex(day_index, method="ffill").fillna(0.0)

    if asset_class == 0:
        sentiment = _load_fng_series()
        if sentiment.empty:
            sentiment_component = pd.Series(0.0, index=day_index)
        else:
            sentiment_component = (-sentiment).reindex(day_index, method="ffill").fillna(0.0)
        stress = 0.55 * vix_component + 0.45 * sentiment_component
    else:
        stress = vix_component

    return stress.clip(-2.0, 2.0)


def build_context_features(timestamps, symbol: str | None = None, asset_class: int | None = None) -> pd.DataFrame:
    ts = _normalize_timestamps(timestamps)
    if len(ts) == 0:
        return pd.DataFrame(columns=CONTEXT_COLUMNS)
    day_index = pd.DatetimeIndex(ts.floor("1D"))
    unique_days = pd.DatetimeIndex(sorted(day_index.unique()))

    macro_events = _build_macro_event_series(unique_days)
    sentiment = _build_sentiment_series(unique_days, asset_class)
    market_events = _build_major_event_series(unique_days, symbol, asset_class)
    macro_event_intensity = macro_events.rolling(3, min_periods=1).sum().clip(0.0, 3.0)
    major_event_intensity = market_events.rolling(3, min_periods=1).sum().clip(-3.0, 3.0)
    sentiment_shock = sentiment.diff().fillna(0.0).clip(-1.5, 1.5)
    cross_asset_stress = _build_cross_asset_stress_series(unique_days, asset_class)

    base = pd.DataFrame(
        {
            "macro_event_flag": macro_events,
            "context_sentiment": sentiment.reindex(unique_days).fillna(0.0),
            "major_market_event_flag": market_events,
            "macro_event_intensity": macro_event_intensity.reindex(unique_days).fillna(0.0),
            "context_sentiment_shock": sentiment_shock.reindex(unique_days).fillna(0.0),
            "major_market_event_intensity": major_event_intensity.reindex(unique_days).fillna(0.0),
            "cross_asset_stress": cross_asset_stress.reindex(unique_days).fillna(0.0),
        },
        index=unique_days,
    )
    joined = base.reindex(day_index).reset_index(drop=True).fillna(0.0)
    return joined
