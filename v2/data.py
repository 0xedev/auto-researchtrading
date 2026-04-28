from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from market_regime import build_regime_frame
import prepare

from .features import calculate_clock_features, timeframe_to_hours
from .manifests import BUNDLE_MANIFESTS


def _base_split(split: str) -> str:
    return split[:-4] if split.endswith("_15m") else split


def split_for_timeframe(split: str, timeframe: str) -> str:
    base = _base_split(split)
    if timeframe == "15m":
        return base if base.endswith("_15m") else f"{base}_15m"
    return base


def _split_bounds(split: str) -> tuple[str, str]:
    splits = {
        "train": (prepare.TRAIN_START, prepare.TRAIN_END),
        "val": (prepare.VAL_START, prepare.VAL_END),
        "test": (prepare.TEST_START, prepare.TEST_END),
        "robustness": (prepare.ROBUST_START, prepare.ROBUST_END),
        "val_15m": (prepare.VAL_START, prepare.VAL_END),
        "train_15m": (prepare.TRAIN_START, prepare.TRAIN_END),
        "oos": ("2025-01-01", "2025-12-31"),
        "oos_15m": ("2025-01-01", "2025-12-31"),
        "2026q1": ("2026-01-01", "2026-03-31"),
        "2026q1_15m": ("2026-01-01", "2026-03-31"),
        "newasset_oos2y": ("2024-04-01", "2026-03-31"),
        "newasset_oos2y_15m": ("2024-04-01", "2026-03-31"),
        "deriv_oss4y": ("2022-04-01", "2026-04-01"),
        "deriv_oss4y_15m": ("2022-04-01", "2026-04-01"),
        "holdout": (prepare.HOLDOUT_START, prepare.HOLDOUT_END),
        "holdout_15m": (prepare.HOLDOUT_START, prepare.HOLDOUT_END),
        # Live trading window — starts before holdout end to include context bars.
        # Add fresh bars to the parquet cache then prepare("live") to get current signals.
        "live": ("2025-07-01", "2030-12-31"),
        "live_15m": ("2025-07-01", "2030-12-31"),
        # Extended training window: includes val period (consumed by research, safe to train on).
        # Use newasset_oos2y as the OOS gate for any model trained on this split.
        "train_extended": (prepare.TRAIN_START, prepare.VAL_END),
        "train_extended_15m": (prepare.TRAIN_START, prepare.VAL_END),
    }
    return splits[split]


def _default_symbol_universe(timeframe: str) -> list[str]:
    if timeframe == "15m":
        return list(prepare.SYMBOLS_15M)
    return list(prepare.TRAIN_SYMBOLS)


def _resample_ohlcv(df: pd.DataFrame, target_hours: int) -> pd.DataFrame:
    frame = df.copy()
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("datetime")
    resampled = frame.resample(f"{int(target_hours)}h", origin="start_day").agg(
        {
            "timestamp": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "funding_rate": "mean",
            "asset_class": "last",
            "bar_interval_hours": "last",
            "has_funding": "last",
            "vix_norm": "last",
            "yield_10y_norm": "last",
            "fed_rate_norm": "last",
        }
    )
    resampled = resampled.dropna(subset=["timestamp", "open", "close"]).reset_index(drop=True)
    resampled["timestamp"] = pd.to_numeric(resampled["timestamp"], errors="coerce").round().astype("Int64")
    resampled = resampled.dropna(subset=["timestamp"]).copy()
    resampled["timestamp"] = resampled["timestamp"].astype("int64")
    resampled["bar_interval_hours"] = float(target_hours)
    return resampled


def _load_symbol_frame(symbol: str, timeframe: str, split: str) -> pd.DataFrame | None:
    effective_split = split_for_timeframe(_base_split(split), timeframe) if timeframe == "15m" else _base_split(split)
    use_15m = timeframe == "15m"
    suffix = "_15m.parquet" if use_15m else "_1h.parquet"
    filepath = os.path.join(prepare.DATA_DIR, f"{symbol}{suffix}")
    if not os.path.exists(filepath):
        return None

    start_str, end_str = _split_bounds(effective_split)
    if effective_split in ("train", "train_15m") and symbol in prepare.SYMBOL_TRAIN_START:
        start_str = max(start_str, prepare.SYMBOL_TRAIN_START[symbol])

    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)

    df = pd.read_parquet(filepath)
    mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
    frame = df[mask].reset_index(drop=True)
    if frame.empty:
        return None

    asset_cls = prepare.ASSET_CLASS.get(symbol, 0)
    frame["asset_class"] = asset_cls
    if asset_cls != 0:
        frame["funding_rate"] = 0.0
    elif "funding_rate" not in frame.columns:
        frame["funding_rate"] = 0.0
    frame["has_funding"] = 1.0 if asset_cls == 0 else 0.0

    if timeframe == "4h":
        frame = prepare._resample_to_4h(frame)
        frame["bar_interval_hours"] = 4.0
    elif timeframe == "1d":
        frame = _resample_ohlcv(frame, 24)
        frame["bar_interval_hours"] = 24.0
    else:
        if not use_15m and (asset_cls in prepare.DAILY_ASSET_CLASSES or symbol in prepare.DAILY_COMMODITIES):
            frame = prepare._resample_daily_to_1h(frame)
            frame["bar_interval_hours"] = 24.0
        else:
            frame["bar_interval_hours"] = 0.25 if use_15m else 1.0

    return frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def load_timeframe_data(timeframe: str, split: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    if timeframe not in {"15m", "1h", "4h", "1d"}:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    symbol_list = symbols or _default_symbol_universe(timeframe)
    result = {}
    for symbol in symbol_list:
        frame = _load_symbol_frame(symbol, timeframe, split)
        if frame is not None and not frame.empty:
            result[symbol] = frame
    return result


def load_bundle_data(
    bundle_name: str,
    split: str,
    symbols: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    bundle = BUNDLE_MANIFESTS[bundle_name]
    return {
        role: load_timeframe_data(timeframe, split, symbols=symbols)
        for role, timeframe in bundle.role_timeframes().items()
    }


def _role_feature_frames(
    bundle_name: str,
    split: str,
    feature_profile: str,
    symbols: list[str] | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, list[str]]]:
    bundle_data = load_bundle_data(bundle_name, split, symbols=symbols)
    role_frames: dict[str, dict[str, pd.DataFrame]] = {}
    role_columns: dict[str, list[str]] = {}

    for role, time_data in bundle_data.items():
        frames = {}
        for symbol, df in time_data.items():
            if len(df) < 120:
                continue
            feat = calculate_clock_features(
                df,
                timeframe=BUNDLE_MANIFESTS[bundle_name].role_timeframes()[role],
                symbol=symbol,
                feature_profile=feature_profile,
            )
            frames[symbol] = feat
        role_frames[role] = frames
        sample = next(iter(frames.values()), None)
        role_columns[role] = [col for col in sample.columns if col != "timestamp"] if sample is not None else []
    return role_frames, role_columns


def build_bundle_dataset(
    bundle_name: str,
    split: str,
    feature_profile: str = "price_context_plus",
    max_symbols: int | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    bundle = BUNDLE_MANIFESTS[bundle_name]
    selected_symbols = list(symbols) if symbols is not None else _default_symbol_universe(bundle.base_tf)
    if max_symbols is not None:
        selected_symbols = selected_symbols[:max_symbols]

    role_frames, role_columns = _role_feature_frames(
        bundle_name,
        split,
        feature_profile,
        symbols=selected_symbols,
    )
    base_frames = role_frames["base"]
    base_items = list(base_frames.items())

    regime_data = load_timeframe_data(bundle.base_tf, split, symbols=[symbol for symbol, _ in base_items])
    regime_frame = build_regime_frame(regime_data)
    regime_map = (
        dict(regime_frame[["timestamp", "regime_family"]].itertuples(index=False, name=None))
        if not regime_frame.empty
        else {}
    )

    rows = []
    for symbol, base_df in base_items:
        base_pref = base_df.rename(columns={col: f"base_{col}" for col in role_columns["base"]})
        base_pref["symbol"] = symbol
        base_pref["timestamp"] = base_df["timestamp"].values
        merged = base_pref.sort_values("timestamp")

        for role in ("fast", "slow"):
            if symbol in role_frames.get(role, {}):
                other = role_frames[role][symbol]
                prefixed = other.rename(columns={col: f"{role}_{col}" for col in role_columns[role]})
                prefixed["timestamp"] = other["timestamp"].values
                merged = pd.merge_asof(
                    merged.sort_values("timestamp"),
                    prefixed.sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
            else:
                for col in role_columns.get(role, []):
                    merged[f"{role}_{col}"] = 0.0

        merged["regime_family"] = merged["timestamp"].map(regime_map).fillna("unknown")
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    dataset = pd.concat(rows).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    rank_sources = {
        "base_clock_ret_24h": "base_ret_rank_24h",
        "base_clock_ret_72h": "base_ret_rank_72h",
        "fast_clock_ret_6h": "fast_ret_rank_6h",
        "base_clock_vol_ratio_24h": "base_vol_ratio_rank_24h",
        "base_clock_bb_width_24h": "base_bb_width_rank_24h",
    }
    for source, target in rank_sources.items():
        if source not in dataset.columns:
            dataset[target] = 0.5
            continue
        dataset[target] = dataset.groupby("timestamp")[source].rank(pct=True, method="average").fillna(0.5)

    return dataset.replace([np.inf, -np.inf], 0.0).fillna(0.0)
