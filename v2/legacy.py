from __future__ import annotations

import os

import numpy as np
import pandas as pd

from market_regime import build_regime_frame
import strategy as strategy_module
from strategy import Strategy
from .data import load_timeframe_data


SUPPORTED_DIRECTIONAL_BUNDLE = ("15m", "1h", "4h")
_LEGACY_BASE_CACHE: dict[tuple[str, str, int | None], pd.DataFrame] = {}
_LEGACY_FRAME_CACHE: dict[tuple[str, str, str, int | None], pd.DataFrame] = {}


def supports_legacy_foundation(bundle) -> bool:
    return (
        bundle.fast_tf,
        bundle.base_tf,
        bundle.slow_tf,
    ) == SUPPORTED_DIRECTIONAL_BUNDLE


def foundation_source_model_set() -> str:
    configured = os.environ.get("AUTOTRADER_V2_FOUNDATION_MODEL_SET", "root").strip()
    return configured or "root"


def _market_cluster(symbol: str) -> str:
    if symbol in {"BTC", "ETH"}:
        return "crypto_majors"
    if symbol in {"SOL", "BNB", "AVAX", "APT", "SUI", "NEAR"}:
        return "crypto_beta"
    if symbol in {"XRP", "ADA", "DOGE", "LINK", "DOT", "ATOM", "UNI"}:
        return "crypto_alt"
    if symbol in {"XAU", "SP500"}:
        return "macro_hedge"
    return "other"


def _series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _structure_score(frame: pd.DataFrame) -> pd.Series:
    liquidity_sweep = _series(frame, "liquidity_sweep")
    msb_status = _series(frame, "msb_status")
    fvg_detected = _series(frame, "fvg_detected")
    ema_200_dist = _series(frame, "ema_200_dist")
    dist_to_vwap = _series(frame, "dist_to_vwap")

    score = pd.Series(0, index=frame.index, dtype=float)
    score += np.where(liquidity_sweep > 0, 1, np.where(liquidity_sweep < 0, -1, 0))
    score += np.where(msb_status > 0, 1, np.where(msb_status < 0, -1, 0))
    score += np.where(fvg_detected > 0, 1, np.where(fvg_detected < 0, -1, 0))
    score += np.where(ema_200_dist > 0, 1, -1)
    score += np.where(dist_to_vwap <= 0, 1, -1)
    return score.astype(int)


def _annotate_trend_1h_directional(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    base = frame.copy()
    market_regime_family = base["regime_family"].fillna("unknown")
    m15 = _series(base, "meta_15m")
    m15_long = _series(base, "meta_long_15m") if "meta_long_15m" in base.columns else m15.copy()
    if "short_conf_15m" in base.columns:
        m15_short = _series(base, "short_conf_15m")
    elif "meta_short_15m" in base.columns:
        m15_short = _series(base, "meta_short_15m")
    else:
        m15_short = m15.copy()
    m1h = _series(base, "meta_1h")
    m4h = _series(base, "meta_4h")
    bull_15m = _series(base, "bull_15m")
    bear_15m = _series(base, "bear_15m")
    rsi_8 = _series(base, "rsi_8b", 50.0)
    market_ret = _series(base, "market_ret")
    regime_sum = _series(base, "market_ret_regime_sum")
    atr_val = _series(base, "atr_val")
    close = _series(base, "close")

    positive_meta = pd.concat([np.maximum(m15_long, m15_short), m1h, m4h], axis=1)
    meta_score = positive_meta.where(positive_meta > 0).mean(axis=1).fillna(0.0)
    supportive_regime = (m1h <= 0) | (m1h > 0.33)
    supportive_regime_4h = (m4h <= 0) | (m4h > 0.42)
    macro_bear = regime_sum < -0.015
    short_regime_ok = macro_bear | (regime_sum < 0.01) | (market_ret < 0.001)

    bull_signal = bull_15m > 0.32
    bear_signal = bear_15m > 0.36
    bull_fortress = bull_signal & (m15_long > 0.28) & (rsi_8 < 65)
    bull_soft = bull_signal & supportive_regime & supportive_regime_4h & (m15_long > 0.50)
    raw_bear_fortress = (bear_15m > 0.58) & (bear_15m > bull_15m + 0.08) & (rsi_8 > 46)
    bear_fortress = short_regime_ok & (((bear_signal) & (m15_short > 0.24) & (rsi_8 > 35)) | raw_bear_fortress)
    bear_soft = (
        short_regime_ok
        & bear_signal
        & supportive_regime
        & supportive_regime_4h
        & (m15_short > 0.42)
        & (bear_15m > bull_15m + 0.02)
    )

    bull_conviction = bull_15m + 0.70 * m15_long
    bear_conviction = bear_15m + 0.70 * m15_short + np.where(macro_bear, 0.05, 0.0)
    prefer_short = bear_conviction > bull_conviction + np.where(macro_bear, 0.02, 0.08)

    vol_pct = atr_val / close.replace(0, np.nan)
    not_falling_knife = market_ret > -0.02
    not_hyper_vol = vol_pct.fillna(0.0) < 0.06
    long_gate_ok = meta_score > 0.36
    macro_bull_ok = (~macro_bear) | (meta_score > 0.40)

    structure_score = _structure_score(base)
    sideways_scale = pd.Series(1.0, index=base.index, dtype=float)
    sideways_fortress = market_regime_family.eq("sideways") & bull_fortress
    sideways_scale = sideways_scale.where(~(sideways_fortress & structure_score.eq(1)), 1.10)
    sideways_scale = sideways_scale.where(
        ~(sideways_fortress & m15_long.between(0.29, 0.33, inclusive="both")),
        sideways_scale * 0.95,
    )

    entry_bull_fortress = (
        bull_fortress
        & ~prefer_short
        & not_falling_knife
        & not_hyper_vol
        & ~raw_bear_fortress
        & (~bear_fortress | (m15_long >= m15_short))
        & macro_bull_ok
        & long_gate_ok
    )
    entry_bull_soft = bull_soft & ~prefer_short & (~bear_soft | (m15_long >= m15_short)) & long_gate_ok & ~entry_bull_fortress

    confidence = (
        0.20
        + 0.40 * bull_15m
        + 0.40 * m15_long
        + 0.15 * np.clip(m1h, 0.0, 1.0)
        + 0.10 * np.clip(m4h, 0.0, 1.0)
        + np.where(entry_bull_soft, 0.03, 0.0)
    )
    confidence *= sideways_scale
    confidence -= np.where(macro_bear, 0.05, 0.0)
    confidence = np.clip(confidence, 0.0, 0.99)

    atr_mult = np.where(macro_bear, 4.0, 7.0)
    stop_distance = np.where(atr_val > 0, atr_mult * atr_val, np.maximum(close * 0.02, 1e-6))

    base["side"] = np.where(entry_bull_fortress | entry_bull_soft, 1, 0)
    base["legacy_signal_active"] = base["side"] != 0
    base["confidence"] = confidence
    base["expected_edge_bps"] = np.maximum(0.0, (confidence - 0.58) * 250.0) * sideways_scale
    base["holding_horizon_hours"] = 3.0
    base["stop_distance"] = stop_distance
    base["reason_tag"] = np.where(entry_bull_fortress, "entry_bull_fortress", np.where(entry_bull_soft, "entry_bull_soft", ""))
    base["structure_score"] = structure_score
    base["meta_score"] = meta_score
    base["supportive_regime"] = supportive_regime.astype(int)
    base["supportive_regime_4h"] = supportive_regime_4h.astype(int)
    base["macro_bear_proxy"] = macro_bear.astype(int)
    return base[base["legacy_signal_active"]].reset_index(drop=True)


def _annotate_bear_1h_calibrated(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    base = frame.copy()
    market_regime_family = base["regime_family"].fillna("unknown")
    m15 = _series(base, "meta_15m")
    m15_long = _series(base, "meta_long_15m") if "meta_long_15m" in base.columns else m15.copy()
    if "short_conf_15m" in base.columns:
        m15_short = _series(base, "short_conf_15m")
    elif "meta_short_15m" in base.columns:
        m15_short = _series(base, "meta_short_15m")
    else:
        m15_short = m15.copy()
    m1h = _series(base, "meta_1h")
    m4h = _series(base, "meta_4h")
    bull_15m = _series(base, "bull_15m")
    bear_15m = _series(base, "bear_15m")
    rsi_8 = _series(base, "rsi_8b", 50.0)
    market_ret = _series(base, "market_ret")
    regime_sum = _series(base, "market_ret_regime_sum")
    atr_val = _series(base, "atr_val")
    close = _series(base, "close")

    positive_meta = pd.concat([np.maximum(m15_long, m15_short), m1h, m4h], axis=1)
    meta_score = positive_meta.where(positive_meta > 0).mean(axis=1).fillna(0.0)
    supportive_regime = (m1h <= 0) | (m1h > 0.33)
    supportive_regime_4h = (m4h <= 0) | (m4h > 0.42)
    macro_bear = regime_sum < -0.015
    short_regime_ok = macro_bear | (regime_sum < 0.01) | (market_ret < 0.001)

    bull_signal = bull_15m > 0.32
    bear_signal = bear_15m > 0.36
    bull_fortress = bull_signal & (m15_long > 0.28) & (rsi_8 < 65)
    bull_soft = bull_signal & supportive_regime & supportive_regime_4h & (m15_long > 0.50)
    raw_bear_fortress = (bear_15m > 0.58) & (bear_15m > bull_15m + 0.08) & (rsi_8 > 46)
    bear_soft = (
        short_regime_ok
        & bear_signal
        & supportive_regime
        & supportive_regime_4h
        & (m15_short > 0.42)
        & (bear_15m > bull_15m + 0.02)
    )
    bull_conviction = bull_15m + 0.70 * m15_long
    bear_conviction = bear_15m + 0.70 * m15_short + np.where(macro_bear, 0.05, 0.0)
    prefer_short = bear_conviction > bull_conviction + np.where(macro_bear, 0.02, 0.08)
    short_gate_ok = meta_score > 0.36
    bear_calibrated = (
        market_regime_family.eq("bear")
        & bear_signal
        & (m15 > 0.24)
        & (rsi_8 > 35)
        & prefer_short
        & short_gate_ok
    )
    bear_calibrated &= ~(bull_fortress & ~prefer_short)
    bear_calibrated &= ~(bull_soft & ~prefer_short & ~bear_soft)
    bear_calibrated &= short_regime_ok | (m15_short > 0.55)
    bear_calibrated &= ~raw_bear_fortress

    confidence = (
        0.18
        + 0.35 * bear_15m
        + 0.45 * m15_short
        + 0.10 * np.clip(m1h, 0.0, 1.0)
        + 0.08 * np.clip(m4h, 0.0, 1.0)
        + np.where(macro_bear, 0.05, 0.0)
        + np.where(prefer_short, 0.04, 0.0)
    )
    confidence = np.clip(confidence, 0.0, 0.99)

    atr_mult = np.where(macro_bear, 4.0, 7.0)
    stop_distance = np.where(atr_val > 0, atr_mult * atr_val, np.maximum(close * 0.02, 1e-6))

    base["side"] = np.where(bear_calibrated, -1, 0)
    base["legacy_signal_active"] = base["side"] != 0
    base["confidence"] = confidence
    base["expected_edge_bps"] = np.maximum(0.0, (confidence - 0.56) * 220.0)
    base["holding_horizon_hours"] = 2.0
    base["stop_distance"] = stop_distance
    base["reason_tag"] = np.where(bear_calibrated, "entry_bear_calibrated", "")
    base["structure_score"] = _structure_score(base)
    base["meta_score"] = meta_score
    base["supportive_regime"] = supportive_regime.astype(int)
    base["supportive_regime_4h"] = supportive_regime_4h.astype(int)
    base["macro_bear_proxy"] = macro_bear.astype(int)
    return base[base["legacy_signal_active"]].reset_index(drop=True)


def _annotate_bear_1h_foundation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    base = frame.copy()
    market_regime_family = base["regime_family"].fillna("unknown")
    m15 = _series(base, "meta_15m")
    m15_long = _series(base, "meta_long_15m") if "meta_long_15m" in base.columns else m15.copy()
    if "short_conf_15m" in base.columns:
        m15_short = _series(base, "short_conf_15m")
    elif "meta_short_15m" in base.columns:
        m15_short = _series(base, "meta_short_15m")
    else:
        m15_short = m15.copy()
    m1h = _series(base, "meta_1h")
    m4h = _series(base, "meta_4h")
    bull_15m = _series(base, "bull_15m")
    bear_15m = _series(base, "bear_15m")
    rsi_8 = _series(base, "rsi_8b", 50.0)
    market_ret = _series(base, "market_ret")
    regime_sum = _series(base, "market_ret_regime_sum")
    atr_val = _series(base, "atr_val")
    close = _series(base, "close")

    positive_meta = pd.concat([np.maximum(m15_long, m15_short), m1h, m4h], axis=1)
    meta_score = positive_meta.where(positive_meta > 0).mean(axis=1).fillna(0.0)
    supportive_regime = (m1h <= 0) | (m1h > 0.33)
    supportive_regime_4h = (m4h <= 0) | (m4h > 0.42)
    macro_bear = regime_sum < -0.015
    short_regime_ok = macro_bear | (regime_sum < 0.01) | (market_ret < 0.001)

    bull_signal = bull_15m > 0.32
    bear_signal = bear_15m > 0.36
    bull_fortress = bull_signal & (m15_long > 0.28) & (rsi_8 < 65)
    raw_bear_fortress = (bear_15m > 0.58) & (bear_15m > bull_15m + 0.08) & (rsi_8 > 46)
    bull_soft = bull_signal & supportive_regime & supportive_regime_4h & (m15_long > 0.50)
    bear_soft = (
        short_regime_ok
        & bear_signal
        & supportive_regime
        & supportive_regime_4h
        & (m15_short > 0.42)
        & (bear_15m > bull_15m + 0.02)
    )

    bull_conviction = bull_15m + 0.70 * m15_long
    bear_conviction = bear_15m + 0.70 * m15_short + np.where(macro_bear, 0.05, 0.0)
    prefer_short = bear_conviction > bull_conviction + np.where(macro_bear, 0.02, 0.08)
    short_gate_ok = meta_score > 0.36

    bear_fortress = short_regime_ok & (((bear_signal) & (m15_short > 0.24) & (rsi_8 > 35)) | raw_bear_fortress)
    bear_calibrated = market_regime_family.eq("bear") & bear_signal & (m15 > 0.24) & (rsi_8 > 35) & prefer_short
    bear_calibrated &= ~(bull_fortress & ~prefer_short)
    bear_calibrated &= ~(bull_soft & ~prefer_short & ~bear_soft)
    bear_calibrated &= short_regime_ok | (m15_short > 0.55)
    bear_calibrated &= ~raw_bear_fortress

    entry_bear_fortress = bear_fortress & (prefer_short | ~bull_fortress) & short_gate_ok
    entry_bear_calibrated = bear_calibrated & short_gate_ok & ~entry_bear_fortress

    confidence = (
        0.18
        + 0.30 * bear_15m
        + 0.40 * m15_short
        + 0.10 * np.clip(m1h, 0.0, 1.0)
        + 0.08 * np.clip(m4h, 0.0, 1.0)
        + np.where(macro_bear, 0.05, 0.0)
        + np.where(prefer_short, 0.04, 0.0)
        + np.where(entry_bear_fortress, 0.08, 0.0)
        + np.where(entry_bear_calibrated, 0.04, 0.0)
    )
    confidence = np.clip(confidence, 0.0, 0.99)

    atr_mult = np.where(macro_bear, 4.0, 7.0)
    stop_distance = np.where(atr_val > 0, atr_mult * atr_val, np.maximum(close * 0.02, 1e-6))
    edge_scale = np.where(entry_bear_fortress, 1.10, 1.0)
    holding_horizon_hours = np.where(entry_bear_fortress, 3.0, 2.0)

    base["side"] = np.where(entry_bear_fortress | entry_bear_calibrated, -1, 0)
    base["legacy_signal_active"] = base["side"] != 0
    base["confidence"] = confidence
    base["expected_edge_bps"] = np.maximum(0.0, (confidence - 0.54) * 240.0) * edge_scale
    base["holding_horizon_hours"] = holding_horizon_hours
    base["stop_distance"] = stop_distance
    base["reason_tag"] = np.where(
        entry_bear_fortress,
        "entry_bear_fortress",
        np.where(entry_bear_calibrated, "entry_bear_calibrated", ""),
    )
    base["structure_score"] = _structure_score(base)
    base["meta_score"] = meta_score
    base["supportive_regime"] = supportive_regime.astype(int)
    base["supportive_regime_4h"] = supportive_regime_4h.astype(int)
    base["macro_bear_proxy"] = macro_bear.astype(int)
    return base[base["legacy_signal_active"]].reset_index(drop=True)

def _build_legacy_base_frame(
    bundle_name: str,
    split: str,
    max_symbols: int | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    symbol_key = tuple(symbols or [])
    key = (bundle_name, split, max_symbols, symbol_key)
    cached = _LEGACY_BASE_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    from v2.manifests import BUNDLE_MANIFESTS

    bundle = BUNDLE_MANIFESTS[bundle_name]
    if not supports_legacy_foundation(bundle):
        return pd.DataFrame()

    base_split = split[:-4] if split.endswith("_15m") else split
    selected_data = load_timeframe_data("1h", base_split, symbols=symbols)
    if not selected_data:
        return pd.DataFrame()
    selected_symbols = list(selected_data)[:max_symbols] if max_symbols is not None else list(selected_data)
    selected_data = {symbol: selected_data[symbol] for symbol in selected_symbols}

    source_model_set = foundation_source_model_set()
    previous_model_set = strategy_module.DEFAULT_MODEL_SET
    strategy_module.DEFAULT_MODEL_SET = "" if source_model_set == "root" else source_model_set
    try:
        legacy = Strategy(timeframe="1h")
        legacy._load_models()
    finally:
        strategy_module.DEFAULT_MODEL_SET = previous_model_set

    regime_frame = build_regime_frame(selected_data)
    regime_map = (
        dict(regime_frame[["timestamp", "regime_family"]].itertuples(index=False, name=None))
        if not regime_frame.empty
        else {}
    )
    main_tables = legacy._build_prediction_tables(selected_data, "1h", include_state=True)
    split_15m = {
        "train": "train_15m",
        "val": "val_15m",
        "oos": "oos_15m",
        "2026q1": "2026q1_15m",
        "newasset_oos2y": "newasset_oos2y_15m",
        "holdout": "holdout_15m",
    }.get(base_split)
    aux_15m_tables = {}
    if split_15m:
        aux_15m_data = load_timeframe_data("15m", split_15m, symbols=selected_symbols)
        aux_15m_tables = legacy._build_prediction_tables(aux_15m_data, "15m")
    aux_4h_data = load_timeframe_data("4h", base_split, symbols=selected_symbols)
    aux_4h_tables = legacy._build_prediction_tables(aux_4h_data, "4h")

    rows = []
    for symbol, df in selected_data.items():
        main_df = main_tables.get(symbol)
        if main_df is None or main_df.empty:
            continue
        base = main_df[
            [
                "timestamp",
                "atr_val",
                "market_ret",
                "rsi_8b",
                "rsi_24b",
                "atr_pct",
                "liquidity_sweep",
                "msb_status",
                "fvg_detected",
                "ob_dist",
                "ema_200_dist",
                "dist_to_vwap",
                "has_funding",
                "bar_interval_hours",
                "macro_event_flag",
                "context_sentiment",
                "major_market_event_flag",
            ]
        ].copy()
        base["bull_15m"] = main_df["bull_1h"].fillna(0.0).values if "bull_1h" in main_df else 0.0
        base["bear_15m"] = main_df["bear_1h"].fillna(0.0).values if "bear_1h" in main_df else 0.0
        base["meta_15m"] = 0.0
        base["meta_long_15m"] = 0.0
        base["meta_short_15m"] = 0.0
        base["short_conf_15m"] = 0.0
        base["short_conf_raw_15m"] = 0.0
        base["short_conf_bear_15m"] = 0.0
        base["meta_1h"] = main_df["meta_1h"].fillna(0.0).values if "meta_1h" in main_df else 0.0
        base["bull_1h"] = main_df["bull_1h"].fillna(0.0).values if "bull_1h" in main_df else 0.0
        base["bear_1h"] = main_df["bear_1h"].fillna(0.0).values if "bear_1h" in main_df else 0.0
        base["meta_4h"] = 0.0

        if symbol in aux_15m_tables:
            merged_15m = pd.merge_asof(
                base[["timestamp"]].sort_values("timestamp"),
                aux_15m_tables[symbol].sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            for col in (
                "bull_15m",
                "bear_15m",
                "meta_15m",
                "meta_long_15m",
                "meta_short_15m",
                "short_conf_15m",
                "short_conf_raw_15m",
                "short_conf_bear_15m",
            ):
                if col in merged_15m.columns:
                    base[col] = merged_15m[col].fillna(base[col])

        if symbol in aux_4h_tables:
            merged_4h = pd.merge_asof(
                base[["timestamp"]].sort_values("timestamp"),
                aux_4h_tables[symbol][["timestamp", "meta_4h"]].sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            base["meta_4h"] = merged_4h["meta_4h"].fillna(base["meta_4h"])

        price_frame = df[["timestamp", "close", "high", "low", "volume", "funding_rate"]].copy()
        merged = base.merge(price_frame, on="timestamp", how="left")
        merged["symbol"] = symbol
        merged["market_cluster"] = _market_cluster(symbol)
        merged["regime_family"] = merged["timestamp"].map(regime_map).fillna("unknown")
        merged["base_close"] = pd.to_numeric(merged["close"], errors="coerce").fillna(0.0)
        merged["base_volume"] = pd.to_numeric(merged["volume"], errors="coerce").fillna(0.0)
        merged["base_bar_interval_hours"] = pd.to_numeric(merged["bar_interval_hours"], errors="coerce").fillna(1.0)
        merged["market_ret_regime_sum"] = _series(merged, "market_ret").rolling(legacy._regime_window, min_periods=1).sum()

        expanded_daily = pd.to_numeric(merged["bar_interval_hours"], errors="coerce").fillna(1.0) >= 24.0
        if expanded_daily.any():
            hour = (pd.to_numeric(merged["timestamp"], errors="coerce").fillna(0).astype("int64") // 3_600_000) % 24
            merged = merged.loc[(~expanded_daily) | hour.eq(0)].copy()
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    frame = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    frame = frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    _LEGACY_BASE_CACHE[key] = frame.copy()
    return frame


def build_legacy_foundation_frame(
    sleeve_name: str,
    bundle_name: str,
    split: str,
    max_symbols: int | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    symbol_key = tuple(symbols or [])
    key = (sleeve_name, bundle_name, split, max_symbols, symbol_key)
    cached = _LEGACY_FRAME_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    base = _build_legacy_base_frame(bundle_name, split, max_symbols=max_symbols, symbols=symbols)
    if base.empty:
        return pd.DataFrame()

    if sleeve_name == "trend_1h_directional":
        frame = _annotate_trend_1h_directional(base)
    elif sleeve_name == "bear_1h_calibrated":
        frame = _annotate_bear_1h_calibrated(base)
    elif sleeve_name == "bear_1h_foundation":
        frame = _annotate_bear_1h_foundation(base)
    else:
        raise ValueError(f"Unsupported legacy foundation sleeve: {sleeve_name}")

    _LEGACY_FRAME_CACHE[key] = frame.copy()
    return frame


def build_trend_1h_directional_frame(
    bundle_name: str,
    split: str,
    max_symbols: int | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    return build_legacy_foundation_frame(
        "trend_1h_directional",
        bundle_name,
        split,
        max_symbols=max_symbols,
        symbols=symbols,
    )
