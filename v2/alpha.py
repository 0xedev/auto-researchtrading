from __future__ import annotations

import numpy as np
import pandas as pd

from .manifests import SLEEVE_MANIFESTS
from .types import SleeveManifest, TimeframeBundle
from .features import bars_for_hours


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def infer_market_cluster(symbol: str) -> str:
    if symbol in {"BTC", "ETH"}:
        return "crypto_majors"
    if symbol in {"SOL", "BNB", "AVAX", "APT", "SUI", "NEAR"}:
        return "crypto_beta"
    if symbol in {"XRP", "ADA", "DOGE", "LINK", "DOT", "ATOM", "UNI"}:
        return "crypto_alt"
    if symbol in {"XAU", "SP500"}:
        return "macro_hedge"
    return "other"


def compute_side_series(frame: pd.DataFrame, manifest: SleeveManifest) -> pd.Series:
    regime = frame.get("regime_family", pd.Series("unknown", index=frame.index)).fillna("unknown")
    base_trend = _col(frame, "base_clock_ema_200h_dist")
    slow_trend = _col(frame, "slow_clock_ema_200h_dist")
    base_rsi = _col(frame, "base_clock_rsi_6h", 50.0)
    base_rsi_slow = _col(frame, "base_clock_rsi_24h", 50.0)
    fast_ret = _col(frame, "fast_clock_ret_6h")
    base_ret = _col(frame, "base_clock_ret_24h")
    base_ret_long = _col(frame, "base_clock_ret_72h")
    fast_liq = _col(frame, "fast_liquidity_sweep")
    fast_msb = _col(frame, "fast_msb_status")
    fast_fvg = _col(frame, "fast_fvg_detected")
    base_vol_ratio = _col(frame, "base_clock_vol_ratio_24h", 1.0)
    base_bb_width = _col(frame, "base_clock_bb_width_24h")
    base_range = _col(frame, "base_clock_range_position_24h", 0.5)
    base_vwap_dist = _col(frame, "base_clock_vwap_dist_24h")
    base_funding = _col(frame, "base_funding_rate")
    base_funding_roc = _col(frame, "base_funding_roc_4b")
    base_ret_rank = _col(frame, "base_ret_rank_24h", 0.5)
    base_ret_rank_long = _col(frame, "base_ret_rank_72h", 0.5)
    fast_ret_rank = _col(frame, "fast_ret_rank_6h", 0.5)
    base_vol_rank = _col(frame, "base_vol_ratio_rank_24h", 0.5)
    base_bb_rank = _col(frame, "base_bb_width_rank_24h", 0.5)
    stress = _col(frame, "base_cross_asset_stress")
    sentiment = _col(frame, "base_context_sentiment")
    sentiment_shock = _col(frame, "base_context_sentiment_shock")
    macro_event = _col(frame, "base_macro_event_flag")
    major_event = _col(frame, "base_major_market_event_flag")

    up_trend = (base_trend > 0) & (slow_trend > 0)
    down_trend = (base_trend < 0) & (slow_trend < 0)
    sideways = regime.eq("sideways")
    bear = regime.eq("bear")

    name = manifest.name
    if name == "trend_breakout":
        return ((up_trend) & (fast_msb > 0) & (base_range > 0.65) & (base_vol_ratio > 1.05)).astype(int)
    if name == "trend_pullback":
        return ((up_trend) & (fast_liq > 0) & base_rsi.between(35, 60) & (_col(frame, "base_clock_dist_to_low_24h") < 0.03)).astype(int)
    if name == "trend_continuation":
        return ((up_trend) & (fast_ret > 0) & (base_ret > 0) & (fast_fvg > 0)).astype(int)
    if name == "sideways_mean_reversion":
        return (sideways & (base_rsi < 35) & (base_range < 0.25)).astype(int)
    if name == "post_extension_snapback":
        return pd.Series(
            np.where(
                sideways & (base_rsi > 72) & (base_vwap_dist > 0.01),
                -1,
                np.where(sideways & (base_rsi < 28) & (base_vwap_dist < -0.01), 1, 0),
            ),
            index=frame.index,
        )
    if name == "volatility_reversion":
        return pd.Series(
            np.where(
                sideways & (base_bb_width > 0.08) & (base_ret > 0.01),
                -1,
                np.where(sideways & (base_bb_width > 0.08) & (base_ret < -0.01), 1, 0),
            ),
            index=frame.index,
        )
    if name == "funding_carry":
        return pd.Series(
            np.where(base_funding > 0.00015, -1, np.where(base_funding < -0.00015, 1, 0)),
            index=frame.index,
        )
    if name == "basis_dislocation":
        return pd.Series(
            np.where(base_funding_roc > 0.00005, -1, np.where(base_funding_roc < -0.00005, 1, 0)),
            index=frame.index,
        )
    if name == "cross_asset_relative_strength":
        return ((up_trend) & (base_ret_rank > 0.75) & (base_ret_long > 0)).astype(int)
    if name == "leader_laggard_rotation":
        return ((fast_ret_rank > 0.75) & (base_ret_rank_long < 0.60) & (base_vol_ratio > 1.0) & (slow_trend >= 0)).astype(int)
    if name == "macro_beta_dispersion":
        return pd.Series(
            np.where(
                (stress > 0.5) & (base_ret_rank > 0.8) & (slow_trend > 0),
                1,
                np.where((stress > 0.5) & (base_ret_rank < 0.2) & (slow_trend < 0), -1, 0),
            ),
            index=frame.index,
        )
    if name == "macro_event_drift":
        event_side = np.sign(sentiment_shock.where(sentiment_shock != 0, sentiment))
        return pd.Series(
            np.where(
                ((macro_event != 0) | (major_event != 0)) & (event_side > 0) & (base_trend >= 0),
                1,
                np.where(((macro_event != 0) | (major_event != 0)) & (event_side < 0) & (base_trend <= 0), -1, 0),
            ),
            index=frame.index,
        )
    if name == "post_event_mean_reversion":
        return pd.Series(
            np.where(
                ((macro_event != 0) | (major_event != 0)) & (sentiment_shock > 0.15) & (base_vwap_dist > 0.01),
                -1,
                np.where(((macro_event != 0) | (major_event != 0)) & (sentiment_shock < -0.15) & (base_vwap_dist < -0.01), 1, 0),
            ),
            index=frame.index,
        )
    if name == "bear_stress_short":
        return (-1 * ((bear | down_trend) & (stress > 0.5) & (fast_msb < 0) & (base_ret < 0)).astype(int))
    if name == "squeeze_failure_short":
        return (-1 * ((base_rsi_slow > 68) & (fast_liq < 0) & (fast_fvg < 0) & (base_ret_rank > 0.7)).astype(int))
    return pd.Series(0, index=frame.index, dtype=int)


def build_training_frame(frame: pd.DataFrame, bundle: TimeframeBundle, manifest: SleeveManifest) -> pd.DataFrame:
    base_bar_hours = float(_col(frame, "base_bar_interval_hours", 1.0).iloc[0] or 1.0)
    horizon_bars = bars_for_hours(manifest.label_horizon_hours or bundle.label_horizon_hours, base_bar_hours)
    base_close = _col(frame, "base_close").replace(0, np.nan)
    symbols = frame["symbol"] if "symbol" in frame.columns else pd.Series("unknown", index=frame.index)
    forward_close = base_close.groupby(symbols, sort=False).shift(-horizon_bars)
    forward_ret = (forward_close / base_close) - 1.0
    vol_window = bars_for_hours(24.0, base_bar_hours)
    returns = base_close.groupby(symbols, sort=False).pct_change(fill_method=None)
    vol = (
        returns.groupby(symbols, sort=False)
        .rolling(vol_window)
        .std()
        .reset_index(level=0, drop=True)
        .replace(0, np.nan)
    )
    scaled_ret = (forward_ret / (vol * np.sqrt(max(horizon_bars, 1)))).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    side = compute_side_series(frame, manifest)
    signed_edge = side * scaled_ret
    target = ((side != 0) & (signed_edge > manifest.target_sigma)).astype(int)
    enriched = frame.copy()
    enriched["side"] = side
    enriched["forward_ret"] = forward_ret.fillna(0.0)
    enriched["scaled_forward_edge"] = signed_edge.fillna(0.0)
    enriched["target"] = target
    enriched["market_cluster"] = enriched["symbol"].map(infer_market_cluster)
    return enriched[enriched["side"] != 0].reset_index(drop=True)


def selected_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "timestamp",
        "symbol",
        "regime_family",
        "side",
        "target",
        "forward_ret",
        "scaled_forward_edge",
        "market_cluster",
    }
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def manifest_by_name(name: str) -> SleeveManifest:
    return SLEEVE_MANIFESTS[name]
