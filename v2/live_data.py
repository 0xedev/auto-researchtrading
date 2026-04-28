"""
Live data utilities for V2 traders.

Handles:
- Parquet cache refresh from exchange bars
- Live signal extraction (latest bar only)
- Position state update after exchange execution
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

import prepare
from execution.v2_paper import V2ShadowState, load_v2_shadow_state, save_v2_shadow_state
from v2.allocator import PortfolioAllocator
from v2.runtime import V2SignalEngine
from v2.types import PortfolioConfig, SleeveSignal

# V2 internal symbol → Binance USD-M Futures symbol
BINANCE_SYMBOL_MAP: dict[str, str] = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "LTC":  "LTCUSDT",
    "BCH":  "BCHUSDT",
    "ETC":  "ETCUSDT",
    "TRX":  "TRXUSDT",
    "AAVE": "AAVEUSDT",
    "FIL":  "FILUSDT",
    "OP":   "OPUSDT",
}

# Deriv API symbol → V2 cache symbol (parquet key)
# Full 11-instrument universe matching the deriv_oss4y OOS evaluation.
DERIV_SYMBOL_MAP: dict[str, str] = {
    "R_10":   "DERIV_V10",
    "R_25":   "DERIV_V25",
    "R_50":   "DERIV_V50",
    "R_75":   "DERIV_V75",
    "R_100":  "DERIV_V100",
    "JD25":   "DERIV_JUMP25",
    "JD100":  "DERIV_JUMP100",
    "stpRNG": "DERIV_STEP",
    "RB100":  "DERIV_RB100",
}


def upsert_cache(symbol: str, df: pd.DataFrame) -> None:
    """Merge fresh bars into the 1h parquet cache. Deduplicates on timestamp."""
    filepath = os.path.join(prepare.DATA_DIR, f"{symbol}_1h.parquet")
    os.makedirs(prepare.DATA_DIR, exist_ok=True)

    incoming = df.copy()
    incoming["timestamp"] = incoming["timestamp"].astype("int64")
    required = ["timestamp", "open", "high", "low", "close", "volume", "funding_rate"]
    for col in required:
        if col not in incoming.columns:
            incoming[col] = 0.0
    incoming = incoming[required]

    if os.path.exists(filepath):
        existing = pd.read_parquet(filepath)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    combined = (
        combined
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    combined.to_parquet(filepath, index=False)


def compute_live_actions(
    bundle_name: str,
    model_set: str,
    portfolio_config: PortfolioConfig,
    active_sleeves: list[str],
    symbols: list[str],
    state_path: str,
    equity_override: float | None = None,
    halt_new_orders: bool = False,
) -> dict[str, Any]:
    """
    Load state, build signal engine on the "live" split, allocate for the
    latest bar, and return a structured result with opens, exits, and state.

    Returns dict with keys:
        latest_ts   — ms timestamp of the bar that was evaluated
        opens       — list of {symbol, side, target_notional_usd, meta}
        exits       — list of {symbol, reason, meta}
        close_prices — {symbol: float} for sizing
        equity      — current equity from state
    """
    state = load_v2_shadow_state(state_path)
    equity = float(equity_override) if equity_override is not None else (float(state.equity) if state.equity else 100_000.0)

    engine = V2SignalEngine(
        bundle_name=bundle_name,
        model_set=model_set,
        active_sleeves=active_sleeves,
        symbols=symbols,
        confidence_overrides=portfolio_config.sleeve_min_confidence_overrides,
    )
    engine.prepare("live")

    if not engine.timestamps:
        return {"latest_ts": None, "opens": [], "exits": [], "close_prices": {}, "equity": equity}

    latest_ts = int(max(engine.timestamps))
    close_prices = engine.close_by_symbol_at_timestamp(latest_ts)

    # ── Exit checks on existing positions ─────────────────────────────────���──
    exits: list[dict] = []
    position_meta = dict(state.position_meta)
    for symbol, notional in list(state.positions.items()):
        if abs(notional) < 1.0:
            continue
        meta = position_meta.get(symbol, {})
        entry_price = float(state.entry_prices.get(symbol, 0.0))
        current_price = float(close_prices.get(symbol, entry_price))
        stop_distance = float(meta.get("stop_distance", 0.0) or 0.0)
        take_profit_r = float(portfolio_config.take_profit_r_multiple or 0.0)

        if portfolio_config.enable_stop_loss and entry_price > 0 and stop_distance > 0 and current_price > 0:
            side = 1.0 if notional > 0 else -1.0
            adverse_move = side * (current_price - entry_price)
            favorable_move = side * (current_price - entry_price)
            if adverse_move <= -stop_distance:
                exits.append({"symbol": symbol, "reason": "stop_loss", "meta": meta})
                continue
            if take_profit_r > 0 and favorable_move >= stop_distance * take_profit_r:
                exits.append({"symbol": symbol, "reason": "take_profit", "meta": meta})
                continue

        opened_ts = int(meta.get("opened_ts", latest_ts))
        max_hold_hours = float(meta.get("max_hold_hours", 0.0) or 0.0)
        age_hours = max(0.0, (latest_ts - opened_ts) / 3_600_000.0)
        if max_hold_hours > 0 and age_hours >= max_hold_hours:
            exits.append({"symbol": symbol, "reason": "max_hold", "meta": meta})

    # ── New open signals ──────────────────────────────────────────────────────
    signals = [] if halt_new_orders else engine.signals_at_timestamp(latest_ts)

    # Remove signals for symbols we're exiting this bar
    exiting = {e["symbol"] for e in exits}
    signals = [s for s in signals if s.symbol not in exiting]

    current_positions = {
        sym: n for sym, n in state.positions.items()
        if sym not in exiting and abs(n) >= 1.0
    }
    allocator = PortfolioAllocator(portfolio_config)
    allocated, _ = allocator.allocate(signals, current_positions, equity)

    opens = []
    for sig in allocated:
        target_notional = equity * (
            portfolio_config.tier_a_notional_pct
            if sig.confidence >= portfolio_config.tier_a_confidence
            else portfolio_config.tier_b_notional_pct
        ) * _effective_weight(sig, portfolio_config) * sig.side
        opens.append({
            "symbol": sig.symbol,
            "side": sig.side,
            "target_notional_usd": float(target_notional),
            "sleeve": sig.sleeve,
            "confidence": float(sig.confidence),
            "stop_distance": float(sig.stop_distance),
            "max_hold_hours": float(sig.holding_horizon_hours),
            "meta": {
                "sleeve": sig.sleeve,
                "bundle": sig.bundle,
                "confidence": float(sig.confidence),
                "stop_distance": float(sig.stop_distance),
                "max_hold_hours": float(sig.holding_horizon_hours),
                "opened_ts": latest_ts,
            },
        })

    return {
        "latest_ts": latest_ts,
        "opens": opens,
        "exits": exits,
        "close_prices": close_prices,
        "equity": equity,
        "halt_new_orders": halt_new_orders,
    }


def _effective_weight(sig: SleeveSignal, config: PortfolioConfig) -> float:
    w = config.sleeve_weight_overrides.get(sig.sleeve, 1.0)
    cap = config.sleeve_cap_overrides.get(sig.sleeve, config.max_per_sleeve_pct)
    return min(w, cap / max(config.tier_a_notional_pct, 1e-6))


def update_state_after_execution(
    state_path: str,
    timestamp: int,
    equity: float,
    positions: dict[str, float],
    entry_prices: dict[str, float],
    position_meta: dict[str, dict],
) -> None:
    """Persist updated positions and equity to state file after real exchange fills."""
    state = load_v2_shadow_state(state_path)
    state.last_timestamp = timestamp
    state.equity = equity
    state.cash = equity - sum(abs(v) for v in positions.values())
    state.positions = dict(positions)
    state.entry_prices = dict(entry_prices)
    state.position_meta = dict(position_meta)
    save_v2_shadow_state(state_path, state)
