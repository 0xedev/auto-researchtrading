from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import prepare
from prepare import PortfolioState

from v2.allocator import PortfolioAllocator
from v2.runtime import V2SignalEngine
from v2.types import PortfolioConfig


@dataclass
class V2ShadowState:
    cash: float = prepare.INITIAL_CAPITAL
    positions: dict = field(default_factory=dict)
    entry_prices: dict = field(default_factory=dict)
    position_meta: dict = field(default_factory=dict)
    recent_entries: dict = field(default_factory=dict)
    equity: float = prepare.INITIAL_CAPITAL
    last_timestamp: int = 0
    total_volume: float = 0.0
    runtime_config: dict = field(default_factory=dict)
    last_kill_switch: dict = field(default_factory=dict)


def load_v2_shadow_state(path: str | Path) -> V2ShadowState:
    target = Path(path)
    if not target.exists():
        return V2ShadowState()
    payload = json.loads(target.read_text())
    return V2ShadowState(
        cash=float(payload.get("cash", prepare.INITIAL_CAPITAL)),
        positions={k: float(v) for k, v in payload.get("positions", {}).items()},
        entry_prices={k: float(v) for k, v in payload.get("entry_prices", {}).items()},
        position_meta=payload.get("position_meta", {}) or {},
        recent_entries={k: int(v) for k, v in (payload.get("recent_entries", {}) or {}).items()},
        equity=float(payload.get("equity", prepare.INITIAL_CAPITAL)),
        last_timestamp=int(payload.get("last_timestamp", 0)),
        total_volume=float(payload.get("total_volume", 0.0)),
        runtime_config=payload.get("runtime_config", {}) or {},
        last_kill_switch=payload.get("last_kill_switch", {}) or {},
    )


def save_v2_shadow_state(path: str | Path, state: V2ShadowState) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))


def _load_kill_switch(path: str | None) -> dict:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text())
    except Exception:
        return {}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _append_log(path: str | Path, record: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _mark_to_market(portfolio: PortfolioState, close_by_symbol: dict[str, float]) -> float:
    unrealized = 0.0
    for symbol, pos in portfolio.positions.items():
        if symbol not in close_by_symbol:
            continue
        entry = portfolio.entry_prices.get(symbol, close_by_symbol[symbol])
        price = close_by_symbol[symbol]
        if entry > 0:
            unrealized += pos * (price - entry) / entry
    portfolio.equity = portfolio.cash + sum(abs(v) for v in portfolio.positions.values()) + unrealized
    return portfolio.equity


def _cooldown_key(signal) -> str:
    scope = str(signal.metadata.get("reentry_cooldown_scope", "symbol") or "symbol")
    if scope == "sleeve":
        return str(signal.sleeve)
    return f"{signal.sleeve}:{signal.symbol}"


def _apply_reentry_cooldown(signals: list, recent_entries: dict[str, int], timestamp: int) -> tuple[list, list[dict]]:
    kept = []
    rejections = []
    unit_scale = 1000 if abs(int(timestamp)) > 1e11 else 1
    for signal in signals:
        cooldown_hours = float(signal.metadata.get("reentry_cooldown_hours", 0.0) or 0.0)
        if cooldown_hours <= 0:
            kept.append(signal)
            continue
        key = _cooldown_key(signal)
        last_entry = int(recent_entries.get(key, 0) or 0)
        if last_entry and (timestamp - last_entry) < (cooldown_hours * 3600 * unit_scale):
            rejections.append(
                {
                    "symbol": signal.symbol,
                    "sleeve": signal.sleeve,
                    "reason": "sleeve_reentry_cooldown",
                }
            )
            continue
        kept.append(signal)
    return kept, rejections


def run_v2_shadow_session(
    bundle_name: str,
    model_set: str,
    split: str,
    portfolio_config: PortfolioConfig,
    state_path: str,
    log_path: str,
    kill_switch_path: str | None = None,
    max_days: int | None = None,
    active_sleeves: list[str] | None = None,
    max_symbols: int | None = None,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
    engine: V2SignalEngine | None = None,
    collect_bar_history: bool = False,
) -> dict:
    if engine is None:
        engine = V2SignalEngine(
            bundle_name=bundle_name,
            model_set=model_set,
            active_sleeves=active_sleeves,
            max_symbols=max_symbols,
        )
        engine.prepare(split)

    state = load_v2_shadow_state(state_path)
    state.runtime_config = {
        "bundle": bundle_name,
        "model_set": model_set,
        "split": split,
        "max_days": max_days,
        "portfolio": asdict(portfolio_config),
        "active_sleeves": active_sleeves or list(engine.sleeve_tables.keys()),
        "kill_switch_path": kill_switch_path,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
    }
    allocator = PortfolioAllocator(portfolio_config)
    portfolio = PortfolioState(
        cash=state.cash,
        positions=dict(state.positions),
        entry_prices=dict(state.entry_prices),
        equity=state.equity,
        timestamp=state.last_timestamp,
    )
    position_meta = dict(state.position_meta)
    recent_entries = dict(state.recent_entries)

    timestamps = [ts for ts in engine.timestamps if ts > state.last_timestamp]
    if start_timestamp is not None:
        timestamps = [ts for ts in timestamps if ts >= int(start_timestamp)]
    if end_timestamp is not None:
        timestamps = [ts for ts in timestamps if ts <= int(end_timestamp)]
    total_bars_available = len(timestamps)
    if max_days is not None and timestamps:
        unit_scale = 1000 if abs(int(timestamps[0])) > 1e11 else 1
        max_ts = timestamps[0] + max_days * 24 * 3600 * unit_scale
        timestamps = [ts for ts in timestamps if ts <= max_ts]

    processed = 0
    first_processed = None
    equity_curve = [float(portfolio.equity)] if collect_bar_history else None
    equity_timestamps = [int(state.last_timestamp or timestamps[0])] if collect_bar_history and timestamps else [int(state.last_timestamp)] if collect_bar_history and state.last_timestamp else []

    for timestamp in timestamps:
        kill_switch = _load_kill_switch(kill_switch_path)
        state.last_kill_switch = kill_switch
        if first_processed is None:
            first_processed = timestamp

        active_candidates = engine.signals_at_timestamp(timestamp)
        active_candidates, cooldown_rejections = _apply_reentry_cooldown(active_candidates, recent_entries, int(timestamp))
        bundle_rows = engine.bundle_frame[engine.bundle_frame["timestamp"] == timestamp]
        close_by_symbol = {
            str(row.symbol): float(row.base_close)
            for row in bundle_rows.itertuples(index=False)
            if hasattr(row, "base_close")
        }

        # Portfolio exits by max-hold.
        close_requests = []
        for symbol, notional in list(portfolio.positions.items()):
            meta = position_meta.get(symbol, {})
            opened_ts = int(meta.get("opened_ts", timestamp))
            max_hold_hours = float(meta.get("max_hold_hours", 0.0) or 0.0)
            age_hours = max(0.0, (timestamp - opened_ts) / 3600.0)
            if max_hold_hours and age_hours >= max_hold_hours:
                close_requests.append(
                    {
                        "symbol": symbol,
                        "target": 0.0,
                        "reason": "max_hold",
                        "meta": meta,
                    }
                )

        _mark_to_market(portfolio, close_by_symbol)
        allocated, rejected = allocator.allocate(
            active_candidates,
            portfolio.positions,
            portfolio.equity,
            current_position_meta=position_meta,
        )
        rejected = list(cooldown_rejections) + list(rejected)

        for reject in rejected:
            _append_log(
                log_path,
                {
                    "timestamp": timestamp,
                    "action": "rejected",
                    "symbol": reject.get("symbol", ""),
                    "sleeve": reject.get("sleeve", ""),
                    "reject_reason": reject.get("reason", ""),
                    "portfolio_equity": portfolio.equity,
                },
            )

        for close_request in close_requests:
            symbol = close_request["symbol"]
            current = portfolio.positions.get(symbol, 0.0)
            if current == 0:
                continue
            price = close_by_symbol.get(symbol, portfolio.entry_prices.get(symbol, 0.0))
            entry = portfolio.entry_prices.get(symbol, price)
            realized = current * (price - entry) / entry if entry > 0 else 0.0
            portfolio.cash += abs(current) + realized
            portfolio.total_volume = getattr(portfolio, "total_volume", 0.0) + abs(current)
            portfolio.positions.pop(symbol, None)
            portfolio.entry_prices.pop(symbol, None)
            meta = position_meta.pop(symbol, {})
            _append_log(
                log_path,
                {
                    "timestamp": timestamp,
                    "action": "close",
                    "symbol": symbol,
                    "signal_tag": meta.get("reason_tag", ""),
                    "sleeve": meta.get("sleeve", ""),
                    "bundle": meta.get("bundle", bundle_name),
                    "signal_regime_family": meta.get("regime_context", "unknown"),
                    "cluster": meta.get("cluster", "other"),
                    "decision_reason": "max_hold",
                    "delta_notional": abs(current),
                    "realized_pnl": realized,
                    "portfolio_equity": portfolio.equity,
                },
            )

        if not kill_switch.get("halt_new_orders"):
            for signal in allocated:
                current = portfolio.positions.get(signal.symbol, 0.0)
                target = signal.target_notional
                delta = target - current
                if abs(delta) < 1.0:
                    continue
                base_close = float(signal.metadata.get("base_close", 0.0))
                base_volume = float(signal.metadata.get("base_volume", 0.0))
                participation = abs(target) / max(base_close * max(base_volume, 1.0), 1.0)
                if participation > portfolio_config.max_participation_rate:
                    _append_log(
                        log_path,
                        {
                            "timestamp": timestamp,
                            "action": "rejected",
                            "symbol": signal.symbol,
                            "sleeve": signal.sleeve,
                            "bundle": signal.bundle,
                            "cluster": signal.cluster,
                            "reject_reason": "participation_cap",
                            "participation": participation,
                            "portfolio_equity": portfolio.equity,
                        },
                    )
                    continue

                slippage = base_close * portfolio_config.slippage_bps / 10000.0
                exec_price = base_close + slippage if delta > 0 else base_close - slippage
                fee = abs(delta) * prepare.TAKER_FEE
                portfolio.total_volume = getattr(portfolio, "total_volume", 0.0) + abs(delta)
                portfolio.cash -= fee
                event = "modify"
                realized = 0.0

                if current == 0:
                    event = "open"
                    portfolio.cash -= abs(target)
                    portfolio.entry_prices[signal.symbol] = exec_price
                    recent_entries[_cooldown_key(signal)] = int(timestamp)
                elif target == 0:
                    event = "close"
                    entry = portfolio.entry_prices.get(signal.symbol, exec_price)
                    realized = current * (exec_price - entry) / entry if entry > 0 else 0.0
                    portfolio.cash += abs(current) + realized
                    portfolio.entry_prices.pop(signal.symbol, None)
                else:
                    old_entry = portfolio.entry_prices.get(signal.symbol, exec_price)
                    if abs(target) > abs(current):
                        added = abs(target) - abs(current)
                        portfolio.cash -= added
                        portfolio.entry_prices[signal.symbol] = ((old_entry * abs(current)) + (exec_price * added)) / max(abs(target), 1.0)
                    elif abs(target) < abs(current):
                        reduced = abs(current) - abs(target)
                        realized = np.sign(current) * reduced * (exec_price - old_entry) / old_entry if old_entry > 0 else 0.0
                        portfolio.cash += reduced + realized
                if target == 0:
                    portfolio.positions.pop(signal.symbol, None)
                    position_meta.pop(signal.symbol, None)
                else:
                    portfolio.positions[signal.symbol] = target
                    position_meta[signal.symbol] = {
                        "bundle": signal.bundle,
                        "sleeve": signal.sleeve,
                        "cluster": signal.cluster,
                        "opened_ts": timestamp,
                        "max_hold_hours": signal.holding_horizon_hours,
                        "regime_context": signal.regime_context,
                        "reason_tag": signal.reason_tag,
                    }
                _append_log(
                    log_path,
                    {
                        "timestamp": timestamp,
                        "action": event,
                        "symbol": signal.symbol,
                        "sleeve": signal.sleeve,
                        "bundle": signal.bundle,
                        "cluster": signal.cluster,
                        "signal_tag": signal.reason_tag,
                        "signal_regime_family": signal.regime_context,
                        "confidence": signal.confidence,
                        "expected_edge_bps": signal.expected_edge_bps,
                        "allocation_tier": signal.metadata.get("allocation_tier", ""),
                        "allocator_score": signal.metadata.get("allocator_score", 0.0),
                        "target_position": target,
                        "delta_notional": delta,
                        "exec_price": exec_price,
                        "fee": fee,
                        "realized_pnl": realized,
                        "participation": participation,
                        "rationale": signal.metadata.get("activation_rule", signal.reason_tag),
                        "decision_reason": signal.reason_tag,
                        "portfolio_equity": portfolio.equity,
                    },
                )
        else:
            for signal in allocated:
                _append_log(
                    log_path,
                    {
                        "timestamp": timestamp,
                        "action": "skipped",
                        "symbol": signal.symbol,
                        "sleeve": signal.sleeve,
                        "bundle": signal.bundle,
                        "cluster": signal.cluster,
                        "skip_reason": "kill_switch_halt_new_orders",
                        "portfolio_equity": portfolio.equity,
                    },
                )

        _mark_to_market(portfolio, close_by_symbol)
        portfolio.timestamp = timestamp
        if collect_bar_history:
            assert equity_curve is not None
            equity_curve.append(float(portfolio.equity))
            equity_timestamps.append(int(timestamp))
        processed += 1

    state.cash = portfolio.cash
    state.positions = dict(portfolio.positions)
    state.entry_prices = dict(portfolio.entry_prices)
    state.position_meta = dict(position_meta)
    state.recent_entries = dict(recent_entries)
    state.equity = portfolio.equity
    state.last_timestamp = portfolio.timestamp
    state.total_volume = float(getattr(portfolio, "total_volume", state.total_volume))
    save_v2_shadow_state(state_path, state)

    return {
        "bundle": bundle_name,
        "model_set": model_set,
        "split": split,
        "bars_processed": processed,
        "total_bars_available": total_bars_available,
        "first_timestamp": first_processed,
        "last_timestamp": state.last_timestamp,
        "equity": state.equity,
        "positions": len(state.positions),
        "equity_curve": equity_curve or [],
        "equity_timestamps": equity_timestamps,
    }
