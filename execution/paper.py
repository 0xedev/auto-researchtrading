from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import prepare
from prepare import BarData, PortfolioState, load_data
from strategy import Strategy


TIMEFRAME_SECS = {"15m": 900, "1h": 3600, "4h": 14400}


@dataclass
class ShadowRiskConfig:
    max_leverage: float = 3.0
    max_symbol_notional_pct: float = 0.35
    state_path: str = "shadow_state.json"
    log_path: str = "shadow_trade_log.jsonl"
    kill_switch_path: str | None = None


@dataclass
class ShadowState:
    cash: float = prepare.INITIAL_CAPITAL
    positions: dict = field(default_factory=dict)
    entry_prices: dict = field(default_factory=dict)
    equity: float = prepare.INITIAL_CAPITAL
    last_timestamp: int = 0
    total_volume: float = 0.0
    strategy_state: dict = field(default_factory=dict)


def load_shadow_state(path: str | Path) -> ShadowState:
    path = Path(path)
    if not path.exists():
        return ShadowState()
    payload = json.loads(path.read_text())
    return ShadowState(
        cash=float(payload.get("cash", prepare.INITIAL_CAPITAL)),
        positions={k: float(v) for k, v in payload.get("positions", {}).items()},
        entry_prices={k: float(v) for k, v in payload.get("entry_prices", {}).items()},
        equity=float(payload.get("equity", prepare.INITIAL_CAPITAL)),
        last_timestamp=int(payload.get("last_timestamp", 0)),
        total_volume=float(payload.get("total_volume", 0.0)),
        strategy_state=payload.get("strategy_state", {}) or {},
    )


def save_shadow_state(path: str | Path, state: ShadowState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))


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
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def _append_log(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _mark_to_market(portfolio: PortfolioState, bar_data: dict) -> float:
    unrealized_pnl = 0.0
    for sym, pos_notional in portfolio.positions.items():
        if sym not in bar_data:
            continue
        current_price = bar_data[sym].close
        entry_price = portfolio.entry_prices.get(sym, current_price)
        if entry_price > 0:
            price_change = (current_price - entry_price) / entry_price
            unrealized_pnl += pos_notional * price_change
    equity = portfolio.cash + sum(abs(v) for v in portfolio.positions.values()) + unrealized_pnl
    portfolio.equity = equity
    return equity


def _extract_signal_metadata(signal, bar, current_pos, target_pos, reason: str = "") -> dict:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    tag = getattr(signal, "tag", "")
    rationale = (
        f"{tag} in {metadata.get('signal_regime_family', 'unknown')} "
        f"meta={metadata.get('meta_score', 0.0):.3f} "
        f"short_conf={metadata.get('short_conf_15m', 0.0):.3f} "
        f"struct={metadata.get('structure_score', 0)} "
        f"sent={metadata.get('context_sentiment', 0.0):.2f}"
    )
    metadata.update(
        {
            "signal_tag": tag,
            "symbol": signal.symbol,
            "current_position": float(current_pos),
            "target_position": float(target_pos),
            "bar_close": float(bar.close),
            "rationale": rationale,
            "decision_reason": reason or metadata.get("size_reason") or metadata.get("exit_reason", ""),
        }
    )
    return metadata


def _apply_fill(portfolio: PortfolioState, signal, bar, risk_config: ShadowRiskConfig):
    current_pos = portfolio.positions.get(signal.symbol, 0.0)
    delta = signal.target_position - current_pos
    if abs(delta) < 1.0:
        return None

    new_positions = dict(portfolio.positions)
    new_positions[signal.symbol] = signal.target_position
    total_exposure = sum(abs(v) for v in new_positions.values())
    if total_exposure > portfolio.equity * risk_config.max_leverage:
        return {"status": "rejected", "reason": "max_leverage"}

    symbol_cap = portfolio.equity * risk_config.max_symbol_notional_pct
    target = signal.target_position
    if abs(target) > symbol_cap:
        target = np.sign(target) * symbol_cap
        delta = target - current_pos
        if abs(delta) < 1.0:
            return {"status": "rejected", "reason": "symbol_cap"}

    slippage = bar.close * prepare.SLIPPAGE_BPS / 10000.0
    exec_price = bar.close + slippage if delta > 0 else bar.close - slippage
    fee = abs(delta) * prepare.TAKER_FEE
    portfolio.cash -= fee

    event = "modify"
    realized_pnl = 0.0
    if target == 0:
        event = "close"
        entry_price = portfolio.entry_prices.get(signal.symbol, exec_price)
        if entry_price > 0:
            realized_pnl = current_pos * (exec_price - entry_price) / entry_price
            portfolio.cash += abs(current_pos) + realized_pnl
        portfolio.positions.pop(signal.symbol, None)
        portfolio.entry_prices.pop(signal.symbol, None)
    elif current_pos == 0:
        event = "open"
        portfolio.cash -= abs(target)
        portfolio.positions[signal.symbol] = target
        portfolio.entry_prices[signal.symbol] = exec_price
    else:
        old_notional = abs(current_pos)
        old_entry = portfolio.entry_prices.get(signal.symbol, exec_price)
        if abs(target) < abs(current_pos):
            reduced = abs(current_pos) - abs(target)
            realized_pnl = (current_pos / abs(current_pos)) * reduced * (exec_price - old_entry) / old_entry if old_entry > 0 else 0.0
            portfolio.cash += reduced + realized_pnl
        elif abs(target) > abs(current_pos):
            added = abs(target) - abs(current_pos)
            portfolio.cash -= added
            if old_notional + added > 0:
                portfolio.entry_prices[signal.symbol] = (old_entry * old_notional + exec_price * added) / (old_notional + added)
        portfolio.positions[signal.symbol] = target

    return {
        "status": "filled",
        "event": event,
        "target_position": float(target),
        "delta": float(delta),
        "exec_price": float(exec_price),
        "fee": float(fee),
        "realized_pnl": float(realized_pnl),
    }


def _build_bar_data(data: dict, timestamps: list[int]) -> tuple[dict, list[int]]:
    indexed = {}
    for symbol, df in data.items():
        df_int = df.copy()
        ts_raw = df_int["timestamp"].astype(np.int64).values
        df_int["timestamp"] = np.where(ts_raw > 1e11, ts_raw // 1000, ts_raw)
        indexed[symbol] = df_int.set_index("timestamp")
    return indexed, timestamps


def run_shadow_session(timeframe: str, split: str, risk_config: ShadowRiskConfig, max_days: int | None = None) -> dict:
    data = load_data(split=split, resample_4h=(timeframe == "4h"))
    strategy = Strategy(timeframe=timeframe)
    strategy.pre_calculate_signals(data, split_name=split)

    state = load_shadow_state(risk_config.state_path)
    if state.strategy_state:
        strategy.trailing_stops = state.strategy_state.get("trailing_stops", {})
        strategy.position_ages = {k: int(v) for k, v in state.strategy_state.get("position_ages", {}).items()}
        strategy._market_ret_buf = list(state.strategy_state.get("market_ret_buf", []))
        strategy.bar_counts = {k: int(v) for k, v in state.strategy_state.get("bar_counts", {}).items()}
        strategy._macro_bear = bool(state.strategy_state.get("macro_bear", False))

    portfolio = PortfolioState(
        cash=state.cash,
        positions=dict(state.positions),
        entry_prices=dict(state.entry_prices),
        equity=state.equity,
        timestamp=state.last_timestamp,
    )

    all_timestamps = set()
    for _, df in data.items():
        ts_raw = df["timestamp"].astype(np.int64).values
        ts_norm = np.where(ts_raw > 1e11, ts_raw // 1000, ts_raw)
        all_timestamps.update(ts_norm.tolist())
    timestamps = sorted(all_timestamps)
    indexed, _ = _build_bar_data(data, timestamps)

    start_ts = state.last_timestamp
    first_processed_ts = None
    processed = 0

    history_buffers = {symbol: [] for symbol in data}
    if start_ts:
        for symbol, count in strategy.bar_counts.items():
            if symbol in indexed and count > 0:
                hist_rows = indexed[symbol].iloc[max(0, count - prepare.LOOKBACK_BARS):count]
                history_buffers[symbol] = hist_rows.reset_index().to_dict("records")

    for ts in timestamps:
        if ts <= start_ts:
            continue
        if first_processed_ts is None:
            first_processed_ts = ts
        if max_days is not None and first_processed_ts is not None:
            if (ts - first_processed_ts) > max_days * 24 * 3600:
                break

        bar_data = {}
        for symbol in data:
            if symbol not in indexed or ts not in indexed[symbol].index:
                continue
            row = indexed[symbol].loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            bar_dict = {
                "timestamp": ts,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "funding_rate": row.get("funding_rate", 0.0),
            }
            history_buffers[symbol].append(bar_dict)
            if len(history_buffers[symbol]) > prepare.LOOKBACK_BARS:
                history_buffers[symbol] = history_buffers[symbol][-prepare.LOOKBACK_BARS:]
            bar_data[symbol] = BarData(
                symbol=symbol,
                timestamp=int(ts),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                funding_rate=row.get("funding_rate", 0.0),
                history=pd.DataFrame(history_buffers[symbol]),
            )

        if not bar_data:
            continue

        portfolio.timestamp = ts
        _mark_to_market(portfolio, bar_data)
        signals = strategy.on_bar(bar_data, portfolio) or []
        kill_switch = _load_kill_switch(risk_config.kill_switch_path)
        halt_new_orders = bool(kill_switch.get("halt_new_orders"))

        for signal in signals:
            if signal.symbol not in bar_data:
                continue
            bar = bar_data[signal.symbol]
            current_pos = portfolio.positions.get(signal.symbol, 0.0)
            target_pos = signal.target_position
            metadata = _extract_signal_metadata(signal, bar, current_pos, target_pos)
            if halt_new_orders and current_pos == 0 and target_pos != 0:
                _append_log(
                    risk_config.log_path,
                    {
                        "timestamp": ts,
                        "action": "skipped",
                        "skip_reason": "kill_switch_halt_new_orders",
                        **metadata,
                    },
                )
                continue

            fill = _apply_fill(portfolio, signal, bar, risk_config)
            if fill is None:
                continue
            if fill["status"] != "filled":
                _append_log(
                    risk_config.log_path,
                    {
                        "timestamp": ts,
                        "action": "rejected",
                        "reject_reason": fill["reason"],
                        **metadata,
                    },
                )
                continue

            state.total_volume += abs(fill["delta"])
            _mark_to_market(portfolio, bar_data)
            _append_log(
                risk_config.log_path,
                {
                    "timestamp": ts,
                    "action": fill["event"],
                    "exec_price": fill["exec_price"],
                    "fee": fill["fee"],
                    "realized_pnl": fill["realized_pnl"],
                    "portfolio_equity": portfolio.equity,
                    **metadata,
                },
            )

        processed += 1
        state.cash = portfolio.cash
        state.positions = dict(portfolio.positions)
        state.entry_prices = dict(portfolio.entry_prices)
        state.equity = portfolio.equity
        state.last_timestamp = ts
        state.strategy_state = {
            "trailing_stops": strategy.trailing_stops,
            "position_ages": strategy.position_ages,
            "market_ret_buf": strategy._market_ret_buf,
            "bar_counts": strategy.bar_counts,
            "macro_bear": strategy._macro_bear,
        }
        save_shadow_state(risk_config.state_path, state)

    return {
        "bars_processed": processed,
        "last_timestamp": state.last_timestamp,
        "equity": state.equity,
        "open_positions": len(state.positions),
        "state_path": risk_config.state_path,
        "log_path": risk_config.log_path,
    }
