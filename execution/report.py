from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .paper import ShadowState, load_shadow_state


def _read_jsonl(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _fmt_ts(ts: int | float | None) -> str:
    if not ts:
        return "n/a"
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def summarize_shadow_run(
    state_path: str | Path,
    log_path: str | Path,
    max_recent: int = 12,
) -> dict:
    state: ShadowState = load_shadow_state(state_path)
    rows = _read_jsonl(log_path)

    gross_exposure = float(sum(abs(v) for v in state.positions.values()))
    gross_leverage = gross_exposure / state.equity if state.equity > 0 else 0.0

    action_counts = Counter()
    signal_counts = Counter()
    regime_counts = Counter()
    decision_reason_counts = Counter()
    symbol_stats = defaultdict(lambda: {"actions": 0, "realized_pnl": 0.0, "fees": 0.0})

    realized_pnl_total = 0.0
    fees_total = 0.0

    for row in rows:
        action = row.get("action", "unknown")
        signal_tag = row.get("signal_tag", "")
        regime = row.get("signal_regime_family", "unknown")
        symbol = row.get("symbol", "")
        reason = row.get("decision_reason", "") or row.get("reject_reason", "") or row.get("skip_reason", "")

        action_counts[action] += 1
        if signal_tag:
            signal_counts[signal_tag] += 1
        if regime:
            regime_counts[regime] += 1
        if reason:
            decision_reason_counts[reason] += 1

        stats = symbol_stats[symbol]
        stats["actions"] += 1
        stats["realized_pnl"] += float(row.get("realized_pnl", 0.0) or 0.0)
        stats["fees"] += float(row.get("fee", 0.0) or 0.0)

        realized_pnl_total += float(row.get("realized_pnl", 0.0) or 0.0)
        fees_total += float(row.get("fee", 0.0) or 0.0)

    open_positions = [
        {
            "symbol": symbol,
            "side": "long" if notional > 0 else "short",
            "notional": float(notional),
            "entry_price": float(state.entry_prices.get(symbol, 0.0)),
        }
        for symbol, notional in sorted(state.positions.items(), key=lambda item: abs(item[1]), reverse=True)
    ]

    recent_actions = []
    for row in rows[-max_recent:]:
        recent_actions.append(
            {
                "timestamp": int(row.get("timestamp", 0) or 0),
                "timestamp_iso": _fmt_ts(row.get("timestamp")),
                "action": row.get("action", ""),
                "symbol": row.get("symbol", ""),
                "signal_tag": row.get("signal_tag", ""),
                "regime": row.get("signal_regime_family", "unknown"),
                "rationale": row.get("rationale", ""),
                "decision_reason": row.get("decision_reason", "") or row.get("reject_reason", "") or row.get("skip_reason", ""),
                "exec_price": float(row.get("exec_price", 0.0) or 0.0),
                "realized_pnl": float(row.get("realized_pnl", 0.0) or 0.0),
                "portfolio_equity": float(row.get("portfolio_equity", 0.0) or 0.0),
            }
        )

    symbol_pnl = [
        {
            "symbol": symbol,
            "actions": int(stats["actions"]),
            "realized_pnl": float(stats["realized_pnl"]),
            "fees": float(stats["fees"]),
        }
        for symbol, stats in sorted(symbol_stats.items(), key=lambda item: item[1]["realized_pnl"], reverse=True)
    ]

    return {
        "portfolio": {
            "equity": float(state.equity),
            "cash": float(state.cash),
            "gross_exposure": gross_exposure,
            "gross_leverage": float(gross_leverage),
            "open_positions": int(len(state.positions)),
            "last_timestamp": int(state.last_timestamp),
            "last_timestamp_iso": _fmt_ts(state.last_timestamp),
            "total_volume": float(state.total_volume),
        },
        "open_positions": open_positions,
        "log_stats": {
            "records": int(len(rows)),
            "realized_pnl_total": float(realized_pnl_total),
            "fees_total": float(fees_total),
            "action_counts": dict(action_counts),
            "signal_counts": dict(signal_counts),
            "regime_counts": dict(regime_counts),
            "decision_reason_counts": dict(decision_reason_counts),
        },
        "symbol_pnl": symbol_pnl,
        "recent_actions": list(reversed(recent_actions)),
    }


def render_shadow_dashboard(summary: dict) -> str:
    portfolio = summary.get("portfolio", {})
    lines = [
        "# Shadow Trading Dashboard",
        "",
        "## Overview",
        "",
        f"- Equity: `{portfolio.get('equity', 0.0):,.2f}`",
        f"- Cash: `{portfolio.get('cash', 0.0):,.2f}`",
        f"- Gross Exposure: `{portfolio.get('gross_exposure', 0.0):,.2f}`",
        f"- Gross Leverage: `{portfolio.get('gross_leverage', 0.0):.3f}`",
        f"- Open Positions: `{portfolio.get('open_positions', 0)}`",
        f"- Last Timestamp: `{portfolio.get('last_timestamp_iso', 'n/a')}`",
        f"- Total Volume: `{portfolio.get('total_volume', 0.0):,.2f}`",
        "",
        "## Open Positions",
        "",
    ]

    positions = summary.get("open_positions", [])
    if positions:
        lines.extend(
            [
                "| Symbol | Side | Notional | Entry Price |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in positions:
            lines.append(
                f"| {row['symbol']} | {row['side']} | {row['notional']:.2f} | {row['entry_price']:.2f} |"
            )
    else:
        lines.append("No open positions.")

    log_stats = summary.get("log_stats", {})
    lines.extend(
        [
            "",
            "## Run Stats",
            "",
            f"- Log Records: `{log_stats.get('records', 0)}`",
            f"- Realized PnL: `{log_stats.get('realized_pnl_total', 0.0):,.2f}`",
            f"- Fees: `{log_stats.get('fees_total', 0.0):,.2f}`",
            "",
            "### Action Mix",
            "",
        ]
    )

    action_counts = log_stats.get("action_counts", {})
    if action_counts:
        for action, count in sorted(action_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{action}`: {count}")
    else:
        lines.append("- No actions recorded.")

    lines.extend(["", "### Signal Mix", ""])
    signal_counts = log_stats.get("signal_counts", {})
    if signal_counts:
        for signal, count in sorted(signal_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{signal}`: {count}")
    else:
        lines.append("- No signal tags recorded.")

    lines.extend(["", "### Regime Mix", ""])
    regime_counts = log_stats.get("regime_counts", {})
    if regime_counts:
        for regime, count in sorted(regime_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{regime}`: {count}")
    else:
        lines.append("- No regime tags recorded.")

    lines.extend(["", "## Symbol PnL", ""])
    symbol_pnl = summary.get("symbol_pnl", [])
    if symbol_pnl:
        lines.extend(
            [
                "| Symbol | Actions | Realized PnL | Fees |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in symbol_pnl[:12]:
            lines.append(
                f"| {row['symbol']} | {row['actions']} | {row['realized_pnl']:.2f} | {row['fees']:.2f} |"
            )
    else:
        lines.append("No symbol-level PnL yet.")

    lines.extend(["", "## Recent Decisions", ""])
    recent_actions = summary.get("recent_actions", [])
    if recent_actions:
        lines.extend(
            [
                "| Time | Action | Symbol | Signal | Regime | Equity | Reason |",
                "| --- | --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in recent_actions:
            reason = (row.get("decision_reason") or row.get("rationale") or "").replace("|", "/")
            lines.append(
                f"| {row['timestamp_iso']} | {row['action']} | {row['symbol']} | {row['signal_tag']} | "
                f"{row['regime']} | {row['portfolio_equity']:.2f} | {reason[:120]} |"
            )
    else:
        lines.append("No recent actions recorded.")

    return "\n".join(lines) + "\n"


def write_shadow_dashboard(
    state_path: str | Path,
    log_path: str | Path,
    dashboard_path: str | Path | None = None,
    summary_json_path: str | Path | None = None,
    max_recent: int = 12,
) -> dict:
    summary = summarize_shadow_run(state_path=state_path, log_path=log_path, max_recent=max_recent)
    if dashboard_path:
        target = Path(dashboard_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_shadow_dashboard(summary), encoding="utf-8")
    if summary_json_path:
        target = Path(summary_json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
