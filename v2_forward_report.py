from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_log_timestamp(line: str) -> datetime | None:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_state_timestamp(value: Any) -> datetime | None:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    if abs(ts) > 1e11:
        ts /= 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _read_state(path: str | Path) -> tuple[dict[str, Any], str | None]:
    target = Path(path)
    if not target.exists():
        return {}, "missing_state"
    try:
        return json.loads(target.read_text(encoding="utf-8")), None
    except Exception:
        return {}, "state_parse_error"


def _read_log(path: str | Path) -> tuple[list[str], str | None]:
    target = Path(path)
    if not target.exists():
        return [], "missing_log"
    try:
        return target.read_text(encoding="utf-8", errors="replace").splitlines(), None
    except Exception:
        return [], "log_parse_error"


def summarize_connector(
    *,
    name: str,
    state_path: str | Path,
    log_path: str | Path,
    now: datetime | None = None,
    max_stale_hours: float = 26.0,
) -> dict[str, Any]:
    now = now or _utc_now()
    state, state_error = _read_state(state_path)
    lines, log_error = _read_log(log_path)

    timestamps = [ts for line in lines if (ts := _parse_log_timestamp(line)) is not None]
    first_log_ts = min(timestamps) if timestamps else None
    last_log_ts = max(timestamps) if timestamps else None
    log_span_days = (
        (last_log_ts - first_log_ts).total_seconds() / 86_400.0
        if first_log_ts and last_log_ts
        else 0.0
    )

    counters = Counter()
    for line in lines:
        if " ERROR" in line or "Traceback" in line:
            counters["errors"] += 1
        if " WARNING" in line:
            counters["warnings"] += 1
        if "V2 signals" in line or (" Equity=$" in line and "opens=" in line and "exits=" in line):
            counters["signal_bars"] += 1
        if "  OPEN" in line or "V2 OPEN" in line or "OPENED" in line:
            counters["opens"] += 1
        if "  CLOSE" in line or "V2 EXIT" in line or "CLOSED" in line:
            counters["closes"] += 1
        if "reconnect" in line.lower():
            counters["reconnects"] += 1
        if "halt" in line.lower():
            counters["halts"] += 1
        if "Invalid API-key" in line or " 401" in line:
            counters["auth_failures"] += 1

    positions = state.get("positions", {}) if isinstance(state.get("positions", {}), dict) else {}
    position_meta = state.get("position_meta", {}) if isinstance(state.get("position_meta", {}), dict) else {}
    last_state_ts = _parse_state_timestamp(state.get("last_timestamp"))
    stale_hours = (now - last_state_ts).total_seconds() / 3600.0 if last_state_ts else None

    alerts: list[dict[str, str]] = []
    if state_error:
        alerts.append({"severity": "critical", "code": state_error, "message": f"{name} state file is unavailable or invalid."})
    if log_error:
        alerts.append({"severity": "warning", "code": log_error, "message": f"{name} log file is unavailable or invalid."})
    if stale_hours is None:
        alerts.append({"severity": "warning", "code": "missing_state_timestamp", "message": f"{name} state has no last_timestamp."})
    elif stale_hours > max_stale_hours:
        alerts.append({"severity": "critical", "code": "stale_state", "message": f"{name} state is stale by {stale_hours:.1f} hours."})
    elif stale_hours < -1.0:
        alerts.append({"severity": "warning", "code": "future_state_timestamp", "message": f"{name} state timestamp is {-stale_hours:.1f} hours in the future."})
    if counters["errors"]:
        alerts.append({"severity": "critical", "code": "log_errors", "message": f"{name} log contains {counters['errors']} errors/tracebacks."})
    if counters["auth_failures"]:
        alerts.append({"severity": "critical", "code": "auth_failures", "message": f"{name} log contains {counters['auth_failures']} API/auth failures."})
    if counters["warnings"]:
        alerts.append({"severity": "warning", "code": "log_warnings", "message": f"{name} log contains {counters['warnings']} warnings."})
    if lines and counters["signal_bars"] == 0:
        alerts.append({"severity": "warning", "code": "no_signal_bars", "message": f"{name} log has no V2 signal bars."})

    return {
        "name": name,
        "state_path": str(state_path),
        "log_path": str(log_path),
        "state_exists": Path(state_path).exists(),
        "log_exists": Path(log_path).exists(),
        "equity": float(state.get("equity", 0.0) or 0.0),
        "cash": float(state.get("cash", 0.0) or 0.0),
        "open_positions": len(positions),
        "positions": positions,
        "position_meta_keys": sorted(position_meta.keys()),
        "last_state_timestamp": state.get("last_timestamp", 0),
        "last_state_timestamp_iso": _iso(last_state_ts),
        "state_stale_hours": stale_hours,
        "log_records": len(lines),
        "first_log_timestamp_iso": _iso(first_log_ts),
        "last_log_timestamp_iso": _iso(last_log_ts),
        "log_span_days": log_span_days,
        "counters": dict(counters),
        "alerts": alerts,
    }


def build_report(
    *,
    connectors: list[dict[str, str]],
    min_days: float = 30.0,
    max_stale_hours: float = 26.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or _utc_now()
    summaries = [
        summarize_connector(
            name=item["name"],
            state_path=item["state_path"],
            log_path=item["log_path"],
            now=now,
            max_stale_hours=max_stale_hours,
        )
        for item in connectors
    ]

    starts = [
        datetime.fromisoformat(row["first_log_timestamp_iso"])
        for row in summaries
        if row.get("first_log_timestamp_iso")
    ]
    ends = [
        datetime.fromisoformat(row["last_log_timestamp_iso"])
        for row in summaries
        if row.get("last_log_timestamp_iso")
    ]
    first_seen = min(starts) if starts else None
    last_seen = max(ends) if ends else None
    observed_days = (
        (last_seen - first_seen).total_seconds() / 86_400.0
        if first_seen and last_seen
        else 0.0
    )

    alerts = []
    for row in summaries:
        alerts.extend({**alert, "connector": row["name"]} for alert in row["alerts"])
        if row["log_span_days"] < min_days:
            alerts.append(
                {
                    "severity": "critical",
                    "code": "insufficient_connector_days",
                    "connector": row["name"],
                    "message": f"{row['name']} has only {row['log_span_days']:.2f} log days; need {min_days:.2f}.",
                }
            )
    if observed_days < min_days:
        alerts.append(
            {
                "severity": "critical",
                "code": "insufficient_forward_days",
                "connector": "portfolio",
                "message": f"Only {observed_days:.2f} observed log days; need {min_days:.2f}.",
            }
        )

    critical_count = sum(1 for alert in alerts if alert["severity"] == "critical")
    warning_count = sum(1 for alert in alerts if alert["severity"] == "warning")
    return {
        "generated_at": now.isoformat(),
        "status": "PASS" if critical_count == 0 else "FAIL",
        "min_days_required": min_days,
        "observed_days": observed_days,
        "first_seen_iso": _iso(first_seen),
        "last_seen_iso": _iso(last_seen),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "connectors": summaries,
        "alerts": alerts,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2 Forward Paper Readiness Report",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{report['status']}`",
        f"- Observed days: `{report['observed_days']:.2f}` / `{report['min_days_required']:.2f}`",
        f"- Critical alerts: `{report['critical_count']}`",
        f"- Warnings: `{report['warning_count']}`",
        "",
        "## Alerts",
        "",
    ]
    if report["alerts"]:
        for alert in report["alerts"]:
            lines.append(f"- `{alert['severity']}` `{alert['connector']}` `{alert['code']}`: {alert['message']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Connectors", ""])
    for connector in report["connectors"]:
        counters = connector.get("counters", {})
        lines.extend(
            [
                f"### {connector['name']}",
                "",
                f"- Equity: `{connector['equity']:,.2f}`",
                f"- Open positions: `{connector['open_positions']}`",
                f"- Last state timestamp: `{connector.get('last_state_timestamp_iso') or 'n/a'}`",
                f"- State stale hours: `{connector['state_stale_hours']:.2f}`" if connector.get("state_stale_hours") is not None else "- State stale hours: `n/a`",
                f"- Log span days: `{connector['log_span_days']:.2f}`",
                f"- Log records: `{connector['log_records']}`",
                f"- Signal bars: `{counters.get('signal_bars', 0)}`",
                f"- Opens / closes: `{counters.get('opens', 0)}` / `{counters.get('closes', 0)}`",
                f"- Errors / warnings: `{counters.get('errors', 0)}` / `{counters.get('warnings', 0)}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def default_connectors(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {"name": "binance", "state_path": args.binance_state, "log_path": args.binance_log},
        {"name": "deriv", "state_path": args.deriv_state, "log_path": args.deriv_log},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V2 forward-paper readiness report")
    parser.add_argument("--binance-state", default="state/v2_binance_live.json")
    parser.add_argument("--binance-log", default="logs/v2_binance.log")
    parser.add_argument("--deriv-state", default="state/v2_deriv_live.json")
    parser.add_argument("--deriv-log", default="logs/v2_deriv.log")
    parser.add_argument("--min-days", type=float, default=30.0)
    parser.add_argument("--max-stale-hours", type=float, default=26.0)
    parser.add_argument("--output-json", default="tmp/v2_forward_report.json")
    parser.add_argument("--output-md", default="tmp/v2_forward_report.md")
    args = parser.parse_args()

    report = build_report(
        connectors=default_connectors(args),
        min_days=args.min_days,
        max_stale_hours=args.max_stale_hours,
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "critical_count": report["critical_count"], "output_json": args.output_json, "output_md": args.output_md}, indent=2))


if __name__ == "__main__":
    main()
