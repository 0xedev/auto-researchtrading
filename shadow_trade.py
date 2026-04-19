import argparse
import contextlib
import io
import json
import sys

from execution import ShadowRiskConfig, resolve_shadow_runtime_config, run_shadow_session, write_shadow_dashboard


def main():
    parser = argparse.ArgumentParser(description="Shadow paper-trading runner with persisted state")
    parser.add_argument("--config-path", default=None, help="Optional JSON config file for runtime, risk, and output paths")
    parser.add_argument("--timeframe", default=None, choices=["15m", "1h", "4h"])
    parser.add_argument("--split", default=None, choices=["val", "oos", "2026q1", "robustness", "holdout", "val_15m", "oos_15m", "2026q1_15m", "holdout_15m"])
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--dashboard-path", default=None)
    parser.add_argument("--summary-json-path", default=None)
    parser.add_argument("--kill-switch-path", default=None)
    parser.add_argument("--recent-actions", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None, help="Optional historical day budget for smoke runs")
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--max-symbol-notional-pct", type=float, default=None)
    parser.add_argument("--json-only", action="store_true", help="Emit only the final result JSON on stdout")
    args = parser.parse_args()

    runtime = resolve_shadow_runtime_config(
        config_path=args.config_path,
        overrides={
            "timeframe": args.timeframe,
            "split": args.split,
            "state_path": args.state_path,
            "log_path": args.log_path,
            "dashboard_path": args.dashboard_path,
            "summary_json_path": args.summary_json_path,
            "kill_switch_path": args.kill_switch_path,
            "recent_actions": args.recent_actions,
            "max_days": args.max_days,
            "max_leverage": args.max_leverage,
            "max_symbol_notional_pct": args.max_symbol_notional_pct,
        },
    )
    paths = runtime["paths"]
    risk = runtime["risk"]
    operator = runtime["operator"]

    config = ShadowRiskConfig(
        max_leverage=risk["max_leverage"],
        max_symbol_notional_pct=risk["max_symbol_notional_pct"],
        state_path=paths["state_path"],
        log_path=paths["log_path"],
        kill_switch_path=paths.get("kill_switch_path"),
        config_source=runtime.get("config_source", "defaults"),
        operator_settings=operator,
    )
    capture = io.StringIO()
    stream = capture if args.json_only else None
    with contextlib.redirect_stdout(stream) if stream is not None else contextlib.nullcontext():
        result = run_shadow_session(
            timeframe=runtime["timeframe"],
            split=runtime["split"],
            risk_config=config,
            max_days=runtime["max_days"],
        )
        summary = write_shadow_dashboard(
            state_path=paths["state_path"],
            log_path=paths["log_path"],
            dashboard_path=paths["dashboard_path"],
            summary_json_path=paths["summary_json_path"],
            max_recent=operator["recent_actions"],
        )
    if args.json_only:
        captured = capture.getvalue().strip()
        if captured:
            sys.stderr.write(captured + "\n")
    result["dashboard_path"] = paths["dashboard_path"]
    result["summary_json_path"] = paths["summary_json_path"]
    result["config_path"] = args.config_path
    result["dashboard_equity"] = summary.get("portfolio", {}).get("equity", 0.0)
    result["dashboard_open_positions"] = summary.get("portfolio", {}).get("open_positions", 0)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
