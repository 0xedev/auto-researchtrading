import argparse
import json

from execution import ShadowRiskConfig, run_shadow_session, write_shadow_dashboard


def main():
    parser = argparse.ArgumentParser(description="Shadow paper-trading runner with persisted state")
    parser.add_argument("--timeframe", default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--split", default="2026q1", choices=["val", "oos", "2026q1", "robustness", "holdout", "val_15m", "oos_15m", "2026q1_15m", "holdout_15m"])
    parser.add_argument("--state-path", default="shadow_state.json")
    parser.add_argument("--log-path", default="shadow_trade_log.jsonl")
    parser.add_argument("--dashboard-path", default="shadow_dashboard.md")
    parser.add_argument("--summary-json-path", default="shadow_summary.json")
    parser.add_argument("--kill-switch-path", default=None)
    parser.add_argument("--recent-actions", type=int, default=12)
    parser.add_argument("--max-days", type=int, default=None, help="Optional historical day budget for smoke runs")
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--max-symbol-notional-pct", type=float, default=0.35)
    args = parser.parse_args()

    config = ShadowRiskConfig(
        max_leverage=args.max_leverage,
        max_symbol_notional_pct=args.max_symbol_notional_pct,
        state_path=args.state_path,
        log_path=args.log_path,
        kill_switch_path=args.kill_switch_path,
    )
    result = run_shadow_session(
        timeframe=args.timeframe,
        split=args.split,
        risk_config=config,
        max_days=args.max_days,
    )
    summary = write_shadow_dashboard(
        state_path=args.state_path,
        log_path=args.log_path,
        dashboard_path=args.dashboard_path,
        summary_json_path=args.summary_json_path,
        max_recent=args.recent_actions,
    )
    result["dashboard_path"] = args.dashboard_path
    result["summary_json_path"] = args.summary_json_path
    result["dashboard_equity"] = summary.get("portfolio", {}).get("equity", 0.0)
    result["dashboard_open_positions"] = summary.get("portfolio", {}).get("open_positions", 0)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
