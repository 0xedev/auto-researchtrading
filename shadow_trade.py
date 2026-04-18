import argparse
import json

from execution import ShadowRiskConfig, run_shadow_session


def main():
    parser = argparse.ArgumentParser(description="Shadow paper-trading runner with persisted state")
    parser.add_argument("--timeframe", default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--split", default="2026q1", choices=["val", "oos", "2026q1", "robustness", "holdout", "val_15m", "oos_15m", "2026q1_15m", "holdout_15m"])
    parser.add_argument("--state-path", default="shadow_state.json")
    parser.add_argument("--log-path", default="shadow_trade_log.jsonl")
    parser.add_argument("--kill-switch-path", default=None)
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
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
