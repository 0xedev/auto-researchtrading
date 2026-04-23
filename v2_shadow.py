import argparse
import json
from pathlib import Path

from execution.report import write_shadow_dashboard
from execution.v2_paper import run_v2_shadow_session
from v2 import PortfolioConfig
from v2.manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS


def _load_portfolio_config(path: str | None) -> dict:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text())


def main():
    parser = argparse.ArgumentParser(description="Run the V2 multi-alpha shadow portfolio")
    parser.add_argument("--bundle", required=True, choices=sorted(BUNDLE_MANIFESTS))
    parser.add_argument("--model-set", required=True)
    parser.add_argument("--split", default="2026q1")
    parser.add_argument("--portfolio-config", default="v2_portfolio.example.json")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--dashboard-path", default=None)
    parser.add_argument("--summary-json-path", default=None)
    parser.add_argument("--kill-switch-path", default=None)
    parser.add_argument("--sleeve", action="append", default=[], help="Repeat to restrict active sleeves.")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()

    config_payload = _load_portfolio_config(args.portfolio_config)
    portfolio_config = PortfolioConfig.from_dict(config_payload.get("portfolio"))

    state_path = args.state_path or config_payload.get("state_path", "v2_shadow_state.json")
    log_path = args.log_path or config_payload.get("log_path", "v2_shadow_log.jsonl")
    dashboard_path = args.dashboard_path or config_payload.get("dashboard_path", "v2_shadow_dashboard.md")
    summary_json_path = args.summary_json_path or config_payload.get("summary_json_path", "v2_shadow_summary.json")
    kill_switch_path = args.kill_switch_path or config_payload.get("kill_switch_path")
    active_sleeves = args.sleeve or config_payload.get("active_sleeves") or sorted(SLEEVE_MANIFESTS)

    result = run_v2_shadow_session(
        bundle_name=args.bundle,
        model_set=args.model_set,
        split=args.split,
        portfolio_config=portfolio_config,
        state_path=state_path,
        log_path=log_path,
        kill_switch_path=kill_switch_path,
        max_days=args.max_days,
        active_sleeves=active_sleeves,
        max_symbols=args.max_symbols,
    )
    summary = write_shadow_dashboard(
        state_path=state_path,
        log_path=log_path,
        dashboard_path=dashboard_path,
        summary_json_path=summary_json_path,
        max_recent=12,
    )
    result["dashboard_path"] = dashboard_path
    result["summary_json_path"] = summary_json_path
    result["dashboard_equity"] = summary.get("portfolio", {}).get("equity", 0.0)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
