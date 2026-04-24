import argparse
import copy
import json
from pathlib import Path

from v2 import PortfolioConfig
from v2.backtest import run_v2_backtest
from v2.manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS
from v2.promotion import PromotionThresholds, build_promotion_report


def _load_portfolio_payload(path: str | None) -> dict:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        raise SystemExit(f"Portfolio config not found: {path}")
    return json.loads(target.read_text(encoding="utf-8"))


def _portfolio_from_path(path: str | None) -> PortfolioConfig:
    payload = _load_portfolio_payload(path)
    return PortfolioConfig.from_dict(payload.get("portfolio"))


def _active_sleeves_from_payload(payload: dict, overrides: list[str]) -> list[str]:
    return overrides or payload.get("active_sleeves") or sorted(SLEEVE_MANIFESTS)


def _derive_stress_config(portfolio_config: PortfolioConfig) -> PortfolioConfig:
    stress_config = copy.deepcopy(portfolio_config)
    stress_config.slippage_bps = max(portfolio_config.slippage_bps + 3.0, portfolio_config.slippage_bps * 2.0)
    stress_config.max_participation_rate = max(portfolio_config.max_participation_rate * 0.5, 0.005)
    return stress_config


def _thresholds_from_args(args: argparse.Namespace) -> PromotionThresholds:
    return PromotionThresholds(
        min_bar_sharpe=args.min_bar_sharpe,
        min_daily_sharpe=args.min_daily_sharpe,
        min_daily_sortino=args.min_daily_sortino,
        min_profit_factor=args.min_profit_factor,
        min_win_rate_pct=args.min_win_rate_pct,
        min_trades_per_day=args.min_trades_per_day,
        max_trades_per_day=args.max_trades_per_day,
        max_drawdown_pct=args.max_drawdown_pct,
        max_concentration_pct=args.max_concentration_pct,
        min_daily_observations=args.min_daily_observations,
    )


def _print_gate_report(report: dict) -> None:
    print("\n" + "=" * 72)
    print(f"  V2 PRODUCTION PROMOTION GATE: {report['status']}")
    print("=" * 72)
    print(f"bundle:             {report['bundle']}")
    print(f"model_set:          {report['model_set']}")
    print(f"portfolio_config:   {report['portfolio_config']}")
    print(f"oos_split:          {report['oos_split']}")
    print(f"fresh_oos_label:    {report['fresh_oos_label'] or '<missing>'}")
    print(f"active_sleeves:     {len(report['active_sleeves'])}")

    print("\nMetrics")
    for family in ("base", "stress"):
        for split, metrics in report.get(family, {}).items():
            print(
                f"  {family}:{split:<7} "
                f"bar_sharpe={metrics['bar_sharpe']:.3f} "
                f"daily_sharpe={metrics['daily_sharpe']:.3f} "
                f"sortino={metrics['daily_sortino']:.3f} "
                f"PF={metrics['profit_factor']:.3f} "
                f"win={metrics['win_rate_pct']:.2f}% "
                f"tpd={metrics['trades_per_day']:.2f} "
                f"DD={metrics['max_drawdown_pct']:.2f}% "
                f"conc={metrics['concentration_pct']:.2f}%"
            )

    failed_checks = [check for check in report["checks"] if not check["passed"]]
    print("\nFailed Checks")
    if not failed_checks:
        print("  none")
    for check in failed_checks:
        print(
            f"  {check['scope']}:{check['name']} "
            f"value={check['value']} target {check['operator']} {check['target']}"
        )
    print(f"\n{report['notes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict V2 production promotion gate")
    parser.add_argument("--bundle", required=True, choices=sorted(BUNDLE_MANIFESTS))
    parser.add_argument("--model-set", required=True)
    parser.add_argument("--portfolio-config", required=True)
    parser.add_argument("--stress-portfolio-config", default=None)
    parser.add_argument("--val-split", default="val", choices=["val", "2026q1", "holdout", "newasset_oos2y"])
    parser.add_argument("--oos-split", default="2026q1", choices=["val", "2026q1", "holdout", "newasset_oos2y"])
    parser.add_argument("--fresh-oos-label", default="")
    parser.add_argument("--sleeve", action="append", default=[], help="Repeat to restrict active sleeves.")
    parser.add_argument("--symbol", action="append", default=[], help="Repeat to restrict the symbol universe.")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON after the text report.")
    parser.add_argument("--final-sign-off-token", default=None)
    parser.add_argument("--min-bar-sharpe", type=float, default=3.5)
    parser.add_argument("--min-daily-sharpe", type=float, default=3.0)
    parser.add_argument("--min-daily-sortino", type=float, default=3.0)
    parser.add_argument("--min-profit-factor", type=float, default=4.0)
    parser.add_argument("--min-win-rate-pct", type=float, default=60.0)
    parser.add_argument("--min-trades-per-day", type=float, default=5.0)
    parser.add_argument("--max-trades-per-day", type=float, default=30.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--max-concentration-pct", type=float, default=30.0)
    parser.add_argument("--min-daily-observations", type=int, default=20)
    args = parser.parse_args()

    holdout_used = args.val_split == "holdout" or args.oos_split == "holdout"
    if holdout_used and args.final_sign_off_token != "FINAL-SIGN-OFF":
        raise SystemExit("Refusing to run V2 holdout without --final-sign-off-token FINAL-SIGN-OFF")

    base_payload = _load_portfolio_payload(args.portfolio_config)
    portfolio_config = PortfolioConfig.from_dict(base_payload.get("portfolio"))
    active_sleeves = _active_sleeves_from_payload(base_payload, args.sleeve)
    if args.stress_portfolio_config:
        stress_config = _portfolio_from_path(args.stress_portfolio_config)
    else:
        stress_config = _derive_stress_config(portfolio_config)

    base_runs = {}
    stress_runs = {}
    for split in (args.val_split, args.oos_split):
        base_runs[split] = run_v2_backtest(
            bundle_name=args.bundle,
            model_set=args.model_set,
            split=split,
            portfolio_config=portfolio_config,
            active_sleeves=active_sleeves,
            max_days=args.max_days,
            max_symbols=args.max_symbols,
            symbols=args.symbol or None,
        )
        stress_runs[split] = run_v2_backtest(
            bundle_name=args.bundle,
            model_set=args.model_set,
            split=split,
            portfolio_config=stress_config,
            active_sleeves=active_sleeves,
            max_days=args.max_days,
            max_symbols=args.max_symbols,
            symbols=args.symbol or None,
        )

    report = build_promotion_report(
        bundle=args.bundle,
        model_set=args.model_set,
        portfolio_config=args.portfolio_config,
        active_sleeves=active_sleeves,
        base_runs=base_runs,
        stress_runs=stress_runs,
        thresholds=_thresholds_from_args(args),
        fresh_oos_label=args.fresh_oos_label or None,
        oos_split=args.oos_split,
        holdout_used=holdout_used,
    )

    _print_gate_report(report)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote promotion report to {args.output}")
    if args.json:
        print("\n" + json.dumps(report, indent=2, sort_keys=True))

    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
