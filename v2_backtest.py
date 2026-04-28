import argparse
import json
from pathlib import Path

from v2 import PortfolioConfig
from v2.backtest import BENCHMARKS, append_v2_results_row, backtest_to_dict, run_v2_backtest
from v2.manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS


def _load_portfolio_config(path: str | None) -> dict:
    if not path:
        return {}
    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text())


def _pass_fail(label: str, value: float, target: float, lower_is_better: bool = False) -> None:
    passed = (value < target) if lower_is_better else (value >= target)
    sym = "PASS" if passed else "FAIL"
    cmp = "<" if lower_is_better else ">="
    print(f"{label:<24} {value:>10.6f}   target {cmp} {target:<8.3f} {sym}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-grade V2 backtest runner")
    parser.add_argument("--bundle", required=True, choices=sorted(BUNDLE_MANIFESTS))
    parser.add_argument("--model-set", required=True)
    parser.add_argument("--split", default="val", choices=["val", "2026q1", "holdout", "newasset_oos2y", "deriv_oss4y"])
    parser.add_argument("--portfolio-config", default="v2_portfolio.example.json")
    parser.add_argument("--sleeve", action="append", default=[], help="Repeat to restrict active sleeves.")
    parser.add_argument("--symbol", action="append", default=[], help="Repeat to restrict the symbol universe.")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--description", default="V2 full backtest iteration")
    parser.add_argument("--label", default=None)
    parser.add_argument("--log-results", action="store_true", help="Append a V2 row to results.tsv")
    parser.add_argument("--results-prefix", default="v2exp")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON after the audit block")
    parser.add_argument("--final-sign-off-token", default=None)
    args = parser.parse_args()

    if args.split == "holdout" and args.final_sign_off_token != "FINAL-SIGN-OFF":
        raise SystemExit("Refusing to run V2 holdout without --final-sign-off-token FINAL-SIGN-OFF")

    config_payload = _load_portfolio_config(args.portfolio_config)
    portfolio_config = PortfolioConfig.from_dict(config_payload.get("portfolio"))
    active_sleeves = args.sleeve or config_payload.get("active_sleeves") or sorted(SLEEVE_MANIFESTS)

    backtest = run_v2_backtest(
        bundle_name=args.bundle,
        model_set=args.model_set,
        split=args.split,
        portfolio_config=portfolio_config,
        active_sleeves=active_sleeves,
        max_days=args.max_days,
        max_symbols=args.max_symbols,
        symbols=args.symbol or None,
    )

    result = backtest["result"]
    top_share = max(
        [float(row.get("pnl_share", 0.0) or 0.0) for row in backtest["summary"].get("sleeve_concentration", [])],
        default=0.0,
    )
    header = f"V2 BACKTEST [{args.bundle}]"
    if args.label:
        header += f" [{args.label}]"
    print("\n" + "=" * 60)
    print(f"  {header}")
    print("=" * 60)
    print(f"split:              {args.split}")
    print(f"model_set:          {args.model_set}")
    print(f"score:              {backtest['score']:.6f}")
    print(f"status:             {backtest['status']}")
    print(f"bar_sharpe:         {backtest['bar_sharpe']:.6f}")
    print(f"daily_sharpe:       {backtest['daily_sharpe']:.6f}")
    print(f"daily_sortino:      {backtest['daily_sortino']:.6f}")
    print(f"daily_obs:          {backtest['daily_observations']}")
    print(f"total_return_pct:   {result.total_return_pct:.6f}")
    print(f"max_drawdown_pct:   {result.max_drawdown_pct:.6f}")
    print(f"num_trades:         {result.num_trades}")
    print(f"trades_per_day:     {backtest['trades_per_day']:.6f}")
    print(f"win_rate_pct:       {result.win_rate_pct:.6f}")
    print(f"profit_factor:      {result.profit_factor:.6f}")
    print(f"annual_turnover:    {result.annual_turnover:.2f}")
    print(f"bars_processed:     {result.bars_processed}/{result.total_bars}")
    print(f"duration_days:      {result.duration_days:.2f}")
    print(f"short_share:        {backtest['short_share']:.6f}")
    print(f"concentration_pct:  {top_share * 100.0:.6f}")

    print("\n" + "=" * 60)
    print("  BENCHMARK AUDIT")
    print("=" * 60)
    _pass_fail("Daily Sharpe", backtest["daily_sharpe"], BENCHMARKS["daily_sharpe"])
    _pass_fail("Win Rate %", result.win_rate_pct, BENCHMARKS["win_rate_pct"])
    _pass_fail("Trades/Day", backtest["trades_per_day"], BENCHMARKS["trades_per_day"])
    _pass_fail("Profit Factor", result.profit_factor, BENCHMARKS["profit_factor"])
    _pass_fail("Max Drawdown %", result.max_drawdown_pct, BENCHMARKS["max_drawdown_pct"], lower_is_better=True)
    print(f"Metric caveat:      {backtest['metric_caveat']}")

    top_sleeves = backtest.get("top_sleeves", []) or []
    if top_sleeves:
        print("\nTop Sleeves")
        for row in top_sleeves:
            print(f"  {row.get('sleeve',''): <28} share={float(row.get('pnl_share', 0.0) or 0.0):.4f}")

    if args.log_results:
        run_id = append_v2_results_row(
            backtest=backtest,
            description=args.description,
            prefix=args.results_prefix,
        )
        print(f"\nLogged to results.tsv as {run_id} [{backtest['status']}]")
    else:
        print("\nSkipping results.tsv append (use --log-results to persist this run).")

    if args.json:
        print("\n" + json.dumps(backtest_to_dict(backtest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
