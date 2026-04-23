import argparse
import json

from v2 import PortfolioConfig
from v2.evaluation import evaluate_candidate, evaluate_model_set
from v2.manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS
from v2.registry import find_live_champion, load_sleeve_registry, save_sleeve_registry, upsert_registry_entry
from v2.types import SleeveRegistryEntry


def _load_portfolio_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Evaluate V2 sleeves or model sets with stress checks")
    parser.add_argument("--bundle", required=True, choices=sorted(BUNDLE_MANIFESTS))
    parser.add_argument("--model-set", required=True)
    parser.add_argument("--portfolio-config", default="v2_portfolio.example.json")
    parser.add_argument("--sleeve", action="append", default=[], help="Repeat to evaluate specific sleeves; omit to evaluate trained model set.")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--rolling-window-days", type=int, default=None)
    parser.add_argument("--rolling-step-days", type=int, default=None)
    parser.add_argument("--rolling-max-windows", type=int, default=None)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--registry-path", default="v2_sleeve_registry.json")
    parser.add_argument("--update-registry", action="store_true")
    parser.add_argument("--promote-if-pass", action="store_true")
    args = parser.parse_args()

    config_payload = _load_portfolio_config(args.portfolio_config)
    portfolio_config = PortfolioConfig.from_dict(config_payload.get("portfolio"))

    if args.sleeve:
        registry = load_sleeve_registry(args.registry_path)
        results = []
        for sleeve_name in args.sleeve:
            if sleeve_name not in SLEEVE_MANIFESTS:
                raise ValueError(f"Unknown sleeve: {sleeve_name}")
            evaluation = evaluate_candidate(
                bundle_name=args.bundle,
                sleeve_name=sleeve_name,
                model_set=args.model_set,
                portfolio_config=portfolio_config,
                max_days=args.max_days,
                max_symbols=args.max_symbols,
                rolling_window_days=args.rolling_window_days,
                rolling_step_days=args.rolling_step_days,
                rolling_max_windows=args.rolling_max_windows,
                include_stress=not args.skip_stress,
            )
            summary = evaluation["summary"]
            champion = find_live_champion(registry, bundle=args.bundle, sleeve=sleeve_name)
            suggested_status = "candidate"
            if summary["meets_gate"]:
                suggested_status = "paper-live"
                if champion is None or (
                    summary["validation_metric"] >= champion.validation_metric
                    and summary["oos_metric"] >= champion.oos_metric
                ):
                    suggested_status = "live-champion"

            evaluation["summary"]["current_champion_model_set"] = champion.model_set if champion else ""
            evaluation["summary"]["suggested_status"] = suggested_status
            results.append(evaluation)

            if args.update_registry:
                status = suggested_status if args.promote_if_pass and summary["meets_gate"] else "candidate"
                entry = SleeveRegistryEntry(
                    sleeve=sleeve_name,
                    bundle=args.bundle,
                    model_set=args.model_set,
                    status=status,
                    validation_metric=summary["validation_metric"],
                    oos_metric=summary["oos_metric"],
                    stress_metric=summary["stress_metric"],
                    concentration_pct=summary["concentration_pct"],
                    trades_per_day=summary["trades_per_day"],
                    short_share=summary["short_share"],
                    shadow_ready=summary["shadow_ready"],
                    last_updated=evaluation["generated_at"],
                    notes=summary["notes"],
                )
                upsert_registry_entry(registry, entry)
                if args.promote_if_pass and status == "live-champion":
                    for row in registry.entries:
                        if (
                            row.bundle == args.bundle
                            and row.sleeve == sleeve_name
                            and row.model_set != args.model_set
                            and row.status == "live-champion"
                        ):
                            row.status = "paper-live"
                save_sleeve_registry(registry, args.registry_path)

        print(json.dumps(results, indent=2, sort_keys=True))
        return

    evaluation = evaluate_model_set(
        bundle_name=args.bundle,
        model_set=args.model_set,
        portfolio_config=portfolio_config,
        active_sleeves=[],
        max_days=args.max_days,
        max_symbols=args.max_symbols,
        rolling_window_days=args.rolling_window_days,
        rolling_step_days=args.rolling_step_days,
        rolling_max_windows=args.rolling_max_windows,
        include_stress=not args.skip_stress,
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
