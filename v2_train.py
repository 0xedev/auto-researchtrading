import argparse
import json

from v2.manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS
from v2.registry import SleeveRegistryEntry, load_sleeve_registry, save_sleeve_registry, upsert_registry_entry
from v2.training import train_sleeve


def main():
    parser = argparse.ArgumentParser(description="Train V2 sleeve models")
    parser.add_argument("--bundle", required=True, choices=sorted(BUNDLE_MANIFESTS))
    parser.add_argument("--sleeve", action="append", default=[], help="Repeat for multiple sleeves; omit to train all.")
    parser.add_argument("--model-set", required=True)
    parser.add_argument("--feature-profile", default="price_context_plus", choices=["price_only", "price_context", "price_context_plus"])
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional smoke-test cap on symbol count.")
    parser.add_argument("--registry-path", default="v2_sleeve_registry.json")
    args = parser.parse_args()

    sleeves = args.sleeve or sorted(SLEEVE_MANIFESTS)
    registry = load_sleeve_registry(args.registry_path)
    results = []
    for sleeve_name in sleeves:
        result = train_sleeve(
            bundle_name=args.bundle,
            sleeve_name=sleeve_name,
            model_set=args.model_set,
            feature_profile=args.feature_profile,
            max_symbols=args.max_symbols,
        )
        results.append(result)
        if result.get("status") in {"trained", "prepared"}:
            entry = SleeveRegistryEntry(
                sleeve=sleeve_name,
                bundle=args.bundle,
                model_set=args.model_set,
                status="candidate",
                validation_metric=float(result.get("val_auc", result.get("avg_confidence", 0.0))),
                oos_metric=0.0,
                concentration_pct=0.0,
                notes=(
                    f"train_pos_rate={result.get('train_pos_rate', 0.0):.3f}"
                    if result.get("status") == "trained"
                    else f"prepared_rows={result.get('rows', 0)}"
                ),
            )
            upsert_registry_entry(registry, entry)
    save_sleeve_registry(registry, args.registry_path)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
