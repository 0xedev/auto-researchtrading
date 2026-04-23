import argparse

from v2.audit import audit_sleeves, audit_to_json
from v2.manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS


def main():
    parser = argparse.ArgumentParser(description="Audit V2 sleeve density across train/val/OOS splits")
    parser.add_argument("--bundle", required=True, choices=sorted(BUNDLE_MANIFESTS))
    parser.add_argument("--sleeve", action="append", default=[], help="Repeat to limit audit to specific sleeves.")
    parser.add_argument(
        "--feature-profile",
        default="price_context_plus",
        choices=["price_only", "price_context", "price_context_plus"],
    )
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--split", action="append", default=[], help="Repeat to override audited splits.")
    args = parser.parse_args()

    sleeves = args.sleeve or sorted(SLEEVE_MANIFESTS)
    splits = tuple(args.split) if args.split else ("train", "val", "2026q1")
    rows = audit_sleeves(
        bundle_name=args.bundle,
        sleeves=sleeves,
        feature_profile=args.feature_profile,
        splits=splits,
        max_symbols=args.max_symbols,
    )
    print(audit_to_json(rows))


if __name__ == "__main__":
    main()
