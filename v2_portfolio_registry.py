"""
Portfolio champion registry CLI.

Usage:
  uv run v2_portfolio_registry.py list
  uv run v2_portfolio_registry.py show <name>
  uv run v2_portfolio_registry.py promote <name> --status live-champion --notes "reason"
  uv run v2_portfolio_registry.py retire <name> --notes "reason"
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

_REGISTRY_PATH = "v2_portfolio_registry.json"


def _load(path: str = _REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _save(registry: dict, path: str = _REGISTRY_PATH) -> None:
    Path(path).write_text(json.dumps(registry, indent=2))


def _find(registry: dict, name: str) -> dict | None:
    return next((c for c in registry["candidates"] if c["name"] == name), None)


def cmd_list(args) -> None:
    registry = _load(args.registry)
    STATUS_ORDER = {"paper-live-candidate": 0, "paper-live": 1, "candidate": 2, "live-champion": 3, "retired": 4}
    candidates = sorted(registry["candidates"], key=lambda c: STATUS_ORDER.get(c["status"], 9))
    print(f"{'Name':<30} {'Status':<14} {'Model Set':<22} {'OOS Sharpe':>10} {'OOS PF':>8} {'Gate':>8}")
    print("-" * 100)
    for c in candidates:
        if args.status and c["status"] != args.status:
            continue
        sharpe = c.get("oos_daily_sharpe") or c.get("val_daily_sharpe") or 0.0
        pf = c.get("oos_pf") or c.get("val_pf") or 0.0
        gate = c.get("promotion_gate_status", "-")
        print(f"{c['name']:<30} {c['status']:<14} {c['model_set']:<22} {sharpe:>10.2f} {pf:>8.2f} {gate:>8}")


def cmd_show(args) -> None:
    registry = _load(args.registry)
    entry = _find(registry, args.name)
    if entry is None:
        print(f"No candidate named {args.name!r}")
        return
    print(json.dumps(entry, indent=2))


def cmd_promote(args) -> None:
    registry = _load(args.registry)
    entry = _find(registry, args.name)
    if entry is None:
        print(f"No candidate named {args.name!r}")
        return
    old_status = entry["status"]
    entry["status"] = args.status
    entry["last_updated"] = str(date.today())
    if args.notes:
        entry["notes"] = (entry.get("notes", "") + f" | {args.notes}").lstrip(" | ")
    _save(registry, args.registry)
    print(f"{args.name}: {old_status} → {args.status}")


def cmd_retire(args) -> None:
    registry = _load(args.registry)
    entry = _find(registry, args.name)
    if entry is None:
        print(f"No candidate named {args.name!r}")
        return
    entry["status"] = "retired"
    entry["last_updated"] = str(date.today())
    if args.notes:
        entry["notes"] = (entry.get("notes", "") + f" | {args.notes}").lstrip(" | ")
    _save(registry, args.registry)
    print(f"{args.name}: retired")


def cmd_add(args) -> None:
    registry = _load(args.registry)
    if _find(registry, args.name):
        print(f"Candidate {args.name!r} already exists — use promote/retire to change status")
        return
    entry = {
        "name": args.name,
        "status": "candidate",
        "config_path": args.config,
        "model_set": args.model_set,
        "bundle": args.bundle,
        "last_updated": str(date.today()),
        "promotion_gate_status": "PENDING",
        "notes": args.notes or "",
    }
    registry["candidates"].append(entry)
    _save(registry, args.registry)
    print(f"Added candidate {args.name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 portfolio champion registry")
    parser.add_argument("--registry", default=_REGISTRY_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all candidates")
    p_list.add_argument("--status", default=None, help="Filter by status")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show full entry")
    p_show.add_argument("name")
    p_show.set_defaults(func=cmd_show)

    p_promote = sub.add_parser("promote", help="Change candidate status")
    p_promote.add_argument("name")
    p_promote.add_argument("--status", required=True, choices=["candidate", "paper-live-candidate", "paper-live", "live-champion", "retired"])
    p_promote.add_argument("--notes", default="")
    p_promote.set_defaults(func=cmd_promote)

    p_retire = sub.add_parser("retire", help="Mark candidate as retired")
    p_retire.add_argument("name")
    p_retire.add_argument("--notes", default="")
    p_retire.set_defaults(func=cmd_retire)

    p_add = sub.add_parser("add", help="Register a new candidate")
    p_add.add_argument("name")
    p_add.add_argument("--config", required=True)
    p_add.add_argument("--model-set", required=True)
    p_add.add_argument("--bundle", default="bundle_intraday_core")
    p_add.add_argument("--notes", default="")
    p_add.set_defaults(func=cmd_add)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
