from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import SleeveRegistry, SleeveRegistryEntry


DEFAULT_REGISTRY_PATH = Path("v2_sleeve_registry.json")


def load_sleeve_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> SleeveRegistry:
    target = Path(path)
    if not target.exists():
        return SleeveRegistry()
    payload = json.loads(target.read_text())
    entries = [SleeveRegistryEntry(**row) for row in payload.get("entries", [])]
    return SleeveRegistry(entries=entries)


def save_sleeve_registry(registry: SleeveRegistry, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"entries": [asdict(entry) for entry in registry.entries]}, indent=2, sort_keys=True))


def upsert_registry_entry(
    registry: SleeveRegistry,
    entry: SleeveRegistryEntry,
) -> SleeveRegistry:
    kept = [
        row
        for row in registry.entries
        if not (
            row.sleeve == entry.sleeve
            and row.bundle == entry.bundle
            and row.model_set == entry.model_set
        )
    ]
    kept.append(entry)
    registry.entries = sorted(kept, key=lambda row: (row.bundle, row.sleeve, row.status, row.model_set))
    return registry


def find_registry_entries(
    registry: SleeveRegistry,
    *,
    bundle: str | None = None,
    sleeve: str | None = None,
    status: str | None = None,
) -> list[SleeveRegistryEntry]:
    rows = registry.entries
    if bundle is not None:
        rows = [row for row in rows if row.bundle == bundle]
    if sleeve is not None:
        rows = [row for row in rows if row.sleeve == sleeve]
    if status is not None:
        rows = [row for row in rows if row.status == status]
    return rows


def find_live_champion(
    registry: SleeveRegistry,
    *,
    bundle: str,
    sleeve: str,
) -> SleeveRegistryEntry | None:
    champions = find_registry_entries(registry, bundle=bundle, sleeve=sleeve, status="live-champion")
    if not champions:
        return None
    champions = sorted(champions, key=lambda row: (row.validation_metric, row.oos_metric, row.model_set), reverse=True)
    return champions[0]
