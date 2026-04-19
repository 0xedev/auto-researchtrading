from __future__ import annotations

import copy
import json
from pathlib import Path


SHADOW_TIMEFRAME_CHOICES = ("15m", "1h", "4h")
SHADOW_SPLIT_CHOICES = (
    "val",
    "oos",
    "2026q1",
    "robustness",
    "holdout",
    "val_15m",
    "oos_15m",
    "2026q1_15m",
    "holdout_15m",
)

DEFAULT_SHADOW_RUNTIME_CONFIG = {
    "timeframe": "1h",
    "split": "2026q1",
    "max_days": None,
    "risk": {
        "max_leverage": 3.0,
        "max_symbol_notional_pct": 0.35,
    },
    "paths": {
        "state_path": "shadow_state.json",
        "log_path": "shadow_trade_log.jsonl",
        "dashboard_path": "shadow_dashboard.md",
        "summary_json_path": "shadow_summary.json",
        "kill_switch_path": None,
    },
    "operator": {
        "recent_actions": 12,
    },
}


def _deep_merge(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_shadow_runtime_config(path: str | Path) -> dict:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Shadow runtime config must be a JSON object")
    return payload


def _set_nested(config: dict, dotted_path: tuple[str, ...], value) -> None:
    cursor = config
    for key in dotted_path[:-1]:
        cursor = cursor[key]
    cursor[dotted_path[-1]] = value


def _validate_shadow_runtime_config(config: dict) -> None:
    timeframe = config.get("timeframe")
    if timeframe not in SHADOW_TIMEFRAME_CHOICES:
        raise ValueError(f"Unsupported shadow timeframe: {timeframe!r}")

    split = config.get("split")
    if split not in SHADOW_SPLIT_CHOICES:
        raise ValueError(f"Unsupported shadow split: {split!r}")

    max_days = config.get("max_days")
    if max_days is not None and int(max_days) <= 0:
        raise ValueError("max_days must be positive when provided")

    risk = config.get("risk", {})
    if float(risk.get("max_leverage", 0.0) or 0.0) <= 0:
        raise ValueError("risk.max_leverage must be positive")
    symbol_cap = float(risk.get("max_symbol_notional_pct", 0.0) or 0.0)
    if symbol_cap <= 0 or symbol_cap > 1:
        raise ValueError("risk.max_symbol_notional_pct must be between 0 and 1")

    operator = config.get("operator", {})
    if int(operator.get("recent_actions", 0) or 0) <= 0:
        raise ValueError("operator.recent_actions must be positive")


def resolve_shadow_runtime_config(
    config_path: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    config = copy.deepcopy(DEFAULT_SHADOW_RUNTIME_CONFIG)
    config_source = "defaults"

    if config_path:
        loaded = load_shadow_runtime_config(config_path)
        _deep_merge(config, loaded)
        config_source = str(Path(config_path))

    override_map = {
        "timeframe": ("timeframe",),
        "split": ("split",),
        "max_days": ("max_days",),
        "max_leverage": ("risk", "max_leverage"),
        "max_symbol_notional_pct": ("risk", "max_symbol_notional_pct"),
        "state_path": ("paths", "state_path"),
        "log_path": ("paths", "log_path"),
        "dashboard_path": ("paths", "dashboard_path"),
        "summary_json_path": ("paths", "summary_json_path"),
        "kill_switch_path": ("paths", "kill_switch_path"),
        "recent_actions": ("operator", "recent_actions"),
    }

    for key, value in (overrides or {}).items():
        if value is None or key not in override_map:
            continue
        _set_nested(config, override_map[key], value)

    config["config_source"] = config_source
    _validate_shadow_runtime_config(config)
    return config
