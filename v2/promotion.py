from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import prepare


@dataclass(frozen=True)
class PromotionThresholds:
    min_bar_sharpe: float = 3.5
    min_daily_sharpe: float = 3.0
    min_daily_sortino: float = 3.0
    min_profit_factor: float = 4.0
    min_win_rate_pct: float = 60.0
    min_trades_per_day: float = 5.0
    max_trades_per_day: float = 30.0
    max_drawdown_pct: float = 10.0
    max_concentration_pct: float = 30.0
    min_daily_observations: int = 20


def _finite_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if math.isnan(numeric):
        return float(default)
    return numeric


def _json_float(value: Any, default: float = 0.0) -> float:
    numeric = _finite_float(value, default)
    if math.isinf(numeric):
        return 1e12 if numeric > 0 else -1e12
    return float(numeric)


def metrics_from_backtest(backtest: dict) -> dict:
    result: prepare.BacktestResult = backtest["result"]
    top_share = max(
        [
            _finite_float(row.get("pnl_share", 0.0))
            for row in backtest.get("summary", {}).get("sleeve_concentration", []) or []
        ],
        default=_finite_float(backtest.get("top_share", 0.0)),
    )
    return {
        "split": str(backtest.get("split", "")),
        "score": _json_float(backtest.get("score", 0.0)),
        "bar_sharpe": _json_float(backtest.get("bar_sharpe", result.sharpe)),
        "daily_sharpe": _json_float(backtest.get("daily_sharpe", 0.0)),
        "daily_sortino": _json_float(backtest.get("daily_sortino", 0.0)),
        "daily_observations": int(backtest.get("daily_observations", 0) or 0),
        "total_return_pct": _json_float(result.total_return_pct),
        "max_drawdown_pct": _json_float(result.max_drawdown_pct),
        "num_trades": int(result.num_trades),
        "trades_per_day": _json_float(backtest.get("trades_per_day", 0.0)),
        "win_rate_pct": _json_float(result.win_rate_pct),
        "profit_factor": _json_float(result.profit_factor),
        "annual_turnover": _json_float(result.annual_turnover),
        "duration_days": _json_float(result.duration_days),
        "bars_processed": int(result.bars_processed),
        "total_bars": int(result.total_bars),
        "short_share": _json_float(backtest.get("short_share", 0.0)),
        "concentration_pct": _json_float(top_share * 100.0),
        "top_sleeves": [
            {
                "sleeve": str(row.get("sleeve", "")),
                "pnl_share": _json_float(row.get("pnl_share", 0.0)),
            }
            for row in (backtest.get("top_sleeves") or backtest.get("summary", {}).get("sleeve_concentration", [])[:3])
        ],
    }


def _check(
    checks: list[dict],
    *,
    name: str,
    scope: str,
    value: float | int | bool | None,
    target: float | int | bool | str,
    operator: str,
    passed: bool,
) -> None:
    checks.append(
        {
            "name": name,
            "scope": scope,
            "value": value,
            "target": target,
            "operator": operator,
            "passed": bool(passed),
        }
    )


def _evaluate_metric_set(
    checks: list[dict],
    *,
    scope: str,
    metrics: dict,
    thresholds: PromotionThresholds,
) -> None:
    bar_sharpe = _finite_float(metrics.get("bar_sharpe"))
    daily_sharpe = _finite_float(metrics.get("daily_sharpe"))
    daily_sortino = _finite_float(metrics.get("daily_sortino"))
    profit_factor = _finite_float(metrics.get("profit_factor"))
    win_rate_pct = _finite_float(metrics.get("win_rate_pct"))
    trades_per_day = _finite_float(metrics.get("trades_per_day"))
    max_drawdown_pct = _finite_float(metrics.get("max_drawdown_pct"))
    concentration_pct = _finite_float(metrics.get("concentration_pct"))
    daily_observations = int(metrics.get("daily_observations", 0) or 0)

    _check(
        checks,
        name="bar_sharpe",
        scope=scope,
        value=bar_sharpe,
        target=thresholds.min_bar_sharpe,
        operator=">=",
        passed=bar_sharpe >= thresholds.min_bar_sharpe,
    )
    _check(
        checks,
        name="daily_sharpe",
        scope=scope,
        value=daily_sharpe,
        target=thresholds.min_daily_sharpe,
        operator=">=",
        passed=daily_sharpe >= thresholds.min_daily_sharpe,
    )
    _check(
        checks,
        name="daily_sortino",
        scope=scope,
        value=daily_sortino,
        target=thresholds.min_daily_sortino,
        operator=">=",
        passed=daily_sortino >= thresholds.min_daily_sortino,
    )
    _check(
        checks,
        name="profit_factor",
        scope=scope,
        value=profit_factor,
        target=thresholds.min_profit_factor,
        operator=">=",
        passed=profit_factor >= thresholds.min_profit_factor,
    )
    _check(
        checks,
        name="win_rate_pct",
        scope=scope,
        value=win_rate_pct,
        target=thresholds.min_win_rate_pct,
        operator=">=",
        passed=win_rate_pct >= thresholds.min_win_rate_pct,
    )
    _check(
        checks,
        name="trades_per_day_min",
        scope=scope,
        value=trades_per_day,
        target=thresholds.min_trades_per_day,
        operator=">=",
        passed=trades_per_day >= thresholds.min_trades_per_day,
    )
    _check(
        checks,
        name="trades_per_day_max",
        scope=scope,
        value=trades_per_day,
        target=thresholds.max_trades_per_day,
        operator="<=",
        passed=trades_per_day <= thresholds.max_trades_per_day,
    )
    _check(
        checks,
        name="max_drawdown_pct",
        scope=scope,
        value=max_drawdown_pct,
        target=thresholds.max_drawdown_pct,
        operator="<",
        passed=max_drawdown_pct < thresholds.max_drawdown_pct,
    )
    _check(
        checks,
        name="concentration_pct",
        scope=scope,
        value=concentration_pct,
        target=thresholds.max_concentration_pct,
        operator="<=",
        passed=concentration_pct <= thresholds.max_concentration_pct,
    )
    _check(
        checks,
        name="daily_observations",
        scope=scope,
        value=daily_observations,
        target=thresholds.min_daily_observations,
        operator=">=",
        passed=daily_observations >= thresholds.min_daily_observations,
    )


def build_promotion_report(
    *,
    bundle: str,
    model_set: str,
    portfolio_config: str,
    active_sleeves: list[str],
    base_runs: dict[str, dict],
    stress_runs: dict[str, dict] | None,
    thresholds: PromotionThresholds | None = None,
    fresh_oos_label: str | None = None,
    oos_split: str = "2026q1",
    holdout_used: bool = False,
) -> dict:
    thresholds = thresholds or PromotionThresholds()
    base_metrics = {split: metrics_from_backtest(run) for split, run in base_runs.items()}
    stress_metrics = {split: metrics_from_backtest(run) for split, run in (stress_runs or {}).items()}
    checks: list[dict] = []

    for split, metrics in base_metrics.items():
        _evaluate_metric_set(
            checks,
            scope=f"base:{split}",
            metrics=metrics,
            thresholds=thresholds,
        )
    for split, metrics in stress_metrics.items():
        _evaluate_metric_set(
            checks,
            scope=f"stress:{split}",
            metrics=metrics,
            thresholds=thresholds,
        )

    _check(
        checks,
        name="stress_replay_present",
        scope="global",
        value=bool(stress_metrics),
        target=True,
        operator="==",
        passed=bool(stress_metrics),
    )
    _check(
        checks,
        name="fresh_oos",
        scope="global",
        value=bool(fresh_oos_label),
        target="non-empty fresh OOS label",
        operator="is",
        passed=bool(fresh_oos_label),
    )
    _check(
        checks,
        name="holdout_untouched",
        scope="global",
        value=not holdout_used,
        target=True,
        operator="==",
        passed=not holdout_used,
    )

    failures = [
        f"{check['scope']}:{check['name']} {check['operator']} {check['target']} "
        f"(value={check['value']})"
        for check in checks
        if not check["passed"]
    ]
    return {
        "passed": not failures,
        "status": "PASS" if not failures else "FAIL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle": bundle,
        "model_set": model_set,
        "portfolio_config": portfolio_config,
        "active_sleeves": list(active_sleeves),
        "oos_split": oos_split,
        "fresh_oos_label": fresh_oos_label or "",
        "thresholds": asdict(thresholds),
        "base": base_metrics,
        "stress": stress_metrics,
        "checks": checks,
        "failures": failures,
        "notes": (
            "Production promotion uses corrected replay metrics, not the legacy score column. "
            "A missing fresh_oos_label is an intentional failure until a clean replacement OOS "
            "or forward paper period is designated."
        ),
    }
