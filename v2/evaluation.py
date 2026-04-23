from __future__ import annotations

import copy
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import prepare
from execution.report import summarize_shadow_run
from execution.v2_paper import run_v2_shadow_session

from .data import build_bundle_dataset
from .features import timeframe_to_hours
from .manifests import BUNDLE_MANIFESTS
from .runtime import V2SignalEngine
from .types import PortfolioConfig


def _read_jsonl(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _days_processed(bundle_name: str, bars_processed: int) -> float:
    base_hours = timeframe_to_hours(BUNDLE_MANIFESTS[bundle_name].base_tf)
    return max((bars_processed * base_hours) / 24.0, 1e-9)


def _summarize_trial(bundle_name: str, result: dict, summary: dict, log_rows: list[dict]) -> dict:
    action_counts = summary.get("log_stats", {}).get("action_counts", {})
    open_rows = [row for row in log_rows if row.get("action") == "open"]
    close_rows = [row for row in log_rows if row.get("action") == "close"]
    days = _days_processed(bundle_name, int(result.get("bars_processed", 0) or 0))
    top_sleeve = None
    top_share = 0.0
    for row in summary.get("sleeve_concentration", []):
        share = float(row.get("pnl_share", 0.0) or 0.0)
        if share > top_share:
            top_share = share
            top_sleeve = row.get("sleeve", "")

    short_opens = sum(1 for row in open_rows if float(row.get("target_position", 0.0) or 0.0) < 0)
    short_share = (short_opens / len(open_rows)) if open_rows else 0.0
    critical_alerts = [
        alert for alert in summary.get("alerts", [])
        if str(alert.get("severity", "")).lower() == "critical"
    ]
    rejection_count = int(action_counts.get("rejected", 0) or 0)
    rejection_rate = rejection_count / max(len(log_rows), 1)
    return_pct = (float(summary.get("portfolio", {}).get("equity", prepare.INITIAL_CAPITAL)) / prepare.INITIAL_CAPITAL) - 1.0
    concentration_penalty = max(0.0, top_share - 0.30)
    rejection_penalty = max(0.0, rejection_rate - 0.10)
    critical_penalty = 0.10 * len(critical_alerts)
    promotion_score = return_pct - concentration_penalty - rejection_penalty - critical_penalty

    close_pnls = [float(row.get("realized_pnl", 0.0) or 0.0) for row in close_rows]
    gross_profit = sum(max(pnl, 0.0) for pnl in close_pnls)
    gross_loss = sum(-min(pnl, 0.0) for pnl in close_pnls)
    winning_closes = sum(1 for pnl in close_pnls if pnl > 0)
    losing_closes = sum(1 for pnl in close_pnls if pnl < 0)
    flat_closes = sum(1 for pnl in close_pnls if pnl == 0)
    win_rate = winning_closes / len(close_pnls) if close_pnls else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    equity_points = [prepare.INITIAL_CAPITAL]
    for row in log_rows:
        if "portfolio_equity" in row and row.get("portfolio_equity") is not None:
            equity_points.append(float(row.get("portfolio_equity", prepare.INITIAL_CAPITAL) or prepare.INITIAL_CAPITAL))
    equity_points.append(float(summary.get("portfolio", {}).get("equity", prepare.INITIAL_CAPITAL) or prepare.INITIAL_CAPITAL))
    peak_equity = equity_points[0]
    action_path_max_drawdown_pct = 0.0
    for equity in equity_points[1:]:
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            action_path_max_drawdown_pct = max(action_path_max_drawdown_pct, (peak_equity - equity) / peak_equity)

    return {
        "score": float(promotion_score),
        "score_kind": "promotion_score",
        "promotion_score": float(promotion_score),
        "return_pct": float(return_pct),
        "raw_return_pct": float(return_pct),
        "days_processed": float(days),
        "bars_processed": int(result.get("bars_processed", 0) or 0),
        "opens": int(action_counts.get("open", 0) or 0),
        "closes": int(action_counts.get("close", 0) or 0),
        "modifies": int(action_counts.get("modify", 0) or 0),
        "rejections": rejection_count,
        "records": int(summary.get("log_stats", {}).get("records", 0) or 0),
        "trades_per_day": float((action_counts.get("open", 0) or 0) / days),
        "realized_pnl_total": float(summary.get("log_stats", {}).get("realized_pnl_total", 0.0) or 0.0),
        "fees_total": float(summary.get("log_stats", {}).get("fees_total", 0.0) or 0.0),
        "winning_closes": int(winning_closes),
        "losing_closes": int(losing_closes),
        "flat_closes": int(flat_closes),
        "win_rate": float(win_rate),
        "profit_factor": None if profit_factor is None else float(profit_factor),
        "action_path_max_drawdown_pct": float(action_path_max_drawdown_pct),
        "gross_leverage": float(summary.get("portfolio", {}).get("gross_leverage", 0.0) or 0.0),
        "top_sleeve": top_sleeve or "",
        "top_sleeve_share": float(top_share),
        "concentration_penalty": float(concentration_penalty),
        "rejection_penalty": float(rejection_penalty),
        "critical_penalty": float(critical_penalty),
        "short_share": float(short_share),
        "critical_alerts": len(critical_alerts),
        "shadow_ready": len(critical_alerts) == 0,
        "alerts": summary.get("alerts", []),
        "action_counts": action_counts,
        "signal_counts": summary.get("log_stats", {}).get("signal_counts", {}),
        "sleeve_counts": summary.get("log_stats", {}).get("sleeve_counts", {}),
        "bundle_counts": summary.get("log_stats", {}).get("bundle_counts", {}),
        "cluster_counts": summary.get("log_stats", {}).get("cluster_counts", {}),
        "decision_reason_counts": summary.get("log_stats", {}).get("decision_reason_counts", {}),
        "sleeve_pnl": summary.get("sleeve_pnl", []),
        "sleeve_concentration": summary.get("sleeve_concentration", []),
    }


def _top_concentration_rows(*metric_sets: dict, limit: int = 3) -> list[dict]:
    by_sleeve: dict[str, float] = {}
    for metrics in metric_sets:
        for row in metrics.get("sleeve_concentration", []) or []:
            sleeve = str(row.get("sleeve", "") or "")
            share = float(row.get("pnl_share", 0.0) or 0.0)
            if not sleeve:
                continue
            by_sleeve[sleeve] = max(by_sleeve.get(sleeve, 0.0), share)
    ranked = sorted(by_sleeve.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [{"sleeve": sleeve, "pnl_share": float(share)} for sleeve, share in ranked]


def _build_portfolio_summary(evaluation: dict) -> dict:
    val_metrics = evaluation.get("base", {}).get("val", {}) or {}
    oos_metrics = evaluation.get("base", {}).get("2026q1", {}) or {}
    stress_payload = evaluation.get("stress", {}) if isinstance(evaluation.get("stress"), dict) else {}
    stress_val = stress_payload.get("val", {}) or {}
    stress_oos = stress_payload.get("2026q1", {}) or {}
    has_stress = bool(stress_payload)

    val_return_metric = float(val_metrics.get("raw_return_pct", val_metrics.get("return_pct", 0.0)) or 0.0)
    oos_return_metric = float(oos_metrics.get("raw_return_pct", oos_metrics.get("return_pct", 0.0)) or 0.0)
    val_promotion_metric = float(val_metrics.get("promotion_score", val_metrics.get("score", 0.0)) or 0.0)
    oos_promotion_metric = float(oos_metrics.get("promotion_score", oos_metrics.get("score", 0.0)) or 0.0)
    if has_stress:
        stress_return_metric = min(
            float(stress_val.get("raw_return_pct", stress_val.get("return_pct", 0.0)) or 0.0),
            float(stress_oos.get("raw_return_pct", stress_oos.get("return_pct", 0.0)) or 0.0),
        )
        stress_promotion_metric = min(
            float(stress_val.get("promotion_score", stress_val.get("score", 0.0)) or 0.0),
            float(stress_oos.get("promotion_score", stress_oos.get("score", 0.0)) or 0.0),
        )
    else:
        stress_return_metric = float(min(val_return_metric, oos_return_metric))
        stress_promotion_metric = 0.0
    top_share = max(
        float(val_metrics.get("top_sleeve_share", 0.0) or 0.0),
        float(oos_metrics.get("top_sleeve_share", 0.0) or 0.0),
    )
    concentration_ready = bool(top_share <= 0.30)
    shadow_ready = bool(val_metrics.get("shadow_ready", False) and oos_metrics.get("shadow_ready", False))
    trades_per_day = max(
        float(val_metrics.get("trades_per_day", 0.0) or 0.0),
        float(oos_metrics.get("trades_per_day", 0.0) or 0.0),
    )
    short_share = max(
        float(val_metrics.get("short_share", 0.0) or 0.0),
        float(oos_metrics.get("short_share", 0.0) or 0.0),
    )
    rolling_base = evaluation.get("rolling", {}).get("base", {}) if isinstance(evaluation.get("rolling"), dict) else {}
    rolling_summaries = {
        split: payload.get("summary", {})
        for split, payload in rolling_base.items()
        if isinstance(payload, dict)
    }
    rolling_positive_rate = min(
        [float(summary.get("positive_window_rate", 0.0) or 0.0) for summary in rolling_summaries.values()],
        default=0.0,
    )
    rolling_mean_return = min(
        [float(summary.get("mean_raw_return_pct", 0.0) or 0.0) for summary in rolling_summaries.values()],
        default=0.0,
    )
    rolling_max_concentration = max(
        [float(summary.get("max_top_sleeve_share", 0.0) or 0.0) for summary in rolling_summaries.values()],
        default=0.0,
    )
    meets_gate = (
        val_return_metric >= 0.0
        and oos_return_metric >= 0.0
        and (stress_return_metric >= 0.0 if has_stress else True)
        and concentration_ready
        and shadow_ready
        and trades_per_day > 0.0
    )
    return {
        "meets_gate": bool(meets_gate),
        "validation_metric": float(val_return_metric),
        "oos_metric": float(oos_return_metric),
        "stress_metric": float(stress_return_metric),
        "validation_promotion_score": float(val_promotion_metric),
        "oos_promotion_score": float(oos_promotion_metric),
        "stress_promotion_score": float(stress_promotion_metric),
        "concentration_pct": float(top_share),
        "concentration_ready": bool(concentration_ready),
        "trades_per_day": float(trades_per_day),
        "short_share": float(short_share),
        "shadow_ready": bool(shadow_ready),
        "has_stress": bool(has_stress),
        "rolling_positive_window_rate": float(rolling_positive_rate),
        "rolling_mean_return_pct": float(rolling_mean_return),
        "rolling_max_concentration_pct": float(rolling_max_concentration),
        "top_sleeves": _top_concentration_rows(val_metrics, oos_metrics),
        "notes": (
            f"val_return={val_return_metric:.4f} "
            f"oos_return={oos_return_metric:.4f} "
            f"stress_return={stress_return_metric:.4f} "
            f"rolling_pos_rate={rolling_positive_rate:.2f} "
            f"top_sleeve={val_metrics.get('top_sleeve', '') or oos_metrics.get('top_sleeve', '')}"
        ).strip(),
    }


def _split_timestamps(
    *,
    bundle_name: str,
    split: str,
    max_symbols: int | None = None,
) -> list[int]:
    frame = build_bundle_dataset(
        bundle_name,
        split=split,
        feature_profile="price_context_plus",
        max_symbols=max_symbols,
    )
    if frame.empty or "timestamp" not in frame.columns:
        return []
    return sorted(frame["timestamp"].dropna().astype("int64").unique().tolist())


def _window_bounds_from_timestamps(
    timestamps: list[int],
    *,
    window_days: int,
    step_days: int | None = None,
    max_windows: int | None = None,
) -> list[tuple[int, int]]:
    if not timestamps or window_days <= 0:
        return []
    unit_scale = 1000 if abs(int(timestamps[0])) > 1e11 else 1
    window_span = int(window_days * 24 * 3600 * unit_scale)
    step_span = int((step_days or window_days) * 24 * 3600 * unit_scale)
    if step_span <= 0:
        step_span = window_span
    last_timestamp = int(timestamps[-1])
    windows: list[tuple[int, int]] = []
    start = int(timestamps[0])
    while start <= last_timestamp:
        end = start + window_span
        active = [ts for ts in timestamps if start <= int(ts) <= end]
        if len(active) >= 2:
            windows.append((int(active[0]), int(active[-1])))
        start += step_span
    if max_windows is not None and len(windows) > max_windows:
        if max_windows <= 1:
            return [windows[-1]]
        last_index = len(windows) - 1
        chosen = sorted({round(i * last_index / (max_windows - 1)) for i in range(max_windows)})
        windows = [windows[index] for index in chosen]
    return windows


def _rolling_window_bounds(
    *,
    bundle_name: str,
    split: str,
    window_days: int,
    step_days: int | None = None,
    max_windows: int | None = None,
    max_symbols: int | None = None,
) -> list[tuple[int, int]]:
    timestamps = _split_timestamps(bundle_name=bundle_name, split=split, max_symbols=max_symbols)
    return _window_bounds_from_timestamps(
        timestamps,
        window_days=window_days,
        step_days=step_days,
        max_windows=max_windows,
    )


def _summarize_rolling_windows(
    windows: list[dict],
    *,
    window_days: int,
    step_days: int | None = None,
) -> dict:
    if not windows:
        return {
            "window_days": int(window_days),
            "step_days": int(step_days or window_days),
            "window_count": 0,
            "positive_window_rate": 0.0,
            "nonnegative_window_rate": 0.0,
            "mean_raw_return_pct": 0.0,
            "min_raw_return_pct": 0.0,
            "mean_trades_per_day": 0.0,
            "max_top_sleeve_share": 0.0,
            "shadow_ready_rate": 0.0,
            "top_sleeves": [],
        }

    raw_returns = [float(row.get("raw_return_pct", row.get("return_pct", 0.0)) or 0.0) for row in windows]
    trades_per_day = [float(row.get("trades_per_day", 0.0) or 0.0) for row in windows]
    top_shares = [float(row.get("top_sleeve_share", 0.0) or 0.0) for row in windows]
    shadow_ready = [1.0 if row.get("shadow_ready", False) else 0.0 for row in windows]
    top_counter = Counter(str(row.get("top_sleeve", "") or "") for row in windows if row.get("top_sleeve"))

    return {
        "window_days": int(window_days),
        "step_days": int(step_days or window_days),
        "window_count": len(windows),
        "positive_window_rate": float(sum(1 for value in raw_returns if value > 0.0) / len(windows)),
        "nonnegative_window_rate": float(sum(1 for value in raw_returns if value >= 0.0) / len(windows)),
        "mean_raw_return_pct": float(sum(raw_returns) / len(windows)),
        "min_raw_return_pct": float(min(raw_returns)),
        "mean_trades_per_day": float(sum(trades_per_day) / len(windows)),
        "max_top_sleeve_share": float(max(top_shares)),
        "shadow_ready_rate": float(sum(shadow_ready) / len(windows)),
        "top_sleeves": [
            {"sleeve": sleeve, "windows": int(count), "share": float(count / len(windows))}
            for sleeve, count in top_counter.most_common(3)
        ],
    }


def run_shadow_trial(
    *,
    bundle_name: str,
    model_set: str,
    split: str,
    portfolio_config: PortfolioConfig,
    active_sleeves: list[str] | None = None,
    max_days: int | None = None,
    max_symbols: int | None = None,
    start_timestamp: int | None = None,
    end_timestamp: int | None = None,
    engine: V2SignalEngine | None = None,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="v2_eval_") as tmpdir:
        root = Path(tmpdir)
        state_path = root / "state.json"
        log_path = root / "log.jsonl"
        result = run_v2_shadow_session(
            bundle_name=bundle_name,
            model_set=model_set,
            split=split,
            portfolio_config=portfolio_config,
            state_path=str(state_path),
            log_path=str(log_path),
            kill_switch_path=None,
            max_days=max_days,
            active_sleeves=active_sleeves,
            max_symbols=max_symbols,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            engine=engine,
        )
        summary = summarize_shadow_run(state_path=state_path, log_path=log_path, max_recent=12)
        log_rows = _read_jsonl(log_path)
    metrics = _summarize_trial(bundle_name, result, summary, log_rows)
    if start_timestamp is not None:
        metrics["window_start"] = int(start_timestamp)
    if end_timestamp is not None:
        metrics["window_end"] = int(end_timestamp)
    return metrics


def evaluate_model_set(
    *,
    bundle_name: str,
    model_set: str,
    portfolio_config: PortfolioConfig,
    active_sleeves: list[str] | None = None,
    splits: tuple[str, ...] = ("val", "2026q1"),
    max_days: int | None = None,
    max_symbols: int | None = None,
    rolling_window_days: int | None = None,
    rolling_step_days: int | None = None,
    rolling_max_windows: int | None = None,
    include_stress: bool = True,
) -> dict:
    engine_cache: dict[str, V2SignalEngine] = {}

    def get_engine(split: str) -> V2SignalEngine:
        if split not in engine_cache:
            engine = V2SignalEngine(
                bundle_name=bundle_name,
                model_set=model_set,
                active_sleeves=active_sleeves,
                max_symbols=max_symbols,
            )
            engine.prepare(split)
            engine_cache[split] = engine
        return engine_cache[split]

    base = {}
    for split in splits:
        base[split] = run_shadow_trial(
            bundle_name=bundle_name,
            model_set=model_set,
            split=split,
            portfolio_config=portfolio_config,
            active_sleeves=active_sleeves,
            max_days=max_days,
            max_symbols=max_symbols,
            engine=get_engine(split),
        )

    stress_config = copy.deepcopy(portfolio_config)
    stress_config.slippage_bps = max(portfolio_config.slippage_bps + 3.0, portfolio_config.slippage_bps * 2.0)
    stress_config.max_participation_rate = max(portfolio_config.max_participation_rate * 0.5, 0.005)

    stressed = {}
    if include_stress:
        for split in splits:
            stressed[split] = run_shadow_trial(
                bundle_name=bundle_name,
                model_set=model_set,
                split=split,
                portfolio_config=stress_config,
                active_sleeves=active_sleeves,
                max_days=max_days,
                max_symbols=max_symbols,
                engine=get_engine(split),
            )

    evaluation = {
        "bundle": bundle_name,
        "model_set": model_set,
        "active_sleeves": active_sleeves or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "stress": stressed,
    }
    if rolling_window_days:
        rolling = {"base": {}, "stress": {}}
        for split in splits:
            bounds = _rolling_window_bounds(
                bundle_name=bundle_name,
                split=split,
                window_days=rolling_window_days,
                step_days=rolling_step_days,
                max_windows=rolling_max_windows,
                max_symbols=max_symbols,
            )
            base_windows = [
                run_shadow_trial(
                    bundle_name=bundle_name,
                    model_set=model_set,
                    split=split,
                    portfolio_config=portfolio_config,
                    active_sleeves=active_sleeves,
                    max_symbols=max_symbols,
                    start_timestamp=start,
                    end_timestamp=end,
                    engine=get_engine(split),
                )
                for start, end in bounds
            ]
            stress_windows = []
            if include_stress:
                stress_windows = [
                    run_shadow_trial(
                        bundle_name=bundle_name,
                        model_set=model_set,
                        split=split,
                        portfolio_config=stress_config,
                        active_sleeves=active_sleeves,
                        max_symbols=max_symbols,
                        start_timestamp=start,
                        end_timestamp=end,
                        engine=get_engine(split),
                    )
                    for start, end in bounds
                ]
            rolling["base"][split] = {
                "summary": _summarize_rolling_windows(
                    base_windows,
                    window_days=rolling_window_days,
                    step_days=rolling_step_days,
                ),
                "windows": base_windows,
            }
            if include_stress:
                rolling["stress"][split] = {
                    "summary": _summarize_rolling_windows(
                        stress_windows,
                        window_days=rolling_window_days,
                        step_days=rolling_step_days,
                    ),
                    "windows": stress_windows,
                }
        evaluation["rolling"] = rolling
    evaluation["summary"] = _build_portfolio_summary(evaluation)
    return evaluation


def evaluate_candidate(
    *,
    bundle_name: str,
    sleeve_name: str,
    model_set: str,
    portfolio_config: PortfolioConfig,
    max_days: int | None = None,
    max_symbols: int | None = None,
    rolling_window_days: int | None = None,
    rolling_step_days: int | None = None,
    rolling_max_windows: int | None = None,
    include_stress: bool = True,
) -> dict:
    evaluation = evaluate_model_set(
        bundle_name=bundle_name,
        model_set=model_set,
        portfolio_config=portfolio_config,
        active_sleeves=[sleeve_name],
        max_days=max_days,
        max_symbols=max_symbols,
        rolling_window_days=rolling_window_days,
        rolling_step_days=rolling_step_days,
        rolling_max_windows=rolling_max_windows,
        include_stress=include_stress,
    )

    val_metrics = evaluation["base"].get("val", {})
    oos_metrics = evaluation["base"].get("2026q1", {})
    stress_payload = evaluation.get("stress", {}) if isinstance(evaluation.get("stress"), dict) else {}
    stress_val = stress_payload.get("val", {})
    stress_oos = stress_payload.get("2026q1", {})
    has_stress = bool(stress_payload)
    top_share = max(
        float(val_metrics.get("top_sleeve_share", 0.0) or 0.0),
        float(oos_metrics.get("top_sleeve_share", 0.0) or 0.0),
    )
    val_return_metric = float(val_metrics.get("raw_return_pct", val_metrics.get("return_pct", 0.0)) or 0.0)
    oos_return_metric = float(oos_metrics.get("raw_return_pct", oos_metrics.get("return_pct", 0.0)) or 0.0)
    val_promotion_metric = float(val_metrics.get("promotion_score", val_metrics.get("score", 0.0)) or 0.0)
    oos_promotion_metric = float(oos_metrics.get("promotion_score", oos_metrics.get("score", 0.0)) or 0.0)
    if has_stress:
        stress_return_metric = min(
            float(stress_val.get("raw_return_pct", stress_val.get("return_pct", 0.0)) or 0.0),
            float(stress_oos.get("raw_return_pct", stress_oos.get("return_pct", 0.0)) or 0.0),
        )
        stress_promotion_metric = min(
            float(stress_val.get("promotion_score", stress_val.get("score", 0.0)) or 0.0),
            float(stress_oos.get("promotion_score", stress_oos.get("score", 0.0)) or 0.0),
        )
    else:
        stress_return_metric = float(min(val_return_metric, oos_return_metric))
        stress_promotion_metric = 0.0
    shadow_ready = bool(val_metrics.get("shadow_ready", False) and oos_metrics.get("shadow_ready", False))
    trades_per_day = max(
        float(val_metrics.get("trades_per_day", 0.0) or 0.0),
        float(oos_metrics.get("trades_per_day", 0.0) or 0.0),
    )
    short_share = max(
        float(val_metrics.get("short_share", 0.0) or 0.0),
        float(oos_metrics.get("short_share", 0.0) or 0.0),
    )
    rolling_base = evaluation.get("rolling", {}).get("base", {}) if isinstance(evaluation.get("rolling"), dict) else {}
    rolling_summaries = {
        split: payload.get("summary", {})
        for split, payload in rolling_base.items()
        if isinstance(payload, dict)
    }
    rolling_positive_rate = min(
        [float(summary.get("positive_window_rate", 0.0) or 0.0) for summary in rolling_summaries.values()],
        default=0.0,
    )
    rolling_mean_return = min(
        [float(summary.get("mean_raw_return_pct", 0.0) or 0.0) for summary in rolling_summaries.values()],
        default=0.0,
    )
    meets_gate = (
        val_return_metric >= 0.0
        and oos_return_metric >= 0.0
        and shadow_ready
        and trades_per_day > 0.0
    )
    return {
        **evaluation,
        "sleeve": sleeve_name,
        "summary": {
            "meets_gate": bool(meets_gate),
            "validation_metric": float(val_return_metric),
            "oos_metric": float(oos_return_metric),
            "stress_metric": float(stress_return_metric),
            "validation_promotion_score": float(val_promotion_metric),
            "oos_promotion_score": float(oos_promotion_metric),
            "stress_promotion_score": float(stress_promotion_metric),
            "concentration_pct": float(top_share),
            "concentration_ready": bool(top_share <= 0.30),
            "trades_per_day": float(trades_per_day),
            "short_share": float(short_share),
            "shadow_ready": bool(shadow_ready),
            "has_stress": bool(has_stress),
            "rolling_positive_window_rate": float(rolling_positive_rate),
            "rolling_mean_return_pct": float(rolling_mean_return),
            "notes": (
                f"val_return={val_metrics.get('return_pct', 0.0):.4f} "
                f"oos_return={oos_metrics.get('return_pct', 0.0):.4f} "
                f"val_promo={val_promotion_metric:.4f} "
                f"oos_promo={oos_promotion_metric:.4f} "
                f"stress_return={stress_return_metric:.4f} "
                f"rolling_pos_rate={rolling_positive_rate:.2f} "
                f"top_sleeve={val_metrics.get('top_sleeve', '') or oos_metrics.get('top_sleeve', '')}"
            ).strip(),
        },
    }
