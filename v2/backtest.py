from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import prepare
from execution.report import summarize_shadow_run
from execution.v2_paper import run_v2_shadow_session

from .evaluation import _read_jsonl
from .features import timeframe_to_hours
from .manifests import BUNDLE_MANIFESTS
from .runtime import V2SignalEngine
from .types import PortfolioConfig


BENCHMARKS = {
    "daily_sharpe": 3.0,
    "win_rate_pct": 60.0,
    "trades_per_day": 1.0,
    "profit_factor": 4.0,
    "max_drawdown_pct": 10.0,
}


def classify_v2_run_status(score: float) -> str:
    if score >= 3.0:
        return "KEEP"
    if score > 0.0:
        return "CANDIDATE"
    return "REVERT"


def _bars_per_year(bundle_name: str) -> float:
    base_hours = timeframe_to_hours(BUNDLE_MANIFESTS[bundle_name].base_tf)
    return 365.25 * 24.0 / max(base_hours, 1e-9)


def _series_from_curve(equity_curve: list[float], timestamps: list[int]) -> pd.Series:
    if not equity_curve or not timestamps or len(equity_curve) != len(timestamps):
        return pd.Series(dtype=float)
    return pd.Series([float(v) for v in equity_curve], index=pd.Index([int(ts) for ts in timestamps], name="timestamp"))


def _timestamp_index_to_datetime(index: pd.Index) -> pd.DatetimeIndex:
    if index.empty:
        return pd.DatetimeIndex([], tz="UTC")
    values = [int(ts) for ts in index.tolist()]
    unit = "ms" if max(abs(ts) for ts in values) > 1e11 else "s"
    return pd.to_datetime(values, unit=unit, utc=True)


def _annualized_sharpe(equity_series: pd.Series, bundle_name: str) -> float:
    if equity_series.empty or len(equity_series) < 3:
        return 0.0
    returns = equity_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return 0.0
    vol = float(returns.std(ddof=0))
    if vol <= 1e-12:
        return 0.0
    return float((returns.mean() / vol) * np.sqrt(_bars_per_year(bundle_name)))


def _max_drawdown_pct(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    running_max = equity_series.cummax().replace(0.0, np.nan)
    drawdown = ((running_max - equity_series) / running_max).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(drawdown.max() * 100.0)


def _duration_days(timestamps: list[int], bundle_name: str) -> float:
    if len(timestamps) >= 2:
        first = int(timestamps[0])
        last = int(timestamps[-1])
        scale = 1000.0 if abs(first) > 1e11 or abs(last) > 1e11 else 1.0
        return max((last - first) / (86400.0 * scale), 0.0)
    if len(timestamps) == 1:
        base_hours = timeframe_to_hours(BUNDLE_MANIFESTS[bundle_name].base_tf)
        return base_hours / 24.0
    return 0.0


def _profit_metrics(close_rows: list[dict]) -> tuple[int, float, float]:
    if not close_rows:
        return 0, 0.0, 0.0
    pnls = [float(row.get("realized_pnl", 0.0) or 0.0) for row in close_rows]
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    gross_profit = sum(max(pnl, 0.0) for pnl in pnls)
    gross_loss = sum(-min(pnl, 0.0) for pnl in pnls)
    win_rate = (wins / len(pnls)) * 100.0 if pnls else 0.0
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    return len(close_rows), float(win_rate), float(profit_factor)


def _annualized_ratio(returns: pd.Series, periods_per_year: float = 252.0, *, downside_only: bool = False) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    if downside_only:
        downside = clean[clean < 0.0]
        if downside.empty:
            return 0.0
        denom = float(np.sqrt(np.mean(np.square(downside))))
    else:
        denom = float(clean.std(ddof=0))
    if denom <= 1e-12:
        return 0.0
    return float((clean.mean() / denom) * np.sqrt(periods_per_year))


def _daily_equity_series(equity_series: pd.Series) -> pd.Series:
    if equity_series.empty:
        return pd.Series(dtype=float)
    dated = pd.Series(equity_series.values, index=_timestamp_index_to_datetime(equity_series.index))
    return dated.groupby(dated.index.normalize()).last()


def _secondary_metrics(equity_series: pd.Series, bundle_name: str) -> dict:
    daily_equity = _daily_equity_series(equity_series)
    daily_returns = daily_equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "bar_sharpe": float(_annualized_sharpe(equity_series, bundle_name)),
        "daily_sharpe": float(_annualized_ratio(daily_returns, 252.0)),
        "daily_sortino": float(_annualized_ratio(daily_returns, 252.0, downside_only=True)),
        "daily_observations": int(len(daily_returns)),
    }


def build_v2_backtest_result(
    *,
    bundle_name: str,
    run_result: dict,
    summary: dict,
    log_rows: list[dict],
) -> prepare.BacktestResult:
    equity_series = _series_from_curve(
        run_result.get("equity_curve", []) or [],
        run_result.get("equity_timestamps", []) or [],
    )
    close_rows = [row for row in log_rows if row.get("action") == "close"]
    num_trades, win_rate_pct, profit_factor = _profit_metrics(close_rows)
    total_bars = int(run_result.get("total_bars_available", run_result.get("bars_processed", 0)) or 0)
    bars_processed = int(run_result.get("bars_processed", 0) or 0)
    duration_days = _duration_days(run_result.get("equity_timestamps", []) or [], bundle_name)
    if duration_days <= 0.0 and bars_processed > 0:
        duration_days = (bars_processed * timeframe_to_hours(BUNDLE_MANIFESTS[bundle_name].base_tf)) / 24.0
    annual_turnover = 0.0
    total_volume = float(summary.get("portfolio", {}).get("total_volume", 0.0) or 0.0)
    if bars_processed > 0:
        annual_turnover = total_volume * (_bars_per_year(bundle_name) / bars_processed)

    final_equity = float(summary.get("portfolio", {}).get("equity", prepare.INITIAL_CAPITAL) or prepare.INITIAL_CAPITAL)
    total_return_pct = ((final_equity / prepare.INITIAL_CAPITAL) - 1.0) * 100.0

    return prepare.BacktestResult(
        sharpe=_annualized_sharpe(equity_series, bundle_name),
        total_return_pct=float(total_return_pct),
        max_drawdown_pct=_max_drawdown_pct(equity_series),
        num_trades=num_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        annual_turnover=float(annual_turnover),
        backtest_seconds=0.0,
        duration_days=float(duration_days),
        timed_out=False,
        bars_processed=bars_processed,
        total_bars=total_bars,
        equity_curve=list(equity_series.values) if not equity_series.empty else [],
        equity_timestamps=[int(ts) for ts in equity_series.index.tolist()] if not equity_series.empty else [],
        bar_regimes=[],
        trade_log=log_rows,
        trade_context_log=log_rows,
    )


def run_v2_backtest(
    *,
    bundle_name: str,
    model_set: str,
    split: str,
    portfolio_config: PortfolioConfig,
    active_sleeves: list[str] | None = None,
    max_days: int | None = None,
    max_symbols: int | None = None,
) -> dict:
    engine = V2SignalEngine(
        bundle_name=bundle_name,
        model_set=model_set,
        active_sleeves=active_sleeves,
        max_symbols=max_symbols,
    )
    engine.prepare(split)

    with tempfile.TemporaryDirectory(prefix="v2_backtest_") as tmpdir:
        root = Path(tmpdir)
        state_path = root / "state.json"
        log_path = root / "log.jsonl"
        run_result = run_v2_shadow_session(
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
            engine=engine,
            collect_bar_history=True,
        )
        summary = summarize_shadow_run(state_path=state_path, log_path=log_path, max_recent=12)
        log_rows = _read_jsonl(log_path)

    result = build_v2_backtest_result(
        bundle_name=bundle_name,
        run_result=run_result,
        summary=summary,
        log_rows=log_rows,
    )
    equity_series = _series_from_curve(result.equity_curve, result.equity_timestamps)
    secondary_metrics = _secondary_metrics(equity_series, bundle_name)
    score = prepare.compute_score(result)
    trades_per_day = (result.num_trades / result.duration_days) if result.duration_days > 0 else 0.0
    top_sleeves = summary.get("sleeve_concentration", [])[:3]
    top_share = max([float(row.get("pnl_share", 0.0) or 0.0) for row in top_sleeves], default=0.0)
    return {
        "bundle": bundle_name,
        "model_set": model_set,
        "split": split,
        "result": result,
        "score": float(score),
        "status": classify_v2_run_status(float(score)),
        "trades_per_day": float(trades_per_day),
        **secondary_metrics,
        "metric_caveat": (
            "score uses the legacy bar-level Sharpe composite from prepare.compute_score; "
            "prefer daily_sharpe and daily_sortino for V2 credibility checks"
        ),
        "top_share": float(top_share),
        "short_share": float(
            sum(
                1
                for row in log_rows
                if row.get("action") == "open" and float(row.get("target_position", 0.0) or 0.0) < 0.0
            )
            / max(1, sum(1 for row in log_rows if row.get("action") == "open"))
        ),
        "summary": summary,
        "top_sleeves": top_sleeves,
        "log_rows": log_rows,
    }


def next_v2_results_id(path: str | Path = "results.tsv", prefix: str = "v2exp") -> str:
    target = Path(path)
    if not target.exists():
        return f"{prefix}1"
    try:
        df = pd.read_csv(target, sep="\t")
        commits = df.get("commit", pd.Series(dtype=str)).astype(str).str.strip()
        matches = commits.str.extract(rf"^{prefix}(\d+)$").dropna()
        if matches.empty:
            return f"{prefix}1"
        last = int(matches.astype(int).max().iloc[0])
        return f"{prefix}{last + 1}"
    except Exception:
        return f"{prefix}1"


def append_v2_results_row(
    *,
    backtest: dict,
    description: str,
    path: str | Path = "results.tsv",
    prefix: str = "v2exp",
) -> str:
    result: prepare.BacktestResult = backtest["result"]
    run_id = next_v2_results_id(path=path, prefix=prefix)
    row = {
        "commit": run_id,
        "score": f"{backtest['score']:.3f}",
        "sharpe": f"{result.sharpe:.3f}",
        "max_dd": f"{result.max_drawdown_pct:.2f}",
        "status": backtest["status"],
        "description": description,
    }
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(
            "\t".join(
                [str(row[column]) for column in ["commit", "score", "sharpe", "max_dd", "status", "description"]]
            )
            + "\n"
        )
    return run_id


def backtest_to_dict(backtest: dict) -> dict:
    result: prepare.BacktestResult = backtest["result"]
    payload = {k: v for k, v in backtest.items() if k not in {"result", "log_rows"}}
    payload["result"] = {
        "sharpe": float(result.sharpe),
        "total_return_pct": float(result.total_return_pct),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "num_trades": int(result.num_trades),
        "win_rate_pct": float(result.win_rate_pct),
        "profit_factor": float(result.profit_factor),
        "annual_turnover": float(result.annual_turnover),
        "duration_days": float(result.duration_days),
        "bars_processed": int(result.bars_processed),
        "total_bars": int(result.total_bars),
    }
    return payload
