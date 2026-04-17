"""
Robustness Evaluation Harness
Runs OOS, Fee Stress, Capacity, Regime Breakdown, and Monte Carlo
for the current loaded strategy/models.

Usage:
    uv run evaluate.py --timeframe 1h --label exp269
    uv run evaluate.py --timeframe 1h --label exp268 --mc-sims 1000
"""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import prepare
from market_regime import detect_regimes as detect_regime_segments, regime_family
from prepare import load_data, run_backtest, compute_score
from strategy import Strategy

TIMEFRAME_SECS = {"15m": 900, "1h": 3600, "4h": 14400}

BENCHMARKS = {
    "sharpe":           3.5,
    "win_rate_pct":    60.0,
    "trades_per_day":   1.0,
    "profit_factor":    4.0,
    "max_drawdown_pct": 10.0,
}

SPLIT_WINDOWS = {
    "oos": (prepare.TEST_END, "2025-01-01", "2025-12-31"),
    "2026q1": ("2025-12-31", "2026-01-01", "2026-03-31"),
    "holdout": ("2025-09-30", prepare.HOLDOUT_START, prepare.HOLDOUT_END),
    "val": (prepare.TRAIN_END, prepare.VAL_START, prepare.VAL_END),
    "robustness": (prepare.TRAIN_START, prepare.ROBUST_START, prepare.ROBUST_END),
}
FRED_SP500_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
BENCHMARK_CACHE = Path.home() / ".cache" / "autotrader" / "benchmarks"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_strategy(timeframe):
    strat = Strategy(timeframe=timeframe)
    return strat


def _normalize_split(timeframe, split):
    if timeframe == "15m" and not split.endswith("_15m"):
        return f"{split}_15m"
    return split


def _base_split_name(split):
    return split[:-4] if split.endswith("_15m") else split


def _split_window(split):
    base = _base_split_name(split)
    prior_end, start, end = SPLIT_WINDOWS.get(base, ("?", "?", "?"))
    return prior_end, start, end


def _load_and_run(timeframe, split, data=None):
    if data is None:
        data = load_data(split, resample_4h=(timeframe == "4h"))
    strat = _make_strategy(timeframe)
    if hasattr(strat, "pre_calculate_signals"):
        strat.pre_calculate_signals(data, split_name=split)
    res = run_backtest(strat, data, bar_interval_sec=TIMEFRAME_SECS[timeframe])
    return res, data


def _pass_fail(label, value, target, lower_is_better=False):
    passed = (value < target) if lower_is_better else (value >= target)
    sym = "✓" if passed else "✗"
    cmp = "<" if lower_is_better else ">="
    print(f"  {sym} {label:<26} {value:>9.3f}   (target {cmp} {target})")


def _to_timestamp(series: pd.Series) -> pd.DatetimeIndex:
    raw = pd.to_numeric(series, errors="coerce").astype("Int64")
    unit = "ms" if raw.dropna().gt(10**11).any() else "s"
    return pd.to_datetime(raw, unit=unit, utc=True)


def _equity_series(result):
    if not result.equity_curve or not result.equity_timestamps:
        return None
    if len(result.equity_curve) != len(result.equity_timestamps):
        return None
    ts = pd.to_datetime(pd.Series(result.equity_timestamps), unit="s", utc=True)
    return pd.Series(result.equity_curve, index=ts).sort_index()


def _benchmark_price_series(data, symbol="SP500"):
    df = data.get(symbol)
    if df is not None and not df.empty:
        ts = _to_timestamp(df["timestamp"])
        prices = pd.Series(pd.to_numeric(df["close"], errors="coerce").values, index=ts).dropna().sort_index()
        if not prices.empty:
            return prices, "parquet"
    return _fred_sp500_series()


def _fred_sp500_series():
    cache_path = BENCHMARK_CACHE / "SP500_fred.csv"
    df = None
    try:
        BENCHMARK_CACHE.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            df = pd.read_csv(FRED_SP500_URL)
            df.to_csv(cache_path, index=False)
        else:
            df = pd.read_csv(cache_path)
    except Exception:
        if not cache_path.exists():
            return None, None
        try:
            df = pd.read_csv(cache_path)
        except Exception:
            return None, None

    if df is None or df.empty or "observation_date" not in df.columns or "SP500" not in df.columns:
        return None, None

    ts = pd.to_datetime(df["observation_date"], utc=True, errors="coerce")
    prices = pd.Series(pd.to_numeric(df["SP500"], errors="coerce").values, index=ts).dropna().sort_index()
    if prices.empty:
        return None, None
    return prices, "fred-daily"


def _annualized_sortino(daily_returns: pd.Series) -> float:
    if daily_returns is None or len(daily_returns) < 2:
        return float("nan")
    downside = np.minimum(daily_returns.values, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))))
    if downside_dev <= 1e-12:
        return float("nan")
    return float((daily_returns.mean() / downside_dev) * np.sqrt(252.0))


def _compute_institutional_metrics(result, data):
    metrics = {
        "sortino": float("nan"),
        "beta_to_spx": float("nan"),
        "alpha_annual_pct": float("nan"),
        "spx_return_pct": float("nan"),
        "excess_return_vs_spx_pct": float("nan"),
        "aligned_days": 0,
        "benchmark_source": None,
    }

    equity = _equity_series(result)
    if equity is None or equity.empty:
        return metrics

    daily_equity = equity.groupby(equity.index.floor("1D")).last()
    portfolio_rets = daily_equity.pct_change().dropna()
    if portfolio_rets.empty:
        return metrics

    metrics["sortino"] = _annualized_sortino(portfolio_rets)

    spx_prices, benchmark_source = _benchmark_price_series(data, symbol="SP500")
    if spx_prices is None:
        return metrics
    metrics["benchmark_source"] = benchmark_source

    daily_spx = spx_prices.groupby(spx_prices.index.floor("1D")).last()
    spx_rets = daily_spx.pct_change().dropna()
    if spx_rets.empty:
        return metrics

    aligned = pd.concat(
        [portfolio_rets.rename("portfolio"), spx_rets.rename("spx")],
        axis=1,
        join="inner",
    ).dropna()
    metrics["aligned_days"] = len(aligned)
    if len(aligned) < 2:
        return metrics

    spx_var = float(aligned["spx"].var(ddof=0))
    if spx_var > 1e-12:
        cov = float(np.cov(aligned["portfolio"], aligned["spx"], ddof=0)[0, 1])
        beta = cov / spx_var
        alpha_daily = float(aligned["portfolio"].mean() - beta * aligned["spx"].mean())
        metrics["beta_to_spx"] = beta
        metrics["alpha_annual_pct"] = alpha_daily * 252.0 * 100.0

    metrics["spx_return_pct"] = float((daily_spx.iloc[-1] / daily_spx.iloc[0] - 1.0) * 100.0)
    metrics["excess_return_vs_spx_pct"] = result.total_return_pct - metrics["spx_return_pct"]
    return metrics


def _print_institutional_metrics(metrics):
    print(f"\n{'='*60}")
    print("  INSTITUTIONAL METRICS")
    print(f"{'='*60}")
    sortino = metrics["sortino"]
    beta = metrics["beta_to_spx"]
    alpha = metrics["alpha_annual_pct"]
    spx_ret = metrics["spx_return_pct"]
    excess = metrics["excess_return_vs_spx_pct"]
    aligned_days = metrics["aligned_days"]
    benchmark_source = metrics["benchmark_source"] or "n/a"

    def _fmt(value, suffix=""):
        return f"{value:.4f}{suffix}" if np.isfinite(value) else "n/a"

    print(f"  Sortino (annualized):      {_fmt(sortino)}")
    print(f"  Beta to SPX:               {_fmt(beta)}")
    print(f"  Alpha vs SPX (annualized): {_fmt(alpha, '%')}")
    print(f"  SPX Return:                {_fmt(spx_ret, '%')}")
    print(f"  Excess Return vs SPX:      {_fmt(excess, '%')}")
    print(f"  Aligned daily samples:     {aligned_days}")
    print(f"  Benchmark source:          {benchmark_source}")
    if not np.isfinite(beta):
        print("  Note: SPX-relative metrics unavailable because SP500 data was missing or too sparse.")


# ─────────────────────────────────────────────────────────────
# 1. OOS Test
# ─────────────────────────────────────────────────────────────

def run_primary_eval(timeframe, label, split):
    split = _normalize_split(timeframe, split)
    split_base = _base_split_name(split)
    prior_end, start, end = _split_window(split)
    print(f"\n{'='*60}")
    header = {
        "oos": "OOS TEST",
        "2026q1": "POST-RESEARCH OOS TEST",
        "holdout": "HOLDOUT TEST",
        "val": "VALIDATION TEST",
        "robustness": "ROBUSTNESS TEST",
    }.get(split_base, "PRIMARY EVAL")
    print(f"  {header}  [{label}]  ({start} → {end})")
    print(f"{'='*60}")
    res, data = _load_and_run(timeframe, split)
    dur = res.duration_days if res.duration_days > 0 else 1
    trades_per_day = res.num_trades / dur
    score = compute_score(res)
    print(f"  Score:          {score:.4f}")
    print(f"  Sharpe:         {res.sharpe:.4f}")
    print(f"  Return:         {res.total_return_pct:.2f}%")
    print(f"  Max DD:         {res.max_drawdown_pct:.2f}%")
    print(f"  Trades:         {res.num_trades}  ({trades_per_day:.2f}/day)")
    print(f"  Win Rate:       {res.win_rate_pct:.2f}%")
    print(f"  Profit Factor:  {res.profit_factor:.3f}")
    print()
    _pass_fail("Sharpe",        res.sharpe,          BENCHMARKS["sharpe"])
    _pass_fail("Win Rate %",    res.win_rate_pct,    BENCHMARKS["win_rate_pct"])
    _pass_fail("Trades/Day",    trades_per_day,      BENCHMARKS["trades_per_day"])
    _pass_fail("Profit Factor", res.profit_factor,   BENCHMARKS["profit_factor"])
    _pass_fail("Max DD %",      res.max_drawdown_pct,BENCHMARKS["max_drawdown_pct"], lower_is_better=True)
    metrics = _compute_institutional_metrics(res, data)
    _print_institutional_metrics(metrics)
    return res, data, score, metrics


# ─────────────────────────────────────────────────────────────
# 2. Fee Stress Test
# ─────────────────────────────────────────────────────────────

def run_fee_stress(timeframe, split, data, label):
    print(f"\n{'='*60}")
    print(f"  FEE STRESS TEST  [{label}]")
    print(f"{'='*60}")
    base_maker  = prepare.MAKER_FEE
    base_taker  = prepare.TAKER_FEE
    base_slip   = prepare.SLIPPAGE_BPS
    print(f"  {'Multiplier':<12} {'Taker (bps)':<14} {'Sharpe':>8} {'Return%':>9} {'Trades':>7} {'PF':>7}")
    print(f"  {'-'*58}")
    for m in [1.0, 1.5, 2.0, 3.0, 5.0]:
        prepare.MAKER_FEE  = base_maker * m
        prepare.TAKER_FEE  = base_taker * m
        prepare.SLIPPAGE_BPS = base_slip * m
        strat = _make_strategy(timeframe)
        if hasattr(strat, "pre_calculate_signals"):
            strat.pre_calculate_signals(data, split_name=split)
        res = run_backtest(strat, data, bar_interval_sec=TIMEFRAME_SECS[timeframe])
        tag = " ← baseline" if m == 1.0 else ""
        print(f"  {m:<12.1f} {prepare.TAKER_FEE*10000:<14.1f} "
              f"{res.sharpe:>8.3f} {res.total_return_pct:>9.2f} "
              f"{res.num_trades:>7d} {res.profit_factor:>7.3f}{tag}")
    prepare.MAKER_FEE  = base_maker
    prepare.TAKER_FEE  = base_taker
    prepare.SLIPPAGE_BPS = base_slip


# ─────────────────────────────────────────────────────────────
# 3. Capacity / Scalability Test
# ─────────────────────────────────────────────────────────────

def run_capacity(timeframe, split, data, label):
    print(f"\n{'='*60}")
    print(f"  CAPACITY TEST  [{label}]")
    print(f"{'='*60}")
    base_cap  = prepare.INITIAL_CAPITAL
    base_slip = prepare.SLIPPAGE_BPS
    print(f"  {'Capital':>12} {'Slip(bps)':>10} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8}")
    print(f"  {'-'*50}")
    for cap in [10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
        prepare.INITIAL_CAPITAL = cap
        prepare.SLIPPAGE_BPS = max(1.0, base_slip + (cap / 100_000) * 0.3)
        strat = _make_strategy(timeframe)
        if hasattr(strat, "pre_calculate_signals"):
            strat.pre_calculate_signals(data, split_name=split)
        res = run_backtest(strat, data, bar_interval_sec=TIMEFRAME_SECS[timeframe])
        tag = " ← baseline" if cap == 100_000 else ""
        print(f"  ${cap:>11,.0f} {prepare.SLIPPAGE_BPS:>10.1f} "
              f"{res.sharpe:>8.3f} {res.total_return_pct:>9.2f} "
              f"{res.max_drawdown_pct:>8.2f}{tag}")
    prepare.INITIAL_CAPITAL = base_cap
    prepare.SLIPPAGE_BPS    = base_slip


# ─────────────────────────────────────────────────────────────
# 4. Regime Breakdown
# ─────────────────────────────────────────────────────────────

def run_regime_breakdown(timeframe, split, data, label):
    print(f"\n{'='*60}")
    print(f"  REGIME BREAKDOWN  [{label}]")
    print(f"{'='*60}")
    segs = detect_regime_segments(data)
    MIN_MS = 14 * 24 * 3600 * 1000
    segs = [(r, a, b) for r, a, b in segs if (b - a) >= MIN_MS]
    if not segs:
        print("  No qualifying regime segments found.")
        return
    print(f"  {'Regime':<22} {'Period':<26} {'Sharpe':>8} {'DD%':>7} {'Trades':>7} {'Ret%':>9}")
    print(f"  {'-'*80}")
    for regime, t_start, t_end in segs:
        rdata = {}
        for sym, df in data.items():
            mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
            rdf  = df[mask].reset_index(drop=True)
            if len(rdf) > 20:
                rdata[sym] = rdf
        if not rdata:
            continue
        strat = _make_strategy(timeframe)
        if hasattr(strat, "pre_calculate_signals"):
            strat.pre_calculate_signals(rdata, split_name=split)
        res = run_backtest(strat, rdata, bar_interval_sec=TIMEFRAME_SECS[timeframe])
        s = pd.Timestamp(t_start, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        e = pd.Timestamp(t_end,   unit="ms", tz="UTC").strftime("%Y-%m-%d")
        print(f"  {regime:<22} {s} → {e}  "
              f"{res.sharpe:>+8.3f} {res.max_drawdown_pct:>7.1f} "
              f"{res.num_trades:>7d} {res.total_return_pct:>+9.2f}%")


def run_regime_trade_attribution(result, label):
    print(f"\n{'='*60}")
    print(f"  REGIME ATTRIBUTION  [{label}]")
    print(f"{'='*60}")

    if not result.bar_regimes:
        print("  No per-bar regime trace found.")
        return

    bar_counts = pd.Series(result.bar_regimes).value_counts()
    total_bars = int(bar_counts.sum())
    print("  Bar Mix")
    print(f"  {'Regime':<22} {'Bars':>8} {'Pct':>8}")
    print(f"  {'-'*40}")
    for regime, count in bar_counts.items():
        pct = 100.0 * count / total_bars if total_bars else 0.0
        print(f"  {regime:<22} {int(count):>8d} {pct:>7.1f}%")

    family_counts = (
        pd.Series(result.bar_regimes)
        .map(regime_family)
        .value_counts()
    )
    print("\n  Bar Mix By Family")
    print(f"  {'Family':<12} {'Bars':>8} {'Pct':>8}")
    print(f"  {'-'*32}")
    for family, count in family_counts.items():
        pct = 100.0 * count / total_bars if total_bars else 0.0
        print(f"  {family:<12} {int(count):>8d} {pct:>7.1f}%")

    closes = [row for row in result.trade_context_log if row.get("event") == "close"]
    if not closes:
        print("\n  No timestamped close events found for regime attribution.")
        return

    df = pd.DataFrame(closes)
    df["side"] = np.where(df["delta"] < 0, "long", "short")
    df["regime_family"] = df["regime"].map(regime_family)

    print("\n  Closed Trades By Regime")
    print(f"  {'Regime':<22} {'Closes':>8} {'Win%':>8} {'Net PnL':>12} {'Avg PnL':>10}")
    print(f"  {'-'*66}")
    grouped = (
        df.groupby("regime")
        .agg(
            closes=("event", "count"),
            win_rate_pct=("pnl", lambda s: 100.0 * float((s > 0).mean())),
            net_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        )
        .reset_index()
    )
    for row in grouped.sort_values("net_pnl", ascending=False).itertuples(index=False):
        print(
            f"  {row.regime:<22} {int(row.closes):>8d} "
            f"{row.win_rate_pct:>7.1f}% {row.net_pnl:>12.2f} {row.avg_pnl:>10.2f}"
        )

    print("\n  Closed Trades By Family")
    print(f"  {'Family':<12} {'Closes':>8} {'Win%':>8} {'Net PnL':>12} {'Avg PnL':>10}")
    print(f"  {'-'*56}")
    family_grouped = (
        df.groupby("regime_family")
        .agg(
            closes=("event", "count"),
            win_rate_pct=("pnl", lambda s: 100.0 * float((s > 0).mean())),
            net_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        )
        .reset_index()
    )
    for row in family_grouped.sort_values("net_pnl", ascending=False).itertuples(index=False):
        print(
            f"  {row.regime_family:<12} {int(row.closes):>8d} "
            f"{row.win_rate_pct:>7.1f}% {row.net_pnl:>12.2f} {row.avg_pnl:>10.2f}"
        )

    side_counts = df.groupby(["regime_family", "side"]).size()
    if not side_counts.empty:
        print("\n  Closed Trades By Family And Side")
        print(f"  {'Family':<12} {'Side':<8} {'Closes':>8}")
        print(f"  {'-'*42}")
        for (family, side), count in side_counts.sort_values(ascending=False).items():
            print(f"  {family:<12} {side:<8} {int(count):>8d}")


# ─────────────────────────────────────────────────────────────
# 5. Monte Carlo
# ─────────────────────────────────────────────────────────────

def run_monte_carlo(oos_result, label, n_sims=1000):
    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SIMULATION  [{label}]  (n={n_sims})")
    print(f"{'='*60}")

    eq = np.array(oos_result.equity_curve)
    if len(eq) < 10:
        print("  Insufficient equity curve data for Monte Carlo.")
        return

    # Trade-level P&L from equity curve daily returns
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    if len(rets) < 10:
        print("  Insufficient returns for Monte Carlo.")
        return

    sharpes, max_dds, total_rets = [], [], []
    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        sim_rets = rng.choice(rets, size=len(rets), replace=True)
        sim_eq   = np.cumprod(1 + sim_rets) * eq[0]
        # Sharpe (annualised assuming 1h bars → 8760 bars/year)
        ann_factor = np.sqrt(8760)
        sh = (sim_rets.mean() / (sim_rets.std() + 1e-9)) * ann_factor
        # Max drawdown
        peak = np.maximum.accumulate(sim_eq)
        dd   = ((sim_eq - peak) / peak).min() * 100
        tr   = (sim_eq[-1] / sim_eq[0] - 1) * 100
        sharpes.append(sh)
        max_dds.append(abs(dd))
        total_rets.append(tr)

    sharpes    = np.array(sharpes)
    max_dds    = np.array(max_dds)
    total_rets = np.array(total_rets)

    p5, p50, p95 = np.percentile(sharpes, [5, 50, 95])
    dd5, dd50, dd95 = np.percentile(max_dds, [5, 50, 95])
    r5, r50, r95 = np.percentile(total_rets, [5, 50, 95])
    prob_beat_target = (sharpes >= BENCHMARKS["sharpe"]).mean() * 100

    print(f"  Metric         {'p5':>8}  {'p50 (median)':>14}  {'p95':>8}")
    print(f"  {'-'*46}")
    print(f"  Sharpe         {p5:>8.3f}  {p50:>14.3f}  {p95:>8.3f}")
    print(f"  Max DD %       {dd5:>8.2f}  {dd50:>14.2f}  {dd95:>8.2f}")
    print(f"  Total Ret %    {r5:>8.2f}  {r50:>14.2f}  {r95:>8.2f}")
    print()
    print(f"  Prob(Sharpe >= {BENCHMARKS['sharpe']:.1f}): {prob_beat_target:.1f}%")
    print(f"  Worst-case Sharpe  (p1): {np.percentile(sharpes, 1):.3f}")
    print(f"  Best-case  Sharpe (p99): {np.percentile(sharpes, 99):.3f}")

    verdict = "ROBUST" if prob_beat_target >= 70 else ("BORDERLINE" if prob_beat_target >= 40 else "FRAGILE")
    print(f"\n  Monte Carlo Verdict: *** {verdict} ***")
    return dict(p5=p5, p50=p50, p95=p95, prob_beat=prob_beat_target)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness Evaluation Harness")
    parser.add_argument("--timeframe",  default="1h",  choices=["15m", "1h", "4h"])
    parser.add_argument("--label",      default="exp269", help="Experiment label for reporting")
    parser.add_argument("--mc-sims",    type=int, default=1000, help="Monte Carlo simulations")
    parser.add_argument("--skip-mc",    action="store_true", help="Skip Monte Carlo (faster)")
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--split",      default="oos", choices=["oos", "2026q1", "val", "robustness"],
                        help="Primary evaluation split (default: oos)")
    parser.add_argument("--holdout",    action="store_true",
                        help="FINAL SIGN-OFF ONLY: run on Oct–Dec 2025 holdout (never re-iterate)")
    args = parser.parse_args()

    tf    = args.timeframe
    lbl   = args.label
    split = args.split

    # ── Holdout friction gate ──────────────────────────────────────────────
    if args.holdout:
        print("\n" + "!" * 60)
        print("  HOLDOUT GATE — READ BEFORE CONTINUING")
        print("  Oct–Dec 2025 is the ONLY truly unseen data remaining.")
        print("  Use this split AT MOST ONCE for final production sign-off.")
        print("  Re-iterating against it invalidates the independence read.")
        print("!" * 60)
        confirm = input("\n  Type FINAL-SIGN-OFF to proceed, or anything else to abort: ").strip()
        if confirm != "FINAL-SIGN-OFF":
            print("  Aborted. Holdout preserved.")
            raise SystemExit(0)
        split = "holdout"
        print()

    print(f"\n{'#'*60}")
    print(f"  ROBUSTNESS SUITE: {lbl.upper()}  |  Timeframe: {tf.upper()}")
    if args.holdout:
        print(f"  *** HOLDOUT MODE: Oct–Dec 2025 (FINAL SIGN-OFF) ***")
    print(f"{'#'*60}")
    t0 = time.time()

    oos_res, oos_data, oos_score, primary_metrics = run_primary_eval(tf, lbl, split)

    # Fee Stress
    run_fee_stress(tf, split, oos_data, lbl)

    # Capacity
    run_capacity(tf, split, oos_data, lbl)

    # Regime Breakdown
    if not args.skip_regime:
        run_regime_breakdown(tf, split, oos_data, lbl)
    run_regime_trade_attribution(oos_res, lbl)

    # Monte Carlo
    mc_result = None
    if not args.skip_mc:
        mc_result = run_monte_carlo(oos_res, lbl, n_sims=args.mc_sims)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {lbl.upper()}")
    print(f"{'='*60}")
    print(f"  OOS Score:    {oos_score:.4f}")
    print(f"  OOS Sharpe:   {oos_res.sharpe:.4f}")
    if np.isfinite(primary_metrics["sortino"]):
        print(f"  OOS Sortino:  {primary_metrics['sortino']:.4f}")
    if np.isfinite(primary_metrics["beta_to_spx"]):
        print(f"  Beta to SPX:  {primary_metrics['beta_to_spx']:.4f}")
    print(f"  OOS MaxDD:    {oos_res.max_drawdown_pct:.2f}%")
    if mc_result:
        print(f"  MC p50 Sharpe: {mc_result['p50']:.4f}")
        print(f"  MC Prob≥{BENCHMARKS['sharpe']:.1f}: {mc_result['prob_beat']:.1f}%")
    print(f"  Total time:    {total:.1f}s")
