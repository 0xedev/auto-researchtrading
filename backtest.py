"""
Unified Backtest Runner.
Usage: uv run backtest.py --timeframe 1h
       uv run backtest.py --timeframe 4h --stress-fees --capacity
"""
import time
import argparse
import signal as sig
import numpy as np
import pandas as pd
import prepare
from prepare import load_data, run_backtest, compute_score, TIME_BUDGET
from strategy import Strategy

TIMEFRAME_SECS = {"15m": 900, "1h": 3600, "4h": 14400}

BENCHMARKS = {
    "sharpe": 3.5,
    "win_rate_pct": 60.0,
    "trades_per_day": 1.0,
    "profit_factor": 4.0,
    "max_drawdown_pct": 10.0,
}


def classify_run_status(result, score, timeframe):
    duration_days = result.duration_days if result.duration_days > 0 else 0.0
    trades_per_day = result.num_trades / duration_days if duration_days > 0 else 0.0

    if timeframe == "15m":
        if (
            score > 0.0
            and result.total_return_pct > 0.0
            and trades_per_day >= 1.0
            and result.max_drawdown_pct < 10.0
        ):
            return "CANDIDATE"
        return "REVERT"

    return "KEEP" if score > 3.0 else "REVERT"


def timeout_handler(signum, frame):
    print("TIMEOUT: backtest exceeded time budget")
    exit(1)

sig.signal(sig.SIGALRM, timeout_handler)
# Alarm set later after args parsed (4H needs extra budget for regime breakdown)


def print_benchmark_audit(result, num_symbols):
    duration_days = result.duration_days if result.duration_days > 0 else 0.0
    trades_per_day = result.num_trades / duration_days if duration_days > 0 else 0.0
    trades_per_day_per_symbol = trades_per_day / num_symbols if num_symbols > 0 else 0.0

    checks = [
        ("Sharpe", result.sharpe, BENCHMARKS["sharpe"], result.sharpe >= BENCHMARKS["sharpe"]),
        ("Win Rate %", result.win_rate_pct, BENCHMARKS["win_rate_pct"], result.win_rate_pct >= BENCHMARKS["win_rate_pct"]),
        (
            "Trades/Day",
            trades_per_day,
            BENCHMARKS["trades_per_day"],
            trades_per_day >= BENCHMARKS["trades_per_day"],
        ),
        (
            "Profit Factor",
            result.profit_factor,
            BENCHMARKS["profit_factor"],
            result.profit_factor >= BENCHMARKS["profit_factor"],
        ),
        (
            "Max Drawdown %",
            result.max_drawdown_pct,
            BENCHMARKS["max_drawdown_pct"],
            result.max_drawdown_pct < BENCHMARKS["max_drawdown_pct"],
        ),
    ]

    print("\n" + "=" * 60)
    print("  BENCHMARK AUDIT")
    print("=" * 60)
    print(f"trades_per_day:             {trades_per_day:.6f}")
    print(f"trades_per_day_per_symbol:  {trades_per_day_per_symbol:.6f}")
    for label, value, target, passed in checks:
        status = "PASS" if passed else "FAIL"
        comparator = "<" if label == "Max Drawdown %" else ">="
        print(f"{label:<24} {value:>10.6f}   target {comparator} {target:<8.3f} {status}")

def detect_regimes(data, anchor_symbol="BTC"):
    """
    Automatically detect market regimes from price data using:
    - 200-bar EMA slope direction (Bull vs Bear trend)
    - 30-bar realized volatility percentile (Low/High vol)
    No hardcoded date labels. The data decides.
    """
    ref_df = data.get(anchor_symbol)
    if ref_df is None:
        ref_df = next(iter(data.values()))

    close = ref_df['close'].reset_index(drop=True)
    timestamps = ref_df['timestamp'].reset_index(drop=True)

    ema200 = close.ewm(span=200, adjust=False).mean()
    ema_slope = ema200.pct_change(20)  # 20-bar momentum of EMA

    returns = close.pct_change()
    vol_30 = returns.rolling(30).std()
    vol_pct = vol_30.rank(pct=True)   # 0.0 to 1.0 percentile rank

    def classify(slope, vol):
        if pd.isna(slope) or pd.isna(vol):
            return "Initializing"
        if slope > 0.01 and vol < 0.60:
            return "Bull (Low Vol)"
        elif slope > 0.01 and vol >= 0.60:
            return "Bull (High Vol)"
        elif slope < -0.01 and vol >= 0.50:
            return "Bear (Panic)"
        elif slope < -0.01 and vol < 0.50:
            return "Bear (Grind)"
        else:
            return "Sideways (Chop)"

    regime_series = [classify(ema_slope.iloc[i], vol_pct.iloc[i]) for i in range(len(close))]

    # Segment into contiguous regime blocks
    segments = []
    prev_regime = None
    seg_start = None
    for i, regime in enumerate(regime_series):
        if regime != prev_regime:
            if prev_regime is not None and prev_regime != "Initializing":
                segments.append((prev_regime, seg_start, timestamps.iloc[i - 1]))
            seg_start = timestamps.iloc[i]
            prev_regime = regime
    if prev_regime and prev_regime != "Initializing":
        segments.append((prev_regime, seg_start, timestamps.iloc[-1]))

    return segments


def print_regimes(result, data):
    print("\n" + "=" * 60)
    print("  AUTO-DETECTED REGIME BREAKDOWN")
    print("  (BTC EMA slope + realized vol percentile)")
    print("=" * 60)

    segments = detect_regimes(data)

    # Merge micro-segments: only report segments > 14 days
    merged = []
    min_duration_ms = 14 * 24 * 3600 * 1000
    for regime, t_start, t_end in segments:
        if (t_end - t_start) < min_duration_ms:
            continue
        merged.append((regime, t_start, t_end))

    if not merged:
        print("  No long-duration regimes detected in data range.")
        return

    print(f"  {'Regime':<22}  {'Period':<28}  {'Sharpe':>7}  {'DD':>6}  {'Trades':>6}  {'Return':>8}")
    print("  " + "-" * 82)

    for regime, t_start, t_end in merged:
        regime_data = {}
        for symbol, df in data.items():
            mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
            rdf = df[mask].reset_index(drop=True)
            if len(rdf) > 10:
                regime_data[symbol] = rdf

        if not regime_data:
            continue

        regime_strategy = Strategy(timeframe="4h")
        regime_result = run_backtest(regime_strategy, regime_data, bar_interval_sec=14400)

        start_str = pd.Timestamp(t_start, unit='ms', tz='UTC').strftime('%Y-%m-%d')
        end_str   = pd.Timestamp(t_end,   unit='ms', tz='UTC').strftime('%Y-%m-%d')
        period_str = f"{start_str} → {end_str}"

        print(f"  {regime:<22}  {period_str:<28}  "
              f"{regime_result.sharpe:+7.3f}  "
              f"{regime_result.max_drawdown_pct:5.1f}%  "
              f"{regime_result.num_trades:6d}  "
              f"{regime_result.total_return_pct:+7.2f}%")

def run_stress_fees(data, timeframe, split_name):
    print("\n" + "=" * 60)
    print("  FEE STRESS TEST (Alpha vs Costs Decay)")
    print("=" * 60)
    
    base_maker = prepare.MAKER_FEE
    base_taker = prepare.TAKER_FEE
    
    multipliers = [1.0, 1.5, 2.0, 3.0]
    for m in multipliers:
        prepare.MAKER_FEE = base_maker * m
        prepare.TAKER_FEE = base_taker * m
        print(f"\nEvaluating with {m}x fees (Taker: {prepare.TAKER_FEE*10000:.1f} bps):")
        
        strat = Strategy(timeframe=timeframe)
        if hasattr(strat, "pre_calculate_signals"):
            strat.pre_calculate_signals(data, split_name=split_name)
        res = run_backtest(strat, data, bar_interval_sec=TIMEFRAME_SECS[timeframe])
        print(f"  Sharpe: {res.sharpe:.3f} | Return: {res.total_return_pct:.2f}% | "
              f"Trades: {res.num_trades} | Profit Factor: {res.profit_factor:.2f}")
              
    prepare.MAKER_FEE = base_maker
    prepare.TAKER_FEE = base_taker

def run_capacity(data, timeframe, split_name):
    print("\n" + "=" * 60)
    print("  CAPACITY ANALYSIS (Liquidity Scaling)")
    print("=" * 60)
    
    base_capital = prepare.INITIAL_CAPITAL
    base_slippage = prepare.SLIPPAGE_BPS
    
    capitals = [10_000, 100_000, 1_000_000, 10_000_000]
    
    for cap in capitals:
        prepare.INITIAL_CAPITAL = cap
        # Dynamically bump slippage up 0.5 bps per $100k deployed as a rough market impact heuristic
        penalty = (cap / 100_000) * 0.5 
        prepare.SLIPPAGE_BPS = max(1.0, base_slippage + penalty)
        
        print(f"\nCapital Size: ${cap:,.0f} (Slippage: {prepare.SLIPPAGE_BPS:.1f} bps):")
        strat = Strategy(timeframe=timeframe)
        if hasattr(strat, "pre_calculate_signals"):
            strat.pre_calculate_signals(data, split_name=split_name)
        res = run_backtest(strat, data, bar_interval_sec=TIMEFRAME_SECS[timeframe])
        print(f"  Sharpe: {res.sharpe:.3f} | Return: {res.total_return_pct:.2f}% | "
              f"DD: {res.max_drawdown_pct:.1f}%")
              
    prepare.INITIAL_CAPITAL = base_capital
    prepare.SLIPPAGE_BPS = base_slippage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Backtest Runner")
    parser.add_argument("--timeframe", type=str, default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--stress-fees", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--oos", action="store_true", help="Run firmly on strictly unseen 2025 quarantine data")
    parser.add_argument("--2026q1", dest="oos_2026q1", action="store_true",
                        help="Run on Jan–Mar 2026 (clean post-research-loop OOS)")
    parser.add_argument("--holdout", action="store_true",
                        help="FINAL SIGN-OFF ONLY: run on Oct–Dec 2025 holdout (NEVER re-iterate against this)")
    parser.add_argument("--description", type=str, default="Auto-research iteration", help="Description for results.tsv")
    parser.add_argument("--leverage", type=float, default=1.0,
                        help="Leverage multiplier applied to all position sizes (default: 1.0 = no leverage)")
    parser.add_argument("--slippage-bps", type=float, default=None,
                        help="Override prepare.SLIPPAGE_BPS for this run only (audit knob)")
    parser.add_argument("--taker-fee-bps", type=float, default=None,
                        help="Override prepare.TAKER_FEE for this run only (audit knob)")
    parser.add_argument("--no-log", action="store_true",
                        help="Do not append result to results.tsv (audit / re-audit runs)")
    parser.add_argument("--label", type=str, default=None,
                        help="Print this label in the results header (audit runs)")

    args = parser.parse_args()

    # Audit overrides — applied BEFORE the strategy and engine see anything.
    # These mutate prepare.* module state for the duration of this process only.
    # When --no-log is set, results.tsv is left alone so audits don't pollute the leaderboard.
    if args.leverage != 1.0:
        print(f"LEVERAGE: {args.leverage}x position sizes — returns and drawdowns scale proportionally")
        # Raise MAX_LEVERAGE ceiling so the engine doesn't block larger positions
        prepare.MAX_LEVERAGE = max(prepare.MAX_LEVERAGE, int(args.leverage * 5))
    if args.slippage_bps is not None:
        print(f"AUDIT OVERRIDE: SLIPPAGE_BPS {prepare.SLIPPAGE_BPS} -> {args.slippage_bps}")
        prepare.SLIPPAGE_BPS = args.slippage_bps
    if args.taker_fee_bps is not None:
        new_taker = args.taker_fee_bps / 10000.0
        print(f"AUDIT OVERRIDE: TAKER_FEE {prepare.TAKER_FEE} -> {new_taker}")
        prepare.TAKER_FEE = new_taker

    # Set OS alarm: 4H needs extra budget for regime breakdown (5 backtests)
    if args.timeframe == "4h":
        sig.alarm(TIME_BUDGET * 6 + 120)  # ~1920s for 4H + 4 regime backtests
    else:
        sig.alarm(1800)              # Absolute Budget for Dual-XGB Research

    # ── Holdout friction gate ─────────────────────────────────────────────────
    # The holdout split (Oct–Dec 2025) must NEVER be used during research iteration.
    # It is reserved for a single final sign-off run before live deployment.
    # If you are running this during a research loop, do NOT use --holdout.
    if args.holdout:
        print("\n" + "!" * 60)
        print("  HOLDOUT GATE — READ BEFORE CONTINUING")
        print("  You are about to run on Oct–Dec 2025, the ONLY truly unseen")
        print("  data remaining. This split must be used AT MOST ONCE for the")
        print("  final production sign-off. Re-iterating against it destroys")
        print("  its independence and invalidates the read.")
        print("!" * 60)
        confirm = input("\n  Type FINAL-SIGN-OFF to proceed, or anything else to abort: ").strip()
        if confirm != "FINAL-SIGN-OFF":
            print("  Aborted. Holdout preserved.")
            raise SystemExit(0)
        print()

    t_start = time.time()

    if args.timeframe == "4h":
        print("Mode: 4H ROBUSTNESS")
        # Ensure 4H fee overrides are active (0.15% RT standard)
        prepare.TAKER_FEE = 0.00075
        prepare.MAKER_FEE = 0.00075
        prepare.SLIPPAGE_BPS = 0.0
        if args.holdout:
            split_name = "holdout"
        elif args.oos_2026q1:
            split_name = "2026q1"
        elif args.oos:
            split_name = "oos"
        else:
            split_name = "robustness"
        data = load_data(split_name, resample_4h=True)
    elif args.timeframe == "15m":
        print("Mode: 15M VALIDATION")
        if args.holdout:
            split_name = "holdout_15m"
        elif args.oos_2026q1:
            split_name = "2026q1_15m"
        elif args.oos:
            split_name = "oos_15m"
        else:
            split_name = "val_15m"
        data = load_data(split_name)
    else:
        print("Mode: 1H VALIDATION")
        if args.holdout:
            split_name = "holdout"
        elif args.oos_2026q1:
            split_name = "2026q1"
        elif args.oos:
            split_name = "oos"
        else:
            split_name = "val"
        data = load_data(split_name)
    
    print(f"Loaded {sum(len(df) for df in data.values())} bars across {len(data)} symbols")
    
    strategy = Strategy(timeframe=args.timeframe)
    if args.leverage != 1.0:
        strategy._leverage_mult = args.leverage
    if hasattr(strategy, 'pre_calculate_signals'):
        strategy.pre_calculate_signals(data, split_name=split_name)
        
    bar_sec = TIMEFRAME_SECS[args.timeframe]
    result = run_backtest(strategy, data, bar_interval_sec=bar_sec)
    score = compute_score(result)
    t_end = time.time()
    
    print("\n" + "=" * 60)
    header = f"  BASELINE RESULTS ({args.timeframe.upper()})"
    if args.label:
        header += f"  [{args.label}]"
    print(header)
    print("=" * 60)
    if args.slippage_bps is not None or args.taker_fee_bps is not None:
        print(f"  audit overrides: slippage_bps={prepare.SLIPPAGE_BPS} taker_fee={prepare.TAKER_FEE}")
    print(f"score:              {score:.6f}")
    print(f"sharpe:             {result.sharpe:.6f}")
    print(f"total_return_pct:   {result.total_return_pct:.6f}")
    print(f"max_drawdown_pct:   {result.max_drawdown_pct:.6f}")
    print(f"num_trades:         {result.num_trades}")
    print(f"win_rate_pct:       {result.win_rate_pct:.6f}")
    print(f"profit_factor:      {result.profit_factor:.6f}")
    print(f"annual_turnover:    {result.annual_turnover:.2f}")
    print(f"backtest_seconds:   {result.backtest_seconds:.1f}")
    print(f"total_seconds:      {t_end - t_start:.1f}")
    print(f"bars_processed:     {result.bars_processed}/{result.total_bars}")
    print(f"timed_out:          {result.timed_out}")
    print_benchmark_audit(result, len(data))

    if result.timed_out:
        print("\nTIMEOUT: prepare.run_backtest hit the internal time budget. Refusing to log a partial result.")
        raise SystemExit(2)
    
    if args.timeframe == "4h" and not args.oos:
        print_regimes(result, data)
        
    if args.stress_fees:
        run_stress_fees(data, args.timeframe, split_name)
        
    if args.capacity:
        run_capacity(data, args.timeframe, split_name)

    if args.no_log:
        print("\n--no-log: skipping results.tsv append (audit run).")
        raise SystemExit(0)

    # Autonomous Logging to results.tsv
    try:
        df_res = pd.read_csv("results.tsv", sep="\t")
        df_res.columns = df_res.columns.str.strip()
        last_exp = df_res[df_res["commit"].str.strip().str.match(r"exp\d+")]["commit"].str.strip().str.extract(r"exp(\d+)").dropna().astype(int).max().iloc[0]
        new_exp_id = f"exp{last_exp + 1}"
    except Exception:
        new_exp_id = "exp1"

    desc = getattr(args, 'description', 'Auto-research iteration')
    status = classify_run_status(result, score, args.timeframe)
    new_row = {
        "commit": new_exp_id,
        "score": f"{score:.3f}",
        "sharpe": f"{result.sharpe:.3f}",
        "max_dd": f"{result.max_drawdown_pct:.2f}",
        "status": status,
        "description": desc
    }
    
    with open("results.tsv", "a") as f:
        f.write("\t".join([str(new_row[c]) for c in ["commit", "score", "sharpe", "max_dd", "status", "description"]]) + "\n")
    print(f"\nLogged to results.tsv as {new_exp_id} [{status}]")
