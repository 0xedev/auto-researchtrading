"""
Robustness Evaluation Harness
Runs OOS, Fee Stress, Capacity, Regime Breakdown, and Monte Carlo
for the current loaded strategy/models.

Usage:
    uv run evaluate.py --timeframe 1h --label exp269
    uv run evaluate.py --timeframe 1h --label exp268 --mc-sims 1000
"""
import argparse
import time
import numpy as np
import pandas as pd
import prepare
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


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_strategy(timeframe):
    strat = Strategy(timeframe=timeframe)
    return strat


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


# ─────────────────────────────────────────────────────────────
# 1. OOS Test
# ─────────────────────────────────────────────────────────────

def run_oos(timeframe, label):
    print(f"\n{'='*60}")
    print(f"  OOS TEST  [{label}]  (2025-01-01 → 2025-12-31)")
    print(f"{'='*60}")
    split = "oos_15m" if timeframe == "15m" else "oos"
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
    return res, data, score


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

def detect_regimes(data, anchor="BTC"):
    ref_df = data.get(anchor, next(iter(data.values())))
    close  = ref_df["close"].reset_index(drop=True)
    ts     = ref_df["timestamp"].reset_index(drop=True)
    ema200 = close.ewm(span=200, adjust=False).mean()
    slope  = ema200.pct_change(20)
    vol30  = close.pct_change().rolling(30).std()
    volpct = vol30.rank(pct=True)

    def classify(s, v):
        if pd.isna(s) or pd.isna(v): return "Initializing"
        if s > 0.01  and v < 0.60:  return "Bull (Low Vol)"
        if s > 0.01  and v >= 0.60: return "Bull (High Vol)"
        if s < -0.01 and v >= 0.50: return "Bear (Panic)"
        if s < -0.01 and v < 0.50:  return "Bear (Grind)"
        return "Sideways"

    regimes = [classify(slope.iloc[i], volpct.iloc[i]) for i in range(len(close))]
    segs, prev, t0 = [], None, None
    for i, r in enumerate(regimes):
        if r != prev:
            if prev and prev != "Initializing":
                segs.append((prev, t0, ts.iloc[i-1]))
            t0, prev = ts.iloc[i], r
    if prev and prev != "Initializing":
        segs.append((prev, t0, ts.iloc[-1]))
    return segs


def run_regime_breakdown(timeframe, split, data, label):
    print(f"\n{'='*60}")
    print(f"  REGIME BREAKDOWN  [{label}]")
    print(f"{'='*60}")
    segs = detect_regimes(data)
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
    parser.add_argument("--split",      default="oos", help="OOS split name")
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
        split = "holdout_15m" if tf == "15m" else "holdout"
        print()

    print(f"\n{'#'*60}")
    print(f"  ROBUSTNESS SUITE: {lbl.upper()}  |  Timeframe: {tf.upper()}")
    if args.holdout:
        print(f"  *** HOLDOUT MODE: Oct–Dec 2025 (FINAL SIGN-OFF) ***")
    print(f"{'#'*60}")
    t0 = time.time()

    # OOS (or holdout)
    if args.holdout:
        # Run directly on the holdout split, bypassing run_oos() which hardcodes "oos"
        holdout_split = "holdout_15m" if tf == "15m" else "holdout"
        print(f"\n{'='*60}")
        print(f"  HOLDOUT TEST  [{lbl}]  ({prepare.HOLDOUT_START} → {prepare.HOLDOUT_END})")
        print(f"{'='*60}")
        oos_res, oos_data = _load_and_run(tf, holdout_split)
        dur = oos_res.duration_days if oos_res.duration_days > 0 else 1
        trades_per_day = oos_res.num_trades / dur
        oos_score = compute_score(oos_res)
        print(f"  Score:          {oos_score:.4f}")
        print(f"  Sharpe:         {oos_res.sharpe:.4f}")
        print(f"  Return:         {oos_res.total_return_pct:.2f}%")
        print(f"  Max DD:         {oos_res.max_drawdown_pct:.2f}%")
        print(f"  Trades:         {oos_res.num_trades}  ({trades_per_day:.2f}/day)")
        print(f"  Win Rate:       {oos_res.win_rate_pct:.2f}%")
        print(f"  Profit Factor:  {oos_res.profit_factor:.3f}")
    else:
        oos_res, oos_data, oos_score = run_oos(tf, lbl)

    # Fee Stress
    run_fee_stress(tf, split, oos_data, lbl)

    # Capacity
    run_capacity(tf, split, oos_data, lbl)

    # Regime Breakdown
    if not args.skip_regime:
        run_regime_breakdown(tf, split, oos_data, lbl)

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
    print(f"  OOS MaxDD:    {oos_res.max_drawdown_pct:.2f}%")
    if mc_result:
        print(f"  MC p50 Sharpe: {mc_result['p50']:.4f}")
        print(f"  MC Prob≥{BENCHMARKS['sharpe']:.1f}: {mc_result['prob_beat']:.1f}%")
    print(f"  Total time:    {total:.1f}s")
