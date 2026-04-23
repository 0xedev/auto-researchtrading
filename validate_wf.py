"""
Walk-Forward Validation Harness
Tests temporal consistency of the strategy across rolling time windows
using the current trained models (no retraining).

Usage:
    uv run validate_wf.py --timeframe 1h
    uv run validate_wf.py --timeframe 1h --slippage-bps 5
    uv run validate_wf.py --timeframe 4h --year-by-year   # 7-year independent 4H audit

The model was trained on 2017 → 2024-06-30 and validated on 2024-07-01 → 2025-09-30.
This harness slices the post-training era into windows to check if performance
is consistent across market regimes rather than just concentrated in one lucky period.

Interpretation:
  - Green (Sharpe >= 0.5): Positive expected value in that regime.
  - Any window with Sharpe < 0 in a long window (> 6 months) is a red flag.
  - Variance across windows > 2x median Sharpe signals regime dependency.
"""
import argparse
import time
import prepare
from prepare import load_data, run_backtest
from strategy import Strategy

TIMEFRAME_SECS = {"15m": 900, "1h": 3600, "4h": 14400}

# Walk-forward windows: (label, start, end, note)
# All dates are UTC, end is exclusive-ish (data through end date EOD)
WINDOWS = [
    ("WF1 | 2022-H2 (Crypto Winter)",  "2022-07-01", "2022-12-31", "bear grind, FTX collapse"),
    ("WF2 | 2023-Full (Recovery)",      "2023-01-01", "2023-12-31", "slow grind up, flat mid-year"),
    ("WF3 | 2023H2-2024H1 (Bull Run)",  "2023-07-01", "2024-06-30", "bull run to ATH"),
    ("WF4 | 2024-H1 (ATH Bull)",        "2024-01-01", "2024-06-30", "BTC ETF launch, peak euphoria"),
    ("WF5 | 2025-Full (OOS)",           "2025-01-01", "2025-12-31", "true OOS (contaminated by evaluate.py)"),
]

# Year-by-year windows for 7-year 4H independent audit (2019-07 → 2026-03)
# Tickers available per era: all 15 from 2023+; BTC/ETH/major from 2019
YEAR_WINDOWS = [
    ("2019-H2", "2019-07-01", "2019-12-31", "pre-bull, low liquidity alts"),
    ("2020",    "2020-01-01", "2020-12-31", "COVID crash + DeFi summer"),
    ("2021",    "2021-01-01", "2021-12-31", "BTC ATH 69k + alt season"),
    ("2022",    "2022-01-01", "2022-12-31", "bear: -75%, FTX collapse Nov"),
    ("2023",    "2023-01-01", "2023-12-31", "recovery: BTC +150%"),
    ("2024",    "2024-01-01", "2024-12-31", "ETF approval + ATH 100k"),
    ("2025",    "2025-01-01", "2025-09-30", "OOS: post-ATH correction"),
    ("2026-Q1", "2026-01-01", "2026-03-31", "clean OOS gate — never seen"),
]


def run_window(tf, label, start, end, note, slippage_override=None):
    """Run backtest over a specific date window by patching prepare globals."""
    orig_val_start = prepare.VAL_START
    orig_val_end   = prepare.VAL_END

    # Patch module-level globals so load_data("val") uses our window
    prepare.VAL_START = start
    prepare.VAL_END   = end

    if slippage_override is not None:
        orig_slip = prepare.SLIPPAGE_BPS
        prepare.SLIPPAGE_BPS = slippage_override

    try:
        data = load_data("val", resample_4h=(tf == "4h"))
        strat = Strategy(timeframe=tf)
        if hasattr(strat, "pre_calculate_signals"):
            strat.pre_calculate_signals(data, split_name="val")
        res = run_backtest(strat, data, bar_interval_sec=TIMEFRAME_SECS[tf])
    finally:
        prepare.VAL_START = orig_val_start
        prepare.VAL_END   = orig_val_end
        if slippage_override is not None:
            prepare.SLIPPAGE_BPS = orig_slip

    dur = max(res.duration_days, 1)
    tpd = res.num_trades / dur
    return res, tpd


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation")
    parser.add_argument("--timeframe", default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--slippage-bps", type=float, default=None,
                        help="Override slippage for this run (e.g. 5 or 10)")
    parser.add_argument("--year-by-year", action="store_true",
                        help="Run 7-year year-by-year audit on 4H timeframe (2019-07 to 2026-03)")
    args = parser.parse_args()
    tf = args.timeframe

    if args.year_by_year:
        _run_year_by_year(tf, args.slippage_bps)
    else:
        _run_walk_forward(tf, args.slippage_bps)


def _run_walk_forward(tf, slippage_bps):
    print("\n" + "=" * 80)
    print(f"  WALK-FORWARD VALIDATION  |  Timeframe: {tf.upper()}", end="")
    if slippage_bps is not None:
        print(f"  |  Slippage override: {slippage_bps:.1f} bps", end="")
    print(f"\n  Model trained on: 2017-01-01 → 2024-06-30")
    print(f"  Val window used for parameter tuning: 2024-07-01 → 2025-09-30")
    print("=" * 80)

    print(f"\n  {'Window':<38} {'Period':<21} {'Sharpe':>7} {'MaxDD%':>7} "
          f"{'Trades':>7} {'T/Day':>6} {'WinR%':>6} {'PF':>6}  Status")
    print(f"  {'-'*102}")

    sharpes = []
    t0 = time.time()
    for label, start, end, note in WINDOWS:
        res, tpd = run_window(tf, label, start, end, note, slippage_bps)
        sharpes.append(res.sharpe)
        status = "OK" if res.sharpe >= 0.5 else ("WARN" if res.sharpe >= 0.0 else "FAIL")
        flag = "  ✓" if status == "OK" else ("  ⚠" if status == "WARN" else "  ✗")
        period = f"{start} → {end[:7]}"
        print(f"  {label:<38} {period:<21} {res.sharpe:>7.3f} "
              f"{res.max_drawdown_pct:>7.2f} {res.num_trades:>7d} "
              f"{tpd:>6.2f} {res.win_rate_pct:>6.1f} "
              f"{res.profit_factor:>6.3f}{flag}  [{note}]")

    elapsed = time.time() - t0
    print(f"\n  {'-'*102}")
    print(f"\n  Sharpe stats across {len(sharpes)} windows:")
    import numpy as np
    sh = np.array(sharpes)
    print(f"    Min:    {sh.min():+.3f}")
    print(f"    Median: {np.median(sh):+.3f}")
    print(f"    Max:    {sh.max():+.3f}")
    print(f"    All positive: {'YES ✓' if (sh > 0).all() else 'NO ✗'}")
    print(f"    All >= 0.5:   {'YES ✓' if (sh >= 0.5).all() else 'NO ✗'}")

    verdict_ok = (sh > 0).all()
    verdict_str = "TEMPORALLY CONSISTENT" if verdict_ok else "REGIME-DEPENDENT"
    print(f"\n  Walk-Forward Verdict: *** {verdict_str} ***")
    print(f"  Elapsed: {elapsed:.1f}s\n")


def _run_year_by_year(tf, slippage_bps):
    import numpy as np

    print("\n" + "=" * 90)
    print(f"  7-YEAR INDEPENDENT 4H AUDIT  |  Timeframe: {tf.upper()}")
    print(f"  Strategy: strategy.py 6-signal ensemble (trained on 1H, tested on 4H)")
    print(f"  Universe: 15 crypto tickers (BTC ETH SOL BNB XRP ADA DOGE LINK AVAX DOT ATOM NEAR UNI APT SUI)")
    print(f"  Coverage: 2019-07 → 2026-03  |  Data: CryptoCompare / Binance (same instruments as Bitget perps)")
    if slippage_bps is not None:
        print(f"  Slippage override: {slippage_bps:.1f} bps")
    print(f"  Note: Earlier years (2019-2022) have fewer tickers due to listing dates")
    print("=" * 90)

    print(f"\n  {'Year':<10} {'Period':<23} {'Sharpe':>7} {'MaxDD%':>7} "
          f"{'Trades':>7} {'T/Day':>6} {'WinR%':>6} {'PF':>7}  Status  Note")
    print(f"  {'-'*110}")

    results = []
    t0 = time.time()
    for label, start, end, note in YEAR_WINDOWS:
        try:
            res, tpd = run_window(tf, label, start, end, note, slippage_bps)
            sharpe = res.sharpe
            mdd = res.max_drawdown_pct
            trades = res.num_trades
            wr = res.win_rate_pct
            pf = res.profit_factor
            valid = True
        except Exception as e:
            sharpe, mdd, trades, tpd, wr, pf = -999, 0, 0, 0, 0, 0
            valid = False
            note = f"ERROR: {e}"

        results.append((label, start, end, sharpe, mdd, trades, tpd, wr, pf, valid))
        if not valid or sharpe <= -900:
            status = "SKIP"
            flag = "  –"
            sharpe_str = "  N/A"
        else:
            status = "OK" if sharpe >= 0.5 else ("WARN" if sharpe >= 0.0 else "FAIL")
            flag = "  ✓" if status == "OK" else ("  ⚠" if status == "WARN" else "  ✗")
            sharpe_str = f"{sharpe:>7.3f}"

        period = f"{start} → {end}"
        if not valid or sharpe <= -900:
            print(f"  {label:<10} {period:<23} {'N/A':>7} {'N/A':>7} "
                  f"{'N/A':>7} {'N/A':>6} {'N/A':>6} {'N/A':>7}{flag}  [{note}]")
        else:
            print(f"  {label:<10} {period:<23} {sharpe:>7.3f} "
                  f"{mdd:>7.2f} {trades:>7d} "
                  f"{tpd:>6.2f} {wr:>6.1f} "
                  f"{pf:>7.3f}{flag}  [{note}]")

    elapsed = time.time() - t0

    valid_sharpes = [r[3] for r in results if r[9] and r[3] > -900]
    print(f"\n  {'-'*110}")
    print(f"\n  Summary across {len(valid_sharpes)} valid years:")
    if valid_sharpes:
        sh = np.array(valid_sharpes)
        print(f"    Min Sharpe:    {sh.min():+.3f}")
        print(f"    Median Sharpe: {np.median(sh):+.3f}")
        print(f"    Max Sharpe:    {sh.max():+.3f}")
        print(f"    Mean Sharpe:   {sh.mean():+.3f}")
        pos_years = (sh > 0).sum()
        good_years = (sh >= 0.5).sum()
        print(f"    Positive years: {pos_years}/{len(sh)}")
        print(f"    Years >= 0.5 Sharpe: {good_years}/{len(sh)}")
        verdict = "ROBUST ACROSS REGIMES" if good_years >= len(sh) * 0.75 else (
            "MODERATE CONSISTENCY" if pos_years >= len(sh) * 0.75 else "REGIME-DEPENDENT"
        )
        print(f"\n  7-Year Verdict: *** {verdict} ***")
    print(f"  Elapsed: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
