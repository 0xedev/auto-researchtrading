"""
Walk-Forward Validation Harness
Tests temporal consistency of the strategy across rolling time windows
using the current trained models (no retraining).

Usage:
    uv run validate_wf.py --timeframe 1h
    uv run validate_wf.py --timeframe 1h --slippage-bps 5

The model was trained on 2017 → 2022-06-30 and validated on 2022-07-01 → 2024-06-30.
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
    args = parser.parse_args()
    tf = args.timeframe

    print("\n" + "=" * 80)
    print(f"  WALK-FORWARD VALIDATION  |  Timeframe: {tf.upper()}", end="")
    if args.slippage_bps is not None:
        print(f"  |  Slippage override: {args.slippage_bps:.1f} bps", end="")
    print(f"\n  Model trained on: 2017-01-01 → 2022-06-30")
    print(f"  Val window used for parameter tuning: 2022-07-01 → 2024-06-30")
    print("=" * 80)

    print(f"\n  {'Window':<38} {'Period':<21} {'Sharpe':>7} {'MaxDD%':>7} "
          f"{'Trades':>7} {'T/Day':>6} {'WinR%':>6} {'PF':>6}  Status")
    print(f"  {'-'*102}")

    sharpes = []
    t0 = time.time()
    for label, start, end, note in WINDOWS:
        res, tpd = run_window(tf, label, start, end, note, args.slippage_bps)
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


if __name__ == "__main__":
    main()
