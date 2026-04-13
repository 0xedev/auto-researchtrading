"""
Feature Drift Monitor (PSI-based)
Detects distribution shift between the training window and recent live data.

Population Stability Index (PSI) thresholds:
  < 0.10  → Stable, no action
  0.10–0.20 → Slight shift, monitor
  > 0.20  → Significant shift — consider retraining

Usage:
    uv run drift_monitor.py                          # train vs most recent 30 days
    uv run drift_monitor.py --recent-days 90         # wider recent window
    uv run drift_monitor.py --ref-split val          # val vs recent (post-val drift)
    uv run drift_monitor.py --symbols BTC ETH SOL    # specific symbols only

Notes:
  - Parquets store raw OHLCV. Features are computed here via calculate_features().
    The recent window loads extra warmup bars (30d) so indicators are properly warmed.
  - PSI uses 10-bucket equal-frequency binning on the reference distribution.
  - A single high-PSI feature may reflect genuine regime change or a data pipeline
    issue — cross-check with raw prices before acting.
"""
import os
import argparse
import numpy as np
import pandas as pd
import prepare
from prepare import load_data, calculate_features, FEATURE_COLS, DATA_DIR


N_BINS   = 10
PSI_OK   = 0.10
PSI_WARN = 0.20

# Features to skip (categorical, not meaningful for PSI)
SKIP_COLS = {"asset_class"}


def compute_psi(ref: np.ndarray, test: np.ndarray, n_bins: int = N_BINS) -> float:
    """Compute Population Stability Index between reference and test arrays."""
    ref  = ref[np.isfinite(ref)]
    test = test[np.isfinite(test)]
    if len(ref) < 20 or len(test) < 20:
        return np.nan

    # Equal-frequency bins from reference distribution
    quantiles  = np.linspace(0, 100, n_bins + 1)
    bin_edges  = np.unique(np.percentile(ref, quantiles))
    if len(bin_edges) < 3:
        return np.nan  # degenerate (near-constant) feature

    ref_counts  = np.histogram(ref,  bins=bin_edges)[0].astype(float)
    test_counts = np.histogram(test, bins=bin_edges)[0].astype(float)

    # Smooth zeros (Laplace smoothing)
    ref_counts  = np.where(ref_counts  == 0, 0.5, ref_counts)
    test_counts = np.where(test_counts == 0, 0.5, test_counts)

    ref_pct  = ref_counts  / ref_counts.sum()
    test_pct = test_counts / test_counts.sum()

    return float(np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct)))


def compute_features_for(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """Compute FEATURE_COLS on a raw OHLCV DataFrame via prepare.calculate_features.

    calculate_features() needs market_vol / market_ret / rel_* columns which are
    normally computed cross-asset in prepare_dataset. For standalone drift checks
    we fill these with NaN (they'll be skipped in PSI if both ref and test are NaN).
    """
    feat_df = calculate_features(df, timeframe=timeframe)
    # Cross-asset features that require the full symbol universe — fill as NaN
    cross_asset = ["market_vol", "market_ret", "rel_ret_1h", "rel_ret_4h", "rel_bb_width"]
    for col in cross_asset:
        if col not in feat_df.columns:
            feat_df[col] = np.nan
    return feat_df


def load_symbol_raw(sym: str, timeframe: str = "1h") -> pd.DataFrame | None:
    """Load full raw parquet for a symbol."""
    suffix = "_15m.parquet" if timeframe == "15m" else "_1h.parquet"
    path = os.path.join(DATA_DIR, f"{sym}{suffix}")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description="Feature Drift Monitor (PSI)")
    parser.add_argument("--recent-days", type=int, default=30,
                        help="Days of recent data to compare against reference (default: 30)")
    parser.add_argument("--ref-split", default="train",
                        choices=["train", "val", "robustness"],
                        help="Reference split to compare against (default: train)")
    parser.add_argument("--timeframe", default="1h", choices=["1h", "15m"])
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Symbols to analyse (default: all available)")
    parser.add_argument("--threshold", type=float, default=PSI_WARN,
                        help=f"PSI alert threshold (default: {PSI_WARN})")
    args = parser.parse_args()
    tf = args.timeframe

    print("\n" + "=" * 72)
    print(f"  FEATURE DRIFT MONITOR  |  PSI-based distribution shift detection")
    print(f"  Reference: '{args.ref_split}' split   |   Recent window: {args.recent_days}d")
    print(f"  Alert threshold: PSI > {args.threshold:.2f}")
    print("=" * 72)

    # ── 1. Reference split ────────────────────────────────────────────────────
    print(f"\nLoading reference split '{args.ref_split}'…")
    ref_raw = load_data(args.ref_split, resample_4h=False)
    symbols = args.symbols if args.symbols else list(ref_raw.keys())
    symbols = [s for s in symbols if s in ref_raw]
    if not symbols:
        print("  [ERROR] No matching symbols found in reference data.")
        return

    print(f"Computing reference features for {len(symbols)} symbols…")
    ref_features = {}
    for sym in symbols:
        try:
            feat = compute_features_for(ref_raw[sym], timeframe=tf)
            ref_features[sym] = feat
        except Exception as e:
            print(f"  [WARN] {sym}: {e}")

    # ── 2. Recent data ────────────────────────────────────────────────────────
    # Load extra warmup bars (30d) so rolling indicators are meaningful
    warmup_ms = int(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=args.recent_days + 30)).timestamp() * 1000
    )
    cutoff_ms = int(
        (pd.Timestamp.utcnow() - pd.Timedelta(days=args.recent_days)).timestamp() * 1000
    )
    print(f"Loading recent {args.recent_days}d from parquet cache (+ 30d warmup)…")
    recent_features = {}
    for sym in symbols:
        df_full = load_symbol_raw(sym, timeframe=tf)
        if df_full is None:
            continue
        df_warm = df_full[df_full["timestamp"] >= warmup_ms].reset_index(drop=True)
        if len(df_warm) < 50:
            print(f"  [WARN] {sym}: insufficient recent data ({len(df_warm)} bars)")
            continue
        try:
            feat_warm = compute_features_for(df_warm, timeframe=tf)
            # Trim to the actual recent window (post-warmup)
            feat_recent = feat_warm[feat_warm.index >= feat_warm.index[
                df_warm["timestamp"].searchsorted(cutoff_ms)
            ]] if "timestamp" not in feat_warm.columns else \
                feat_warm[feat_warm["timestamp"] >= cutoff_ms]
            # Fallback: use last N rows
            if len(feat_recent) < 24:
                bars_needed = args.recent_days * (24 if tf == "1h" else 96)
                feat_recent = feat_warm.tail(bars_needed)
            if len(feat_recent) >= 24:
                recent_features[sym] = feat_recent
        except Exception as e:
            print(f"  [WARN] {sym}: feature build failed: {e}")

    if not recent_features:
        print("\n  [ERROR] Could not build features for recent data.")
        print("  Ensure parquet cache is up-to-date: uv run prepare.py\n")
        return

    print(f"  Reference symbols: {len(ref_features)}  |  Recent symbols: {len(recent_features)}")

    # ── 3. PSI per feature ────────────────────────────────────────────────────
    feat_cols = [c for c in FEATURE_COLS if c not in SKIP_COLS]
    psi_results = []  # (feature, median_psi, max_psi, n_symbols)

    for feat in feat_cols:
        psi_vals = []
        for sym in symbols:
            if sym not in ref_features or sym not in recent_features:
                continue
            ref_df  = ref_features[sym]
            rec_df  = recent_features[sym]
            if feat not in ref_df.columns or feat not in rec_df.columns:
                continue
            ref_arr = ref_df[feat].dropna().values
            rec_arr = rec_df[feat].dropna().values
            psi = compute_psi(ref_arr, rec_arr)
            if np.isfinite(psi):
                psi_vals.append(psi)
        if psi_vals:
            psi_results.append((feat, float(np.median(psi_vals)), float(np.max(psi_vals)), len(psi_vals)))

    if not psi_results:
        print("  [ERROR] No PSI values computed. Check FEATURE_COLS availability in parquets.")
        return

    psi_results.sort(key=lambda x: x[1], reverse=True)
    alerts   = [r for r in psi_results if r[1] >= args.threshold]
    warnings = [r for r in psi_results if PSI_OK <= r[1] < args.threshold]
    stable   = [r for r in psi_results if r[1] < PSI_OK]

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print(f"\n  {'Feature':<22} {'Med PSI':>9} {'Max PSI':>9} {'N_syms':>7}  Status")
    print(f"  {'-'*65}")

    def prow(r, status):
        print(f"  {r[0]:<22} {r[1]:>9.4f} {r[2]:>9.4f} {r[3]:>7d}  {status}")

    for r in alerts:
        prow(r, f"ALERT  ⚠")
    for r in warnings:
        prow(r, "MONITOR")
    for r in stable:
        prow(r, "stable")

    print(f"\n  {'─'*65}")
    print(f"  Checked {len(psi_results)} features across {len(recent_features)} symbols")
    print(f"  Stable  (< {PSI_OK:.2f}):   {len(stable)}")
    print(f"  Monitor ({PSI_OK:.2f}–{args.threshold:.2f}): {len(warnings)}")
    print(f"  ALERT   (>= {args.threshold:.2f}): {len(alerts)}")

    if alerts:
        names = ", ".join(r[0] for r in alerts)
        print(f"\n  HIGH-DRIFT FEATURES: {names}")
        print("  → Retrain or exclude these features before live deployment.")
        verdict = "DRIFT DETECTED — retrain recommended"
    elif warnings:
        verdict = "MARGINAL DRIFT — monitor daily"
    else:
        verdict = "STABLE — inputs consistent with training distribution"

    print(f"\n  Drift Verdict: *** {verdict} ***\n")


if __name__ == "__main__":
    main()
