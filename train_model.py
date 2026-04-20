import pandas as pd
import numpy as np
import argparse
import joblib
import os
import time
import json
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed
from hmmlearn import hmm

from prepare import (
    calculate_features, load_data, prepare_dataset,
    get_n_trees_depth, run_backtest, compute_score,
    TRAIN_START, TEST_END, VAL_START, VAL_END,
    SYMBOLS, SYMBOLS_15M, ASSET_CLASS, SYMBOL_TRAIN_START, FEATURE_COLS
)

def build_model(n_trees, depth, learning_rate):
    # early_stopping_rounds set in constructor for sklearn wrapper (XGB 1.6+)
    return xgb.XGBClassifier(
        n_estimators=n_trees,
        max_depth=depth,
        learning_rate=learning_rate,
        n_jobs=-1,
        tree_method='hist',
        colsample_bytree=0.7,
        subsample=0.8,
        random_state=42,
        early_stopping_rounds=20 
    )


def _output_dir(model_set: str | None) -> Path:
    base = Path("models")
    if model_set:
        base = base / model_set
    base.mkdir(parents=True, exist_ok=True)
    return base


def _save_model(model, out_dir: Path, filename: str) -> None:
    model.save_model(str(out_dir / filename))


def _write_metadata(out_dir: Path, metadata: dict) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _fit_short_calibrator(mode: str, raw_probs: np.ndarray, labels: np.ndarray):
    raw_probs = np.asarray(raw_probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if mode == "raw":
        return None
    if mode == "platt":
        calibrator = LogisticRegression(random_state=42, max_iter=1000)
        calibrator.fit(raw_probs.reshape(-1, 1), labels)
        return calibrator
    if mode == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, labels)
        return calibrator
    return None


def _build_bear_short_target(default_target: np.ndarray, sample_meta: pd.DataFrame, mode: str) -> np.ndarray:
    default_target = np.asarray(default_target, dtype=int)
    if mode == "default":
        return default_target

    meta = sample_meta.reset_index(drop=True).copy()
    bear_mask = meta["regime_family"].eq("bear").values
    if bear_mask.sum() == 0:
        return default_target

    rel_forward = pd.to_numeric(meta.get("relative_forward_ret", 0.0), errors="coerce").fillna(0.0).values
    scaled_forward = pd.to_numeric(meta.get("scaled_forward_ret", 0.0), errors="coerce").fillna(0.0).values
    sentiment_shock = pd.to_numeric(meta.get("context_sentiment_shock", 0.0), errors="coerce").fillna(0.0).values
    market_event = pd.to_numeric(meta.get("major_market_event_flag", 0.0), errors="coerce").fillna(0.0).values
    stress = pd.to_numeric(meta.get("cross_asset_stress", 0.0), errors="coerce").fillna(0.0).values

    positive_mask = bear_mask & (default_target == 1)
    if positive_mask.sum() < 200:
        return default_target

    if mode == "relative_tail":
        rel_q = np.quantile(rel_forward[positive_mask], 0.45)
        scaled_q = np.quantile(scaled_forward[positive_mask], 0.45)
        shock_q = np.quantile(sentiment_shock[positive_mask], 0.35)
        stress_q = np.quantile(stress[positive_mask], 0.60)
        enhanced = (
            positive_mask
            & (
                (rel_forward <= rel_q)
                | (scaled_forward <= scaled_q)
                | (market_event < 0)
                | (sentiment_shock <= shock_q)
                | (stress >= stress_q)
            )
        )
        return enhanced.astype(int)

    return default_target


def validate_directional_models(long_model, short_model, long_meta_model, short_meta_model, X_val, y_val, feature_cols):
    long_probs = long_model.predict_proba(X_val[feature_cols])[:, 1]
    short_probs = short_model.predict_proba(X_val[feature_cols])[:, 1]
    long_meta_probs = long_meta_model.predict_proba(X_val[feature_cols])[:, 1]
    short_meta_probs = short_meta_model.predict_proba(X_val[feature_cols])[:, 1]

    lead_pred = np.where(
        (long_probs > short_probs) & (long_probs > 0.5),
        1,
        np.where((short_probs > long_probs) & (short_probs > 0.5), 2, 0),
    )
    meta_probs = np.where(lead_pred == 1, long_meta_probs, np.where(lead_pred == 2, short_meta_probs, 0.0))

    is_trend_pred = lead_pred != 0
    base_wr = accuracy_score(y_val[is_trend_pred], lead_pred[is_trend_pred]) if any(is_trend_pred) else 0.0

    sniper_mask = is_trend_pred & (meta_probs > 0.65)
    sniper_wr = accuracy_score(y_val[sniper_mask], lead_pred[sniper_mask]) if any(sniper_mask) else 0.0
    return base_wr, sniper_wr


def train_macro_hmm():
    """Trains a 4-state Multivariate HMM on BTC, ETH, XAU, and SP500.
    Identifies hidden macro regimes for the strategy to adapt to.
    """
    print("Training Multivariate Macro HMM (Medallion Regime Engine)...")
    cutoff_ms = int(pd.Timestamp(VAL_START, tz="UTC").timestamp() * 1000)
    
    macro_assets = ["BTC", "ETH", "XAU", "SP500"]
    data_frames = {}
    
    for symbol in macro_assets:
        try:
            path = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "data", f"{symbol}_1h.parquet")
            df = pd.read_parquet(path)
            df = df[pd.to_numeric(df["timestamp"], errors="coerce") < cutoff_ms].copy()
            if df.empty:
                print(f"Warning: No pre-validation data left for macro asset {symbol}")
                continue
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_ret'].rolling(24).std()
            data_frames[symbol] = df[['timestamp', 'log_ret', 'volatility']].dropna()
        except Exception as e:
            print(f"Warning: Could not load macro asset {symbol}: {e}")
            
    if len(data_frames) < 3:
        print("Error: Insufficient macro data for HMM training.")
        return

    # Join on timestamp
    merged = None
    for symbol, df in data_frames.items():
        df = df.rename(columns={'log_ret': f'{symbol}_ret', 'volatility': f'{symbol}_vol'})
        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on='timestamp', how='inner')
            
    # Bounded features for the HMM: [BTC_RSI, BTC_Vol_Ratio, ETH_BTC_Spread_RSI, BTC_Mom_Ratio, Global_Vol]
    merged['ETH_BTC_ratio'] = merged['ETH_ret'] - merged['BTC_ret']
    
    # 1. BTC RSI (Bounded 0-100)
    # We'll calculate a simple 14-period RSI here natively
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    merged['BTC_RSI'] = calc_rsi(merged['BTC_ret'], 14)
    merged['ETH_BTC_RSI'] = calc_rsi(merged['ETH_BTC_ratio'], 14)
    
    # 2. Volatility Ratio (Spike detection)
    merged['BTC_vol_24h'] = merged['BTC_ret'].rolling(24).std()
    merged['BTC_vol_ratio'] = merged['BTC_vol'] / merged['BTC_vol_24h'].replace(0, 1e-10)
    
    # 3. Momentum Ratio (Trend detection)
    merged['BTC_mom'] = merged['BTC_ret'].rolling(24).sum()
    
    features = ['BTC_RSI', 'BTC_vol_ratio', 'ETH_BTC_RSI', 'BTC_mom', 'BTC_vol']
    merged = merged.replace([np.inf, -np.inf], 0).ffill().fillna(0)
    X_raw = merged[features].values
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Gaussian HMM with 4 regimes
    model = hmm.GaussianHMM(
        n_components=4, 
        covariance_type="full", 
        n_iter=2000, 
        random_state=42,
        init_params="stmc" 
    )
    model.fit(X_scaled)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/macro_hmm.joblib")
    joblib.dump(scaler, "models/macro_scaler.joblib")
    print(f"Successfully saved HMM and Scaler ({X_scaled.shape}) to models/ using data before {VAL_START}")


def train(
    timeframe,
    specialists=True,
    feature_profile="price_only",
    model_set=None,
    short_calibration_mode="raw",
    train_bear_short_meta=False,
    bear_short_target_mode="default",
):
    print(f"Training Cross-Sectional Sniper (Classifier) for {timeframe} bars...")
    X, y, y_meta, states, sample_meta, features = prepare_dataset(timeframe, "train", feature_profile=feature_profile)
    
    if X is None or len(X) < 1000:
        print("Insufficient data for training.")
        return
        
    X_train, X_val, y_train, y_val, y_meta_train, y_meta_val, s_train, s_val, meta_train, meta_val = train_test_split(
        X, y, y_meta, states, sample_meta, test_size=0.2, shuffle=False
    )

    n_trees = args.n_trees if args.n_trees else 250
    depth = args.depth if args.depth else {"1h": 14, "4h": 16, "15m": 12}.get(timeframe, 14)
    out_dir = _output_dir(model_set)

    metadata = {
        "model_set": model_set or "root",
        "timeframe": timeframe,
        "feature_profile": feature_profile,
        "feature_columns": list(features),
        "short_confidence": {
            "mode": short_calibration_mode if timeframe == "15m" else "raw",
            "calibrator_artifact": None,
            "bear_short_artifact": None,
            "bear_short_target_mode": bear_short_target_mode if timeframe == "15m" else "default",
        },
    }

    if specialists:
        print("Training Regime-Specific Specialists (Medallion Strategy)...")
        for state in range(4):
            mask_train = (s_train == state)
            mask_val = (s_val == state)
            
            if mask_train.sum() < 200: 
                print(f"Skipping State {state}: Insufficient data ({mask_train.sum()} samples)")
                continue
            
            print(f"--- Training Specialist: State {state} ---")
            X_s = X_train[mask_train]
            y_s = y_train[mask_train]
            ym_s = y_meta_train[mask_train]
            
            X_v = X_val[mask_val][features]
            y_v = y_val[mask_val]
            ym_v = y_meta_val[mask_val]
            
            # 1. Lead Model
            model = build_model(n_trees, depth, 0.05)
            w_s = compute_sample_weight(class_weight="balanced", y=y_s)
            
            if len(X_v) > 50:
                model.fit(X_s, y_s, sample_weight=w_s, eval_set=[(X_v, y_v)], verbose=False)
            else:
                model.set_params(early_stopping_rounds=None)
                model.fit(X_s, y_s, sample_weight=w_s)
            _save_model(model, out_dir, f"lead_{timeframe}_s{state}.json")
            
            # 2. Meta Model
            meta_target = (ym_s == 1).astype(int)
            meta_val_target = (ym_v == 1).astype(int)
            w_meta = compute_sample_weight(class_weight="balanced", y=meta_target)
            
            meta_model = build_model(n_trees, depth - 2, 0.03)
            if len(X_v) > 50:
                meta_model.fit(X_s, meta_target, sample_weight=w_meta, eval_set=[(X_v, meta_val_target)], verbose=False)
            else:
                meta_model.set_params(early_stopping_rounds=None)
                meta_model.fit(X_s, meta_target, sample_weight=w_meta)
            _save_model(meta_model, out_dir, f"meta_{timeframe}_s{state}.json")
    
    # Also train a "Global" fallback model

    if timeframe == "15m":
        print(f"Training Directional Split Models (n={n_trees}, d={depth})...")
        long_target = (y_train == 1).astype(int)
        short_target = (y_train == 2).astype(int)
        long_meta_target = ((y_train == 1) & (y_meta_train == 1)).astype(int)
        short_meta_target = ((y_train == 2) & (y_meta_train == 1)).astype(int)

        print("Training Stage 1A: Long Lead (15m)...")
        long_model = build_model(n_trees, depth, 0.05)
        long_model.fit(X_train, long_target, eval_set=[(X_val[features], (y_val == 1).astype(int))], verbose=False)

        print("Training Stage 1B: Short Lead (15m)...")
        short_model = build_model(n_trees, depth, 0.05)
        short_model.fit(X_train, short_target, eval_set=[(X_val[features], (y_val == 2).astype(int))], verbose=False)

        print("Training Stage 2A: Long Meta (15m)...")
        long_meta_model = build_model(n_trees, depth - 2, 0.03)
        long_meta_model.fit(X_train, long_meta_target, eval_set=[(X_val[features], ((y_val == 1) & (y_meta_val == 1)).astype(int))], verbose=False)

        print("Training Stage 2B: Short Meta (15m)...")
        short_meta_model = build_model(n_trees, depth - 2, 0.03)
        short_meta_model.fit(X_train, short_meta_target, eval_set=[(X_val[features], ((y_val == 2) & (y_meta_val == 1)).astype(int))], verbose=False)

        _save_model(long_model, out_dir, "lead_long_15m.xgb")
        _save_model(short_model, out_dir, "lead_short_15m.xgb")
        _save_model(long_meta_model, out_dir, "meta_long_15m.xgb")
        _save_model(short_meta_model, out_dir, "meta_short_15m.xgb")

        if train_bear_short_meta:
            bear_mask_train = meta_train["regime_family"].eq("bear").values
            bear_mask_val = meta_val["regime_family"].eq("bear").values
            if bear_mask_train.sum() >= 200 and bear_mask_val.sum() >= 50:
                bear_short_target_train = _build_bear_short_target(short_meta_target, meta_train, bear_short_target_mode)
                bear_short_target_val = _build_bear_short_target(
                    ((y_val == 2) & (y_meta_val == 1)).astype(int),
                    meta_val,
                    bear_short_target_mode,
                )
                print("Training Stage 2C: Bear-Specific Short Meta (15m)...")
                bear_short_meta_model = build_model(n_trees, depth - 2, 0.03)
                bear_short_meta_model.fit(
                    X_train.loc[bear_mask_train, features],
                    bear_short_target_train[bear_mask_train],
                    eval_set=[(X_val.loc[bear_mask_val, features], bear_short_target_val[bear_mask_val])],
                    verbose=False,
                )
                _save_model(bear_short_meta_model, out_dir, "meta_short_bear_15m.xgb")
                metadata["short_confidence"]["bear_short_artifact"] = "meta_short_bear_15m.xgb"
            else:
                print("Skipping bear-specific short meta: insufficient bear-family samples.")

        # Keep the aggregate 15m artifacts around for fallback compatibility.
        _save_model(long_model, out_dir, "lead_15m.xgb")
        _save_model(long_meta_model, out_dir, "meta_15m.xgb")

        if short_calibration_mode in {"platt", "isotonic"}:
            val_short_target = ((y_val == 2) & (y_meta_val == 1)).astype(int)
            raw_short_probs = short_meta_model.predict_proba(X_val[features])[:, 1]
            calibrator = _fit_short_calibrator(short_calibration_mode, raw_short_probs, val_short_target)
            if calibrator is not None:
                artifact_name = f"short_conf_15m_{short_calibration_mode}.joblib"
                joblib.dump(calibrator, out_dir / artifact_name)
                metadata["short_confidence"]["calibrator_artifact"] = artifact_name
        elif short_calibration_mode == "rank":
            metadata["short_confidence"]["calibrator_artifact"] = None

        base_wr, sniper_wr = validate_directional_models(
            long_model, short_model, long_meta_model, short_meta_model, X_val, y_val, features
        )

        if not model_set:
            os.makedirs(os.path.expanduser("~/.cache/autotrader"), exist_ok=True)
            long_model.save_model(os.path.expanduser("~/.cache/autotrader/model_long_15m.xgb"))
            short_model.save_model(os.path.expanduser("~/.cache/autotrader/model_short_15m.xgb"))
            long_meta_model.save_model(os.path.expanduser("~/.cache/autotrader/meta_model_long_15m.xgb"))
            short_meta_model.save_model(os.path.expanduser("~/.cache/autotrader/meta_model_short_15m.xgb"))
            long_model.save_model(os.path.expanduser("~/.cache/autotrader/model_15m.xgb"))
            long_meta_model.save_model(os.path.expanduser("~/.cache/autotrader/meta_model_15m.xgb"))
    else:
        print(f"Training Primary XGBClassifier (n={n_trees}, d={depth})...")
        print(f"Training Stage 1: Directional Lead ({timeframe})...")
        model = build_model(n_trees, depth, 0.05)
        # Global fallback uses full validation set for early stopping
        model.fit(X_train, y_train, eval_set=[(X_val[features], y_val)], verbose=False)

        print(f"Training Stage 2: Meta-Labeling ({timeframe})...")
        meta_target = (y_meta_train == 1).astype(int)
        meta_val_target = (y_meta_val == 1).astype(int)
        meta_model = build_model(n_trees, depth - 2, 0.03)
        meta_model.fit(X_train, meta_target, eval_set=[(X_val[features], meta_val_target)], verbose=False)

        _save_model(model, out_dir, f"lead_{timeframe}.json")
        _save_model(meta_model, out_dir, f"meta_{timeframe}.json")

        # Validation / Meta-Diagnostics
        val_preds = model.predict(X_val[features])
        val_meta_probs = meta_model.predict_proba(X_val[features])[:, 1]

        is_trend_pred = (val_preds != 0)
        base_wr = accuracy_score(y_val[is_trend_pred], val_preds[is_trend_pred]) if any(is_trend_pred) else 0.0

        sniper_mask = is_trend_pred & (val_meta_probs > 0.65)
        sniper_wr = accuracy_score(y_val[sniper_mask], val_preds[sniper_mask]) if any(sniper_mask) else 0.0

        if not model_set:
            model.save_model(os.path.expanduser(f"~/.cache/autotrader/model_{timeframe}.json"))
            meta_model.save_model(os.path.expanduser(f"~/.cache/autotrader/meta_model_{timeframe}.json"))
    
    print("\n" + "="*60)
    print(f"  VALIDATION RESULTS: {timeframe.upper()} SNIPER")
    print("="*60)
    print(f"Base Win Rate:    {base_wr:.3f}")
    print(f"Sniper Win Rate (Meta > 0.65): {sniper_wr:.3f}")
    
    if sniper_wr > 0.60:
        print("<< SNIPER TARGET REACHED >>")
    _write_metadata(out_dir, metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dual XGB sniper")
    parser.add_argument("--timeframe", type=str, default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--n_trees", type=int, default=100)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--train_hmm", action="store_true", help="Also retrain the macro HMM")
    parser.add_argument("--feature-profile", type=str, default="price_only", choices=["price_only", "price_context", "price_context_plus"])
    parser.add_argument("--model-set", type=str, default=None, help="Optional models/<name>/ output directory")
    parser.add_argument("--short-calibration-mode", type=str, default="raw", choices=["raw", "platt", "isotonic", "rank", "bear_model"])
    parser.add_argument("--train-bear-short-meta", action="store_true",
                        help="For 15m, train an additional bear-family short meta model artifact")
    parser.add_argument(
        "--bear-short-target-mode",
        type=str,
        default="default",
        choices=["default", "relative_tail"],
        help="For 15m bear-short artifact, choose the target construction mode",
    )
    args = parser.parse_args()
    
    if args.train_hmm:
        train_macro_hmm()
        
    train(
        args.timeframe,
        feature_profile=args.feature_profile,
        model_set=args.model_set,
        short_calibration_mode=args.short_calibration_mode,
        train_bear_short_meta=args.train_bear_short_meta,
        bear_short_target_mode=args.bear_short_target_mode,
    )
