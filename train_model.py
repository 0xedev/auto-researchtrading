import pandas as pd
import numpy as np
import argparse
import joblib
import os
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
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


def validate_directional_models(long_model, short_model, long_meta_model, short_meta_model, X_val, y_val):
    long_probs = long_model.predict_proba(X_val[FEATURE_COLS])[:, 1]
    short_probs = short_model.predict_proba(X_val[FEATURE_COLS])[:, 1]
    long_meta_probs = long_meta_model.predict_proba(X_val[FEATURE_COLS])[:, 1]
    short_meta_probs = short_meta_model.predict_proba(X_val[FEATURE_COLS])[:, 1]

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
    
    macro_assets = ["BTC", "ETH", "XAU", "SP500"]
    data_frames = {}
    
    for symbol in macro_assets:
        try:
            path = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "data", f"{symbol}_1h.parquet")
            df = pd.read_parquet(path)
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
    print(f"Successfully saved HMM and Scaler ({X_scaled.shape}) to models/")


def train(timeframe, specialists=True):
    print(f"Training Cross-Sectional Sniper (Classifier) for {timeframe} bars...")
    X, y, y_meta, states, features = prepare_dataset(timeframe, "train")
    
    if X is None or len(X) < 1000:
        print("Insufficient data for training.")
        return
        
    X_train, X_val, y_train, y_val, y_meta_train, y_meta_val, s_train, s_val = train_test_split(
        X, y, y_meta, states, test_size=0.2, shuffle=False
    )

    n_trees = args.n_trees if args.n_trees else 250
    depth = args.depth if args.depth else {"1h": 14, "4h": 16, "15m": 12}.get(timeframe, 14)
    os.makedirs("models", exist_ok=True)

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
            
            X_v = X_val[mask_val][FEATURE_COLS]
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
            model.save_model(f"models/lead_{timeframe}_s{state}.json")
            
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
            meta_model.save_model(f"models/meta_{timeframe}_s{state}.json")
    
    # Also train a "Global" fallback model

    if timeframe == "15m":
        print(f"Training Directional Split Models (n={n_trees}, d={depth})...")
        long_target = (y_train == 1).astype(int)
        short_target = (y_train == 2).astype(int)
        long_meta_target = ((y_train == 1) & (y_meta_train == 1)).astype(int)
        short_meta_target = ((y_train == 2) & (y_meta_train == 1)).astype(int)

        print("Training Stage 1A: Long Lead (15m)...")
        long_model = build_model(n_trees, depth, 0.05)
        long_model.fit(X_train, long_target, eval_set=[(X_val[FEATURE_COLS], (y_val == 1).astype(int))], verbose=False)

        print("Training Stage 1B: Short Lead (15m)...")
        short_model = build_model(n_trees, depth, 0.05)
        short_model.fit(X_train, short_target, eval_set=[(X_val[FEATURE_COLS], (y_val == 2).astype(int))], verbose=False)

        print("Training Stage 2A: Long Meta (15m)...")
        long_meta_model = build_model(n_trees, depth - 2, 0.03)
        long_meta_model.fit(X_train, long_meta_target, eval_set=[(X_val[FEATURE_COLS], ((y_val == 1) & (y_meta_val == 1)).astype(int))], verbose=False)

        print("Training Stage 2B: Short Meta (15m)...")
        short_meta_model = build_model(n_trees, depth - 2, 0.03)
        short_meta_model.fit(X_train, short_meta_target, eval_set=[(X_val[FEATURE_COLS], ((y_val == 2) & (y_meta_val == 1)).astype(int))], verbose=False)

        long_model.save_model("models/lead_long_15m.xgb")
        short_model.save_model("models/lead_short_15m.xgb")
        long_meta_model.save_model("models/meta_long_15m.xgb")
        short_meta_model.save_model("models/meta_short_15m.xgb")

        # Keep the aggregate 15m artifacts around for fallback compatibility.
        long_model.save_model("models/lead_15m.xgb")
        long_meta_model.save_model("models/meta_15m.xgb")

        base_wr, sniper_wr = validate_directional_models(
            long_model, short_model, long_meta_model, short_meta_model, X_val, y_val
        )

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
        model.fit(X_train, y_train, eval_set=[(X_val[FEATURE_COLS], y_val)], verbose=False)

        print(f"Training Stage 2: Meta-Labeling ({timeframe})...")
        meta_target = (y_meta_train == 1).astype(int)
        meta_val_target = (y_meta_val == 1).astype(int)
        meta_model = build_model(n_trees, depth - 2, 0.03)
        meta_model.fit(X_train, meta_target, eval_set=[(X_val[FEATURE_COLS], meta_val_target)], verbose=False)

        model.save_model(f"models/lead_{timeframe}.json")
        meta_model.save_model(f"models/meta_{timeframe}.json")

        # Validation / Meta-Diagnostics
        val_preds = model.predict(X_val[FEATURE_COLS])
        val_meta_probs = meta_model.predict_proba(X_val[FEATURE_COLS])[:, 1]

        is_trend_pred = (val_preds != 0)
        base_wr = accuracy_score(y_val[is_trend_pred], val_preds[is_trend_pred]) if any(is_trend_pred) else 0.0

        sniper_mask = is_trend_pred & (val_meta_probs > 0.65)
        sniper_wr = accuracy_score(y_val[sniper_mask], val_preds[sniper_mask]) if any(sniper_mask) else 0.0

        model.save_model(os.path.expanduser(f"~/.cache/autotrader/model_{timeframe}.json"))
        meta_model.save_model(os.path.expanduser(f"~/.cache/autotrader/meta_model_{timeframe}.json"))
    
    print("\n" + "="*60)
    print(f"  VALIDATION RESULTS: {timeframe.upper()} SNIPER")
    print("="*60)
    print(f"Base Win Rate:    {base_wr:.3f}")
    print(f"Sniper Win Rate (Meta > 0.65): {sniper_wr:.3f}")
    
    if sniper_wr > 0.60:
        print("<< SNIPER TARGET REACHED >>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dual XGB sniper")
    parser.add_argument("--timeframe", type=str, default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--n_trees", type=int, default=100)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--train_hmm", action="store_true", help="Also retrain the macro HMM")
    args = parser.parse_args()
    
    if args.train_hmm:
        train_macro_hmm()
        
    train(args.timeframe)
