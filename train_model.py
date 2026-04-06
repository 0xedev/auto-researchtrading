import pandas as pd
import numpy as np
import argparse
import joblib
import os
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from joblib import Parallel, delayed

from prepare import (
    calculate_features, load_data, prepare_dataset, 
    get_n_trees_depth, run_backtest, compute_score,
    TRAIN_START, TEST_END, VAL_START, VAL_END,
    SYMBOLS, SYMBOLS_15M, ASSET_CLASS, SYMBOL_TRAIN_START, FEATURE_COLS
)

def build_model(n_trees, depth, learning_rate):
    return xgb.XGBClassifier(
        n_estimators=n_trees,
        max_depth=depth,
        learning_rate=learning_rate,
        n_jobs=-1,
        tree_method='hist',
        colsample_bytree=0.7,
        random_state=42,
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


def train(timeframe):
    print(f"Training Cross-Sectional Sniper (Classifier) for {timeframe} bars...")
    X, y, y_meta, _, features = prepare_dataset(timeframe, "train")
    
    if X is None or len(X) < 1000:
        print("Insufficient data for training.")
        return
        
    X_train, X_val, y_train, y_val, y_meta_train, y_meta_val = train_test_split(
        X, y, y_meta, test_size=0.2, shuffle=False
    )

    unique_labels, label_counts = np.unique(y_train, return_counts=True)
    print(
        "Lead label distribution:",
        {int(k): int(v) for k, v in zip(unique_labels.tolist(), label_counts.tolist())}
    )
    
    n_trees = args.n_trees if args.n_trees else 100
    depth = args.depth if args.depth else {"1h": 12, "4h": 14, "15m": 12}.get(timeframe, 12)
    os.makedirs("models", exist_ok=True)

    if timeframe == "15m":
        print(f"Training Directional Split Models (n={n_trees}, d={depth})...")
        long_target = (y_train == 1).astype(int)
        short_target = (y_train == 2).astype(int)
        long_meta_target = ((y_train == 1) & (y_meta_train == 1)).astype(int)
        short_meta_target = ((y_train == 2) & (y_meta_train == 1)).astype(int)

        print("Training Stage 1A: Long Lead (15m)...")
        long_model = build_model(n_trees, depth, 0.05)
        long_model.fit(X_train, long_target)

        print("Training Stage 1B: Short Lead (15m)...")
        short_model = build_model(n_trees, depth, 0.05)
        short_model.fit(X_train, short_target)

        print("Training Stage 2A: Long Meta (15m)...")
        long_meta_model = build_model(n_trees, depth - 2, 0.03)
        long_meta_model.fit(X_train, long_meta_target)

        print("Training Stage 2B: Short Meta (15m)...")
        short_meta_model = build_model(n_trees, depth - 2, 0.03)
        short_meta_model.fit(X_train, short_meta_target)

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
        model.fit(X_train, y_train)

        print(f"Training Stage 2: Meta-Labeling ({timeframe})...")
        meta_target = (y_meta_train == 1).astype(int)
        meta_model = build_model(n_trees, depth - 2, 0.03)
        meta_model.fit(X_train, meta_target)

        model.save_model(f"models/lead_{timeframe}.xgb")
        meta_model.save_model(f"models/meta_{timeframe}.xgb")

        val_preds = model.predict(X_val[FEATURE_COLS])
        val_meta_probs = meta_model.predict_proba(X_val[FEATURE_COLS])[:, 1]

        is_trend_pred = (val_preds != 0)
        base_wr = accuracy_score(y_val[is_trend_pred], val_preds[is_trend_pred]) if any(is_trend_pred) else 0.0

        sniper_mask = is_trend_pred & (val_meta_probs > 0.65)
        sniper_wr = accuracy_score(y_val[sniper_mask], val_preds[sniper_mask]) if any(sniper_mask) else 0.0

        os.makedirs(os.path.expanduser("~/.cache/autotrader"), exist_ok=True)
        model.save_model(os.path.expanduser(f"~/.cache/autotrader/model_{timeframe}.xgb"))
        meta_model.save_model(os.path.expanduser(f"~/.cache/autotrader/meta_model_{timeframe}.xgb"))
    
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
    args = parser.parse_args()
    train(args.timeframe)
