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

def train(timeframe):
    print(f"Training Cross-Sectional Sniper (Classifier) for {timeframe} bars...")
    X, y, y_meta, _, features = prepare_dataset(timeframe, "train")
    
    if X is None or len(X) < 1000:
        print("Insufficient data for training.")
        return
        
    X_train, X_val, y_train, y_val, y_meta_train, y_meta_val = train_test_split(
        X, y, y_meta, test_size=0.2, shuffle=False
    )
    
    n_trees = args.n_trees if args.n_trees else 100
    depth = args.depth if args.depth else {"1h": 12, "4h": 14, "15m": 12}.get(timeframe, 12)
    # Balanced weights for outliers
    from sklearn.utils import class_weight
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    
    print(f"Training Primary XGBClassifier (n={n_trees}, d={depth})...")
    # Stage 1: Directional Lead (Predict Bull/Bear/Neutral)
    print(f"Training Stage 1: Directional Lead ({timeframe})...")
    model = xgb.XGBClassifier(n_estimators=n_trees, max_depth=depth, learning_rate=0.05, n_jobs=-1,
                              tree_method='hist', random_state=42)
    model.fit(X_train, y_train)

    # Stage 2: Meta-Labeling (Predict Confidence of hitting TP)
    print(f"Training Stage 2: Meta-Labeling ({timeframe})...")
    meta_target = (y_meta_train == 1).astype(int) # 1 if Hit Profit Target, 0 otherwise
    meta_model = xgb.XGBClassifier(n_estimators=n_trees, max_depth=depth-2, learning_rate=0.03, n_jobs=-1,
                                   tree_method='hist', random_state=42)
    meta_model.fit(X_train, meta_target)
    
    # Save the Lead & Meta pair
    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/lead_{timeframe}.xgb")
    meta_model.save_model(f"models/meta_{timeframe}.xgb")
    
    # 3. Validation
    val_preds = model.predict(X_val[FEATURE_COLS])
    val_meta_probs = meta_model.predict_proba(X_val[FEATURE_COLS])[:, 1]
    
    # Sniper logic
    is_trend_pred = (val_preds != 0)
    base_wr = accuracy_score(y_val[is_trend_pred], val_preds[is_trend_pred]) if any(is_trend_pred) else 0.0
    
    sniper_mask = is_trend_pred & (val_meta_probs > 0.65)
    sniper_wr = accuracy_score(y_val[sniper_mask], val_preds[sniper_mask]) if any(sniper_mask) else 0.0
    
    print("\n" + "="*60)
    print(f"  VALIDATION RESULTS: {timeframe.upper()} SNIPER")
    print("="*60)
    print(f"Base Win Rate:    {base_wr:.3f}")
    print(f"Sniper Win Rate (Meta > 0.65): {sniper_wr:.3f}")
    
    if sniper_wr > 0.60:
        print("<< SNIPER TARGET REACHED >>")
    
    # 4. Save Models
    os.makedirs(os.path.expanduser("~/.cache/autotrader"), exist_ok=True)
    model.save_model(os.path.expanduser(f"~/.cache/autotrader/model_{timeframe}.xgb"))
    meta_model.save_model(os.path.expanduser(f"~/.cache/autotrader/meta_model_{timeframe}.xgb"))

if __name__ == "__main__":
    parser.add_argument("--timeframe", type=str, default="1h", choices=["15m", "1h", "4h"])
    parser.add_argument("--n_trees", type=int, default=100)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()
    train(args.timeframe)
