import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from prepare import load_data

MODEL_PATH = os.path.expanduser("~/.cache/autotrader/model.joblib")

def calculate_features(df):
    """Vectorized feature calculation for a single symbol dataframe."""
    df = df.copy()
    close = df['close']
    
    # 1. Returns
    df['ret_1h'] = close.pct_change(1)
    df['ret_4h'] = close.pct_change(4)
    df['ret_12h'] = close.pct_change(12)
    df['ret_24h'] = close.pct_change(24)
    df['ret_48h'] = close.pct_change(48)
    
    # 2. RSI (Simplified Vectorized)
    def v_rsi(s, p):
        d = s.diff()
        g = (d.where(d > 0, 0)).rolling(window=p).mean()
        l = (-d.where(d < 0, 0)).rolling(window=p).mean()
        rs = g / l.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    df['rsi_8'] = v_rsi(close, 8)
    df['rsi_24'] = v_rsi(close, 24)
    
    # 3. MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_line - signal_line
    
    # 4. BB Width
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_width'] = (4 * std) / sma.replace(0, 1e-10)
    
    # 5. Volatility
    df['vol_24h'] = df['ret_1h'].rolling(24).std()
    
    # 6. Targets (Next 4h return)
    # Target: Predict next 4h return
    df['target_ret'] = df['close'].shift(-4).pct_change(4).shift(-4)
    
    # 7. CLEANUP: Replace inf with NaN and drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Label: 1 for Buy (>0.15%), 2 for Sell (<-0.15%), 0 for Neutral
    fee_threshold = 0.0015
    df['label'] = 0
    df.loc[df['target_ret'] > fee_threshold, 'label'] = 1
    df.loc[df['target_ret'] < -fee_threshold, 'label'] = 2
    
    return df

def train():
    print("Loading 20-year training data...")
    # Using 'train' split which is 2004 - June 2024
    data_dict = load_data(split="train")
    
    all_features = []
    symbols = sorted(list(data_dict.keys()))
    
    for i, symbol in enumerate(symbols):
        print(f"Processing {symbol}...")
        df = data_dict[symbol]
        df_feat = calculate_features(df)
        df_feat['symbol_idx'] = i # One-hot or label encoding
        all_features.append(df_feat)
        
    full_df = pd.concat(all_features)
    
    features = ['ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'ret_48h', 'rsi_8', 'rsi_24', 'macd_hist', 'bb_width', 'vol_24h', 'symbol_idx']
    X = full_df[features]
    y = full_df['label']
    
    print(f"Training on {len(X)} samples with {len(features)} features...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        tree_method='hist', # Fast on CPU, faster on GPU
        device='cuda' if os.path.exists('/dev/nvidia0') else 'cpu',
        random_state=42
    )
    
    model.fit(X, y)
    
    print(f"Saving model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    
    # Basic Feature Importance
    importances = model.feature_importances_
    for f, imp in zip(features, importances):
        print(f"Feature {f:12s}: {imp:.4f}")

if __name__ == "__main__":
    train()
