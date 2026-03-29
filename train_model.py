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
    df['macd_line'] = macd_line # RAW (matching v4 breakthrough)
    
    # 4. BB Width
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_width'] = (4 * std) / sma.replace(0, 1e-10)
    
    # 5. Macro Regime (EMA 200)
    ema_200 = close.ewm(span=200, adjust=False).mean()
    df['ema_200_dist'] = (close - ema_200) / close
    
    # 6. Volatility
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

def load_broad_train_data():
    """Load training data using full historical range to leverage synthetic altcoin data."""
    from prepare import DATA_DIR
    # Use broad date range: altcoin parquets have synthetic data from 2004
    # BTC/ETH/SOL only start Jun 2023 but others have 20-year synthetic data
    BROAD_START = "2004-01-01"
    BROAD_END = "2024-06-30"
    start_ms = int(pd.Timestamp(BROAD_START, tz='UTC').timestamp() * 1000)
    end_ms = int(pd.Timestamp(BROAD_END, tz='UTC').timestamp() * 1000)
    data_dict = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('_1h.parquet'):
            continue
        symbol = fname.replace('_1h.parquet', '')
        df = pd.read_parquet(os.path.join(DATA_DIR, fname))
        mask = (df['timestamp'] >= start_ms) & (df['timestamp'] < end_ms)
        split_df = df[mask].reset_index(drop=True)
        if len(split_df) > 0:
            data_dict[symbol] = split_df
    return data_dict

def train():
    print("Loading training data (broad historical range for altcoin synthetic data)...")
    from strategy import ACTIVE_SYMBOLS
    data_dict = load_broad_train_data()
    all_features = []
    symbols = ACTIVE_SYMBOLS

    for i, symbol in enumerate(symbols):
        if symbol not in data_dict:
            continue
        print(f"Processing {symbol} (idx={i}, rows={len(data_dict[symbol])})...")
        df = data_dict[symbol]
        df_feat = calculate_features(df)
        df_feat['symbol_idx'] = i # One-hot or label encoding
        all_features.append(df_feat)
        
    full_df = pd.concat(all_features)
    
    features = ['ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'ret_48h', 'rsi_8', 'rsi_24', 'macd_hist', 'macd_line', 'bb_width', 'ema_200_dist', 'vol_24h', 'symbol_idx']
    X = full_df[features]
    y = full_df['label']
    
    print(f"Training on {len(X)} samples with {len(features)} features...")
    print(f"Class Distribution: {y.value_counts(normalize=True).to_dict()}")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        reg_alpha=1.0,
        colsample_bytree=0.7,
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
