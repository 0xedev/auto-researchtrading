"""
Autotrader backtesting engine. Fixed evaluation harness — DO NOT MODIFY.
Downloads Hyperliquid historical data, runs backtests, computes scores.

Usage:
    python prepare.py                  # download data
    python prepare.py --symbols BTC    # download specific symbols
"""

import os
import sys
import time
import math
import signal
import argparse
import joblib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import pyarrow.parquet as pq
from hmmlearn import hmm

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 1200             # Increased for complex 17-symbol dual-model backtests
INITIAL_CAPITAL = 100_000.0    # $100K starting capital
MAKER_FEE = 0.0002             # 2 bps
TAKER_FEE = 0.0005             # 5 bps
SLIPPAGE_BPS = 1.0             # 1 bps simulated slippage
MAX_LEVERAGE = 20              # max leverage allowed
LOOKBACK_BARS = 500            # history buffer provided to strategy
BAR_INTERVAL = "1h"

SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK",
    "AVAX", "DOT", "ATOM", "NEAR", "UNI", "APT", "SUI",
    "XAU", "SP500"
]

SYMBOLS_15M = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK",
    "AVAX", "DOT", "ATOM", "NEAR", "UNI", "APT", "SUI", "XAU"
]  # SP500/DXY excluded — no free 15min intraday source

# Earliest Binance USDT listing dates (approx) for 15m downloads
BINANCE_15M_START = {
    "BTC": "2017-08-17", "ETH": "2017-08-17", "SOL": "2020-08-11",
    "BNB": "2017-11-06", "XRP": "2018-04-23", "ADA": "2018-04-17",
    "DOGE": "2019-07-05", "LINK": "2019-01-16", "AVAX": "2020-09-22",
    "DOT": "2020-08-19", "ATOM": "2019-04-29", "NEAR": "2020-10-15",
    "UNI": "2020-09-17", "APT": "2022-10-19", "SUI": "2023-05-03",
}

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
HF_XAU_15M_URL = "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_15m_data.jsonl"

# Date splits (UTC timestamps) - 2025 Strictly Quarantined as OOS
# Training: 2017-01-01 → 2022-06-30 (pre-crash baseline)
# Validation: 2022-07-01 → 2024-06-30 (2 full years: bear, recovery, early bull)
# Robustness: 2018-01-01 → 2024-06-30 (all meaningful crypto history)
# OOS: 2025-01-01 → 2025-12-31 (full year of unseen data)
TRAIN_START = "2017-01-01"   # Default global start (overridden per-symbol below)
TRAIN_END   = "2022-06-30"   # Cut before validation starts
VAL_START   = "2022-07-01"   # 2-year window: crash → recovery → bull
VAL_END     = "2024-06-30"
TEST_START  = "2022-07-01"
TEST_END    = "2024-06-30"
ROBUST_START = "2018-01-01"
ROBUST_END   = "2024-06-30"

# Per-symbol earliest usable training start
# Universal goal: use deepest available price history for each asset
SYMBOL_TRAIN_START = {
    "BTC":   "2011-08-01",  # Earliest reliable CryptoCompare exchange data
    "ETH":   "2015-08-01",  # Ethereum mainnet launch
    "XRP":   "2013-08-01",  # XRP had liquid markets from 2013
    "BNB":   "2017-07-01",
    "ADA":   "2017-10-01",
    "DOGE":  "2014-01-01",  # DOGE launched Jan 2014
    "LINK":  "2017-09-01",
    "DOT":   "2020-08-01",
    "ATOM":  "2019-04-01",
    "AVAX":  "2020-09-01",
    "NEAR":  "2020-10-01",
    "UNI":   "2020-09-01",
    "APT":   "2022-10-01",
    "SUI":   "2023-05-01",
    "SOL":   "2020-04-01",
    "XAU":   "2004-01-01",  # Gold — full history
    "SP500": "2004-01-01",  # S&P — full history
}

# Asset class labels — universal feature, works across any market
# 0=Crypto (volatile, 24/7), 1=Commodity (macro-driven), 2=Equity Index (session-based)
ASSET_CLASS = {
    "BTC": 0, "ETH": 0, "SOL": 0, "BNB": 0, "XRP": 0,
    "ADA": 0, "DOGE": 0, "LINK": 0, "AVAX": 0, "DOT": 0,
    "ATOM": 0, "NEAR": 0, "UNI": 0, "APT": 0, "SUI": 0,
    "XAU": 1,
    "SP500": 2,
}

HOURS_PER_YEAR = 8760

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Quantitative Feature Columns
FEATURE_COLS = [
    'ret_1h', 'ret_4h', 'ret_12h', 'ret_24h', 'ret_48h', 
    'rsi_8', 'rsi_24', 'macd_hist', 'macd_line', 
    'bb_width', 'ema_200_dist', 'vol_24h', 'atr_pct', 
    'dist_to_high', 'dist_to_low', 'vol_ratio_24h', 'asset_class',
    'dist_to_vwap', 'vol_ema_50', 'market_vol', 'market_ret',
    'rel_ret_1h', 'rel_ret_4h', 'rel_bb_width',
    'fvg_detected', 'msb_status', 'ob_dist', 'frac_diff_close'
]

def get_frac_diff_weights(d, size):
    """Calculates weights for fractional differentiation."""
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1]).reshape(-1, 1)

def apply_frac_diff(series, d, threshold=1e-5):
    """Applies fixed-window fractional differentiation to a series."""
    weights = get_frac_diff_weights(d, size=50) # Fixed window of 50 for HFT
    res = []
    for i in range(len(series)):
        if i < 50:
            res.append(0)
            continue
        window = series.iloc[i-50:i].values.reshape(-1, 1)
        res.append(np.dot(weights.T, window)[0][0])
    return pd.Series(res, index=series.index)

def calculate_features(df, timeframe="1h"):
    """Vectorized feature calculation for a single symbol dataframe."""
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 1. Returns
    df['ret_1h'] = close.pct_change(1)
    df['ret_4h'] = close.pct_change(4)
    df['ret_12h'] = close.pct_change(12)
    df['ret_24h'] = close.pct_change(24)
    df['ret_48h'] = close.pct_change(48)
    
    # 2. RSI
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
    df['macd_line'] = macd_line 
    
    # 4. BB Width (35-bar)
    sma35 = close.rolling(35).mean()
    std35 = close.rolling(35).std()
    df['bb_width'] = (4 * std35) / sma35.replace(0, 1e-10)
    
    # 5. ATR (Average True Range)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / close
    
    # 6. Donchian Distance
    df['donchian_high_20'] = high.rolling(20).max()
    df['donchian_low_20'] = low.rolling(20).min()
    df['dist_to_high'] = (df['donchian_high_20'] - close) / close
    df['dist_to_low'] = (close - df['donchian_low_20']) / close
    
    # 7. Macro Regime (EMA 200)
    ema_200 = close.ewm(span=200, adjust=False).mean()
    df['ema_200_dist'] = (close - ema_200) / close
    
    # 8. Volume momentum
    df['vol_ratio_24h'] = df['volume'] / df['volume'].rolling(24).mean().replace(0, 1e-10)
    df['vol_24h'] = df['volume'].rolling(24).mean()
    df['vol_ema_50'] = df['volume'].ewm(span=50, adjust=False).mean()
    
    # 9. VWAP (Approximate via typical price)
    tp = (high + low + close) / 3
    df['vwap'] = (tp * df['volume']).rolling(24).sum() / df['volume'].rolling(24).sum().replace(0, 1e-10)
    df['dist_to_vwap'] = (close - df['vwap']) / close

    # 10. Microstructure: Refined FVG (Fair Value Gaps >= 0.5 ATR)
    atr = df['atr_14']
    df['fvg_bull'] = ((df['low'] > df['high'].shift(2)) & (df['low'] - df['high'].shift(2) > 0.5 * atr)).astype(int)
    df['fvg_bear'] = ((df['high'] < df['low'].shift(2)) & (df['low'].shift(2) - df['high'] > 0.5 * atr)).astype(int)
    df['fvg_detected'] = df['fvg_bull'] - df['fvg_bear']

    # 11. Liquidity Sweep Detection
    # Price dips below 24-period low but closes back above it within 1-2 bars
    local_low = df['low'].rolling(24).min().shift(1)
    local_high = df['high'].rolling(24).max().shift(1)
    
    is_low_sweep = (df['low'] < local_low) & (close > local_low)
    is_high_sweep = (df['high'] > local_high) & (close < local_high)
    df['liquidity_sweep'] = np.where(is_low_sweep, 1, np.where(is_high_sweep, -1, 0))

    # 12. Market Structure Break (MSB) Proxy
    recent_high = df['high'].rolling(24).max()
    recent_low = df['low'].rolling(24).min()
    df['msb_status'] = np.where(close > recent_high.shift(1), 1, np.where(close < recent_low.shift(1), -1, 0))

    # 12. Order Block (OB) Distance
    # Approximate OB as the high-volume candle before a MSB
    df['ob_zone'] = np.where(df['volume'] > df['volume'].rolling(24).mean() * 1.5, close, np.nan)
    df['ob_zone'] = df['ob_zone'].ffill()
    df['ob_dist'] = (close - df['ob_zone']) / close

    # 13. Fractional Differentiation (Stationary Memory)
    # d=0.4 is the industry sweet spot for crypto ADF stationarity
    df['frac_diff_close'] = apply_frac_diff(close, d=0.4)

    # 14. Macro Regime (HMM Slot)
    # This will be populated by the strategy/backtester using the saved HMM model
    df['macro_state'] = 0 

    # 15. Sanitization
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

# Index data types 

@dataclass
class BarData:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: float
    history: pd.DataFrame  # last LOOKBACK_BARS bars

@dataclass
class Signal:
    symbol: str
    target_position: float   # target USD notional (signed: +long, -short)
    order_type: str = "market"

@dataclass
class PortfolioState:
    cash: float
    positions: dict          # symbol -> signed USD notional
    entry_prices: dict       # symbol -> avg entry price
    equity: float = 0.0
    timestamp: int = 0

@dataclass
class BacktestResult:
    sharpe: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    annual_turnover: float = 0.0
    backtest_seconds: float = 0.0
    duration_days: float = 0.0
    equity_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

def get_triple_barrier_labels(df, timeframe):
    """De Prado's Triple Barrier Method for conviction labeling.
    1 = Hit PT, 2 = Hit SL, 0 = Vertical/Breath exit.
    """
    close = df['close']
    vol = close.pct_change().rolling(24).std().fillna(0.01)
    lookahead = {"1h": 48, "4h": 24, "15m": 64}.get(timeframe, 24)
    
    labels = np.zeros(len(df))
    for i in range(len(df) - lookahead):
        price_now = close.iloc[i]
        pt = price_now * (1 + 2.0 * vol.iloc[i])
        sl = price_now * (1 - 1.0 * vol.iloc[i])
        
        # Check future path
        future_path = close.iloc[i+1 : i+lookahead]
        hit_pt = future_path[future_path >= pt].index
        hit_sl = future_path[future_path <= sl].index
        
        first_pt = hit_pt[0] if len(hit_pt) > 0 else 9e18
        first_sl = hit_sl[0] if len(hit_sl) > 0 else 9e18
        
        if first_pt < first_sl and first_pt != 9e18:
            labels[i] = 1 # Profit
        elif first_sl < first_pt and first_sl != 9e18:
            labels[i] = 2 # Loss
        else:
            labels[i] = 0 # Vertical
            
    return labels


def get_directional_labels(df, timeframe):
    """Volatility-normalized directional labels.

    The 15m system was previously labeled only on extreme 4-bar outliers, which
    produced very sparse action. For intraday alpha discovery we use a slightly
    longer horizon and normalized forward returns so the lead model sees more
    tradable opportunities without discarding volatility context.
    """
    close = df["close"]
    vol = close.pct_change().rolling(24).std().replace(0, np.nan)

    horizon_map = {"15m": 8, "1h": 4, "4h": 4}
    threshold_map = {"15m": 0.50, "1h": 1.50, "4h": 1.50}

    horizon = horizon_map.get(timeframe, 4)
    threshold = threshold_map.get(timeframe, 1.50)

    forward_ret = close.shift(-horizon) / close - 1
    scaled_ret = forward_ret / (vol * np.sqrt(horizon))

    labels = np.zeros(len(df))
    labels[scaled_ret > threshold] = 1
    labels[scaled_ret < -threshold] = 2
    return labels, forward_ret

def get_n_trees_depth(timeframe):
    n_trees = 100
    depth = {"1h": 12, "4h": 14, "15m": 12}.get(timeframe, 12)
    return n_trees, depth

def prepare_dataset(timeframe, split_name):
    # This logic is now shared by the trainer
    data_dict = load_data(split=split_name, resample_4h=(timeframe == "4h"))
    all_features, all_y, all_y_meta, all_states = [], [], [], []
    
    # Pre-calculate Market Regime (Systemic Beta)
    print("Pre-calculating Market Regime Context...")
    market_vols = []
    market_rets = []
    for s, df in data_dict.items():
        if len(df) < 300: continue
        market_vols.append(calculate_features(df, timeframe=timeframe)['bb_width'])
        market_rets.append(df['close'].pct_change())
    
    m_vol = pd.concat(market_vols, axis=1).median(axis=1).fillna(0)
    m_ret = pd.concat(market_rets, axis=1).median(axis=1).fillna(0)

    for symbol, df in data_dict.items():
        df_feat = calculate_features(df, timeframe=timeframe)
        if len(df_feat) < 300: continue
        
        # Inject Market Context
        df_feat['market_vol'] = m_vol
        df_feat['market_ret'] = m_ret
        market_ret_4h = m_ret.rolling(4).sum().fillna(0)
        df_feat['rel_ret_1h'] = df_feat['ret_1h'] - df_feat['market_ret']
        df_feat['rel_ret_4h'] = df_feat['ret_4h'] - market_ret_4h
        df_feat['rel_bb_width'] = df_feat['bb_width'] - df_feat['market_vol']
        
        labels, forward_ret = get_directional_labels(df_feat, timeframe)

        meta = get_triple_barrier_labels(df_feat, timeframe)

        valid_mask = ~(df_feat[FEATURE_COLS].isna().any(axis=1) | forward_ret.isna() | np.isinf(forward_ret))
        
        sliced_X = df_feat[FEATURE_COLS][valid_mask].values[50:-100]
        sliced_y = labels[valid_mask][50:-100]
        sliced_meta = meta[valid_mask][50:-100]
        sliced_states = df_feat['macro_state'][valid_mask][50:-100].values

        all_features.append(pd.DataFrame(sliced_X, columns=FEATURE_COLS))
        all_y.append(sliced_y)
        all_y_meta.append(sliced_meta)
        all_states.append(sliced_states)

    if not all_features: return None, None, None, None, None
    return pd.concat(all_features), np.concatenate(all_y), np.concatenate(all_y_meta), np.concatenate(all_states), FEATURE_COLS

# Binance symbol mapping
BINANCE_SYMBOL_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "BNB": "BNBUSDT", "XRP": "XRPUSDT", "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT", "LINK": "LINKUSDT", "AVAX": "AVAXUSDT",
    "DOT": "DOTUSDT", "ATOM": "ATOMUSDT", "NEAR": "NEARUSDT",
    "UNI": "UNIUSDT", "APT": "APTUSDT", "SUI": "SUIUSDT",
}

def _download_cryptocompare_candles(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download hourly OHLCV from CryptoCompare (no geo-restrictions)."""
    all_rows = []
    # CryptoCompare uses 'toTs' (end timestamp in seconds) and returns up to 2000 bars
    current_end = end_ms // 1000
    start_s = start_ms // 1000

    while current_end > start_s:
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": 2000,
            "toTs": current_end,
        }
        resp = requests.get(CRYPTOCOMPARE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("Data", {}).get("Data", [])
        if not bars:
            break
        for bar in bars:
            ts_s = bar["time"]
            if ts_s < start_s:
                continue
            all_rows.append({
                "timestamp": ts_s * 1000,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volumefrom", 0)),
            })
        # Move window back
        earliest = bars[0]["time"]
        if earliest >= current_end:
            break
        current_end = earliest - 1
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def _download_binance_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download funding rate history from Binance Futures (no API key needed)."""
    binance_sym = BINANCE_SYMBOL_MAP.get(symbol)
    if not binance_sym:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    all_rows = []
    current = start_ms
    while current < end_ms:
        params = {
            "symbol": binance_sym,
            "startTime": current,
            "endTime": min(current + 90 * 24 * 3600 * 1000, end_ms),
            "limit": 1000,
        }
        try:
            resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                # No data in this window — skip ahead (funding may start later)
                current += 90 * 24 * 3600 * 1000
                continue
            for row in data:
                all_rows.append({
                    "timestamp": int(row["fundingTime"]),
                    "funding_rate": float(row["fundingRate"]),
                })
            current = int(data[-1]["fundingTime"]) + 1
        except Exception:
            break
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return pd.DataFrame(all_rows)


def _download_hl_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download funding rate history from Hyperliquid."""
    all_rows = []
    current = start_ms
    while current < end_ms:
        body = {
            "type": "fundingHistory",
            "coin": symbol,
            "startTime": current,
            "endTime": min(current + 30 * 24 * 3600 * 1000, end_ms),
        }
        try:
            resp = requests.post(HL_INFO_URL, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for row in data:
                all_rows.append({
                    "timestamp": int(row["time"]),
                    "funding_rate": float(row["fundingRate"]),
                })
            current = int(data[-1]["time"]) + 1
        except Exception:
            break
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return pd.DataFrame(all_rows)


def _download_hl_candles(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download OHLCV candles from Hyperliquid."""
    all_rows = []
    current = start_ms
    chunk_ms = 30 * 24 * 3600 * 1000  # 30 days
    while current < end_ms:
        body = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": min(current + chunk_ms, end_ms),
            }
        }
        try:
            resp = requests.post(HL_INFO_URL, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                current += chunk_ms
                continue
            for row in data:
                all_rows.append({
                    "timestamp": int(row["t"]),
                    "open": float(row["o"]),
                    "high": float(row["h"]),
                    "low": float(row["l"]),
                    "close": float(row["c"]),
                    "volume": float(row["v"]),
                })
            current = int(data[-1]["t"]) + 3600 * 1000
        except Exception:
            current += chunk_ms
        time.sleep(0.2)
    return pd.DataFrame(all_rows)


def _download_binance_15m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download 15-minute OHLCV from Binance Spot (no API key needed)."""
    binance_sym = BINANCE_SYMBOL_MAP.get(symbol)
    if not binance_sym:
        return pd.DataFrame()

    all_rows = []
    current = start_ms
    chunk_ms = 1000 * 15 * 60 * 1000  # 1000 bars * 15min in ms

    while current < end_ms:
        params = {
            "symbol": binance_sym,
            "interval": "15m",
            "startTime": current,
            "endTime": min(current + chunk_ms, end_ms),
            "limit": 1000,
        }
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for bar in data:
                all_rows.append({
                    "timestamp": int(bar[0]),
                    "open": float(bar[1]),
                    "high": float(bar[2]),
                    "low": float(bar[3]),
                    "close": float(bar[4]),
                    "volume": float(bar[5]),
                })
            current = int(data[-1][0]) + 15 * 60 * 1000
        except Exception as e:
            print(f"    Binance 15m error for {symbol}: {e}")
            break
        time.sleep(0.1)  # Binance rate limit: 1200 req/min

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def _download_hf_xau_15m() -> pd.DataFrame:
    """Download XAU 15-minute data from HuggingFace JSONL dataset."""
    import json
    print("    Downloading XAU 15m from HuggingFace (may take a moment)...")
    try:
        resp = requests.get(HF_XAU_15M_URL, timeout=120, stream=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"    HuggingFace download failed: {e}")
        return pd.DataFrame()

    all_rows = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            row = json.loads(line)
            # Parse "YYYY.MM.DD HH:MM" format
            dt = pd.to_datetime(row["Date"], format="%Y.%m.%d %H:%M", utc=True)
            all_rows.append({
                "timestamp": int(dt.timestamp() * 1000),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0)),
            })
        except Exception:
            continue

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"    XAU: parsed {len(df)} bars from HuggingFace")
    return df


def download_data(symbols=None):
    """Download historical OHLCV + funding data for all symbols."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if symbols is None:
        symbols = SYMBOLS

    start_ms = int(pd.Timestamp(TRAIN_START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(TEST_END, tz="UTC").timestamp() * 1000)

    for symbol in symbols:
        filepath = os.path.join(DATA_DIR, f"{symbol}_1h.parquet")
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            print(f"  {symbol}: already have {len(existing)} bars")
            continue

        print(f"  {symbol}: downloading candles from CryptoCompare...")

        # Use CryptoCompare for reliable historical OHLCV (no geo-restrictions)
        df = _download_cryptocompare_candles(symbol, start_ms, end_ms)
        if len(df) < 100:
            print(f"  {symbol}: CryptoCompare insufficient ({len(df)} bars), trying HL...")
            df = _download_hl_candles(symbol, "1h", start_ms, end_ms)

        if df.empty:
            print(f"  {symbol}: NO DATA AVAILABLE, skipping")
            continue

        # Download funding rates (Binance first, fallback to Hyperliquid)
        print(f"  {symbol}: downloading funding rates from Binance...")
        funding = _download_binance_funding(symbol, start_ms, end_ms)
        if funding.empty:
            print(f"  {symbol}: Binance funding unavailable, trying Hyperliquid...")
            funding = _download_hl_funding(symbol, start_ms, end_ms)

        # Merge
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if not funding.empty:
            funding = funding.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            # Merge nearest — funding is every 8h, candles every 1h
            df = pd.merge_asof(df, funding, on="timestamp", direction="backward")
        if "funding_rate" not in df.columns:
            df["funding_rate"] = 0.0
        df["funding_rate"] = df["funding_rate"].fillna(0.0)

        df.to_parquet(filepath, index=False)
        print(f"  {symbol}: saved {len(df)} bars to {filepath}")


def download_15m_data(symbols=None):
    """Download 15-minute OHLCV data for SYMBOLS_15M."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if symbols is None:
        symbols = SYMBOLS_15M

    end_ms = int(pd.Timestamp(TEST_END, tz="UTC").timestamp() * 1000)

    for symbol in symbols:
        filepath = os.path.join(DATA_DIR, f"{symbol}_15m.parquet")
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            print(f"  {symbol}: already have {len(existing)} 15m bars")
            continue

        if symbol == "XAU":
            df = _download_hf_xau_15m()
        else:
            start_date = BINANCE_15M_START.get(symbol, "2020-01-01")
            start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
            print(f"  {symbol}: downloading 15m candles from Binance (since {start_date})...")
            df = _download_binance_15m(symbol, start_ms, end_ms)

        if df is None or df.empty:
            print(f"  {symbol}: NO 15m DATA AVAILABLE, skipping")
            continue

        # No funding rate on 15min source
        df["funding_rate"] = 0.0

        df.to_parquet(filepath, index=False)
        print(f"  {symbol}: saved {len(df)} 15m bars to {filepath}")


def _resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H dataframe to 4H candles (aligned to 00:00, 04:00, etc. UTC)."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('datetime')
    resampled = df.resample('4h', origin='start_day').agg({
        'timestamp': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'funding_rate': 'mean',
    })
    resampled = resampled.dropna(subset=['open', 'close'])
    resampled = resampled.reset_index(drop=True)
    return resampled


def load_data(split: str = "val", resample_4h: bool = False, resample_15m: bool = False) -> dict:
    """Load OHLCV+funding data for the given split. Returns {symbol: DataFrame}.
    
    Per-symbol training starts: newer assets (APT, SUI, SOL, etc.) use their
    real listing date as the training floor so we never train on zero-padded noise.
    For val/oos/robustness splits the global split dates are used for all symbols.
    When resample_15m=True, loads from {symbol}_15m.parquet files instead of 1h.
    """
    splits = {
        "train":      (TRAIN_START, TRAIN_END),
        "val":        (VAL_START,   VAL_END),
        "test":       (TEST_START,  TEST_END),
        "robustness": (ROBUST_START, ROBUST_END),
        "val_15m":    (VAL_START,   VAL_END),
        "train_15m":  (TRAIN_START, TRAIN_END),
        "oos":        ("2025-01-01", "2025-12-31"),
        "oos_15m":    ("2025-01-01", "2025-12-31"),
    }
    assert split in splits, f"split must be one of {list(splits.keys())}"
    global_start_str, end_str = splits[split]
    end_ms = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)

    # Determine which symbols and file suffix to use
    use_15m = resample_15m or split.endswith("_15m")
    symbol_list = SYMBOLS_15M if use_15m else SYMBOLS
    suffix = "_15m.parquet" if use_15m else "_1h.parquet"

    result = {}
    for symbol in symbol_list:
        filepath = os.path.join(DATA_DIR, f"{symbol}{suffix}")
        if not os.path.exists(filepath):
            continue
        df = pd.read_parquet(filepath)

        # Apply per-symbol training start floor only on training splits
        if split in ("train", "train_15m") and symbol in SYMBOL_TRAIN_START:
            sym_start = SYMBOL_TRAIN_START[symbol]
            # Use the later of the global split start and the symbol's listing date
            effective_start = max(global_start_str, sym_start)
        else:
            effective_start = global_start_str

        start_ms = int(pd.Timestamp(effective_start, tz="UTC").timestamp() * 1000)
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
        split_df = df[mask].reset_index(drop=True)
        if len(split_df) > 0:
            if resample_4h:
                split_df = _resample_to_4h(split_df)
            # Inject asset_class — universal feature for cross-asset generalization
            split_df["asset_class"] = ASSET_CLASS.get(symbol, 0)
            # Zero funding_rate for non-perp assets (XAU, SP500) so model
            # doesn't learn perp-specific patterns as universal signals
            if ASSET_CLASS.get(symbol, 0) != 0:
                split_df["funding_rate"] = 0.0
            elif "funding_rate" not in split_df.columns:
                split_df["funding_rate"] = 0.0
            result[symbol] = split_df
    return result


# ---------------------------------------------------------------------------
# Backtesting engine (DO NOT CHANGE)
# ---------------------------------------------------------------------------

def run_backtest(strategy, data: dict, bar_interval_sec: int = 3600) -> BacktestResult:
    """
    Run strategy over data. Returns BacktestResult with full metrics.
    Enforces TIME_BUDGET.
    """
    t_start = time.time()

    # Build unified timeline with integer second precision
    all_timestamps = set()
    for symbol, df in data.items():
        ts_raw = df["timestamp"].astype(np.int64).values
        # Auto-detect milliseconds and convert to seconds
        ts_norm = np.where(ts_raw > 1e11, ts_raw // 1000, ts_raw)
        all_timestamps.update(ts_norm.tolist())
    timestamps = sorted(all_timestamps)

    if not timestamps:
        return BacktestResult()

    # Index data by (symbol, timestamp) with integer second precision
    indexed = {}
    for symbol, df in data.items():
        df_int = df.copy()
        ts_raw = df_int["timestamp"].astype(np.int64).values
        df_int["timestamp"] = np.where(ts_raw > 1e11, ts_raw // 1000, ts_raw)
        indexed[symbol] = df_int.set_index("timestamp")

    # Portfolio state
    portfolio = PortfolioState(
        cash=INITIAL_CAPITAL,
        positions={},
        entry_prices={},
        equity=INITIAL_CAPITAL,
        timestamp=0,
    )

    bars_per_year = 365.25 * 24 * 3600 / bar_interval_sec
    bars_per_funding = 8 * 3600 / bar_interval_sec  # bars in 8h funding period
    equity_curve = [INITIAL_CAPITAL]
    bar_returns = []
    trade_log = []
    total_volume = 0.0
    prev_equity = INITIAL_CAPITAL

    # History buffers
    history_buffers = {symbol: [] for symbol in data}

    for ts in timestamps:
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            break

        portfolio.timestamp = ts

        # Build bar data
        bar_data = {}
        for symbol in data:
            if symbol not in indexed or ts not in indexed[symbol].index:
                continue
            row = indexed[symbol].loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Update history buffer
            bar_dict = {
                "timestamp": ts,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "funding_rate": row.get("funding_rate", 0.0),
            }
            history_buffers[symbol].append(bar_dict)
            if len(history_buffers[symbol]) > LOOKBACK_BARS:
                history_buffers[symbol] = history_buffers[symbol][-LOOKBACK_BARS:]

            hist_df = pd.DataFrame(history_buffers[symbol])

            # Standardize BarData timestamp to seconds
            bar_ts = ts
            if bar_ts > 1e11: bar_ts //= 1000

            bar_data[symbol] = BarData(
                symbol=symbol,
                timestamp=int(bar_ts),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                funding_rate=row.get("funding_rate", 0.0),
                history=hist_df,
            )

        if not bar_data:
            continue

        # Update portfolio equity (mark-to-market)
        unrealized_pnl = 0.0
        for sym, pos_notional in portfolio.positions.items():
            if sym in bar_data:
                current_price = bar_data[sym].close
                entry_price = portfolio.entry_prices.get(sym, current_price)
                if entry_price > 0:
                    price_change = (current_price - entry_price) / entry_price
                    unrealized_pnl += pos_notional * price_change

        portfolio.equity = portfolio.cash + sum(abs(v) for v in portfolio.positions.values()) + unrealized_pnl

        # Apply funding rates (on open positions)
        for sym, pos_notional in list(portfolio.positions.items()):
            if sym in bar_data:
                fr = bar_data[sym].funding_rate
                # Funding: longs pay when positive, shorts receive
                # Applied every 8h, but we have hourly bars so scale by 1/8
                funding_payment = pos_notional * fr / bars_per_funding
                portfolio.cash -= funding_payment

        # Get signals from strategy
        signals = strategy.on_bar(bar_data, portfolio)

        # Execute signals
        for sig in (signals or []):
            if sig.symbol not in bar_data:
                continue

            current_price = bar_data[sig.symbol].close
            current_pos = portfolio.positions.get(sig.symbol, 0.0)
            delta = sig.target_position - current_pos

            if abs(delta) < 1.0:  # < $1 change, skip
                continue

            # Check leverage constraint
            new_positions = dict(portfolio.positions)
            new_positions[sig.symbol] = sig.target_position
            total_exposure = sum(abs(v) for v in new_positions.values())
            if total_exposure > portfolio.equity * MAX_LEVERAGE:
                continue

            # Apply slippage and fees
            slippage = current_price * SLIPPAGE_BPS / 10000
            fee_rate = TAKER_FEE
            if delta > 0:  # buying
                exec_price = current_price + slippage
            else:  # selling
                exec_price = current_price - slippage

            fee = abs(delta) * fee_rate
            portfolio.cash -= fee
            total_volume += abs(delta)

            # Update position
            if sig.target_position == 0:
                # Closing position — realize PnL
                if sig.symbol in portfolio.entry_prices:
                    entry = portfolio.entry_prices[sig.symbol]
                    if entry > 0:
                        pnl = current_pos * (exec_price - entry) / entry
                        portfolio.cash += abs(current_pos) + pnl
                    del portfolio.entry_prices[sig.symbol]
                if sig.symbol in portfolio.positions:
                    del portfolio.positions[sig.symbol]
                trade_log.append(("close", sig.symbol, delta, exec_price, pnl if 'pnl' in dir() else 0))
            else:
                if current_pos == 0:
                    # Opening new position
                    portfolio.cash -= abs(sig.target_position)
                    portfolio.positions[sig.symbol] = sig.target_position
                    portfolio.entry_prices[sig.symbol] = exec_price
                    trade_log.append(("open", sig.symbol, delta, exec_price, 0))
                else:
                    # Modifying position
                    old_notional = abs(current_pos)
                    old_entry = portfolio.entry_prices.get(sig.symbol, exec_price)
                    # Realize PnL on reduced portion
                    if abs(sig.target_position) < abs(current_pos):
                        reduced = abs(current_pos) - abs(sig.target_position)
                        if old_entry > 0:
                            pnl = (current_pos / abs(current_pos)) * reduced * (exec_price - old_entry) / old_entry
                        else:
                            pnl = 0
                        portfolio.cash += reduced + pnl
                    elif abs(sig.target_position) > abs(current_pos):
                        added = abs(sig.target_position) - abs(current_pos)
                        portfolio.cash -= added
                        # Weighted average entry
                        if old_notional + added > 0:
                            new_entry = (old_entry * old_notional + exec_price * added) / (old_notional + added)
                            portfolio.entry_prices[sig.symbol] = new_entry
                    portfolio.positions[sig.symbol] = sig.target_position
                    trade_log.append(("modify", sig.symbol, delta, exec_price, 0))

        # Recalculate equity after trades
        unrealized_pnl = 0.0
        for sym, pos_notional in portfolio.positions.items():
            if sym in bar_data:
                current_price = bar_data[sym].close
                entry_price = portfolio.entry_prices.get(sym, current_price)
                if entry_price > 0:
                    price_change = (current_price - entry_price) / entry_price
                    unrealized_pnl += pos_notional * price_change

        current_equity = portfolio.cash + sum(abs(v) for v in portfolio.positions.values()) + unrealized_pnl
        equity_curve.append(current_equity)

        # Hourly return
        if prev_equity > 0:
            bar_returns.append((current_equity - prev_equity) / prev_equity)
        prev_equity = current_equity

        # Liquidation check
        if current_equity < INITIAL_CAPITAL * 0.01:
            break

    t_end = time.time()

    duration_days = 0.0
    if len(timestamps) > 1:
        duration_days = (timestamps[-1] - timestamps[0]) / (1000 * 60 * 60 * 24)

    # Compute metrics
    returns = np.array(bar_returns) if bar_returns else np.array([0.0])
    eq = np.array(equity_curve)

    # Sharpe ratio (annualized from hourly)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Total return
    final_equity = eq[-1] if len(eq) > 0 else INITIAL_CAPITAL
    total_return_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / np.where(peak > 0, peak, 1)
    max_drawdown_pct = drawdown.max() * 100

    # Win rate and profit factor
    trade_pnls = [t[4] for t in trade_log if t[0] == "close"]
    num_trades = len(trade_log)
    if trade_pnls:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        win_rate_pct = len(wins) / len(trade_pnls) * 100 if trade_pnls else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0

    # Annual turnover
    num_bars = len(timestamps)
    if num_bars > 0:
        annual_turnover = total_volume * (bars_per_year / num_bars)
    else:
        annual_turnover = 0.0

    return BacktestResult(
        sharpe=sharpe,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        num_trades=num_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        annual_turnover=annual_turnover,
        backtest_seconds=t_end - t_start,
        duration_days=duration_days,
        equity_curve=equity_curve,
        trade_log=trade_log,
    )

# ---------------------------------------------------------------------------
# Evaluation metric (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def compute_score(result: BacktestResult) -> float:
    """
    Composite risk-adjusted score (HIGHER is better).

    score = sharpe * sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty

    Hard cutoffs for degenerate strategies.
    """
    # Hard cutoffs
    if result.num_trades < 10:
        return -999.0
    if result.duration_days > 0 and (result.num_trades / result.duration_days) < 1.0:
        return -999.0    # Strictly enforced minimum: 1 trade per day
    if result.max_drawdown_pct > 50.0:
        return -999.0
    final_equity = result.equity_curve[-1] if result.equity_curve else INITIAL_CAPITAL
    if final_equity < INITIAL_CAPITAL * 0.5:
        return -999.0

    # Trade count factor: full credit at 50+ trades
    trade_count_factor = min(result.num_trades / 50.0, 1.0)

    # Drawdown penalty: no penalty below 15%, then 5x per additional percent
    drawdown_penalty = max(0, result.max_drawdown_pct - 15.0) * 0.05

    # Turnover penalty: penalize excessive churning (>500x annual)
    turnover_ratio = result.annual_turnover / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 0
    turnover_penalty = max(0, turnover_ratio - 500) * 0.001

    score = result.sharpe * math.sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty
    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autotrader")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to download (default: all)")
    parser.add_argument("--mode", choices=["data", "15m"], default="data",
                        help="'data' = 1H candles (default), '15m' = 15-minute candles")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    if args.mode == "15m":
        print("Downloading 15-minute data...")
        download_15m_data(args.symbols)
    else:
        print("Downloading data...")
        download_data(args.symbols)
    print()
    print("Done! Ready to backtest.")
