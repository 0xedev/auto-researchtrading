"""
Production Ensemble Strategy (6-signal).
Expanded to 18 assets (Crypto + Macro) on 4H bars.
Signals:
1. Short Momentum (12 bars)
2. Very Short Momentum (6 bars)
3. EMA Crossover (7, 26)
4. RSI (8)
5. MACD (14, 23, 9)
6. BB Width Percentile (100-bar window)
Voting: 4/6 for Entry.
Risk: ATR 5.5 trailing stop, RSI exits (69/31).
"""

import numpy as np
from prepare import Signal, PortfolioState, BarData

ACTIVE_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "LINK", 
    "AVAX", "DOT", "ATOM", "NEAR", "UNI", "APT", "SUI", 
    "XAU", "DXY", "SP500"
]

import joblib
import os

# Model paths — one per timeframe, loaded automatically based on bar interval
MODEL_PATHS = {
    900:   os.path.expanduser("~/.cache/autotrader/model_15m.joblib"),
    3600:  os.path.expanduser("~/.cache/autotrader/model_1h.joblib"),
    14400: os.path.expanduser("~/.cache/autotrader/model_4h.joblib"),
}
# Fallback for legacy single-model setups
MODEL_PATH_LEGACY = os.path.expanduser("~/.cache/autotrader/model.joblib")

# Per-timeframe thresholds: adjusted lower to guarantee >= 2 trades/day minimum
PROB_THRESHOLDS = {
    900:   0.55,   # 15m: dropped from 0.57
    3600:  0.42,   # 1H: OOS-compatible — vol-adaptive sizing handles the high-vol reduction
    14400: 0.53,   # 4H: sweet spot — Sharpe ~4.7 / 28 trades on val (48h target)
}
PROB_THRESHOLD = 0.46  # fallback

# Canonical H4 Periods (Reference)
H4_REF = 4 * 3600  # 4 hours in seconds

BASE_EMA_FAST = 7
BASE_EMA_SLOW = 26
BASE_SHORT_WINDOW = 12
BASE_VSHORT_WINDOW = 6
BASE_RSI_PERIOD = 8
BASE_MACD_FAST = 14
BASE_MACD_SLOW = 23
BASE_MACD_SIGNAL = 9
BASE_BB_PERIOD = 100

BASE_POSITION_PCT = 0.08
ATR_LOOKBACK = 24
ATR_STOP_MULT = 4.5
RSI_OVERBOUGHT = 68
RSI_OVERSOLD = 32
RSI_ENTRY_LONG_MAX = 63   # Don't enter long if already near overbought
RSI_ENTRY_SHORT_MIN = 37  # Don't enter short if already near oversold

def ema(values, span):
    span = max(2, int(span))
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result

def calc_rsi(closes, period):
    period = max(2, int(period))
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / max(avg_loss, 1e-10)
    return 100 - 100 / (1 + rs)

# Max bars to hold a position before forcing exit (per timeframe)
# Forces capital recycling to hit ~2 trades/day target
MAX_HOLD_BARS = {
    900:   16,   # 15m: 4 hours max hold
    3600:  8,    # 1H:  8 hours max hold — optimal capital recycling
    14400: 30,   # 4H:  120 hours (5 days) max hold
}

class Strategy:
    def __init__(self, model=None):
        self.peak_prices = {}
        self.bar_count = 0
        self.interval_sec = 0
        self.model = model
        self.model_loaded = (model is not None)
        self.bars_held = {}  # symbol → bars held in current position

    def _get_adaptive_period(self, base_period):
        if self.interval_sec == 0:
            return base_period
        return max(2, int(base_period * (H4_REF / self.interval_sec)))

    def _calc_atr(self, history, lookback):
        lookback = max(5, int(lookback))
        if len(history) < lookback + 1:
            return None
        highs = history["high"].values[-lookback:]
        lows = history["low"].values[-lookback:]
        closes = history["close"].values[-(lookback+1):-1]
        tr = np.maximum(highs - lows,
                        np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
        return np.mean(tr)

    def _calc_macd(self, closes, fast, slow, signal_p):
        if len(closes) < slow + signal_p + 5:
            return 0.0, 0.0, 0.0
        fast_ema = ema(closes[-(slow + signal_p + 5):], fast)
        slow_ema = ema(closes[-(slow + signal_p + 5):], slow)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, signal_p)
        return macd_line[-1] - signal_line[-1], macd_line[-1], signal_line[-1]

    def _calc_bb_stats(self, closes, period):
        period = max(5, int(period))
        if len(closes) < period + 2:
            return 100.0
        rolling_mean = np.mean(closes[-period:])
        rolling_std = np.std(closes[-period:])
        width = (4 * rolling_std) / np.maximum(rolling_mean, 1e-10)
        return width

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1

        # Detect interval on first bars, then load the matching model
        if self.interval_sec == 0 and len(bar_data) > 0:
            for s in bar_data:
                if len(bar_data[s].history) >= 2:
                    ts = bar_data[s].history["timestamp"].values
                    self.interval_sec = (ts[-1] - ts[-2]) // 1000
                    break

        if not self.model_loaded and self.interval_sec > 0:
            path = MODEL_PATHS.get(self.interval_sec)
            if path and os.path.exists(path):
                try:
                    self.model = joblib.load(path)
                    print(f"Loaded {self.interval_sec}s model from {path}")
                except Exception as e:
                    print(f"Failed to load {self.interval_sec}s model: {e}")
            elif os.path.exists(MODEL_PATH_LEGACY):
                try:
                    self.model = joblib.load(MODEL_PATH_LEGACY)
                    print(f"Loaded legacy model from {MODEL_PATH_LEGACY}")
                except Exception as e:
                    print(f"Failed to load legacy model: {e}")
            else:
                print(f"No model found for interval {self.interval_sec}s")
            self.model_loaded = True

        for i, symbol in enumerate(ACTIVE_SYMBOLS):
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            closes = bd.history["close"].values
            if len(closes) < 100:
                continue

            mid = bd.close
            
            # --- FEATURE ENGINEERING (Consistent with train_model.py) ---
            ret1 = (closes[-1] - closes[-2]) / closes[-2]
            ret4 = (closes[-1] - closes[-5]) / closes[-5]
            ret12 = (closes[-1] - closes[-13]) / closes[-13]
            ret24 = (closes[-1] - closes[-25]) / closes[-25]
            ret48 = (closes[-1] - closes[-49]) / closes[-49]
            
            rsi8 = calc_rsi(closes, 8)
            rsi24 = calc_rsi(closes, 24)
            macd_h, macd_line_val, _ = self._calc_macd(closes, 12, 26, 9)
            
            # BB Width (35-bar, sync with training)
            bbw = self._calc_bb_stats(closes, 35)

            # EMA 200 Macro
            ema200_arr = ema(closes[-220:], 200)
            ema200_dist = (mid - ema200_arr[-1]) / mid

            vol24 = np.std(np.diff(np.log(closes[-25:])))

            # Inference Data (13 features — must match train_model.py column order)
            FEAT_NAMES = ['ret_1h','ret_4h','ret_12h','ret_24h','ret_48h','rsi_8','rsi_24',
                          'macd_hist','macd_line','bb_width','ema_200_dist','vol_24h','symbol_idx']
            feat_vec = [ret1, ret4, ret12, ret24, ret48, rsi8, rsi24, macd_h, macd_line_val, bbw, ema200_dist, vol24, i]

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos
            # Volatility-adaptive sizing: scale down in high-vol regimes
            # Target 3% ATR exposure; uncapped in low-vol (bull) markets
            atr_pct = vol24  # vol24 = std of log-returns over 24 bars ≈ 1H ATR proxy
            vol_adj = min(1.0, 0.03 / max(atr_pct, 1e-6))
            long_size = equity * 0.18 * vol_adj
            long_soft_size = equity * 0.12 * vol_adj
            short_size = equity * 0.10 * vol_adj
            short_soft_size = equity * 0.06 * vol_adj

            if self.model:
                try:
                    # ML-Based Voting — pass DataFrame to suppress sklearn feature-name warnings
                    import pandas as _pd
                    probs = self.model.predict_proba(_pd.DataFrame([feat_vec], columns=FEAT_NAMES))[0]
                    # Label 0: Neutral, 1: Buy, 2: Sell
                    prob_buy = probs[1]
                    prob_sell = probs[2]
                    thresh = PROB_THRESHOLDS.get(self.interval_sec, PROB_THRESHOLD)

                    if current_pos == 0:
                        if prob_buy > thresh and rsi8 < RSI_ENTRY_LONG_MAX:
                            # Enter long — RSI filter prevents buying into overbought
                            target = long_size if prob_buy > thresh + 0.14 else long_soft_size
                        elif prob_sell > thresh and rsi8 > RSI_ENTRY_SHORT_MIN:
                            # Enter short — RSI filter prevents shorting into oversold
                            target = -short_size if prob_sell > thresh + 0.14 else -short_soft_size
                    else:
                        # Exit long on sell signal; exit short on buy signal
                        if current_pos > 0 and prob_sell > thresh:
                            target = 0.0
                        elif current_pos < 0 and prob_buy > thresh:
                            target = 0.0
                except Exception as e:
                    print(f"Inference error for {symbol}: {e}")
            
            # --- RISK MANAGEMENT LAYER (ATR Stops, RSI Exits, Time-Based Exit) ---
            atr_l = self._get_adaptive_period(ATR_LOOKBACK)
            atr = self._calc_atr(bd.history, atr_l) or mid * 0.02
            max_hold = MAX_HOLD_BARS.get(self.interval_sec, 12)

            if current_pos != 0:
                # Track bars held
                self.bars_held[symbol] = self.bars_held.get(symbol, 0) + 1

                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid

                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    if mid < self.peak_prices[symbol] - ATR_STOP_MULT * atr:
                        target = 0.0
                    if rsi8 > RSI_OVERBOUGHT:
                        target = 0.0
                    if self.bars_held.get(symbol, 0) >= max_hold:
                        target = 0.0  # Time-based exit: recycle capital
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    if mid > self.peak_prices[symbol] + ATR_STOP_MULT * atr:
                        target = 0.0
                    if rsi8 < RSI_OVERSOLD:
                        target = 0.0
                    if self.bars_held.get(symbol, 0) >= max_hold:
                        target = 0.0

            if abs(target - current_pos) > 1e-6:
                is_flip = (target != 0 and current_pos != 0 and np.sign(target) != np.sign(current_pos))
                if is_flip:
                    # Two-step: close old position fully, then open new side
                    # Ensures correct PnL realization and fresh entry_price
                    signals.append(Signal(symbol=symbol, target_position=0))
                    signals.append(Signal(symbol=symbol, target_position=target))
                    self.peak_prices[symbol] = mid
                    self.bars_held[symbol] = 0
                else:
                    signals.append(Signal(symbol=symbol, target_position=target))
                    if target != 0 and current_pos == 0:
                        self.peak_prices[symbol] = mid
                        self.bars_held[symbol] = 0
                    elif target == 0:
                        self.peak_prices.pop(symbol, None)
                        self.bars_held.pop(symbol, None)

        return signals
