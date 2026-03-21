"""
strategy.py — ML-powered trading strategy.

Primary path: load trained Transformer/LSTM model → compute features →
  predict direction + confidence → Kelly-sized positions.

Fallback: if model.pt not found, uses proven rule-based signals (Exp32).

The agent modifies this file (and train.py) in the autonomous loop.
"""

import os
import numpy as np
from prepare import Signal, PortfolioState, BarData

# Optional ML imports — fail gracefully
try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from features import compute_feature_vector, N_FEATURES
    FEATURES_OK = True
except ImportError:
    FEATURES_OK = False

MODEL_PATH = os.path.expanduser("~/.cache/autotrader/model.pt")

# ── Shared constants ───────────────────────────────────────────────────────────
ACTIVE_SYMBOLS = ["BTC", "ETH", "SOL"]
SYMBOL_WEIGHTS = {"BTC": 0.33, "ETH": 0.33, "SOL": 0.33}

# Risk management (applies to BOTH ML and rule-based paths)
ATR_LOOKBACK     = 24
ATR_STOP_MULT    = 4.5
VOL_LOOKBACK     = 36
TARGET_VOL       = 0.015
BASE_POSITION_PCT = 0.09
COOLDOWN_BARS    = 2
RSI_OVERBOUGHT   = 70
RSI_OVERSOLD     = 30
TAKE_PROFIT_PCT  = 99.0
DD_REDUCE_THRESHOLD = 0.20
DD_REDUCE_SCALE  = 0.5
CORR_LOOKBACK    = 72
HIGH_CORR_THRESHOLD = 0.85

# ML-specific
DIRECTION_THRESHOLD = 0.15   # min |predicted return| to open position
MIN_CONFIDENCE      = 0.55   # min max-softmax confidence to trade
KELLY_MAX_FRACTION  = 0.20   # cap Kelly fraction at 20% of equity

# Rule-based fallback constants (Exp32 — score 9.382 on val)
SHORT_WINDOW = 6
MED_WINDOW   = 12
MED2_WINDOW  = 24
LONG_WINDOW  = 36
EMA_FAST     = 7
EMA_SLOW     = 26
RSI_PERIOD   = 8
RSI_BULL     = 50
RSI_BEAR     = 50
MACD_FAST    = 14
MACD_SLOW    = 23
MACD_SIGNAL  = 9
BB_PERIOD    = 7
FUNDING_LOOKBACK = 24
BASE_THRESHOLD   = 0.012
BTC_OPPOSE_THRESHOLD = -99.0
PYRAMID_THRESHOLD = 0.015
PYRAMID_SIZE      = 0.0
HIGH_CORR_RULE    = 99.0
MIN_VOTES         = 4   # out of 6 signals

# ── Utility functions ──────────────────────────────────────────────────────────

def _ema(values, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _calc_rsi(closes, period):
    if len(closes) < period + 1:
        return 50.0
    d = np.diff(closes[-(period + 1):])
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    rs = np.mean(g) / max(np.mean(l), 1e-10)
    return 100 - 100 / (1 + rs)


def _calc_macd(closes):
    if len(closes) < MACD_SLOW + MACD_SIGNAL + 5:
        return 0.0
    src = closes[-(MACD_SLOW + MACD_SIGNAL + 5):]
    ef = _ema(src, MACD_FAST)
    es = _ema(src, MACD_SLOW)
    macd = ef - es
    sig = _ema(macd, MACD_SIGNAL)
    return macd[-1] - sig[-1]


def _calc_bb_width_pctile(closes, period):
    if len(closes) < period * 3:
        return 50.0
    widths = []
    for i in range(period * 2, len(closes)):
        w = closes[i - period:i]
        sma = np.mean(w)
        widths.append((2 * np.std(w)) / sma if sma > 0 else 0)
    if len(widths) < 2:
        return 50.0
    return 100 * np.sum(np.array(widths) <= widths[-1]) / len(widths)


def _calc_atr(history, lookback):
    if len(history) < lookback + 1:
        return None
    h = history["high"].values[-lookback:]
    l = history["low"].values[-lookback:]
    c = history["close"].values[-(lookback + 1):-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    return np.mean(tr)


def _calc_vol(closes, lookback):
    if len(closes) < lookback:
        return TARGET_VOL
    return max(np.std(np.diff(np.log(closes[-lookback:]))), 1e-6)


def _kelly_size(confidence, win_rate, avg_win, avg_loss, equity, weight):
    """Compute Kelly-fraction position size."""
    if avg_loss < 1e-10 or avg_win < 1e-10:
        return equity * BASE_POSITION_PCT * weight
    kelly = win_rate / avg_loss - (1 - win_rate) / avg_win
    kelly = max(0.0, min(kelly, KELLY_MAX_FRACTION))
    confidence_scale = max(0.0, (confidence - MIN_CONFIDENCE) / (1 - MIN_CONFIDENCE))
    return equity * kelly * confidence_scale * weight


# ── ML model loader ────────────────────────────────────────────────────────────

class _ModelWrapper:
    """Thin wrapper around a loaded PyTorch model for inference."""

    def __init__(self, checkpoint):
        if not TORCH_OK:
            raise ImportError("torch not available")
        from train import TransformerModel, LSTMModel, CNNTransformerModel

        arch = checkpoint.get("arch", "transformer")
        n_features = checkpoint.get("n_features", N_FEATURES)
        seq_len = checkpoint.get("seq_len", 168)
        d_model = checkpoint.get("d_model", 256)
        n_heads = checkpoint.get("n_heads", 8)
        n_layers = checkpoint.get("n_layers", 4)
        d_ffn = checkpoint.get("d_ffn", 512)

        if arch == "lstm":
            model = LSTMModel(n_features=n_features)
        elif arch == "cnn_transformer":
            model = CNNTransformerModel(n_features=n_features, d_model=d_model)
        else:
            model = TransformerModel(
                n_features=n_features, d_model=d_model, n_heads=n_heads,
                n_layers=n_layers, d_ffn=d_ffn, seq_len=seq_len,
            )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.model = model
        self.seq_len = seq_len
        self.n_features = n_features
        self.val_sharpe = checkpoint.get("val_sharpe", 0.0)
        self.arch = arch

    def predict(self, feature_history: np.ndarray) -> float:
        """
        Predict return weight for a single symbol.
        feature_history: (seq_len, n_features) numpy array
        Returns: float in [-1, 1] — predicted position weight
        """
        with torch.no_grad():
            x = torch.from_numpy(feature_history.astype(np.float32)).unsqueeze(0)
            pred = self.model(x)
            return float(pred.item())


# ── Main Strategy class ────────────────────────────────────────────────────────

class Strategy:
    def __init__(self):
        # Shared state
        self.entry_prices = {}
        self.peak_prices = {}
        self.atr_at_entry = {}
        self.peak_equity = 100_000.0
        self.exit_bar = {}
        self.bar_count = 0
        self.pyramided = {}

        # ML: feature history buffers (symbol → list of feature vectors)
        self.feature_history = {sym: [] for sym in ACTIVE_SYMBOLS}
        self.daily_macro = None     # updated every 24 bars
        self.macro_last_update = -999

        # Rule-based: BTC momentum tracking
        self.btc_momentum = 0.0

        # Load model
        self.model = self._load_model()
        if self.model:
            print(f"[Strategy] ML model loaded (arch={self.model.arch}, "
                  f"val_sharpe={self.model.val_sharpe:.3f})")
        else:
            print("[Strategy] No model found — using rule-based fallback (Exp32)")

    def _load_model(self):
        if not TORCH_OK or not FEATURES_OK:
            return None
        if not os.path.exists(MODEL_PATH):
            return None
        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            return _ModelWrapper(checkpoint)
        except Exception as e:
            print(f"[Strategy] model load failed: {e} — using rule-based")
            return None

    def on_bar(self, bar_data, portfolio):
        self.bar_count += 1

        equity = max(portfolio.equity, portfolio.cash)
        self.peak_equity = max(self.peak_equity, equity)
        current_dd = (self.peak_equity - equity) / max(self.peak_equity, 1.0)
        dd_scale = 1.0
        if current_dd > DD_REDUCE_THRESHOLD:
            dd_scale = max(DD_REDUCE_SCALE,
                          1.0 - (current_dd - DD_REDUCE_THRESHOLD) * 5)

        if self.model is not None:
            return self._ml_on_bar(bar_data, portfolio, equity, dd_scale)
        return self._rule_based_on_bar(bar_data, portfolio, equity, dd_scale)

    # ── ML inference path ────────────────────────────────────────────────────

    def _ml_on_bar(self, bar_data, portfolio, equity, dd_scale):
        signals = []
        seq_len = self.model.seq_len

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < 50:
                continue

            mid = bd.close

            # ── Build feature vector ─────────────────────────────────────────
            if FEATURES_OK:
                ts_ms = bd.timestamp
                fv = compute_feature_vector(bd, self.daily_macro, ts_ms)
                self.feature_history[symbol].append(fv)
                if len(self.feature_history[symbol]) > seq_len + 10:
                    self.feature_history[symbol] = self.feature_history[symbol][-(seq_len + 10):]
            else:
                continue

            if len(self.feature_history[symbol]) < seq_len:
                continue

            feat_seq = np.array(self.feature_history[symbol][-seq_len:], dtype=np.float32)
            feat_seq = np.where(np.isfinite(feat_seq), feat_seq, 0.0)

            # ── Model inference ──────────────────────────────────────────────
            try:
                pred_weight = self.model.predict(feat_seq)
            except Exception:
                continue

            # ── Risk management ──────────────────────────────────────────────
            atr = _calc_atr(bd.history, ATR_LOOKBACK)
            if atr is None:
                atr = self.atr_at_entry.get(symbol, mid * 0.02)
            realized_vol = _calc_vol(bd.history["close"].values, VOL_LOOKBACK)

            weight = SYMBOL_WEIGHTS.get(symbol, 0.33)
            confidence = min(abs(pred_weight), 1.0)

            # Kelly-based sizing: simple estimate from recent returns
            closes = bd.history["close"].values
            if len(closes) >= 50:
                rets = np.diff(np.log(closes[-50:]))
                wins = rets[rets > 0]
                losses = rets[rets <= 0]
                if len(wins) > 0 and len(losses) > 0:
                    size = _kelly_size(
                        confidence,
                        len(wins) / len(rets),
                        np.mean(wins),
                        np.mean(np.abs(losses)),
                        equity, weight,
                    )
                else:
                    size = equity * BASE_POSITION_PCT * weight
            else:
                size = equity * BASE_POSITION_PCT * weight

            # Vol scaling
            vol_ratio = realized_vol / max(TARGET_VOL, 1e-6)
            vol_scale = 1.0 / max(vol_ratio, 0.5)
            vol_scale = np.clip(vol_scale, 0.3, 2.0)
            size = size * vol_scale * dd_scale

            current_pos = portfolio.positions.get(symbol, 0.0)
            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < COOLDOWN_BARS
            target = current_pos

            rsi = _calc_rsi(closes, RSI_PERIOD)

            # ── Entry signals ────────────────────────────────────────────────
            if current_pos == 0 and not in_cooldown:
                if pred_weight > DIRECTION_THRESHOLD and confidence >= MIN_CONFIDENCE:
                    target = size
                    self.pyramided[symbol] = False
                elif pred_weight < -DIRECTION_THRESHOLD and confidence >= MIN_CONFIDENCE:
                    target = -size
                    self.pyramided[symbol] = False

            # ── Exit / stop management ───────────────────────────────────────
            elif current_pos != 0:
                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid

                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    stop = self.peak_prices[symbol] - ATR_STOP_MULT * atr
                    if mid < stop:
                        target = 0.0
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    stop = self.peak_prices[symbol] + ATR_STOP_MULT * atr
                    if mid > stop:
                        target = 0.0

                # RSI extremes exit
                if current_pos > 0 and rsi > RSI_OVERBOUGHT:
                    target = 0.0
                elif current_pos < 0 and rsi < RSI_OVERSOLD:
                    target = 0.0

                # Take profit
                if symbol in self.entry_prices:
                    pnl = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                    if current_pos < 0:
                        pnl = -pnl
                    if pnl > TAKE_PROFIT_PCT:
                        target = 0.0

                # Reverse on strong opposing signal
                if (not in_cooldown and
                        current_pos > 0 and pred_weight < -DIRECTION_THRESHOLD and
                        confidence >= MIN_CONFIDENCE):
                    target = -size
                elif (not in_cooldown and
                        current_pos < 0 and pred_weight > DIRECTION_THRESHOLD and
                        confidence >= MIN_CONFIDENCE):
                    target = size

            # ── Emit signal ──────────────────────────────────────────────────
            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                self._update_tracking(symbol, target, current_pos, mid, bd)

        # Update daily macro features once every 24 bars
        if (self.bar_count - self.macro_last_update) >= 24 and FEATURES_OK:
            self._update_macro(bar_data)
            self.macro_last_update = self.bar_count

        return signals

    def _update_macro(self, bar_data):
        """Refresh daily macro features from cached data."""
        try:
            from prepare_extended import load_daily_data
            from features import compute_macro_features
            daily = load_daily_data()
            if daily:
                self.daily_macro = compute_macro_features(daily)
        except Exception:
            pass

    # ── Rule-based fallback (Exp32 — proven score 9.382) ─────────────────────

    def _rule_based_on_bar(self, bar_data, portfolio, equity, dd_scale):
        signals = []

        if "BTC" in bar_data and len(bar_data["BTC"].history) >= LONG_WINDOW + 1:
            btc_c = bar_data["BTC"].history["close"].values
            self.btc_momentum = (btc_c[-1] - btc_c[-MED2_WINDOW]) / btc_c[-MED2_WINDOW]

        btc_eth_corr = self._calc_correlation(bar_data)

        for symbol in ACTIVE_SYMBOLS:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            min_bars = max(LONG_WINDOW, EMA_SLOW, MACD_SLOW + MACD_SIGNAL + 5, BB_PERIOD * 3) + 1
            if len(bd.history) < min_bars:
                continue

            closes = bd.history["close"].values
            mid = bd.close

            realized_vol = _calc_vol(closes, VOL_LOOKBACK)
            vol_ratio = realized_vol / TARGET_VOL
            dyn_threshold = BASE_THRESHOLD * (0.3 + vol_ratio * 0.7)
            dyn_threshold = np.clip(dyn_threshold, 0.005, 0.020)

            ret_vshort = (closes[-1] - closes[-SHORT_WINDOW]) / closes[-SHORT_WINDOW]
            ret_short  = (closes[-1] - closes[-MED_WINDOW])  / closes[-MED_WINDOW]

            mom_bull   = ret_short > dyn_threshold
            mom_bear   = ret_short < -dyn_threshold
            vshort_bull = ret_vshort > dyn_threshold * 0.7
            vshort_bear = ret_vshort < -dyn_threshold * 0.7

            ef = _ema(closes[-(EMA_SLOW + 10):], EMA_FAST)
            es = _ema(closes[-(EMA_SLOW + 10):], EMA_SLOW)
            ema_bull = ef[-1] > es[-1]
            ema_bear = ef[-1] < es[-1]

            rsi = _calc_rsi(closes, RSI_PERIOD)
            rsi_bull = rsi > RSI_BULL
            rsi_bear = rsi < RSI_BEAR

            macd_hist = _calc_macd(closes)
            macd_bull = macd_hist > 0
            macd_bear = macd_hist < 0

            bb_pctile = _calc_bb_width_pctile(closes, BB_PERIOD)
            bb_compressed = bb_pctile < 90

            bull_votes = sum([mom_bull, vshort_bull, ema_bull, rsi_bull, macd_bull, bb_compressed])
            bear_votes = sum([mom_bear, vshort_bear, ema_bear, rsi_bear, macd_bear, bb_compressed])

            btc_confirm = True
            if symbol != "BTC":
                if bull_votes >= MIN_VOTES and self.btc_momentum < BTC_OPPOSE_THRESHOLD:
                    btc_confirm = False
                if bear_votes >= MIN_VOTES and self.btc_momentum > -BTC_OPPOSE_THRESHOLD:
                    btc_confirm = False

            bullish = bull_votes >= MIN_VOTES and btc_confirm
            bearish = bear_votes >= MIN_VOTES and btc_confirm

            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < COOLDOWN_BARS

            weight = SYMBOL_WEIGHTS.get(symbol, 0.33)
            if btc_eth_corr > HIGH_CORR_RULE and symbol == "SOL":
                weight *= 0.5
            size = equity * BASE_POSITION_PCT * weight * dd_scale

            funding = bd.history["funding_rate"].values[-FUNDING_LOOKBACK:]
            avg_funding = np.mean(funding) if len(funding) >= FUNDING_LOOKBACK else 0.0

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            if current_pos == 0:
                if not in_cooldown:
                    if bullish:
                        target = size
                        self.pyramided[symbol] = False
                    elif bearish:
                        target = -size
                        self.pyramided[symbol] = False
            else:
                atr = _calc_atr(bd.history, ATR_LOOKBACK) or self.atr_at_entry.get(symbol, mid * 0.02)

                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid

                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    if mid < self.peak_prices[symbol] - ATR_STOP_MULT * atr:
                        target = 0.0
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    if mid > self.peak_prices[symbol] + ATR_STOP_MULT * atr:
                        target = 0.0

                if symbol in self.entry_prices:
                    pnl = (mid - self.entry_prices[symbol]) / self.entry_prices[symbol]
                    if current_pos < 0:
                        pnl = -pnl
                    if pnl > TAKE_PROFIT_PCT:
                        target = 0.0

                if current_pos > 0 and rsi > RSI_OVERBOUGHT:
                    target = 0.0
                elif current_pos < 0 and rsi < RSI_OVERSOLD:
                    target = 0.0

                if current_pos > 0 and bearish and not in_cooldown:
                    target = -size
                elif current_pos < 0 and bullish and not in_cooldown:
                    target = size

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                self._update_tracking(symbol, target, current_pos, mid, bd)

        return signals

    def _calc_correlation(self, bar_data):
        if "BTC" not in bar_data or "ETH" not in bar_data:
            return 0.5
        bh = bar_data["BTC"].history
        eh = bar_data["ETH"].history
        if len(bh) < CORR_LOOKBACK or len(eh) < CORR_LOOKBACK:
            return 0.5
        br = np.diff(np.log(bh["close"].values[-CORR_LOOKBACK:]))
        er = np.diff(np.log(eh["close"].values[-CORR_LOOKBACK:]))
        if len(br) < 10:
            return 0.5
        corr = np.corrcoef(br, er)[0, 1]
        return corr if np.isfinite(corr) else 0.5

    def _update_tracking(self, symbol, target, current_pos, mid, bd):
        """Update entry prices, peak prices, and exit bar tracking."""
        if target != 0 and current_pos == 0:
            self.entry_prices[symbol] = mid
            self.peak_prices[symbol] = mid
            self.atr_at_entry[symbol] = _calc_atr(bd.history, ATR_LOOKBACK) or mid * 0.02
        elif target == 0:
            self.entry_prices.pop(symbol, None)
            self.peak_prices.pop(symbol, None)
            self.atr_at_entry.pop(symbol, None)
            self.pyramided.pop(symbol, None)
            self.exit_bar[symbol] = self.bar_count
        elif (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
            self.entry_prices[symbol] = mid
            self.peak_prices[symbol] = mid
            self.atr_at_entry[symbol] = _calc_atr(bd.history, ATR_LOOKBACK) or mid * 0.02
            self.pyramided[symbol] = False
