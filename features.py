"""
features.py — Shared feature engineering library.

~200 features across 5 groups. All pure numpy, no pandas in hot path.
Shared by train.py (batch) and strategy.py (online inference).

Groups:
  A: Proven signals from current strategy          (~18)
  B: PDF-derived technical indicators              (~51)
  C: Cross-asset macro features (daily, filled)   (~40)
  D: Temporal / calendar features                  (~9)
  E: Statistical / regime features                (~10)
  Total: 128 per-symbol + macro                  = N_FEATURES

N_FEATURES = 128  (70 symbol + 9 temporal + 9 stat + 40 macro)
"""

import numpy as np

N_FEATURES = 128  # must match across features.py, train.py, strategy.py

# ─── Internal helpers ──────────────────────────────────────────────────────────

def _ema(values, span):
    alpha = 2.0 / (span + 1)
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _safe(x, lo=-3.0, hi=3.0):
    """Clip and ensure finite."""
    v = float(x)
    if not np.isfinite(v):
        return 0.0
    return max(lo, min(hi, v))


# ─── Group A: Proven signals ───────────────────────────────────────────────────

def _rsi(closes, period):
    if len(closes) < period + 1:
        return 0.0
    d = np.diff(closes[-(period + 1):])
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.mean(g)
    al = np.mean(l)
    rs = ag / max(al, 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return (rsi - 50) / 50  # [-1, 1]


def _ema_cross(closes, fast, slow):
    if len(closes) < slow + 5:
        return 0.0
    src = closes[-(slow + 5):]
    ef = _ema(src, fast)
    es = _ema(src, slow)
    mid = closes[-1]
    if mid < 1e-10:
        return 0.0
    return _safe((ef[-1] - es[-1]) / mid / 0.05, -1, 1)


def _macd_hist(closes, fast=14, slow=23, signal=9):
    need = slow + signal + 5
    if len(closes) < need:
        return 0.0
    src = closes[-need:]
    ef = _ema(src, fast)
    es = _ema(src, slow)
    macd = ef - es
    sig = _ema(macd, signal)
    hist = macd[-1] - sig[-1]
    mid = closes[-1]
    return _safe(hist / max(mid * 0.001, 1e-10) / 20, -1, 1)


def _bb_position(closes, period=20):
    if len(closes) < period:
        return 0.5
    w = closes[-period:]
    sma = np.mean(w)
    std = np.std(w)
    if std < 1e-10:
        return 0.5
    pos = (closes[-1] - (sma - 2 * std)) / (4 * std)
    return float(np.clip(pos, 0.0, 1.0))


def _bb_width(closes, period=20):
    """BB width as percentile of recent history [0,1]."""
    if len(closes) < period * 3:
        return 0.5
    widths = []
    for i in range(period, len(closes)):
        w = closes[i - period:i]
        sma = np.mean(w)
        if sma > 1e-10:
            widths.append(np.std(w) * 4 / sma)
    if len(widths) < 2:
        return 0.5
    return float(np.sum(np.array(widths) <= widths[-1]) / len(widths))


def _atr_norm(highs, lows, closes, period=24):
    if len(closes) < period + 1:
        return 0.02
    h = highs[-period:]
    l = lows[-period:]
    c = closes[-(period + 1):-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    return float(np.clip(np.mean(tr) / closes[-1], 0, 0.1))


def _momentum(closes, window):
    if len(closes) < window + 1:
        return 0.0
    return _safe((closes[-1] - closes[-window]) / closes[-window], -1, 1)


def _volume_ratio(volumes, period=24):
    if len(volumes) < period:
        return 0.5
    avg = np.mean(volumes[-period:])
    if avg < 1e-10:
        return 0.5
    return float(np.clip(volumes[-1] / avg / 5.0, 0, 1))


def _funding_features(funding, period=24):
    """Returns (avg_8h, avg_24h, std_24h) each in [-1,1]."""
    scale = 0.001
    if len(funding) < 8:
        return 0.0, 0.0, 0.0
    avg8 = _safe(np.mean(funding[-8:]) / scale / 5, -1, 1)
    if len(funding) >= period:
        avg24 = _safe(np.mean(funding[-period:]) / scale / 5, -1, 1)
        std24 = float(np.clip(np.std(funding[-period:]) / scale / 3, 0, 1))
    else:
        avg24, std24 = 0.0, 0.0
    return avg8, avg24, std24


def _sma_dist(closes, period):
    if len(closes) < period:
        return 0.0
    sma = np.mean(closes[-period:])
    return _safe((closes[-1] - sma) / sma / 0.1, -1, 1)


# ─── Group B: Technical indicators ────────────────────────────────────────────

def _cci(highs, lows, closes, period=20):
    if len(closes) < period:
        return 0.0
    tp = (highs[-period:] + lows[-period:] + closes[-period:]) / 3.0
    sma = np.mean(tp)
    mad = np.mean(np.abs(tp - sma))
    if mad < 1e-10:
        return 0.0
    return _safe((tp[-1] - sma) / (0.015 * mad) / 200, -1, 1)


def _keltner_position(highs, lows, closes, period=20):
    if len(closes) < period + 1:
        return 0.5
    ema_val = _ema(closes[-(period + 5):], period)[-1]
    h = highs[-period:]
    l = lows[-period:]
    c = closes[-(period + 1):-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    atr = np.mean(tr)
    upper = ema_val + 2.0 * atr
    lower = ema_val - 2.0 * atr
    return float(np.clip((closes[-1] - lower) / (upper - lower + 1e-10), 0.0, 1.0))


def _bb_squeeze(highs, lows, closes, bb_period=20, kc_period=20):
    """1 if BB is inside KC (squeeze = pending breakout)."""
    if len(closes) < max(bb_period, kc_period) + 2:
        return 0.0
    w = closes[-bb_period:]
    sma = np.mean(w)
    std = np.std(w)
    bb_up = sma + 2 * std
    bb_lo = sma - 2 * std
    ema_val = _ema(closes[-(kc_period + 5):], kc_period)[-1]
    h = highs[-kc_period:]
    l = lows[-kc_period:]
    c = closes[-(kc_period + 1):-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    atr = np.mean(tr)
    kc_up = ema_val + 1.5 * atr
    kc_lo = ema_val - 1.5 * atr
    return float(bb_up <= kc_up and bb_lo >= kc_lo)


def _adx(highs, lows, closes, period=14):
    """Returns (adx_norm, direction) both in [-1,1]."""
    if len(closes) < period * 2 + 2:
        return 0.0, 0.0
    h = highs[-(period * 2 + 1):]
    l = lows[-(period * 2 + 1):]
    c = closes[-(period * 2 + 2):]
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    mdm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.mean(tr[-period:])
    if atr < 1e-10:
        return 0.0, 0.0
    pdi = np.mean(pdm[-period:]) / atr * 100
    mdi = np.mean(mdm[-period:]) / atr * 100
    di_sum = pdi + mdi
    if di_sum < 1e-10:
        return 0.0, 0.0
    adx = np.abs(pdi - mdi) / di_sum
    direction = (pdi - mdi) / di_sum
    return float(np.clip(adx, 0, 1)), float(np.clip(direction, -1, 1))


def _stochastic(highs, lows, closes, k=14, d=3):
    """Returns (%K - 0.5)*2, (%D - 0.5)*2 in [-1,1]."""
    if len(closes) < k + d:
        return 0.0, 0.0
    hh = np.max(highs[-k:])
    ll = np.min(lows[-k:])
    rng = hh - ll
    if rng < 1e-10:
        return 0.0, 0.0
    stoch_k = (closes[-1] - ll) / rng
    # approximate %D
    k_vals = []
    for i in range(d):
        end = len(closes) - i
        if end < k:
            k_vals.append(0.5)
            continue
        hh_i = np.max(highs[end - k:end])
        ll_i = np.min(lows[end - k:end])
        r_i = hh_i - ll_i
        k_vals.append((closes[end - 1] - ll_i) / r_i if r_i > 1e-10 else 0.5)
    stoch_d = np.mean(k_vals)
    return (stoch_k - 0.5) * 2, (stoch_d - 0.5) * 2


def _williams_r(highs, lows, closes, period=14):
    if len(closes) < period:
        return 0.0
    hh = np.max(highs[-period:])
    ll = np.min(lows[-period:])
    rng = hh - ll
    if rng < 1e-10:
        return 0.0
    wr = (hh - closes[-1]) / rng  # 0=overbought, 1=oversold
    return (wr - 0.5) * 2  # [-1, 1]


def _cmf(highs, lows, closes, volumes, period=20):
    if len(closes) < period:
        return 0.0
    h = highs[-period:]
    l = lows[-period:]
    c = closes[-period:]
    v = volumes[-period:]
    rng = np.where(h - l < 1e-10, 1.0, h - l)
    mf = ((c - l) - (h - c)) / rng * v
    vol_sum = np.sum(v)
    return float(np.clip(np.sum(mf) / max(vol_sum, 1e-10), -1, 1))


def _obv_trend(closes, volumes, period=24):
    """OBV slope correlation [-1,1]."""
    if len(closes) < period:
        return 0.0
    obv = np.zeros(period)
    obv[0] = volumes[-period]
    for i in range(1, period):
        if closes[-period + i] > closes[-period + i - 1]:
            obv[i] = obv[i - 1] + volumes[-period + i]
        elif closes[-period + i] < closes[-period + i - 1]:
            obv[i] = obv[i - 1] - volumes[-period + i]
        else:
            obv[i] = obv[i - 1]
    x = np.arange(period, dtype=float)
    if np.std(obv) < 1e-10:
        return 0.0
    return float(np.clip(np.corrcoef(x, obv)[0, 1], -1, 1))


def _roc(closes, period):
    if len(closes) < period + 1:
        return 0.0
    return _safe((closes[-1] - closes[-period - 1]) / closes[-period - 1], -1, 1)


def _trend_strength(closes, period=20):
    """Linear regression correlation [-1,1] (direction included)."""
    if len(closes) < period:
        return 0.0
    y = closes[-period:]
    x = np.arange(period, dtype=float)
    if np.std(y) < 1e-10:
        return 0.0
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1))


def _channel_breakout(closes, period=20):
    """Channel breakout [-1,1]."""
    if len(closes) < period + 1:
        return 0.0
    prior = closes[-(period + 1):-1]
    upper = np.max(prior)
    lower = np.min(prior)
    rng = upper - lower
    if rng < 1e-10:
        return 0.0
    cur = closes[-1]
    if cur > upper:
        return 1.0
    if cur < lower:
        return -1.0
    return float((cur - lower) / rng * 2 - 1)


def _bar_patterns(opens, closes, highs, lows):
    """Returns (inside_bar, doji, engulfing) each 0/1 or [-1,1]."""
    if len(opens) < 2:
        return 0.0, 0.0, 0.0
    inside = float(highs[-1] <= highs[-2] and lows[-1] >= lows[-2])
    rng = highs[-1] - lows[-1]
    body = abs(closes[-1] - opens[-1])
    doji = float(rng > 1e-10 and body / rng < 0.1)
    cur_body = closes[-1] - opens[-1]
    prior_body = closes[-2] - opens[-2]
    if prior_body < 0 and cur_body > abs(prior_body):
        engulf = 1.0
    elif prior_body > 0 and -cur_body > prior_body:
        engulf = -1.0
    else:
        engulf = 0.0
    return inside, doji, engulf


def _donchian_position(closes, highs, lows, period=20):
    if len(closes) < period:
        return 0.5
    dc_h = np.max(highs[-period:])
    dc_l = np.min(lows[-period:])
    rng = dc_h - dc_l
    return float(np.clip((closes[-1] - dc_l) / (rng + 1e-10), 0, 1))


def _higher_low_pattern(closes, period=5):
    if len(closes) < period * 2:
        return 0.0
    recent_h = np.max(closes[-period:])
    recent_l = np.min(closes[-period:])
    prior_h = np.max(closes[-period * 2:-period])
    prior_l = np.min(closes[-period * 2:-period])
    if recent_h > prior_h and recent_l > prior_l:
        return 1.0
    if recent_h < prior_h and recent_l < prior_l:
        return -1.0
    return 0.0


def _true_range_ratio(highs, lows, closes, period=24):
    """Current true range vs average — expansion indicator."""
    if len(closes) < period + 1:
        return 0.5
    h = highs[-period:]
    l = lows[-period:]
    c = closes[-(period + 1):-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    avg_tr = np.mean(tr[:-1]) if len(tr) > 1 else tr[0]
    if avg_tr < 1e-10:
        return 0.5
    return float(np.clip(tr[-1] / avg_tr / 3.0, 0, 1))


def _vol_regime(closes, short=12, long=48):
    if len(closes) < long:
        return 0.0
    sv = np.std(np.diff(np.log(closes[-short:])))
    lv = np.std(np.diff(np.log(closes[-long:])))
    return float(np.clip((sv / max(lv, 1e-10) - 1), -1, 1))


# ─── Group C: Cross-asset macro features ──────────────────────────────────────

def compute_macro_features(daily_closes):
    """
    Compute 40 cross-asset macro features.
    daily_closes: dict of {symbol: np.array of daily close prices}
    Returns: np.array(40,)
    """
    out = np.zeros(40, dtype=float)
    i = 0

    def _get(sym):
        return daily_closes.get(sym, np.array([]))

    g = _get("GOLD")
    s = _get("SILVER")
    o = _get("OIL")
    spx = _get("SPX")
    dxy = _get("DXY")
    btcd = _get("BTC_DAILY")
    tlt = _get("TLT")

    # GOLD momentum (5d, 20d, 60d)
    for arr, w in [(g, 5), (g, 20), (g, 60)]:
        if len(arr) > w:
            out[i] = _safe((arr[-1] - arr[-w]) / arr[-w], -1, 1)
        i += 1

    # DXY trend (20d EMA direction)
    if len(dxy) > 25:
        e = _ema(dxy[-25:], 20)
        out[i] = _safe((e[-1] - e[-5]) / (dxy[-1] + 1e-10) / 0.02, -1, 1)
    i += 1

    # OIL volatility regime
    if len(o) > 20:
        v = np.std(np.diff(np.log(o[-20:] + 1e-10)))
        out[i] = float(np.clip(v / 0.03, 0, 3) / 3)
    i += 1

    # Gold/Silver ratio (risk indicator, typical ~75)
    if len(g) > 1 and len(s) > 1 and s[-1] > 1e-10:
        out[i] = _safe((g[-1] / s[-1] - 75) / 15, -1, 1)
    i += 1

    # BTC/Gold ratio vs 20d avg
    if len(btcd) > 20 and len(g) > 20 and g[-1] > 1e-10:
        ratio = btcd[-1] / g[-1]
        avg = np.mean(btcd[-20:] / np.maximum(g[-20:], 1e-10))
        out[i] = _safe((ratio - avg) / (avg + 1e-10), -1, 1)
    i += 1

    # SPX 20d trend
    if len(spx) > 20:
        out[i] = _safe((spx[-1] - spx[-20]) / spx[-20], -1, 1)
    i += 1

    # TLT momentum (5d, 20d — risk-off signal)
    for w in [5, 20]:
        if len(tlt) > w:
            out[i] = _safe((tlt[-1] - tlt[-w]) / tlt[-w], -1, 1)
        i += 1

    # Oil/Gold ratio (inflation signal)
    if len(o) > 20 and len(g) > 20 and g[-1] > 1e-10:
        ratio = o[-1] / g[-1]
        avg = np.mean(o[-20:] / np.maximum(g[-20:], 1e-10))
        out[i] = _safe((ratio - avg) / (avg + 1e-10), -1, 1)
    i += 1

    # DXY vs Gold correlation (20d)
    period = 20
    if len(dxy) > period and len(g) > period:
        dr = np.diff(np.log(dxy[-period:] + 1e-10))
        gr = np.diff(np.log(g[-period:] + 1e-10))
        if np.std(dr) > 1e-10 and np.std(gr) > 1e-10:
            out[i] = float(np.clip(np.corrcoef(dr, gr)[0, 1], -1, 1))
    i += 1

    # BTC vs SPX correlation (20d)
    if len(btcd) > period and len(spx) > period:
        br = np.diff(np.log(btcd[-period:] + 1e-10))
        sr = np.diff(np.log(spx[-period:] + 1e-10))
        if np.std(br) > 1e-10 and np.std(sr) > 1e-10:
            out[i] = float(np.clip(np.corrcoef(br, sr)[0, 1], -1, 1))
    i += 1

    # DXY momentum (5d, 20d)
    for w in [5, 20]:
        if len(dxy) > w:
            out[i] = _safe((dxy[-1] - dxy[-w]) / dxy[-w], -1, 1)
        i += 1

    # SPX vs TLT ratio (risk-on/off regime)
    if len(spx) > 20 and len(tlt) > 20 and tlt[-1] > 1e-10:
        ratio = spx[-1] / tlt[-1]
        avg = np.mean(spx[-20:] / np.maximum(tlt[-20:], 1e-10))
        out[i] = _safe((ratio - avg) / (avg + 1e-10), -1, 1)
    i += 1

    # GOLD vs SPX correlation (20d)
    if len(g) > period and len(spx) > period:
        gr = np.diff(np.log(g[-period:] + 1e-10))
        sr = np.diff(np.log(spx[-period:] + 1e-10))
        if np.std(gr) > 1e-10 and np.std(sr) > 1e-10:
            out[i] = float(np.clip(np.corrcoef(gr, sr)[0, 1], -1, 1))
    i += 1

    # OIL momentum (5d, 20d)
    for w in [5, 20]:
        if len(o) > w:
            out[i] = _safe((o[-1] - o[-w]) / o[-w], -1, 1)
        i += 1

    # Cross-asset average volatility
    vols = []
    for arr in [g, o, spx]:
        if len(arr) > 20:
            vols.append(np.std(np.diff(np.log(arr[-20:] + 1e-10))))
    if vols:
        out[i] = float(np.clip(np.mean(vols) / 0.015, 0, 3) / 3)
    i += 1

    # BTC daily momentum (5d, 20d)
    for w in [5, 20]:
        if len(btcd) > w:
            out[i] = _safe((btcd[-1] - btcd[-w]) / btcd[-w], -1, 1)
        i += 1

    # SILVER momentum (5d)
    if len(s) > 5:
        out[i] = _safe((s[-1] - s[-5]) / s[-5], -1, 1)
    i += 1

    # TLT vs DXY correlation (flight to safety vs dollar)
    if len(tlt) > period and len(dxy) > period:
        tr2 = np.diff(np.log(tlt[-period:] + 1e-10))
        dr2 = np.diff(np.log(dxy[-period:] + 1e-10))
        if np.std(tr2) > 1e-10 and np.std(dr2) > 1e-10:
            out[i] = float(np.clip(np.corrcoef(tr2, dr2)[0, 1], -1, 1))
    i += 1

    # OIL vs SPX correlation
    if len(o) > period and len(spx) > period:
        or2 = np.diff(np.log(o[-period:] + 1e-10))
        sr2 = np.diff(np.log(spx[-period:] + 1e-10))
        if np.std(or2) > 1e-10 and np.std(sr2) > 1e-10:
            out[i] = float(np.clip(np.corrcoef(or2, sr2)[0, 1], -1, 1))
    i += 1

    # Remaining slots zeroed
    return out


# ─── Group D: Temporal features ───────────────────────────────────────────────

def compute_temporal_features(timestamp_ms):
    """Returns np.array(9,) of calendar features."""
    import datetime
    import calendar as cal
    try:
        dt = datetime.datetime.utcfromtimestamp(int(timestamp_ms) / 1000)
        hour = dt.hour
        dow = dt.weekday()
        dom = dt.day
        last_day = cal.monthrange(dt.year, dt.month)[1]
        days_left = (last_day - dom) / last_day
    except Exception:
        return np.zeros(9, dtype=float)
    return np.array([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
        float(np.clip(days_left, 0, 1)),
        float(dow >= 5),          # weekend
        float(13 <= hour <= 21),  # US market hours
        float(7 <= hour <= 15),   # EU market hours
        float(0 <= hour <= 7),    # Asia market hours
    ], dtype=float)


# ─── Group E: Statistical / regime features ───────────────────────────────────

def _rvol(closes, period):
    if len(closes) < period + 1:
        return 0.02
    return float(np.clip(np.std(np.diff(np.log(closes[-period:] + 1e-10))), 0, 0.1))


def _autocorr(closes, lag=1, period=24):
    if len(closes) < period + lag + 1:
        return 0.0
    rets = np.diff(np.log(closes[-period - lag:] + 1e-10))
    if len(rets) < lag + 2:
        return 0.0
    r1 = rets[lag:]
    r2 = rets[:-lag]
    if np.std(r1) < 1e-10 or np.std(r2) < 1e-10:
        return 0.0
    return float(np.clip(np.corrcoef(r1, r2)[0, 1], -1, 1))


def _skewness(closes, period=20):
    if len(closes) < period + 1:
        return 0.0
    rets = np.diff(np.log(closes[-period:] + 1e-10))
    std = np.std(rets)
    if std < 1e-10:
        return 0.0
    return float(np.clip(np.mean(((rets - np.mean(rets)) / std) ** 3) / 3, -1, 1))


def _hurst(closes, period=100):
    """Hurst exponent: >0.5 trending, <0.5 mean-reverting, ~0.5 random."""
    if len(closes) < period:
        return 0.5
    prices = np.log(closes[-period:] + 1e-10)
    lags_log, rs_log = [], []
    for lag in [4, 8, 16, 32, 64]:
        if lag >= period:
            continue
        rs_list = []
        n_chunks = period // lag
        for j in range(n_chunks):
            chunk = prices[j * lag:(j + 1) * lag]
            rets = np.diff(chunk)
            if len(rets) < 2:
                continue
            cumdev = np.cumsum(rets - np.mean(rets))
            std = np.std(rets)
            if std > 1e-10:
                rs_list.append((np.max(cumdev) - np.min(cumdev)) / std)
        if rs_list:
            lags_log.append(np.log(lag))
            rs_log.append(np.log(np.mean(rs_list) + 1e-10))
    if len(lags_log) < 2:
        return 0.5
    lags_log = np.array(lags_log)
    rs_log = np.array(rs_log)
    if np.std(lags_log) < 1e-10:
        return 0.5
    hurst = np.cov(lags_log, rs_log)[0, 1] / np.var(lags_log)
    return float(np.clip(hurst, 0.0, 1.0))


def _kelly_fraction(closes, period=50):
    if len(closes) < period + 1:
        return 0.0
    rets = np.diff(np.log(closes[-period:] + 1e-10))
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    wp = len(wins) / len(rets)
    aw = np.mean(wins)
    al = np.mean(np.abs(losses))
    if al < 1e-10 or aw < 1e-10:
        return 0.0
    kelly = wp / al - (1 - wp) / aw
    return float(np.clip(kelly, 0.0, 1.0))


# ─── Master feature computation ───────────────────────────────────────────────

def compute_feature_vector(bar_data, macro_features=None, timestamp_ms=None):
    """
    Compute N_FEATURES-length feature vector for a single symbol/bar.

    bar_data : BarData object (has .history DataFrame with OHLCV + funding_rate)
    macro_features : np.array(40,) from compute_macro_features(), or None
    timestamp_ms : int, epoch ms (used for temporal features)

    Returns: np.array(N_FEATURES,) — all finite floats
    """
    hist = bar_data.history
    closes = hist["close"].values.astype(float)
    highs = hist["high"].values.astype(float)
    lows = hist["low"].values.astype(float)
    opens = hist["open"].values.astype(float) if "open" in hist.columns else closes
    volumes = hist["volume"].values.astype(float)
    funding = hist["funding_rate"].values.astype(float) if "funding_rate" in hist.columns else np.zeros(len(closes))
    ts_ms = timestamp_ms or (int(hist["timestamp"].values[-1]) if "timestamp" in hist.columns else 0)

    f = []

    # ── Group A: Proven signals (18 features) ──────────────────────────────────
    f.append(_rsi(closes, 8))                         # 1
    f.append(_rsi(closes, 14))                        # 2
    f.append(_ema_cross(closes, 7, 26))               # 3
    f.append(_ema_cross(closes, 12, 48))              # 4
    f.append(_ema_cross(closes, 24, 72))              # 5
    f.append(_macd_hist(closes, 14, 23, 9))           # 6
    f.append(_macd_hist(closes, 5, 13, 5))            # 7  fast MACD
    f.append(_bb_position(closes, 20))                # 8
    f.append(_bb_width(closes, 20))                   # 9
    f.append(_bb_position(closes, 7))                 # 10  short BB
    f.append(_atr_norm(highs, lows, closes, 24))      # 11
    f.append(_momentum(closes, 6))                    # 12
    f.append(_momentum(closes, 12))                   # 13
    f.append(_momentum(closes, 24))                   # 14
    f.append(_momentum(closes, 36))                   # 15
    f.append(_momentum(closes, 48))                   # 16
    f.append(_volume_ratio(volumes, 24))              # 17
    f.append(_volume_ratio(volumes, 5))               # 18

    # ── Group A: Price vs MA (5 features) ─────────────────────────────────────
    f.append(_sma_dist(closes, 20))                   # 19
    f.append(_sma_dist(closes, 50))                   # 20
    f.append(_sma_dist(closes, 100) if len(closes) >= 100 else 0.0)  # 21
    f.append(_sma_dist(closes, 200) if len(closes) >= 200 else 0.0)  # 22

    # Funding features (3)
    fa8, fa24, fstd = _funding_features(funding, 24)
    f.append(fa8)                                     # 23
    f.append(fa24)                                    # 24
    f.append(fstd)                                    # 25

    # ── Group B: Technical indicators (~45 features) ───────────────────────────
    f.append(_cci(highs, lows, closes, 20))           # 26
    f.append(_cci(highs, lows, closes, 10))           # 27
    f.append(_keltner_position(highs, lows, closes))  # 28
    f.append(_bb_squeeze(highs, lows, closes))        # 29

    adx_v, adx_d = _adx(highs, lows, closes, 14)
    f.append(adx_v)                                   # 30
    f.append(adx_d)                                   # 31

    sk, sd = _stochastic(highs, lows, closes, 14, 3)
    f.append(sk)                                      # 32
    f.append(sd)                                      # 33
    f.append(sk - sd)                                 # 34  %K-%D signal

    f.append(_williams_r(highs, lows, closes, 14))   # 35
    f.append(_williams_r(highs, lows, closes, 20))   # 36
    f.append(_cmf(highs, lows, closes, volumes, 20)) # 37
    f.append(_obv_trend(closes, volumes, 24))         # 38

    f.append(_roc(closes, 1))                        # 39
    f.append(_roc(closes, 3))                        # 40
    f.append(_roc(closes, 5))                        # 41
    f.append(_roc(closes, 10))                       # 42
    f.append(_roc(closes, 20))                       # 43

    f.append(_higher_low_pattern(closes, 5))         # 44
    f.append(_higher_low_pattern(closes, 10))        # 45
    f.append(_channel_breakout(closes, 20))          # 46
    f.append(_channel_breakout(closes, 50) if len(closes) >= 51 else 0.0)  # 47

    inside, doji, engulf = _bar_patterns(opens, closes, highs, lows)
    f.append(inside)                                 # 48
    f.append(doji)                                   # 49
    f.append(engulf)                                 # 50

    f.append(_trend_strength(closes, 20))            # 51
    f.append(_trend_strength(closes, 50) if len(closes) >= 50 else 0.0)  # 52
    f.append(_donchian_position(closes, highs, lows, 20))  # 53
    f.append(_true_range_ratio(highs, lows, closes, 24))   # 54
    f.append(_vol_regime(closes, 12, 48))            # 55

    # ROC acceleration (roc3 change)
    roc3_now = _roc(closes, 3)
    roc3_prior = _roc(closes[:-3], 3) if len(closes) > 6 else 0.0
    f.append(float(np.clip(roc3_now - roc3_prior, -1, 1)))  # 56

    # Bar body ratio
    rng = highs[-1] - lows[-1]
    body = abs(closes[-1] - opens[-1])
    f.append(float(body / rng if rng > 1e-10 else 0.5))  # 57

    # Price gap
    if len(opens) >= 2 and closes[-2] > 1e-10:
        f.append(_safe((opens[-1] - closes[-2]) / closes[-2] / 0.03, -1, 1))  # 58
    else:
        f.append(0.0)

    # RSI(4) — very short term
    f.append(_rsi(closes, 4) if len(closes) >= 5 else 0.0)  # 59

    # EMA ultra-short
    f.append(_ema_cross(closes, 3, 10))              # 60

    # ── Group D: Temporal (9 features) ────────────────────────────────────────
    temporal = compute_temporal_features(ts_ms)
    f.extend(temporal.tolist())                      # 61-69

    # ── Group E: Statistical (9 features) ─────────────────────────────────────
    f.append(_rvol(closes, 6))                       # 70
    f.append(_rvol(closes, 24))                      # 71
    f.append(_rvol(closes, 48) if len(closes) >= 49 else 0.02)  # 72
    f.append(_autocorr(closes, 1, 24))              # 73
    f.append(_autocorr(closes, 2, 24))              # 74
    f.append(_skewness(closes, 20))                 # 75
    f.append(_hurst(closes, 100) if len(closes) >= 100 else 0.5)  # 76
    f.append(_kelly_fraction(closes, 50) if len(closes) >= 51 else 0.0)  # 77

    # Volume-price trend correlation
    if len(closes) >= 24 and len(volumes) >= 24:
        rets = np.diff(np.log(closes[-24:] + 1e-10))
        vols_slice = volumes[-23:]
        if np.std(rets) > 1e-10 and np.std(vols_slice) > 1e-10:
            f.append(float(np.clip(np.corrcoef(rets, vols_slice)[0, 1], -1, 1)))  # 78
        else:
            f.append(0.0)
    else:
        f.append(0.0)

    # ── Group C: Macro (40 features) ──────────────────────────────────────────
    if macro_features is not None and len(macro_features) >= 40:
        f.extend(macro_features[:40].tolist())       # 79-118
    else:
        f.extend([0.0] * 40)

    # Padding to exactly N_FEATURES = 128
    while len(f) < N_FEATURES:
        f.append(0.0)
    f = f[:N_FEATURES]

    arr = np.array(f, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def compute_feature_matrix(history_df, macro_closes=None, seq_len=500):
    """
    Build (T, N_FEATURES) feature matrix for training.

    history_df : DataFrame [timestamp, open, high, low, close, volume, funding_rate]
    macro_closes : dict {symbol: np.array of daily closes}
    seq_len : max look-back per bar

    Returns: (features: np.array (T, N_FEATURES), timestamps: np.array (T,))
    """
    n = len(history_df)
    if n < 2:
        return np.zeros((0, N_FEATURES), dtype=float), np.array([])

    macro_feats = compute_macro_features(macro_closes) if macro_closes else np.zeros(40, dtype=float)

    class _FakeBar:
        def __init__(self, hist):
            self.history = hist

    features_list = []
    timestamps = []

    for i in range(1, n):
        start = max(0, i - seq_len + 1)
        hist_slice = history_df.iloc[start:i + 1].reset_index(drop=True)
        ts_ms = int(history_df["timestamp"].iloc[i]) if "timestamp" in history_df.columns else 0
        fv = compute_feature_vector(_FakeBar(hist_slice), macro_feats, ts_ms)
        features_list.append(fv)
        timestamps.append(ts_ms)

    return np.array(features_list, dtype=float), np.array(timestamps)
