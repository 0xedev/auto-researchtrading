import os
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from prepare import calculate_features, FEATURE_COLS

@dataclass
class Signal:
    symbol: str
    target_position: float
    order_type: str = "market"

class Strategy:
    def __init__(self, timeframe: str):
        self.timeframe_arg = timeframe
        self.interval_sec = self._parse_timeframe(timeframe)
        
        self.models = {}
        self.meta_models = {}
        self.symbol_caches = {} 
        self.bar_counts = {}   
        self.models_loaded = False
        
        # State tracking
        self.trailing_stops = {}

    def _parse_timeframe(self, tf: str) -> int:
        if tf == "15m": return 15 * 60
        if tf == "1h": return 3600
        if tf == "4h": return 14400
        return 0

    def _load_models(self):
        for tf in ["15m", "1h", "4h"]:
            m_path = f"models/lead_{tf}.xgb"
            meta_path = f"models/meta_{tf}.xgb"
            
            if os.path.exists(m_path):
                self.models[tf] = xgb.XGBClassifier()
                self.models[tf].load_model(m_path)
            
            if os.path.exists(meta_path):
                self.meta_models[tf] = xgb.XGBClassifier()
                self.meta_models[tf].load_model(meta_path)
                
        self.models_loaded = True
        print(f"LOADED QUANTUM FORTRESS: {list(self.models.keys())}")

    def pre_calculate_signals(self, data_dict):
        """Vectorized pre-calculation for Quantum Meta-Labeling Fortress."""
        if not self.models_loaded:
            self._load_models()

        # 1. Pre-calculate Systemic Beta (Market Context)
        print("Calculating Global Market Context...")
        all_vols = {}
        all_rets = {}
        for symbol, df in data_dict.items():
            feat = calculate_features(df, timeframe=self.timeframe_arg)
            all_vols[symbol] = feat['bb_width']
            all_rets[symbol] = df['close'].pct_change()
            
        m_vol = pd.DataFrame(all_vols).median(axis=1).fillna(0)
        m_ret = pd.DataFrame(all_rets).median(axis=1).fillna(0)

        print(f"Building Quantum Meta-Labeling Fortress for all symbols...")
        for symbol, df in data_dict.items():
            df_feat = calculate_features(df, timeframe=self.timeframe_arg)
            df_feat['market_vol'] = m_vol
            df_feat['market_ret'] = m_ret
            X = df_feat[FEATURE_COLS]
            
            # Lead Inference (Direction)
            probs_15m = self.models["15m"].predict_proba(X) if "15m" in self.models else np.zeros((len(X), 3))
            
            # Meta Inference (Profitability Confidence)
            meta_15m = self.meta_models["15m"].predict_proba(X)[:, 1] if "15m" in self.meta_models else np.zeros(len(X))
            meta_1h = self.meta_models["1h"].predict_proba(X)[:, 1] if "1h" in self.meta_models else np.zeros(len(X))
            meta_4h = self.meta_models["4h"].predict_proba(X)[:, 1] if "4h" in self.meta_models else np.zeros(len(X))
            
            cache_list = []
            for i in range(len(df)):
                cache_list.append({
                    'bull_15m': probs_15m[i, 1] if probs_15m.shape[1] > 1 else 0,
                    'bear_15m': probs_15m[i, 2] if probs_15m.shape[1] > 2 else 0,
                    'meta_15m': meta_15m[i],
                    'meta_1h': meta_1h[i],
                    'meta_4h': meta_4h[i],
                    'atr_val': df_feat['atr_14'].iloc[i],
                    'market_ret': df_feat['market_ret'].iloc[i]
                })
            self.symbol_caches[symbol] = cache_list
            self.bar_counts[symbol] = 0
            
        print(f"Market-Aware Fortress cache warmed for {len(self.symbol_caches)} symbols.")

    def on_bar(self, bar_data, portfolio):
        if not self.models_loaded:
            self._load_models()

        equity = portfolio.equity
        signals = []

        for symbol, bar in bar_data.items():
            pos = portfolio.positions.get(symbol, 0.0)
            
            if symbol not in self.symbol_caches:
                continue
            
            cache = self.symbol_caches[symbol]
            idx = self.bar_counts.get(symbol, 0)
            if idx >= len(cache): continue
            
            row = cache[idx]
            self.bar_counts[symbol] += 1

            m15 = row['meta_15m']
            m1h = row['meta_1h']
            m4h = row['meta_4h']
            
            # Nexus Confluence: Dynamic Thresholding for Alpha Surge
            # We use the top-tier of the meta-probability distribution
            is_fortress = (m15 > 0.50) and (m1h > 0.40) and (m4h > 0.30)
            
            if pos != 0:
                # Dynamic Volatility-Adjusted Trailing Stop
                if pos > 0:
                    if bar.close < self.trailing_stops.get(symbol, 0):
                        signals.append(Signal(symbol, 0.0))
                    self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), bar.high - stop_dist)
                else:
                    if bar.close > self.trailing_stops.get(symbol, 9e18):
                        signals.append(Signal(symbol, 0.0))
                    self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, 9e18), bar.low + stop_dist)
                continue

            if is_fortress:
                # Institutional Sizing: 4% per symbol to hit high diversified Sharpe
                size = equity * 0.04 
                
                if row['bull_15m'] > 0.35:
                    signals.append(Signal(symbol, size))
                    self.trailing_stops[symbol] = bar.low - 3.0 * row['atr_val']
                elif row['bear_15m'] > 0.35:
                    signals.append(Signal(symbol, -size))
                    self.trailing_stops[symbol] = bar.high + 3.0 * row['atr_val']

        return signals
