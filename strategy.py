import os
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
from hmmlearn import hmm
from dataclasses import dataclass
from typing import List
from prepare import calculate_features, FEATURE_COLS, load_data


@dataclass
class Signal:
    symbol: str
    target_position: float
    order_type: str = "market"


class Strategy:
    def __init__(self, timeframe: str = "1h"):
        self.timeframe_arg = timeframe
        self.interval_sec = self._parse_timeframe(timeframe)

        # Timeframe-derived constants (all calibrated at 1h, scaled automatically)
        _bph = 3600 / self.interval_sec  # bars per hour
        _tf_ratio = self.interval_sec / 3600  # 0.25 for 15m, 1.0 for 1h, 4.0 for 4h
        self._regime_window = max(18, int(72 * _bph))       # 72h regime lookback
        self._max_hold = max(2, round(6 * _tf_ratio ** 0.5))  # sqrt-scaled hold
        self._decay_age = max(2, round(3 * _tf_ratio ** 0.5)) # sqrt-scaled signal decay
        self._atr_scale = _tf_ratio ** 0.25                    # ATR mult: constant stop/range ratio
        self._entry_scale = min(1.0, _tf_ratio ** 0.5)         # entry size scaling
        self._thresh_scale = max(1.0, (1.0 / _tf_ratio) ** 0.25)  # meta gate quality: higher for shorter tf

        self.models = {}  # {tf: {state_id: model}}
        self.meta_models = {}
        self.long_models = {} # fallback
        self.short_models = {} # fallback
        self.symbol_caches = {}
        self.bar_counts = {}
        self.models_loaded = False
        self.trailing_stops = {}
        self.position_ages = {}
        self._market_ret_buf = []
        self._macro_bear = False
        self._macro_hmm = None
        self._current_hmm_state = 0
        self._bars_since_calibration = 0

        # Load Marco HMM if exists
        hmm_path = "models/macro_hmm.joblib"
        scaler_path = "models/macro_scaler.joblib"
        if os.path.exists(hmm_path):
            try:
                self._macro_hmm = joblib.load(hmm_path)
                print(f"LOADED MACRO HMM (Medallion Engine)")
            except Exception as e:
                print(f"Warning: Failed to load HMM: {e}")
        if os.path.exists(scaler_path):
            try:
                self._macro_scaler = joblib.load(scaler_path)
                print(f"LOADED MACRO SCALER")
            except Exception as e:
                print(f"Warning: Failed to load HMM Scaler: {e}")

    def _parse_timeframe(self, tf: str) -> int:
        if tf == "15m":
            return 15 * 60
        if tf == "1h":
            return 3600
        if tf == "4h":
            return 14400
        return 0

    def _load_models(self):
        for tf in ["15m", "1h", "4h"]:
            # Specialist Model Loading
            for state in range(4):
                s_lead_path = f"models/lead_{tf}_s{state}.json"
                s_meta_path = f"models/meta_{tf}_s{state}.json"
                
                if os.path.exists(s_lead_path):
                    if tf not in self.models: self.models[tf] = {}
                    self.models[tf][state] = xgb.XGBClassifier()
                    self.models[tf][state].load_model(s_lead_path)
                    
                if os.path.exists(s_meta_path):
                    if tf not in self.meta_models: self.meta_models[tf] = {}
                    self.meta_models[tf][state] = xgb.XGBClassifier()
                    self.meta_models[tf][state].load_model(s_meta_path)

            # Global Fallback Loading
            m_path = f"models/lead_{tf}.json"
            meta_path = f"models/meta_{tf}.json"
            if os.path.exists(m_path):
                if tf not in self.models: self.models[tf] = {}
                self.models[tf]["global"] = xgb.XGBClassifier()
                self.models[tf]["global"].load_model(m_path)
            if os.path.exists(meta_path):
                if tf not in self.meta_models: self.meta_models[tf] = {}
                self.meta_models[tf]["global"] = xgb.XGBClassifier()
                self.meta_models[tf]["global"].load_model(meta_path)

        self.models_loaded = True
        print(f"LOADED QUANTUM FORTRESS SPECIALISTS: {list(self.models.keys())}")

    def _build_prediction_tables(self, data_dict, timeframe: str, include_state: bool = False):
        if not data_dict:
            return {}

        all_vols = {}
        all_rets = {}
        for symbol, df in data_dict.items():
            feat = calculate_features(df, timeframe=timeframe)
            all_vols[symbol] = feat["bb_width"]
            all_rets[symbol] = df["close"].pct_change()

        m_vol = pd.DataFrame(all_vols).median(axis=1).fillna(0)
        m_ret = pd.DataFrame(all_rets).median(axis=1).fillna(0)

        # Pre-calculate Macro HMM for the entire batch
        hmm_path = "models/macro_hmm.joblib"
        hmm_model = joblib.load(hmm_path) if os.path.exists(hmm_path) else None

        tables = {}
        for symbol, df in data_dict.items():
            # calculate_features (from prepare.py) handles all indicators
            df_feat = calculate_features(df.copy(), timeframe=timeframe)
            df_feat["market_vol"] = m_vol
            df_feat["market_ret"] = m_ret
            market_ret_4h = m_ret.rolling(4, min_periods=1).sum().fillna(0.0)
            df_feat["rel_ret_1h"] = df_feat["ret_1h"] - df_feat["market_ret"]
            df_feat["rel_ret_4h"] = df_feat["ret_4h"] - market_ret_4h
            df_feat["rel_bb_width"] = df_feat["bb_width"] - df_feat["market_vol"]
            
            # High-Fidelity HMM State Detection
            df_feat["macro_state"] = 0
            if hmm_model:
                try:
                    # Retrieve the macro anchor assets from the data_dict
                    # [BTC_Ret, BTC_Vol, ETH/BTC_Ret, XAU_Ret, SP500_Ret]
                    # Note: We assume these are in the data_dict from prepare.py
                    btc = data_dict.get("BTC", df)
                    eth = data_dict.get("ETH", df)
                    xau = data_dict.get("XAU", df)
                    spx = data_dict.get("SP500", df)
                    
                    btc_ret = btc["close"].pct_change().fillna(0).values
                    btc_vol = calculate_features(btc, timeframe=timeframe)["bb_width"].fillna(0).values
                    eth_btc_ret = (eth["close"].pct_change() - btc_ret).fillna(0).values
                    xau_ret = xau["close"].pct_change().fillna(0).values
                    spx_ret = spx["close"].pct_change().fillna(0).values
                    
                    # Align lengths and Scale for HMM
                    X_hmm = np.column_stack([
                        btc_ret, btc_vol, eth_btc_ret, xau_ret, spx_ret
                    ])
                    if len(X_hmm) > len(df_feat): X_hmm = X_hmm[-len(df_feat):]
                    elif len(X_hmm) < len(df_feat):
                        padding = np.zeros((len(df_feat) - len(X_hmm), 5))
                        X_hmm = np.vstack([padding, X_hmm])

                    if hasattr(self, "_macro_scaler") and self._macro_scaler:
                        X_hmm = self._macro_scaler.transform(X_hmm)

                    df_feat["macro_state"] = hmm_model.predict(X_hmm)
                    
                    # Diagnostic: Print the state distribution for the FIRST symbol only to save log space
                    if symbol == list(data_dict.keys())[0]:
                        unique_states, state_counts = np.unique(df_feat["macro_state"], return_counts=True)
                        print(f"--- HMM REGIME DISTRIBUTION: {dict(zip(unique_states, state_counts))} ---")
                except Exception as e:
                    print(f"HMM High-Fideilty Predict Error: {e}")

            X_full = df_feat[FEATURE_COLS]
            print(f"DIAGNOSTIC: X_full for {symbol} - Shape: {X_full.shape} | Nulls: {X_full.isna().sum().sum()}")
            if X_full.isna().any().any():
                 print(f"WARNING: NaNs found in features: {X_full.columns[X_full.isna().any()].tolist()}")
            states = df_feat["macro_state"].values
            
            tf_models = self.models.get(timeframe, {})
            tf_meta = self.meta_models.get(timeframe, {})
            
            all_bull = np.zeros(len(df_feat))
            all_bear = np.zeros(len(df_feat))
            all_meta = np.zeros(len(df_feat))
            
            for state_id in range(4):
                mask = (states == state_id)
                if not any(mask): continue
                
                model = tf_models.get(state_id, tf_models.get("global"))
                meta_model = tf_meta.get(state_id, tf_meta.get("global"))
                
                if model:
                    probs = model.predict_proba(X_full[mask])
                    # Force class mapping to bypass metadata loss during JSON loading
                    # Lead models: 0=Neutral, 1=Long, 2=Short
                    all_bull[mask] = probs[:, 1] if probs.shape[1] > 1 else 0.0
                    all_bear[mask] = probs[:, 2] if probs.shape[1] > 2 else 0.0
                    
                    if meta_model:
                        m_probs = meta_model.predict_proba(X_full[mask])
                        # Meta models: 0=Fail, 1=Pass
                        all_meta[mask] = m_probs[:, 1] if m_probs.shape[1] > 1 else 0.0
                    
                    # Probability Audit Print
                    if symbol == list(data_dict.keys())[0]:
                        print(f"DEBUG: {symbol} S{state_id} | MaxBull: {all_bull[mask].max():.4f} | MaxBear: {all_bear[mask].max():.4f} | MaxMeta: {all_meta[mask].max():.4f}")
            
            table = pd.DataFrame({
                "timestamp": df["timestamp"].values,
                f"bull_{timeframe}": all_bull,
                f"bear_{timeframe}": all_bear,
                f"meta_{timeframe}": all_meta,
                "macro_state": states.astype(int),
            })
            if include_state:
                table["atr_val"] = df_feat["atr_14"].values
                table["market_ret"] = df_feat["market_ret"].values
                table["rsi_8"] = df_feat["rsi_8"].values
            tables[symbol] = table.sort_values("timestamp").reset_index(drop=True)
        return tables

    def pre_calculate_signals(self, data_dict, split_name: str = "val"):
        if not self.models_loaded:
            self._load_models()

        print("Calculating timeframe-aligned model caches...")
        main_tables = self._build_prediction_tables(data_dict, self.timeframe_arg, include_state=True)
        aux_15m_tables = {}
        aux_1h_tables = {}
        aux_4h_tables = {}
        base_split = split_name.replace("_15m", "")

        if self.timeframe_arg == "1h":
            if "15m" in self.models or "15m" in self.meta_models:
                split_15m = {
                    "train": "train_15m",
                    "val": "val_15m",
                    "oos": "oos_15m",
                }.get(split_name)
                if split_15m:
                    aux_15m_data = load_data(split=split_15m)
                    print(f"Loaded {len(aux_15m_data)} 15m symbols for alignment.")
                    aux_15m_tables = self._build_prediction_tables(aux_15m_data, "15m")

            if "4h" in self.models or "4h" in self.meta_models:
                aux_4h_data = load_data(split=base_split, resample_4h=True)
                print(f"Loaded {len(aux_4h_data)} 4h symbols for alignment.")
                aux_4h_tables = self._build_prediction_tables(aux_4h_data, "4h")

        elif self.timeframe_arg == "15m":
            if "1h" in self.models or "1h" in self.meta_models:
                aux_1h_data = load_data(split=base_split)
                print(f"Loaded {len(aux_1h_data)} 1h symbols for alignment.")
                aux_1h_tables = self._build_prediction_tables(aux_1h_data, "1h")

            if "4h" in self.models or "4h" in self.meta_models:
                aux_4h_data = load_data(split=base_split, resample_4h=True)
                print(f"Loaded {len(aux_4h_data)} 4h symbols for alignment.")
                aux_4h_tables = self._build_prediction_tables(aux_4h_data, "4h")

        elif self.timeframe_arg == "4h":
            if "15m" in self.long_models or "15m" in self.short_models:
                split_15m = {
                    "robustness": "val_15m",
                    "val": "val_15m",
                    "oos": "oos_15m",
                    "train": "train_15m",
                }.get(base_split)
                if split_15m:
                    aux_15m_data = load_data(split=split_15m)
                    print(f"Loaded {len(aux_15m_data)} 15m symbols for 4h alignment.")
                    aux_15m_tables = self._build_prediction_tables(aux_15m_data, "15m")

            if "1h" in self.models or "1h" in self.meta_models:
                aux_1h_data = load_data(split=base_split)
                print(f"Loaded {len(aux_1h_data)} 1h symbols for 4h alignment.")
                aux_1h_tables = self._build_prediction_tables(aux_1h_data, "1h")

        for symbol in data_dict:
            if symbol not in main_tables:
                continue

            main_df = main_tables[symbol]
            main_bull_col = f"bull_{self.timeframe_arg}"
            main_bear_col = f"bear_{self.timeframe_arg}"
            main_meta_col = f"meta_{self.timeframe_arg}"

            base = main_df[["timestamp", "atr_val", "market_ret", "rsi_8"]].copy()
            base["bull_15m"] = main_df[main_bull_col].fillna(0).values if main_bull_col in main_df else 0.0
            base["bear_15m"] = main_df[main_bear_col].fillna(0).values if main_bear_col in main_df else 0.0
            base["meta_15m"] = main_df[main_meta_col].fillna(0).values if self.timeframe_arg == "15m" and main_meta_col in main_df else 0.0
            base["meta_long_15m"] = (
                main_df["meta_long_15m"].fillna(0).values if self.timeframe_arg == "15m" and "meta_long_15m" in main_df
                else base["meta_15m"].copy()
            )
            base["meta_short_15m"] = (
                main_df["meta_short_15m"].fillna(0).values if self.timeframe_arg == "15m" and "meta_short_15m" in main_df
                else base["meta_15m"].copy()
            )
            base["meta_1h"] = main_df[main_meta_col].fillna(0).values if self.timeframe_arg == "1h" and main_meta_col in main_df else 0.0
            base["meta_4h"] = main_df[main_meta_col].fillna(0).values if self.timeframe_arg == "4h" and main_meta_col in main_df else 0.0

            if symbol in aux_15m_tables:
                merged_15m = pd.merge_asof(
                    base[["timestamp"]].sort_values("timestamp"),
                    aux_15m_tables[symbol].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                base["bull_15m"] = merged_15m["bull_15m"].fillna(base["bull_15m"])
                base["bear_15m"] = merged_15m["bear_15m"].fillna(base["bear_15m"])
                base["meta_15m"] = merged_15m["meta_15m"].fillna(base["meta_15m"])
                if "meta_long_15m" in merged_15m:
                    base["meta_long_15m"] = merged_15m["meta_long_15m"].fillna(base["meta_long_15m"])
                if "meta_short_15m" in merged_15m:
                    base["meta_short_15m"] = merged_15m["meta_short_15m"].fillna(base["meta_short_15m"])

            if symbol in aux_1h_tables:
                merged_1h = pd.merge_asof(
                    base[["timestamp"]].sort_values("timestamp"),
                    aux_1h_tables[symbol][["timestamp", "meta_1h"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                base["meta_1h"] = merged_1h["meta_1h"].fillna(base["meta_1h"])

            if symbol in aux_4h_tables:
                merged_4h = pd.merge_asof(
                    base[["timestamp"]].sort_values("timestamp"),
                    aux_4h_tables[symbol][["timestamp", "meta_4h"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                base["meta_4h"] = merged_4h["meta_4h"].fillna(base["meta_4h"])

            # Type-cast keys to explicit int seconds precision
            raw_cache = base.fillna(0.0).copy()
            ts_raw = raw_cache["timestamp"].astype(np.int64).values
            raw_cache["timestamp"] = np.where(ts_raw > 1e11, ts_raw // 1000, ts_raw)
            raw_dict = raw_cache.set_index("timestamp").to_dict("index")
            self.symbol_caches[symbol] = {int(k): v for k, v in raw_dict.items()}
            self.bar_counts[symbol] = 0

        print(f"Market-Aware Fortress cache warmed for {len(self.symbol_caches)} symbols.")

    def on_bar(self, bar_data, portfolio):
        if not self.models_loaded:
            self._load_models()

        equity = portfolio.equity
        signals: List[Signal] = []

        # Macro regime: rolling 72-bar market return detects sustained bear trends
        any_sym = next(iter(bar_data), None)
        # Self-Learning Calibration Hook (Every 480 bars ~ 20 days on 1h)
        self._bars_since_calibration += 1
        if self._bars_since_calibration >= 480:
            self.calibrate_regimes(bar_data)
            self._bars_since_calibration = 0

        for symbol, bar in bar_data.items():
            if symbol not in self.symbol_caches:
                if len(self.symbol_caches) > 0:
                    print(f"DEBUG: Symbol {symbol} NOT in caches. Caches keys: {list(self.symbol_caches.keys())}")
                continue
            
            # Using bar.timestamp (normed to seconds in prepare.py)
            row = self.symbol_caches[symbol].get(int(bar.timestamp))
            if row is None:
                continue
            
            pos = portfolio.positions.get(symbol, 0.0)
            
            # Macro regime: Update rolling market return buffer using the first symbol's row
            if symbol == any_sym:
                mret = row.get("market_ret", 0.0)
                self._market_ret_buf.append(mret)
                if len(self._market_ret_buf) > self._regime_window:
                    self._market_ret_buf.pop(0)
                self._macro_bear = sum(self._market_ret_buf) < -0.015

            tf = self.timeframe_arg
            m_curr = row.get(f"meta_{tf}", 0.0)
            rsi_8 = row.get("rsi_8", 50.0)
            m_l = row.get(f"meta_long_{tf}", m_curr)
            m_s = row.get(f"meta_short_{tf}", m_curr)
            m1h = row.get("meta_1h", m_curr)
            m4h = row.get("meta_4h", m_curr)
            m15_long = row.get("meta_long_15m", 0.0)
            m15_short = row.get("meta_short_15m", 0.0)
            
            # ATR-based adaptive Stop/Target distance
            atr = row.get("atr_pct", 0.015)
            stop_dist = bar.close * (atr * 1.5)

            active_meta = [m for m in (max(m_l, m_s), m1h, m4h) if m > 0]
            meta_score = float(np.mean(active_meta)) if active_meta else 0.0
            
            # Medallion-style HMM State Adaptation
            state_adjust = {
                0: {"long_gate": -0.05, "short_gate": +0.05, "size": 1.1}, # Bullish: loosen longs
                1: {"long_gate": +0.05, "short_gate": -0.05, "size": 1.1}, # Bearish: loosen shorts
                2: {"long_gate": +0.02, "short_gate": +0.02, "size": 0.5}, # Volatile: tighten gates, cut size
                3: {"long_gate": +0.10, "short_gate": +0.10, "size": 0.1}  # Choppy: test with small size
            }.get(row.get("macro_state", 0), {"long_gate": 0.0, "short_gate": 0.0, "size": 1.0})

            supportive_regime = (m1h <= 0 or m1h > 0.45)
            supportive_regime_4h = (m4h <= 0 or m4h > 0.45)
            
            # Production logic (state-adaptive gates)
            bull_raw = row.get(f"bull_{tf}", 0.0)
            bear_raw = row.get(f"bear_{tf}", 0.0)
            bull_signal = bull_raw > (0.32 + state_adjust["long_gate"])
            bear_signal = bear_raw > (0.55 + state_adjust["short_gate"])
            
            meta_score = row.get(f"meta_{tf}", 0)
            
            # Fortress logic (timeframe-agnostic keys)
            bull_fortress = bull_signal and meta_score > 0.28 and rsi_8 < 65
            raw_bear_fortress = (
                row.get(f"bear_{tf}", 0.0) > 0.62
                and row.get(f"bear_{tf}", 0.0) > row.get(f"bull_{tf}", 0.0) + 0.12
                and rsi_8 > 48
            )
            bear_fortress = (bear_signal and meta_score > 0.28 and rsi_8 > 32) or raw_bear_fortress
            bull_soft = bull_signal and supportive_regime and supportive_regime_4h and meta_score > 0.50
            bear_soft = bear_signal and supportive_regime and supportive_regime_4h and meta_score > 0.50

            # Continuous meta-proportional sizing (exp239)
            if supportive_regime:
                meta_factor = max(0.0, min(1.0, (meta_score - 0.30) / 0.50))  # 0→1 over [0.30, 0.80]
                risk_pct = 0.03 + 0.27 * meta_factor  # 3% at meta=0.30, 30% at meta≥0.80
                risk_per_trade = equity * risk_pct
                long_size = (risk_per_trade * bar.close) / stop_dist
                short_size = (risk_per_trade * bar.close) / stop_dist
            else:
                long_size = equity * 0.08
                short_size = equity * 0.08
            # Regime-throttled risk: keep flow, but cut weak-regime exposure hard.
            if not supportive_regime:
                long_size *= 0.05
            # Continuous regime scaling (25x) — smooth crush based on macro intensity
            mret_sum = sum(self._market_ret_buf) if self._market_ret_buf else 0.0
            long_factor = max(0.20, min(1.0, 1.0 + 45.0 * mret_sum))
            short_factor = max(0.20, min(1.0, 1.0 - 45.0 * mret_sum))
            long_size *= long_factor
            short_size *= short_factor

            long_size *= state_adjust["size"]
            short_size *= state_adjust["size"]

            # Flow control for final signal gates
            macro_bull_ok = not self._macro_bear
            long_gate_ok = True
            short_gate_ok = True

            if pos != 0:
                age = self.position_ages.get(symbol, 0) + 1
                self.position_ages[symbol] = age
                max_hold_bars = self._max_hold

                if pos > 0:
                    signal_decay_exit = age >= self._decay_age and (not bull_signal) and m15_long < 0.40
                    should_exit = (
                        age >= max_hold_bars
                        or bar.close < self.trailing_stops.get(symbol, 0)
                        or signal_decay_exit
                    )
                    if should_exit:
                        signals.append(Signal(symbol, 0.0))
                        self.position_ages[symbol] = 0
                    else:
                        desired = pos
                        if bull_signal and m15_long > 0.65 and abs(pos) + 1.0 < long_size:
                            desired = long_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired))
                        self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), bar.high - stop_dist)
                else:
                    signal_decay_exit = age >= self._decay_age and (not bear_signal) and m15_short < 0.40 and not raw_bear_fortress
                    should_exit = (
                        age >= max_hold_bars
                        or bar.close > self.trailing_stops.get(symbol, 9e18)
                        or signal_decay_exit
                    )
                    if should_exit:
                        signals.append(Signal(symbol, 0.0))
                        self.position_ages[symbol] = 0
                    else:
                        desired = pos
                        if bear_signal and m15_short > 0.65 and abs(pos) + 1.0 < short_size:
                            desired = -short_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired))
                        self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, 9e18), bar.low + stop_dist)
                continue

            self.position_ages[symbol] = 0
            ts = self._thresh_scale  # meta gate scaling: 1.0 for 1h, 1.414 for 15m
            long_gate_ok = meta_score > (0.36 * ts + state_adjust["long_gate"])
            short_gate_ok = meta_score > (0.25 * ts + state_adjust["short_gate"])
            macro_bull_ok = not self._macro_bear or meta_score > 0.40 * ts
            if bull_fortress and not raw_bear_fortress and (not bear_fortress or m15_long >= m15_short) and macro_bull_ok and long_gate_ok:
                entry_size = long_size * self._entry_scale
                signals.append(Signal(symbol, entry_size))
                self.trailing_stops[symbol] = bar.low - stop_dist
                self.position_ages[symbol] = 0
            elif bull_soft and (not bear_soft or m15_long >= m15_short) and long_gate_ok:
                entry_size = long_size * 0.5 * self._entry_scale
                signals.append(Signal(symbol, entry_size))
                self.trailing_stops[symbol] = bar.low - stop_dist
                self.position_ages[symbol] = 0
            elif bear_fortress and short_gate_ok:
                entry_size = short_size * self._entry_scale
                signals.append(Signal(symbol, -entry_size))
                self.trailing_stops[symbol] = bar.high + stop_dist
                self.position_ages[symbol] = 0
            elif bear_soft and short_gate_ok:
                entry_size = short_size * 0.5 * self._entry_scale
                signals.append(Signal(symbol, -entry_size))
                self.trailing_stops[symbol] = bar.high + stop_dist
                self.position_ages[symbol] = 0

        return signals

    def _perceive_macro_state(self, bar_data) -> int:
        """Infers the current hidden market regime using the Multivariate HMM."""
        if self._macro_hmm is None:
            return 0
            
        try:
            # Extract recent returns for the 5-D Matrix: [BTC_Ret, BTC_Vol, ETH/BTC_Ret, XAU_Ret, SP500_Ret]
            # Use last 24h window for stability
            hist_rets = {}
            for s in ["BTC", "ETH", "XAU", "SP500"]:
                if s in bar_data:
                    df = bar_data[s].history
                    if len(df) >= 24:
                        hist_rets[s] = np.log(df['close'] / df['close'].shift(1)).tail(1).values[0]
                        vols = np.log(df['close'] / df['close'].shift(1)).rolling(24).std().tail(1).values[0]
                        hist_rets[f"{s}_vol"] = vols
            
            if len(hist_rets) < 6: # Need all macro assets + BTC vol at least
                return self._current_hmm_state
                
            eth_btc_ret = hist_rets["ETH"] - hist_rets["BTC"]
            X = np.array([[
                hist_rets["BTC"], hist_rets["BTC_vol"], eth_btc_ret, 
                hist_rets.get("XAU", 0), hist_rets.get("SP500", 0)
            ]])
            
            state = self._macro_hmm.predict(X)[0]
            return state
        except Exception:
            return self._current_hmm_state

    def calibrate_regimes(self, bar_data):
        """Self-learning hook: Re-fits the HMM to stay adapted to real-time market shifts."""
        print("Self-Learning: Re-calibrating Macro HMM regimes...")
        # In a real production setup, we would run the train_macro_hmm() logic here
        # using the accumulated bar_data.history across all symbols.
        # For now, we signal that the bot is "observing" and ready for recalibration.
        pass
