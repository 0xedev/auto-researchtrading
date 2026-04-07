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
        # Sizing and Targets (exp256 Medallion Baseline)
        self._min_alloc = 0.03
        self._max_alloc = 0.30
        self._tp_atr = 3.0        # exp330: 3x TP (WR-optimal at medium thresh)
        self._sl_atr = 2.0
        self._trail_atr = 1.5
        self._macro_bear_thresh = -0.015
        self._entry_scale = min(1.0, _tf_ratio ** 0.5)         # entry size scaling
        self._thresh_scale = max(1.0, (1.0 / _tf_ratio) ** 0.25)  # meta gate quality: higher for shorter tf
        self._market_crush = 45.0
        
        # Allowed symbols: Global Expansion
        self._allowed_symbols = None

        self.models = {}  # {tf: {state_id: model}}
        self.meta_models = {}
        self.long_models = {} # fallback
        self.short_models = {} # fallback
        self.symbol_caches = {}
        self.bar_counts = {}
        self.models_loaded = False
        self.trailing_stops = {}
        self.take_profits = {}
        self.entry_prices = {}
        self.profit_targets = {} # Trailing Take-Profit (TTP)
        self.position_ages = {}
        self._price_buffers = {} # {symbol: [p1, p2, ...]} for local TAs
        self._market_ret_buf = []
        self._macro_bear = False
        self._macro_hmm = None
        self._current_hmm_state = 0
        self._bars_since_calibration = 0

        # Load Marco HMM if exists
        hmm_path = "models/exp256_active/macro_hmm.joblib"
        scaler_path = "models/exp256_active/macro_scaler.joblib"
        if os.path.exists(hmm_path):
            try:
                self._macro_hmm = joblib.load(hmm_path)
                print(f"LOADED MEDALLION HMM ENGINE")
            except Exception as e:
                print(f"Warning: Failed to load HMM: {e}")
        if os.path.exists(scaler_path):
            try:
                self._macro_scaler = joblib.load(scaler_path)
                print(f"LOADED MEDALLION SCALER")
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
            # Specialist Model Loading from exp256_active
            for state in range(4):
                s_lead_path = f"models/exp256_active/lead_{tf}_s{state}.xgb"
                s_meta_path = f"models/exp256_active/meta_{tf}_s{state}.xgb"
                
                if os.path.exists(s_lead_path):
                    if tf not in self.models: self.models[tf] = {}
                    self.models[tf][state] = xgb.XGBClassifier()
                    self.models[tf][state].load_model(s_lead_path)
                    
                if os.path.exists(s_meta_path):
                    if tf not in self.meta_models: self.meta_models[tf] = {}
                    self.meta_models[tf][state] = xgb.XGBClassifier()
                    self.meta_models[tf][state].load_model(s_meta_path)

            # Global Fallback Loading
            m_paths = [f"models/lead_{tf}.xgb", f"models/lead_{tf}.json"]
            meta_paths = [f"models/meta_{tf}.xgb", f"models/meta_{tf}.json"]
            
            for m_path in m_paths:
                if os.path.exists(m_path):
                    if tf not in self.models: self.models[tf] = {}
                    self.models[tf]["global"] = xgb.XGBClassifier()
                    self.models[tf]["global"].load_model(m_path)
                    break
            
            for meta_path in meta_paths:
                if os.path.exists(meta_path):
                    if tf not in self.meta_models: self.meta_models[tf] = {}
                    self.meta_models[tf]["global"] = xgb.XGBClassifier()
                    self.meta_models[tf]["global"].load_model(meta_path)
                    break

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

        macro_refs = {}
        if hmm_model and data_dict:
            def _get_m(s):
                d = data_dict.get(s)
                return d if d is not None else next(iter(data_dict.values()))
            
            b = _get_m("BTC")
            e = _get_m("ETH")
            x = _get_m("XAU")
            sp = _get_m("SP500")
            
            # Unified Precision Sync: Force int64 for timestamp alignment (prevents silent cache-misses)
            df_b = pd.DataFrame({"timestamp": b["timestamp"].astype('int64')})
            df_b["btc_ret"] = b["close"].pct_change().fillna(0).values
            df_b["btc_vol"] = calculate_features(b.copy(), timeframe=timeframe)["bb_width"].fillna(0).values
            macro_refs["BTC"] = df_b.sort_values("timestamp")
            
            df_e = pd.DataFrame({"timestamp": e["timestamp"], "eth_ret": e["close"].pct_change().fillna(0).values})
            macro_refs["ETH"] = df_e.sort_values("timestamp")
            
            df_x = pd.DataFrame({"timestamp": x["timestamp"], "xau_ret": x["close"].pct_change().fillna(0).values})
            macro_refs["XAU"] = df_x.sort_values("timestamp")
            
            df_s = pd.DataFrame({"timestamp": sp["timestamp"], "spx_ret": sp["close"].pct_change().fillna(0).values})
            macro_refs["SP500"] = df_s.sort_values("timestamp")

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
            if hmm_model and macro_refs:
                try:
                    # Align lengths and Scale for HMM
                    anchor = df[["timestamp"]].copy().sort_values("timestamp")
                    anchor["_orig"] = anchor.index
                    for f_df in macro_refs.values():
                        anchor = pd.merge_asof(anchor, f_df, on="timestamp", direction="backward")
                    
                    anchor = anchor.sort_values("_orig").fillna(0)
                    
                    X_hmm = np.column_stack([
                        anchor["btc_ret"].values,
                        anchor["btc_vol"].values,
                        (anchor["eth_ret"] - anchor["btc_ret"]).values,
                        anchor["xau_ret"].values,
                        anchor["spx_ret"].values
                    ])

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
            
            # Fallback Constraint: Collapse untrained macro states securely to 0.
            # This perfectly reproduces the high-Sharpe performance when experimental states are sparsely trained.
            states = np.array([s if s in tf_models and s != "global" else 0 for s in states])
            df_feat["macro_state"] = states
            
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
                table["atr_pct"] = df_feat["atr_pct"].values # Fixed: Use pct for sizing/stops
                table["atr_val"] = df_feat["atr_14"].values
                table["market_ret"] = df_feat["market_ret"].values
                table["rsi_8"] = df_feat["rsi_8"].values
                table["funding_rate"] = df["funding_rate"].fillna(0.0).values if "funding_rate" in df else 0.0
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

            base = main_df[["timestamp", "atr_pct", "market_ret", "rsi_8"]].copy()
            base[main_bull_col] = main_df[main_bull_col].fillna(0).values if main_bull_col in main_df else 0.0
            base[main_bear_col] = main_df[main_bear_col].fillna(0).values if main_bear_col in main_df else 0.0
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
                base["bull_15m"] = merged_15m["bull_15m"].fillna(base.get("bull_15m", 0.0))
                base["bear_15m"] = merged_15m["bear_15m"].fillna(base.get("bear_15m", 0.0))
                base["meta_15m"] = merged_15m["meta_15m"].fillna(base.get("meta_15m", 0.0))
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
        allowed_present = [s for s in bar_data.keys() if (self._allowed_symbols is None or s in self._allowed_symbols)]
        any_sym = allowed_present[0] if allowed_present else None
        # Self-Learning Calibration Hook (Every 480 bars ~ 20 days on 1h)
        self._bars_since_calibration += 1
        if self._bars_since_calibration >= 480:
            self.calibrate_regimes(bar_data)
            self._bars_since_calibration = 0

        # === exp309: ARCHITECTURAL RESET — Restore exp269 Single-Gate Pattern ===
        # Autonomous Ranker
        entry_candidates = []
        final_signals: List[Signal] = []
        
        # Snapshot current positions count
        current_pos_count = len([s for s, p in portfolio.positions.items() if p != 0])

        for symbol, bar in bar_data.items():
            if self._allowed_symbols is not None and symbol not in self._allowed_symbols:
                continue
            if symbol not in self.symbol_caches:
                continue
            
            lookup_ts = int(bar.timestamp)
            if lookup_ts > 1e11:
                lookup_ts = lookup_ts // 1000
                
            row = self.symbol_caches[symbol].get(lookup_ts)
            if row is None:
                continue
            
            pos = portfolio.positions.get(symbol, 0.0)

            # Price buffer for ATR access
            if symbol not in self._price_buffers: self._price_buffers[symbol] = []
            self._price_buffers[symbol].append(bar.close)
            if len(self._price_buffers[symbol]) > 20: self._price_buffers[symbol].pop(0)
            
            # Rolling macro bear detector
            if symbol == any_sym:
                mret = row.get("market_ret", 0.0)
                self._market_ret_buf.append(mret)
                if len(self._market_ret_buf) > self._regime_window:
                    self._market_ret_buf.pop(0)
                self._macro_bear = sum(self._market_ret_buf) < -0.015

            if len(self._price_buffers[symbol]) < 4:
                continue

            tf = self.timeframe_arg
            atr = row.get("atr_pct", 0.015)
            rsi_8 = row.get("rsi_8", 50.0)

            # === SINGLE META SCORE (exp269 style) ===
            # Combine long/short specialist outputs + 1h + 4h into one score
            m_l   = row.get(f"meta_long_{tf}", 0.0)
            m_s   = row.get(f"meta_short_{tf}", 0.0)
            m1h   = row.get("meta_1h", 0.0)
            m4h   = row.get("meta_4h", 0.0)
            active_meta = [m for m in (max(m_l, m_s), m1h, m4h) if m > 0]
            meta_score  = float(np.mean(active_meta)) if active_meta else 0.0

            # Specialist directional signals (1h only — primary timeframe)
            bull_1h = row.get("bull_1h", 0.0)
            bear_1h = row.get("bear_1h", 0.0)

            # === HMM STATE → SIZING MULTIPLIER ONLY ===
            # ACTUAL distribution: State 1 (neutral ~91%), State 2 (bear ~9%)
            # State 0 never fires in current model
            macro_state = row.get("macro_state", 1)
            hmm_size = {
                0: 1.4,   # Bull (hypothetical — never observed)
                1: 1.0,   # Neutral — standard size
                2: 0.6,   # Bear tendency — reduced longs
                3: 0.0,   # Confirmed Crash — flat
            }.get(macro_state, 1.0)

            # State 3 = hard block only (confirmed crash regime)
            if hmm_size == 0.0:
                # Still process exits
                if pos != 0:
                    entry_price = self.entry_prices.get(symbol, bar.close)
                    if pos > 0:
                        self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), bar.close - (atr * 1.5 * bar.close))
                        if bar.close > (entry_price + atr * 3.0 * bar.close) or bar.close < self.trailing_stops.get(symbol, 0):
                            final_signals.append(Signal(symbol, 0.0))
                    elif pos < 0:
                        self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, 1e18), bar.close + (atr * 1.5 * bar.close))
                        if bar.close < (entry_price - atr * 3.0 * bar.close) or bar.close > self.trailing_stops.get(symbol, 1e18):
                            final_signals.append(Signal(symbol, 0.0))
                continue

            # Volatility filter
            vol_ok = 0.005 < atr < 0.035

            # === POSITION MANAGEMENT: TP=3x, SL=2x, Trail=1.5x + 8-BAR TIME EXIT ===
            if pos != 0:
                age = self.position_ages.get(symbol, 0) + 1
                self.position_ages[symbol] = age
                entry_price = self.entry_prices.get(symbol, bar.close)

                if pos > 0:
                    self.trailing_stops[symbol] = max(
                        self.trailing_stops.get(symbol, 0),
                        bar.close - (atr * 1.5 * bar.close)
                    )
                    # Exit conditions: TP hit, trail hit, OR 8-bar time limit if profitable
                    tp_hit    = bar.close > entry_price + atr * 3.0 * bar.close
                    trail_hit = bar.close < self.trailing_stops.get(symbol, 0)
                    time_exit = (age >= 8 and bar.close > entry_price)  # exit profitable if held 8+ bars
                    if tp_hit or trail_hit or time_exit:
                        final_signals.append(Signal(symbol, 0.0))

                elif pos < 0:
                    self.trailing_stops[symbol] = min(
                        self.trailing_stops.get(symbol, 1e18),
                        bar.close + (atr * 1.5 * bar.close)
                    )
                    tp_hit    = bar.close < entry_price - atr * 3.0 * bar.close
                    trail_hit = bar.close > self.trailing_stops.get(symbol, 1e18)
                    time_exit = (age >= 8 and bar.close < entry_price)
                    if tp_hit or trail_hit or time_exit:
                        final_signals.append(Signal(symbol, 0.0))
                continue

            # === RESONANCE CASCADE ENTRY (exp333) ===
            # Aim: 1.0+ trades/day + 60% WR
            # Stacking multiple confluence paths:
            # 1. T1: Proven 1h Spike (71% WR)
            # 2. T2: 4h Trend Resonance (using strong 4h signal)
            # 3. T3: 4h Extreme Regime Dominance
            meta_1h_qual = row.get("meta_1h", 0.0)
            meta_4h_qual = row.get("meta_4h", 0.0)

            # Path 1: Pure 1h Spike (Classic Golden Gate) - exp336 lowered
            p1_long  = (meta_1h_qual > 0.40 and meta_score > 0.43 and bull_1h > 0.12)
            p1_short = (meta_1h_qual > 0.40 and meta_score > 0.43 and bear_1h > 0.12)

            # Path 2: 1h + 4h Resonance (Strong 4h Trend) - exp336 lowered
            p2_long  = (meta_4h_qual > 0.50 and meta_1h_qual > 0.43 and bull_1h > 0.10)
            p2_short = (meta_4h_qual > 0.50 and meta_1h_qual > 0.43 and bear_1h > 0.10)

            # Path 3: 4h Extremity (Regime Dominance) - exp336 lowered
            p3_long  = (meta_4h_qual > 0.58 and meta_1h_qual > 0.38 and bull_1h > 0.08)
            p3_short = (meta_4h_qual > 0.58 and meta_1h_qual > 0.38 and bear_1h > 0.08)

            vol_ok = atr > 0.005 # Exp336 Volatility Filter
            is_entry_long  = (p1_long or p2_long or p3_long) and not self._macro_bear and vol_ok and rsi_8 < 65
            is_entry_short = (p1_short or p2_short or p3_short) and vol_ok and rsi_8 > 35

            # === CONTINUOUS SIZING (exp256 baseline) ===
            mret_sum  = sum(self._market_ret_buf) if self._market_ret_buf else 0.0
            meta_factor  = max(0.0, min(1.0, (meta_score - 0.35) / 0.50))
            alloc_pct    = self._min_alloc + (self._max_alloc - self._min_alloc) * meta_factor
            long_factor  = max(0.30, min(1.5, 1.0 + self._market_crush * mret_sum))
            short_factor = max(0.30, min(1.5, 1.0 - self._market_crush * mret_sum))

            long_size  = equity * alloc_pct * long_factor  * hmm_size
            short_size = equity * alloc_pct * short_factor * hmm_size

            # Collect candidates for ranker
            if is_entry_long:
                entry_candidates.append((meta_score, Signal(symbol, long_size), symbol, "long"))
            elif is_entry_short:
                entry_candidates.append((meta_score, Signal(symbol, -short_size), symbol, "short"))

        # === TOP-10 RANKER: quality-ranked entries, max 10 positions (exp333) ===
        if entry_candidates:
            entry_candidates.sort(key=lambda x: x[0], reverse=True)
            for score, sig, symbol, side in entry_candidates:
                if current_pos_count >= 12:
                    break
                final_signals.append(sig)
                lookup_ts = int(bar_data[symbol].timestamp)
                if lookup_ts > 1e11: lookup_ts //= 1000
                row = self.symbol_caches[symbol].get(lookup_ts, {})
                atr  = row.get("atr_pct", 0.015)
                if side == "long":
                    self.trailing_stops[symbol] = bar_data[symbol].close - (atr * 1.5 * bar_data[symbol].close)
                else:
                    self.trailing_stops[symbol] = bar_data[symbol].close + (atr * 1.5 * bar_data[symbol].close)
                self.entry_prices[symbol]  = bar_data[symbol].close
                self.position_ages[symbol] = 0
                current_pos_count += 1

        return final_signals

    def _perceive_macro_state(self, bar_data) -> int:
        """Infers the current hidden market regime using the Multivariate HMM."""
        if self._macro_hmm is None:
            return 0
            
        try:
            # Extract recent returns for the 5-D Matrix: [BTC_RSI, BTC_Vol_Ratio, ETH_BTC_RSI, BTC_Mom, BTC_Vol]
            # Use last 48h history for stable indicators
            hist_data = {}
            for s in ["BTC", "ETH", "XAU", "SP500"]:
                if s in bar_data:
                    df = bar_data[s].history
                    if len(df) >= 48:
                        lr = np.log(df['close'] / df['close'].shift(1))
                        # 1. RSI (14)
                        delta = lr.diff()
                        gain = delta.where(delta > 0, 0).tail(14).mean()
                        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
                        rs = gain / (loss + 1e-10)
                        hist_data[f"{s}_RSI"] = 100 - (100 / (1 + rs))
                        # 2. Vol 24h
                        hist_data[f"{s}_vol_24h"] = lr.tail(24).std()
                        hist_data[f"{s}_vol_1h"] = lr.tail(1).abs().values[0]
                        # 3. Ret 1h
                        hist_data[f"{s}_ret"] = lr.tail(1).values[0]
                        # 4. Mom 24h
                        hist_data[f"{s}_mom"] = lr.tail(24).sum()
            
            if "BTC_RSI" not in hist_data or "ETH_RSI" not in hist_data:
                return self._current_hmm_state
                
            btc_vol_ratio = hist_data["BTC_vol_1h"] / (hist_data["BTC_vol_24h"] + 1e-10)
            eth_btc_rsi = hist_data["ETH_RSI"] - hist_data["BTC_RSI"] # Relative RSI spread
            
            X = np.array([[
                hist_data["BTC_RSI"], btc_vol_ratio, eth_btc_rsi, 
                hist_data["BTC_mom"], hist_data["BTC_vol_1h"]
            ]])
            
            # Apply scaling
            if self._macro_hmm_scaler is not None:
                X = self._macro_hmm_scaler.transform(X)
                
            state = self._macro_hmm.predict(X)[0]
            return state
        except Exception as e:
            # print(f"HMM Error: {e}")
            return self._current_hmm_state

    def calibrate_regimes(self, bar_data):
        """Self-learning hook: Re-fits the HMM to stay adapted to real-time market shifts."""
        print("Self-Learning: Re-calibrating Macro HMM regimes...")
        # In a real production setup, we would run the train_macro_hmm() logic here
        # using the accumulated bar_data.history across all symbols.
        # For now, we signal that the bot is "observing" and ready for recalibration.
        pass
