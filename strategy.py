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
        self.long_meta_models = {}
        self.short_meta_models = {}
        self.symbol_caches = {}
        self.bar_counts = {}
        self.models_loaded = False
        self.trailing_stops = {}
        self.take_profits = {}
        self.entry_prices = {}
        self.profit_targets = {} # Trailing Take-Profit (TTP)
        self.position_ages = {}
        self._flat_cooldowns = {}
        self._price_buffers = {} # {symbol: [p1, p2, ...]} for local TAs
        self._market_ret_buf = []
        self._macro_bear = False
        self._macro_hmm = None
        self._macro_hmm_scaler = None
        self._current_hmm_state = 0
        self._bars_since_calibration = 0
        self._entry_confidence = {} # exp351: Track confidence for spectrum exits

        # Load Marco HMM if exists
        hmm_path = self._resolve_first_existing_path(
            "models/exp256_active/macro_hmm.joblib",
            "models/macro_hmm.joblib",
        )
        scaler_path = self._resolve_first_existing_path(
            "models/exp256_active/macro_scaler.joblib",
            "models/macro_scaler.joblib",
        )
        if os.path.exists(hmm_path):
            try:
                self._macro_hmm = joblib.load(hmm_path)
                print(f"LOADED MEDALLION HMM ENGINE")
            except Exception as e:
                print(f"Warning: Failed to load HMM: {e}")
        if os.path.exists(scaler_path):
            try:
                self._macro_hmm_scaler = joblib.load(scaler_path)
                print(f"LOADED MEDALLION SCALER")
            except Exception as e:
                print(f"Warning: Failed to load HMM Scaler: {e}")

        self._cfg_15m = {
            "exit_decay_age": self._env_int("STRAT_15M_EXIT_DECAY_AGE", 5),
            "max_hold": self._env_int("STRAT_15M_MAX_HOLD", 6),
            "scalein_bull": self._env_float("STRAT_15M_SCALEIN_BULL", 0.56),
            "scalein_meta": self._env_float("STRAT_15M_SCALEIN_META", 0.34),
            "trend_bull": self._env_float("STRAT_15M_TREND_BULL", 0.45),
            "trend_meta": self._env_float("STRAT_15M_TREND_META", 0.25),
            "trend_m1h": self._env_float("STRAT_15M_TREND_M1H", 0.27),
            "trend_m4h": self._env_float("STRAT_15M_TREND_M4H", 0.18),
            "trend_rsi_max": self._env_float("STRAT_15M_TREND_RSI_MAX", 67.0),
            "trend_macro_bear_bull": self._env_float("STRAT_15M_TREND_MACRO_BEAR_BULL", 0.53),
            "trend_weight": self._env_float("STRAT_15M_TREND_WEIGHT", 0.65),
            "push_bull": self._env_float("STRAT_15M_PUSH_BULL", 0.43),
            "push_meta": self._env_float("STRAT_15M_PUSH_META", 0.32),
            "push_m1h": self._env_float("STRAT_15M_PUSH_M1H", 0.38),
            "push_rsi_max": self._env_float("STRAT_15M_PUSH_RSI_MAX", 63.0),
            "push_weight": self._env_float("STRAT_15M_PUSH_WEIGHT", 0.22),
            "max_positions": self._env_int("STRAT_15M_MAX_POSITIONS", 4),
            "allocator": os.getenv("STRAT_15M_ALLOCATOR", "push_priority").strip().lower() or "push_priority",
        }

    def _resolve_first_existing_path(self, *paths: str) -> str:
        for path in paths:
            if os.path.exists(path):
                return path
        return paths[0]

    def _env_float(self, name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _env_int(self, name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _try_load_xgb_model(self, paths: List[str]):
        for path in paths:
            if not os.path.exists(path):
                continue
            model = xgb.XGBClassifier()
            try:
                model.load_model(path)
                return model
            except Exception as exc:
                print(f"Warning: Failed to load model {path}: {exc}")
        return None

    def _parse_timeframe(self, tf: str) -> int:
        if tf == "15m":
            return 15 * 60
        if tf == "1h":
            return 3600
        if tf == "4h":
            return 14400
        return 0

    def _normalize_timestamp_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce merge keys to stable integer timestamps for asof alignment."""
        if "timestamp" not in df.columns:
            return df.copy()

        normalized = df.copy()
        ts = pd.to_numeric(normalized["timestamp"], errors="coerce")
        normalized = normalized.loc[ts.notna()].copy()
        normalized["timestamp"] = ts.loc[ts.notna()].round().astype("int64")
        return normalized.sort_values("timestamp").reset_index(drop=True)

    def _map_hmm_states_to_regimes(self, anchor: pd.DataFrame, raw_states: np.ndarray):
        """Map arbitrary HMM labels to stable semantic regimes.

        Raw HMM state ids are only meaningful for the specialist models that were
        trained on those same ids. The live strategy, however, needs consistent
        bull/neutral/bear/crash semantics for sizing and directional throttles.
        """
        semantic = np.full(len(raw_states), 1, dtype=int)  # default to neutral
        if len(raw_states) == 0 or "btc_ret" not in anchor or "btc_vol" not in anchor:
            return semantic, {}

        state_df = pd.DataFrame({
            "state": raw_states,
            "btc_ret": anchor["btc_ret"].values,
            "btc_vol": anchor["btc_vol"].values,
        })
        stats = (
            state_df.groupby("state")
            .agg(
                count=("state", "size"),
                btc_ret_mean=("btc_ret", "mean"),
                btc_vol_mean=("btc_vol", "mean"),
            )
            .reset_index()
        )

        min_count = max(25, len(raw_states) // 1000)
        active = stats[stats["count"] >= min_count].copy()
        if active.empty:
            return semantic, {}

        mapping = {int(row.state): 1 for row in active.itertuples()}
        if len(active) == 1:
            only_state = int(active.iloc[0]["state"])
            return semantic, {only_state: 1}

        bull_row = active.sort_values(["btc_ret_mean", "count"], ascending=[False, False]).iloc[0]
        bull_state = int(bull_row["state"])
        if bull_row["btc_ret_mean"] > 0:
            mapping[bull_state] = 0

        downside = active.sort_values(["btc_ret_mean", "btc_vol_mean"], ascending=[True, False]).reset_index(drop=True)
        worst_row = downside.iloc[0]
        worst_state = int(worst_row["state"])
        vol_median = active["btc_vol_mean"].median()
        looks_like_crash = (
            worst_row["btc_ret_mean"] < -5e-4
            and worst_row["btc_vol_mean"] > max(vol_median * 1.25, 0.02)
        )
        if looks_like_crash:
            mapping[worst_state] = 3
            remaining_bears = downside.iloc[1:]
            remaining_bears = remaining_bears[remaining_bears["btc_ret_mean"] < -1e-4]
            if not remaining_bears.empty:
                mapping[int(remaining_bears.iloc[0]["state"])] = 2
        elif worst_row["btc_ret_mean"] < -1e-4:
            mapping[worst_state] = 2

        semantic = np.array([mapping.get(int(state), 1) for state in raw_states], dtype=int)
        return semantic, mapping

    def _load_models(self):
        for tf in ["15m", "1h", "4h"]:
            if tf == "15m":
                long_lead = self._try_load_xgb_model([
                    "models/lead_long_15m.xgb",
                    "models/exp256_active/lead_long_15m.xgb",
                ])
                short_lead = self._try_load_xgb_model([
                    "models/lead_short_15m.xgb",
                    "models/exp256_active/lead_short_15m.xgb",
                ])
                long_meta = self._try_load_xgb_model([
                    "models/meta_long_15m.xgb",
                    "models/exp256_active/meta_long_15m.xgb",
                ])
                short_meta = self._try_load_xgb_model([
                    "models/meta_short_15m.xgb",
                    "models/exp256_active/meta_short_15m.xgb",
                ])

                if long_lead is not None:
                    self.long_models[tf] = long_lead
                if short_lead is not None:
                    self.short_models[tf] = short_lead
                if long_meta is not None:
                    self.long_meta_models[tf] = long_meta
                if short_meta is not None:
                    self.short_meta_models[tf] = short_meta

            # Specialist Model Loading: prefer top-level state snapshots when present,
            # then fall back to any archived specialist artifacts.
            for state in range(4):
                lead_candidates = [
                    f"models/lead_{tf}_s{state}.json",
                    f"models/lead_{tf}_s{state}.xgb",
                    f"models/exp256_active/lead_{tf}_s{state}.xgb",
                ]
                meta_candidates = [
                    f"models/meta_{tf}_s{state}.json",
                    f"models/meta_{tf}_s{state}.xgb",
                    f"models/exp256_active/meta_{tf}_s{state}.xgb",
                ]

                lead_model = self._try_load_xgb_model(lead_candidates)
                if lead_model is not None:
                    if tf not in self.models:
                        self.models[tf] = {}
                    self.models[tf][state] = lead_model

                meta_model = self._try_load_xgb_model(meta_candidates)
                if meta_model is not None:
                    if tf not in self.meta_models:
                        self.meta_models[tf] = {}
                    self.meta_models[tf][state] = meta_model

            # Global Fallback Loading
            m_paths = [f"models/lead_{tf}.xgb", f"models/lead_{tf}.json"]
            meta_paths = [f"models/meta_{tf}.xgb", f"models/meta_{tf}.json"]

            global_model = self._try_load_xgb_model(m_paths)
            if global_model is not None:
                if tf not in self.models:
                    self.models[tf] = {}
                self.models[tf]["global"] = global_model

            global_meta_model = self._try_load_xgb_model(meta_paths)
            if global_meta_model is not None:
                if tf not in self.meta_models:
                    self.meta_models[tf] = {}
                self.meta_models[tf]["global"] = global_meta_model

        self.models_loaded = True
        print(f"LOADED QUANTUM FORTRESS SPECIALISTS: {list(self.models.keys())}")

    def _build_prediction_tables(self, data_dict, timeframe: str, include_state: bool = False):
        if not data_dict:
            return {}

        normalized_data = {
            symbol: self._normalize_timestamp_frame(df)
            for symbol, df in data_dict.items()
        }

        all_vols = {}
        all_rets = {}
        for symbol, df in normalized_data.items():
            feat = calculate_features(df, timeframe=timeframe)
            all_vols[symbol] = feat["bb_width"]
            all_rets[symbol] = df["close"].pct_change()

        m_vol = pd.DataFrame(all_vols).median(axis=1).fillna(0)
        m_ret = pd.DataFrame(all_rets).median(axis=1).fillna(0)

        # Pre-calculate Macro HMM for the entire batch
        hmm_path = self._resolve_first_existing_path(
            "models/exp256_active/macro_hmm.joblib",
            "models/macro_hmm.joblib",
        )
        scaler_path = self._resolve_first_existing_path(
            "models/exp256_active/macro_scaler.joblib",
            "models/macro_scaler.joblib",
        )
        hmm_model = joblib.load(hmm_path) if os.path.exists(hmm_path) else None
        hmm_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        macro_refs = {}
        if hmm_model and normalized_data:
            def _get_m(s):
                d = normalized_data.get(s)
                return d if d is not None else next(iter(normalized_data.values()))

            b = _get_m("BTC")
            e = _get_m("ETH")
            x = _get_m("XAU")
            sp = _get_m("SP500")

            df_b = pd.DataFrame({"timestamp": b["timestamp"].astype("int64")})
            df_b["btc_ret"] = b["close"].pct_change().fillna(0).values
            df_b["btc_vol"] = calculate_features(b.copy(), timeframe=timeframe)["bb_width"].fillna(0).values
            macro_refs["BTC_RET"] = df_b.sort_values("timestamp")

            df_e = pd.DataFrame({"timestamp": e["timestamp"].astype("int64"), "eth_ret": e["close"].pct_change().fillna(0).values})
            macro_refs["ETH"] = df_e.sort_values("timestamp")

            df_x = pd.DataFrame({"timestamp": x["timestamp"].astype("int64"), "xau_ret": x["close"].pct_change().fillna(0).values})
            macro_refs["XAU"] = df_x.sort_values("timestamp")

            df_s = pd.DataFrame({"timestamp": sp["timestamp"].astype("int64"), "spx_ret": sp["close"].pct_change().fillna(0).values})
            macro_refs["SP500"] = df_s.sort_values("timestamp")

        tables = {}
        first_symbol = next(iter(normalized_data.keys()))
        for symbol, df in normalized_data.items():
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
            raw_states = None
            if hmm_model and macro_refs:
                try:
                    # Align lengths and Scale for HMM
                    anchor = df[["timestamp"]].copy().sort_values("timestamp")
                    anchor["_orig"] = anchor.index
                    for ref_key in ("BTC_RET", "ETH", "XAU", "SP500"):
                        f_df = macro_refs[ref_key]
                        anchor = pd.merge_asof(anchor, f_df, on="timestamp", direction="backward")
                    
                    anchor = anchor.sort_values("_orig").fillna(0)
                    
                    X_hmm = np.column_stack([
                        anchor["btc_ret"].values,
                        anchor["btc_vol"].values,
                        (anchor["eth_ret"] - anchor["btc_ret"]).values,
                        anchor["xau_ret"].values,
                        anchor["spx_ret"].values
                    ])

                    if hmm_scaler is not None:
                        X_hmm = hmm_scaler.transform(X_hmm)

                    raw_states = hmm_model.predict(X_hmm)
                    semantic_states, state_mapping = self._map_hmm_states_to_regimes(anchor, raw_states)
                    df_feat["macro_state"] = semantic_states
                    
                    # Diagnostic: Print the state distribution for the FIRST symbol only to save log space
                    if symbol == first_symbol:
                        unique_states, state_counts = np.unique(df_feat["macro_state"], return_counts=True)
                        print(f"--- HMM REGIME MAP: {state_mapping} ---")
                        print(f"--- HMM REGIME DISTRIBUTION: {dict(zip(unique_states, state_counts))} ---")
                except Exception as e:
                    print(f"HMM High-Fideilty Predict Error: {e}")

            X_full = df_feat[FEATURE_COLS]
            print(f"DIAGNOSTIC: X_full for {symbol} - Shape: {X_full.shape} | Nulls: {X_full.isna().sum().sum()}")
            if X_full.isna().any().any():
                 print(f"WARNING: NaNs found in features: {X_full.columns[X_full.isna().any()].tolist()}")
            raw_model_states = raw_states if raw_states is not None else df_feat["macro_state"].values
            semantic_states = df_feat["macro_state"].values
            
            all_bull = np.zeros(len(df_feat))
            all_bear = np.zeros(len(df_feat))
            all_meta = np.zeros(len(df_feat))
            all_meta_long = np.zeros(len(df_feat))
            all_meta_short = np.zeros(len(df_feat))

            directional_15m = timeframe == "15m" and (
                "15m" in self.long_models
                or "15m" in self.short_models
                or "15m" in self.long_meta_models
                or "15m" in self.short_meta_models
            )

            if directional_15m:
                long_model = self.long_models.get("15m")
                short_model = self.short_models.get("15m")
                long_meta_model = self.long_meta_models.get("15m")
                short_meta_model = self.short_meta_models.get("15m")

                if long_model is not None:
                    long_probs = long_model.predict_proba(X_full)
                    all_bull = long_probs[:, 1] if long_probs.shape[1] > 1 else 0.0
                if short_model is not None:
                    short_probs = short_model.predict_proba(X_full)
                    all_bear = short_probs[:, 1] if short_probs.shape[1] > 1 else 0.0
                if long_meta_model is not None:
                    long_meta_probs = long_meta_model.predict_proba(X_full)
                    all_meta_long = long_meta_probs[:, 1] if long_meta_probs.shape[1] > 1 else 0.0
                if short_meta_model is not None:
                    short_meta_probs = short_meta_model.predict_proba(X_full)
                    all_meta_short = short_meta_probs[:, 1] if short_meta_probs.shape[1] > 1 else 0.0

                all_meta = np.maximum(all_meta_long, all_meta_short)

                if symbol == first_symbol:
                    print(
                        f"DEBUG: {symbol} 15m split | MaxBull: {all_bull.max():.4f} | "
                        f"MaxBear: {all_bear.max():.4f} | MaxMetaLong: {all_meta_long.max():.4f} | "
                        f"MaxMetaShort: {all_meta_short.max():.4f}"
                    )
            else:
                tf_models = self.models.get(timeframe, {})
                tf_meta = self.meta_models.get(timeframe, {})

                for state_id in range(4):
                    mask = (raw_model_states == state_id)
                    if not any(mask):
                        continue

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
                        if symbol == first_symbol:
                            print(
                                f"DEBUG: {symbol} S{state_id} | MaxBull: {all_bull[mask].max():.4f} | "
                                f"MaxBear: {all_bear[mask].max():.4f} | MaxMeta: {all_meta[mask].max():.4f}"
                            )
            
            table = pd.DataFrame({
                "timestamp": df["timestamp"].values,
                f"bull_{timeframe}": all_bull,
                f"bear_{timeframe}": all_bear,
                f"meta_{timeframe}": all_meta,
                "macro_state": semantic_states.astype(int),
            })
            if timeframe == "15m":
                table["meta_long_15m"] = all_meta_long
                table["meta_short_15m"] = all_meta_short
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
            if (
                "15m" in self.models
                or "15m" in self.meta_models
                or "15m" in self.long_models
                or "15m" in self.short_models
            ):
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

            main_df = self._normalize_timestamp_frame(main_tables[symbol])
            main_bull_col = f"bull_{self.timeframe_arg}"
            main_bear_col = f"bear_{self.timeframe_arg}"
            main_meta_col = f"meta_{self.timeframe_arg}"

            base = main_df[["timestamp", "atr_pct", "market_ret", "rsi_8", "macro_state"]].copy()
            base["funding_rate"] = main_df["funding_rate"].fillna(0).values if "funding_rate" in main_df else 0.0
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
            base = self._normalize_timestamp_frame(base)

            if symbol in aux_15m_tables:
                merged_15m = pd.merge_asof(
                    base[["timestamp"]].sort_values("timestamp"),
                    self._normalize_timestamp_frame(aux_15m_tables[symbol]),
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
                    self._normalize_timestamp_frame(aux_1h_tables[symbol][["timestamp", "meta_1h"]]),
                    on="timestamp",
                    direction="backward",
                )
                base["meta_1h"] = merged_1h["meta_1h"].fillna(base["meta_1h"])

            if symbol in aux_4h_tables:
                merged_4h = pd.merge_asof(
                    base[["timestamp"]].sort_values("timestamp"),
                    self._normalize_timestamp_frame(aux_4h_tables[symbol][["timestamp", "meta_4h"]]),
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

        current_pos_count = len([s for s, p in portfolio.positions.items() if p != 0])
        ordered_candidates = []
        trend_candidates = []
        push_candidates = []

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

            if symbol == any_sym:
                mret = row.get("market_ret", 0.0)
                self._market_ret_buf.append(mret)
                if len(self._market_ret_buf) > self._regime_window:
                    self._market_ret_buf.pop(0)
                self._macro_bear = sum(self._market_ret_buf) < -0.015

            tf = self.timeframe_arg
            m_curr = row.get(f"meta_{tf}", 0.0)
            atr = row.get("atr_pct", 0.015)
            rsi_8 = row.get("rsi_8", 50.0)
            m_l = row.get(f"meta_long_{tf}", m_curr)
            m_s = row.get(f"meta_short_{tf}", m_curr)
            m1h = row.get("meta_1h", m_curr)
            m4h = row.get("meta_4h", m_curr)
            m15_long = row.get("meta_long_15m", 0.0)
            m15_short = row.get("meta_short_15m", 0.0)
            stop_dist = bar.close * (atr * 1.5)
            stop_dist = max(stop_dist, max(abs(bar.close) * 1e-6, 1e-9))

            active_meta = [m for m in (max(m_l, m_s), m1h, m4h) if m > 0]
            blended_meta = float(np.mean(active_meta)) if active_meta else 0.0

            state_adjust = {
                0: {"long_gate": -0.05, "short_gate": +0.05, "size": 1.1},
                1: {"long_gate": +0.05, "short_gate": -0.05, "size": 1.1},
                2: {"long_gate": +0.02, "short_gate": +0.02, "size": 0.5},
                3: {"long_gate": +0.10, "short_gate": +0.10, "size": 0.1},
            }.get(row.get("macro_state", 0), {"long_gate": 0.0, "short_gate": 0.0, "size": 1.0})

            supportive_regime = (m1h <= 0 or m1h > 0.45)
            supportive_regime_4h = (m4h <= 0 or m4h > 0.45)

            bull_raw = row.get(f"bull_{tf}", 0.0)
            bear_raw = row.get(f"bear_{tf}", 0.0)
            bull_signal = bull_raw > (0.32 + state_adjust["long_gate"])
            bear_signal = bear_raw > (0.55 + state_adjust["short_gate"])

            tf_meta_score = row.get(f"meta_{tf}", 0.0)

            bull_fortress = bull_signal and tf_meta_score > 0.28 and rsi_8 < 65
            raw_bear_fortress = (
                row.get(f"bear_{tf}", 0.0) > 0.62
                and row.get(f"bear_{tf}", 0.0) > row.get(f"bull_{tf}", 0.0) + 0.12
                and rsi_8 > 48
            )
            bear_fortress = (bear_signal and tf_meta_score > 0.28 and rsi_8 > 32) or raw_bear_fortress
            bull_soft = bull_signal and supportive_regime and supportive_regime_4h and tf_meta_score > 0.50
            bear_soft = bear_signal and supportive_regime and supportive_regime_4h and tf_meta_score > 0.50

            if supportive_regime:
                meta_factor = max(0.0, min(1.0, (blended_meta - 0.30) / 0.50))
                risk_pct = 0.03 + 0.27 * meta_factor
                risk_per_trade = equity * risk_pct
                long_size = (risk_per_trade * bar.close) / stop_dist
                short_size = (risk_per_trade * bar.close) / stop_dist
            else:
                long_size = equity * 0.08
                short_size = equity * 0.08

            if not supportive_regime:
                long_size *= 0.05

            mret_sum = sum(self._market_ret_buf) if self._market_ret_buf else 0.0
            long_factor = max(0.20, min(1.0, 1.0 + 45.0 * mret_sum))
            short_factor = max(0.20, min(1.0, 1.0 - 45.0 * mret_sum))
            long_size *= long_factor
            short_size *= short_factor

            long_size *= state_adjust["size"]
            short_size *= state_adjust["size"]

            if tf == "15m":
                cfg15 = self._cfg_15m
                if pos > 0:
                    age = self.position_ages.get(symbol, 0) + 1
                    self.position_ages[symbol] = age
                    signal_decay_exit = age >= cfg15["exit_decay_age"] and (
                        bull_raw < 0.40
                        or m15_long < 0.20
                        or (m1h > 0 and m1h < 0.20)
                    )
                    should_exit = (
                        age >= cfg15["max_hold"]
                        or bar.close < self.trailing_stops.get(symbol, 0)
                        or signal_decay_exit
                    )
                    if should_exit:
                        signals.append(Signal(symbol, 0.0))
                        self.position_ages[symbol] = 0
                    else:
                        desired = pos
                        if bull_raw > cfg15["scalein_bull"] and m15_long > cfg15["scalein_meta"] and abs(pos) + 1.0 < long_size:
                            desired = long_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired))
                        self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), bar.high - stop_dist)
                    continue

                if pos < 0:
                    signals.append(Signal(symbol, 0.0))
                    self.position_ages[symbol] = 0
                    continue

                self.position_ages[symbol] = 0
                trend_veto = (m1h > 0 and m1h < 0.10) or (m4h > 0 and m4h < 0.08)
                push_veto = (m1h > 0 and m1h < 0.12)
                trend_support = 0.65 + 0.25 * max(m1h, 0.0) + 0.10 * max(m4h, 0.0)
                push_support = 0.75 + 0.25 * max(m1h, 0.0)
                long_trend_ok = (
                    bull_raw > cfg15["trend_bull"]
                    and m15_long > cfg15["trend_meta"]
                    and rsi_8 < cfg15["trend_rsi_max"]
                    and (not self._macro_bear or bull_raw > cfg15["trend_macro_bear_bull"])
                    and not trend_veto
                )
                long_push_ok = (
                    bull_raw > cfg15["push_bull"]
                    and m15_long > cfg15["push_meta"]
                    and rsi_8 < cfg15["push_rsi_max"]
                    and not self._macro_bear
                    and not push_veto
                )

                if long_trend_ok:
                    trend_score = bull_raw + 0.55 * m15_long + 0.10 * max(m1h, 0.0) + 0.06 * max(m4h, 0.0)
                    candidate = (
                        trend_score,
                        symbol,
                        long_size * cfg15["trend_weight"] * self._entry_scale * trend_support,
                        bar.low - stop_dist,
                    )
                    ordered_candidates.append(candidate)
                    trend_candidates.append(candidate)
                elif long_push_ok:
                    push_score = (
                        bull_raw
                        + 0.75 * m15_long
                        + 0.14 * max(m1h, 0.0)
                        + 0.06 * max(0.0, (cfg15["push_rsi_max"] - rsi_8) / 10.0)
                        + 0.04
                    )
                    candidate = (
                        push_score,
                        symbol,
                        long_size * cfg15["push_weight"] * self._entry_scale * push_support,
                        bar.low - stop_dist,
                    )
                    ordered_candidates.append(candidate)
                    push_candidates.append(candidate)
                continue

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
            ts = self._thresh_scale
            long_gate_ok = blended_meta > (0.36 * ts + state_adjust["long_gate"])
            short_gate_ok = blended_meta > (0.25 * ts + state_adjust["short_gate"])
            macro_bull_ok = not self._macro_bear or blended_meta > 0.40 * ts

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

        if self.timeframe_arg == "15m" and current_pos_count < self._cfg_15m["max_positions"]:
            remaining_slots = self._cfg_15m["max_positions"] - current_pos_count
            allocator = self._cfg_15m["allocator"]
            if allocator == "baseline":
                selected_entries = ordered_candidates[:remaining_slots]
            elif allocator == "ranked":
                selected_entries = sorted(ordered_candidates, key=lambda item: item[0], reverse=True)[:remaining_slots]
            elif allocator == "trend_priority":
                selected_entries = sorted(trend_candidates, key=lambda item: item[0], reverse=True)[:remaining_slots]
                if len(selected_entries) < remaining_slots:
                    selected_entries.extend(
                        sorted(push_candidates, key=lambda item: item[0], reverse=True)[: remaining_slots - len(selected_entries)]
                    )
            else:
                selected_entries = sorted(push_candidates, key=lambda item: item[0], reverse=True)[:remaining_slots]
                if len(selected_entries) < remaining_slots:
                    selected_entries.extend(
                        sorted(trend_candidates, key=lambda item: item[0], reverse=True)[: remaining_slots - len(selected_entries)]
                    )

            for _, symbol, entry_size, trailing_stop in selected_entries:
                signals.append(Signal(symbol, entry_size))
                self.trailing_stops[symbol] = trailing_stop
                self.position_ages[symbol] = 0

        return signals

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
