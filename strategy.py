import os
from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from market_regime import build_regime_frame
from prepare import FEATURE_COLS, calculate_features, load_data


DEFAULT_MODEL_SET = os.environ.get("AUTOTRADER_MODEL_SET", "exp256_active")


@dataclass
class Signal:
    symbol: str
    target_position: float
    order_type: str = "market"
    tag: str = ""
    metadata: dict | None = None


class Strategy:
    def __init__(self, timeframe: str = "1h"):
        self.timeframe_arg = timeframe
        self.interval_sec = self._parse_timeframe(timeframe)

        # Timeframe-derived constants (all calibrated at 1h, scaled automatically)
        bars_per_hour = 3600 / self.interval_sec
        tf_ratio = self.interval_sec / 3600
        self._regime_window = max(18, int(72 * bars_per_hour))
        self._max_hold = max(2, round(3 * tf_ratio ** 0.5))
        self._decay_age = max(2, round(3 * tf_ratio ** 0.5))
        self._atr_scale = tf_ratio ** 0.25
        self._entry_scale = min(1.0, tf_ratio ** 0.5)
        self._thresh_scale = max(1.0, (1.0 / tf_ratio) ** 0.25)
        self._leverage_mult = 1.0  # set externally (e.g. backtest.py --leverage)

        self.models = {}
        self.meta_models = {}
        self.long_models = {}
        self.short_models = {}
        self.long_meta_models = {}
        self.short_meta_models = {}
        self.bear_short_meta_models = {}
        self.symbol_caches = {}
        self.bar_counts = {}
        self.models_loaded = False
        self.trailing_stops = {}
        self.position_ages = {}
        self._market_ret_buf = []
        self._macro_bear = False
        self.regime_family_by_ts = {}
        self.model_metadata = {}
        self.feature_profile = "price_only"
        self.short_conf_mode = os.environ.get("AUTOTRADER_SHORT_CONF_MODE")
        self.short_conf_calibrator = None
        self.bear_short_blend = float(os.environ.get("AUTOTRADER_BEAR_SHORT_BLEND", "1.0"))
        self.bear_short_decay_blend = float(
            os.environ.get("AUTOTRADER_BEAR_SHORT_DECAY_BLEND", str(self.bear_short_blend))
        )
        self.bear_short_delta_cap = float(os.environ.get("AUTOTRADER_BEAR_SHORT_DELTA_CAP", "1.0"))
        self.bear_short_negative_scale = float(os.environ.get("AUTOTRADER_BEAR_SHORT_NEGATIVE_SCALE", "1.0"))
        self.bear_short_size_bonus = float(os.environ.get("AUTOTRADER_BEAR_SHORT_SIZE_BONUS", "0.0"))
        self.bear_short_size_edge = float(os.environ.get("AUTOTRADER_BEAR_SHORT_SIZE_EDGE", "0.0"))
        self.sideways_structure_boost = float(os.environ.get("AUTOTRADER_SIDEWAYS_STRUCTURE_BOOST", "1.10"))
        self.sideways_structure_boost_min = int(float(os.environ.get("AUTOTRADER_SIDEWAYS_STRUCTURE_BOOST_MIN", "1")))
        self.sideways_structure_boost_max = int(float(os.environ.get("AUTOTRADER_SIDEWAYS_STRUCTURE_BOOST_MAX", "1")))
        self.sideways_structure_penalty = float(os.environ.get("AUTOTRADER_SIDEWAYS_STRUCTURE_PENALTY", "1.0"))
        self.sideways_structure_penalty_min = int(float(os.environ.get("AUTOTRADER_SIDEWAYS_STRUCTURE_PENALTY_MIN", "999")))
        self.sideways_structure_penalty_max = int(float(os.environ.get("AUTOTRADER_SIDEWAYS_STRUCTURE_PENALTY_MAX", "999")))
        self.sideways_conf_penalty = float(os.environ.get("AUTOTRADER_SIDEWAYS_CONF_PENALTY", "0.95"))
        self.sideways_conf_penalty_min = float(os.environ.get("AUTOTRADER_SIDEWAYS_CONF_PENALTY_MIN", "0.29"))
        self.sideways_conf_penalty_max = float(os.environ.get("AUTOTRADER_SIDEWAYS_CONF_PENALTY_MAX", "0.33"))

    def _parse_timeframe(self, tf: str) -> int:
        if tf == "15m":
            return 15 * 60
        if tf == "1h":
            return 3600
        if tf == "4h":
            return 14400
        return 0

    def _resolve_model_path(self, *filenames: str) -> str | None:
        search_dirs = []
        if DEFAULT_MODEL_SET:
            search_dirs.append(os.path.join("models", DEFAULT_MODEL_SET))
        search_dirs.append("models")

        seen = set()
        for directory in search_dirs:
            if directory in seen:
                continue
            seen.add(directory)
            for filename in filenames:
                path = os.path.join(directory, filename)
                if os.path.exists(path):
                    return path
        return None

    def _load_model_metadata(self) -> dict:
        metadata_path = self._resolve_model_path("metadata.json")
        if not metadata_path:
            return {}
        try:
            return pd.read_json(metadata_path, typ="series").to_dict()
        except Exception:
            try:
                import json
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return {}

    def _resolve_feature_frame(self, model, df_feat: pd.DataFrame) -> pd.DataFrame:
        booster = model.get_booster()
        feature_names = booster.feature_names or FEATURE_COLS
        return df_feat.reindex(columns=feature_names, fill_value=0.0)

    def _load_short_calibrator(self):
        if self.short_conf_mode not in {"platt", "isotonic"}:
            return None
        short_meta = self.model_metadata.get("short_confidence", {}) if isinstance(self.model_metadata, dict) else {}
        artifact = short_meta.get("calibrator_artifact")
        if not artifact:
            artifact = f"short_conf_15m_{self.short_conf_mode}.joblib"
        path = self._resolve_model_path(artifact)
        if not path:
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    def _apply_short_calibration(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if self.short_conf_mode == "raw" or self.short_conf_calibrator is None:
            return probs
        if self.short_conf_mode == "platt":
            return self.short_conf_calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        if self.short_conf_mode == "isotonic":
            return np.asarray(self.short_conf_calibrator.predict(probs), dtype=float)
        return probs

    def _apply_cross_sectional_short_ranks(self, tables):
        if not tables:
            return tables
        merged = []
        for symbol, table in tables.items():
            if "short_conf_raw_15m" not in table.columns:
                continue
            subset = table[["timestamp", "short_conf_raw_15m"]].copy()
            subset["symbol"] = symbol
            merged.append(subset)
        if not merged:
            return tables
        rank_df = pd.concat(merged, ignore_index=True)
        rank_df["short_conf_rank_15m"] = rank_df.groupby("timestamp")["short_conf_raw_15m"].rank(pct=True, method="average")
        rank_map = {
            (row.symbol, int(row.timestamp)): float(row.short_conf_rank_15m)
            for row in rank_df.itertuples(index=False)
        }
        for symbol, table in tables.items():
            if "short_conf_raw_15m" not in table.columns:
                continue
            table["short_conf_rank_15m"] = [
                rank_map.get((symbol, int(ts)), 0.0) for ts in table["timestamp"].values
            ]
            table["short_conf_15m"] = table["short_conf_rank_15m"]
        return tables

    def _blend_bear_short_conf(self, raw_value: float, bear_value: float, blend: float) -> float:
        blend = max(0.0, min(1.0, float(blend)))
        raw_value = float(raw_value)
        bear_value = float(bear_value)
        delta = bear_value - raw_value
        if delta < 0:
            neg_scale = max(0.0, min(1.0, float(self.bear_short_negative_scale)))
            bear_value = raw_value + delta * neg_scale
        blended = (1.0 - blend) * raw_value + blend * bear_value
        cap = max(0.0, float(self.bear_short_delta_cap))
        if cap < 1.0:
            blended = min(max(blended, raw_value - cap), raw_value + cap)
        return float(blended)

    def _load_models(self):
        self.model_metadata = self._load_model_metadata()
        short_conf_meta = self.model_metadata.get("short_confidence", {}) if isinstance(self.model_metadata, dict) else {}
        if not self.short_conf_mode:
            self.short_conf_mode = short_conf_meta.get("mode", "raw")
        self.feature_profile = self.model_metadata.get("feature_profile", "price_only") if isinstance(self.model_metadata, dict) else "price_only"
        loaded_paths = {}
        for tf in ["15m", "1h", "4h"]:
            path_map = {
                "lead": self._resolve_model_path(f"lead_{tf}.xgb", f"lead_{tf}.json"),
                "meta": self._resolve_model_path(f"meta_{tf}.xgb", f"meta_{tf}.json"),
                "long": self._resolve_model_path(f"lead_long_{tf}.xgb", f"lead_long_{tf}.json"),
                "short": self._resolve_model_path(f"lead_short_{tf}.xgb", f"lead_short_{tf}.json"),
                "long_meta": self._resolve_model_path(f"meta_long_{tf}.xgb", f"meta_long_{tf}.json"),
                "short_meta": self._resolve_model_path(f"meta_short_{tf}.xgb", f"meta_short_{tf}.json"),
                "bear_short_meta": self._resolve_model_path(f"meta_short_bear_{tf}.xgb", f"meta_short_bear_{tf}.json"),
            }

            if path_map["lead"]:
                self.models[tf] = xgb.XGBClassifier()
                self.models[tf].load_model(path_map["lead"])
                loaded_paths[f"{tf}:lead"] = path_map["lead"]

            if path_map["meta"]:
                self.meta_models[tf] = xgb.XGBClassifier()
                self.meta_models[tf].load_model(path_map["meta"])
                loaded_paths[f"{tf}:meta"] = path_map["meta"]

            if path_map["long"]:
                self.long_models[tf] = xgb.XGBClassifier()
                self.long_models[tf].load_model(path_map["long"])
                loaded_paths[f"{tf}:long"] = path_map["long"]

            if path_map["short"]:
                self.short_models[tf] = xgb.XGBClassifier()
                self.short_models[tf].load_model(path_map["short"])
                loaded_paths[f"{tf}:short"] = path_map["short"]

            if path_map["long_meta"]:
                self.long_meta_models[tf] = xgb.XGBClassifier()
                self.long_meta_models[tf].load_model(path_map["long_meta"])
                loaded_paths[f"{tf}:long_meta"] = path_map["long_meta"]

            if path_map["short_meta"]:
                self.short_meta_models[tf] = xgb.XGBClassifier()
                self.short_meta_models[tf].load_model(path_map["short_meta"])
                loaded_paths[f"{tf}:short_meta"] = path_map["short_meta"]

            if path_map["bear_short_meta"]:
                self.bear_short_meta_models[tf] = xgb.XGBClassifier()
                self.bear_short_meta_models[tf].load_model(path_map["bear_short_meta"])
                loaded_paths[f"{tf}:bear_short_meta"] = path_map["bear_short_meta"]

        self.short_conf_calibrator = self._load_short_calibrator()
        self.models_loaded = True
        print(f"LOADED MODEL SET: {DEFAULT_MODEL_SET}")
        for key in sorted(loaded_paths):
            print(f"  {key:<14} {loaded_paths[key]}")
        print(f"  short_conf_mode {self.short_conf_mode}")

    def _build_prediction_tables(self, data_dict, timeframe: str, include_state: bool = False):
        if not data_dict:
            return {}

        all_vols = {}
        all_rets = {}
        for symbol, df in data_dict.items():
            feat = calculate_features(df, timeframe=timeframe, symbol=symbol, feature_profile="price_context")
            all_vols[symbol] = feat["bb_width"]
            all_rets[symbol] = df["close"].pct_change()

        m_vol = pd.DataFrame(all_vols).median(axis=1).fillna(0)
        m_ret = pd.DataFrame(all_rets).median(axis=1).fillna(0)

        tables = {}
        for symbol, df in data_dict.items():
            df_feat = calculate_features(df, timeframe=timeframe, symbol=symbol, feature_profile="price_context")
            df_feat["market_vol"] = m_vol
            df_feat["market_ret"] = m_ret
            market_ret_4h = m_ret.rolling(4, min_periods=1).sum().fillna(0.0)
            df_feat["rel_ret_1h"] = df_feat["ret_1h"] - df_feat["market_ret"]
            df_feat["rel_ret_4h"] = df_feat["ret_4h"] - market_ret_4h
            df_feat["rel_bb_width"] = df_feat["bb_width"] - df_feat["market_vol"]

            use_directional = (
                timeframe == "15m" and timeframe in self.long_models and timeframe in self.short_models
            )

            if use_directional:
                bull = self.long_models[timeframe].predict_proba(
                    self._resolve_feature_frame(self.long_models[timeframe], df_feat)
                )[:, 1]
                bear = self.short_models[timeframe].predict_proba(
                    self._resolve_feature_frame(self.short_models[timeframe], df_feat)
                )[:, 1]
                meta_long = (
                    self.long_meta_models[timeframe].predict_proba(
                        self._resolve_feature_frame(self.long_meta_models[timeframe], df_feat)
                    )[:, 1]
                    if timeframe in self.long_meta_models
                    else np.zeros(len(df_feat))
                )
                meta_short_raw = (
                    self.short_meta_models[timeframe].predict_proba(
                        self._resolve_feature_frame(self.short_meta_models[timeframe], df_feat)
                    )[:, 1]
                    if timeframe in self.short_meta_models
                    else np.zeros(len(df_feat))
                )
                meta_short = self._apply_short_calibration(meta_short_raw)
                bear_meta_short = (
                    self.bear_short_meta_models[timeframe].predict_proba(
                        self._resolve_feature_frame(self.bear_short_meta_models[timeframe], df_feat)
                    )[:, 1]
                    if timeframe in self.bear_short_meta_models
                    else np.zeros(len(df_feat))
                )
                table = pd.DataFrame(
                    {
                        "timestamp": df["timestamp"].values,
                        f"bull_{timeframe}": bull,
                        f"bear_{timeframe}": bear,
                        f"meta_{timeframe}": np.maximum(meta_long, meta_short),
                        f"meta_long_{timeframe}": meta_long,
                        f"meta_short_{timeframe}": meta_short_raw,
                        f"short_conf_raw_{timeframe}": meta_short_raw,
                        f"short_conf_{timeframe}": meta_short,
                        f"short_conf_bear_{timeframe}": bear_meta_short,
                    }
                )
            else:
                probs = (
                    self.models[timeframe].predict_proba(self._resolve_feature_frame(self.models[timeframe], df_feat))
                    if timeframe in self.models
                    else None
                )
                meta = (
                    self.meta_models[timeframe].predict_proba(
                        self._resolve_feature_frame(self.meta_models[timeframe], df_feat)
                    )[:, 1]
                    if timeframe in self.meta_models
                    else np.zeros(len(df_feat))
                )
                table = pd.DataFrame(
                    {
                        "timestamp": df["timestamp"].values,
                        f"bull_{timeframe}": (
                            probs[:, 1] if probs is not None and probs.shape[1] > 1 else np.zeros(len(df_feat))
                        ),
                        f"bear_{timeframe}": (
                            probs[:, 2] if probs is not None and probs.shape[1] > 2 else np.zeros(len(df_feat))
                        ),
                        f"meta_{timeframe}": meta,
                    }
                )

            if include_state:
                table["atr_val"] = df_feat["atr_14"].values
                table["market_ret"] = df_feat["market_ret"].values
                table["rsi_8"] = df_feat["rsi_8"].values
                table["rsi_24"] = df_feat["rsi_24"].values
                table["atr_pct"] = df_feat["atr_pct"].values
                table["liquidity_sweep"] = df_feat["liquidity_sweep"].values
                table["msb_status"] = df_feat["msb_status"].values
                table["fvg_detected"] = df_feat["fvg_detected"].values
                table["ob_dist"] = df_feat["ob_dist"].values
                table["ema_200_dist"] = df_feat["ema_200_dist"].values
                table["dist_to_vwap"] = df_feat["dist_to_vwap"].values
                table["macro_event_flag"] = df_feat["macro_event_flag"].values
                table["context_sentiment"] = df_feat["context_sentiment"].values
                table["major_market_event_flag"] = df_feat["major_market_event_flag"].values

            tables[symbol] = table.sort_values("timestamp").reset_index(drop=True)

        if timeframe == "15m" and self.short_conf_mode == "rank":
            tables = self._apply_cross_sectional_short_ranks(tables)

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
        regime_frame = build_regime_frame(data_dict)
        self.regime_family_by_ts = (
            dict(regime_frame[["timestamp", "regime_family"]].itertuples(index=False, name=None))
            if not regime_frame.empty
            else {}
        )

        if self.timeframe_arg == "1h":
            if "15m" in self.models or "15m" in self.meta_models:
                split_15m = {
                    "train":   "train_15m",
                    "val":     "val_15m",
                    "oos":     "oos_15m",
                    "2026q1":  "2026q1_15m",
                    "holdout": "holdout_15m",
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

            base = main_df[
                [
                    "timestamp",
                    "atr_val",
                    "market_ret",
                    "rsi_8",
                    "rsi_24",
                    "atr_pct",
                    "liquidity_sweep",
                    "msb_status",
                    "fvg_detected",
                    "ob_dist",
                    "ema_200_dist",
                    "dist_to_vwap",
                    "macro_event_flag",
                    "context_sentiment",
                    "major_market_event_flag",
                ]
            ].copy()
            base["bull_15m"] = main_df[main_bull_col].fillna(0).values if main_bull_col in main_df else 0.0
            base["bear_15m"] = main_df[main_bear_col].fillna(0).values if main_bear_col in main_df else 0.0
            base["meta_15m"] = (
                main_df[main_meta_col].fillna(0).values
                if self.timeframe_arg == "15m" and main_meta_col in main_df
                else 0.0
            )
            base["meta_long_15m"] = (
                main_df["meta_long_15m"].fillna(0).values
                if self.timeframe_arg == "15m" and "meta_long_15m" in main_df
                else base["meta_15m"].copy()
            )
            base["meta_short_15m"] = (
                main_df["meta_short_15m"].fillna(0).values
                if self.timeframe_arg == "15m" and "meta_short_15m" in main_df
                else base["meta_15m"].copy()
            )
            base["short_conf_15m"] = (
                main_df["short_conf_15m"].fillna(0).values
                if self.timeframe_arg == "15m" and "short_conf_15m" in main_df
                else base["meta_short_15m"].copy()
            )
            base["short_conf_raw_15m"] = (
                main_df["short_conf_raw_15m"].fillna(0).values
                if self.timeframe_arg == "15m" and "short_conf_raw_15m" in main_df
                else base["meta_short_15m"].copy()
            )
            base["short_conf_bear_15m"] = (
                main_df["short_conf_bear_15m"].fillna(0).values
                if self.timeframe_arg == "15m" and "short_conf_bear_15m" in main_df
                else 0.0
            )
            base["meta_1h"] = (
                main_df[main_meta_col].fillna(0).values
                if self.timeframe_arg == "1h" and main_meta_col in main_df
                else 0.0
            )
            base["bull_1h"] = (
                main_df[main_bull_col].fillna(0).values
                if self.timeframe_arg == "1h" and main_bull_col in main_df
                else 0.0
            )
            base["bear_1h"] = (
                main_df[main_bear_col].fillna(0).values
                if self.timeframe_arg == "1h" and main_bear_col in main_df
                else 0.0
            )
            base["meta_4h"] = (
                main_df[main_meta_col].fillna(0).values
                if self.timeframe_arg == "4h" and main_meta_col in main_df
                else 0.0
            )

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
                if "short_conf_15m" in merged_15m:
                    base["short_conf_15m"] = merged_15m["short_conf_15m"].fillna(base["short_conf_15m"])
                if "short_conf_raw_15m" in merged_15m:
                    base["short_conf_raw_15m"] = merged_15m["short_conf_raw_15m"].fillna(base["short_conf_raw_15m"])
                if "short_conf_bear_15m" in merged_15m:
                    base["short_conf_bear_15m"] = merged_15m["short_conf_bear_15m"].fillna(base["short_conf_bear_15m"])

            if symbol in aux_1h_tables:
                aux_1h_cols = ["timestamp", "meta_1h"]
                if "bull_1h" in aux_1h_tables[symbol].columns:
                    aux_1h_cols.extend(["bull_1h", "bear_1h"])
                merged_1h = pd.merge_asof(
                    base[["timestamp"]].sort_values("timestamp"),
                    aux_1h_tables[symbol][aux_1h_cols].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                base["meta_1h"] = merged_1h["meta_1h"].fillna(base["meta_1h"])
                if "bull_1h" in merged_1h.columns:
                    base["bull_1h"] = merged_1h["bull_1h"].fillna(base.get("bull_1h", 0.0))
                    base["bear_1h"] = merged_1h["bear_1h"].fillna(base.get("bear_1h", 0.0))

            if symbol in aux_4h_tables:
                merged_4h = pd.merge_asof(
                    base[["timestamp"]].sort_values("timestamp"),
                    aux_4h_tables[symbol][["timestamp", "meta_4h"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                base["meta_4h"] = merged_4h["meta_4h"].fillna(base["meta_4h"])

            self.symbol_caches[symbol] = base.fillna(0.0).to_dict("records")
            self.bar_counts[symbol] = 0

        print(f"Market-Aware Fortress cache warmed for {len(self.symbol_caches)} symbols.")

    def on_bar(self, bar_data, portfolio):
        if not self.models_loaded:
            self._load_models()

        equity = portfolio.equity
        signals: List[Signal] = []

        any_sym = next(iter(bar_data), None)
        if any_sym and any_sym in self.symbol_caches:
            idx = self.bar_counts.get(any_sym, 0)
            if idx < len(self.symbol_caches[any_sym]):
                mret = self.symbol_caches[any_sym][idx].get("market_ret", 0.0)
                self._market_ret_buf.append(mret)
                if len(self._market_ret_buf) > self._regime_window:
                    self._market_ret_buf.pop(0)
                self._macro_bear = sum(self._market_ret_buf) < -0.015

        for symbol, bar in bar_data.items():
            pos = portfolio.positions.get(symbol, 0.0)

            if symbol not in self.symbol_caches:
                continue

            cache = self.symbol_caches[symbol]
            idx = self.bar_counts.get(symbol, 0)
            if idx >= len(cache):
                continue

            row = cache[idx]
            self.bar_counts[symbol] += 1

            m15 = row["meta_15m"]
            rsi_8 = row.get("rsi_8", 50.0)
            market_regime_family = self.regime_family_by_ts.get(int(row["timestamp"]), "unknown")
            m15_long = row.get("meta_long_15m", m15)
            m15_short_raw = row.get("meta_short_15m", m15)
            m15_short = row.get("short_conf_15m", m15_short_raw)
            m15_short_decay = m15_short
            bear_short_model = row.get("short_conf_bear_15m", m15_short)
            if self.short_conf_mode in {"bear_model", "bear_overlay"} and market_regime_family == "bear":
                m15_short = self._blend_bear_short_conf(m15_short_raw, bear_short_model, self.bear_short_blend)
                m15_short_decay = self._blend_bear_short_conf(
                    m15_short_raw, bear_short_model, self.bear_short_decay_blend
                )
            short_gate_conf = m15_short if self.short_conf_mode in {"raw", "bear_overlay"} else m15_short_raw
            bear_calibrated_gate = m15 if self.short_conf_mode in {"raw", "bear_overlay"} else m15_short_raw
            m1h = row["meta_1h"]
            m4h = row["meta_4h"]
            macro_event_flag = row.get("macro_event_flag", 0.0)
            context_sentiment = row.get("context_sentiment", 0.0)
            major_market_event_flag = row.get("major_market_event_flag", 0.0)

            active_meta = [m for m in (max(m15_long, m15_short), m1h, m4h) if m > 0]
            meta_score = float(np.mean(active_meta)) if active_meta else 0.0
            supportive_regime = (m1h <= 0 or m1h > 0.45)
            supportive_regime_4h = (m4h <= 0 or m4h > 0.45)
            atr_mult = (4.0 if self._macro_bear else 7.0) * self._atr_scale
            stop_dist = atr_mult * row["atr_val"] if row["atr_val"] > 0 else max(bar.close * 0.02, 1e-6)
            mret_bar = row.get("market_ret", 0.0)
            mret_sum = sum(self._market_ret_buf) if self._market_ret_buf else 0.0
            short_regime_ok = self._macro_bear or mret_sum < 0.01 or mret_bar < 0.001
            bull_signal = row["bull_15m"] > 0.32
            bear_signal = row["bear_15m"] > 0.36
            # bull_1h / bear_1h available but too noisy from 3-class model (see exp462-469)
            bull_fortress = bull_signal and m15_long > 0.28 and rsi_8 < 65
            raw_bear_fortress = (
                row["bear_15m"] > 0.58
                and row["bear_15m"] > row["bull_15m"] + 0.08
                and rsi_8 > 46
            )
            bull_soft = bull_signal and supportive_regime and supportive_regime_4h and m15_long > 0.50
            bull_conviction = row["bull_15m"] + 0.70 * m15_long
            bear_conviction = row["bear_15m"] + 0.70 * m15_short + (0.05 if self._macro_bear else 0.0)
            prefer_short = bear_conviction > bull_conviction + (0.02 if self._macro_bear else 0.08)
            prefer_long = bull_conviction > bear_conviction + 0.05
            bear_calibrated = market_regime_family == "bear" and bear_signal and bear_calibrated_gate > 0.24 and rsi_8 > 35 and prefer_short
            bear_soft = (
                short_regime_ok
                and bear_signal
                and supportive_regime
                and supportive_regime_4h
                and short_gate_conf > 0.42
                and row["bear_15m"] > row["bull_15m"] + 0.02
            )
            bear_fortress = short_regime_ok and ((bear_signal and short_gate_conf > 0.24 and rsi_8 > 35) or raw_bear_fortress)
            sideways_structure_score = 0
            liquidity_sweep = row.get("liquidity_sweep", 0.0)
            msb_status = row.get("msb_status", 0.0)
            fvg_detected = row.get("fvg_detected", 0.0)
            ema_200_dist = row.get("ema_200_dist", 0.0)
            dist_to_vwap = row.get("dist_to_vwap", 0.0)
            if liquidity_sweep > 0:
                sideways_structure_score += 1
            elif liquidity_sweep < 0:
                sideways_structure_score -= 1
            if msb_status > 0:
                sideways_structure_score += 1
            elif msb_status < 0:
                sideways_structure_score -= 1
            if fvg_detected > 0:
                sideways_structure_score += 1
            elif fvg_detected < 0:
                sideways_structure_score -= 1
            if ema_200_dist > 0:
                sideways_structure_score += 1
            else:
                sideways_structure_score -= 1
            if dist_to_vwap <= 0:
                sideways_structure_score += 1
            else:
                sideways_structure_score -= 1
            structure_boost_ok = (
                self.sideways_structure_boost > 1.0
                and self.sideways_structure_boost_min <= sideways_structure_score <= self.sideways_structure_boost_max
            )
            structure_penalty_ok = (
                self.sideways_structure_penalty < 1.0
                and self.sideways_structure_penalty_min <= sideways_structure_score <= self.sideways_structure_penalty_max
            )
            conf_penalty_ok = (
                self.sideways_conf_penalty < 1.0
                and self.sideways_conf_penalty_min <= m15_long <= self.sideways_conf_penalty_max
            )
            sideways_bull_fortress_scale = (
                self.sideways_structure_boost
                if (market_regime_family == "sideways" and bull_fortress and structure_boost_ok)
                else 1.0
            )
            if market_regime_family == "sideways" and bull_fortress and structure_penalty_ok:
                sideways_bull_fortress_scale *= self.sideways_structure_penalty
            if market_regime_family == "sideways" and bull_fortress and conf_penalty_ok:
                sideways_bull_fortress_scale *= self.sideways_conf_penalty

            def signal_metadata(tag: str, size_reason: str = "", exit_reason: str = "") -> dict:
                return {
                    "signal_tag": tag,
                    "signal_regime_family": market_regime_family,
                    "meta_score": float(meta_score),
                    "m15_long": float(m15_long),
                    "m15_short_raw": float(m15_short_raw),
                    "short_conf_15m": float(m15_short),
                    "short_conf_mode": self.short_conf_mode,
                    "structure_score": int(sideways_structure_score),
                    "macro_event_flag": float(macro_event_flag),
                    "context_sentiment": float(context_sentiment),
                    "major_market_event_flag": float(major_market_event_flag),
                    "size_reason": size_reason,
                    "exit_reason": exit_reason,
                }

            if supportive_regime:
                meta_factor = max(0.0, min(1.0, (meta_score - 0.25) / 0.60))
                risk_pct = 0.03 + 0.23 * meta_factor
                risk_per_trade = equity * risk_pct
                long_size = (risk_per_trade * bar.close) / stop_dist
                short_size = (risk_per_trade * bar.close) / stop_dist
            else:
                long_size = equity * 0.08
                short_size = equity * 0.08
            # exp366 (KEPT): cap per-position notional at 1.0x equity.
            # Risk-parity sizing can imply 3x+ leverage per trade; the 1.0x cap
            # shaves the blowup tail (OOS Sh +0.04@1bp, +0.058@5bp, DD -1pt)
            # without touching trade count or WR. exp367 (0.60x) was tested and
            # found non-monotonic — tighter than 1.0x hurts Sharpe.
            long_size = min(long_size, equity)
            short_size = min(short_size, equity)

            # exp371-373 KEPT. Non-supp shrink trajectory:
            # 0.05->0.025 (+0.06 OOS), 0.025->0.0125 (+0.011), 0.0125->0.005 (+0.006).
            # exp374: 0.005 -> 0.001 (5x) — near-full gate, search for inflection.
            if not supportive_regime:
                long_size *= 0.001
            if market_regime_family == "bull":
                long_size *= 1.25
            if market_regime_family == "sideways":
                long_size *= 0.45

            long_factor = max(0.05, min(1.0, 1.0 + 8.0 * mret_sum))
            short_factor = max(0.05, min(1.0, 1.0 - 8.0 * mret_sum))
            long_size *= long_factor
            short_size *= short_factor
            if prefer_short:
                short_size *= 1.10
            elif prefer_long:
                long_size *= 1.05
            if (
                self.short_conf_mode in {"bear_model", "bear_overlay"}
                and market_regime_family == "bear"
                and self.bear_short_size_bonus > 0.0
                and bear_short_model > m15_short_raw + self.bear_short_size_edge
                and prefer_short
            ):
                short_size *= 1.0 + self.bear_short_size_bonus
            short_size = min(short_size, equity)

            if pos != 0:
                age = self.position_ages.get(symbol, 0) + 1
                self.position_ages[symbol] = age
                max_hold_bars = self._max_hold

                if pos > 0:
                    long_decay_age = self._decay_age
                    if market_regime_family == "sideways":
                        long_decay_age = max(2, self._decay_age - 1)
                    signal_decay_exit = age >= long_decay_age and (not bull_signal) and m15_long < 0.40
                    should_exit = (
                        age >= max_hold_bars
                        or bar.close < self.trailing_stops.get(symbol, 0)
                        or signal_decay_exit
                    )
                    if should_exit:
                        if age >= max_hold_bars:
                            exit_tag = "close_long_max_hold"
                        elif bar.close < self.trailing_stops.get(symbol, 0):
                            exit_tag = "close_long_stop"
                        else:
                            exit_tag = "close_long_decay"
                        signals.append(Signal(symbol, 0.0, tag=exit_tag, metadata=signal_metadata(exit_tag, exit_reason=exit_tag)))
                        self.position_ages[symbol] = 0
                    else:
                        desired = pos
                        if bull_signal and m15_long > 0.60 and abs(pos) + 1.0 < long_size:
                            desired = long_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired, tag="add_long", metadata=signal_metadata("add_long", size_reason="trend_strength_add")))
                        self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), bar.high - stop_dist)
                else:
                    short_decay_age = max(2, self._decay_age - 1)
                    short_max_hold_bars = max(2, self._max_hold - 1)
                    signal_decay_exit = (
                        age >= short_decay_age
                        and (
                            (not bear_signal)
                            or m15_short_decay < 0.45
                            or row["bear_15m"] < row["bull_15m"] + 0.01
                        )
                        and not raw_bear_fortress
                    )
                    should_exit = (
                        age >= short_max_hold_bars
                        or bar.close > self.trailing_stops.get(symbol, 9e18)
                        or signal_decay_exit
                    )
                    if should_exit:
                        if age >= short_max_hold_bars:
                            exit_tag = "close_short_max_hold"
                        elif bar.close > self.trailing_stops.get(symbol, 9e18):
                            exit_tag = "close_short_stop"
                        else:
                            exit_tag = "close_short_decay"
                        signals.append(Signal(symbol, 0.0, tag=exit_tag, metadata=signal_metadata(exit_tag, exit_reason=exit_tag)))
                        self.position_ages[symbol] = 0
                    else:
                        desired = pos
                        if bear_signal and short_gate_conf > 0.65 and abs(pos) + 1.0 < short_size:
                            desired = -short_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired, tag="add_short", metadata=signal_metadata("add_short", size_reason="short_conviction_add")))
                        self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, 9e18), bar.low + stop_dist)
                continue

            self.position_ages[symbol] = 0
            thresh_scale = self._thresh_scale
            long_gate_ok = meta_score > 0.36 * thresh_scale
            short_gate_ok = meta_score > 0.36 * thresh_scale
            macro_bull_ok = not self._macro_bear or meta_score > 0.40 * thresh_scale
            not_falling_knife = mret_bar > -0.02
            vol_pct = row["atr_val"] / bar.close if bar.close > 0 else 0.0
            not_hyper_vol = vol_pct < 0.06

            vol_scale = max(0.5, min(1.0, 0.015 / max(vol_pct, 1e-6)))
            rsi_scale = max(0.5, 1.0 - max(0.0, rsi_8 - 40) / 40)
            mret_scale = max(0.5, min(1.5, 1.0 + 40.0 * mret_bar))
            funding = bar.funding_rate if hasattr(bar, 'funding_rate') else 0.0
            fund_scale = max(0.5, min(2.5, 1.0 - 3000.0 * funding))
            if bull_fortress and not prefer_short and not_falling_knife and not_hyper_vol and not raw_bear_fortress and (not bear_fortress or m15_long >= m15_short) and macro_bull_ok and long_gate_ok:
                entry_size = long_size * self._entry_scale * vol_scale * rsi_scale * mret_scale * fund_scale * self._leverage_mult
                if market_regime_family == "sideways":
                    entry_size *= sideways_bull_fortress_scale
                signals.append(Signal(symbol, entry_size, tag="entry_bull_fortress", metadata=signal_metadata("entry_bull_fortress", size_reason="bull_fortress_core")))
                self.trailing_stops[symbol] = bar.low - stop_dist
                self.position_ages[symbol] = 0
            elif bull_soft and not prefer_short and (not bear_soft or m15_long >= m15_short) and long_gate_ok:
                entry_size = long_size * 0.5 * self._entry_scale * self._leverage_mult
                signals.append(Signal(symbol, entry_size, tag="entry_bull_soft", metadata=signal_metadata("entry_bull_soft", size_reason="bull_soft_half")))
                self.trailing_stops[symbol] = bar.low - stop_dist
                self.position_ages[symbol] = 0
            elif bear_fortress and (prefer_short or not bull_fortress) and short_gate_ok:
                entry_size = short_size * self._entry_scale * self._leverage_mult
                signals.append(Signal(symbol, -entry_size, tag="entry_bear_fortress", metadata=signal_metadata("entry_bear_fortress", size_reason="bear_fortress_core")))
                self.trailing_stops[symbol] = bar.high + stop_dist
                self.position_ages[symbol] = 0
            elif bear_calibrated and short_gate_ok:
                entry_size = short_size * 0.65 * self._entry_scale * self._leverage_mult
                signals.append(Signal(symbol, -entry_size, tag="entry_bear_calibrated", metadata=signal_metadata("entry_bear_calibrated", size_reason="calibrated_bear_short")))
                self.trailing_stops[symbol] = bar.high + stop_dist
                self.position_ages[symbol] = 0
            elif bear_soft and (prefer_short or not bull_soft) and short_gate_ok:
                entry_size = short_size * 0.5 * self._entry_scale * self._leverage_mult
                signals.append(Signal(symbol, -entry_size, tag="entry_bear_soft", metadata=signal_metadata("entry_bear_soft", size_reason="bear_soft_half")))
                self.trailing_stops[symbol] = bar.high + stop_dist
                self.position_ages[symbol] = 0

        return signals
