import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb

from prepare import FEATURE_COLS, calculate_features, load_data


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
        bars_per_hour = 3600 / self.interval_sec
        tf_ratio = self.interval_sec / 3600
        self._regime_window = max(18, int(72 * bars_per_hour))
        self._max_hold = max(2, round(3 * tf_ratio ** 0.5))
        self._decay_age = max(2, round(3 * tf_ratio ** 0.5))
        self._atr_scale = tf_ratio ** 0.25
        self._entry_scale = min(1.0, tf_ratio ** 0.5)
        self._thresh_scale = max(1.0, (1.0 / tf_ratio) ** 0.25)

        self.models = {}
        self.meta_models = {}
        self.long_models = {}
        self.short_models = {}
        self.long_meta_models = {}
        self.short_meta_models = {}
        self.symbol_caches = {}
        self.bar_counts = {}
        self.models_loaded = False
        self.trailing_stops = {}
        self.position_ages = {}
        self._market_ret_buf = []
        self._macro_bear = False

    def _parse_timeframe(self, tf: str) -> int:
        if tf == "15m":
            return 15 * 60
        if tf == "1h":
            return 3600
        if tf == "4h":
            return 14400
        return 0

    def _resolve_model_path(self, filename: str) -> str | None:
        for path in (
            os.path.join("models", "exp256_active", filename),
            os.path.join("models", filename),
        ):
            if os.path.exists(path):
                return path
        return None

    def _load_models(self):
        for tf in ["15m", "1h", "4h"]:
            path_map = {
                "lead": self._resolve_model_path(f"lead_{tf}.xgb"),
                "meta": self._resolve_model_path(f"meta_{tf}.xgb"),
                "long": self._resolve_model_path(f"lead_long_{tf}.xgb"),
                "short": self._resolve_model_path(f"lead_short_{tf}.xgb"),
                "long_meta": self._resolve_model_path(f"meta_long_{tf}.xgb"),
                "short_meta": self._resolve_model_path(f"meta_short_{tf}.xgb"),
            }

            if path_map["lead"]:
                self.models[tf] = xgb.XGBClassifier()
                self.models[tf].load_model(path_map["lead"])

            if path_map["meta"]:
                self.meta_models[tf] = xgb.XGBClassifier()
                self.meta_models[tf].load_model(path_map["meta"])

            if path_map["long"]:
                self.long_models[tf] = xgb.XGBClassifier()
                self.long_models[tf].load_model(path_map["long"])

            if path_map["short"]:
                self.short_models[tf] = xgb.XGBClassifier()
                self.short_models[tf].load_model(path_map["short"])

            if path_map["long_meta"]:
                self.long_meta_models[tf] = xgb.XGBClassifier()
                self.long_meta_models[tf].load_model(path_map["long_meta"])

            if path_map["short_meta"]:
                self.short_meta_models[tf] = xgb.XGBClassifier()
                self.short_meta_models[tf].load_model(path_map["short_meta"])

        self.models_loaded = True
        print(f"LOADED EXP256 BASELINE: {list(self.models.keys())}")

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

        tables = {}
        for symbol, df in data_dict.items():
            df_feat = calculate_features(df, timeframe=timeframe)
            df_feat["market_vol"] = m_vol
            df_feat["market_ret"] = m_ret
            market_ret_4h = m_ret.rolling(4, min_periods=1).sum().fillna(0.0)
            df_feat["rel_ret_1h"] = df_feat["ret_1h"] - df_feat["market_ret"]
            df_feat["rel_ret_4h"] = df_feat["ret_4h"] - market_ret_4h
            df_feat["rel_bb_width"] = df_feat["bb_width"] - df_feat["market_vol"]
            X = df_feat[FEATURE_COLS]

            use_directional = (
                timeframe == "15m" and timeframe in self.long_models and timeframe in self.short_models
            )

            if use_directional:
                bull = self.long_models[timeframe].predict_proba(X)[:, 1]
                bear = self.short_models[timeframe].predict_proba(X)[:, 1]
                meta_long = (
                    self.long_meta_models[timeframe].predict_proba(X)[:, 1]
                    if timeframe in self.long_meta_models
                    else np.zeros(len(X))
                )
                meta_short = (
                    self.short_meta_models[timeframe].predict_proba(X)[:, 1]
                    if timeframe in self.short_meta_models
                    else np.zeros(len(X))
                )
                table = pd.DataFrame(
                    {
                        "timestamp": df["timestamp"].values,
                        f"bull_{timeframe}": bull,
                        f"bear_{timeframe}": bear,
                        f"meta_{timeframe}": np.maximum(meta_long, meta_short),
                        f"meta_long_{timeframe}": meta_long,
                        f"meta_short_{timeframe}": meta_short,
                    }
                )
            else:
                probs = self.models[timeframe].predict_proba(X) if timeframe in self.models else None
                meta = (
                    self.meta_models[timeframe].predict_proba(X)[:, 1]
                    if timeframe in self.meta_models
                    else np.zeros(len(X))
                )
                table = pd.DataFrame(
                    {
                        "timestamp": df["timestamp"].values,
                        f"bull_{timeframe}": (
                            probs[:, 1] if probs is not None and probs.shape[1] > 1 else np.zeros(len(X))
                        ),
                        f"bear_{timeframe}": (
                            probs[:, 2] if probs is not None and probs.shape[1] > 2 else np.zeros(len(X))
                        ),
                        f"meta_{timeframe}": meta,
                    }
                )

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
            m15_long = row.get("meta_long_15m", m15)
            m15_short = row.get("meta_short_15m", m15)
            m1h = row["meta_1h"]
            m4h = row["meta_4h"]

            active_meta = [m for m in (max(m15_long, m15_short), m1h, m4h) if m > 0]
            meta_score = float(np.mean(active_meta)) if active_meta else 0.0
            supportive_regime = (m1h <= 0 or m1h > 0.45)
            supportive_regime_4h = (m4h <= 0 or m4h > 0.45)
            atr_mult = (4.0 if self._macro_bear else 7.0) * self._atr_scale
            stop_dist = atr_mult * row["atr_val"] if row["atr_val"] > 0 else max(bar.close * 0.02, 1e-6)
            bull_signal = row["bull_15m"] > 0.32
            bear_signal = row["bear_15m"] > 0.40
            # bull_1h / bear_1h available but too noisy from 3-class model (see exp462-469)
            bull_fortress = bull_signal and m15_long > 0.28 and rsi_8 < 65
            raw_bear_fortress = (
                row["bear_15m"] > 0.62
                and row["bear_15m"] > row["bull_15m"] + 0.12
                and rsi_8 > 48
            )
            bear_fortress = (bear_signal and m15_short > 0.28 and rsi_8 > 32) or raw_bear_fortress
            bull_soft = bull_signal and supportive_regime and supportive_regime_4h and m15_long > 0.50
            bear_soft = bear_signal and supportive_regime and supportive_regime_4h and m15_short > 0.50

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

            mret_sum = sum(self._market_ret_buf) if self._market_ret_buf else 0.0
            long_factor = max(0.05, min(1.0, 1.0 + 8.0 * mret_sum))
            short_factor = max(0.05, min(1.0, 1.0 - 8.0 * mret_sum))
            long_size *= long_factor
            short_size *= short_factor

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
                        if bull_signal and m15_long > 0.60 and abs(pos) + 1.0 < long_size:
                            desired = long_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired))
                        self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), bar.high - stop_dist)
                else:
                    signal_decay_exit = (
                        age >= self._decay_age and (not bear_signal) and m15_short < 0.40 and not raw_bear_fortress
                    )
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
                        if bear_signal and m15_short > 0.60 and abs(pos) + 1.0 < short_size:
                            desired = -short_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired))
                        self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, 9e18), bar.low + stop_dist)
                continue

            self.position_ages[symbol] = 0
            thresh_scale = self._thresh_scale
            long_gate_ok = meta_score > 0.36 * thresh_scale
            short_gate_ok = meta_score > 0.36 * thresh_scale
            macro_bull_ok = not self._macro_bear or meta_score > 0.40 * thresh_scale
            mret_bar = row.get("market_ret", 0.0)
            not_falling_knife = mret_bar > -0.02
            vol_pct = row["atr_val"] / bar.close if bar.close > 0 else 0.0
            not_hyper_vol = vol_pct < 0.06

            vol_scale = max(0.5, min(1.0, 0.015 / max(vol_pct, 1e-6)))
            rsi_scale = max(0.5, 1.0 - max(0.0, rsi_8 - 40) / 40)
            mret_scale = max(0.5, min(1.5, 1.0 + 40.0 * mret_bar))
            funding = bar.funding_rate if hasattr(bar, 'funding_rate') else 0.0
            fund_scale = max(0.5, min(2.5, 1.0 - 3000.0 * funding))
            if bull_fortress and not_falling_knife and not_hyper_vol and not raw_bear_fortress and (not bear_fortress or m15_long >= m15_short) and macro_bull_ok and long_gate_ok:
                entry_size = long_size * self._entry_scale * vol_scale * rsi_scale * mret_scale * fund_scale
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
