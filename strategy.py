import os
import xgboost as xgb
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from prepare import calculate_features, FEATURE_COLS, load_data


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

    def _load_models(self):
        for tf in ["15m", "1h", "4h"]:
            m_path = f"models/lead_{tf}.xgb"
            meta_path = f"models/meta_{tf}.xgb"
            long_path = f"models/lead_long_{tf}.xgb"
            short_path = f"models/lead_short_{tf}.xgb"
            long_meta_path = f"models/meta_long_{tf}.xgb"
            short_meta_path = f"models/meta_short_{tf}.xgb"

            if os.path.exists(m_path):
                self.models[tf] = xgb.XGBClassifier()
                self.models[tf].load_model(m_path)

            if os.path.exists(meta_path):
                self.meta_models[tf] = xgb.XGBClassifier()
                self.meta_models[tf].load_model(meta_path)

            if os.path.exists(long_path):
                self.long_models[tf] = xgb.XGBClassifier()
                self.long_models[tf].load_model(long_path)

            if os.path.exists(short_path):
                self.short_models[tf] = xgb.XGBClassifier()
                self.short_models[tf].load_model(short_path)

            if os.path.exists(long_meta_path):
                self.long_meta_models[tf] = xgb.XGBClassifier()
                self.long_meta_models[tf].load_model(long_meta_path)

            if os.path.exists(short_meta_path):
                self.short_meta_models[tf] = xgb.XGBClassifier()
                self.short_meta_models[tf].load_model(short_meta_path)

        self.models_loaded = True
        print(f"LOADED QUANTUM FORTRESS: {list(self.models.keys())}")

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
                meta_long = self.long_meta_models[timeframe].predict_proba(X)[:, 1] if timeframe in self.long_meta_models else np.zeros(len(X))
                meta_short = self.short_meta_models[timeframe].predict_proba(X)[:, 1] if timeframe in self.short_meta_models else np.zeros(len(X))
                table = pd.DataFrame({
                    "timestamp": df["timestamp"].values,
                    f"bull_{timeframe}": bull,
                    f"bear_{timeframe}": bear,
                    f"meta_{timeframe}": np.maximum(meta_long, meta_short),
                    f"meta_long_{timeframe}": meta_long,
                    f"meta_short_{timeframe}": meta_short,
                })
            else:
                probs = self.models[timeframe].predict_proba(X) if timeframe in self.models else None
                meta = self.meta_models[timeframe].predict_proba(X)[:, 1] if timeframe in self.meta_models else np.zeros(len(X))
                table = pd.DataFrame({
                    "timestamp": df["timestamp"].values,
                    f"bull_{timeframe}": probs[:, 1] if probs is not None and probs.shape[1] > 1 else np.zeros(len(X)),
                    f"bear_{timeframe}": probs[:, 2] if probs is not None and probs.shape[1] > 2 else np.zeros(len(X)),
                    f"meta_{timeframe}": meta,
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

            self.symbol_caches[symbol] = base.fillna(0.0).to_dict("records")
            self.bar_counts[symbol] = 0

        print(f"Market-Aware Fortress cache warmed for {len(self.symbol_caches)} symbols.")

    def on_bar(self, bar_data, portfolio):
        if not self.models_loaded:
            self._load_models()

        equity = portfolio.equity
        signals: List[Signal] = []

        # Macro regime: rolling 72-bar market return detects sustained bear trends
        any_sym = next(iter(bar_data), None)
        if any_sym and any_sym in self.symbol_caches:
            idx = self.bar_counts.get(any_sym, 0)
            if idx < len(self.symbol_caches[any_sym]):
                mret = self.symbol_caches[any_sym][idx].get("market_ret", 0.0)
                self._market_ret_buf.append(mret)
                if len(self._market_ret_buf) > 72:
                    self._market_ret_buf.pop(0)
                self._macro_bear = sum(self._market_ret_buf) < -0.02

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
            atr_mult = 3.0 if self._macro_bear else 7.0
            stop_dist = atr_mult * row["atr_val"] if row["atr_val"] > 0 else max(bar.close * 0.02, 1e-6)
            bull_signal = row["bull_15m"] > 0.32
            bear_signal = row["bear_15m"] > 0.55
            bull_fortress = bull_signal and m15_long > 0.28 and rsi_8 < 68
            raw_bear_fortress = (
                row["bear_15m"] > 0.62
                and row["bear_15m"] > row["bull_15m"] + 0.12
                and rsi_8 > 48
            )
            bear_fortress = (bear_signal and m15_short > 0.28 and rsi_8 > 32) or raw_bear_fortress
            bull_soft = bull_signal and supportive_regime and supportive_regime_4h and m15_long > 0.50
            bear_soft = bear_signal and supportive_regime and supportive_regime_4h and m15_short > 0.50

            # meta_score-based continuous sizing with regime gating
            if meta_score > 0.65 and supportive_regime:
                risk_per_trade = equity * 0.2200; long_size = (risk_per_trade * bar.close) / stop_dist
                short_size = (risk_per_trade * bar.close) / stop_dist
            elif meta_score > 0.50 and supportive_regime:
                risk_per_trade = equity * 0.1100; long_size = (risk_per_trade * bar.close) / stop_dist
                short_size = (risk_per_trade * bar.close) / stop_dist
            elif supportive_regime:
                risk_per_trade = equity * 0.0900; long_size = (risk_per_trade * bar.close) / stop_dist
                short_size = (risk_per_trade * bar.close) / stop_dist
            else:
                long_size = equity * 0.08
                short_size = equity * 0.08
            # Regime-throttled risk: keep flow, but cut weak-regime exposure hard.
            if not supportive_regime:
                long_size *= 0.05
            # Continuous regime scaling (25x) — smooth crush based on macro intensity
            mret_sum = sum(self._market_ret_buf) if self._market_ret_buf else 0.0
            long_factor = max(0.20, min(1.0, 1.0 + 25.0 * mret_sum))
            short_factor = max(0.20, min(1.0, 1.0 - 25.0 * mret_sum))
            long_size *= long_factor
            short_size *= short_factor

            if pos != 0:
                age = self.position_ages.get(symbol, 0) + 1
                self.position_ages[symbol] = age
                max_hold_bars = 3 if self.timeframe_arg == "15m" else 6

                if pos > 0:
                    signal_decay_exit = age >= 3 and (not bull_signal) and m15_long < 0.40
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
                        half_size = long_size * 0.5
                        if bull_signal and m15_long > 0.65 and abs(pos) + 1.0 < long_size:
                            desired = long_size
                        elif False and (not bull_soft) and abs(pos) - 1.0 > half_size:
                            desired = half_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired))
                        self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), bar.high - stop_dist)
                else:
                    signal_decay_exit = age >= 3 and (not bear_signal) and m15_short < 0.40 and not raw_bear_fortress
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
                        half_size = short_size * 0.5
                        if bear_signal and m15_short > 0.65 and abs(pos) + 1.0 < short_size:
                            desired = -short_size
                        elif False and (not bear_soft) and abs(pos) - 1.0 > half_size:
                            desired = -half_size
                        if abs(desired - pos) > 1.0:
                            signals.append(Signal(symbol, desired))
                        self.trailing_stops[symbol] = min(self.trailing_stops.get(symbol, 9e18), bar.low + stop_dist)
                continue

            self.position_ages[symbol] = 0
            meta_gate_ok = meta_score > 0.355
            macro_bull_ok = not self._macro_bear or meta_score > 0.40
            if bull_fortress and not raw_bear_fortress and (not bear_fortress or m15_long >= m15_short) and macro_bull_ok and meta_gate_ok:
                entry_size = long_size * 0.5 if self.timeframe_arg == "15m" else long_size
                signals.append(Signal(symbol, entry_size))
                self.trailing_stops[symbol] = bar.low - stop_dist
                self.position_ages[symbol] = 0
            elif bull_soft and (not bear_soft or m15_long >= m15_short) and meta_gate_ok:
                entry_size = long_size * 0.25 if self.timeframe_arg == "15m" else long_size * 0.5
                signals.append(Signal(symbol, entry_size))
                self.trailing_stops[symbol] = bar.low - stop_dist
                self.position_ages[symbol] = 0
            elif bear_fortress and meta_gate_ok:
                entry_size = short_size * 0.5 if self.timeframe_arg == "15m" else short_size
                signals.append(Signal(symbol, -entry_size))
                self.trailing_stops[symbol] = bar.high + stop_dist
                self.position_ages[symbol] = 0
            elif bear_soft and meta_gate_ok:
                entry_size = short_size * 0.25 if self.timeframe_arg == "15m" else short_size * 0.5
                signals.append(Signal(symbol, -entry_size))
                self.trailing_stops[symbol] = bar.high + stop_dist
                self.position_ages[symbol] = 0
            # exp149: 15m Front-run logic (increase frequency without fee-bleed)
            if self.timeframe_arg == '1h' and bull_fortress and m15_long > 0.65: entry_size = long_size

        return signals
