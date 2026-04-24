from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .alpha import build_training_frame, manifest_by_name, selected_feature_columns
from .data import build_bundle_dataset
from .legacy import build_legacy_foundation_frame
from .manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS
from .types import SleeveSignal


class V2SignalEngine:
    def __init__(
        self,
        bundle_name: str,
        model_set: str,
        active_sleeves: list[str] | None = None,
        feature_profile: str = "price_context_plus",
        max_symbols: int | None = None,
        symbols: list[str] | None = None,
        confidence_overrides: dict[str, float] | None = None,
    ):
        self.bundle_name = bundle_name
        self.bundle = BUNDLE_MANIFESTS[bundle_name]
        self.model_set = model_set
        self.feature_profile = feature_profile
        self.max_symbols = max_symbols
        self.symbols = list(symbols) if symbols is not None else None
        self.active_sleeves = active_sleeves or list(SLEEVE_MANIFESTS.keys())
        self.confidence_overrides = dict(confidence_overrides or {})
        self.bundle_frame = pd.DataFrame()
        self.timestamps: list[int] = []
        self.sleeve_tables: dict[str, pd.DataFrame] = {}
        self.sleeve_rows_by_timestamp: dict[str, dict[int, list[dict]]] = {}
        self.sleeve_features: dict[str, list[str]] = {}
        self.models: dict[str, xgb.XGBClassifier] = {}
        self.bundle_close_by_timestamp: dict[int, dict[str, float]] = {}
        self.bundle_funding_by_timestamp: dict[int, dict[str, dict[str, float]]] = {}

    def _build_bundle_close_index(self) -> None:
        self.bundle_close_by_timestamp = {}
        self.bundle_funding_by_timestamp = {}
        if self.bundle_frame.empty or "timestamp" not in self.bundle_frame.columns:
            return
        close_frame = self.bundle_frame.copy()
        for column, default in (
            ("base_funding_rate", 0.0),
            ("base_has_funding", 0.0),
            ("base_bar_interval_hours", 1.0),
        ):
            if column not in close_frame.columns:
                close_frame[column] = default
        close_frame = close_frame[
            [
                "timestamp",
                "symbol",
                "base_close",
                "base_funding_rate",
                "base_has_funding",
                "base_bar_interval_hours",
            ]
        ].copy()
        for row in close_frame.itertuples(index=False):
            timestamp = int(row.timestamp)
            close_by_symbol = self.bundle_close_by_timestamp.setdefault(timestamp, {})
            close_by_symbol[str(row.symbol)] = float(row.base_close)
            funding_by_symbol = self.bundle_funding_by_timestamp.setdefault(timestamp, {})
            funding_by_symbol[str(row.symbol)] = {
                "funding_rate": float(row.base_funding_rate),
                "has_funding": float(row.base_has_funding),
                "bar_interval_hours": float(row.base_bar_interval_hours),
            }

    def _build_sleeve_timestamp_index(self, sleeve_name: str) -> dict[int, list[dict]]:
        frame = self.sleeve_tables.get(sleeve_name, pd.DataFrame())
        rows_by_timestamp: dict[int, list[dict]] = {}
        if frame.empty or "timestamp" not in frame.columns:
            self.sleeve_rows_by_timestamp[sleeve_name] = rows_by_timestamp
            return rows_by_timestamp
        for row in frame.itertuples(index=False):
            payload = row._asdict() if hasattr(row, "_asdict") else dict(row)
            rows_by_timestamp.setdefault(int(payload["timestamp"]), []).append(payload)
        self.sleeve_rows_by_timestamp[sleeve_name] = rows_by_timestamp
        return rows_by_timestamp

    def close_by_symbol_at_timestamp(self, timestamp: int) -> dict[str, float]:
        return self.bundle_close_by_timestamp.get(int(timestamp), {})

    def funding_by_symbol_at_timestamp(self, timestamp: int) -> dict[str, dict[str, float]]:
        return self.bundle_funding_by_timestamp.get(int(timestamp), {})

    def prepare(self, split: str) -> None:
        self.bundle_frame = build_bundle_dataset(
            self.bundle_name,
            split=split,
            feature_profile=self.feature_profile,
            max_symbols=self.max_symbols,
            symbols=self.symbols,
        )
        self.timestamps = sorted(self.bundle_frame["timestamp"].unique().tolist()) if not self.bundle_frame.empty else []
        self.sleeve_rows_by_timestamp = {}
        self._build_bundle_close_index()

        for sleeve_name in self.active_sleeves:
            metadata_path = Path("models") / self.model_set / "v2" / self.bundle_name / f"{sleeve_name}.metadata.json"
            if not metadata_path.exists():
                continue

            metadata = json.loads(metadata_path.read_text())
            manifest = manifest_by_name(sleeve_name)
            if metadata.get("model_family") == "legacy_strategy" or manifest.model_family == "legacy_strategy":
                sleeve_frame = build_legacy_foundation_frame(
                    sleeve_name=sleeve_name,
                    bundle_name=self.bundle_name,
                    split=split,
                    max_symbols=self.max_symbols,
                    symbols=self.symbols,
                )
                if sleeve_frame.empty:
                    continue
                self.sleeve_tables[sleeve_name] = sleeve_frame
                self.sleeve_features[sleeve_name] = []
                self._build_sleeve_timestamp_index(sleeve_name)
                continue

            model_path = Path("models") / self.model_set / "v2" / self.bundle_name / f"{sleeve_name}.json"
            if not model_path.exists():
                continue
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            feature_cols = metadata.get("feature_columns", [])
            sleeve_frame = build_training_frame(self.bundle_frame, self.bundle, manifest)
            if sleeve_frame.empty:
                continue
            sleeve_frame = sleeve_frame.copy()
            sleeve_frame["confidence"] = model.predict_proba(sleeve_frame[feature_cols])[:, 1]
            self.models[sleeve_name] = model
            self.sleeve_features[sleeve_name] = feature_cols
            self.sleeve_tables[sleeve_name] = sleeve_frame
            self._build_sleeve_timestamp_index(sleeve_name)

    def signals_at_timestamp(self, timestamp: int) -> list[SleeveSignal]:
        signals: list[SleeveSignal] = []
        timestamp = int(timestamp)
        for sleeve_name in self.sleeve_tables:
            manifest = manifest_by_name(sleeve_name)
            rows_by_timestamp = self.sleeve_rows_by_timestamp.get(sleeve_name)
            if rows_by_timestamp is None:
                rows_by_timestamp = self._build_sleeve_timestamp_index(sleeve_name)
            rows = rows_by_timestamp.get(timestamp, [])
            if not rows:
                continue
            for row in rows:
                confidence = float(row.get("confidence", 0.0))
                min_confidence = float(self.confidence_overrides.get(sleeve_name, manifest.min_confidence))
                if confidence < min_confidence:
                    continue
                side = int(row.get("side", 0))
                if side == 0:
                    continue
                close = float(row.get("base_close", 0.0))
                atr_pct = abs(float(row.get("base_clock_atr_pct_24h", 0.0)))
                stop_distance = float(row.get("stop_distance", max(close * atr_pct * manifest.stop_atr_mult, close * 0.0025)))
                expected_edge_bps = float(
                    row.get(
                        "expected_edge_bps",
                        max(0.0, (confidence - manifest.min_confidence) * manifest.edge_scale_bps),
                    )
                )
                holding_horizon_hours = float(row.get("holding_horizon_hours", manifest.max_hold_hours))
                reason_tag = str(row.get("reason_tag", sleeve_name))
                metadata = {
                    "bundle": self.bundle_name,
                    "sleeve": sleeve_name,
                    "family": manifest.family,
                    "strategy_cluster": manifest.cluster,
                    "market_cluster": row.get("market_cluster", "other"),
                    "base_close": close,
                    "base_volume": float(row.get("base_volume", 0.0)),
                    "allocator_confidence": confidence,
                    "min_confidence": min_confidence,
                    "activation_rule": manifest.activation_rule,
                    "portfolio_weight": float(manifest.portfolio_weight),
                    "reentry_cooldown_hours": float(manifest.reentry_cooldown_hours),
                    "reentry_cooldown_scope": str(manifest.reentry_cooldown_scope),
                    "meta_score": float(row.get("meta_score", 0.0)),
                    "structure_score": int(row.get("structure_score", 0)),
                }
                signals.append(
                    SleeveSignal(
                        bundle=self.bundle_name,
                        sleeve=sleeve_name,
                        symbol=str(row.get("symbol")),
                        side=side,
                        confidence=confidence,
                        expected_edge_bps=expected_edge_bps,
                        holding_horizon_hours=holding_horizon_hours,
                        stop_distance=stop_distance,
                        target_notional=0.0,
                        regime_context=str(row.get("regime_family", "unknown")),
                        reason_tag=reason_tag,
                        cluster=str(manifest.cluster),
                        timestamp=timestamp,
                        metadata=metadata,
                    )
                )
        return signals
