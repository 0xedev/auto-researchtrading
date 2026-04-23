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
    ):
        self.bundle_name = bundle_name
        self.bundle = BUNDLE_MANIFESTS[bundle_name]
        self.model_set = model_set
        self.feature_profile = feature_profile
        self.max_symbols = max_symbols
        self.active_sleeves = active_sleeves or list(SLEEVE_MANIFESTS.keys())
        self.bundle_frame = pd.DataFrame()
        self.timestamps: list[int] = []
        self.sleeve_tables: dict[str, pd.DataFrame] = {}
        self.sleeve_features: dict[str, list[str]] = {}
        self.models: dict[str, xgb.XGBClassifier] = {}

    def prepare(self, split: str) -> None:
        self.bundle_frame = build_bundle_dataset(
            self.bundle_name,
            split=split,
            feature_profile=self.feature_profile,
            max_symbols=self.max_symbols,
        )
        self.timestamps = sorted(self.bundle_frame["timestamp"].unique().tolist()) if not self.bundle_frame.empty else []

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
                )
                if sleeve_frame.empty:
                    continue
                self.sleeve_tables[sleeve_name] = sleeve_frame
                self.sleeve_features[sleeve_name] = []
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

    def signals_at_timestamp(self, timestamp: int) -> list[SleeveSignal]:
        signals: list[SleeveSignal] = []
        for sleeve_name, frame in self.sleeve_tables.items():
            manifest = manifest_by_name(sleeve_name)
            rows = frame[frame["timestamp"] == timestamp]
            if rows.empty:
                continue
            for row in rows.itertuples(index=False):
                confidence = float(getattr(row, "confidence", 0.0))
                if confidence < manifest.min_confidence:
                    continue
                side = int(getattr(row, "side", 0))
                if side == 0:
                    continue
                close = float(getattr(row, "base_close", getattr(row, "base_close", 0.0)))
                atr_pct = abs(float(getattr(row, "base_clock_atr_pct_24h", 0.0)))
                stop_distance = float(getattr(row, "stop_distance", max(close * atr_pct * manifest.stop_atr_mult, close * 0.0025)))
                expected_edge_bps = float(
                    getattr(
                        row,
                        "expected_edge_bps",
                        max(0.0, (confidence - manifest.min_confidence) * manifest.edge_scale_bps),
                    )
                )
                holding_horizon_hours = float(getattr(row, "holding_horizon_hours", manifest.max_hold_hours))
                reason_tag = str(getattr(row, "reason_tag", sleeve_name))
                metadata = {
                    "bundle": self.bundle_name,
                    "sleeve": sleeve_name,
                    "family": manifest.family,
                    "strategy_cluster": manifest.cluster,
                    "market_cluster": getattr(row, "market_cluster", "other"),
                    "base_close": close,
                    "base_volume": float(getattr(row, "base_volume", 0.0)),
                    "allocator_confidence": confidence,
                    "activation_rule": manifest.activation_rule,
                    "portfolio_weight": float(manifest.portfolio_weight),
                    "reentry_cooldown_hours": float(manifest.reentry_cooldown_hours),
                    "reentry_cooldown_scope": str(manifest.reentry_cooldown_scope),
                    "meta_score": float(getattr(row, "meta_score", 0.0)),
                    "structure_score": int(getattr(row, "structure_score", 0)),
                }
                signals.append(
                    SleeveSignal(
                        bundle=self.bundle_name,
                        sleeve=sleeve_name,
                        symbol=str(getattr(row, "symbol")),
                        side=side,
                        confidence=confidence,
                        expected_edge_bps=expected_edge_bps,
                        holding_horizon_hours=holding_horizon_hours,
                        stop_distance=stop_distance,
                        target_notional=0.0,
                        regime_context=str(getattr(row, "regime_family", "unknown")),
                        reason_tag=reason_tag,
                        cluster=str(manifest.cluster),
                        timestamp=int(timestamp),
                        metadata=metadata,
                    )
                )
        return signals
