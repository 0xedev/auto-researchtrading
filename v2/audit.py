from __future__ import annotations

import json

import pandas as pd

from .alpha import build_training_frame, manifest_by_name
from .data import build_bundle_dataset
from .legacy import build_legacy_foundation_frame
from .manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS
from .training import choose_temporal_split


def _split_metrics(frame: pd.DataFrame, bundle_name: str, sleeve_name: str, split: str, max_symbols: int | None = None) -> dict:
    bundle = BUNDLE_MANIFESTS[bundle_name]
    manifest = manifest_by_name(sleeve_name)
    if manifest.model_family == "legacy_strategy":
        labeled = build_legacy_foundation_frame(
            sleeve_name=sleeve_name,
            bundle_name=bundle_name,
            split=split,
            max_symbols=max_symbols,
        )
        if labeled.empty:
            return {
                "split": split,
                "rows": 0,
                "positive_rows": 0,
                "positive_rate": 0.0,
                "long_rows": 0,
                "short_rows": 0,
                "unique_symbols": 0,
                "unique_timestamps": 0,
            }
        return {
            "split": split,
            "rows": int(len(labeled)),
            "positive_rows": int(len(labeled)),
            "positive_rate": 1.0,
            "long_rows": int((labeled["side"] > 0).sum()),
            "short_rows": int((labeled["side"] < 0).sum()),
            "unique_symbols": int(labeled["symbol"].nunique()),
            "unique_timestamps": int(labeled["timestamp"].nunique()),
            "avg_confidence": float(labeled["confidence"].mean()),
            "signal_mix": labeled["reason_tag"].value_counts().to_dict() if "reason_tag" in labeled.columns else {},
        }

    labeled = build_training_frame(frame, bundle, manifest)
    if labeled.empty:
        return {
            "split": split,
            "rows": 0,
            "positive_rows": 0,
            "positive_rate": 0.0,
            "long_rows": 0,
            "short_rows": 0,
            "unique_symbols": 0,
            "unique_timestamps": 0,
        }

    metrics = {
        "split": split,
        "rows": int(len(labeled)),
        "positive_rows": int(labeled["target"].sum()),
        "positive_rate": float(labeled["target"].mean()),
        "long_rows": int((labeled["side"] > 0).sum()),
        "short_rows": int((labeled["side"] < 0).sum()),
        "unique_symbols": int(labeled["symbol"].nunique()),
        "unique_timestamps": int(labeled["timestamp"].nunique()),
    }
    if split == "train":
        split_result = choose_temporal_split(labeled)
        if split_result is None:
            metrics["temporal_split"] = None
        else:
            _, _, split_meta = split_result
            metrics["temporal_split"] = split_meta
    return metrics


def audit_sleeves(
    *,
    bundle_name: str,
    sleeves: list[str] | None = None,
    feature_profile: str = "price_context_plus",
    splits: tuple[str, ...] = ("train", "val", "2026q1"),
    max_symbols: int | None = None,
) -> list[dict]:
    sleeve_names = sleeves or sorted(SLEEVE_MANIFESTS)
    datasets = {
        split: build_bundle_dataset(bundle_name, split=split, feature_profile=feature_profile, max_symbols=max_symbols)
        for split in splits
    }

    rows = []
    for sleeve_name in sleeve_names:
        manifest = manifest_by_name(sleeve_name)
        row = {
            "bundle": bundle_name,
            "sleeve": sleeve_name,
            "family": manifest.family,
            "side_mode": manifest.side_mode,
            "feature_profile": feature_profile,
            "target_sigma": float(manifest.target_sigma),
            "min_confidence": float(manifest.min_confidence),
            "label_horizon_hours": float(manifest.label_horizon_hours),
            "max_hold_hours": float(manifest.max_hold_hours),
            "splits": {},
        }
        for split, frame in datasets.items():
            if manifest.model_family == "legacy_strategy":
                row["splits"][split] = _split_metrics(pd.DataFrame(), bundle_name, sleeve_name, split, max_symbols=max_symbols)
            else:
                row["splits"][split] = _split_metrics(frame, bundle_name, sleeve_name, split, max_symbols=max_symbols)
        rows.append(row)
    return rows


def audit_to_json(rows: list[dict]) -> str:
    return json.dumps(rows, indent=2, sort_keys=True)
