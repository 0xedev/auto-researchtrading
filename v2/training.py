from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from .alpha import build_training_frame, manifest_by_name, selected_feature_columns
from .data import build_bundle_dataset
from .legacy import build_legacy_foundation_frame, foundation_source_model_set
from .manifests import BUNDLE_MANIFESTS


def _artifact_dir(model_set: str, bundle_name: str) -> Path:
    target = Path("models") / model_set / "v2" / bundle_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def choose_temporal_split(
    train_frame: pd.DataFrame,
    *,
    min_train_pos: int = 50,
    min_val_pos: int = 10,
    preferred_fraction: float = 0.8,
) -> tuple[pd.Series, pd.Series, dict] | None:
    timestamps = np.sort(train_frame["timestamp"].unique())
    if len(timestamps) < 20:
        return None

    candidate_fracs = [preferred_fraction, 0.75, 0.70, 0.67, 0.60, 0.55]
    for frac in candidate_fracs:
        split_idx = int(len(timestamps) * frac)
        split_idx = min(max(split_idx, 1), len(timestamps) - 1)
        cutoff = timestamps[split_idx - 1]
        train_mask = train_frame["timestamp"] <= cutoff
        val_mask = ~train_mask
        train_pos = int(train_frame.loc[train_mask, "target"].sum())
        val_pos = int(train_frame.loc[val_mask, "target"].sum())
        if train_pos >= min_train_pos and val_pos >= min_val_pos:
            return train_mask, val_mask, {
                "split_fraction": float(frac),
                "cutoff_timestamp": int(cutoff),
                "train_pos": train_pos,
                "val_pos": val_pos,
                "train_rows": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
            }

    return None


def train_sleeve(
    bundle_name: str,
    sleeve_name: str,
    model_set: str,
    feature_profile: str = "price_context_plus",
    max_symbols: int | None = None,
    split: str = "train",
) -> dict:
    bundle = BUNDLE_MANIFESTS[bundle_name]
    manifest = manifest_by_name(sleeve_name)
    if manifest.model_family == "legacy_strategy":
        source_split = ""
        train_frame = pd.DataFrame()
        for candidate_split in ("train", "val", "2026q1"):
            train_frame = build_legacy_foundation_frame(
                sleeve_name=sleeve_name,
                bundle_name=bundle_name,
                split=candidate_split,
                max_symbols=max_symbols,
            )
            if not train_frame.empty:
                source_split = candidate_split
                break
        if train_frame.empty:
            return {
                "status": "skipped",
                "reason": "empty_legacy_frame",
                "sleeve": sleeve_name,
                "bundle": bundle_name,
            }

        artifact_dir = _artifact_dir(model_set, bundle_name)
        metadata_path = artifact_dir / f"{sleeve_name}.metadata.json"
        signal_counts = train_frame["reason_tag"].value_counts().to_dict() if "reason_tag" in train_frame.columns else {}
        regime_counts = train_frame["regime_family"].value_counts().to_dict() if "regime_family" in train_frame.columns else {}
        metrics = {
            "rows": int(len(train_frame)),
            "unique_symbols": int(train_frame["symbol"].nunique()),
            "unique_timestamps": int(train_frame["timestamp"].nunique()),
            "avg_confidence": float(train_frame["confidence"].mean()),
            "signal_counts": signal_counts,
            "regime_counts": regime_counts,
            "source_model_set": foundation_source_model_set(),
            "source_split": source_split,
        }
        metadata = {
            "bundle": bundle_name,
            "sleeve": sleeve_name,
            "model_family": manifest.model_family,
            "feature_profile": manifest.feature_profile,
            "manifest": manifest.__dict__,
            "metrics": metrics,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return {
            "status": "prepared",
            "bundle": bundle_name,
            "sleeve": sleeve_name,
            "metadata_path": str(metadata_path),
            **metrics,
        }

    dataset = build_bundle_dataset(bundle_name, split=split, feature_profile=feature_profile, max_symbols=max_symbols)
    if dataset.empty:
        return {"status": "skipped", "reason": "empty_dataset", "sleeve": sleeve_name, "bundle": bundle_name}

    train_frame = build_training_frame(dataset, bundle, manifest)
    if train_frame.empty or len(train_frame) < 500:
        return {"status": "skipped", "reason": "insufficient_active_rows", "rows": int(len(train_frame)), "sleeve": sleeve_name, "bundle": bundle_name}

    feature_cols = selected_feature_columns(train_frame)
    if not feature_cols:
        return {"status": "skipped", "reason": "no_features", "sleeve": sleeve_name, "bundle": bundle_name}

    split_result = choose_temporal_split(train_frame)
    if split_result is None:
        return {
            "status": "skipped",
            "reason": "insufficient_temporal_split",
            "sleeve": sleeve_name,
            "bundle": bundle_name,
        }
    train_mask, val_mask, split_meta = split_result

    X_train = train_frame.loc[train_mask, feature_cols]
    y_train = train_frame.loc[train_mask, "target"].astype(int)
    X_val = train_frame.loc[val_mask, feature_cols]
    y_val = train_frame.loc[val_mask, "target"].astype(int)

    if y_train.sum() < 50 or y_val.sum() < 10:
        return {
            "status": "skipped",
            "reason": "insufficient_positive_samples",
            "train_pos": int(y_train.sum()),
            "val_pos": int(y_val.sum()),
            "sleeve": sleeve_name,
            "bundle": bundle_name,
        }

    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        tree_method="hist",
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, sample_weight=weights, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_pred = (val_probs >= manifest.min_confidence).astype(int)
    metrics = {
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate": float(y_val.mean()),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_auc": float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.0,
        "val_confident_rate": float((val_probs >= manifest.min_confidence).mean()),
        "split_fraction": float(split_meta["split_fraction"]),
        "train_pos": int(split_meta["train_pos"]),
        "val_pos": int(split_meta["val_pos"]),
    }

    artifact_dir = _artifact_dir(model_set, bundle_name)
    model_path = artifact_dir / f"{sleeve_name}.json"
    metadata_path = artifact_dir / f"{sleeve_name}.metadata.json"
    model.save_model(model_path)
    metadata = {
        "bundle": bundle_name,
        "sleeve": sleeve_name,
        "model_family": manifest.model_family,
        "feature_profile": feature_profile,
        "feature_columns": feature_cols,
        "manifest": manifest.__dict__,
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return {
        "status": "trained",
        "bundle": bundle_name,
        "sleeve": sleeve_name,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        **metrics,
    }
