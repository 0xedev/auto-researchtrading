# Models Directory

## Active Models (root)
These are loaded by `strategy.py`. Do not rename or move them.

| Model | Type | Description |
|---|---|---|
| `lead_15m.xgb` | XGBoost 3-class | 15m directional classifier (bull/bear/neutral) |
| `lead_1h.xgb` | XGBoost 3-class | 1h directional classifier |
| `lead_4h.xgb` | XGBoost 3-class | 4h directional classifier |
| `lead_long_15m.xgb` | XGBoost binary | 15m long signal (P(bull) — main entry driver) |
| `lead_short_15m.xgb` | XGBoost binary | 15m short signal (P(bear)) |
| `meta_15m.xgb` | XGBoost binary | 15m meta quality (trade/no-trade) |
| `meta_1h.xgb` | XGBoost binary | 1h meta quality — used for regime gating |
| `meta_4h.xgb` | XGBoost binary | 4h meta quality — used for regime gating |
| `meta_long_15m.xgb` | XGBoost binary | 15m long meta quality |
| `meta_short_15m.xgb` | XGBoost binary | 15m short meta quality |

All models: 100 trees, depth 12 (15m) or 14 (4h), 13 features.
Trained on: 2017-01-01 to 2022-06-30 (per-symbol start dates vary).

## Archived Versions

| Directory | When | Notes |
|---|---|---|
| `exp256_active/` | 2026-04-05 | Snapshot of current active models (exp256 baseline: Sharpe 1.86) |
| `exp256_pre_retrain/` | 2026-04-04 | Models before last retrain attempt (safe rollback point) |

## How Models Are Used

- **15m directional** (`lead_long/short_15m`) are the primary entry signal drivers
- **1h/4h meta** (`meta_1h`, `meta_4h`) control regime gating (supportive_regime threshold 0.45)
- **Meta score** is the average of active metas across timeframes — gates entries and sizes positions
- The 1h/4h lead models produce 3-class probabilities too dilute for direct entries (~0.02 per class)
