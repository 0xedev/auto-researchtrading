# Models Directory

This directory now contains a mix of live fallback artifacts, archived snapshots, and training outputs.

## Directory Layout

- `models/`: root artifacts used by training, dataset prep, and default fallback loading
- `models/exp256_active/`: archived snapshot of an earlier active model set
- `models/exp256_pre_retrain/`: rollback snapshot from before the later retrain attempt

## What The Current Strategy Loads

The live loader in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L60) now pins the default control to `exp256_active`.

The search order is:

1. `models/exp256_active/`
2. `models/`

Within each directory, generic lead/meta artifacts prefer `.xgb` before `.json`. The current default control is the `exp256_active` snapshot because it reproduces the accepted `2026q1` validation read under the live `strategy.py`.

## Root Artifacts Present Today

### Global fallback models

- `lead_15m.xgb`
- `lead_1h.xgb`
- `lead_1h.json`
- `lead_4h.xgb`
- `meta_15m.xgb`
- `meta_1h.xgb`
- `meta_1h.json`
- `meta_4h.xgb`

Root artifacts are fallback/training outputs rather than the first source of truth for the default control.

### Additional experimental artifacts

- `lead_long_15m.xgb`
- `lead_short_15m.xgb`
- `meta_long_15m.xgb`
- `meta_short_15m.xgb`
- `lead_1h_s0.json`
- `lead_1h_s0.xgb`
- `meta_1h_s0.json`
- `meta_1h_s0.xgb`

These come from training experiments and compatibility outputs. They are not the primary root fallback path used by the current default strategy.

### Macro regime artifacts

- `macro_hmm.joblib`
- `macro_scaler.joblib`

These are produced by [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L57) when `--train_hmm` is used.

They are consumed by:

- dataset preparation in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L356)
- prediction-table HMM inference in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L141)

## Snapshot Directories

### `exp256_active/`

Contains a frozen snapshot of:

- `lead_15m.xgb`
- `lead_1h.xgb`
- `lead_4h.xgb`
- `lead_long_15m.xgb`
- `lead_short_15m.xgb`
- `meta_15m.xgb`
- `meta_1h.xgb`
- `meta_4h.xgb`
- `meta_long_15m.xgb`
- `meta_short_15m.xgb`

This directory is now the default pinned control.

### `exp490_production/`

Contains an archived `1h` comparison set:

- `lead_1h.json`
- `meta_1h.json`
- `lead_1h_s0.json`
- `lead_1h_s2.json`
- `meta_1h_s0.json`
- `meta_1h_s2.json`
- `macro_hmm.joblib`
- `macro_scaler.joblib`

### `exp256_pre_retrain/`

Contains the same artifact layout as a pre-retrain rollback point.

## How Training Writes Files

From [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L135):

- `15m` training writes split long/short artifacts plus compatibility fallbacks:
  - `lead_long_15m.xgb`
  - `lead_short_15m.xgb`
  - `meta_long_15m.xgb`
  - `meta_short_15m.xgb`
  - `lead_15m.xgb`
  - `meta_15m.xgb`
- `1h` and `4h` training writes:
  - `lead_{tf}.json`
  - `meta_{tf}.json`
- when enough HMM-state data exists, training can also emit:
  - `lead_{tf}_s{state}.json`
  - `meta_{tf}_s{state}.json`

Because the root directory may contain both older `.xgb` and newer `.json` files, verify which files are present before assuming which artifact the strategy will actually load.

## New Metadata And Short-Confidence Artifacts

Training now also supports model-set metadata plus short-confidence artifacts:

- `metadata.json`
  - feature profile (`price_only` or `price_context`)
  - short-confidence mode (`raw`, `platt`, `isotonic`, `rank`, or `bear_model`)
  - calibrator artifact name when present
  - bear-only short artifact name when present
- `short_conf_15m_platt.joblib`
- `short_conf_15m_isotonic.joblib`
- `meta_short_bear_15m.xgb`

The live strategy reads a single normalized short-confidence field from these artifacts when configured, while preserving raw `meta_short_15m` for diagnostics and control comparisons.

## Practical Notes

- The default 1h strategy currently relies most on root 1h directional probabilities and 1h/4h meta gating.
- 15m split artifacts are available for experimentation and alignment, but they are not the dominant live entry driver in the current default branch.
- If you retrain models, double-check the loader paths in `strategy.py` so the new artifacts are actually being used.
