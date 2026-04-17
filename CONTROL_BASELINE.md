# Control Baseline

This document defines the frozen research control used for stage 2 of the competitive roadmap.

## Active Control

- Primary lane: autonomous, explainable, market-neutral AI hedge fund candidate
- Primary research path: `1h`
- Active model set: `exp256_active`
- Primary control split:
  - train: `2017-01-01` to `2022-06-30`
  - val: `2022-07-01` to `2024-06-30`
- Preserved final sign-off split:
  - holdout: `2025-10-01` to `2025-12-31`
- Clean post-research OOS check:
  - `2026q1`

## Model Pinning

The default model pin is now `exp256_active`.

At runtime, `strategy.py` resolves model files in this order:

1. `models/exp256_active/`
2. `models/`

This preserves the currently working control that reproduces the known `2026q1` validation read.

Override only when doing explicit baseline maintenance:

```bash
AUTOTRADER_MODEL_SET=exp490_production uv run backtest.py --timeframe 1h --2026q1 --no-log
```

## What Is Frozen

The following are part of the control and should not be changed casually:

- the split windows in `prepare.py`
- the default model pin in `strategy.py`
- the interpretation of holdout and `2026q1`
- the benchmark lens in `backtest.py` / `evaluate.py`

## Archived Comparison Set

`exp490_production` is kept as an archived comparison set, not the default live control.

When pinned explicitly, it does not reproduce the currently accepted `2026q1` read under the current `strategy.py` logic, so it should be treated as an audit branch rather than the default baseline.

## Allowed Work On Top Of The Control

- new evaluation metrics
- docs and reporting improvements
- live execution planning
- explainability instrumentation
- strategy experiments that do not redefine the control itself
