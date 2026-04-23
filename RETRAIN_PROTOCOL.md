# Retrain Protocol

This document defines how model retraining should be done from the frozen
`exp494` strategy shell.

## Purpose

Retraining is allowed to improve model quality, but it is not allowed to blur
the line between:

- model-side changes
- shell/rule changes
- control drift

The goal is to answer a narrow question cleanly:

> Does a new model artifact improve the `exp494` trading shell without
> introducing new failure modes?

## Frozen References

### Strategy Control

- active shell: `exp494`
- current best validation result:
  - Sharpe: `2.243`
  - PF: `12.744`
  - MaxDD: `0.438`
- clean post-research OOS check (`2026q1`):
  - Sharpe: `1.983`
  - PF: `5.893`
  - MaxDD: `0.549`

Important:

- those `exp494` validation figures were recorded on the older pre-`exp503`
  validation window
- the current code now uses:
  - train: `2017-01-01` to `2024-06-30`
  - val: `2024-07-01` to `2025-09-30`

So new retrain work must be judged on the current split definitions in
[prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py:102), even
if the shell reference is still `exp494`.

### Runtime Model Pin

- default model set: `exp256_active`
- runtime resolution order:
  1. `models/exp256_active/`
  2. `models/`

Do not overwrite `exp256_active` during normal research.

## Non-Negotiable Rules

1. Never retrain into the root `models/` directory during experiments.
2. Never overwrite `models/exp256_active/` unless doing explicit control maintenance.
3. Every retrain must use a tagged `--model-set`.
4. Every retrain branch must be logged in `results.tsv`, including rejects.
5. `research_loop.py` remains out of scope. Retrains are manual and explicit.
6. Holdout remains untouched.

## Required Retrain Workflow

1. Start from the frozen `exp494` strategy shell.
2. Train only into a tagged directory:

```bash
uv run train_model.py --timeframe 15m --model-set expXXX_name ...
```

3. Evaluate in the smallest safe way first:
   - if it is a partial 15m short-side artifact, test it as an overlay before replacing the full shell
   - if diagnostics already show negative OOS separation, skip the full backtest and log `NO_BACKTEST`

4. Only after the branch clears the cheap diagnostic gate should it run:

```bash
uv run backtest.py --timeframe 1h --no-log
uv run evaluate.py --timeframe 1h --split 2026q1 --label expXXX
```

5. Compare against `exp494`, not against historical peaks like `exp269`.
6. Do not present a post-`exp503` validation score as directly comparable to the
   older `exp494` validation score unless the split window is explicitly noted.

## What Counts As A Valid Model-Side Win

A retrained model branch counts as a valid win over `exp494` only if all of the
following are true:

1. The strategy shell is unchanged, or the shell change is strictly limited to
   loading the new artifact.
2. Validation does not regress below the current control on primary quality:
   - Sharpe `>= 2.243`
   - MaxDD `<= 0.438`
3. `2026q1` does not regress:
   - Sharpe `>= 1.983`
   - MaxDD `<= 0.549`
4. Trades/day stays above the harness floor.
5. Short-side branches must increase bear-family short quality without adding
   sideways-short leakage.

If a branch improves PF but fails the Sharpe/DD bar, it is a near miss, not a
keep.

## Overlay-First Rule

Use overlay-style evaluation when the retrain changes only one artifact family,
especially for:

- `meta_short_15m`
- bear-only short meta artifacts
- short-confidence calibrators

Do not replace the whole 15m stack first unless the overlay version has already
shown clean value.

## When To Skip A Full Backtest

Skip the expensive full backtest and log `NO_BACKTEST` when early diagnostics
already show one of these:

- bear overlay reduces bear confidence in both `val` and `2026q1`
- raw bear entries are removed with no compensating OOS additions
- sideways-short leakage clearly increases in cache-level diagnostics
- a partial artifact is inert relative to the control

## Suggested Command Shapes

### Tagged retrain

```bash
uv run train_model.py --timeframe 15m --model-set expXXX_tag --feature-profile price_only
```

### Tagged retrain with explicit train-window trim

```bash
uv run train_model.py \
  --timeframe 15m \
  --model-set expXXX_tag \
  --train-start 2020-01-01 \
  --trim-train-end 2022-06-30
```

### Explicit model-pin evaluation

```bash
AUTOTRADER_MODEL_SET=expXXX_tag uv run backtest.py --timeframe 1h --no-log
AUTOTRADER_MODEL_SET=expXXX_tag uv run evaluate.py --timeframe 1h --split 2026q1 --label expXXX_tag
```

## Current Interpretation

The old setup may be partially capped, but the first cleanup priority is not
"retrain everything." It is:

1. keep the control pin stable
2. keep retrains tagged and isolated
3. evaluate model changes without shell drift
4. only keep retrains that beat `exp494` on both validation quality and
   `2026q1` robustness
