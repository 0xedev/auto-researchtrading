# CLAUDE.md

This file gives code agents the current operating context for this repository.

## Project Overview

This repository is a trading research workspace built around:

- cached 1h and 15m market data
- a fixed backtest engine in `prepare.py`
- XGBoost directional and meta models
- a macro HMM regime layer
- a live strategy implementation in `strategy.py`
- validation and OOS robustness runners

The older "single mutable file with 20+ Sharpe" story still exists in historical docs, but it is not the right mental model for the current audited branch.

## Commands

```bash
# Refresh data caches
uv run prepare.py
uv run prepare.py --mode 15m

# Train models
uv run train_model.py --timeframe 1h
uv run train_model.py --timeframe 4h
uv run train_model.py --timeframe 15m
uv run train_model.py --timeframe 1h --train_hmm

# Validation and robustness backtests
uv run backtest.py --timeframe 1h
uv run backtest.py --timeframe 15m
uv run backtest.py --timeframe 4h --stress-fees --capacity
uv run backtest.py --timeframe 1h --oos

# Cost stress tests (audit knobs, --no-log to avoid polluting leaderboard)
uv run backtest.py --timeframe 1h --oos --slippage-bps 5 --no-log
uv run backtest.py --timeframe 1h --oos --slippage-bps 10 --no-log

# Full OOS robustness suite
uv run evaluate.py --timeframe 1h --label exp269

# Walk-forward validation (temporal consistency across rolling windows)
uv run validate_wf.py --timeframe 1h
uv run validate_wf.py --timeframe 1h --slippage-bps 5

# Feature drift monitoring (PSI-based, run before live deployment)
uv run drift_monitor.py
uv run drift_monitor.py --recent-days 90 --ref-split train

# FINAL PRODUCTION SIGN-OFF ONLY (interactive confirmation required)
# DO NOT run during research iteration — destroys holdout independence
uv run backtest.py --timeframe 1h --holdout
uv run evaluate.py --timeframe 1h --holdout --label final

# Benchmark comparison
uv run run_benchmarks.py
```

There are no unit tests or CI checks. Backtests and OOS evaluation are the verification path.

## Working Assumptions

- `strategy.py` is still the primary experiment surface.
- `prepare.py`, `backtest.py`, and `benchmarks/` should normally be treated as fixed infrastructure.
- Updating docs or maintenance files is allowed when the task explicitly asks for it.
- Old materials that quote 20+ Sharpe are historical. On the corrected harness, realistic hourly Sharpe is generally much lower.

## Architecture

### Data

Defined in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L38):

- 17 symbols on 1h bars
- 16 symbols on 15m bars
- train / val / robustness / oos splits with fixed dates in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L61)

1h candles come from CryptoCompare with Hyperliquid fallback.
Funding comes from Binance with Hyperliquid fallback.
15m candles come from Binance spot plus a HuggingFace XAU source.

### Features

Feature generation lives in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L146) and includes:

- multi-horizon returns
- RSI and MACD
- Bollinger width and ATR
- Donchian / VWAP distance
- market context and relative-strength features
- simple market-structure proxies
- fractional differentiation

### Models

Training lives in [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L135).

Current stack:

- XGBoost directional classifiers
- XGBoost meta models
- optional state-specific artifacts when enough HMM-state data exists
- macro HMM training in [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L57)

Artifact layout is documented in [models/MODELS.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/models/MODELS.md).

### Strategy Runtime

The live strategy flow is:

1. load model artifacts in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L89)
2. precompute prediction tables and HMM state assignments in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L127)
3. warm symbol caches in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L271)
4. run the event-driven entry/exit logic in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L395)

The current default 1h branch primarily uses:

- 1h directional probabilities for entries
- 1h and 4h meta quality as confirmation
- HMM-aware size multipliers
- ATR / ratchet / time-based exits
- ranked candidate selection with capped open positions

### Backtest Engine

The fixed engine and score function live in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L858) and [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L1138).

Important guardrails:

- fewer than 10 trades gives `-999`
- fewer than 1 trade per day gives `-999`
- drawdown above 50% gives `-999`
- final equity below 50% gives `-999`

Benchmark audit thresholds used by the CLIs:

- Sharpe >= 3.5
- Win rate >= 60%
- Trades/day >= 1.0
- Profit factor >= 4.0
- Max drawdown < 10%

## Holdout Discipline

`prepare.py` defines `HOLDOUT_START = "2025-10-01"` and `HOLDOUT_END = "2025-12-31"`.

**This is the only truly unseen data remaining.** The OOS period (Jan–Dec 2025) was
observed hundreds of times during the research loop and is no longer independent.

Rules:
- Never run `evaluate.py` or `backtest.py --holdout` during research iteration.
- Use `--holdout` exactly ONCE at final production sign-off.
- Both CLIs gate behind an interactive "FINAL-SIGN-OFF" confirmation string.

## Production-Readiness Status (as of exp490, April 2026)

| Gap | Status | Result |
|-----|--------|--------|
| Cost model (10bp stress) | ✅ Done | OOS Sharpe 1.96 at 10bp, PF=4.81 |
| Walk-forward validation | ✅ Done (see validate_wf.py) | Run and check results |
| Hold-out carve-out | ✅ Done | Oct–Dec 2025 locked, gate added |
| OOS contamination | ⚠️ Structural | OOS 2025 was seen ~hundreds of times |
| Model drift monitoring | ✅ Done (see drift_monitor.py) | Run before live deployment |

## Key Files

- [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py)
- [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py)
- [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py)
- [backtest.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/backtest.py)
- [evaluate.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/evaluate.py)
- [validate_wf.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/validate_wf.py) — walk-forward temporal consistency harness
- [drift_monitor.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/drift_monitor.py) — PSI-based feature drift monitor
- [run_benchmarks.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/run_benchmarks.py)
- [strategy_exp269.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy_exp269.py)
- [strategy_diff.txt](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy_diff.txt)

## Historical Docs

Treat these as archive material, not as the source of truth for the current branch:

- [POST.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/POST.md)
- [TWITTER_THREAD.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/TWITTER_THREAD.md)
- older README sections that mention 20+ Sharpe before the March 2026 audit
