<p align="center">
  <img src="assets/logo.png" alt="Nunchi" width="480" />
</p>

<h3 align="center">Multi-Timeframe Trading Research Workspace</h3>

<p align="center">
  XGBoost + HMM regime research for cross-asset trading on a fixed backtest harness
</p>

---

This repository is the current research workspace behind the trading experiments. The live codebase is no longer just a single-file strategy loop: it now includes data preparation, model training, model snapshots, a fixed backtest engine, benchmark runners, and an out-of-sample robustness suite.

The active product lane and frozen research control are documented in:

- [POSITIONING.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/POSITIONING.md)
- [CONTROL_BASELINE.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/CONTROL_BASELINE.md)

> Historical note
>
> Several files in this repo still preserve the earlier pre-audit autonomous-loop story. Those materials are kept for reference, but they do not describe the current fixed-engine setup. After the March 2026 engine audit, realistic hourly Sharpe on this harness is usually in the low single digits, not the 20+ range quoted in older materials.

## Quick Start

### Prerequisites

- Python 3.10+
- `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```bash
git clone https://github.com/Nunchi-trade/auto-researchtrading.git
cd auto-researchtrading
uv run prepare.py
uv run prepare.py --mode 15m
```

1h data is cached to `~/.cache/autotrader/data/*_1h.parquet`.
15m data is cached to `~/.cache/autotrader/data/*_15m.parquet`.

## Main Commands

```bash
# Validation backtest on the default 1h path
uv run backtest.py --timeframe 1h

# 4h robustness pass with optional fee/capacity sweeps
uv run backtest.py --timeframe 4h --stress-fees --capacity

# Full OOS robustness suite on 2025 data
uv run evaluate.py --timeframe 1h --label exp269

# Clean post-research OOS check without consuming holdout
uv run evaluate.py --timeframe 1h --split 2026q1 --label exp269

# Retrain XGBoost models
uv run train_model.py --timeframe 1h
uv run train_model.py --timeframe 4h
uv run train_model.py --timeframe 15m

# Train a tagged experimental model set with short-confidence plumbing
uv run train_model.py --timeframe 15m --model-set exp_short_platt --short-calibration-mode platt --train-bear-short-meta

# Run the shadow paper-trading engine with persisted state
uv run shadow_trade.py --timeframe 1h --split 2026q1 --state-path shadow_state.json --log-path shadow_trade_log.jsonl --max-days 5

# Retrain the macro HMM as well
uv run train_model.py --timeframe 1h --train_hmm

# Run reference benchmarks
uv run run_benchmarks.py

# Refresh research memory from results.tsv
uv run analyze_results.py --update-memory

# Verify harness assresearch_loop.pyumptions before a research session
uv run verify_harness.py

```

There are no unit tests or CI pipelines. Validation is done through `backtest.py`, `evaluate.py`, and benchmark comparison.

## Current Setup

### Data Universe

- 1h universe: 17 symbols in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L38)
  - BTC, ETH, SOL, BNB, XRP, ADA, DOGE, LINK, AVAX, DOT, ATOM, NEAR, UNI, APT, SUI, XAU, SP500
- 15m universe: 16 symbols in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L44)
  - Same set minus SP500
- 1h candles come from CryptoCompare with Hyperliquid fallback; funding comes from Binance with Hyperliquid fallback in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L457)
- 15m candles come from Binance spot plus a HuggingFace XAU source in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L610)

### Splits

The fixed date windows come from [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L61):

- `train`: 2017-01-01 to 2022-06-30, with per-symbol start floors
- `val`: 2022-07-01 to 2024-06-30
- `robustness`: 2018-01-01 to 2024-06-30
- `oos`: 2025-01-01 to 2025-12-31

### Models

- Feature generation happens in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L146)
- Training is XGBoost based, not RandomForest, in [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L21)
- The repo also trains and uses a macro HMM regime model in [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L57)
- Current model artifacts and loading behavior are documented in [models/MODELS.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/models/MODELS.md)
- The default control model pin is `exp256_active`

### Strategy Runtime

The live strategy is [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L19).

Its runtime flow is:

1. Load model artifacts in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L89)
2. Build per-symbol prediction tables with engineered features and HMM states in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L127)
3. Warm timeframe-aligned caches in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L271)
4. On each bar, combine 1h directional outputs with 1h/4h meta gating, HMM-aware sizing, ATR-based exits, and a ranked entry allocator in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L395)

The default research path is the 1h strategy plus OOS evaluation with `evaluate.py`.

## Validation Lens

### Score

The score is computed in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L1138):

```text
score = sharpe * sqrt(min(num_trades / 50, 1.0)) - drawdown_penalty - turnover_penalty
```

Hard cutoffs send the score to `-999` when:

- fewer than 10 trades
- fewer than 1 trade per day
- max drawdown above 50%
- final equity below 50% of initial capital

### Benchmark Audit

`backtest.py` and `evaluate.py` use the same benchmark lens from [backtest.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/backtest.py#L17) and [evaluate.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/evaluate.py#L20):

- Sharpe >= 3.5
- Win rate >= 60%
- Trades/day >= 1.0
- Profit factor >= 4.0
- Max drawdown < 10%

These are audit thresholds, not guarantees that a branch is good enough to ship.

### Institutional Metrics

`evaluate.py` now also reports:

- annualized Sortino
- beta to `SP500` using the parquet benchmark when present, with daily FRED fallback
- annualized alpha vs `SP500`
- excess return vs `SP500`
- lane attribution by entry tag
- confidence / structure / hold-time bucket diagnostics
- short-confidence base-rate vs realized bear-trade distribution
- event-conditioned attribution using deterministic context features

## Workflow

For day-to-day research:

1. Refresh [RESEARCH_MEMORY.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/RESEARCH_MEMORY.md) with `uv run analyze_results.py --update-memory`
2. Run `uv run verify_harness.py` to catch drift in artifacts, dependencies, or cache assumptions
3. Refresh data if needed with `prepare.py`
4. Retrain models if the experiment needs new artifacts
5. Change `strategy.py`
6. Run `uv run backtest.py --timeframe 1h --no-log`
7. Run `uv run evaluate.py --timeframe 1h --label <label>`
   - optional clean post-research check: `uv run evaluate.py --timeframe 1h --split 2026q1 --label <label>`
8. Compare against `run_benchmarks.py`

`research_loop.py` is retained as historical automation infrastructure, but the active workflow is manual experimentation plus explicit `results.tsv` logging.

`backtest.py` also appends a summary row to `results.tsv`.

The new research infra is:

- [analyze_results.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/analyze_results.py): summarizes `results.tsv`, detects plateau risk, and refreshes [RESEARCH_MEMORY.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/RESEARCH_MEMORY.md)
- [verify_harness.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/verify_harness.py): checks schema, dependency drift, model-path drift, and cache presence
- [research_loop.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/research_loop.py): runs verifier, backtest, memory refresh, and optional OOS evaluation as one cycle
- [RESEARCH_MEMORY.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/RESEARCH_MEMORY.md): the lightweight guidance layer the AI researcher should read before each mutation
- [POSITIONING.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/POSITIONING.md): exact lane and non-goals for the project
- [CONTROL_BASELINE.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/CONTROL_BASELINE.md): frozen control definition and model-pin rules

Reference snapshots for comparison:

- [strategy_exp269.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy_exp269.py)
- [strategy_diff.txt](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy_diff.txt)

## Project Structure

```text
├── strategy.py
├── strategy_exp269.py
├── strategy_diff.txt
├── prepare.py
├── backtest.py
├── evaluate.py
├── train_model.py
├── run_benchmarks.py
├── benchmarks/
├── models/
├── results.tsv
├── program.md
├── CLAUDE.md
├── STRATEGIES.md
├── POST.md
├── TWITTER_THREAD.md
└── pyproject.toml
```

## Historical Documents

These files are kept as archive material and should be read as historical context, not as the source of truth for the current branch:

- [POST.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/POST.md)
- [TWITTER_THREAD.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/TWITTER_THREAD.md)

The current source of truth is the code plus:

- [program.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/program.md)
- [CLAUDE.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/CLAUDE.md)
- [.github/copilot-instructions.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/.github/copilot-instructions.md)
- [models/MODELS.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/models/MODELS.md)

## Attribution

Built on the autoresearch pattern popularized by Karpathy-style experiment loops. Market data comes from CryptoCompare, Binance, Hyperliquid, and the XAU source linked in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L58).

<p align="center">
  <sub>Built by <a href="https://nunchi.trade">Nunchi</a> • MIT License</sub>
</p>
