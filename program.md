# Current Experiment Program

This file is the operational guide for the current branch.

## Ground Truth

- The backtest engine in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py) is the source of truth.
- The engine was audited in March 2026. Old 20+ Sharpe claims elsewhere in the repo are historical and should not be treated as current expectations.
- The live stack is XGBoost plus macro HMM regime tooling, not the older simplified single-file story.

## Current Objective

Improve the fixed-engine score and robustness of [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py) while keeping the workflow reproducible.

The default path is:

1. Prepare data
2. Optionally retrain models
3. Update `strategy.py`
4. Run validation backtests
5. Run OOS robustness checks
6. Compare against benchmarks

## Files That Matter

- [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py): primary experiment surface
- [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py): feature engine, dataset prep, data loading, backtest engine, score function
- [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py): XGBoost and HMM training
- [backtest.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/backtest.py): validation and robustness CLI
- [evaluate.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/evaluate.py): OOS suite with fee stress, capacity, regime breakdown, and Monte Carlo
- [run_benchmarks.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/run_benchmarks.py): benchmark comparison
- [analyze_results.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/analyze_results.py): results analyzer and research-memory refresher
- [verify_harness.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/verify_harness.py): harness drift checker
- [research_loop.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/research_loop.py): guided experiment-cycle runner
- [RESEARCH_MEMORY.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/RESEARCH_MEMORY.md): short-lived strategic memory for the AI researcher
- [models/MODELS.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/models/MODELS.md): current artifact map

Helpful comparison artifacts:

- [strategy_exp269.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy_exp269.py)
- [strategy_diff.txt](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy_diff.txt)

## Current Workflow

### 1. Refresh data when needed

```bash
uv run prepare.py
uv run prepare.py --mode 15m
```

### 2. Retrain models when the experiment requires it

```bash
uv run train_model.py --timeframe 1h
uv run train_model.py --timeframe 4h
uv run train_model.py --timeframe 15m
uv run train_model.py --timeframe 1h --train_hmm
```

### 3. Run validation backtests

```bash
uv run backtest.py --timeframe 1h
uv run backtest.py --timeframe 4h --stress-fees --capacity
uv run backtest.py --timeframe 1h --oos
```

### 4. Refresh research memory and verify the harness

```bash
uv run analyze_results.py --update-memory
uv run verify_harness.py
```

### 5. Run the full OOS suite

```bash
uv run evaluate.py --timeframe 1h --label exp269
```

### 6. Optionally run one guided cycle end to end

```bash
uv run research_loop.py --timeframe 1h --description "test idea"
```

### 7. Compare with benchmarks

```bash
uv run run_benchmarks.py
```

## Rules

- Treat `strategy.py` as the default experiment surface.
- Treat `prepare.py`, `backtest.py`, and `benchmarks/` as fixed infrastructure unless the task is explicitly repository maintenance.
- Do not use old README or social-post metrics as the current baseline.
- Use the fixed-engine score and OOS behavior as the decision rule.
- Prefer small, auditable changes over broad rewrites.

## Current Codebase Facts

### Data windows

From [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L61):

- `train`: 2017-01-01 to 2022-06-30, with per-symbol floors
- `val`: 2022-07-01 to 2024-06-30
- `robustness`: 2018-01-01 to 2024-06-30
- `oos`: 2025-01-01 to 2025-12-31

### Universe

- 17 symbols on 1h data
- 16 symbols on 15m data

### Model stack

- XGBoost directional and meta models in [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L135)
- Macro HMM regime training in [train_model.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/train_model.py#L57)
- Cached probability tables and HMM state mapping in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L127)

### Default strategy behavior

The current `1h` path in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py#L395):

- warms cached predictions before the backtest
- uses 1h directional probabilities as the main entry signal
- blends 1h and 4h meta quality into a single gate
- applies HMM-aware sizing and ATR/ratchet exits
- ranks entry candidates and caps concurrent positions

## Score And Audit

The score in [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py#L1138) is:

```text
score = sharpe * sqrt(min(num_trades / 50, 1.0)) - drawdown_penalty - turnover_penalty
```

Hard cutoffs:

- fewer than 10 trades
- fewer than 1 trade per day
- drawdown above 50%
- final equity below 50% of initial capital

Benchmark audit thresholds from [backtest.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/backtest.py#L17) and [evaluate.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/evaluate.py#L20):

- Sharpe >= 3.5
- Win rate >= 60%
- Trades/day >= 1.0
- Profit factor >= 4.0
- Max drawdown < 10%

## Results Logging

- `backtest.py` appends summary rows to `results.tsv`
- keep experiment labels and descriptions meaningful
- use `evaluate.py` labels to tie OOS results back to a strategy snapshot

## Research Infra

- Refresh [RESEARCH_MEMORY.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/RESEARCH_MEMORY.md) from `results.tsv` before a new mutation.
- Use [analyze_results.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/analyze_results.py) to detect plateau risk, recent keep-rate collapse, and repeated revert-heavy themes.
- Use [verify_harness.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/verify_harness.py) to catch path drift, dependency drift, results-schema drift, and missing cached data before trusting a run.
- Use [research_loop.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/research_loop.py) when you want explicit loop control instead of relying on prompt obedience alone.
- Treat [RESEARCH_MEMORY.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/RESEARCH_MEMORY.md) as the short strategic brief that sits between `program.md` and the next `strategy.py` edit.

## Historical References

These are still useful context, but they are not the current operating manual:

- [STRATEGIES.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/STRATEGIES.md)
- [POST.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/POST.md)
- [TWITTER_THREAD.md](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/TWITTER_THREAD.md)

## NEVER STOP

Once the research loop has begun, do not pause just because a branch is difficult, a result is disappointing, or a recent idea failed.

- Keep iterating until explicitly interrupted by the user.
- Use the current audited workflow: prepare data if needed, retrain only when necessary, update `strategy.py`, run validation backtests, run OOS checks, and compare against benchmarks.
- Refresh `RESEARCH_MEMORY.md` and run `verify_harness.py` so the loop stays grounded in recent evidence and known drift.
- Let the fixed-engine score, OOS behavior, and benchmark audit decide what survives.
- If an experiment fails, learn from it, restore a sane baseline, and continue searching.
- Focus on robust, non-overfit improvements across regimes rather than chasing flashy in-sample numbers.
