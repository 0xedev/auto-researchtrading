# Workspace Instructions

This repository is a trading research workspace with a fixed backtest engine, model-training scripts, and a live strategy implementation.

## Primary guidance

- For strategy experiments, prefer changing `strategy.py`.
- Treat `prepare.py`, `backtest.py`, and `benchmarks/` as fixed infrastructure unless the task explicitly involves maintenance.
- Updating docs is allowed when the request is about documentation or repo cleanup.
- Do not introduce new dependencies unless explicitly requested.
- Validation is by backtest and OOS evaluation, not by unit tests.

## Key files

- `strategy.py` - primary experiment surface
- `prepare.py` - data loading, feature engineering, backtest engine, score function
- `train_model.py` - XGBoost and macro HMM training
- `backtest.py` - validation and robustness runner
- `evaluate.py` - OOS suite with fee stress, capacity, regime breakdown, and Monte Carlo
- `run_benchmarks.py` - reference strategy comparison
- `analyze_results.py` - results analyzer and research-memory refresher
- `verify_harness.py` - harness drift checker
- `research_loop.py` - guided experiment-cycle runner
- `RESEARCH_MEMORY.md` - lightweight guidance layer for the AI researcher
- `models/MODELS.md` - current artifact map
- `program.md` - current workflow and repo facts
- `README.md` - user-facing repo overview

## Recommended workflow

1. Read `program.md`, `RESEARCH_MEMORY.md`, `README.md`, and `models/MODELS.md`.
2. Refresh the memory block with `uv run analyze_results.py --update-memory`.
3. Run `uv run verify_harness.py`.
4. Refresh data if needed with `uv run prepare.py` and `uv run prepare.py --mode 15m`.
5. Retrain models only if the experiment needs new artifacts.
6. Update `strategy.py`.
7. Run `uv run backtest.py --timeframe 1h` or `uv run research_loop.py --timeframe 1h --description "<idea>"`.
8. Run `uv run evaluate.py --timeframe 1h --label <label>`.
9. Compare with `uv run run_benchmarks.py`.

## Important repository context

- The backtest engine was audited in March 2026. Old 20+ Sharpe claims elsewhere in the repo are historical.
- The current stack is XGBoost plus macro HMM regime tooling, not RandomForest.
- The default research path is the 1h strategy with validation in `backtest.py` and OOS checks in `evaluate.py`.
- Scores above 5.0 on the corrected harness should be treated with skepticism and checked carefully for overfitting or data leakage.

## Useful commands

- `uv run prepare.py`
- `uv run prepare.py --mode 15m`
- `uv run train_model.py --timeframe 1h`
- `uv run train_model.py --timeframe 4h`
- `uv run train_model.py --timeframe 15m`
- `uv run train_model.py --timeframe 1h --train_hmm`
- `uv run backtest.py --timeframe 1h`
- `uv run evaluate.py --timeframe 1h --label exp269`
- `uv run run_benchmarks.py`

## Notes for the agent

- The repository is research-first, not production-first.
- Historical posts and threads are archive material; use the code and current docs as the source of truth.
- Keep explanations aligned with the live audited branch, not the earlier autonomous-marketing narrative.
