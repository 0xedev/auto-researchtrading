# autotrader

Autonomous trading strategy research on Hyperliquid perpetual futures.

## 🚨 CRITICAL ENGINE AUDIT (March 2026)

**The evaluation harness (`prepare.py`) was found to have a severe accounting bug (#4)** in its position reversal logic. This bug incorrectly inflated PnL and Sharpe ratios (previously reporting Sharpe > 20).

The engine has been **FIXED**. Real-world Sharpe ratios for these hourly strategies are in the 0.0-3.0 range. Any result higher than 5.0 should be treated with extreme skepticism and checked for overfitting (#2).

## Current Leaderboard (FIXED ENGINE)

Your goal is to discover strategies that beat the robust baseline established on the corrected harness.

## Leaderboard (Adaptive Engine + 18 Symbols)

| Rank | Strategy               | Sharpe | Drawdown | Turnover | Score |
| :--- | :--------------------- | :----- | :------- | :------- | :---- |
| 1    | `regime_mm`            | 2.228  | 4.7%     | 13,451   | 2.228 |
| 2    | `simple_momentum`      | 2.122  | 5.6%     | 1,310    | 2.122 |
| 3    | `xgb_iteration_v1`     | 1.800  | 5.0%     | 1.86M    | 1.800 |
| 4    | `adaptive_ensemble_h1` | 1.054  | 16.3%    | 0.86M    | 0.991 |
| 5    | `xgb_baseline_v0`      | 0.671  | 0.7%     | 417k     | 0.671 |

> [!IMPORTANT]
> **Breakthrough (Iteration 1)**: By adding 48h momentum and 24h RSI features and lowering the entry threshold to 53%, the model achieved a **1.80 Sharpe** with a **67.6% win rate**. It is now significantly outperforming the hardcoded ensemble.

**New Target**: Beat the simple momentum baseline of **2.122**.

## 🏗️ Experimentation Workflow

1. **Modify `strategy.py`** — This is the only mutable file.
2. **Run `uv run backtest.py`** — Evaluates on validation data (Jul 2024 - Mar 2025).
3. **Budget**: 120 seconds. Ensure signal calculations are vectorized for speed.
4. **Scoring**: `score = sharpe * sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autotrader/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: `prepare.py`, `strategy.py`, `backtest.py`, this file.
4. **Verify data exists**: `ls ~/.cache/autotrader/data/`
5. **Initialize results.tsv**: `echo -e "commit\tscore\tsharpe\tmax_dd\tstatus\tdescription" > results.tsv`
6. **Confirm and go**.

## Experimentation

Each experiment runs a backtest . Launch: `uv run backtest.py`.

**What you CAN do:**

- Modify `strategy.py` — this is the only file you edit. Everything is fair game.

**What you CANNOT do:**

- Modify `prepare.py`, `backtest.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at test set data.

**The goal: get the highest `score`.** Higher is better. Baseline is 2.724.

## Output format

```
grep "^score:" run.log
```

## Results TSV

```
commit	score	sharpe	max_dd	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at git state
2. Modify `strategy.py` with an experimental idea
3. git commit
4. `uv run backtest.py > run.log 2>&1`
5. `grep "^score:\|^sharpe:\|^max_drawdown_pct:" run.log`
6. If empty → crashed. `tail -n 50 run.log`, fix or skip.
7. Record in results.tsv (untracked)
8. If score IMPROVED (higher than best so far): keep
9. If score equal or worse: `git reset --hard HEAD~1`

## 🧪 Research Lessons

- **Simplicity Wins**: Complexity often hides overfitting. The original ensemble performed worse on the fixed engine than a simple momentum trend follower.
- **BB Squeeze Alpha**: Bollinger Band width compression is a verified event-driven signal (#2) but works best as a sparse filter, not an always-on voting signal.
- **Turnover Management**: High-frequency flipping leads to massive fee bleed and turnover penalties. Aim for <500x annual turnover.
- **Benchmark Lens Matters**: The built-in `score` only enforces overall trades/day, not trades/day/symbol. Always read the benchmark audit before declaring a strategy "good".
- **Directional 15m Split Helps Quality, Not Frequency**: Separate long/short 15m lead+meta models improved win rate and profit factor, but did not solve the opportunity-count bottleneck by themselves.
- **Current Safe Restore Point**: The exp35-style 15m shell is the last trustworthy live baseline. When structural experiments fail, restore to that before continuing.
- **Dense Labels Unlock Frequency but Can Destroy Alpha**: Moving 15m labels from sparse outliers to dense normalized thresholds made frequency achievable, but the first dense-label branches produced severe fee bleed, poor PF, and catastrophic drawdowns.
- **Medium-Density Labels Were Too Sparse**: Raising the 15m normalized threshold to `0.70` improved selectivity, but starved the model again unless thresholds were relaxed so much that quality collapsed.
- **Relative-Strength Features Alone Were Not Enough**: Adding cross-sectional features like `rel_ret_1h`, `rel_ret_4h`, and `rel_bb_width` improved some validation diagnostics, but did not automatically translate into a profitable live backtest.
- **Mixed-Generation Model Inputs Need Alignment**: When feature sets change, older 1h/4h models will crash or silently misbehave unless prediction inputs are reindexed to each model's trained feature names.
- **Ranker / Top-N Allocator Family Is Rejected For Now**: Converting 15m signals into a cross-sectional ranker solved the frequency target too aggressively, but every tested version overtraded and blew up quality. Do not repeat raw top-N 15m ranking without a fundamentally different gating layer.
- **Cooldowns Alone Do Not Save Bad Entry Quality**: Adding top-1 picks, cooldowns, and longer holds reduced churn, but still left PF sub-1 and drawdowns above 90% in the ranker branch.
- **Paper-Driven Next Step**: The next structural branch should be regime-conditioned 15m models and/or event-driven sampling, not more threshold sweeps or more ranker tuning.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder. The loop runs until interrupted. Focus on discovering robust, non-overfit alpha across multiple regimes.
