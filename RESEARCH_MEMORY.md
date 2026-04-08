# Research Memory

Read this before mutating [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py).

This file is the lightweight guidance layer for the AI researcher. Manual notes stay outside the generated block. The generated block is refreshed by `analyze_results.py` and `research_loop.py`.

## Stable Rules

- Treat [prepare.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/prepare.py), [backtest.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/backtest.py), and [evaluate.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/evaluate.py) as the fixed harness unless the task is repository maintenance.
- Prefer small, auditable changes in [strategy.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/strategy.py).
- Let the fixed-engine score, OOS behavior, and benchmark audit decide what survives.
- When the loop starts plateauing, switch from threshold churn to structural changes.

## Human Notes

- `exp340` is the first current-session 15m branch with positive validation and positive 2025 OOS. Treat it as the live control until a better branch beats it on both.
- `exp340` OOS early read: Sharpe about `0.28`, PF about `4.29`, drawdown about `1.8%`, trades/day about `1.24`. The edge is real but modest.
- Main weakness is fee sensitivity, not opportunity starvation. At higher fee multipliers the branch degrades quickly, so the next search should target lower churn or better signal weighting.
- The `push` entry family is much higher quality than the `trend` family in current 15m diagnostics. Favor reallocating risk toward push entries before loosening gates broadly.

<!-- BEGIN AUTO SUMMARY -->
## Auto Summary
- Updated: 2026-04-08 23:19 UTC
- Total logged runs: 291
- Best run: `exp269` | score 3.245 | Medallion Council v1.39: 4-Regime Specialists + HMM calibration + Unified Precision + balanced retraining. Trades unlocked.
- Last run: `exp363` | CANDIDATE | score 0.937 | apr08-vast4090 15m soft slow-tf ranked p3 push_relax push_bias
- Last KEEP: `exp269` | score 3.245 | Medallion Council v1.39: 4-Regime Specialists + HMM calibration + Unified Precision + balanced retraining. Trades unlocked.
- Last promising run: `exp363` | score 0.937 | apr08-vast4090 15m soft slow-tf ranked p3 push_relax push_bias
- Recent keep rate: 0.00% over the last 20 runs
- Recent progress rate: 45.00% over the last 20 runs
- Runs since best: 93
- Suggested mode: `exploit`
- Revert-heavy themes: apr08, vast4090, 15m, campaign, control
- Keep-heavy themes: apr08, vast4090, 15m, push_relax, campaign

## Next Guidance
- Recent keep rate is healthy enough to keep exploiting the current family.
- Bias toward incremental improvements around the current baseline before widening scope.
<!-- END AUTO SUMMARY -->

## Open Hypotheses

- Capture one or two active hypotheses at a time.

## Rejected Families

- Record rejected families in one line each with the reason they failed.
