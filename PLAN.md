# V2 Renaissance-Style Multi-Alpha Platform

## Current Stage
As of `2026-04-22`, this plan is in the **portfolio proving and diversification stage**.

What is already true:
- the parallel V2 platform exists beside the frozen `exp494` control
- role-based `fast/base/slow` bundles are implemented
- V2 training, audit, evaluation, registry, allocator, and shadow runtime all exist
- V2 now also has a benchmark-style full-period backtest path via `v2_backtest.py`
- the `15m` split mismatch was fixed for the old trainer before V2 sleeve work continued
- V2 now has `18` manifests live in code:
  - `15` experimental sleeves from the plan source list
  - `3` legacy foundation sleeves currently sourced from the root artifact set by default:
    - `trend_1h_directional`
    - `bear_1h_calibrated`
    - `bear_1h_foundation`

What is not true yet:
- we do **not** have `5-8` live-quality independent sleeves
- the current lead candidate does satisfy the `<=30%` sleeve concentration target on the base full splits, but that has **not** yet been confirmed under stress / harsher replay assumptions
- V2 still does not use the legacy `backtest.py` engine, but it now has its own benchmark-style replay/backtest path
- most historical V2 reads were still probe or rolling-window evaluations, so the new full-period backtest path now needs to become part of the normal research loop

Current practical stage label:
- **Stage 1 complete:** platform foundation
- **Stage 2 in progress:** sleeve wave build-out and foundation sleeve porting
- **Stage 3 in progress:** portfolio allocator and concentration reduction
- **Stage 4 partially complete:** shadow execution and evaluation
- **Stage 5 early / partial:** challenger registry and rolling evaluation exist, but promotion cadence and portfolio champion workflow are not mature yet

Latest V2 read worth tracking:
- full 8-symbol benchmark-style replay on `v2_wave4_carry` with the small cross-asset trim policy in [v2_portfolio.wave4_trim_rs.json](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/v2_portfolio.wave4_trim_rs.json):
  - `val`: return `1042.89%`, daily Sharpe `11.57`, trades/day `15.93`, short share `30.96%`, top sleeve concentration `29.50%`
  - `2026q1`: return `40.05%`, daily Sharpe `13.39`, trades/day `18.85`, short share `29.99%`, top sleeve concentration `26.83%`
- conclusion: V2 now has its first full-split portfolio that clears the `<=30%` sleeve concentration target on both `val` and `2026q1`; the next blocker is no longer diversification, but realism/stress confirmation

Current leading V2 candidate:
- `v2_wave4_carry` with the small `cross_asset_relative_strength` trim from [v2_portfolio.wave4_trim_rs.json](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/v2_portfolio.wave4_trim_rs.json)
  - adds `funding_carry` and `post_event_mean_reversion` on top of the family-cluster-fixed `v2_wave3_orth`, then lightly trims cross-asset concentration in the allocator
  - full 8-symbol replay:
    - `val`: legacy score `15.06`, daily Sharpe `11.57`, return `1042.89%`, trades/day `15.93`, short share `30.96%`, top sleeve concentration `29.50%`
    - `2026q1`: legacy score `17.73`, daily Sharpe `13.39`, return `40.05%`, trades/day `18.85`, short share `29.99%`, top sleeve concentration `26.83%`
- interpretation:
  - `funding_carry` proved to be the real new contributor; it added a meaningful fourth pillar while `post_event_mean_reversion` stayed nearly inert at default confidence
  - the tiny cross-asset trim was enough to convert wave4 from “almost there” into the first V2 portfolio that passes the concentration target on both full splits
  - it is still not a V2 champion because replay realism and stress still need confirmation, but the alpha-breadth part of the plan is now materially closer to done

Useful research note:
- `post_event_mean_reversion` currently looks more like a policy-gated sleeve than a bad model:
  - metadata looked good, but default runtime confidence produced almost no live signals
  - a new V2 confidence-override path now exists so that sleeve can be tested without rewriting manifests
  - the first event-floor test at `0.30` increased live event actions but did not move portfolio-level outcomes materially, so that is not the best next lever right now

Immediate next focus:
- stress-test the trimmed `v2_wave4_carry` benchmark so the new breadth survives a less optimistic replay lens
- use the stricter V2 benchmark audit in `v2_evaluate.py`:
  - bar Sharpe
  - win rate
  - profit factor
  - max drawdown
  - trades/day
  instead of the older raw-return-only promotion lens
- compare the trimmed wave4 benchmark against the current realism gates and shadow/capacity assumptions before any promotion language gets stronger
- keep the confidence-override path available for later sleeve activation work, but do not spend the next loop on `post_event_mean_reversion` unless a better event-side hypothesis appears
- if full 8-symbol stress replay remains too slow for routine research, add a faster dedicated stress harness rather than relying on the heaviest end-to-end evaluator path every time

## Summary
Build a **parallel V2 platform** beside the frozen `exp494` control. The goal is not “one better strategy,” but a **research-and-allocation machine** that manages **12-20 candidate alphas**, promotes **5-8 live sleeves**, supports **role-based timeframe bundles**, and compounds many weak-to-medium edges under strict risk and anti-overfitting controls.

Default targets for V2:
- `12-20` candidate alphas in research
- `5-8` active sleeves in the live paper portfolio
- no sleeve contributes more than `30%` of portfolio PnL over validation + `2026q1`
- first production-quality density target: `5-15` trades/day portfolio-wide
- second-stage density target after allocator maturity: `15-30` trades/day
- first market scope: **crypto primary + macro/cross-asset context**
- first supported bundles:
  - `fast=15m, base=1h, slow=4h`
  - `fast=1h, base=4h, slow=1d`

## Implementation Changes
### 1. Create a new V2 platform layer
Status: **mostly complete**
- Build V2 beside the current shell; do not rewrite `exp494` in place.
- Replace hardcoded `15m/1h/4h` assumptions with a `bundle` config using `fast`, `base`, and `slow` roles.
- Convert feature lookbacks, label horizons, decay ages, and hold limits from bar-count semantics to clock-time semantics.
- Standardize model outputs into one signal interface:
  - `bundle`
  - `sleeve`
  - `symbol`
  - `side`
  - `confidence`
  - `expected_edge_bps`
  - `holding_horizon_hours`
  - `stop_distance`
  - `target_notional`
  - `regime_context`
  - `reason_tag`
- Fix the current 15m training-path mismatch before V2 alpha work starts, so native fast-timeframe training is trustworthy.

### 2. Build the first 14 candidate alpha sleeves
Status: **in progress**
Implement these as independent sleeves with their own labels, features, diagnostics, and artifacts:

- Trend family:
  - `trend_breakout`
  - `trend_pullback`
  - `trend_continuation`
- Mean-reversion family:
  - `sideways_mean_reversion`
  - `post_extension_snapback`
  - `volatility_reversion`
- Carry / basis family:
  - `funding_carry`
  - `basis_dislocation`
- Relative / cross-asset family:
  - `cross_asset_relative_strength`
  - `leader_laggard_rotation`
  - `macro_beta_dispersion`
- Event family:
  - `macro_event_drift`
  - `post_event_mean_reversion`
- Short family:
  - `bear_stress_short`
  - `squeeze_failure_short`

Defaults:
- Use XGBoost as the primary model family for every sleeve in V2 phase 1.
- Keep HMM as regime/context input only.
- Keep structured data to market + macro + funding/basis + structured event/sentiment features; do not make free-form LLM output part of trade decisions.

Current note:
- this section is now beyond the original wording: there are currently `15` experimental sleeves plus `3` foundation sleeves in code
- the main blocker is no longer sleeve existence; it is sleeve independence and OOS contribution

### 3. Add a portfolio allocator instead of one dominant lane
Status: **in progress**
- Stop promoting raw signals directly into trades.
- Add a portfolio allocator that ranks all sleeve signals each bar by:
  - expected edge
  - confidence
  - sleeve diversification
  - correlation clustering
  - regime fit
  - current exposure budget
- Introduce sizing tiers:
  - `Tier A`: full size
  - `Tier B`: half size
  - `Tier C`: monitor-only or reject
- Enforce portfolio limits:
  - max gross exposure
  - max net exposure
  - per-sleeve cap
  - per-symbol cap
  - correlated-cluster cap
  - short-side minimum activity floor once short sleeves are live
- Add sleeve concentration reporting and automatic rejection of sleeves that are inert, duplicative, or dominate too much of the portfolio.

Current note:
- allocator, sleeve caps, cluster caps, and concentration reporting exist
- exposure-aware scoring and lightweight cooldown/weight scaffolding exist
- the unresolved problem is still concentration, not missing allocator code

### 4. Upgrade execution, capacity, and shadow operations to portfolio-grade
Status: **partially complete**
- Extend shadow execution to consume the standardized V2 signal interface and allocator decisions.
- Add portfolio-state reconciliation, sleeve-level exposure tracking, and capital allocation logs.
- Add capacity modeling to every evaluation:
  - spread/slippage stress
  - participation-rate assumptions
  - funding drag
  - concentration penalties
- Require shadow-readiness before promotion:
  - restart-safe paper runs
  - kill-switch enforcement
  - no duplicate orders
  - stable state reconciliation
  - rationale-rich operator dashboard and machine-readable summaries

Current note:
- V2 shadow runtime is real and restart-safe
- sleeve/cluster attribution and operator summaries exist
- evaluation supports stress mode
- benchmark-grade full-period V2 backtesting now exists via `v2_backtest.py`
- daily-level V2 metrics are now part of the benchmark lens; bar-level Sharpe remains useful but should not be over-interpreted on short probes
- what is still missing is broader use of that path in routine V2 research plus richer capacity/fee stress parity with the legacy evaluation stack

### 5. Add continuous adaptation without uncontrolled overfitting
Status: **early / partial**
- Use champion/challenger promotion at the **sleeve** level and the **portfolio** level.
- Retrain on a schedule, not ad hoc:
  - weekly data refresh
  - monthly challenger generation
  - quarterly sleeve retirement review
- Use rolling walk-forward training with fixed clean OOS gates and untouched holdout.
- Promotion rules:
  - challenger sleeve must improve validation and `2026q1`
  - challenger must not worsen capacity, concentration, or leakage metrics
  - sleeves can be promoted, demoted, or retired independently
- Maintain a sleeve registry with status:
  - `candidate`
  - `paper-live`
  - `live-champion`
  - `retired`

Current note:
- sleeve registry, candidate evaluation, and rolling-window reads exist
- there is not yet a mature monthly challenger flow, portfolio champion workflow, or reliable sleeve retirement cadence

## Public Interfaces / Config
- Add a `bundle manifest` defining `fast/base/slow` timeframes, clock-time label horizons, and feature horizon settings.
- Add a `sleeve manifest` defining sleeve family, feature profile, target construction, allowed sides, and promotion thresholds.
- Extend training/evaluation/runtime CLIs to accept:
  - `--bundle`
  - `--sleeve`
  - `--model-set`
  - `--portfolio-config`
- Standardize outputs across training, evaluation, and shadow execution so every sleeve produces comparable diagnostics and allocator inputs.

## Test Plan
- Bundle correctness:
  - each bundle uses the correct native timeframe data and split mapping
  - clock-time features/labels preserve meaning across bundles
- Sleeve correctness:
  - every sleeve emits attribution, leakage, and concentration metrics
  - no sleeve may be promoted if inert or strongly duplicative
- Portfolio correctness:
  - portfolio reaches `5-15` trades/day in stage 1 without PF/DD collapse
  - no sleeve exceeds the `30%` PnL concentration cap
  - short-side sleeves increase bear participation without sideways-short leakage
- Execution correctness:
  - multi-day shadow runs pass restart, reconciliation, kill-switch, and summary generation checks
  - capacity and slippage stress remain within predefined promotion bounds
- Anti-overfitting correctness:
  - every promoted sleeve beats the current champion on validation and `2026q1`
  - final holdout remains untouched until explicit sign-off

## Assumptions and Defaults
- The current `exp494` shell remains the frozen benchmark and audit fallback.
- V2 is a **parallel platform**, not an in-place rewrite.
- The first serious target is a **Renaissance-style research machine**, not immediate live autonomy with real capital.
- Market scope stays **crypto plus macro/cross-asset context** for phase 1.
- Data scope is **market + macro + structured events/sentiment**, not broad alt-data from day one.
- The goal is many independent edges under portfolio construction, not one universal model.
