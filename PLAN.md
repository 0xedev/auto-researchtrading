# V2 Renaissance-Style Multi-Alpha Platform

## Current Stage
As of `2026-04-24`, this plan is in the **reality-check and V2 rebuild stage**.

Production readiness rating:
- **Current production readiness:** `3/10`
- **Research platform maturity:** `5-7/10`, depending on whether we mean infrastructure or validated alpha
- **Reason:** the system now has serious research and replay infrastructure, but the latest corrected V2 candidate still fails production gates on PF, concentration, win rate, clean OOS discipline, and live-shadow proof.

## Production Checklist
This is the checklist that must be green before V2 can be considered production-ready. The current goal is **paper-production readiness first**, not real-capital autonomy.

### A. Strategy Quality
- [x] Legacy benchmark exists and is tracked in `results.tsv`.
- [x] V2 multi-sleeve framework exists beside the legacy control.
- [x] V2 has multiple candidate sleeves, not only one dominant strategy idea.
- [x] V2 remains directionally profitable after replay hardening.
- [x] Corrected ablations identify which sleeves are real pillars versus weak support.
- [ ] V2 has `5-8` genuinely live-quality independent sleeves.
- [ ] No sleeve contributes more than `30%` of PnL on full corrected `val` and full corrected `2026q1`.
- [ ] V2 clears PF target on corrected base replay.
- [ ] V2 clears PF target on corrected stress replay.
- [ ] V2 clears win-rate or equivalent payoff-quality target after fees, slippage, funding, and partial reductions.
- [ ] V2 beats the legacy line on an apples-to-apples replacement standard.
- [ ] V2 survives a fresh OOS split that was not consumed during wave selection.
- [ ] Final holdout remains untouched until explicit final sign-off.

### B. Replay And Backtest Realism
- [x] V2 has full-period benchmark replay via `v2_backtest.py`.
- [x] V2 uses daily Sharpe and daily Sortino as credibility checks.
- [x] V2 no longer relies only on short probe windows.
- [x] V2 max-hold exits now use real millisecond-aware clock timing.
- [x] V2 max-hold exits now pay slippage and taker fees.
- [x] V2 applies base-time funding drag in replay.
- [x] V2 PF and win-rate now use net realized PnL including allocated entry fees and partial reductions.
- [x] V2 replay uses timestamp-indexed lookups instead of repeated per-bar DataFrame filtering.
- [ ] V2 has nonlinear market-impact modeling, not only fixed bps slippage plus participation rejection.
- [ ] V2 has borrow/short financing assumptions beyond funding-rate cashflows.
- [ ] V2 reports bar Sharpe, daily Sharpe, Sortino, PF, win rate, DD, turnover, trade density, and concentration in a consistent results schema.
- [ ] V2 results columns in `results.tsv` are clearly separated from legacy columns where metrics are not directly comparable.
- [ ] V2 stress tests cover multiple fee/slippage/participation regimes, not one harsh config.

### C. Portfolio Construction
- [x] Portfolio allocator exists and consumes standardized V2 sleeve signals.
- [x] Allocator supports sleeve caps, cluster caps, symbol caps, gross/net exposure limits, tier sizing, and short-share pressure.
- [x] Strategy-family clustering was fixed to use sleeve manifests instead of symbol market buckets.
- [x] Corrected ablations identified `basis_dislocation`, `cross_asset_relative_strength`, and `post_extension_snapback` as the current core sleeve family.
- [x] Weak corrected sleeves `funding_carry` and `trend_1h_directional` were pruned into the wave5 seed.
- [ ] Current wave5 seed still fails concentration target.
- [ ] Current wave5 seed still fails PF target.
- [ ] Mean-reversion sleeves need redesigned hold/exit behavior under real clock timing.
- [ ] Cross-asset and basis sleeves need concentration controls that do not collapse return.
- [ ] Carry sleeves need to be rebuilt under actual funding drag before they can re-enter the book.
- [ ] Portfolio promotion requires sleeve ablation, pairwise ablation, and cluster-level ablation.

### D. Data And Modeling
- [x] The old `15m` train split mismatch was fixed before V2 sleeve work continued.
- [x] Role-based `fast/base/slow` bundle architecture exists.
- [x] Clock-time features exist for V2.
- [x] V2 has `18` manifests live in code: `15` experimental sleeves plus `3` legacy foundation sleeves.
- [x] External context and funding/context features are available in the pipeline.
- [ ] Current V2 model waves have consumed `val` and `2026q1`; they are no longer pristine OOS for V2.
- [ ] Need a fresh clean OOS slice or forward paper period for V2 promotion.
- [ ] Need scheduled champion/challenger retraining instead of ad hoc wave selection.
- [ ] Need feature/sleeve drift diagnostics before paper-live promotion.
- [ ] Need stronger short-side sleeve breadth; current short sleeves are still too thin.
- [ ] Need a second bundle proof, especially `fast=1h, base=4h, slow=1d`, before claiming timeframe portability.

### E. Shadow Execution And Operations
- [x] Shadow runtime exists for V2.
- [x] Shadow state is restart-safe.
- [x] Operator dashboard and JSON summaries exist.
- [x] Kill-switch handling exists.
- [x] Logs include sleeve, bundle, cluster, rationale, fees, realized PnL, and portfolio context.
- [ ] Need a multi-day continuous shadow run for the current corrected seed.
- [ ] Need restart/reconciliation proof on the current corrected seed, not only earlier smoke runs.
- [ ] Need kill-switch, stale-data, duplicate-order, and orphan-position validation in realistic shadow sessions.
- [ ] Need alert thresholds for PF deterioration, sleeve concentration drift, funding drag, and rejection spikes.
- [ ] Need daily operator report comparing expected replay behavior versus paper-shadow behavior.
- [ ] Need explicit real-capital ban until shadow gates are passed.

### F. Governance And Promotion
- [x] `results.tsv` tracks legacy and V2 experiment history.
- [x] Historical overstated V2 rows were preserved rather than rewritten.
- [x] Corrected follow-up rows `v2exp9` and `v2exp10` document the realism downgrade.
- [x] `PLAN.md` now treats V2 as a rebuild candidate, not a production candidate.
- [ ] Promotion gates must be written as code, not only prose.
- [ ] V2 experiment rows need a richer schema or sidecar JSON so daily Sharpe, PF, concentration, stress metrics, active sleeves, and config path are machine-readable.
- [ ] Need a formal rule for when `2026q1` is considered consumed and which split replaces it.
- [ ] Need a champion registry for portfolio-level candidates, not only sleeve-level candidates.
- [ ] Need a retirement policy for weak sleeves and stale artifacts.

### Done Right
- We did **not** bury the replay bugs. The max-hold timing issue, max-hold friction issue, funding omission, and fee-blind PF accounting were fixed and then rebenchmarked.
- We preserved historical V2 rows in `results.tsv` and added corrected follow-up rows instead of rewriting the past.
- We moved from hypey Sharpe readings to a stricter view using daily metrics, net trade accounting, funding drag, and full-period replay.
- We found that V2 still has real directional edge after correction, but we downgraded the production claim when PF and concentration failed.
- We turned ablation findings into named seed configs rather than leaving them as vague notes:
  - [v2_portfolio.wave5_seed_pruned.json](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/v2_portfolio.wave5_seed_pruned.json)
  - [v2_portfolio.wave5_seed_pruned_stress.json](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/v2_portfolio.wave5_seed_pruned_stress.json)

### Next Production Milestone
The next meaningful milestone is **not live trading**. It is a corrected V2 paper candidate that clears:
- full `val` and full replacement OOS daily Sharpe target
- PF target after net fees, slippage, funding, and partial reductions
- concentration `<=30%`
- stress replay under multiple friction profiles
- multi-day paper-shadow run with restart and kill-switch proof
- no untouched holdout usage before explicit final sign-off

## Current Research State
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
- the latest fully corrected V2 benchmark candidate no longer satisfies the `<=30%` sleeve concentration target, and it also fails the PF gate badly
- V2 still does not use the legacy `backtest.py` engine, but it now has its own benchmark-style replay/backtest path
- most historical V2 reads were still probe or rolling-window evaluations, so the new full-period backtest path now needs to become part of the normal research loop

Current practical stage label:
- **Stage 1 complete:** platform foundation
- **Stage 2 in progress:** sleeve wave build-out and foundation sleeve porting
- **Stage 3 in progress:** portfolio allocator, realism hardening, and concentration reduction
- **Stage 4 partially complete:** shadow execution and evaluation
- **Stage 5 early / partial:** challenger registry and rolling evaluation exist, but promotion cadence and portfolio champion workflow are not mature yet

Latest V2 read worth tracking:
- the big V2 benchmark claims from `v2exp6`-`v2exp8` are no longer trustworthy as stated
  - after fixing three more realism/metric issues:
    - millisecond-aware max-hold timing in [execution/v2_paper.py](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/execution/v2_paper.py)
    - base-time funding drag in V2 replay
    - net realized trade accounting including allocated entry fees and partial reductions
  - the same [v2_portfolio.wave4_quality2.json](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/v2_portfolio.wave4_quality2.json) policy reran much lower on full 8-symbol replay:
    - `val`: legacy score `5.67`, daily Sharpe `5.92`, PF `1.98`, return `473.50%`, trades/day `7.59`, concentration `39.51%`
    - `2026q1`: legacy score `3.32`, daily Sharpe `3.59`, PF `1.49`, return `17.74%`, trades/day `8.69`, concentration `35.93%`
- conclusion:
  - the V2 edge is probably real
  - the old Sharpe/PF story was materially overstated
  - V2 is back to “promising prototype” rather than “near-replacement benchmark”
  - the corrected ablation read is now clearer too:
    - `basis_dislocation` is non-negotiable
    - `cross_asset_relative_strength` is still a real edge sleeve even though it drives concentration
    - `funding_carry` and `trend_1h_directional` are weak enough to prune from the next seed
    - `post_extension_snapback` and `sideways_mean_reversion` still carry edge, but are the main redesign targets for hold/exit logic rather than obvious sleeves to delete

Current leading V2 candidate:
- there is no trustworthy promotion-ready V2 benchmark candidate right now
  - `v2_wave4_carry` / `quality2` is still the best-known policy family structurally
  - but after the latest replay hardening it fails the replacement bar on:
    - PF
    - concentration
    - win rate
- the current best **wave5 seed** is now:
  - [v2_portfolio.wave5_seed_pruned.json](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/v2_portfolio.wave5_seed_pruned.json)
  - this drops `funding_carry` and `trend_1h_directional` from the full wave4 book
  - full corrected base replay:
    - `val`: legacy score `5.73`, daily Sharpe `5.97`, PF `1.99`, return `477.66%`, trades/day `6.99`, concentration `37.16%`
    - `2026q1`: legacy score `3.77`, daily Sharpe `4.01`, PF `1.49`, return `17.98%`, trades/day `7.81`, concentration `41.30%`
  - matching stress seed:
    - [v2_portfolio.wave5_seed_pruned_stress.json](/Users/ayobamiadefolalu/Downloads/auto-researchtrading/v2_portfolio.wave5_seed_pruned_stress.json)
    - `val`: daily Sharpe `5.12`, PF `1.86`, concentration `39.23%`
    - `2026q1`: daily Sharpe `3.10`, PF `1.51`, concentration `35.47%`
- interpretation:
  - `basis_dislocation`, `cross_asset_relative_strength`, and `post_extension_snapback` still look like the main real sleeves
  - `funding_carry` is no longer strong enough to save the book once funding is actually charged
  - the legacy directional foundation sleeve is now mostly ballast / narrative, not a material alpha contributor
  - V2 now needs another true research wave, not just another promotion pass

Useful research note:
- `post_event_mean_reversion` currently looks more like a policy-gated sleeve than a bad model:
  - metadata looked good, but default runtime confidence produced almost no live signals
  - a new V2 confidence-override path now exists so that sleeve can be tested without rewriting manifests
  - the first event-floor test at `0.30` increased live event actions but did not move portfolio-level outcomes materially, so that is not the best next lever right now

Immediate next focus:
- redesign around the corrected reality instead of trying to rescue the old wave4 policy by tiny allocator nudges
- specifically:
  - start from the new wave5 pruned seed instead of the full wave4 book
  - inspect why max-hold-corrected sleeves are still profitable in aggregate but now over-concentrated and low-PF
  - reduce dependence on `cross_asset_relative_strength`, `basis_dislocation`, and `post_extension_snapback`
  - retune hold horizons and sleeve exit behavior now that max-hold uses the real clock
  - rebuild stress-aware carry sleeves with actual funding charged if carry is to remain in V2 at all
- run a longer paper-shadow validation pass only after a single policy clears corrected base and corrected stress bars again
- compare any rebuilt V2 candidate against the legacy control on explicit replacement criteria, not just standalone V2 strength
- keep using the stricter V2 benchmark audit in `v2_evaluate.py`:
  - bar Sharpe
  - win rate
  - profit factor
  - max drawdown
  - trades/day
- keep the confidence-override path available for later sleeve activation work, but do not spend the next loop on `post_event_mean_reversion` unless a better event-side hypothesis appears
- continue using the faster indexed replay path for routine wave work; the main blocker is now policy fit and replay realism, not raw throughput

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
