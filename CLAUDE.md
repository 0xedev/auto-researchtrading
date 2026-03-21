# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run backtest.py           # run one backtest on val data (prints score/sharpe/drawdown)
uv run prepare.py            # download BTC/ETH/SOL hourly data (one-time, ~1 min)
uv run run_benchmarks.py     # compare 5 reference strategies
uv run train.py              # train ML model on GPU → saves ~/.cache/autotrader/model.pt
uv run prepare_extended.py   # download macro daily data (requires TWELVE_DATA_API_KEY)
uv run fetch_docs.py         # download 56 trading PDFs from tradebridge/DOCs on GitHub
uv run extract_knowledge.py  # extract trading insights from PDFs → docs/knowledge.json (requires GOOGLE_API_KEY)
```

Output is grep-able:
```bash
uv run backtest.py | grep "^score:\|^sharpe:\|^max_drawdown"
uv run train.py | grep "^score:\|^sharpe:\|^val_loss:"
```

## Architecture

Two separate systems sharing `strategy.py` as a bridge:

### 1. Rule-based experiment loop (original, 103 experiments)
- **Only `strategy.py` is mutable.** `prepare.py` and `backtest.py` are the fixed harness.
- `backtest.py` calls `strategy.Strategy().on_bar(bar_data, portfolio)` and scores on val data (Jul 2024–Mar 2025).
- `prepare.py` contains the data pipeline AND the backtest engine AND the scoring formula — all locked.
- Scoring: `score = sharpe * sqrt(min(trades/50, 1)) - drawdown_penalty - turnover_penalty`. Hard cutoffs: <10 trades or >50% drawdown → -999.
- Best known score: **20.634** (branch `autotrader/mar10c`), Sharpe 20.634, max drawdown 0.3%.

### 2. ML system (new, GPU-native)
- `train.py` — agent-mutable Transformer/LSTM. Trains on BTC/ETH/SOL hourly features → saves `model.pt`.
- `features.py` — 128 features per bar (N_FEATURES = 128). Shared by train.py and strategy.py. Groups: A (proven signals), B (technical indicators), C (cross-asset macro), D (temporal), E (statistical).
- `prepare_extended.py` — Twelve Data API pipeline for GOLD, SILVER, OIL, SPX, DXY, TLT daily data (macro context features only, not traded).
- `strategy.py` loads `model.pt` at init if it exists; falls back to rule-based Exp32 logic if not.
- ML path: `compute_feature_vector(bar_data)` → Transformer → predicted return weight in [-1,1] → Kelly-sized position.

### Data flow
```
CryptoCompare + Hyperliquid → prepare.py → ~/.cache/autotrader/data/{BTC,ETH,SOL}_1h.parquet
Twelve Data API             → prepare_extended.py → ~/.cache/autotrader/data/{GOLD,OIL,...}_1d.parquet
                                                              ↓
                                                         features.py (128 features)
                                                              ↓
                                                         train.py → model.pt
                                                              ↓
strategy.py (ML or rule-based) ← backtest.py (fixed harness, val period only)
```

## Key constraints

- **Never modify** `prepare.py`, `backtest.py`, or anything in `benchmarks/`. These are the fixed eval harness.
- `strategy.py`, `train.py`, `features.py` are the only mutable files in the experiment loop.
- `N_FEATURES = 128` must be identical in `features.py` and `train.py`. Mismatch causes silent shape errors.
- Val period is hard-coded in `prepare.py`: Jul 2024 – Mar 2025. Training uses Jun 2023 – Jun 2024.
- `strategy.py` must import only from `prepare` (for `Signal`, `PortfolioState`, `BarData`) plus optional torch/features.

## Experiment loop pattern

```
1. Modify train.py (architecture, loss, hyperparameters) or strategy.py
2. git commit
3. uv run train.py > run.log 2>&1
4. uv run backtest.py >> run.log 2>&1
5. grep "^score:" run.log
6. If score improved: keep. Else: git reset --hard HEAD~1
```

Record results in `results.tsv`: `commit\tscore\tsharpe\tmax_dd\tstatus\tdescription`

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `TWELVE_DATA_API_KEY` | Optional | Macro daily data (GOLD, OIL, SPX, DXY, TLT). Without it, macro features are zeros. |
| `GOOGLE_API_KEY` | Optional | `extract_knowledge.py` — PDF extraction via Gemini Flash. Not needed for training. |

## ML model details

- Architecture options: `transformer` (default), `lstm`, `cnn_transformer` — set `ARCH` in `train.py`
- Loss: Sharpe loss (directly maximizes risk-adjusted return, not directional accuracy)
- Saved to `~/.cache/autotrader/model.pt` as a checkpoint dict with `arch`, `state_dict`, `n_features`, `val_sharpe`
- Precision: `bf16` on A100+, `fp16` on RTX 30xx/40xx, `fp32` on CPU/MPS
- `torch.compile` enabled by default (`COMPILE_MODEL = True`) for ~30% speedup on CUDA
