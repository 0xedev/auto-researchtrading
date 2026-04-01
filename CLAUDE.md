# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nunchi Auto-Research Trading: an autonomous strategy research framework where Claude iteratively evolves trading strategies on Hyperliquid perpetual futures. The system modifies `strategy.py`, backtests it, keeps improvements, and reverts failures — fully autonomous (Baseline Sharpe ~1.0-2.1 on 18 assets).

## Commands

```bash
# Download/refresh 1H market data (cached to ~/.cache/autotrader/data/)
uv run prepare.py

# Download 15-minute data (Binance + HuggingFace XAU)
uv run prepare.py --mode 15m

# Train ML models (per-timeframe) & Run Monte Carlo tests
uv run train_model.py                    # 1H model → model_1h.joblib
uv run train_model.py --timeframe 4h     # 4H model → model_4h.joblib
uv run train_model.py --timeframe 4h --monte-carlo 1000  # 1,000 parallel GPU simulations

# Run baseline validation backtest (1H bars)
uv run backtest.py --timeframe 1h

# Run robustness test (4H bars, 2004–2024, 0.15% RT costs, regime breakdown)
uv run backtest.py --timeframe 4h

# Run Fee Stress Test & Capacity Constraints
uv run backtest.py --timeframe 4h --stress-fees --capacity --oos
```

There are no tests, linters, or CI/CD pipelines. Validation is done entirely through backtest scores.

## Architecture

### Immutable vs Mutable Boundary

**Only `strategy.py` is mutable during experiments.** Everything else — especially `prepare.py` (backtesting engine, ~2100 lines) — is fixed infrastructure.

### Data Flow

1. `prepare.py` downloads and caches OHLCV + funding rate data for 17 assets (15 crypto, 2 macro: XAU, SP500)
   - 1H data: CryptoCompare candles + Binance/Hyperliquid funding rates
   - 15m data: Binance spot klines (16 symbols, no SP500/DXY) + HuggingFace XAU
2. `train_model.py --timeframe {1h,4h,15m}` trains per-timeframe RandomForest models
3. `backtest.py --timeframe 1h` calls `prepare.load_data(split="val")` then `prepare.run_backtest(strategy, data)`
4. `backtest.py --timeframe 4h` runs robustness test: 4H bars, full history, 0.15% RT costs, per-year regime breakdown, with `--stress-fees` and `--capacity` matrix options.
5. The engine iterates bars, calling `strategy.on_bar(bar_data, portfolio) → List[Signal]` each step
6. Results: Sharpe, total return, max drawdown, trade count, win rate, profit factor, turnover, i want at least 2 trades per day.

### Multi-Timeframe Architecture

- **15m model** (`model_15m.joblib`): Intraday momentum, highest signal frequency
- **1H model** (`model_1h.joblib`): Swing trades, baseline timeframe
- **4H model** (`model_4h.joblib`): Trend-level alpha, highest conviction
- Same 13 features across all timeframes — bar-count periods map to different real-time windows
- Each model learns its own timeframe's statistical distribution

### Strategy Interface

Every strategy must implement:

```python
class Strategy:
    def __init__(self): ...
    def on_bar(self, bar_data: dict, portfolio: PortfolioState) -> list[Signal]:
        # bar_data: dict[symbol → BarData] with .close, .open, .high, .low, .volume, .funding_rate, .history (last 500 bars)
        # portfolio: .cash, .positions, .entry_prices, .equity
        # Returns: list of Signal(symbol, target_position)
```

### Scoring Function

```
score = sharpe × √(trade_count_factor) - drawdown_penalty - turnover_penalty
```

Hard cutoffs (score → -999): <10 trades, >50% drawdown, final equity <50% of initial.

### Key Constants (prepare.py)

- Initial capital: $100K, Max leverage: 20x
- Fees (1H harness): 2 bps maker, 5 bps taker, 1 bps slippage
- Fees (robustness/4H): 7.5 bps per side (0.15% RT), no slippage
- Backtest time budget: 120s (1H), 300s (robustness)
- History buffer: 500 bars per `on_bar` call
- 17 symbols: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, LINK, AVAX, DOT, ATOM, NEAR, UNI, APT, SUI, XAU, SP500
- 16 symbols (15m): Same minus SP500 (no free 15m source)
- Train period: 2004-01-01 to 2024-06-30
- Validation period: 2024-07-01 to 2025-03-31
- Robustness period: 2004-01-01 to 2025-03-31

### Benchmarks (benchmarks/)

Five reference strategies: `regime_mm.py`, `avellaneda_mm.py`, `funding_arb.py`, `mean_reversion.py`, `momentum_breakout.py`. All implement the same Strategy interface.

## Autonomous Experiment Loop

The workflow (described in `program.md`) is: edit `strategy.py` → commit → `uv run backtest.py` → if score improved keep, else `git reset`. Each experiment is one atomic commit (e.g., `exp251: RSI exit 69/31`). The evolution log lives in `STRATEGIES.md`.

## Key Lessons

- Removing complexity often improved performance (the "Great Simplification")
- RSI period 8 beats standard 14 for hourly crypto
- Uniform position sizing outperformed momentum-weighted
- Wider trailing stops (5.5x ATR vs 3.5x) improved Sharpe significantly
- Single-timeframe models produce too few trades (41-77) — multi-timeframe ensemble needed
- RandomForest reduced symbol_idx dominance (32% vs 56% XGBoost)
- Must validate on 2+ years across bull/bear/sideways to avoid overfitting to a single regime
- Funding rates: Binance has deepest history (2019+), not needed for XAU/SP500

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder or check arxiv for papers. The loop runs until interrupted. Focus on discovering robust, non-overfit alpha across multiple regimes and timeframes.
