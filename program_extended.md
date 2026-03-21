# autotrader — Extended ML System (Karpathy-Aligned)

Autonomous trading research using GPU-native deep learning.
The agent reads academic research, implements it in `train.py`, trains on GPU,
backtests, and keeps improvements. No human in the loop.

---

## Architecture

```
PDFs (64 local + AlphaXiv live) → features.py (~128 features, pure numpy)
Daily data  (2008+, 7 symbols)  ─┐
Hourly data (2yr, 10 symbols)   ─┴→ train.py (Transformer/LSTM on GPU)
                                       ↓
                                   model.pt (~/.cache/autotrader/)
                                       ↓
                                   strategy.py (ML inference + Kelly sizing)
                                       ↓
                                   backtest.py → score
                                       ↑ keep / git reset --hard HEAD~1
                                   Claude modifies train.py
```

---

## Mutable Files (agent modifies these)

| File | Role | What to change |
|------|------|----------------|
| `train.py` | Model architecture, loss, training loop | Architecture, hyperparameters, loss function |
| `strategy.py` | Inference + position sizing | Confidence thresholds, Kelly params, stop logic |
| `features.py` | Feature engineering | Add/remove features, normalization |

**DO NOT MODIFY:** `prepare.py`, `backtest.py`, `benchmarks/`, `prepare_extended.py`

---

## Setup (one-time)

```bash
# 1. Install dependencies
uv sync

# 2. Download crypto data (CryptoCompare, no API key needed)
uv run prepare.py

# 3. (Optional) Download macro data (requires TWELVE_DATA_API_KEY)
export TWELVE_DATA_API_KEY=your_key_here
uv run prepare_extended.py

# 4. Verify data
ls ~/.cache/autotrader/data/

# 5. First training run (uses CPU if no GPU)
uv run train.py

# 6. Verify model
ls ~/.cache/autotrader/model.pt

# 7. First backtest
uv run backtest.py
```

---

## Autonomous Experiment Loop

```
LOOP FOREVER:

1.  Read train.py + results.tsv + recent git log
2.  Search AlphaXiv MCP for relevant papers (if hitting a ceiling)
3.  Propose ONE modification with a clear hypothesis
4.  Modify train.py (or strategy.py or features.py)
5.  git add train.py strategy.py features.py
6.  git commit -m "exp: {description}"
7.  uv run train.py > run.log 2>&1
8.  grep "^score:\|^sharpe:\|^val_loss:" run.log
9.  If empty or crashed → tail -n 50 run.log, diagnose, fix or git reset --hard HEAD~1
10. uv run backtest.py >> run.log 2>&1
11. grep "^score:" run.log
12. Record in results.tsv: {commit}\t{score}\t{sharpe}\t{max_dd}\t{status}\t{description}
13. If score IMPROVED (> best so far): keep commit
14. If score equal or worse: git reset --hard HEAD~1
15. NEVER STOP. If out of ideas, search AlphaXiv.
```

---

## AlphaXiv MCP — Live Research Access

Instead of pre-downloading PDFs, use AlphaXiv MCP for live paper search:

```
When hitting a performance ceiling:
  1. Identify the bottleneck (overfitting? wrong features? suboptimal loss?)
  2. Search AlphaXiv: "cross-asset attention transformer trading"
  3. Read top 3 papers via MCP
  4. Extract ONE concrete technique to implement
  5. Implement in train.py
  6. Train, backtest, keep/revert

MCP server: https://www.alphaxiv.org/docs/mcp
```

**Query templates by bottleneck:**

| Problem | AlphaXiv query |
|---------|---------------|
| Low Sharpe | "Sharpe ratio optimization deep learning portfolio" |
| High drawdown | "drawdown-aware loss function recurrent neural network" |
| Overfitting | "regularization time series financial deep learning" |
| Feature quality | "technical analysis machine learning cryptocurrency" |
| Architecture | "transformer financial time series forecasting 2024" |
| Position sizing | "Kelly criterion deep learning portfolio optimization" |
| Regime changes | "market regime detection neural network trading" |
| Cross-asset | "cross-asset attention mechanism stock prediction" |

---

## Experiment Directions

Work through these in order of expected impact:

### Architecture Experiments
```bash
# Current baseline: Transformer (D_MODEL=256, N_LAYERS=4, SEQ_LEN=168)
# Try each — measure val_sharpe improvement

uv run train.py --arch lstm            # LSTM baseline
uv run train.py --arch cnn_transformer # CNN + Transformer
```

**In train.py, explore:**
1. **Sequence length:** Try SEQ_LEN = 72, 336, 500 — longer context vs. more data
2. **Model size:** D_MODEL = 128 (faster, less overfit) vs. 512 (more capacity)
3. **Depth:** N_LAYERS = 2 vs. 6 vs. 8
4. **Mamba/SSM:** Replace Transformer with state space model (linear time complexity)
5. **PatchTST:** Divide sequence into patches of length 16, treat as tokens (like ViT)
6. **Ensemble:** Train 3 models, average predictions

### Loss Function Experiments
```python
# In train.py — change the loss call in the training loop

# Current: sharpe_loss(pred, y)
# Try:
loss = sortino_loss(pred, y)               # penalize only downside vol
loss = combined_loss(pred, y, alpha=0.8)   # Sharpe + MSE stability

# New ideas to implement:
def calmar_loss(pred, future_returns, window=252):
    """Optimize Calmar ratio (return / max drawdown)."""
    ...

def max_sharpe_with_constraints(pred, future_returns, max_dd=0.15):
    """Sharpe loss with drawdown constraint penalty."""
    ...
```

### Feature Experiments
In `features.py`, add/remove feature groups and update N_FEATURES:

```python
# Test which groups help:
# Comment out Group C (macro) → retrain → compare val_sharpe
# Comment out Group B (technical) → retrain → compare
# Add Group F: market microstructure (bid-ask spread proxy, volume imbalance)
```

### Multi-task Learning
```python
# In train.py — predict direction + volatility simultaneously
# Two output heads: direction ([-1,1]) + vol (sigma)

class MultiTaskHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.direction_head = nn.Sequential(Linear(d_model, 64), GELU(), Linear(64, 1), Tanh())
        self.vol_head = nn.Sequential(Linear(d_model, 64), GELU(), Linear(64, 1), Softplus())

# Loss: alpha * sharpe_loss(direction) + beta * vol_mse_loss(vol)
```

### Regime Conditioning
```python
# Use vol regime as conditioning signal
# High-vol regime: more conservative (smaller Kelly fraction)
# Low-vol / squeeze: aggressive (BB squeeze = pending breakout)

# In strategy.py, override position size:
if self.feature_history[symbol][-1][54] > 0.3:  # feature 54 = vol_regime
    size *= 0.5  # reduce in high-vol regime
```

### Cross-asset Attention
```python
# In train.py — custom cross-asset attention block
# BTC, ETH, SOL each attend to macro symbols (GOLD, DXY, SPX)

class CrossAssetAttention(nn.Module):
    """Each trading symbol attends to macro symbols."""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, trading_features, macro_features):
        # trading: (batch, seq, d_model), macro: (batch, n_macro, d_model)
        out, _ = self.attn(trading_features, macro_features, macro_features)
        return out
```

---

## Scoring Reference

```
score = sharpe * sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty

trade_count_factor = min(num_trades / 50, 1.0)
drawdown_penalty   = max(0, max_drawdown_pct - 15) * 0.05
turnover_penalty   = max(0, annual_turnover/capital - 500) * 0.001

Hard cutoffs: <10 trades → -999 | >50% drawdown → -999 | lost >50% → -999
```

**Current baselines:**
- Rule-based simple_momentum: 2.724 ← minimum target
- Rule-based Exp32 (BB width): ~9.38 on val ← strategy.py fallback

**ML targets:**
- Val Sharpe > 3.0: good start
- Val Sharpe > 5.0: strong signal
- Score > 10.0: competitive with production strategies

---

## Results TSV Format

```tsv
commit	score	sharpe	max_dd	status	description
abc1234	9.382	2.814	8.2%	keep	rule-based Exp32 baseline
def5678	11.241	3.102	9.1%	keep	Transformer SEQ_LEN=168 D_MODEL=256
ghi9012	-999	n/a	n/a	revert	LSTM overfitting — too few samples
```

---

## Debugging Checklist

**Training crashes:**
```bash
tail -n 50 run.log          # inspect error
# Common: CUDA OOM → reduce BATCH_SIZE in train.py
# Common: NaN loss → check features.py for _safe() guards
# Common: import error → uv sync
```

**Score is -999:**
```bash
# < 10 trades → strategy is too selective. Lower thresholds:
#   DIRECTION_THRESHOLD = 0.10  (in strategy.py)
#   MIN_CONFIDENCE = 0.50
# > 50% drawdown → reduce ATR_STOP_MULT or BASE_POSITION_PCT
```

**Model not loading:**
```bash
ls ~/.cache/autotrader/model.pt
# If missing → run: uv run train.py
# If wrong N_FEATURES → N_FEATURES mismatch between train.py and features.py
```

**Validating feature count:**
```bash
python -c "from features import N_FEATURES; print(N_FEATURES)"
python -c "from train import N_FEATURES; print(N_FEATURES)"
# These MUST match
```

---

## GPU Notes

| GPU | Expected training time | Recommended PRECISION |
|-----|----------------------|----------------------|
| RTX 3090 / 4090 | 5–10 min | "bf16" |
| A100 40GB | 3–5 min | "bf16" |
| H100 | 2–3 min | "bf16" |
| RTX 2080 / 3080 | 10–20 min | "fp16" |
| CPU (M2/M3 Mac) | 30–60 min | "fp32" |

If CUDA OOM:
```python
# In train.py:
BATCH_SIZE = 128    # reduce from 256
D_MODEL    = 128    # reduce from 256
SEQ_LEN    = 72     # reduce from 168
```

---

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue.
You are autonomous. If you run out of ideas, search AlphaXiv. The loop runs until interrupted.

Minimum: run 20 experiments before concluding anything.
Success criterion: score > 15.0 (meaningfully above rule-based baseline).
