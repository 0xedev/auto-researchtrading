"""
train.py — GPU-native Transformer training for crypto directional prediction.

Agent-mutable file: Claude modifies architecture, hyperparameters, and loss
function autonomously. This is the Karpathy-equivalent core.

Usage:
    uv run train.py              # train on GPU, saves model.pt
    uv run train.py --cpu        # force CPU (slow, for debugging)
    uv run train.py --arch lstm  # use LSTM instead of Transformer

Output format (grep-able by experiment loop):
    score: {val_sharpe:.3f}
    sharpe: {val_sharpe:.3f}
    val_loss: {val_loss:.4f}
    train_loss: {train_loss:.4f}
    epochs: {n_epochs}
"""

import os
import sys
import time
import math
import argparse

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_OK = True
except ImportError:
    print("ERROR: torch not installed. Run: uv add torch")
    sys.exit(1)

# ── Hyperparameters (agent tunes these) ──────────────────────────────────────
SEQ_LEN        = 168      # 1 week of hourly bars (context window)
N_FEATURES     = 128      # must match features.N_FEATURES
D_MODEL        = 256      # transformer embedding dimension
N_HEADS        = 8        # attention heads (D_MODEL must be divisible)
N_LAYERS       = 4        # transformer encoder layers
D_FFN          = 512      # feed-forward network dimension
DROPOUT        = 0.15     # dropout rate
LR             = 3e-4     # AdamW learning rate
WEIGHT_DECAY   = 0.05     # AdamW weight decay
WARMUP_STEPS   = 200      # LR warmup steps
BATCH_SIZE     = 256      # per-GPU batch size
GRAD_CLIP      = 1.0      # gradient clipping norm
MAX_EPOCHS     = 30       # max training epochs
PATIENCE       = 5        # early stopping patience (epochs without improvement)
PRECISION      = "bf16"   # "bf16" on A100+, "fp16" on RTX 30xx/40xx, "fp32" on CPU
COMPILE_MODEL  = True     # torch.compile for ~30% speedup (PyTorch 2.0+)
MODEL_PATH     = os.path.expanduser("~/.cache/autotrader/model.pt")
ARCH           = "transformer"   # "transformer" | "lstm" | "cnn_transformer"
# ─────────────────────────────────────────────────────────────────────────────


# ── Loss functions ──────────────────────────────────────────────────────────

def sharpe_loss(pred_returns, future_returns, eps=1e-6):
    """
    Directly maximize Sharpe ratio.
    pred_returns: (batch,) — predicted portfolio return weights
    future_returns: (batch,) — actual next-bar returns
    """
    portfolio_ret = pred_returns * future_returns
    mean = portfolio_ret.mean()
    std = portfolio_ret.std() + eps
    return -(mean / std)


def sortino_loss(pred_returns, future_returns, eps=1e-6):
    """Sortino: only penalizes downside volatility."""
    portfolio_ret = pred_returns * future_returns
    mean = portfolio_ret.mean()
    downside = portfolio_ret[portfolio_ret < 0]
    if len(downside) == 0:
        return -mean / eps
    dstd = downside.std() + eps
    return -(mean / dstd)


def combined_loss(pred_returns, future_returns, alpha=0.7, eps=1e-6):
    """Weighted combination of Sharpe loss + MSE for stability."""
    sharpe = sharpe_loss(pred_returns, future_returns, eps)
    mse = F.mse_loss(pred_returns, future_returns.clamp(-0.1, 0.1))
    return alpha * sharpe + (1 - alpha) * mse


# ── Model architectures ──────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Learned positional encoding (not sinusoidal — financial has no natural period)."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class TransformerModel(nn.Module):
    """
    Causal Transformer encoder for time series.
    Input:  (batch, seq_len, n_features)
    Output: (batch,) — predicted return weight in [-1, 1]
    """
    def __init__(self, n_features=N_FEATURES, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ffn=D_FFN, dropout=DROPOUT, seq_len=SEQ_LEN):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Learned positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Causal mask (upper triangular, prevents future leakage)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),  # output in [-1, 1] = predicted position weight
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        seq = x.size(1)
        mask = self.causal_mask[:seq, :seq]

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h, mask=mask, is_causal=True)

        # Use last token representation for prediction
        out = self.head(h[:, -1, :])  # (batch, 1)
        return out.squeeze(-1)  # (batch,)


class LSTMModel(nn.Module):
    """
    Multi-layer LSTM for time series prediction.
    Alternative architecture — agent can switch via ARCH hyperparameter.
    """
    def __init__(self, n_features=N_FEATURES, hidden=512, n_layers=4,
                 dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.input_proj(x)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNTransformerModel(nn.Module):
    """
    1D-CNN for local pattern extraction → Transformer for long-range dependencies.
    """
    def __init__(self, n_features=N_FEATURES, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=2, dropout=DROPOUT):
        super().__init__()
        # Local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (batch, seq, d_model)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return self.head(h[:, -1, :]).squeeze(-1)


# ── Dataset ──────────────────────────────────────────────────────────────────

class TradingDataset(Dataset):
    """
    Sliding-window dataset for trading time series.
    Each sample: (features[t-seq_len:t], return[t+1])
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int):
        assert len(features) == len(labels), "features and labels must have same length"
        assert features.shape[1] == N_FEATURES, \
            f"Expected {N_FEATURES} features, got {features.shape[1]}"
        n = len(features)
        # Build sliding windows
        self.X = []
        self.y = []
        for i in range(seq_len, n):
            self.X.append(features[i - seq_len:i])
            self.y.append(labels[i])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    # Cosine decay
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Training loop ─────────────────────────────────────────────────────────────

def build_model(arch=ARCH):
    if arch == "lstm":
        return LSTMModel()
    elif arch == "cnn_transformer":
        return CNNTransformerModel()
    else:
        return TransformerModel()


def train(args):
    # ── Device setup ─────────────────────────────────────────────────────────
    if args.cpu:
        device = torch.device("cpu")
        precision = "fp32"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        precision = PRECISION
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        precision = "fp32"  # MPS doesn't support bf16/fp16 in all ops
    else:
        device = torch.device("cpu")
        precision = "fp32"

    print(f"Device: {device} | Precision: {precision} | Arch: {args.arch}")

    # ── Data loading ─────────────────────────────────────────────────────────
    try:
        from features import compute_feature_matrix, N_FEATURES as NF, compute_macro_features
        from prepare_extended import load_extended_data, load_daily_data
        from prepare import load_data
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    assert NF == N_FEATURES, f"N_FEATURES mismatch: features.py={NF}, train.py={N_FEATURES}"

    print("\nLoading training data...")
    train_data = load_data("train")
    val_data = load_data("val")
    macro_daily = load_daily_data()

    if not train_data:
        print("ERROR: No training data. Run: python prepare.py")
        sys.exit(1)

    # Build feature matrices for each symbol
    train_features_all = []
    train_labels_all = []
    val_features_all = []
    val_labels_all = []

    for symbol in ["BTC", "ETH", "SOL"]:
        if symbol not in train_data:
            print(f"  [SKIP] {symbol}: no train data")
            continue

        df_train = train_data[symbol]
        df_val = val_data.get(symbol)

        print(f"  {symbol}: building features for {len(df_train)} train bars...")
        feat_train, ts_train = compute_feature_matrix(df_train, macro_daily)
        if len(feat_train) < SEQ_LEN + 10:
            print(f"  [SKIP] {symbol}: insufficient features")
            continue

        closes_train = df_train["close"].values.astype(float)
        labels_train = np.zeros(len(feat_train), dtype=np.float32)
        for i in range(len(feat_train)):
            bar_idx = i + 1  # offset because compute_feature_matrix starts at bar 1
            if bar_idx + 1 < len(closes_train):
                ret = (closes_train[bar_idx + 1] - closes_train[bar_idx]) / closes_train[bar_idx]
                labels_train[i] = np.clip(ret, -0.1, 0.1)

        train_features_all.append(feat_train)
        train_labels_all.append(labels_train)

        if df_val is not None and len(df_val) >= SEQ_LEN + 10:
            print(f"  {symbol}: building features for {len(df_val)} val bars...")
            feat_val, _ = compute_feature_matrix(df_val, macro_daily)
            closes_val = df_val["close"].values.astype(float)
            labels_val = np.zeros(len(feat_val), dtype=np.float32)
            for i in range(len(feat_val)):
                bar_idx = i + 1
                if bar_idx + 1 < len(closes_val):
                    ret = (closes_val[bar_idx + 1] - closes_val[bar_idx]) / closes_val[bar_idx]
                    labels_val[i] = np.clip(ret, -0.1, 0.1)
            val_features_all.append(feat_val)
            val_labels_all.append(labels_val)

    if not train_features_all:
        print("ERROR: No features generated. Check data.")
        sys.exit(1)

    train_features = np.concatenate(train_features_all, axis=0)
    train_labels = np.concatenate(train_labels_all, axis=0)

    # Shuffle training data (important: shuffle samples, not sequences)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_features))
    train_features = train_features[idx]
    train_labels = train_labels[idx]

    train_ds = TradingDataset(train_features, train_labels, SEQ_LEN)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    if val_features_all:
        val_features = np.concatenate(val_features_all, axis=0)
        val_labels = np.concatenate(val_labels_all, axis=0)
        val_ds = TradingDataset(val_features, val_labels, SEQ_LEN)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        has_val = True
    else:
        has_val = False
        print("[WARN] No validation data available.")

    print(f"\nDataset: {len(train_ds)} train samples", end="")
    if has_val:
        print(f", {len(val_ds)} val samples")
    else:
        print()

    # ── Model setup ──────────────────────────────────────────────────────────
    model = build_model(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.arch} | Parameters: {n_params:,}")

    if COMPILE_MODEL and hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  [WARN] torch.compile failed: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        fused=(device.type == "cuda"),
    )

    total_steps = MAX_EPOCHS * len(train_loader)
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_sharpe = -999.0
    best_epoch = 0
    patience_count = 0
    step = 0

    print("\n" + "─" * 60)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Update LR
            lr = get_lr(step, WARMUP_STEPS, total_steps, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model(x)
                    loss = sharpe_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = sharpe_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            if torch.isfinite(loss):
                epoch_loss += loss.item()
                n_batches += 1
            step += 1

        train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # ── Validation ───────────────────────────────────────────────────────
        if has_val:
            model.eval()
            val_preds = []
            val_actuals = []
            val_loss_sum = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    if use_amp:
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            pred = model(x)
                    else:
                        pred = model(x)
                    val_preds.append(pred.cpu().numpy())
                    val_actuals.append(y.cpu().numpy())
                    if torch.isfinite(sharpe_loss(pred, y)):
                        val_loss_sum += sharpe_loss(pred, y).item()
                        n_val_batches += 1

            preds = np.concatenate(val_preds)
            actuals = np.concatenate(val_actuals)

            # Compute val Sharpe from predictions
            portfolio_rets = preds * actuals
            if portfolio_rets.std() > 1e-8:
                val_sharpe = portfolio_rets.mean() / portfolio_rets.std() * math.sqrt(8760)
            else:
                val_sharpe = 0.0
            val_loss = val_loss_sum / max(n_val_batches, 1)

            improved = val_sharpe > best_val_sharpe
            if improved:
                best_val_sharpe = val_sharpe
                best_epoch = epoch
                patience_count = 0
                _save_model(model, args.arch, val_sharpe)
            else:
                patience_count += 1

            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_sharpe={val_sharpe:.3f} | "
                  f"{'✓ BEST' if improved else f'patience={patience_count}'} | "
                  f"{elapsed:.1f}s")

            if patience_count >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (best: epoch {best_epoch})")
                break
        else:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train_loss={train_loss:.4f} | {elapsed:.1f}s")
            _save_model(model, args.arch, 0.0)

    # ── Final results (grep-able format) ──────────────────────────────────────
    print("\n" + "─" * 60)
    final_sharpe = best_val_sharpe if has_val else 0.0
    print(f"score: {final_sharpe:.3f}")
    print(f"sharpe: {final_sharpe:.3f}")
    print(f"val_loss: {val_loss:.4f}" if has_val else "val_loss: n/a")
    print(f"train_loss: {train_loss:.4f}")
    print(f"epochs: {best_epoch if has_val else MAX_EPOCHS}")
    print(f"model: {MODEL_PATH}")
    print("─" * 60)


def _save_model(model, arch, val_sharpe):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    # Unwrap compiled model if needed
    raw_model = getattr(model, "_orig_mod", model)
    checkpoint = {
        "arch": arch,
        "state_dict": raw_model.state_dict(),
        "n_features": N_FEATURES,
        "seq_len": SEQ_LEN,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ffn": D_FFN,
        "val_sharpe": val_sharpe,
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"  → Saved model.pt (val_sharpe={val_sharpe:.3f})")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML trading model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--arch", default=ARCH,
                        choices=["transformer", "lstm", "cnn_transformer"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max epochs")
    args = parser.parse_args()

    MAX_EPOCHS = args.epochs
    t_start = time.time()
    train(args)
    print(f"\nTotal time: {time.time() - t_start:.1f}s")
