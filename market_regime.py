from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_COLUMNS = ["timestamp", "regime", "regime_family"]


def regime_family(label: str) -> str:
    if not label:
        return "unknown"
    if label.startswith("Bull"):
        return "bull"
    if label.startswith("Bear"):
        return "bear"
    if label == "Sideways":
        return "sideways"
    return "unknown"


def build_regime_frame(data: dict[str, pd.DataFrame], anchor: str = "BTC") -> pd.DataFrame:
    if not data:
        return pd.DataFrame(columns=REGIME_COLUMNS)

    ref_df = data.get(anchor)
    if ref_df is None:
        ref_df = next(iter(data.values()))
    if ref_df is None or ref_df.empty:
        return pd.DataFrame(columns=REGIME_COLUMNS)

    close = pd.to_numeric(ref_df["close"], errors="coerce").reset_index(drop=True)
    ts = pd.to_numeric(ref_df["timestamp"], errors="coerce").astype("Int64").reset_index(drop=True)

    ema200 = close.ewm(span=200, adjust=False).mean()
    slope = ema200.pct_change(20)
    vol30 = close.pct_change().rolling(30).std()
    volpct = vol30.rank(pct=True)

    labels = np.full(len(close), "Sideways", dtype=object)
    labels[(slope > 0.01) & (volpct < 0.60)] = "Bull (Low Vol)"
    labels[(slope > 0.01) & (volpct >= 0.60)] = "Bull (High Vol)"
    labels[(slope < -0.01) & (volpct >= 0.50)] = "Bear (Panic)"
    labels[(slope < -0.01) & (volpct < 0.50)] = "Bear (Grind)"
    labels[slope.isna() | volpct.isna()] = "Initializing"

    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "regime": labels,
        }
    ).dropna(subset=["timestamp"])
    frame["timestamp"] = frame["timestamp"].astype("int64")
    frame["regime_family"] = frame["regime"].map(regime_family)
    return frame


def detect_regimes(data: dict[str, pd.DataFrame], anchor: str = "BTC") -> list[tuple[str, int, int]]:
    frame = build_regime_frame(data, anchor=anchor)
    if frame.empty:
        return []

    segs: list[tuple[str, int, int]] = []
    prev: str | None = None
    t0: int | None = None
    rows = list(frame[["timestamp", "regime"]].itertuples(index=False, name=None))
    for timestamp, regime in rows:
        if regime != prev:
            if prev and prev != "Initializing" and t0 is not None:
                segs.append((prev, int(t0), int(last_ts)))
            t0 = int(timestamp)
            prev = regime
        last_ts = int(timestamp)

    if prev and prev != "Initializing" and t0 is not None:
        segs.append((prev, int(t0), int(rows[-1][0])))
    return segs
