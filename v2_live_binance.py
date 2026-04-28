"""
v2_live_binance.py — V2 multi-alpha live/paper trader for Binance USD-M Futures

Trades the V2 quality13_wave5 champion config on Binance Futures.
Uses the same signal engine as the validated shadow runs.

Demo (testnet):
    uv run v2_live_binance.py --portfolio-config v2_portfolio.wave5_quality13_pesfix.json --testnet --dry-run
    uv run v2_live_binance.py --portfolio-config v2_portfolio.wave5_quality13_pesfix.json --testnet

Live (real money — requires explicit --live flag):
    uv run v2_live_binance.py --portfolio-config v2_portfolio.wave5_quality13_pesfix.json --live

Symbols used on testnet: BTC, ETH, SOL, quality13 OOS basket, and XAU when exchangeInfo lists them.
Symbols used on live endpoint: quality13 basket (LTC, BCH, ETC, TRX, AAVE, FIL, OP)

Background:
    nohup uv run v2_live_binance.py \\
        --portfolio-config v2_portfolio.wave5_quality13_pesfix.json --testnet \\
        >> logs/v2_binance.log 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import time
import urllib.parse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from execution.v2_paper import load_v2_shadow_state

# ── .env loader ───────────────────────────────────────────────────────────────
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_dotenv()

# ── Endpoints ─────────────────────────────────────────────────────────────────
TESTNET_BASE = "https://testnet.binancefuture.com"
LIVE_BASE    = "https://fapi.binance.com"

# V2 symbol → Binance USD-M Futures symbol
SYMBOL_MAP: dict[str, str] = {
    "BTC":  "BTCUSDT",
    "ETH":  "ETHUSDT",
    "SOL":  "SOLUSDT",
    "LTC":  "LTCUSDT",
    "BCH":  "BCHUSDT",
    "ETC":  "ETCUSDT",
    "TRX":  "TRXUSDT",
    "AAVE": "AAVEUSDT",
    "FIL":  "FILUSDT",
    "OP":   "OPUSDT",
    "XAU":  "XAUUSDT",
}

# Testnet supports the canonical fresh-OOS basket plus XAUUSDT as of 2026-04-28.
# The runner still filters this list through exchangeInfo on startup.
TESTNET_SYMBOLS  = ["BTC", "ETH", "SOL", "LTC", "BCH", "ETC", "TRX", "AAVE", "FIL", "OP", "XAU"]
# Full quality13 champion basket (live endpoint)
LIVE_SYMBOLS     = ["LTC", "BCH", "ETC", "TRX", "AAVE", "FIL", "OP"]

LEVERAGE              = 3           # conservative for paper validation
MARGIN_MODE           = "ISOLATED"
MIN_NOTIONAL          = 10.0        # USD
EXCHANGE_MIN_NOTIONAL = 100.0       # Binance minimum per order
SL_PCT                = 0.025       # 2.5% adverse move → STOP_MARKET
TP_PCT                = 0.060       # 6.0% favorable move → TAKE_PROFIT_MARKET
RECV_WINDOW           = 5000
BAR_BUFFER_SECS       = 8           # wait after candle close before reading
FETCH_LIMIT_1H        = 600         # enough bars for V2 feature lookback

LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
log = logging.getLogger("v2_binance")

IS_TTY  = sys.stdout.isatty()
_RESET  = "\033[0m"; _BOLD = "\033[1m"; _GREEN = "\033[92m"
_RED    = "\033[91m"; _CYN  = "\033[96m"; _YLW  = "\033[93m"; _DIM = "\033[2m"


def _cpnl(v: float) -> str:
    s = f"${v:+.2f}"
    return f"{_GREEN}{s}{_RESET}" if v >= 0 else f"{_RED}{s}{_RESET}"


def live_mode_unlocked(stdin_is_tty: bool, env: dict | None = None) -> bool:
    env = env or os.environ
    return stdin_is_tty or env.get("ALLOW_REAL_MONEY") == "YES_I_UNDERSTAND"


def configure_file_logging(log_path: str) -> None:
    if not log_path:
        return
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    abs_target = str(target.resolve())
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == abs_target:
            return
    handler = logging.FileHandler(abs_target, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
    root.addHandler(handler)
    log.info("File logging enabled: %s", abs_target)


def _render_binance_dashboard(
    endpoint: str,
    equity: float,
    close_prices: dict,
    positions: dict,
    entry_prices: dict,
    bar_count: int,
    last_bar_str: str,
    next_bar_ts: float,
    dry_run: bool,
):
    if not IS_TTY:
        return
    W   = 70
    now = datetime.now(tz=timezone.utc)
    nb  = datetime.fromtimestamp(next_bar_ts, tz=timezone.utc).strftime("%H:%M:%S UTC")
    sys.stdout.write("\033[H\033[J")
    mode = f"{_RED}[DRY-RUN]{_RESET}" if dry_run else ""
    print(f"{_BOLD}{'═'*W}{_RESET}")
    print(f"{_BOLD}  V2 BINANCE TRADER{_RESET}  {mode}  │  {_CYN}{now.strftime('%Y-%m-%d %H:%M:%S')} UTC{_RESET}")
    print(f"  Endpoint: {_CYN}{endpoint}{_RESET}   "
          f"Equity: {_BOLD}${equity:,.2f}{_RESET}   "
          f"Positions: {len(positions)}   "
          f"Next bar: {_YLW}{nb}{_RESET}  #{bar_count}")
    if last_bar_str:
        print(f"  Last bar: {_DIM}{last_bar_str}{_RESET}")
    print(f"{_DIM}{'─'*W}{_RESET}")
    print(f"{_BOLD}  PRICES{_RESET}")
    for sym, price in sorted(close_prices.items()):
        pstr = f"{price:,.4f}" if price < 10_000 else f"{price:,.2f}"
        in_pos = sym in positions
        tag = f"  {_GREEN}●{_RESET}" if in_pos else ""
        print(f"    {_BOLD}{sym:<6}{_RESET}  {_CYN}{pstr}{_RESET}{tag}")
    print(f"{_DIM}{'─'*W}{_RESET}")
    print(f"{_BOLD}  OPEN POSITIONS{_RESET}")
    if not positions:
        print(f"    {_DIM}none{_RESET}")
    else:
        print(f"    {_BOLD}{'SYM':<6}{'DIR':<7}{'NOTIONAL':>12}{'ENTRY':>12}{'NOW':>12}{'P&L':>10}{_RESET}")
        for sym, notional in positions.items():
            entry = entry_prices.get(sym, 0.0)
            cur   = close_prices.get(sym, entry)
            side  = 1.0 if notional >= 0 else -1.0
            pct   = side * (cur / entry - 1) if entry else 0.0
            pnl   = abs(notional) * pct
            d_tag = (f"{_GREEN}▲ LONG {_RESET}" if notional >= 0
                     else f"{_RED}▼ SHORT{_RESET}")
            cur_str   = f"{cur:,.4f}" if cur < 10_000 else f"{cur:,.2f}"
            entry_str = f"{entry:,.4f}" if entry < 10_000 else f"{entry:,.2f}"
            print(f"    {_BOLD}{sym:<6}{_RESET}{d_tag}"
                  f"  ${abs(notional):>10,.0f}"
                  f"  {entry_str:>11}"
                  f"  {cur_str:>11}"
                  f"  {_cpnl(pnl)}")
    print(f"{_BOLD}{'═'*W}{_RESET}")
    sys.stdout.flush()


# ── Binance USD-M Futures REST client ─────────────────────────────────────────
class BinanceFutures:
    def __init__(self, api_key: str, secret: str, base_url: str):
        self.api_key = api_key
        self._secret = secret.encode("utf-8")
        self.base    = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign(self, params: dict) -> dict:
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = RECV_WINDOW
        qs  = urllib.parse.urlencode(params)
        sig = hmac.new(self._secret, qs.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _get(self, path: str, params: dict = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params = self._sign(params)
        r = self._session.get(f"{self.base}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, params: dict) -> dict:
        params = self._sign(dict(params))
        r = self._session.post(
            f"{self.base}{path}",
            data=urllib.parse.urlencode(params),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        if not r.ok:
            try:
                code = r.json().get("code", 0)
            except Exception:
                code = 0
            if code not in (-4046,):
                log.error("POST %s %d: %s", path, r.status_code, r.text[:300])
            r.raise_for_status()
        return r.json()

    def _delete(self, path: str, params: dict) -> dict:
        params = self._sign(dict(params))
        r = self._session.delete(f"{self.base}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        return self._get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def funding_rate(self, symbol: str, limit: int = 100) -> list:
        return self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})

    def exchange_info(self) -> dict:
        return self._get("/fapi/v1/exchangeInfo")

    def symbols_available(self) -> set:
        info = self.exchange_info()
        return {s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"}

    def account(self) -> dict:
        return self._get("/fapi/v2/account", signed=True)

    def position_risk(self) -> list:
        return self._get("/fapi/v2/positionRisk", {}, signed=True)

    def open_orders(self, symbol: str | None = None) -> list:
        params = {"symbol": symbol} if symbol else {}
        return self._get("/fapi/v1/openOrders", params, signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._post("/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def set_margin_type(self, symbol: str, margin_type: str) -> dict:
        try:
            return self._post("/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type})
        except requests.HTTPError as exc:
            if exc.response is not None and "No need to change" in exc.response.text:
                log.debug("  %s margin type already %s", symbol, margin_type)
                return {}
            raise

    def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
        params = {"symbol": symbol, "side": side, "type": "MARKET", "quantity": f"{qty:.8f}"}
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._post("/fapi/v1/order", params)

    def cancel_all_orders(self, symbol: str) -> dict:
        result = {}
        for path in ("/fapi/v1/allOpenOrders", "/fapi/v1/algoOpenOrders"):
            try:
                result[path] = self._delete(path, {"symbol": symbol})
            except Exception as exc:
                log.debug("cancel_all_orders(%s %s): %s", symbol, path, exc)
        return result

    def stop_market_order(self, symbol: str, side: str, stop_price: float) -> dict:
        return self._post("/fapi/v1/algoOrder", {
            "algoType": "CONDITIONAL", "symbol": symbol, "side": side,
            "type": "STOP_MARKET", "triggerPrice": f"{stop_price:.4f}",
            "workingType": "CONTRACT_PRICE", "closePosition": "true",
        })

    def take_profit_order(self, symbol: str, side: str, stop_price: float) -> dict:
        return self._post("/fapi/v1/algoOrder", {
            "algoType": "CONDITIONAL", "symbol": symbol, "side": side,
            "type": "TAKE_PROFIT_MARKET", "triggerPrice": f"{stop_price:.4f}",
            "workingType": "CONTRACT_PRICE", "closePosition": "true",
        })


# ── Market data helpers ───────────────────────────────────────────────────────
def fetch_and_cache(client: BinanceFutures, v2_sym: str, bn_sym: str) -> Optional[pd.DataFrame]:
    """Fetch 1h klines + funding for `bn_sym`, update V2 parquet cache, return DataFrame."""
    from v2.live_data import upsert_cache

    try:
        raw = client.klines(bn_sym, "1h", limit=FETCH_LIMIT_1H)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            log.warning("%-8s not available on this endpoint — skipping", bn_sym)
            return None
        raise
    if not raw or len(raw) < 10:
        return None

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "num_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ])[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    df["funding_rate"] = 0.0

    try:
        for fr in client.funding_rate(bn_sym, limit=100):
            ts   = int(fr["fundingTime"])
            rate = float(fr["fundingRate"])
            mask = (df["timestamp"] >= ts) & (df["timestamp"] < ts + 8 * 3_600_000)
            df.loc[mask, "funding_rate"] = rate
    except Exception as exc:
        log.debug("Funding unavailable %s: %s", bn_sym, exc)

    df = df.reset_index(drop=True)
    upsert_cache(v2_sym, df)
    return df


def get_lot_rules(client: BinanceFutures) -> dict:
    info  = client.exchange_info()
    rules = {}
    for s in info.get("symbols", []):
        bn_sym = s["symbol"]
        if bn_sym not in SYMBOL_MAP.values():
            continue
        step, min_q = 1.0, 0.001
        for f in s.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                step  = float(f["stepSize"])
                min_q = float(f["minQty"])
        rules[bn_sym] = {"step": step, "min_qty": min_q}
    return rules


def round_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    decimals = max(0, round(-math.log10(step + 1e-15)))
    return round(round(qty / step) * step, decimals)


def sync_portfolio_equity(client: BinanceFutures) -> float:
    acct = client.account()
    return float(acct.get("totalWalletBalance", 0.0)) + float(acct.get("totalUnrealizedProfit", 0.0))


def _order_avg_price(order: dict, fallback: float) -> float:
    for key in ("avgPrice", "price"):
        try:
            value = float(order.get(key, 0.0))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return fallback


def reconcile_binance_positions(
    client: BinanceFutures,
    active_map: dict[str, str],
    state_path: str,
    close_prices: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict], bool]:
    """Return exchange positions and whether local state disagrees with exchange state."""
    inverse = {bn: sym for sym, bn in active_map.items()}
    close_prices = close_prices or {}
    positions: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    position_meta: dict[str, dict] = {}
    for row in client.position_risk():
        bn_sym = row.get("symbol", "")
        sym = inverse.get(bn_sym)
        if not sym:
            continue
        qty = float(row.get("positionAmt", 0.0) or 0.0)
        if abs(qty) <= 0:
            continue
        entry = float(row.get("entryPrice", 0.0) or 0.0)
        mark = float(row.get("markPrice", 0.0) or 0.0) or close_prices.get(sym, entry)
        raw_notional = float(row.get("notional", 0.0) or 0.0)
        notional = raw_notional if abs(raw_notional) > 0 else qty * mark
        positions[sym] = float(notional)
        entry_prices[sym] = entry if entry > 0 else mark
        position_meta[sym] = {
            "exchange": "binance_usdm",
            "exchange_symbol": bn_sym,
            "reconciled": True,
            "position_amt": qty,
        }

    local = load_v2_shadow_state(state_path)
    local_positions = {k: round(float(v), 6) for k, v in local.positions.items() if abs(float(v)) >= MIN_NOTIONAL}
    exchange_positions = {k: round(float(v), 6) for k, v in positions.items() if abs(float(v)) >= MIN_NOTIONAL}
    mismatch = local_positions != exchange_positions
    return positions, entry_prices, position_meta, mismatch


def seconds_until_next_bar(bar_seconds: int = 3600) -> float:
    now = time.time()
    nxt = (int(now / bar_seconds) + 1) * bar_seconds + BAR_BUFFER_SECS
    return max(1.0, nxt - now)


# ── Order execution ───────────────────────────────────────────────────────────
def execute_opens(
    client: BinanceFutures,
    opens: list[dict],
    close_prices: dict[str, float],
    lot_rules: dict,
    active_map: dict[str, str],
    dry_run: bool,
) -> dict[str, dict]:
    """Execute V2 open actions. Returns {symbol: fill_meta}."""
    fills = {}
    for action in opens:
        sym    = action["symbol"]
        bn_sym = active_map.get(sym)
        if not bn_sym:
            log.debug("  OPEN %-6s — no exchange mapping, skip", sym)
            continue
        price  = close_prices.get(sym, 0.0)
        if price <= 0:
            log.warning("  OPEN %-6s — no close price, skip", sym)
            continue

        target_usd = float(action["target_notional_usd"])
        if abs(target_usd) < MIN_NOTIONAL:
            log.debug("  OPEN %-6s — notional $%.0f below minimum, skip", sym, abs(target_usd))
            continue

        rules = lot_rules.get(bn_sym, {"step": 0.001, "min_qty": 0.001})
        qty   = round_qty(abs(target_usd) / price, rules["step"])
        if qty * price < EXCHANGE_MIN_NOTIONAL:
            qty = math.ceil(EXCHANGE_MIN_NOTIONAL / price / rules["step"]) * rules["step"]
        side = "BUY" if target_usd > 0 else "SELL"

        log.info("  OPEN   %-6s  %-4s  qty=%.4f  ~$%.0f  @ %.4f%s",
                 sym, side, qty, qty * price, price, "  [DRY]" if dry_run else "")
        if not dry_run:
            try:
                client.cancel_all_orders(bn_sym)
                res = client.market_order(bn_sym, side, qty)
                log.info("         orderId=%s  status=%s", res.get("orderId"), res.get("status"))
                fill_price = _order_avg_price(res, price)
                close_side = "SELL" if side == "BUY" else "BUY"
                sl_price = fill_price * (1 - SL_PCT) if side == "BUY" else fill_price * (1 + SL_PCT)
                tp_price = fill_price * (1 + TP_PCT) if side == "BUY" else fill_price * (1 - TP_PCT)
                try:
                    client.stop_market_order(bn_sym, close_side, sl_price)
                    client.take_profit_order(bn_sym, close_side, tp_price)
                    log.info("         SL=%.4f  TP=%.4f", sl_price, tp_price)
                except Exception as exc:
                    log.warning("         SL/TP failed %s: %s", sym, exc)
            except Exception as exc:
                log.error("  OPEN  %s failed: %s", sym, exc)
                continue
        else:
            res = {}
            fill_price = price
        meta = dict(action["meta"])
        meta.update({
            "exchange": "binance_usdm",
            "exchange_symbol": bn_sym,
            "order_id": res.get("orderId"),
            "filled_qty": qty,
            "requested_notional": target_usd,
            "fill_price": fill_price,
        })
        fills[sym] = {"entry_price": fill_price, "notional": target_usd, "meta": meta}
    return fills


def execute_exits(
    client: BinanceFutures,
    exits: list[dict],
    positions: dict[str, float],
    close_prices: dict[str, float],
    lot_rules: dict,
    active_map: dict[str, str],
    dry_run: bool,
) -> set[str]:
    """Execute V2 exit actions. Returns set of closed symbols."""
    closed = set()
    for action in exits:
        sym    = action["symbol"]
        bn_sym = active_map.get(sym)
        if not bn_sym:
            continue
        notional = positions.get(sym, 0.0)
        if abs(notional) < MIN_NOTIONAL:
            continue
        price  = close_prices.get(sym, 0.0)
        if price <= 0:
            continue
        rules  = lot_rules.get(bn_sym, {"step": 0.001, "min_qty": 0.001})
        qty    = round_qty(abs(notional) / price, rules["step"])
        side   = "SELL" if notional > 0 else "BUY"

        log.info("  CLOSE  %-6s  %-4s  qty=%.4f  ~$%.0f  reason=%s%s",
                 sym, side, qty, qty * price, action["reason"],
                 "  [DRY]" if dry_run else "")
        if not dry_run:
            try:
                client.cancel_all_orders(bn_sym)
                res = client.market_order(bn_sym, side, qty, reduce_only=True)
                log.info("         orderId=%s  status=%s", res.get("orderId"), res.get("status"))
            except Exception as exc:
                log.error("  CLOSE %s failed: %s", sym, exc)
                continue
        closed.add(sym)
    return closed


# ── Main loop ─────────────────────────────────────────────────────────────────
def run(
    portfolio_config_path: str,
    model_set: str,
    use_testnet: bool,
    dry_run: bool,
    state_path: str,
    log_path: str,
):
    from v2.live_data import compute_live_actions, update_state_after_execution, BINANCE_SYMBOL_MAP
    from v2.types import PortfolioConfig
    import json

    config_payload = json.loads(Path(portfolio_config_path).read_text())
    portfolio_config = PortfolioConfig.from_dict(config_payload.get("portfolio"))
    active_sleeves = config_payload.get("active_sleeves") or None

    api_key  = os.environ.get("BINANCE_API_KEY", "")
    secret   = os.environ.get("BINANCE_API_SECRET", "")
    base_url = TESTNET_BASE if use_testnet else LIVE_BASE

    if not dry_run and (not api_key or not secret):
        log.error("Set BINANCE_API_KEY and BINANCE_API_SECRET in .env (or use --dry-run)")
        sys.exit(1)

    client = BinanceFutures(api_key, secret, base_url)
    log.info("Endpoint: %s  |  dry_run=%s  model_set=%s", base_url, dry_run, model_set)

    # Discover available symbols
    try:
        available_bn = client.symbols_available()
    except Exception as exc:
        log.warning("exchange_info failed: %s — assuming all symbols", exc)
        available_bn = set(SYMBOL_MAP.values())

    candidate_symbols = TESTNET_SYMBOLS if use_testnet else LIVE_SYMBOLS
    active_map = {sym: SYMBOL_MAP[sym] for sym in candidate_symbols
                  if sym in SYMBOL_MAP and SYMBOL_MAP[sym] in available_bn}
    log.info("Active symbols: %s", list(active_map.keys()))

    if not active_map:
        log.error("No tradable symbols found on %s — check API keys", base_url)
        sys.exit(1)

    lot_rules = get_lot_rules(client)

    if not dry_run:
        for sym, bn_sym in active_map.items():
            try:
                client.set_margin_type(bn_sym, MARGIN_MODE)
                client.set_leverage(bn_sym, LEVERAGE)
                log.info("  %-6s  %s  %dx leverage", sym, MARGIN_MODE, LEVERAGE)
            except Exception as exc:
                log.error("  Setup %s failed: %s", sym, exc)
                sys.exit(2)

        try:
            startup_equity = sync_portfolio_equity(client)
            positions, entry_prices, position_meta, state_mismatch = reconcile_binance_positions(
                client, active_map, state_path
            )
            open_order_count = sum(len(client.open_orders(bn_sym)) for bn_sym in active_map.values())
        except Exception as exc:
            log.error("Startup account/position reconciliation failed: %s", exc)
            sys.exit(2)
    else:
        startup_equity = 100_000.0
        positions, entry_prices, position_meta, state_mismatch = {}, {}, {}, False
        open_order_count = 0

    os.makedirs(os.path.dirname(state_path) if os.path.dirname(state_path) else ".", exist_ok=True)
    halt_new_orders = bool(state_mismatch or open_order_count > 0)
    if halt_new_orders:
        log.error(
            "Startup reconciliation mismatch/open-orders detected; halting new orders "
            "(state_mismatch=%s open_orders=%d). Exits/reconciliation only.",
            state_mismatch,
            open_order_count,
        )
    update_state_after_execution(
        state_path=state_path,
        timestamp=int(time.time() * 1000),
        equity=startup_equity,
        positions=positions,
        entry_prices=entry_prices,
        position_meta=position_meta,
    )

    # Dashboard state (updated after each bar, read by display loop)
    _dash: dict = {
        "close_prices": {},
        "equity": 100_000.0,
        "last_bar_str": "",
        "next_bar_ts": (int(time.time() / 3600) + 1) * 3600 + BAR_BUFFER_SECS,
    }
    endpoint_label = (TESTNET_BASE if use_testnet else LIVE_BASE).replace("https://", "")

    bar_count = 0
    while True:
        bar_count += 1
        _dash["next_bar_ts"] = (int(time.time() / 3600) + 1) * 3600 + BAR_BUFFER_SECS
        log.info("─── Bar %-4d  %s ─────────────────────",
                 bar_count, datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

        # 1. Refresh parquet cache with live bars
        close_prices_raw = {}
        for sym, bn_sym in active_map.items():
            df = fetch_and_cache(client, sym, bn_sym)
            if df is not None and not df.empty:
                close_prices_raw[sym] = float(df.iloc[-1]["close"])
                log.info("  %-6s  close=%.4f  bars=%d", sym, close_prices_raw[sym], len(df))
        _dash["close_prices"] = dict(close_prices_raw)

        if not close_prices_raw:
            log.error("No market data — skipping bar")
            time.sleep(seconds_until_next_bar())
            continue

        # 2. Get V2 actions for latest bar
        equity = 100_000.0
        if not dry_run:
            try:
                equity = sync_portfolio_equity(client)
            except Exception as exc:
                log.error("Portfolio sync failed: %s — halting trader", exc)
                sys.exit(2)
        _dash["equity"] = equity

        try:
            result = compute_live_actions(
                bundle_name="bundle_intraday_core",
                model_set=model_set,
                portfolio_config=portfolio_config,
                active_sleeves=active_sleeves,
                symbols=list(active_map.keys()),
                state_path=state_path,
                equity_override=equity,
                halt_new_orders=halt_new_orders,
            )
        except Exception as exc:
            log.exception("V2 signal engine error: %s", exc)
            time.sleep(seconds_until_next_bar())
            continue

        latest_ts    = result["latest_ts"]
        opens        = result["opens"]
        exits        = result["exits"]
        close_prices = result.get("close_prices") or close_prices_raw

        log.info("  Equity=$%.2f  ts=%s  opens=%d  exits=%d",
                 equity, latest_ts, len(opens), len(exits))

        # 3. Execute exits first, then opens
        closed_syms = execute_exits(
            client, exits, positions, close_prices, lot_rules, active_map, dry_run)
        for sym in closed_syms:
            positions.pop(sym, None)
            entry_prices.pop(sym, None)
            position_meta.pop(sym, None)

        fill_meta = execute_opens(
            client, opens, close_prices, lot_rules, active_map, dry_run)
        for sym, fill in fill_meta.items():
            positions[sym]    = fill["notional"]
            entry_prices[sym] = fill["entry_price"]
            position_meta[sym] = fill["meta"]

        # 4. Persist V2 state so next bar sees current positions
        if latest_ts:
            update_state_after_execution(
                state_path=state_path,
                timestamp=latest_ts,
                equity=equity,
                positions=positions,
                entry_prices=entry_prices,
                position_meta=position_meta,
            )

        log.info("  Open positions: %s",
                 {k: f"${v:+.0f}" for k, v in positions.items()} or "none")

        _dash["last_bar_str"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        next_bar_ts = (int(time.time() / 3600) + 1) * 3600 + BAR_BUFFER_SECS
        _dash["next_bar_ts"] = next_bar_ts
        next_dt = datetime.fromtimestamp(next_bar_ts, tz=timezone.utc).strftime("%H:%M:%S UTC")
        log.info("  Next bar at %s", next_dt)

        # Display loop — refreshes every 2s while waiting for next bar
        while time.time() < next_bar_ts - 1:
            _render_binance_dashboard(
                endpoint=endpoint_label,
                equity=_dash["equity"],
                close_prices=_dash["close_prices"],
                positions=positions,
                entry_prices=entry_prices,
                bar_count=bar_count,
                last_bar_str=_dash["last_bar_str"],
                next_bar_ts=_dash["next_bar_ts"],
                dry_run=dry_run,
            )
            time.sleep(2)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="V2 Binance USD-M Futures live/paper trader")
    p.add_argument("--portfolio-config", default="v2_portfolio.wave5_quality13_pesfix.json",
                   help="Portfolio config JSON path")
    p.add_argument("--model-set", default="v2_wave5_extended",
                   help="Model set name (default: v2_wave5_extended)")
    p.add_argument("--state-path", default="tmp/v2_binance_state.json",
                   help="State file path for position persistence")
    p.add_argument("--log-path", default="logs/v2_binance.log",
                   help="Trade log path")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch data and compute signals but place NO orders")
    p.add_argument("--testnet", action="store_true", default=True,
                   help="Use Binance Futures testnet (default, symbols: BTC/ETH/SOL)")
    p.add_argument("--live", action="store_true",
                   help="Use live endpoint with quality13 basket — REAL MONEY")
    args = p.parse_args()

    if not Path(args.portfolio_config).exists():
        print(f"ERROR: portfolio config not found: {args.portfolio_config}")
        sys.exit(1)

    use_testnet = not args.live
    if args.live:
        if sys.stdin.isatty():
            answer = input(
                "\n⚠  WARNING: --live uses REAL MONEY on https://fapi.binance.com\n"
                "   Type 'yes i understand' to proceed: "
            )
            if answer.strip().lower() != "yes i understand":
                print("Aborted.")
                sys.exit(0)
        elif not live_mode_unlocked(False):
            log.error(
                "--live requested non-interactively without ALLOW_REAL_MONEY=YES_I_UNDERSTAND; aborting"
            )
            sys.exit(2)

    os.makedirs("logs", exist_ok=True)
    configure_file_logging(args.log_path)
    run(
        portfolio_config_path=args.portfolio_config,
        model_set=args.model_set,
        use_testnet=use_testnet,
        dry_run=args.dry_run,
        state_path=args.state_path,
        log_path=args.log_path,
    )
