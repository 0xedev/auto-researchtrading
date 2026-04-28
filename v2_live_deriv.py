"""
v2_live_deriv.py — V2 multi-alpha live trader for Deriv synthetic indices

Trades Volatility 75 Index (R_75 ↔ DERIV_V75) and Step Index (stpRNG ↔ DERIV_STEP)
using the V2 XGBoost + HMM portfolio engine (same engine as shadow validation).

Products: Multipliers (MULTUP / MULTDOWN)
  - MULTUP  = long  (position grows as price rises)
  - MULTDOWN = short (position grows as price falls)
  - No fixed expiry — stays open until sold, SL hit, or TP hit
  - Max loss = stake amount

Setup:
  Add to .env:
    DERIV_API_TOKEN=your_token_here
    DERIV_APP_ID=1        # Deriv's public demo app_id (or your own registered app_id)

Usage:
    # Dry-run — fetches live data and computes signals, places NO orders
    uv run v2_live_deriv.py --portfolio-config v2_portfolio.wave5_quality13_pesfix.json --dry-run

    # Demo/virtual account (uses token's VRTC account)
    uv run v2_live_deriv.py --portfolio-config v2_portfolio.wave5_quality13_pesfix.json

    # Background
    nohup uv run v2_live_deriv.py \\
        --portfolio-config v2_portfolio.wave5_quality13_pesfix.json \\
        >> logs/v2_deriv.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from execution.v2_paper import load_v2_shadow_state

try:
    import websocket  # websocket-client
except ImportError:
    print("ERROR: websocket-client not installed. Run: uv pip install websocket-client")
    sys.exit(1)

try:
    import certifi
    _SSL_OPT = {"ca_certs": certifi.where()}
except ImportError:
    import ssl
    _SSL_OPT = {"cert_reqs": ssl.CERT_NONE}


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

# ── Constants ─────────────────────────────────────────────────────────────────
DERIV_WS_URL     = "wss://ws.binaryws.com/websockets/v3"
DEFAULT_APP_ID   = "1"
GRANULARITY      = 3600    # 1h candles (matches V2 training timeframe)
BAR_SECONDS      = 3600    # bar boundary detection period
CANDLE_COUNT     = 600     # fetch 600 candles (~25 days of 1h bars for feature context)
MAX_STAKE_PCT    = 0.02    # max 2% of balance per trade
BAR_SLEEP_BUFFER = 5       # seconds after bar boundary before fetching candles
SL_PCT           = 0.75    # stop_loss  = 75% of stake (server-side, instant)
TP_PCT           = 2.00    # take_profit = 200% of stake (server-side, instant)
DISPLAY_INTERVAL = 2       # seconds between live terminal refreshes
IS_TTY           = sys.stdout.isatty()

# V2 cache symbol → Deriv API symbol.
# This is the live-tradable subset confirmed by contracts_for/multiplier lookup.
# The 11-instrument synthetic OOS generator remains robustness-only evidence.
DERIV_SYMBOL_MAP: dict[str, str] = {
    "DERIV_V10":    "R_10",    # Volatility 10 Index
    "DERIV_V25":    "R_25",    # Volatility 25 Index
    "DERIV_V50":    "R_50",    # Volatility 50 Index
    "DERIV_V75":    "R_75",    # Volatility 75 Index
    "DERIV_V100":   "R_100",   # Volatility 100 Index
    "DERIV_CRASH300": "CRASH300N",  # Crash 300 Index
    "DERIV_BOOM300":  "BOOM300N",   # Boom 300 Index
    "DERIV_JUMP25":  "JD25",   # Jump 25 Index
    "DERIV_JUMP100": "JD100",  # Jump 100 Index
    "DERIV_STEP":   "stpRNG",  # Step Index
    "DERIV_RB100":  "RB100",   # Range Break 100
}

# Safe fallback multipliers when contract metadata lookup fails.
# Lower multipliers for high-vol indices (CRASH/BOOM) to keep stake reasonable.
DEFAULT_MULTIPLIERS: dict[str, int] = {
    "DERIV_V10":    200,
    "DERIV_V25":    100,
    "DERIV_V50":     50,
    "DERIV_V75":     50,
    "DERIV_V100":    20,
    "DERIV_CRASH300": 20,
    "DERIV_BOOM300":  20,
    "DERIV_JUMP25":  50,
    "DERIV_JUMP100": 30,
    "DERIV_STEP":   750,
    "DERIV_RB100":   50,
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  %(levelname)-8s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
)
log = logging.getLogger("v2_deriv")


def configure_file_logging(log_path: str) -> None:
    if not log_path:
        return
    target = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == target:
            return
    handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATEFMT))
    root.addHandler(handler)
    log.info("File logging enabled: %s", target)


# ── Deriv WebSocket client ────────────────────────────────────────────────────
class DerivClient:
    """
    Thread-safe synchronous wrapper around the Deriv WebSocket API.
    Runs a background reader thread; requests are sent with a req_id and the
    caller blocks on a threading.Event until the matching response arrives.
    """

    def __init__(self, api_token: str, app_id: str):
        self._api_token = api_token
        self._app_id    = app_id
        self._ws: Optional[websocket.WebSocket] = None
        self._req_id    = 0
        self._lock      = threading.Lock()
        self._pending: dict = {}
        self._stream_cb: dict = {}
        self._type_cb: dict = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._running   = False
        self.account_id: Optional[str] = None
        self.balance: float = 0.0
        self.currency: str  = "USD"

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self):
        url = f"{DERIV_WS_URL}?app_id={self._app_id}"
        log.info("Connecting to %s", url)
        self._ws      = websocket.create_connection(url, timeout=30, sslopt=_SSL_OPT)
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="deriv-reader"
        )
        self._reader_thread.start()
        resp = self._send_sync({"authorize": self._api_token})
        if "error" in resp:
            raise RuntimeError(f"Deriv auth failed: {resp['error']['message']}")
        acct = resp.get("authorize", {})
        self.account_id = acct.get("loginid")
        self.balance    = float(acct.get("balance", 0))
        self.currency   = acct.get("currency", "USD")
        log.info("Authorized as %s  balance=%.2f %s", self.account_id, self.balance, self.currency)

    def disconnect(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    @property
    def is_alive(self) -> bool:
        return (self._running
                and self._reader_thread is not None
                and self._reader_thread.is_alive())

    def reconnect(self):
        log.info("Reconnecting WebSocket…")
        self._running = False
        try:
            self._ws.close()
        except Exception:
            pass
        time.sleep(2)
        self.connect()

    def refresh_balance(self):
        resp = self._send_sync({"balance": 1})
        if "balance" in resp:
            self.balance = float(resp["balance"]["balance"])
        return self.balance

    # ── Internal reader ───────────────────────────────────────────────────────

    def _read_loop(self):
        while self._running:
            try:
                raw = self._ws.recv()
                if not raw:
                    continue
                msg = json.loads(raw)
            except Exception as exc:
                if self._running:
                    log.error("WS recv error: %s", exc)
                break
            rid   = msg.get("req_id")
            mtype = msg.get("msg_type", "")
            if mtype in self._type_cb:
                try:
                    self._type_cb[mtype](msg)
                except Exception as e:
                    log.error("Type callback error [%s]: %s", mtype, e)
            if rid and rid in self._stream_cb:
                try:
                    self._stream_cb[rid](msg)
                except Exception as e:
                    log.error("Stream callback error: %s", e)
            if rid and rid in self._pending:
                entry = self._pending[rid]
                entry["result"] = msg
                entry["event"].set()

    def _next_req_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def _send_sync(self, msg: dict, timeout: float = 30.0) -> dict:
        rid = self._next_req_id()
        msg["req_id"] = rid
        evt = threading.Event()
        self._pending[rid] = {"event": evt, "result": None}
        try:
            self._ws.send(json.dumps(msg))
        except Exception as exc:
            del self._pending[rid]
            raise RuntimeError(f"WS send failed: {exc}") from exc
        if not evt.wait(timeout):
            self._pending.pop(rid, None)
            raise TimeoutError(f"No response for req_id={rid} after {timeout}s")
        return self._pending.pop(rid)["result"]

    # ── Market data ───────────────────────────────────────────────────────────

    def get_candles(self, symbol: str, count: int = CANDLE_COUNT,
                    granularity: int = GRANULARITY) -> pd.DataFrame:
        """Fetch historical OHLCV candles. Returns DataFrame sorted oldest→newest."""
        resp = self._send_sync({
            "ticks_history": symbol,
            "end":           "latest",
            "count":         count,
            "style":         "candles",
            "granularity":   granularity,
        }, timeout=45)
        if "error" in resp:
            raise RuntimeError(f"ticks_history({symbol}) error: {resp['error']['message']}")
        candles = resp.get("candles", [])
        if not candles:
            raise ValueError(f"No candles returned for {symbol}")
        df = pd.DataFrame(candles)
        df = df.rename(columns={"epoch": "timestamp"})
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["funding_rate"] = 0.0
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_latest_spot(self, symbol: str) -> float:
        resp = self._send_sync({"ticks": symbol})
        if "error" in resp:
            raise RuntimeError(f"ticks({symbol}) error: {resp['error']['message']}")
        return float(resp["tick"]["quote"])

    def subscribe_ticks(self, symbol: str, callback) -> None:
        """
        Subscribe to live tick stream. Deriv pushes msg_type='tick' for all symbols
        with no per-symbol req_id — routed by symbol field in _multi_tick closure chain.
        """
        old_cb = self._type_cb.get("tick")

        def _multi_tick(msg):
            tick = msg.get("tick", {})
            sym  = tick.get("symbol", "")
            if sym == symbol:
                try:
                    callback(float(tick["quote"]), int(tick["epoch"]))
                except Exception as e:
                    log.debug("Tick callback error (%s): %s", symbol, e)
            elif old_cb:
                old_cb(msg)

        self._type_cb["tick"] = _multi_tick
        rid = self._next_req_id()
        msg = {"ticks": symbol, "subscribe": 1, "req_id": rid}
        evt = threading.Event()
        self._pending[rid] = {"event": evt, "result": None}
        self._ws.send(json.dumps(msg))
        evt.wait(10)
        self._pending.pop(rid, None)
        log.info("Subscribed to live ticks: %s", symbol)

    # ── Trading ───────────────────────────────────────────────────────────────

    def get_proposal(self, symbol: str, direction: str, stake: float,
                     multiplier: int, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> dict:
        """
        Get a price proposal for a Multiplier contract.
        direction: "up" → MULTUP, "down" → MULTDOWN
        SL/TP must be set on the proposal, not on buy (Deriv Multiplier API constraint).
        """
        contract_type = "MULTUP" if direction == "up" else "MULTDOWN"
        payload: dict = {
            "proposal":      1,
            "contract_type": contract_type,
            "amount":        round(stake, 2),
            "basis":         "stake",
            "multiplier":    multiplier,
            "symbol":        symbol,
            "currency":      "USD",
        }
        limit: dict = {}
        if stop_loss is not None:
            limit["stop_loss"]   = round(stop_loss, 2)
        if take_profit is not None:
            limit["take_profit"] = round(take_profit, 2)
        if limit:
            payload["limit_order"] = limit
        resp = self._send_sync(payload, timeout=15)
        if "error" in resp:
            raise RuntimeError(f"proposal({symbol} {direction}) error: {resp['error']['message']}")
        return resp["proposal"]

    def get_multiplier_choices(self, symbol: str) -> list[int]:
        resp = self._send_sync({"contracts_for": symbol}, timeout=20)
        if "error" in resp:
            raise RuntimeError(f"contracts_for({symbol}) error: {resp['error']['message']}")
        choices: set[int] = set()
        for contract in resp.get("contracts_for", {}).get("available", []):
            if contract.get("contract_type") not in ("MULTUP", "MULTDOWN"):
                continue
            for value in contract.get("multiplier_range", []):
                try:
                    choices.add(int(value))
                except (TypeError, ValueError):
                    continue
        if not choices:
            raise RuntimeError(f"contracts_for({symbol}) returned no multiplier_range")
        return sorted(choices)

    def buy_contract(self, proposal_id: str, price: float) -> dict:
        payload: dict = {
            "buy":   proposal_id,
            "price": round(price * 1.01, 2),  # +1% slippage tolerance
        }
        resp = self._send_sync(payload, timeout=15)
        if "error" in resp:
            raise RuntimeError(f"buy({proposal_id}) error: {resp['error']['message']}")
        return resp["buy"]

    def sell_contract(self, contract_id: int) -> dict:
        resp = self._send_sync({"sell": contract_id, "price": 0}, timeout=15)
        if "error" in resp:
            ec = resp["error"].get("code", "")
            if ec not in ("BI004", "InvalidContractProposalExpired"):
                raise RuntimeError(f"sell({contract_id}) error: {resp['error']['message']}")
            return {}
        return resp.get("sell", {})

    def get_open_contracts(self) -> list:
        resp = self._send_sync({"portfolio": 1}, timeout=15)
        if "error" in resp:
            return []
        return resp.get("portfolio", {}).get("contracts", [])


# ── Position tracker ──────────────────────────────────────────────────────────

@dataclass
class _Position:
    """Tracks the Deriv-specific state needed to close a V2 position."""
    contract_id: int
    direction: str    # "up" or "down"
    stake: float
    multiplier: int
    entry_spot: float
    v2_symbol: str    # e.g. "DERIV_V75"
    deriv_symbol: str # e.g. "R_75"


def _reconcile_deriv_positions(state_path: str, open_contracts: list[dict]) -> tuple[dict[str, _Position], bool]:
    """Rebuild Deriv positions from local state and flag unknown exchange contracts."""
    state = load_v2_shadow_state(state_path)
    open_ids = {int(c.get("contract_id", 0)) for c in open_contracts if c.get("contract_id")}
    positions: dict[str, _Position] = {}
    known_ids: set[int] = set()
    for sym, meta in state.position_meta.items():
        contract_id = int(meta.get("deriv_contract_id", 0) or 0)
        if contract_id <= 0 or contract_id not in open_ids:
            continue
        stake = float(meta.get("deriv_stake", 0.0) or 0.0)
        multiplier = int(meta.get("deriv_multiplier", DEFAULT_MULTIPLIERS.get(sym, 50)) or 50)
        direction = str(meta.get("deriv_direction", "up"))
        deriv_symbol = str(meta.get("deriv_symbol", DERIV_SYMBOL_MAP.get(sym, "")))
        entry_spot = float(state.entry_prices.get(sym, 0.0) or 0.0)
        positions[sym] = _Position(
            contract_id=contract_id,
            direction=direction,
            stake=stake,
            multiplier=multiplier,
            entry_spot=entry_spot,
            v2_symbol=sym,
            deriv_symbol=deriv_symbol,
        )
        known_ids.add(contract_id)
    unknown_contracts = bool(open_ids - known_ids)
    return positions, unknown_contracts


# ── Data cache helpers ────────────────────────────────────────────────────────

def _fetch_and_cache(client: DerivClient, v2_sym: str, deriv_sym: str) -> pd.DataFrame:
    """
    Fetch 1h candles from Deriv and upsert into the V2 parquet cache.
    Deriv timestamps are Unix seconds; upsert_cache expects milliseconds.
    """
    from v2.live_data import upsert_cache
    df = client.get_candles(deriv_sym, count=CANDLE_COUNT, granularity=GRANULARITY)
    df_ms = df.copy()
    df_ms["timestamp"] = df_ms["timestamp"].astype("int64") * 1000  # seconds → ms
    upsert_cache(v2_sym, df_ms)
    log.debug("Cached %d bars for %s (%s)", len(df_ms), v2_sym, deriv_sym)
    return df


# ── Execution helpers ─────────────────────────────────────────────────────────

def _execute_opens(
    client: DerivClient,
    opens: list[dict],
    multipliers: dict[str, int],
    balance: float,
    open_positions: dict[str, "_Position"],
    dry_run: bool,
) -> dict[str, "_Position"]:
    """
    Open new Deriv Multiplier contracts for each V2 open signal.
    Returns a dict of newly opened positions (v2_symbol → _Position).
    """
    new_positions: dict[str, _Position] = {}
    for action in opens:
        v2_sym    = action["symbol"]
        deriv_sym = DERIV_SYMBOL_MAP.get(v2_sym)
        if deriv_sym is None:
            log.warning("No Deriv symbol for V2 symbol %s — skipping open", v2_sym)
            continue
        if v2_sym in open_positions:
            log.info("  SKIP OPEN  %-12s already has open contract %d",
                     v2_sym, open_positions[v2_sym].contract_id)
            continue

        multiplier = multipliers.get(v2_sym, DEFAULT_MULTIPLIERS.get(v2_sym, 50))
        target_notional = abs(float(action["target_notional_usd"]))
        # Stake = target notional / multiplier, capped at MAX_STAKE_PCT of balance
        stake = max(round(min(target_notional / multiplier, balance * MAX_STAKE_PCT), 2), 1.0)
        direction = "up" if float(action["side"]) > 0 else "down"
        sl_amt = round(stake * SL_PCT, 2)
        tp_amt = round(stake * TP_PCT, 2)

        log.info("  V2 OPEN  %-12s  %s  notional=$%.0f → stake=$%.2f  x%d  SL=$%.2f  TP=$%.2f",
                 v2_sym, direction.upper(), target_notional, stake, multiplier, sl_amt, tp_amt)

        if dry_run:
            continue
        try:
            prop   = client.get_proposal(deriv_sym, direction, stake, multiplier,
                                         stop_loss=sl_amt, take_profit=tp_amt)
            price  = float(prop["ask_price"])
            bought = client.buy_contract(prop["id"], price)
            cid    = int(bought["contract_id"])
            spot   = float(bought.get("spot", 0))
            log.info("  OPENED  %-12s  contract=%d  entry_spot=%.5f", v2_sym, cid, spot)
            new_positions[v2_sym] = _Position(
                contract_id=cid,
                direction=direction,
                stake=stake,
                multiplier=multiplier,
                entry_spot=spot,
                v2_symbol=v2_sym,
                deriv_symbol=deriv_sym,
            )
        except Exception as exc:
            log.error("  OPEN FAILED  %-12s: %s", v2_sym, exc)

    return new_positions


def _execute_exits(
    client: DerivClient,
    exits: list[dict],
    open_positions: dict[str, "_Position"],
    dry_run: bool,
) -> list[str]:
    """
    Close Deriv contracts for each V2 exit signal.
    Returns list of V2 symbols successfully closed.
    """
    closed: list[str] = []
    for action in exits:
        v2_sym = action["symbol"]
        pos = open_positions.get(v2_sym)
        if pos is None:
            log.debug("  EXIT %s: no tracked contract — already closed or never opened", v2_sym)
            closed.append(v2_sym)
            continue
        reason = action.get("reason", "unknown")
        log.info("  V2 EXIT  %-12s  reason=%s  contract=%d", v2_sym, reason, pos.contract_id)
        if dry_run:
            closed.append(v2_sym)
            continue
        try:
            sold = client.sell_contract(pos.contract_id)
            sold_for = float(sold.get("sold_for", 0)) if sold else 0.0
            log.info("  CLOSED  %-12s  contract=%d  sold_for=$%.2f",
                     v2_sym, pos.contract_id, sold_for)
            closed.append(v2_sym)
        except Exception as exc:
            log.error("  EXIT FAILED  %-12s  contract=%d: %s",
                      v2_sym, pos.contract_id, exc)
    return closed


# ── Main run loop ─────────────────────────────────────────────────────────────

_RESET = "\033[0m"; _BOLD = "\033[1m"; _GREEN = "\033[92m"
_RED   = "\033[91m"; _CYN  = "\033[96m"; _YLW  = "\033[93m"; _DIM = "\033[2m"


def _cpnl(v: float) -> str:
    s = f"${v:+.2f}"
    return f"{_GREEN}{s}{_RESET}" if v >= 0 else f"{_RED}{s}{_RESET}"


def _render_dashboard(client, open_positions, live_spots, last_bar_ts, bar_count, next_bar_ts):
    if not IS_TTY:
        return
    W   = 70
    now = datetime.now(tz=timezone.utc)
    nb  = datetime.fromtimestamp(next_bar_ts, tz=timezone.utc).strftime("%H:%M:%S UTC")
    sys.stdout.write("\033[H\033[J")
    print(f"{_BOLD}{'═'*W}{_RESET}")
    print(f"{_BOLD}  V2 DERIV TRADER{_RESET}  │  {_CYN}{now.strftime('%Y-%m-%d %H:%M:%S')} UTC{_RESET}")
    print(f"  Account: {_CYN}{client.account_id}{_RESET}   "
          f"Balance: {_BOLD}${client.balance:,.2f}{_RESET}   "
          f"Positions: {len(open_positions)}   "
          f"Next bar: {_YLW}{nb}{_RESET}  #{bar_count}")
    if last_bar_ts:
        bar_dt = datetime.fromtimestamp(last_bar_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  Last bar: {_DIM}{bar_dt} UTC{_RESET}")
    print(f"{_DIM}{'─'*W}{_RESET}")
    print(f"{_BOLD}  LIVE PRICES{_RESET}")
    for v2_sym, deriv_sym in DERIV_SYMBOL_MAP.items():
        price = live_spots.get(v2_sym)
        p_str = f"{price:,.4f}" if price else "…"
        print(f"    {_BOLD}{v2_sym:<14}{_RESET} [{deriv_sym:<8}]  {_CYN}{p_str}{_RESET}")
    print(f"{_DIM}{'─'*W}{_RESET}")
    print(f"{_BOLD}  OPEN POSITIONS  "
          f"{_DIM}(SL={SL_PCT*100:.0f}% · TP={TP_PCT*100:.0f}% of stake){_RESET}")
    if not open_positions:
        print(f"    {_DIM}none{_RESET}")
    else:
        print(f"    {_BOLD}{'V2 SYM':<14}{'DIR':<7}{'STAKE':>8}{'ENTRY':>14}{'NOW':>14}{'P&L':>9}{_RESET}")
        for v2_sym, pos in open_positions.items():
            cur = live_spots.get(v2_sym, pos.entry_spot)
            pct = (cur / pos.entry_spot - 1) if pos.entry_spot else 0.0
            if pos.direction == "down":
                pct = -pct
            raw_pnl = pct * pos.stake * pos.multiplier
            d_tag = (f"{_GREEN}▲ UP  {_RESET}" if pos.direction == "up"
                     else f"{_RED}▼ DOWN{_RESET}")
            print(f"    {_BOLD}{v2_sym:<14}{_RESET}{d_tag}"
                  f"  ${pos.stake:>7.2f}"
                  f"  {pos.entry_spot:>13.5f}"
                  f"  {cur:>13.5f}"
                  f"  {_cpnl(raw_pnl)}")
    print(f"{_BOLD}{'═'*W}{_RESET}")
    sys.stdout.flush()


def run(
    portfolio_config_path: str,
    model_set: str,
    bundle_name: str,
    state_path: str,
    dry_run: bool,
    active_sleeves: list[str] | None = None,
):
    # ── Load V2 config ────────────────────────────────────────────────────────
    with open(portfolio_config_path) as f:
        raw_cfg = json.load(f)

    from v2.types import PortfolioConfig
    from v2.live_data import compute_live_actions, update_state_after_execution

    portfolio_config = PortfolioConfig.from_dict(raw_cfg.get("portfolio", raw_cfg))
    sleeves = active_sleeves or raw_cfg.get("active_sleeves", [])
    symbols = list(DERIV_SYMBOL_MAP.keys())

    log.info("V2 Deriv trader starting%s", " [DRY-RUN]" if dry_run else "")
    log.info("  bundle=%s  model_set=%s  symbols=%s", bundle_name, model_set, symbols)
    log.info("  sleeves=%d  state=%s", len(sleeves), state_path)

    # ── Connect ───────────────────────────────────────────────────────────────
    api_token = os.environ.get("DERIV_API_TOKEN", "")
    app_id    = os.environ.get("DERIV_APP_ID", DEFAULT_APP_ID)
    if not app_id.strip().isdigit():
        log.warning("DERIV_APP_ID '%s' not numeric — using '1'", app_id)
        app_id = "1"
    if not api_token and not dry_run:
        log.error("DERIV_API_TOKEN not set."); sys.exit(1)

    def _connect() -> DerivClient:
        delay = 5
        while True:
            try:
                c = DerivClient(api_token or "demo", app_id)
                c.connect()
                return c
            except Exception as exc:
                log.error("Connect failed: %s — retry in %ds", exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)

    client = _connect()

    # ── Resolve multipliers ───────────────────────────────────────────────────
    multipliers: dict[str, int] = {}
    for v2_sym, deriv_sym in DERIV_SYMBOL_MAP.items():
        fallback = DEFAULT_MULTIPLIERS[v2_sym]
        try:
            choices = client.get_multiplier_choices(deriv_sym)
            chosen  = min(choices)
            multipliers[v2_sym] = chosen
            log.info("Multiplier %-14s allowed=%s  using x%d", v2_sym, choices, chosen)
        except Exception as exc:
            multipliers[v2_sym] = fallback
            log.warning("Multiplier lookup failed %-14s (%s) — fallback x%d",
                        v2_sym, exc, fallback)

    # ── Shared state ──────────────────────────────────────────────────────────
    live_spots:   dict[str, float] = {}
    last_bar_ts:  list[Optional[int]] = [None]
    bar_count     = 0
    next_bar_ts   = (int(time.time() / BAR_SECONDS) + 1) * BAR_SECONDS + BAR_SLEEP_BUFFER
    _bar_lock     = threading.Lock()
    _last_bar_id  = [0]

    # In-memory Deriv contract tracking, rebuilt from persisted state on startup.
    try:
        open_positions, halt_new_orders = _reconcile_deriv_positions(state_path, client.get_open_contracts())
    except Exception as exc:
        log.warning("Deriv startup reconciliation failed: %s — halting new orders", exc)
        open_positions, halt_new_orders = {}, True
    if halt_new_orders:
        log.warning("Unknown Deriv open contracts or reconciliation failure; exits/reconciliation only")
    elif open_positions:
        log.info("Reconciled Deriv positions: %s", list(open_positions))

    # ── Bar processing ────────────────────────────────────────────────────────
    def _process_bar(_trigger_sym: str):
        nonlocal bar_count, next_bar_ts

        current_bar_id = int(time.time() // BAR_SECONDS)
        with _bar_lock:
            if current_bar_id <= _last_bar_id[0]:
                return
            _last_bar_id[0] = current_bar_id

        bar_count += 1
        next_bar_ts = (int(time.time() / BAR_SECONDS) + 1) * BAR_SECONDS + BAR_SLEEP_BUFFER
        log.info("─── Bar %d  %s UTC  balance=$%.2f",
                 bar_count,
                 datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                 client.balance)

        try:
            # 1. Refresh parquet cache with latest 1h bars
            for v2_sym, deriv_sym in DERIV_SYMBOL_MAP.items():
                try:
                    _fetch_and_cache(client, v2_sym, deriv_sym)
                except Exception as exc:
                    log.warning("  Cache refresh failed %-14s: %s", v2_sym, exc)

            # 2. Run V2 signal engine
            result = compute_live_actions(
                bundle_name=bundle_name,
                model_set=model_set,
                portfolio_config=portfolio_config,
                active_sleeves=sleeves,
                symbols=symbols,
                state_path=state_path,
                equity_override=float(client.balance),
                halt_new_orders=halt_new_orders,
            )
            latest_ts = result.get("latest_ts")
            opens     = result.get("opens", [])
            exits     = result.get("exits", [])
            equity    = result.get("equity", client.balance)
            close_prices = result.get("close_prices", {})

            if latest_ts:
                last_bar_ts[0] = int(latest_ts) // 1000  # ms → s for display

            log.info("  V2 signals  opens=%d  exits=%d  equity=%.2f  ts=%s",
                     len(opens), len(exits), equity, latest_ts)

            # 3. Execute exits first
            closed = _execute_exits(client, exits, open_positions, dry_run)
            for sym in closed:
                open_positions.pop(sym, None)

            # 4. Execute opens
            client.refresh_balance()
            new_pos = _execute_opens(
                client, opens, multipliers, client.balance, open_positions, dry_run
            )
            open_positions.update(new_pos)

            # 5. Persist state to V2ShadowState
            positions: dict[str, float] = {}
            entry_prices: dict[str, float] = {}
            position_meta: dict[str, dict] = {}
            for sym, pos in open_positions.items():
                notional = pos.stake * pos.multiplier
                positions[sym]     = notional if pos.direction == "up" else -notional
                entry_prices[sym]  = pos.entry_spot
                # Persist Deriv contract details inside V2 meta so monitor loop can close
                position_meta[sym] = {
                    "deriv_contract_id": pos.contract_id,
                    "deriv_multiplier":  pos.multiplier,
                    "deriv_stake":       pos.stake,
                    "deriv_direction":   pos.direction,
                    "deriv_symbol":      pos.deriv_symbol,
                    "opened_ts":         latest_ts or 0,
                }

            ts_ms = int(latest_ts) if latest_ts else int(time.time() * 1000)
            update_state_after_execution(
                state_path=state_path,
                timestamp=ts_ms,
                equity=float(client.balance),
                positions=positions,
                entry_prices=entry_prices,
                position_meta=position_meta,
            )

        except Exception as exc:
            log.exception("Bar processing error: %s", exc)

    # ── Tick callbacks ────────────────────────────────────────────────────────
    def _make_tick_cb(v2_sym: str):
        def _on_tick(price: float, epoch: int):
            live_spots[v2_sym] = price
            cur_bar = int(epoch // BAR_SECONDS)
            prev    = _last_bar_id[0]
            if cur_bar > prev:
                log.info("Bar boundary detected by tick (%s)  epoch=%d", v2_sym, epoch)
                time.sleep(BAR_SLEEP_BUFFER)
                threading.Thread(
                    target=_process_bar, args=(v2_sym,),
                    daemon=True, name=f"bar-{v2_sym}"
                ).start()
        return _on_tick

    # ── Subscribe and start ───────────────────────────────────────────────────
    def _subscribe_all():
        for v2_sym, deriv_sym in DERIV_SYMBOL_MAP.items():
            _last_bar_id[0] = max(_last_bar_id[0], int(time.time() // BAR_SECONDS) - 1)
            try:
                client.subscribe_ticks(deriv_sym, _make_tick_cb(v2_sym))
            except Exception as exc:
                log.error("subscribe_ticks(%s) failed: %s", deriv_sym, exc)

    _subscribe_all()

    # ── SL/TP monitor + auto-reconnect (polls every 60s) ─────────────────────
    def _monitor_loop():
        nonlocal halt_new_orders
        while True:
            time.sleep(60)
            if not client.is_alive:
                log.warning("WS connection lost — reconnecting…")
                try:
                    client.reconnect()
                    _subscribe_all()
                    reconciled, mismatch = _reconcile_deriv_positions(state_path, client.get_open_contracts())
                    open_positions.clear()
                    open_positions.update(reconciled)
                    if mismatch:
                        halt_new_orders = True
                        log.warning("Unknown Deriv contracts after reconnect; halting new orders")
                    log.info("Reconnected and re-subscribed")
                except Exception as exc:
                    log.error("Reconnect failed: %s", exc)
                continue
            if not open_positions:
                continue
            try:
                open_cids = {int(c["contract_id"]) for c in client.get_open_contracts()}
                sl_tp_hit = [
                    sym for sym, pos in list(open_positions.items())
                    if not dry_run and pos.contract_id not in open_cids
                ]
                if sl_tp_hit:
                    client.refresh_balance()
                    # Remove closed positions from state
                    remaining_pos = {
                        sym: pos for sym, pos in open_positions.items()
                        if sym not in sl_tp_hit
                    }
                    positions: dict[str, float] = {}
                    entry_prices: dict[str, float] = {}
                    position_meta: dict[str, dict] = {}
                    for sym, pos in remaining_pos.items():
                        notional = pos.stake * pos.multiplier
                        positions[sym]     = notional if pos.direction == "up" else -notional
                        entry_prices[sym]  = pos.entry_spot
                        position_meta[sym] = {
                            "deriv_contract_id": pos.contract_id,
                            "deriv_multiplier":  pos.multiplier,
                            "deriv_stake":       pos.stake,
                            "deriv_direction":   pos.direction,
                            "deriv_symbol":      pos.deriv_symbol,
                            "opened_ts":         0,
                        }
                    for sym in sl_tp_hit:
                        pos = open_positions.pop(sym, None)
                        if pos:
                            log.info("  SL/TP hit  %-14s  contract=%d", sym, pos.contract_id)
                    update_state_after_execution(
                        state_path=state_path,
                        timestamp=int(time.time() * 1000),
                        equity=float(client.balance),
                        positions=positions,
                        entry_prices=entry_prices,
                        position_meta=position_meta,
                    )
                    log.info("  Balance after SL/TP: $%.2f", client.balance)
            except Exception as exc:
                log.warning("Monitor error: %s", exc)

    threading.Thread(target=_monitor_loop, daemon=True, name="monitor").start()
    log.info("Tick subscriptions active. Monitor running (poll 60s).")

    # ── Main loop: live display ───────────────────────────────────────────────
    while True:
        _render_dashboard(client, open_positions, live_spots,
                          last_bar_ts[0], bar_count, next_bar_ts)
        time.sleep(DISPLAY_INTERVAL)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="V2 Deriv synthetic index live trader")
    ap.add_argument(
        "--portfolio-config",
        default="v2_portfolio.wave5_quality13_pesfix.json",
        help="Path to V2 portfolio config JSON (default: quality13_pesfix)",
    )
    ap.add_argument(
        "--model-set",
        default="v2_wave5_extended",
        help="V2 model set name (default: v2_wave5_extended)",
    )
    ap.add_argument(
        "--bundle",
        default="bundle_intraday_core",
        help="V2 bundle name (default: bundle_intraday_core)",
    )
    ap.add_argument(
        "--state-path",
        default="state/v2_deriv_live.json",
        help="Path to V2 shadow state JSON (created if absent)",
    )
    ap.add_argument(
        "--log-path",
        default="logs/v2_deriv.log",
        help="Structured operator log path",
    )
    ap.add_argument(
        "--active-sleeves",
        default="",
        help="Comma-separated sleeve names (default: use portfolio config list)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data and compute signals but place NO orders",
    )
    args = ap.parse_args()

    if not os.path.exists(args.portfolio_config):
        log.error("Portfolio config not found: %s", args.portfolio_config)
        sys.exit(1)

    sleeves = (
        [s.strip() for s in args.active_sleeves.split(",") if s.strip()]
        if args.active_sleeves else None
    )

    os.makedirs("logs", exist_ok=True)
    configure_file_logging(args.log_path)
    state_dir = os.path.dirname(os.path.abspath(args.state_path))
    os.makedirs(state_dir, exist_ok=True)

    run(
        portfolio_config_path=args.portfolio_config,
        model_set=args.model_set,
        bundle_name=args.bundle,
        state_path=args.state_path,
        dry_run=args.dry_run,
        active_sleeves=sleeves,
    )


if __name__ == "__main__":
    main()
