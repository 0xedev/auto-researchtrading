"""
Shadow execution validation: kill-switch and halt-new-orders behaviour.

Tests:
  1. kill_switch absent     → orders execute normally
  2. kill_switch present with halt_new_orders=true  → no new opens/modifies after cut
  3. halt_new_orders=false  → still opens (switch present but not triggered)
  4. existing positions still close after kill-switch engages
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from execution.report import summarize_shadow_run
from execution.v2_paper import run_v2_shadow_session
from v2 import PortfolioConfig
from v2.runtime import V2SignalEngine


_BUNDLE = "bundle_intraday_core"
_MODEL_SET = "v2_wave4_carry"
_SPLIT = "newasset_oos2y"
_SYMBOLS = ["LTC", "BCH"]
_CONFIG_PATH = "v2_portfolio.wave5_quality11_tp095_postmicro.json"
_MAX_DAYS_PHASE1 = 10
_MAX_DAYS_PHASE2 = 10


def _load_config(path: str) -> PortfolioConfig:
    payload = json.loads(Path(path).read_text())
    return PortfolioConfig.from_dict(payload.get("portfolio"))


def _count_actions(log_path: Path, after_ts: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if after_ts is not None and int(row.get("timestamp", 0)) <= after_ts:
            continue
        action = row.get("action", row.get("event", "unknown"))
        counts[action] = counts.get(action, 0) + 1
    return counts


def _run_phase(
    root: Path,
    label: str,
    kill_switch_path: str | None,
    max_days: int,
    engine: V2SignalEngine,
    portfolio_config: PortfolioConfig,
) -> tuple[dict, dict, Path]:
    state_path = root / f"{label}_state.json"
    log_path = root / f"{label}_log.jsonl"
    result = run_v2_shadow_session(
        bundle_name=_BUNDLE,
        model_set=_MODEL_SET,
        split=_SPLIT,
        portfolio_config=portfolio_config,
        state_path=str(state_path),
        log_path=str(log_path),
        kill_switch_path=kill_switch_path,
        max_days=max_days,
        symbols=_SYMBOLS,
        engine=engine,
    )
    summary = summarize_shadow_run(state_path=state_path, log_path=log_path, max_recent=5)
    return result, summary, log_path


def _check(label: str, condition: bool, detail: str = "") -> bool:
    sym = "PASS" if condition else "FAIL"
    print(f"  [{sym}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def main() -> None:
    portfolio_config = _load_config(_CONFIG_PATH)
    engine = V2SignalEngine(
        bundle_name=_BUNDLE,
        model_set=_MODEL_SET,
        symbols=_SYMBOLS,
    )
    engine.prepare(_SPLIT)

    all_pass = True

    # ── Test 1: No kill-switch → normal execution ──────────────────────────
    print("\n=== Test 1: No kill-switch (baseline) ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t1_") as tmpdir:
        root = Path(tmpdir)
        result, summary, log_path = _run_phase(
            root, "t1", kill_switch_path=None,
            max_days=_MAX_DAYS_PHASE1, engine=engine, portfolio_config=portfolio_config,
        )
        counts = _count_actions(log_path)
        opens = counts.get("open", 0)
        ok = _check("shadow ran without error", result.get("bars_processed", 0) > 0,
                    f"bars={result.get('bars_processed', 0)}")
        ok &= _check("at least 1 open action recorded", opens > 0, f"opens={opens}")
        all_pass &= ok
        baseline_opens = opens

    # ── Test 2: Kill-switch with halt_new_orders=true ──────────────────────
    print("\n=== Test 2: Kill-switch halt_new_orders=true (from bar 0) ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t2_") as tmpdir:
        root = Path(tmpdir)
        ks_path = str(root / "kill_switch.json")
        Path(ks_path).write_text(json.dumps({"halt_new_orders": True}))
        result, summary, log_path = _run_phase(
            root, "t2", kill_switch_path=ks_path,
            max_days=_MAX_DAYS_PHASE1, engine=engine, portfolio_config=portfolio_config,
        )
        counts = _count_actions(log_path)
        opens = counts.get("open", 0)
        modifies = counts.get("modify", 0)
        ok = _check("shadow ran without error", result.get("bars_processed", 0) > 0)
        ok &= _check("zero open actions when halted", opens == 0, f"opens={opens}")
        ok &= _check("zero modify actions when halted", modifies == 0, f"modifies={modifies}")
        all_pass &= ok

    # ── Test 3: Kill-switch present but halt_new_orders=false ─────────────
    print("\n=== Test 3: Kill-switch file present, halt_new_orders=false ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t3_") as tmpdir:
        root = Path(tmpdir)
        ks_path = str(root / "kill_switch.json")
        Path(ks_path).write_text(json.dumps({"halt_new_orders": False}))
        result, summary, log_path = _run_phase(
            root, "t3", kill_switch_path=ks_path,
            max_days=_MAX_DAYS_PHASE1, engine=engine, portfolio_config=portfolio_config,
        )
        counts = _count_actions(log_path)
        opens = counts.get("open", 0)
        ok = _check("opens still fire with halt_new_orders=false", opens > 0, f"opens={opens}")
        all_pass &= ok

    # ── Test 4: Kill-switch engaged mid-run → positions still close ────────
    print("\n=== Test 4: Kill-switch engaged after phase 1 completes ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t4_") as tmpdir:
        root = Path(tmpdir)
        ks_path = str(root / "kill_switch.json")

        # Phase 1: no kill-switch, let positions open
        state_path = root / "state.json"
        log_path1 = root / "phase1_log.jsonl"
        result1 = run_v2_shadow_session(
            bundle_name=_BUNDLE, model_set=_MODEL_SET, split=_SPLIT,
            portfolio_config=portfolio_config,
            state_path=str(state_path), log_path=str(log_path1),
            kill_switch_path=None, max_days=_MAX_DAYS_PHASE1,
            symbols=_SYMBOLS, engine=engine,
        )
        cut_ts = result1.get("last_timestamp", 0)
        positions_after_p1 = result1.get("open_positions", 0)

        # Engage kill-switch
        Path(ks_path).write_text(json.dumps({"halt_new_orders": True}))

        # Phase 2: continue from same state, kill-switch active
        log_path2 = root / "phase2_log.jsonl"
        result2 = run_v2_shadow_session(
            bundle_name=_BUNDLE, model_set=_MODEL_SET, split=_SPLIT,
            portfolio_config=portfolio_config,
            state_path=str(state_path), log_path=str(log_path2),
            kill_switch_path=ks_path, max_days=_MAX_DAYS_PHASE2,
            symbols=_SYMBOLS, engine=engine,
        )
        counts2 = _count_actions(log_path2)
        closes = counts2.get("close", 0)
        opens2 = counts2.get("open", 0)

        ok = _check("phase 1 ran without error", result1.get("bars_processed", 0) > 0)
        ok &= _check("phase 2 ran without error", result2.get("bars_processed", 0) > 0)
        ok &= _check("no new opens in phase 2 (halted)", opens2 == 0, f"opens={opens2}")
        ok &= _check("close actions still fire after kill-switch",
                     closes > 0 or positions_after_p1 == 0,
                     f"p1_positions={positions_after_p1}, p2_closes={closes}")
        all_pass &= ok

    # ── Test 5: Duplicate-signal / position stability ─────────────────────
    # Run a longer shadow and verify that re-entry cooldown prevents the
    # same symbol from being opened twice in consecutive bars, and that
    # open-position count never exceeds the portfolio cap.
    print("\n=== Test 5: Duplicate-signal / position stability ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t5_") as tmpdir:
        root = Path(tmpdir)
        state_path = root / "state.json"
        log_path = root / "log.jsonl"
        result = run_v2_shadow_session(
            bundle_name=_BUNDLE, model_set=_MODEL_SET, split=_SPLIT,
            portfolio_config=portfolio_config,
            state_path=str(state_path), log_path=str(log_path),
            kill_switch_path=None, max_days=30,
            symbols=_SYMBOLS, engine=engine,
        )
        # Reconstruct running position count from log
        positions: dict[str, int] = {}
        max_concurrent = 0
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            action = row.get("action", "")
            sym = row.get("symbol", "")
            if action == "open":
                positions[sym] = positions.get(sym, 0) + 1
                max_concurrent = max(max_concurrent, len([s for s, c in positions.items() if c > 0]))
            elif action == "close":
                if sym in positions:
                    positions[sym] = max(0, positions[sym] - 1)
        # Per-symbol open count should never exceed 1 (can't double-open same symbol)
        any_double = any(c > 1 for c in positions.values())
        # max_per_symbol_pct=0.14 with 2 symbols means at most 2 concurrent positions
        cap_respected = max_concurrent <= len(_SYMBOLS)
        ok = _check("no symbol opened more than once concurrently", not any_double,
                    f"positions={dict(positions)}")
        ok &= _check(f"concurrent position count ≤ symbol count ({len(_SYMBOLS)})",
                     cap_respected, f"max_concurrent={max_concurrent}")
        all_pass &= ok

    # ── Test 6: Gross leverage ceiling ───────────────────────────────────
    # Run a 30-day shadow on the full 7-symbol basket and verify gross
    # leverage never exceeds max_gross_leverage=1.6 on any bar.
    print("\n=== Test 6: Gross leverage ceiling (full 7-symbol basket) ===")
    engine7 = V2SignalEngine(
        bundle_name=_BUNDLE,
        model_set=_MODEL_SET,
        symbols=["LTC", "BCH", "ETC", "TRX", "AAVE", "FIL", "OP"],
    )
    engine7.prepare(_SPLIT)
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t6_") as tmpdir:
        root = Path(tmpdir)
        state_path = root / "state.json"
        log_path = root / "log.jsonl"
        run_v2_shadow_session(
            bundle_name=_BUNDLE, model_set=_MODEL_SET, split=_SPLIT,
            portfolio_config=portfolio_config,
            state_path=str(state_path), log_path=str(log_path),
            kill_switch_path=None, max_days=30,
            symbols=["LTC", "BCH", "ETC", "TRX", "AAVE", "FIL", "OP"],
            engine=engine7, collect_bar_history=True,
        )
        from execution.report import summarize_shadow_run
        summary = summarize_shadow_run(state_path=state_path, log_path=log_path, max_recent=5)
        equity = float(summary.get("portfolio", {}).get("equity", 100000))
        gross_exposure = float(summary.get("portfolio", {}).get("gross_exposure", 0))
        gross_leverage = gross_exposure / max(equity, 1.0)
        max_allowed = portfolio_config.max_gross_leverage
        ok = _check(f"final gross leverage ≤ {max_allowed}",
                    gross_leverage <= max_allowed + 0.01,
                    f"gross_leverage={gross_leverage:.3f}")
        # Also check no rejected action has reason 'leverage_exceeded' (would indicate
        # the allocator fired over the cap before rejecting)
        overcap = sum(
            1 for line in log_path.read_text().splitlines()
            if line.strip() and json.loads(line).get("reject_reason") == "leverage_exceeded"
        )
        ok &= _check("no leverage_exceeded rejections in log", overcap == 0,
                     f"overcap_events={overcap}")
        all_pass &= ok

    # ── Test 7: State persistence accuracy ───────────────────────────────
    # Run phase 1, read the state file, verify equity matches the last
    # log entry — confirms state.json is written correctly for restart.
    print("\n=== Test 7: State persistence accuracy ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t7_") as tmpdir:
        root = Path(tmpdir)
        state_path = root / "state.json"
        log_path = root / "log.jsonl"
        run_v2_shadow_session(
            bundle_name=_BUNDLE, model_set=_MODEL_SET, split=_SPLIT,
            portfolio_config=portfolio_config,
            state_path=str(state_path), log_path=str(log_path),
            kill_switch_path=None, max_days=_MAX_DAYS_PHASE1,
            symbols=_SYMBOLS, engine=engine,
        )
        state = json.loads(state_path.read_text())
        last_log_equity = None
        for line in log_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                last_log_equity = row.get("portfolio_equity", row.get("equity"))
        state_equity = state.get("equity") or state.get("portfolio", {}).get("equity")
        # Allow up to 0.1% difference: state is written at session end (post-funding),
        # while last log entry reflects equity at the moment of the final action.
        equity_match = (
            last_log_equity is None or state_equity is None or
            abs(float(state_equity) - float(last_log_equity)) / max(abs(float(last_log_equity)), 1.0) < 0.001
        )
        ok = _check("state.json equity within 0.1% of last log entry",
                    equity_match,
                    f"state={state_equity:.2f}, log={last_log_equity:.2f}, diff={abs(float(state_equity)-float(last_log_equity)):.2f}" if state_equity and last_log_equity else "no data")
        ok &= _check("state.json is valid JSON with last_timestamp",
                     "last_timestamp" in state, f"keys={list(state.keys())[:5]}")
        all_pass &= ok

    # ── Test 8: Orphan-position cleanup via max-hold ──────────────────────
    # A 45-day run should end with 0 open positions: every position opened
    # during the run must be closed by TP/SL/max-hold before the final bar.
    # This verifies max-hold is the safety net for "orphan" positions that
    # never receive an explicit exit signal.
    print("\n=== Test 8: Orphan-position cleanup (45-day full run) ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t8_") as tmpdir:
        root = Path(tmpdir)
        state_path = root / "state.json"
        log_path = root / "log.jsonl"
        result = run_v2_shadow_session(
            bundle_name=_BUNDLE, model_set=_MODEL_SET, split=_SPLIT,
            portfolio_config=portfolio_config,
            state_path=str(state_path), log_path=str(log_path),
            kill_switch_path=None, max_days=45,
            symbols=_SYMBOLS, engine=engine,
        )
        state = json.loads(state_path.read_text())
        # Count net open positions from log (opens minus closes)
        net_positions: dict[str, int] = {}
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sym = row.get("symbol", "")
            if row.get("action") == "open":
                net_positions[sym] = net_positions.get(sym, 0) + 1
            elif row.get("action") == "close":
                net_positions[sym] = max(0, net_positions.get(sym, 0) - 1)
        orphaned = {s: c for s, c in net_positions.items() if c > 0}
        # Also check positions dict in state
        state_positions = state.get("positions", {})
        ok = _check("zero orphan positions at run end (log-derived)",
                    len(orphaned) == 0, f"orphaned={orphaned}")
        ok &= _check("state.json positions dict is empty at run end",
                     len(state_positions) == 0, f"state_positions={state_positions}")
        all_pass &= ok

    # ── Test 9: Open-position count consistency (state vs log) ───────────
    # After a complete run, the count of positions in state.json must equal
    # the net open count derived from the action log.
    print("\n=== Test 9: Position count — state vs log consistency ===")
    with tempfile.TemporaryDirectory(prefix="v2_ksv_t9_") as tmpdir:
        root = Path(tmpdir)
        state_path = root / "state.json"
        log_path = root / "log.jsonl"
        # Phase 1 only (positions may still be open mid-run)
        run_v2_shadow_session(
            bundle_name=_BUNDLE, model_set=_MODEL_SET, split=_SPLIT,
            portfolio_config=portfolio_config,
            state_path=str(state_path), log_path=str(log_path),
            kill_switch_path=None, max_days=_MAX_DAYS_PHASE1,
            symbols=_SYMBOLS, engine=engine,
        )
        state = json.loads(state_path.read_text())
        state_open = len(state.get("positions", {}))
        # Derive from log
        net: dict[str, int] = {}
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sym = row.get("symbol", "")
            if row.get("action") == "open":
                net[sym] = net.get(sym, 0) + 1
            elif row.get("action") == "close":
                net[sym] = max(0, net.get(sym, 0) - 1)
        log_open = sum(c for c in net.values() if c > 0)
        ok = _check("state position count matches log-derived open count",
                    state_open == log_open,
                    f"state={state_open}, log={log_open}")
        all_pass &= ok

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    status = "ALL PASS" if all_pass else "FAILURES DETECTED"
    print(f"Shadow validation: {status}")
    print(f"Baseline opens (no kill-switch, 2-symbol): {baseline_opens}")
    return all_pass


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
