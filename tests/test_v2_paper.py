import json
import tempfile
import unittest
from pathlib import Path

import prepare

from execution.v2_paper import V2ShadowState, run_v2_shadow_session, save_v2_shadow_state
from v2.types import PortfolioConfig
from v2.types import SleeveSignal


class _FakeV2Engine:
    def __init__(self, timestamps, close_by_timestamp, signals_by_timestamp=None, funding_by_timestamp=None):
        self.timestamps = timestamps
        self._close_by_timestamp = close_by_timestamp
        self._signals_by_timestamp = signals_by_timestamp or {}
        self._funding_by_timestamp = funding_by_timestamp or {}
        self.sleeve_tables = {}

    def signals_at_timestamp(self, timestamp):
        return list(self._signals_by_timestamp.get(timestamp, []))

    def close_by_symbol_at_timestamp(self, timestamp):
        return self._close_by_timestamp.get(timestamp, {})

    def funding_by_symbol_at_timestamp(self, timestamp):
        return self._funding_by_timestamp.get(timestamp, {})


class V2PaperTests(unittest.TestCase):
    def test_max_hold_close_applies_slippage_and_fee(self):
        opened_ts = 1_700_000_000_000
        close_ts = opened_ts + 7_200_000
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            log_path = root / "log.jsonl"
            config = PortfolioConfig(slippage_bps=0.0, tier_a_notional_pct=0.001, tier_b_notional_pct=0.001)

            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    cash=prepare.INITIAL_CAPITAL - 100.0,
                    positions={"BTC": 100.0},
                    entry_prices={"BTC": 100.0},
                    position_fee_basis={"BTC": 100.0 * prepare.TAKER_FEE},
                    position_meta={
                        "BTC": {
                            "bundle": "bundle_intraday_core",
                            "sleeve": "trend_pullback",
                            "cluster": "trend",
                            "opened_ts": opened_ts,
                            "max_hold_hours": 1.0,
                            "regime_context": "bull",
                            "reason_tag": "trend_pullback",
                        }
                    },
                    equity=prepare.INITIAL_CAPITAL,
                    last_timestamp=opened_ts,
                ),
            )

            engine = _FakeV2Engine(
                timestamps=[close_ts],
                close_by_timestamp={close_ts: {"BTC": 110.0}},
            )
            config = PortfolioConfig(slippage_bps=10.0)

            result = run_v2_shadow_session(
                bundle_name="bundle_intraday_core",
                model_set="unused",
                split="val",
                portfolio_config=config,
                state_path=str(state_path),
                log_path=str(log_path),
                engine=engine,
            )

            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(result["bars_processed"], 1)
            self.assertEqual(len(rows), 1)
            close_row = rows[0]
            expected_exec_price = 110.0 - (110.0 * 10.0 / 10000.0)
            expected_fee = 100.0 * prepare.TAKER_FEE
            expected_entry_fee = 100.0 * prepare.TAKER_FEE
            expected_gross_realized = 100.0 * (expected_exec_price - 100.0) / 100.0
            expected_realized = expected_gross_realized - expected_entry_fee - expected_fee

            self.assertEqual(close_row["action"], "close")
            self.assertEqual(close_row["decision_reason"], "max_hold")
            self.assertAlmostEqual(close_row["exec_price"], expected_exec_price)
            self.assertAlmostEqual(close_row["fee"], expected_fee)
            self.assertAlmostEqual(close_row["entry_fee_allocated"], expected_entry_fee)
            self.assertAlmostEqual(close_row["gross_realized_pnl"], expected_gross_realized)
            self.assertAlmostEqual(close_row["realized_pnl"], expected_realized)

            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            expected_cash = (prepare.INITIAL_CAPITAL - 100.0) - expected_fee + 100.0 + expected_gross_realized
            self.assertAlmostEqual(saved_state["cash"], expected_cash)
            self.assertEqual(saved_state["positions"], {})

    def test_open_then_max_hold_close_tracks_entry_fee_basis_in_realized_pnl(self):
        open_ts = 1_700_000_000_000
        close_ts = open_ts + 7_200_000
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            log_path = root / "log.jsonl"
            config = PortfolioConfig(slippage_bps=0.0, tier_a_notional_pct=0.001, tier_b_notional_pct=0.001)

            open_signal = SleeveSignal(
                bundle="bundle_intraday_core",
                sleeve="trend_pullback",
                symbol="BTC",
                side=1,
                confidence=0.7,
                expected_edge_bps=100.0,
                holding_horizon_hours=1.0,
                stop_distance=1.0,
                target_notional=100.0,
                regime_context="bull",
                reason_tag="trend_pullback",
                cluster="trend",
                timestamp=open_ts,
                metadata={"base_close": 100.0, "base_volume": 1_000_000.0},
            )

            run_v2_shadow_session(
                bundle_name="bundle_intraday_core",
                model_set="unused",
                split="val",
                portfolio_config=config,
                state_path=str(state_path),
                log_path=str(log_path),
                engine=_FakeV2Engine(
                    timestamps=[open_ts],
                    close_by_timestamp={open_ts: {"BTC": 100.0}},
                    signals_by_timestamp={open_ts: [open_signal]},
                ),
            )
            log_path.unlink()

            run_v2_shadow_session(
                bundle_name="bundle_intraday_core",
                model_set="unused",
                split="val",
                portfolio_config=config,
                state_path=str(state_path),
                log_path=str(log_path),
                engine=_FakeV2Engine(
                    timestamps=[close_ts],
                    close_by_timestamp={close_ts: {"BTC": 110.0}},
                ),
            )

            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            close_row = rows[0]
            expected_fee = 100.0 * prepare.TAKER_FEE
            expected_gross_realized = 10.0
            expected_realized = expected_gross_realized - expected_fee - expected_fee
            self.assertAlmostEqual(close_row["gross_realized_pnl"], expected_gross_realized)
            self.assertAlmostEqual(close_row["entry_fee_allocated"], expected_fee)
            self.assertAlmostEqual(close_row["fee"], expected_fee)
            self.assertAlmostEqual(close_row["realized_pnl"], expected_realized)

    def test_funding_is_applied_to_cash_each_bar(self):
        opened_ts = 1_700_000_000_000
        bar_ts = opened_ts + 3_600_000
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            log_path = root / "log.jsonl"

            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    cash=prepare.INITIAL_CAPITAL - 100.0,
                    positions={"BTC": 100.0},
                    entry_prices={"BTC": 100.0},
                    equity=prepare.INITIAL_CAPITAL,
                    last_timestamp=opened_ts,
                ),
            )

            engine = _FakeV2Engine(
                timestamps=[bar_ts],
                close_by_timestamp={bar_ts: {"BTC": 100.0}},
                funding_by_timestamp={
                    bar_ts: {
                        "BTC": {
                            "funding_rate": 0.008,
                            "has_funding": 1.0,
                            "bar_interval_hours": 1.0,
                        }
                    }
                },
            )

            run_v2_shadow_session(
                bundle_name="bundle_intraday_core",
                model_set="unused",
                split="val",
                portfolio_config=PortfolioConfig(slippage_bps=0.0),
                state_path=str(state_path),
                log_path=str(log_path),
                engine=engine,
                collect_bar_history=True,
            )

            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            expected_funding = 100.0 * 0.008 / 8.0
            expected_cash = (prepare.INITIAL_CAPITAL - 100.0) - expected_funding
            self.assertAlmostEqual(saved_state["cash"], expected_cash)

    def test_max_hold_uses_millisecond_timestamps_correctly(self):
        opened_ts = 1_700_000_000_000
        bar_ts = opened_ts + 3_600_000
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            log_path = root / "log.jsonl"

            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    cash=prepare.INITIAL_CAPITAL - 100.0,
                    positions={"BTC": 100.0},
                    entry_prices={"BTC": 100.0},
                    position_meta={
                        "BTC": {
                            "bundle": "bundle_intraday_core",
                            "sleeve": "trend_pullback",
                            "cluster": "trend",
                            "opened_ts": opened_ts,
                            "max_hold_hours": 2.0,
                            "regime_context": "bull",
                            "reason_tag": "trend_pullback",
                        }
                    },
                    equity=prepare.INITIAL_CAPITAL,
                    last_timestamp=opened_ts,
                ),
            )

            engine = _FakeV2Engine(
                timestamps=[bar_ts],
                close_by_timestamp={bar_ts: {"BTC": 101.0}},
            )

            result = run_v2_shadow_session(
                bundle_name="bundle_intraday_core",
                model_set="unused",
                split="val",
                portfolio_config=PortfolioConfig(slippage_bps=0.0),
                state_path=str(state_path),
                log_path=str(log_path),
                engine=engine,
            )

            rows = []
            if log_path.exists():
                rows = [
                    json.loads(line)
                    for line in log_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            self.assertEqual(result["bars_processed"], 1)
            self.assertEqual(rows, [])
            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("BTC", saved_state["positions"])

    def test_opt_in_stop_loss_closes_position(self):
        opened_ts = 1_700_000_000_000
        bar_ts = opened_ts + 3_600_000
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            log_path = root / "log.jsonl"

            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    cash=prepare.INITIAL_CAPITAL - 100.0,
                    positions={"BTC": 100.0},
                    entry_prices={"BTC": 100.0},
                    position_fee_basis={"BTC": 100.0 * prepare.TAKER_FEE},
                    position_meta={
                        "BTC": {
                            "bundle": "bundle_intraday_core",
                            "sleeve": "trend_pullback",
                            "cluster": "trend",
                            "opened_ts": opened_ts,
                            "max_hold_hours": 24.0,
                            "stop_distance": 2.0,
                            "regime_context": "bull",
                            "reason_tag": "trend_pullback",
                        }
                    },
                    equity=prepare.INITIAL_CAPITAL,
                    last_timestamp=opened_ts,
                ),
            )

            run_v2_shadow_session(
                bundle_name="bundle_intraday_core",
                model_set="unused",
                split="val",
                portfolio_config=PortfolioConfig(slippage_bps=0.0, enable_stop_loss=True),
                state_path=str(state_path),
                log_path=str(log_path),
                engine=_FakeV2Engine(
                    timestamps=[bar_ts],
                    close_by_timestamp={bar_ts: {"BTC": 97.5}},
                ),
            )

            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["action"], "close")
            self.assertEqual(rows[0]["decision_reason"], "stop_loss")

    def test_opt_in_take_profit_closes_position(self):
        opened_ts = 1_700_000_000_000
        bar_ts = opened_ts + 3_600_000
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            log_path = root / "log.jsonl"

            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    cash=prepare.INITIAL_CAPITAL - 100.0,
                    positions={"BTC": 100.0},
                    entry_prices={"BTC": 100.0},
                    position_meta={
                        "BTC": {
                            "bundle": "bundle_intraday_core",
                            "sleeve": "trend_pullback",
                            "cluster": "trend",
                            "opened_ts": opened_ts,
                            "max_hold_hours": 24.0,
                            "stop_distance": 2.0,
                            "regime_context": "bull",
                            "reason_tag": "trend_pullback",
                        }
                    },
                    equity=prepare.INITIAL_CAPITAL,
                    last_timestamp=opened_ts,
                ),
            )

            run_v2_shadow_session(
                bundle_name="bundle_intraday_core",
                model_set="unused",
                split="val",
                portfolio_config=PortfolioConfig(
                    slippage_bps=0.0,
                    enable_stop_loss=True,
                    take_profit_r_multiple=1.5,
                ),
                state_path=str(state_path),
                log_path=str(log_path),
                engine=_FakeV2Engine(
                    timestamps=[bar_ts],
                    close_by_timestamp={bar_ts: {"BTC": 103.2}},
                ),
            )

            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["action"], "close")
            self.assertEqual(rows[0]["decision_reason"], "take_profit")


if __name__ == "__main__":
    unittest.main()
