import json
import tempfile
import unittest
from pathlib import Path

import prepare

from execution.v2_paper import V2ShadowState, run_v2_shadow_session, save_v2_shadow_state
from v2.types import PortfolioConfig


class _FakeV2Engine:
    def __init__(self, timestamps, close_by_timestamp):
        self.timestamps = timestamps
        self._close_by_timestamp = close_by_timestamp
        self.sleeve_tables = {}

    def signals_at_timestamp(self, timestamp):
        return []

    def close_by_symbol_at_timestamp(self, timestamp):
        return self._close_by_timestamp.get(timestamp, {})


class V2PaperTests(unittest.TestCase):
    def test_max_hold_close_applies_slippage_and_fee(self):
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
                            "opened_ts": 0,
                            "max_hold_hours": 1.0,
                            "regime_context": "bull",
                            "reason_tag": "trend_pullback",
                        }
                    },
                    equity=prepare.INITIAL_CAPITAL,
                    last_timestamp=0,
                ),
            )

            engine = _FakeV2Engine(
                timestamps=[7200],
                close_by_timestamp={7200: {"BTC": 110.0}},
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
            expected_realized = 100.0 * (expected_exec_price - 100.0) / 100.0

            self.assertEqual(close_row["action"], "close")
            self.assertEqual(close_row["decision_reason"], "max_hold")
            self.assertAlmostEqual(close_row["exec_price"], expected_exec_price)
            self.assertAlmostEqual(close_row["fee"], expected_fee)
            self.assertAlmostEqual(close_row["realized_pnl"], expected_realized)

            saved_state = json.loads(state_path.read_text(encoding="utf-8"))
            expected_cash = (prepare.INITIAL_CAPITAL - 100.0) - expected_fee + 100.0 + expected_realized
            self.assertAlmostEqual(saved_state["cash"], expected_cash)
            self.assertEqual(saved_state["positions"], {})


if __name__ == "__main__":
    unittest.main()
