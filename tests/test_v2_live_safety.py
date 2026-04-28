import tempfile
import unittest
from pathlib import Path

from execution.v2_paper import V2ShadowState, save_v2_shadow_state
from v2_live_binance import live_mode_unlocked, reconcile_binance_positions
from v2_live_deriv import _reconcile_deriv_positions


class _FakeBinanceClient:
    def __init__(self, positions):
        self._positions = positions

    def position_risk(self):
        return self._positions


class V2LiveSafetyTests(unittest.TestCase):
    def test_noninteractive_live_requires_explicit_unlock(self):
        self.assertFalse(live_mode_unlocked(False, env={}))
        self.assertFalse(live_mode_unlocked(False, env={"ALLOW_REAL_MONEY": "yes"}))
        self.assertTrue(live_mode_unlocked(False, env={"ALLOW_REAL_MONEY": "YES_I_UNDERSTAND"}))
        self.assertTrue(live_mode_unlocked(True, env={}))

    def test_binance_reconcile_flags_state_exchange_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            save_v2_shadow_state(
                state_path,
                V2ShadowState(positions={}, entry_prices={}, equity=5_000.0),
            )
            client = _FakeBinanceClient(
                [
                    {
                        "symbol": "LTCUSDT",
                        "positionAmt": "2.0",
                        "entryPrice": "50.0",
                        "markPrice": "55.0",
                        "notional": "110.0",
                    }
                ]
            )
            positions, entry_prices, meta, mismatch = reconcile_binance_positions(
                client,
                {"LTC": "LTCUSDT"},
                str(state_path),
            )

            self.assertTrue(mismatch)
            self.assertEqual(positions["LTC"], 110.0)
            self.assertEqual(entry_prices["LTC"], 50.0)
            self.assertEqual(meta["LTC"]["exchange_symbol"], "LTCUSDT")

    def test_deriv_reconcile_rebuilds_tracked_contracts_and_flags_unknowns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    positions={"DERIV_V75": 100.0},
                    entry_prices={"DERIV_V75": 1234.5},
                    position_meta={
                        "DERIV_V75": {
                            "deriv_contract_id": 42,
                            "deriv_multiplier": 50,
                            "deriv_stake": 2.0,
                            "deriv_direction": "up",
                            "deriv_symbol": "R_75",
                        }
                    },
                ),
            )

            positions, mismatch = _reconcile_deriv_positions(
                str(state_path),
                [{"contract_id": 42}, {"contract_id": 999}],
            )

            self.assertTrue(mismatch)
            self.assertIn("DERIV_V75", positions)
            self.assertEqual(positions["DERIV_V75"].contract_id, 42)
            self.assertEqual(positions["DERIV_V75"].entry_spot, 1234.5)


if __name__ == "__main__":
    unittest.main()
