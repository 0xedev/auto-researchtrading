import tempfile
import unittest
from pathlib import Path

from execution.paper import ShadowState, load_shadow_state, save_shadow_state


class ShadowStateTests(unittest.TestCase):
    def test_shadow_state_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            original = ShadowState(
                cash=123.0,
                positions={"BTC": 50.0},
                entry_prices={"BTC": 40000.0},
                equity=125.0,
                last_timestamp=1234567890,
                total_volume=75.0,
                strategy_state={"position_ages": {"BTC": 2}},
                runtime_config={"split": "2026q1"},
                last_kill_switch={"halt_new_orders": True},
            )
            save_shadow_state(path, original)
            loaded = load_shadow_state(path)
            self.assertEqual(loaded.cash, original.cash)
            self.assertEqual(loaded.positions, original.positions)
            self.assertEqual(loaded.entry_prices, original.entry_prices)
            self.assertEqual(loaded.strategy_state, original.strategy_state)
            self.assertEqual(loaded.runtime_config, original.runtime_config)
            self.assertEqual(loaded.last_kill_switch, original.last_kill_switch)


if __name__ == "__main__":
    unittest.main()
