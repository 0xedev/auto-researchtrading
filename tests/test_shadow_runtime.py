import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from execution.paper import ShadowRiskConfig, run_shadow_session


class _FakeStrategy:
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.trailing_stops = {}
        self.position_ages = {}
        self._market_ret_buf = []
        self.bar_counts = {}
        self._macro_bear = False

    def pre_calculate_signals(self, data, split_name=None):
        return None

    def on_bar(self, bar_data, portfolio):
        return []


class ShadowRuntimeTests(unittest.TestCase):
    def test_shadow_session_resume_skips_processed_bars(self):
        rows = [
            {"timestamp": 1713481200, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "funding_rate": 0.0},
            {"timestamp": 1713484800, "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 11.0, "funding_rate": 0.0},
            {"timestamp": 1713488400, "open": 101.5, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 12.0, "funding_rate": 0.0},
        ]
        fake_data = {"BTC": pd.DataFrame(rows)}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = ShadowRiskConfig(
                state_path=str(root / "shadow_state.json"),
                log_path=str(root / "shadow_log.jsonl"),
                kill_switch_path=str(root / "kill_switch.json"),
            )

            with patch("execution.paper.load_data", return_value=fake_data), patch("execution.paper.Strategy", _FakeStrategy):
                first = run_shadow_session("1h", "2026q1", config)
                second = run_shadow_session("1h", "2026q1", config)

            self.assertEqual(first["bars_processed"], 3)
            self.assertEqual(second["bars_processed"], 0)
            self.assertEqual(first["last_timestamp"], rows[-1]["timestamp"])
            self.assertEqual(second["last_timestamp"], rows[-1]["timestamp"])


if __name__ == "__main__":
    unittest.main()
