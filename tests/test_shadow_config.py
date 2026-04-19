import json
import tempfile
import unittest
from pathlib import Path

from execution.config import resolve_shadow_runtime_config


class ShadowConfigTests(unittest.TestCase):
    def test_config_merge_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "shadow_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "timeframe": "4h",
                        "risk": {
                            "max_leverage": 2.2,
                            "max_symbol_notional_pct": 0.2,
                        },
                        "operator": {"recent_actions": 8},
                    }
                ),
                encoding="utf-8",
            )

            resolved = resolve_shadow_runtime_config(
                config_path=config_path,
                overrides={
                    "timeframe": "1h",
                    "max_days": 3,
                    "recent_actions": 5,
                },
            )

            self.assertEqual(resolved["timeframe"], "1h")
            self.assertEqual(resolved["split"], "2026q1")
            self.assertEqual(resolved["max_days"], 3)
            self.assertAlmostEqual(resolved["risk"]["max_leverage"], 2.2)
            self.assertAlmostEqual(resolved["risk"]["max_symbol_notional_pct"], 0.2)
            self.assertEqual(resolved["operator"]["recent_actions"], 5)
            self.assertEqual(resolved["config_source"], str(config_path))


if __name__ == "__main__":
    unittest.main()
