from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


class V2ForwardPaperScriptTests(unittest.TestCase):
    def test_forward_launcher_refuses_live_mode(self) -> None:
        result = subprocess.run(
            ["bash", "run_v2_forward_paper.sh", "--live"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing --live", result.stderr)

    def test_start_script_supports_dry_run_without_live_default(self) -> None:
        text = Path("start_v2_traders.sh").read_text(encoding="utf-8")

        self.assertIn('LIVE_FLAG="--testnet"', text)
        self.assertIn('--dry-run)', text)
        self.assertIn("$DRY_RUN_FLAG", text)

    def test_dashboard_launcher_refuses_live_mode(self) -> None:
        result = subprocess.run(
            ["bash", "run_v2_dashboards.sh", "--live"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing --live", result.stderr)

    def test_dashboard_launcher_uses_tmux_and_safe_endpoints(self) -> None:
        text = Path("run_v2_dashboards.sh").read_text(encoding="utf-8")

        self.assertIn("tmux new-session", text)
        self.assertIn("--testnet", text)
        self.assertIn("DERIV DEMO DASHBOARD", text)


if __name__ == "__main__":
    unittest.main()
