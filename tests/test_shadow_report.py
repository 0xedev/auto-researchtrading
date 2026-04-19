import json
import tempfile
import unittest
from pathlib import Path

from execution.paper import ShadowState, save_shadow_state
from execution.report import render_shadow_dashboard, summarize_shadow_run, write_shadow_dashboard


class ShadowReportTests(unittest.TestCase):
    def test_shadow_report_summary_and_dashboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            log_path = root / "shadow_log.jsonl"
            dashboard_path = root / "dashboard.md"
            summary_path = root / "summary.json"

            save_shadow_state(
                state_path,
                ShadowState(
                    cash=9000.0,
                    positions={"BTC": 1200.0, "ETH": -500.0},
                    entry_prices={"BTC": 42000.0, "ETH": 2500.0},
                    equity=10300.0,
                    last_timestamp=1713484800,
                    total_volume=3100.0,
                    runtime_config={
                        "source": "shadow_config.example.json",
                        "timeframe": "1h",
                        "split": "2026q1",
                        "max_days": 5,
                        "risk": {"max_leverage": 3.0, "max_symbol_notional_pct": 0.15},
                        "paths": {"kill_switch_path": "shadow_kill_switch.json"},
                        "operator": {"recent_actions": 5},
                    },
                    last_kill_switch={"halt_new_orders": True},
                ),
            )

            rows = [
                {
                    "timestamp": 1713481200,
                    "action": "open",
                    "symbol": "BTC",
                    "signal_tag": "entry_bull_fortress",
                    "signal_regime_family": "sideways",
                    "decision_reason": "trend_strength_add",
                    "rationale": "entry_bull_fortress in sideways meta=0.61",
                    "exec_price": 42000.0,
                    "fee": 2.0,
                    "realized_pnl": 0.0,
                    "portfolio_equity": 10000.0,
                },
                {
                    "timestamp": 1713484800,
                    "action": "close",
                    "symbol": "ETH",
                    "signal_tag": "close_short_max_hold",
                    "signal_regime_family": "bear",
                    "decision_reason": "close_short_max_hold",
                    "rationale": "close_short_max_hold in bear meta=0.58",
                    "exec_price": 2450.0,
                    "fee": 1.5,
                    "realized_pnl": 35.0,
                    "portfolio_equity": 10300.0,
                },
            ]
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            summary = summarize_shadow_run(state_path=state_path, log_path=log_path, max_recent=5)
            self.assertEqual(summary["portfolio"]["open_positions"], 2)
            self.assertAlmostEqual(summary["log_stats"]["realized_pnl_total"], 35.0)
            self.assertEqual(summary["log_stats"]["action_counts"]["open"], 1)
            self.assertEqual(summary["log_stats"]["action_counts"]["close"], 1)
            self.assertEqual(summary["log_stats"]["signal_counts"]["entry_bull_fortress"], 1)
            self.assertTrue(summary["kill_switch"]["halt_new_orders"])
            self.assertEqual(summary["runtime_config"]["timeframe"], "1h")
            self.assertGreaterEqual(len(summary["alerts"]), 1)

            dashboard = render_shadow_dashboard(summary)
            self.assertIn("# Shadow Trading Dashboard", dashboard)
            self.assertIn("## Control Plane", dashboard)
            self.assertIn("## Alerts", dashboard)
            self.assertIn("## Open Positions", dashboard)
            self.assertIn("entry_bull_fortress", dashboard)
            self.assertIn("close_short_max_hold", dashboard)

            written = write_shadow_dashboard(
                state_path=state_path,
                log_path=log_path,
                dashboard_path=dashboard_path,
                summary_json_path=summary_path,
                max_recent=5,
            )
            self.assertTrue(dashboard_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(written["portfolio"]["open_positions"], 2)


if __name__ == "__main__":
    unittest.main()
