from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import unittest

from v2_forward_report import build_report, render_markdown


class V2ForwardReportTests(unittest.TestCase):
    def _write_state(self, root: Path, name: str, ts: datetime, equity: float = 10_000.0) -> Path:
        path = root / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "cash": equity,
                    "equity": equity,
                    "last_timestamp": int(ts.timestamp() * 1000),
                    "positions": {},
                    "position_meta": {},
                }
            ),
            encoding="utf-8",
        )
        return path

    def _write_log(self, root: Path, name: str, lines: list[str]) -> Path:
        path = root / f"{name}.log"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_report_passes_when_each_connector_has_clean_forward_span(self) -> None:
        now = datetime(2026, 4, 28, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            connectors = []
            for name in ("binance", "deriv"):
                state = self._write_state(root, name, now)
                log = self._write_log(
                    root,
                    name,
                    [
                        "2026-03-28 00:00:00 INFO V2 signals rows=10",
                        "2026-04-28 00:00:00 INFO V2 signals rows=12",
                    ],
                )
                connectors.append({"name": name, "state_path": str(state), "log_path": str(log)})

            report = build_report(connectors=connectors, min_days=30.0, now=now)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["critical_count"], 0)
        self.assertGreaterEqual(report["observed_days"], 30.0)

    def test_report_fails_on_short_connector_span_and_auth_errors(self) -> None:
        now = datetime(2026, 4, 28, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binance_state = self._write_state(root, "binance", now)
            binance_log = self._write_log(
                root,
                "binance",
                [
                    "2026-04-27 00:00:00 INFO V2 signals rows=10",
                    "2026-04-28 00:00:00 ERROR Invalid API-key, IP, or permissions for action. 401",
                ],
            )
            deriv_state = self._write_state(root, "deriv", now)
            deriv_log = self._write_log(
                root,
                "deriv",
                [
                    "2026-03-28 00:00:00 INFO V2 signals rows=10",
                    "2026-04-28 00:00:00 INFO V2 signals rows=12",
                ],
            )

            report = build_report(
                connectors=[
                    {"name": "binance", "state_path": str(binance_state), "log_path": str(binance_log)},
                    {"name": "deriv", "state_path": str(deriv_state), "log_path": str(deriv_log)},
                ],
                min_days=30.0,
                now=now,
            )

        codes = {alert["code"] for alert in report["alerts"]}
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("insufficient_connector_days", codes)
        self.assertIn("log_errors", codes)
        self.assertIn("auth_failures", codes)

    def test_markdown_renders_alerts_and_connector_sections(self) -> None:
        now = datetime(2026, 4, 28, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self._write_state(root, "binance", now)
            log = self._write_log(root, "binance", ["2026-04-28 00:00:00 INFO V2 signals rows=10"])
            report = build_report(
                connectors=[{"name": "binance", "state_path": str(state), "log_path": str(log)}],
                min_days=30.0,
                now=now,
            )

        md = render_markdown(report)
        self.assertIn("# V2 Forward Paper Readiness Report", md)
        self.assertIn("### binance", md)
        self.assertIn("insufficient_connector_days", md)

    def test_binance_equity_summary_counts_as_signal_bar(self) -> None:
        now = datetime(2026, 4, 28, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self._write_state(root, "binance", now)
            log = self._write_log(
                root,
                "binance",
                ["2026-04-28 11:33:51  INFO       Equity=$4978.51  ts=1777370400000  opens=0  exits=0"],
            )
            report = build_report(
                connectors=[{"name": "binance", "state_path": str(state), "log_path": str(log)}],
                min_days=30.0,
                now=now,
            )

        self.assertEqual(report["connectors"][0]["counters"]["signal_bars"], 1)
        self.assertNotIn("no_signal_bars", {alert["code"] for alert in report["alerts"]})


if __name__ == "__main__":
    unittest.main()
