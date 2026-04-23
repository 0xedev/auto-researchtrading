import tempfile
import unittest
from pathlib import Path

import prepare

from v2.backtest import (
    _secondary_metrics,
    _series_from_curve,
    append_v2_results_row,
    build_v2_backtest_result,
    next_v2_results_id,
)


class V2BacktestTests(unittest.TestCase):
    def test_build_v2_backtest_result_computes_core_metrics(self):
        run_result = {
            "equity_curve": [prepare.INITIAL_CAPITAL, 101_000.0, 99_000.0, 103_000.0],
            "equity_timestamps": [0, 3_600, 7_200, 10_800],
            "bars_processed": 4,
            "total_bars_available": 4,
        }
        summary = {
            "portfolio": {
                "equity": 103_000.0,
                "total_volume": 25_000.0,
            }
        }
        log_rows = [
            {"action": "open", "target_position": 5_000.0},
            {"action": "close", "realized_pnl": 200.0},
            {"action": "open", "target_position": -3_000.0},
            {"action": "close", "realized_pnl": -100.0},
        ]

        result = build_v2_backtest_result(
            bundle_name="bundle_intraday_core",
            run_result=run_result,
            summary=summary,
            log_rows=log_rows,
        )

        self.assertEqual(result.num_trades, 2)
        self.assertAlmostEqual(result.total_return_pct, 3.0)
        self.assertAlmostEqual(result.win_rate_pct, 50.0)
        self.assertAlmostEqual(result.profit_factor, 2.0)
        self.assertGreater(result.sharpe, 0.0)
        self.assertGreater(result.max_drawdown_pct, 0.0)
        self.assertGreater(result.annual_turnover, 0.0)

        secondary = _secondary_metrics(
            _series_from_curve(result.equity_curve, result.equity_timestamps),
            "bundle_intraday_core",
        )
        self.assertGreaterEqual(secondary["daily_observations"], 0)
        self.assertIn("daily_sharpe", secondary)
        self.assertIn("daily_sortino", secondary)

    def test_next_v2_results_id_and_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "results.tsv"
            target.write_text(
                "commit\tscore\tsharpe\tmax_dd\tstatus\tdescription\n"
                "exp494\t2.245\t2.245\t0.47\tKEEP\tlegacy\n"
                "v2exp1\t0.812\t1.200\t4.20\tCANDIDATE\tv2 run\n",
                encoding="utf-8",
            )
            self.assertEqual(next_v2_results_id(target), "v2exp2")

            backtest = {
                "score": 0.8123,
                "status": "CANDIDATE",
                "result": prepare.BacktestResult(
                    sharpe=1.2,
                    max_drawdown_pct=4.2,
                ),
            }
            run_id = append_v2_results_row(
                backtest=backtest,
                description="fresh v2 run",
                path=target,
            )
            self.assertEqual(run_id, "v2exp2")
            contents = target.read_text(encoding="utf-8")
            self.assertIn("v2exp2", contents)
            self.assertIn("fresh v2 run", contents)


if __name__ == "__main__":
    unittest.main()
