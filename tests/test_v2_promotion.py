import unittest

import prepare

from v2.promotion import PromotionThresholds, build_promotion_report, metrics_from_backtest


def _fake_backtest(**overrides):
    result = prepare.BacktestResult(
        sharpe=overrides.get("bar_sharpe", 4.2),
        total_return_pct=overrides.get("total_return_pct", 25.0),
        max_drawdown_pct=overrides.get("max_drawdown_pct", 4.0),
        num_trades=overrides.get("num_trades", 120),
        win_rate_pct=overrides.get("win_rate_pct", 65.0),
        profit_factor=overrides.get("profit_factor", 4.5),
        annual_turnover=overrides.get("annual_turnover", 100.0),
        duration_days=overrides.get("duration_days", 20.0),
        bars_processed=overrides.get("bars_processed", 480),
        total_bars=overrides.get("total_bars", 480),
    )
    top_share = overrides.get("top_share", 0.25)
    return {
        "split": overrides.get("split", "val"),
        "result": result,
        "score": overrides.get("score", 4.0),
        "bar_sharpe": overrides.get("bar_sharpe", 4.2),
        "daily_sharpe": overrides.get("daily_sharpe", 3.4),
        "daily_sortino": overrides.get("daily_sortino", 3.2),
        "daily_observations": overrides.get("daily_observations", 30),
        "trades_per_day": overrides.get("trades_per_day", 6.0),
        "short_share": overrides.get("short_share", 0.2),
        "summary": {
            "sleeve_concentration": [
                {"sleeve": "basis_dislocation", "pnl_share": top_share},
            ],
        },
        "top_sleeves": [
            {"sleeve": "basis_dislocation", "pnl_share": top_share},
        ],
    }


class V2PromotionGateTests(unittest.TestCase):
    def test_metrics_from_backtest_uses_percent_concentration(self):
        metrics = metrics_from_backtest(_fake_backtest(top_share=0.294))

        self.assertAlmostEqual(metrics["concentration_pct"], 29.4)
        self.assertAlmostEqual(metrics["profit_factor"], 4.5)
        self.assertAlmostEqual(metrics["daily_sortino"], 3.2)

    def test_promotion_report_passes_when_all_gates_clear(self):
        report = build_promotion_report(
            bundle="bundle_intraday_core",
            model_set="v2_test",
            portfolio_config="portfolio.json",
            active_sleeves=["basis_dislocation", "cross_asset_relative_strength"],
            base_runs={
                "val": _fake_backtest(split="val"),
                "2026q1": _fake_backtest(split="2026q1"),
            },
            stress_runs={
                "val": _fake_backtest(split="val", profit_factor=4.1, daily_sharpe=3.1),
                "2026q1": _fake_backtest(split="2026q1", profit_factor=4.0, daily_sharpe=3.0),
            },
            fresh_oos_label="forward-paper-2026q2",
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])

    def test_promotion_report_fails_pf_concentration_and_missing_fresh_oos(self):
        report = build_promotion_report(
            bundle="bundle_intraday_core",
            model_set="v2_bad",
            portfolio_config="portfolio.json",
            active_sleeves=["cross_asset_relative_strength"],
            base_runs={
                "val": _fake_backtest(split="val", profit_factor=1.9, top_share=0.37),
                "2026q1": _fake_backtest(split="2026q1", profit_factor=1.5, top_share=0.41),
            },
            stress_runs={
                "val": _fake_backtest(split="val", profit_factor=1.8, top_share=0.39),
                "2026q1": _fake_backtest(split="2026q1", profit_factor=1.5, top_share=0.35),
            },
            thresholds=PromotionThresholds(),
        )

        self.assertFalse(report["passed"])
        failure_text = "\n".join(report["failures"])
        self.assertIn("profit_factor", failure_text)
        self.assertIn("concentration_pct", failure_text)
        self.assertIn("fresh_oos", failure_text)


if __name__ == "__main__":
    unittest.main()
