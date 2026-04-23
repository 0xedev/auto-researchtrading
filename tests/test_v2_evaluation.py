import unittest
from unittest.mock import patch

from v2.evaluation import _summarize_trial, _window_bounds_from_timestamps, evaluate_candidate, evaluate_model_set
from v2.types import PortfolioConfig


class V2EvaluationTests(unittest.TestCase):
    def test_summarize_trial_extracts_density_and_concentration(self):
        result = {"bars_processed": 48}
        summary = {
            "portfolio": {"equity": 102000.0, "gross_leverage": 0.5},
            "log_stats": {
                "records": 3,
                "realized_pnl_total": 250.0,
                "fees_total": 10.0,
                "action_counts": {"open": 2, "close": 1},
                "signal_counts": {"trend_pullback": 2},
                "sleeve_counts": {"trend_pullback": 2},
                "bundle_counts": {"bundle_intraday_core": 2},
                "cluster_counts": {"trend": 2},
                "decision_reason_counts": {"trend_pullback": 2},
            },
            "alerts": [],
            "sleeve_pnl": [{"sleeve": "trend_pullback", "actions": 2, "realized_pnl": 250.0, "fees": 10.0}],
            "sleeve_concentration": [{"sleeve": "trend_pullback", "pnl_share": 0.25}],
        }
        log_rows = [
            {"action": "open", "target_position": 5000.0},
            {"action": "open", "target_position": -3000.0},
            {"action": "close", "target_position": 0.0, "realized_pnl": 250.0},
        ]
        metrics = _summarize_trial("bundle_intraday_core", result, summary, log_rows)
        self.assertAlmostEqual(metrics["days_processed"], 2.0)
        self.assertAlmostEqual(metrics["trades_per_day"], 1.0)
        self.assertAlmostEqual(metrics["short_share"], 0.5)
        self.assertAlmostEqual(metrics["top_sleeve_share"], 0.25)
        self.assertAlmostEqual(metrics["raw_return_pct"], 0.02)
        self.assertAlmostEqual(metrics["promotion_score"], 0.02)
        self.assertEqual(metrics["score_kind"], "promotion_score")
        self.assertAlmostEqual(metrics["win_rate"], 1.0)
        self.assertIsNone(metrics["profit_factor"])
        self.assertTrue(metrics["shadow_ready"])

    def test_evaluate_candidate_uses_raw_returns_for_single_sleeve_gate(self):
        evaluation = {
            "bundle": "bundle_intraday_core",
            "model_set": "v2_test",
            "generated_at": "2026-04-22T00:00:00+00:00",
            "base": {
                "val": {
                    "bar_sharpe": 4.2,
                    "daily_sharpe": 3.4,
                    "win_rate_pct": 61.0,
                    "profit_factor": 4.6,
                    "max_drawdown_pct": 6.0,
                    "raw_return_pct": 0.012,
                    "return_pct": 0.012,
                    "promotion_score": -0.688,
                    "top_sleeve_share": 1.0,
                    "shadow_ready": True,
                    "trades_per_day": 1.4,
                    "short_share": 0.0,
                    "top_sleeve": "trend_1h_directional",
                },
                "2026q1": {
                    "bar_sharpe": 3.9,
                    "daily_sharpe": 3.1,
                    "win_rate_pct": 60.5,
                    "profit_factor": 4.2,
                    "max_drawdown_pct": 5.0,
                    "raw_return_pct": 0.006,
                    "return_pct": 0.006,
                    "promotion_score": -0.694,
                    "top_sleeve_share": 1.0,
                    "shadow_ready": True,
                    "trades_per_day": 1.1,
                    "short_share": 0.0,
                    "top_sleeve": "trend_1h_directional",
                },
            },
            "stress": {
                "val": {
                    "bar_sharpe": 3.8,
                    "daily_sharpe": 3.0,
                    "win_rate_pct": 60.0,
                    "profit_factor": 4.0,
                    "max_drawdown_pct": 6.5,
                    "shadow_ready": True,
                    "trades_per_day": 1.2,
                    "raw_return_pct": 0.003,
                    "promotion_score": -0.697,
                },
                "2026q1": {
                    "bar_sharpe": 3.6,
                    "daily_sharpe": 2.9,
                    "win_rate_pct": 60.0,
                    "profit_factor": 4.0,
                    "max_drawdown_pct": 6.0,
                    "shadow_ready": True,
                    "trades_per_day": 1.0,
                    "raw_return_pct": 0.001,
                    "promotion_score": -0.699,
                },
            },
        }
        with patch("v2.evaluation.evaluate_model_set", return_value=evaluation):
            summary = evaluate_candidate(
                bundle_name="bundle_intraday_core",
                sleeve_name="trend_1h_directional",
                model_set="v2_test",
                portfolio_config=PortfolioConfig(),
            )["summary"]

        self.assertTrue(summary["meets_gate"])
        self.assertEqual(summary["metric_kind"], "bar_sharpe")
        self.assertAlmostEqual(summary["validation_metric"], 4.2)
        self.assertAlmostEqual(summary["oos_metric"], 3.9)
        self.assertAlmostEqual(summary["stress_metric"], 3.6)
        self.assertAlmostEqual(summary["validation_return_pct"], 0.012)
        self.assertAlmostEqual(summary["validation_promotion_score"], -0.688)
        self.assertFalse(summary["concentration_ready"])
        self.assertTrue(summary["validation_audit"]["passed"])

    def test_evaluate_model_set_adds_portfolio_summary(self):
        trial_rows = [
            {
                "bar_sharpe": 4.4,
                "daily_sharpe": 3.6,
                "win_rate_pct": 62.0,
                "profit_factor": 4.8,
                "max_drawdown_pct": 4.2,
                "raw_return_pct": 0.012,
                "return_pct": 0.012,
                "promotion_score": 0.008,
                "top_sleeve_share": 0.28,
                "shadow_ready": True,
                "trades_per_day": 2.4,
                "short_share": 0.16,
                "top_sleeve": "post_extension_snapback",
                "sleeve_concentration": [{"sleeve": "post_extension_snapback", "pnl_share": 0.28}],
            },
            {
                "bar_sharpe": 4.1,
                "daily_sharpe": 3.3,
                "win_rate_pct": 61.0,
                "profit_factor": 4.3,
                "max_drawdown_pct": 4.5,
                "raw_return_pct": 0.009,
                "return_pct": 0.009,
                "promotion_score": 0.005,
                "top_sleeve_share": 0.27,
                "shadow_ready": True,
                "trades_per_day": 2.0,
                "short_share": 0.14,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.27}],
            },
            {
                "bar_sharpe": 3.9,
                "daily_sharpe": 3.1,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "shadow_ready": True,
                "raw_return_pct": 0.004,
                "return_pct": 0.004,
                "promotion_score": 0.001,
                "top_sleeve_share": 0.26,
                "shadow_ready": True,
                "trades_per_day": 1.9,
                "short_share": 0.12,
                "top_sleeve": "post_extension_snapback",
                "sleeve_concentration": [{"sleeve": "post_extension_snapback", "pnl_share": 0.26}],
            },
            {
                "bar_sharpe": 3.7,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "shadow_ready": True,
                "raw_return_pct": 0.002,
                "return_pct": 0.002,
                "promotion_score": -0.001,
                "top_sleeve_share": 0.25,
                "shadow_ready": True,
                "trades_per_day": 1.8,
                "short_share": 0.13,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.25}],
            },
        ]
        with patch("v2.evaluation.run_shadow_trial", side_effect=trial_rows), patch(
            "v2.evaluation.V2SignalEngine"
        ) as engine_cls:
            engine_cls.return_value.prepare.return_value = None
            evaluation = evaluate_model_set(
                bundle_name="bundle_intraday_core",
                model_set="v2_test",
                portfolio_config=PortfolioConfig(),
            )

        summary = evaluation["summary"]
        self.assertTrue(summary["meets_gate"])
        self.assertEqual(summary["metric_kind"], "bar_sharpe")
        self.assertAlmostEqual(summary["validation_metric"], 4.4)
        self.assertAlmostEqual(summary["oos_metric"], 4.1)
        self.assertAlmostEqual(summary["stress_metric"], 3.7)
        self.assertAlmostEqual(summary["validation_return_pct"], 0.012)
        self.assertTrue(summary["concentration_ready"])
        self.assertEqual(summary["top_sleeves"][0]["sleeve"], "post_extension_snapback")
        self.assertTrue(summary["validation_audit"]["passed"])

    def test_window_bounds_from_timestamps_downsamples_evenly(self):
        timestamps = [i * 24 * 3600 for i in range(10)]
        windows = _window_bounds_from_timestamps(
            timestamps,
            window_days=2,
            step_days=1,
            max_windows=3,
        )
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], (0, 2 * 24 * 3600))
        self.assertEqual(windows[-1], (8 * 24 * 3600, 9 * 24 * 3600))

    def test_evaluate_model_set_adds_rolling_summary(self):
        base_trials = [
            {
                "bar_sharpe": 4.1,
                "daily_sharpe": 3.3,
                "win_rate_pct": 61.0,
                "profit_factor": 4.2,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.010,
                "return_pct": 0.010,
                "promotion_score": 0.004,
                "top_sleeve_share": 0.41,
                "shadow_ready": True,
                "trades_per_day": 3.0,
                "short_share": 0.10,
                "top_sleeve": "sideways_mean_reversion",
                "sleeve_concentration": [{"sleeve": "sideways_mean_reversion", "pnl_share": 0.41}],
            },
            {
                "bar_sharpe": 3.9,
                "daily_sharpe": 3.1,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.005,
                "return_pct": 0.005,
                "promotion_score": 0.001,
                "top_sleeve_share": 0.33,
                "shadow_ready": True,
                "trades_per_day": 2.5,
                "short_share": 0.12,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.33}],
            },
            {
                "bar_sharpe": 4.0,
                "daily_sharpe": 3.2,
                "win_rate_pct": 60.5,
                "profit_factor": 4.1,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.009,
                "return_pct": 0.009,
                "promotion_score": 0.003,
                "top_sleeve_share": 0.29,
                "shadow_ready": True,
                "trades_per_day": 2.0,
                "short_share": 0.15,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.29}],
            },
            {
                "bar_sharpe": 3.8,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.004,
                "return_pct": 0.004,
                "promotion_score": 0.000,
                "top_sleeve_share": 0.28,
                "shadow_ready": True,
                "trades_per_day": 1.8,
                "short_share": 0.14,
                "top_sleeve": "post_extension_snapback",
                "sleeve_concentration": [{"sleeve": "post_extension_snapback", "pnl_share": 0.28}],
            },
        ]
        rolling_trials = [
            {
                "bar_sharpe": 3.8,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.003,
                "return_pct": 0.003,
                "promotion_score": 0.001,
                "top_sleeve_share": 0.38,
                "shadow_ready": True,
                "trades_per_day": 2.1,
                "short_share": 0.10,
                "top_sleeve": "sideways_mean_reversion",
                "sleeve_concentration": [{"sleeve": "sideways_mean_reversion", "pnl_share": 0.38}],
            },
            {
                "bar_sharpe": 3.7,
                "daily_sharpe": 2.9,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": -0.001,
                "return_pct": -0.001,
                "promotion_score": -0.003,
                "top_sleeve_share": 0.45,
                "shadow_ready": True,
                "trades_per_day": 1.9,
                "short_share": 0.12,
                "top_sleeve": "sideways_mean_reversion",
                "sleeve_concentration": [{"sleeve": "sideways_mean_reversion", "pnl_share": 0.45}],
            },
            {
                "bar_sharpe": 3.8,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.002,
                "return_pct": 0.002,
                "promotion_score": -0.001,
                "top_sleeve_share": 0.31,
                "shadow_ready": True,
                "trades_per_day": 2.0,
                "short_share": 0.11,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.31}],
            },
            {
                "bar_sharpe": 3.8,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.001,
                "return_pct": 0.001,
                "promotion_score": -0.002,
                "top_sleeve_share": 0.30,
                "shadow_ready": True,
                "trades_per_day": 1.7,
                "short_share": 0.10,
                "top_sleeve": "post_extension_snapback",
                "sleeve_concentration": [{"sleeve": "post_extension_snapback", "pnl_share": 0.30}],
            },
            {
                "bar_sharpe": 3.9,
                "daily_sharpe": 3.1,
                "win_rate_pct": 60.0,
                "profit_factor": 4.1,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.004,
                "return_pct": 0.004,
                "promotion_score": 0.001,
                "top_sleeve_share": 0.35,
                "shadow_ready": True,
                "trades_per_day": 2.3,
                "short_share": 0.13,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.35}],
            },
            {
                "bar_sharpe": 3.8,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.002,
                "return_pct": 0.002,
                "promotion_score": -0.001,
                "top_sleeve_share": 0.29,
                "shadow_ready": True,
                "trades_per_day": 1.8,
                "short_share": 0.14,
                "top_sleeve": "post_extension_snapback",
                "sleeve_concentration": [{"sleeve": "post_extension_snapback", "pnl_share": 0.29}],
            },
            {
                "bar_sharpe": 3.9,
                "daily_sharpe": 3.1,
                "win_rate_pct": 60.0,
                "profit_factor": 4.1,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.005,
                "return_pct": 0.005,
                "promotion_score": 0.001,
                "top_sleeve_share": 0.34,
                "shadow_ready": True,
                "trades_per_day": 2.1,
                "short_share": 0.15,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.34}],
            },
            {
                "bar_sharpe": 3.8,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 5.0,
                "raw_return_pct": 0.003,
                "return_pct": 0.003,
                "promotion_score": 0.000,
                "top_sleeve_share": 0.27,
                "shadow_ready": True,
                "trades_per_day": 1.9,
                "short_share": 0.12,
                "top_sleeve": "cross_asset_relative_strength",
                "sleeve_concentration": [{"sleeve": "cross_asset_relative_strength", "pnl_share": 0.27}],
            },
        ]
        with patch("v2.evaluation.run_shadow_trial", side_effect=base_trials + rolling_trials), patch(
            "v2.evaluation._rolling_window_bounds",
            return_value=[(1, 2), (3, 4)],
        ), patch("v2.evaluation.V2SignalEngine") as engine_cls:
            engine_cls.return_value.prepare.return_value = None
            evaluation = evaluate_model_set(
                bundle_name="bundle_intraday_core",
                model_set="v2_test",
                portfolio_config=PortfolioConfig(),
                rolling_window_days=5,
                rolling_step_days=3,
            )

        rolling_summary = evaluation["rolling"]["base"]["val"]["summary"]
        self.assertEqual(rolling_summary["window_count"], 2)
        self.assertAlmostEqual(rolling_summary["positive_window_rate"], 0.5)
        self.assertEqual(rolling_summary["top_sleeves"][0]["sleeve"], "sideways_mean_reversion")
        self.assertAlmostEqual(evaluation["summary"]["rolling_positive_window_rate"], 0.5)

    def test_evaluate_model_set_can_skip_stress(self):
        trial_rows = [
            {
                "bar_sharpe": 4.0,
                "daily_sharpe": 3.2,
                "win_rate_pct": 61.0,
                "profit_factor": 4.2,
                "max_drawdown_pct": 4.5,
                "raw_return_pct": 0.011,
                "return_pct": 0.011,
                "promotion_score": 0.006,
                "top_sleeve_share": 0.29,
                "shadow_ready": True,
                "trades_per_day": 2.2,
                "short_share": 0.10,
                "top_sleeve": "trend_1h_directional",
                "sleeve_concentration": [{"sleeve": "trend_1h_directional", "pnl_share": 0.29}],
            },
            {
                "bar_sharpe": 3.8,
                "daily_sharpe": 3.0,
                "win_rate_pct": 60.0,
                "profit_factor": 4.0,
                "max_drawdown_pct": 4.8,
                "raw_return_pct": 0.007,
                "return_pct": 0.007,
                "promotion_score": 0.002,
                "top_sleeve_share": 0.27,
                "shadow_ready": True,
                "trades_per_day": 1.9,
                "short_share": 0.12,
                "top_sleeve": "post_extension_snapback",
                "sleeve_concentration": [{"sleeve": "post_extension_snapback", "pnl_share": 0.27}],
            },
        ]
        with patch("v2.evaluation.run_shadow_trial", side_effect=trial_rows), patch(
            "v2.evaluation.V2SignalEngine"
        ) as engine_cls:
            engine_cls.return_value.prepare.return_value = None
            evaluation = evaluate_model_set(
                bundle_name="bundle_intraday_core",
                model_set="v2_test",
                portfolio_config=PortfolioConfig(),
                include_stress=False,
            )

        self.assertEqual(evaluation["stress"], {})
        self.assertFalse(evaluation["summary"]["has_stress"])
        self.assertTrue(evaluation["summary"]["meets_gate"])


if __name__ == "__main__":
    unittest.main()
