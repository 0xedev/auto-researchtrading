import unittest

import pandas as pd

from v2.runtime import V2SignalEngine


class V2RuntimeTests(unittest.TestCase):
    def test_signals_use_manifest_cluster_and_preserve_market_cluster_metadata(self):
        engine = V2SignalEngine(
            bundle_name="bundle_intraday_core",
            model_set="unused",
            active_sleeves=["post_extension_snapback"],
        )
        engine.sleeve_tables["post_extension_snapback"] = pd.DataFrame(
            [
                {
                    "timestamp": 123,
                    "symbol": "BTC",
                    "confidence": 0.70,
                    "side": 1,
                    "base_close": 100.0,
                    "base_clock_atr_pct_24h": 0.01,
                    "holding_horizon_hours": 8.0,
                    "reason_tag": "post_extension_snapback",
                    "market_cluster": "crypto_majors",
                    "regime_family": "sideways",
                }
            ]
        )

        signals = engine.signals_at_timestamp(123)

        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.cluster, "mean_reversion")
        self.assertEqual(signal.metadata["strategy_cluster"], "mean_reversion")
        self.assertEqual(signal.metadata["market_cluster"], "crypto_majors")

    def test_signals_respect_confidence_overrides(self):
        engine = V2SignalEngine(
            bundle_name="bundle_intraday_core",
            model_set="unused",
            active_sleeves=["post_event_mean_reversion"],
            confidence_overrides={"post_event_mean_reversion": 0.30},
        )
        engine.sleeve_tables["post_event_mean_reversion"] = pd.DataFrame(
            [
                {
                    "timestamp": 456,
                    "symbol": "ETH",
                    "confidence": 0.35,
                    "side": -1,
                    "base_close": 100.0,
                    "base_clock_atr_pct_24h": 0.01,
                    "holding_horizon_hours": 8.0,
                    "reason_tag": "post_event_mean_reversion",
                    "market_cluster": "crypto_majors",
                    "regime_family": "event",
                }
            ]
        )

        signals = engine.signals_at_timestamp(456)

        self.assertEqual(len(signals), 1)
        self.assertAlmostEqual(signals[0].metadata["min_confidence"], 0.30)

    def test_bundle_close_index_returns_symbol_prices_by_timestamp(self):
        engine = V2SignalEngine(
            bundle_name="bundle_intraday_core",
            model_set="unused",
            active_sleeves=[],
        )
        engine.bundle_frame = pd.DataFrame(
            [
                {"timestamp": 100, "symbol": "BTC", "base_close": 101.0},
                {"timestamp": 100, "symbol": "ETH", "base_close": 202.0},
                {"timestamp": 200, "symbol": "BTC", "base_close": 103.0},
            ]
        )

        engine._build_bundle_close_index()

        self.assertEqual(engine.close_by_symbol_at_timestamp(100), {"BTC": 101.0, "ETH": 202.0})
        self.assertEqual(engine.close_by_symbol_at_timestamp(200), {"BTC": 103.0})
        self.assertEqual(engine.close_by_symbol_at_timestamp(300), {})


if __name__ == "__main__":
    unittest.main()
