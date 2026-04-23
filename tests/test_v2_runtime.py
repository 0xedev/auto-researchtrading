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


if __name__ == "__main__":
    unittest.main()
