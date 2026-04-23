import unittest

import pandas as pd

from v2.legacy import _annotate_bear_1h_calibrated, _annotate_bear_1h_foundation, _annotate_trend_1h_directional


class V2LegacyTests(unittest.TestCase):
    def test_annotate_directional_entries_emits_exp494_style_long_signal(self):
        frame = pd.DataFrame(
            [
                {
                    "timestamp": 1,
                    "symbol": "BTC",
                    "regime_family": "bull",
                    "meta_15m": 0.40,
                    "meta_long_15m": 0.62,
                    "short_conf_15m": 0.18,
                    "meta_1h": 0.55,
                    "meta_4h": 0.60,
                    "bull_15m": 0.58,
                    "bear_15m": 0.12,
                    "rsi_8b": 49.0,
                    "market_ret": 0.002,
                    "market_ret_regime_sum": 0.010,
                    "atr_val": 15.0,
                    "close": 30000.0,
                    "liquidity_sweep": 1.0,
                    "msb_status": 1.0,
                    "fvg_detected": 1.0,
                    "ema_200_dist": 0.03,
                    "dist_to_vwap": -0.01,
                }
            ]
        )
        labeled = _annotate_trend_1h_directional(frame)
        self.assertEqual(len(labeled), 1)
        self.assertEqual(int(labeled.iloc[0]["side"]), 1)
        self.assertEqual(labeled.iloc[0]["reason_tag"], "entry_bull_fortress")
        self.assertGreater(float(labeled.iloc[0]["confidence"]), 0.58)
        self.assertGreater(float(labeled.iloc[0]["expected_edge_bps"]), 0.0)

    def test_annotate_bear_calibrated_emits_exp494_style_short_signal(self):
        frame = pd.DataFrame(
            [
                {
                    "timestamp": 1,
                    "symbol": "BTC",
                    "regime_family": "bear",
                    "meta_15m": 0.31,
                    "meta_long_15m": 0.14,
                    "short_conf_15m": 0.52,
                    "meta_1h": 0.41,
                    "meta_4h": 0.46,
                    "bull_15m": 0.09,
                    "bear_15m": 0.58,
                    "rsi_8b": 47.0,
                    "market_ret": -0.004,
                    "market_ret_regime_sum": -0.020,
                    "atr_val": 18.0,
                    "close": 28000.0,
                    "liquidity_sweep": -1.0,
                    "msb_status": -1.0,
                    "fvg_detected": -1.0,
                    "ema_200_dist": -0.04,
                    "dist_to_vwap": 0.02,
                }
            ]
        )
        labeled = _annotate_bear_1h_calibrated(frame)
        self.assertEqual(len(labeled), 1)
        self.assertEqual(int(labeled.iloc[0]["side"]), -1)
        self.assertEqual(labeled.iloc[0]["reason_tag"], "entry_bear_calibrated")
        self.assertGreater(float(labeled.iloc[0]["confidence"]), 0.56)
        self.assertGreater(float(labeled.iloc[0]["expected_edge_bps"]), 0.0)

    def test_annotate_bear_foundation_prefers_fortress_short_signal(self):
        frame = pd.DataFrame(
            [
                {
                    "timestamp": 1,
                    "symbol": "BTC",
                    "regime_family": "bear",
                    "meta_15m": 0.31,
                    "meta_long_15m": 0.12,
                    "short_conf_15m": 0.33,
                    "meta_1h": 0.44,
                    "meta_4h": 0.49,
                    "bull_15m": 0.08,
                    "bear_15m": 0.62,
                    "rsi_8b": 48.0,
                    "market_ret": -0.005,
                    "market_ret_regime_sum": -0.021,
                    "atr_val": 19.0,
                    "close": 27000.0,
                    "liquidity_sweep": -1.0,
                    "msb_status": -1.0,
                    "fvg_detected": -1.0,
                    "ema_200_dist": -0.04,
                    "dist_to_vwap": 0.03,
                }
            ]
        )
        labeled = _annotate_bear_1h_foundation(frame)
        self.assertEqual(len(labeled), 1)
        self.assertEqual(int(labeled.iloc[0]["side"]), -1)
        self.assertEqual(labeled.iloc[0]["reason_tag"], "entry_bear_fortress")
        self.assertGreater(float(labeled.iloc[0]["confidence"]), 0.54)
        self.assertGreater(float(labeled.iloc[0]["expected_edge_bps"]), 0.0)


if __name__ == "__main__":
    unittest.main()
