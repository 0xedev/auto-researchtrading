import unittest

import pandas as pd

from v2.alpha import build_training_frame
from v2.types import SleeveManifest, TimeframeBundle


class V2AlphaTests(unittest.TestCase):
    def test_build_training_frame_groups_forward_returns_by_symbol(self):
        bundle = TimeframeBundle(name="test", fast_tf="15m", base_tf="1h", slow_tf="4h", label_horizon_hours=1.0)
        manifest = SleeveManifest(
            name="cross_asset_relative_strength",
            family="relative_strength",
            side_mode="long",
            label_horizon_hours=1.0,
            target_sigma=0.1,
        )
        frame = pd.DataFrame(
            [
                {"timestamp": 1, "symbol": "A", "base_close": 100.0, "base_bar_interval_hours": 1.0, "regime_family": "bull", "base_clock_ema_200h_dist": 0.1, "slow_clock_ema_200h_dist": 0.1, "base_ret_rank_24h": 0.9, "base_clock_ret_72h": 0.1},
                {"timestamp": 2, "symbol": "A", "base_close": 110.0, "base_bar_interval_hours": 1.0, "regime_family": "bull", "base_clock_ema_200h_dist": 0.1, "slow_clock_ema_200h_dist": 0.1, "base_ret_rank_24h": 0.9, "base_clock_ret_72h": 0.1},
                {"timestamp": 1, "symbol": "B", "base_close": 200.0, "base_bar_interval_hours": 1.0, "regime_family": "bull", "base_clock_ema_200h_dist": 0.1, "slow_clock_ema_200h_dist": 0.1, "base_ret_rank_24h": 0.9, "base_clock_ret_72h": 0.1},
                {"timestamp": 2, "symbol": "B", "base_close": 190.0, "base_bar_interval_hours": 1.0, "regime_family": "bull", "base_clock_ema_200h_dist": 0.1, "slow_clock_ema_200h_dist": 0.1, "base_ret_rank_24h": 0.9, "base_clock_ret_72h": 0.1},
            ]
        )
        labeled = build_training_frame(frame, bundle, manifest)
        returns = dict(zip(zip(labeled["symbol"], labeled["timestamp"]), labeled["forward_ret"]))
        self.assertAlmostEqual(returns[("A", 1)], 0.10)
        self.assertAlmostEqual(returns[("B", 1)], -0.05)


if __name__ == "__main__":
    unittest.main()
