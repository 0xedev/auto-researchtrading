import unittest

from v2.manifests import BUNDLE_MANIFESTS, SLEEVE_MANIFESTS


class V2ManifestTests(unittest.TestCase):
    def test_required_bundles_exist(self):
        self.assertIn("bundle_intraday_core", BUNDLE_MANIFESTS)
        self.assertIn("bundle_swing_core", BUNDLE_MANIFESTS)
        self.assertEqual(BUNDLE_MANIFESTS["bundle_intraday_core"].role_timeframes(), {"fast": "15m", "base": "1h", "slow": "4h"})
        self.assertEqual(BUNDLE_MANIFESTS["bundle_swing_core"].role_timeframes(), {"fast": "1h", "base": "4h", "slow": "1d"})

    def test_candidate_sleeve_count_matches_requested_list(self):
        self.assertEqual(len(SLEEVE_MANIFESTS), 18)
        for name in ("trend_1h_directional", "bear_1h_calibrated", "bear_1h_foundation", "trend_breakout", "sideways_mean_reversion", "funding_carry", "macro_event_drift", "bear_stress_short"):
            self.assertIn(name, SLEEVE_MANIFESTS)


if __name__ == "__main__":
    unittest.main()
