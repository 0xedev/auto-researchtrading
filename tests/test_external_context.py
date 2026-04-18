import unittest

import pandas as pd

from external_context import CONTEXT_COLUMNS, build_context_features


class ExternalContextTests(unittest.TestCase):
    def test_build_context_features_has_expected_columns(self):
        timestamps = pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC").astype("int64") // 10**9
        frame = build_context_features(timestamps, symbol="BTC", asset_class=0)
        self.assertEqual(list(frame.columns), CONTEXT_COLUMNS)
        self.assertEqual(len(frame), len(timestamps))
        self.assertTrue(frame.notna().all().all())


if __name__ == "__main__":
    unittest.main()
