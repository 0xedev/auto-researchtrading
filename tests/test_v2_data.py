import unittest

from v2.data import split_for_timeframe


class V2DataTests(unittest.TestCase):
    def test_split_for_timeframe_handles_15m_suffixes(self):
        self.assertEqual(split_for_timeframe("train", "15m"), "train_15m")
        self.assertEqual(split_for_timeframe("2026q1", "15m"), "2026q1_15m")
        self.assertEqual(split_for_timeframe("train_15m", "1h"), "train")
        self.assertEqual(split_for_timeframe("2026q1_15m", "4h"), "2026q1")


if __name__ == "__main__":
    unittest.main()
