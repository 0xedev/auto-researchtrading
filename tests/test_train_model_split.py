import unittest
from unittest.mock import patch

import pandas as pd

import train_model


class TrainModelSplitTests(unittest.TestCase):
    def test_15m_training_uses_train_15m_split(self):
        captured = {}

        def fake_prepare_dataset(timeframe, split_name, feature_profile="price_only"):
            captured["timeframe"] = timeframe
            captured["split_name"] = split_name
            return None, None, None, None, None, None

        with patch("train_model.prepare_dataset", side_effect=fake_prepare_dataset):
            train_model.train("15m")

        self.assertEqual(captured["timeframe"], "15m")
        self.assertEqual(captured["split_name"], "train_15m")


if __name__ == "__main__":
    unittest.main()
