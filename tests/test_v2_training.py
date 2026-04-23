import unittest

import pandas as pd

from v2.training import choose_temporal_split


class V2TrainingTests(unittest.TestCase):
    def test_choose_temporal_split_finds_valid_cutoff(self):
        frame = pd.DataFrame(
            {
                "timestamp": list(range(20)),
                "target": [1] * 12 + [0] * 4 + [1] * 4,
            }
        )
        result = choose_temporal_split(frame, min_train_pos=6, min_val_pos=2)
        self.assertIsNotNone(result)
        train_mask, val_mask, meta = result
        self.assertGreater(meta["train_pos"], 5)
        self.assertGreater(meta["val_pos"], 1)
        self.assertGreater(train_mask.sum(), 0)
        self.assertGreater(val_mask.sum(), 0)


if __name__ == "__main__":
    unittest.main()
