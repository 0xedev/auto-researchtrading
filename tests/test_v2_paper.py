import unittest

from execution.v2_paper import _apply_reentry_cooldown
from v2.types import SleeveSignal


class V2PaperTests(unittest.TestCase):
    def test_apply_reentry_cooldown_respects_sleeve_scope(self):
        signal = SleeveSignal(
            "bundle_intraday_core",
            "post_extension_snapback",
            "BTC",
            1,
            0.75,
            20.0,
            12.0,
            0.0,
            0.0,
            "sideways",
            "post_extension_snapback",
            "crypto_majors",
            10_000,
            metadata={"reentry_cooldown_hours": 4.0, "reentry_cooldown_scope": "sleeve"},
        )
        kept, rejected = _apply_reentry_cooldown([signal], {"post_extension_snapback": 1_000}, 10_000)
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "sleeve_reentry_cooldown")

    def test_apply_reentry_cooldown_allows_signal_after_window(self):
        signal = SleeveSignal(
            "bundle_intraday_core",
            "post_extension_snapback",
            "BTC",
            1,
            0.75,
            20.0,
            12.0,
            0.0,
            0.0,
            "sideways",
            "post_extension_snapback",
            "crypto_majors",
            20_000,
            metadata={"reentry_cooldown_hours": 4.0, "reentry_cooldown_scope": "symbol"},
        )
        kept, rejected = _apply_reentry_cooldown([signal], {"post_extension_snapback:BTC": 1_000}, 20_000)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
