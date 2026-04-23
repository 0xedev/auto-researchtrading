import unittest

from v2.audit import audit_sleeves


class V2AuditTests(unittest.TestCase):
    def test_audit_sleeves_returns_split_metrics(self):
        rows = audit_sleeves(
            bundle_name="bundle_intraday_core",
            sleeves=["trend_pullback"],
            feature_profile="price_only",
            splits=("train",),
            max_symbols=2,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["sleeve"], "trend_pullback")
        self.assertIn("train", row["splits"])
        self.assertIn("rows", row["splits"]["train"])
        self.assertIn("positive_rows", row["splits"]["train"])


if __name__ == "__main__":
    unittest.main()
