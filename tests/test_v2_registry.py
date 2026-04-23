import tempfile
import unittest
from pathlib import Path

from v2.registry import find_live_champion, load_sleeve_registry, save_sleeve_registry, upsert_registry_entry
from v2.types import SleeveRegistry, SleeveRegistryEntry


class V2RegistryTests(unittest.TestCase):
    def test_registry_round_trip_and_upsert(self):
        registry = SleeveRegistry()
        entry = SleeveRegistryEntry(
            sleeve="trend_breakout",
            bundle="bundle_intraday_core",
            model_set="v2_test",
            status="candidate",
            validation_metric=0.61,
        )
        upsert_registry_entry(registry, entry)
        upsert_registry_entry(
            registry,
            SleeveRegistryEntry(
                sleeve="trend_breakout",
                bundle="bundle_intraday_core",
                model_set="v2_test",
                status="paper-live",
                validation_metric=0.64,
            ),
        )
        upsert_registry_entry(
            registry,
            SleeveRegistryEntry(
                sleeve="trend_breakout",
                bundle="bundle_intraday_core",
                model_set="v2_alt",
                status="live-champion",
                validation_metric=0.72,
                oos_metric=0.68,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "registry.json"
            save_sleeve_registry(registry, path)
            loaded = load_sleeve_registry(path)

        self.assertEqual(len(loaded.entries), 2)
        updated = [row for row in loaded.entries if row.model_set == "v2_test"][0]
        self.assertEqual(updated.status, "paper-live")
        self.assertAlmostEqual(updated.validation_metric, 0.64)
        champion = find_live_champion(loaded, bundle="bundle_intraday_core", sleeve="trend_breakout")
        self.assertIsNotNone(champion)
        self.assertEqual(champion.model_set, "v2_alt")


if __name__ == "__main__":
    unittest.main()
