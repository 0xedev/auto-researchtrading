import unittest

from v2.allocator import PortfolioAllocator
from v2.types import PortfolioConfig, SleeveSignal


class V2AllocatorTests(unittest.TestCase):
    def test_allocator_applies_symbol_sleeve_and_cluster_caps(self):
        config = PortfolioConfig(
            max_gross_leverage=1.0,
            max_net_exposure_pct=1.0,
            max_per_symbol_pct=0.10,
            max_per_sleeve_pct=0.15,
            max_per_cluster_pct=0.12,
            tier_a_confidence=0.70,
            tier_b_confidence=0.60,
            tier_a_notional_pct=0.10,
            tier_b_notional_pct=0.06,
        )
        allocator = PortfolioAllocator(config)
        signals = [
            SleeveSignal("bundle", "trend_breakout", "BTC", 1, 0.72, 40.0, 12.0, 100.0, 0.0, "bull", "trend_breakout", "crypto_majors", 1),
            SleeveSignal("bundle", "trend_breakout", "ETH", 1, 0.71, 38.0, 12.0, 100.0, 0.0, "bull", "trend_breakout", "crypto_majors", 1),
            SleeveSignal("bundle", "bear_stress_short", "SOL", -1, 0.61, 30.0, 12.0, 100.0, 0.0, "bear", "bear_stress_short", "crypto_beta", 1),
        ]
        allocated, rejected = allocator.allocate(signals, current_positions={}, equity=100_000.0)

        self.assertEqual(len(allocated), 2)
        self.assertTrue(any(signal.symbol == "BTC" for signal in allocated))
        self.assertTrue(any(signal.symbol == "SOL" for signal in allocated))
        self.assertTrue(any(row["reason"] in {"cluster_cap", "sleeve_cap"} for row in rejected))

    def test_allocator_uses_confidence_tiers(self):
        config = PortfolioConfig(tier_a_confidence=0.7, tier_b_confidence=0.6, tier_a_notional_pct=0.1, tier_b_notional_pct=0.05)
        allocator = PortfolioAllocator(config)
        signals = [
            SleeveSignal("bundle", "trend_breakout", "BTC", 1, 0.75, 30.0, 12.0, 0.0, 0.0, "bull", "trend_breakout", "crypto_majors", 1),
            SleeveSignal("bundle", "trend_pullback", "ETH", 1, 0.62, 20.0, 12.0, 0.0, 0.0, "bull", "trend_pullback", "crypto_majors", 1),
            SleeveSignal("bundle", "trend_continuation", "SOL", 1, 0.55, 15.0, 12.0, 0.0, 0.0, "bull", "trend_continuation", "crypto_beta", 1),
        ]
        allocated, rejected = allocator.allocate(signals, current_positions={}, equity=100_000.0)

        by_symbol = {signal.symbol: signal for signal in allocated}
        self.assertAlmostEqual(abs(by_symbol["BTC"].target_notional), 10_000.0)
        self.assertAlmostEqual(abs(by_symbol["ETH"].target_notional), 5_000.0)
        self.assertTrue(any(row["reason"] == "tier_c" for row in rejected))

    def test_allocator_limits_new_positions_per_sleeve(self):
        config = PortfolioConfig(
            max_per_sleeve_pct=1.0,
            max_per_cluster_pct=1.0,
            max_per_symbol_pct=0.2,
            max_new_positions_per_sleeve=1,
            max_new_positions_per_cluster=3,
            tier_a_confidence=0.7,
            tier_b_confidence=0.6,
            tier_a_notional_pct=0.1,
            tier_b_notional_pct=0.05,
        )
        allocator = PortfolioAllocator(config)
        signals = [
            SleeveSignal("bundle", "post_extension_snapback", "BTC", -1, 0.80, 30.0, 12.0, 0.0, 0.0, "sideways", "post_extension_snapback", "crypto_majors", 1),
            SleeveSignal("bundle", "post_extension_snapback", "ETH", -1, 0.79, 29.0, 12.0, 0.0, 0.0, "sideways", "post_extension_snapback", "crypto_majors", 1),
            SleeveSignal("bundle", "cross_asset_relative_strength", "SOL", 1, 0.78, 28.0, 12.0, 0.0, 0.0, "bull", "cross_asset_relative_strength", "crypto_beta", 1),
        ]
        allocated, rejected = allocator.allocate(signals, current_positions={}, equity=100_000.0)

        self.assertEqual(len([row for row in allocated if row.sleeve == "post_extension_snapback"]), 1)
        self.assertTrue(any(row["reason"] == "sleeve_new_position_cap" for row in rejected))

    def test_allocator_penalizes_existing_sleeve_and_cluster_exposure(self):
        config = PortfolioConfig(
            max_per_sleeve_pct=1.0,
            max_per_cluster_pct=1.0,
            max_per_symbol_pct=0.2,
            diversification_penalty=0.50,
            tier_a_confidence=0.7,
            tier_b_confidence=0.6,
            tier_a_notional_pct=0.1,
            tier_b_notional_pct=0.05,
        )
        allocator = PortfolioAllocator(config)
        signals = [
            SleeveSignal("bundle", "trend_breakout", "ETH", 1, 0.75, 30.0, 12.0, 0.0, 0.0, "bull", "trend_breakout", "crypto_majors", 1),
            SleeveSignal("bundle", "cross_asset_relative_strength", "SOL", 1, 0.75, 30.0, 12.0, 0.0, 0.0, "bull", "cross_asset_relative_strength", "crypto_beta", 1),
        ]
        allocated, _ = allocator.allocate(
            signals,
            current_positions={"BTC": 10_000.0},
            equity=100_000.0,
            current_position_meta={"BTC": {"sleeve": "trend_breakout", "cluster": "crypto_majors"}},
        )

        self.assertEqual(len(allocated), 2)
        self.assertEqual(allocated[0].sleeve, "cross_asset_relative_strength")
        self.assertGreater(
            allocated[0].metadata["allocator_effective_score"],
            allocated[1].metadata["allocator_effective_score"],
        )

    def test_allocator_applies_portfolio_weight_to_score_and_notional(self):
        config = PortfolioConfig(
            max_per_sleeve_pct=1.0,
            max_per_cluster_pct=1.0,
            max_per_symbol_pct=1.0,
            tier_a_confidence=0.7,
            tier_b_confidence=0.6,
            tier_a_notional_pct=0.1,
            tier_b_notional_pct=0.05,
        )
        allocator = PortfolioAllocator(config)
        low_weight = SleeveSignal("bundle", "sideways_mean_reversion", "ETH", 1, 0.75, 30.0, 12.0, 0.0, 0.0, "sideways", "sideways_mean_reversion", "crypto_majors", 1, metadata={"portfolio_weight": 0.5})
        high_weight = SleeveSignal("bundle", "cross_asset_relative_strength", "BTC", 1, 0.75, 30.0, 12.0, 0.0, 0.0, "bull", "cross_asset_relative_strength", "crypto_majors", 1, metadata={"portfolio_weight": 1.2})
        allocated, _ = allocator.allocate([low_weight, high_weight], current_positions={}, equity=100_000.0)

        by_symbol = {signal.symbol: signal for signal in allocated}
        self.assertGreater(by_symbol["BTC"].metadata["allocator_score"], by_symbol["ETH"].metadata["allocator_score"])
        self.assertAlmostEqual(abs(by_symbol["BTC"].target_notional), 12_000.0)
        self.assertAlmostEqual(abs(by_symbol["ETH"].target_notional), 5_000.0)


if __name__ == "__main__":
    unittest.main()
