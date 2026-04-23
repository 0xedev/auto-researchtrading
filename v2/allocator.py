from __future__ import annotations

from collections import defaultdict

from .types import PortfolioConfig, SleeveSignal


class PortfolioAllocator:
    def __init__(self, config: PortfolioConfig):
        self.config = config

    def allocate(
        self,
        signals: list[SleeveSignal],
        current_positions: dict[str, float],
        equity: float,
        current_position_meta: dict[str, dict] | None = None,
    ) -> tuple[list[SleeveSignal], list[dict]]:
        if equity <= 0:
            return [], []

        rejections: list[dict] = []
        shortlisted: dict[str, SleeveSignal] = {}
        for signal in signals:
            existing = shortlisted.get(signal.symbol)
            if existing is None or self._score(signal) > self._score(existing):
                shortlisted[signal.symbol] = signal

        ranked = sorted(shortlisted.values(), key=self._score, reverse=True)

        allocated: list[SleeveSignal] = []
        gross_limit = equity * self.config.max_gross_leverage
        net_limit = equity * self.config.max_net_exposure_pct
        gross_used = sum(abs(v) for v in current_positions.values())
        net_used = sum(v for v in current_positions.values())
        current_position_meta = current_position_meta or {}
        sleeve_notional, cluster_notional, sleeve_open_positions, cluster_open_positions = self._seed_current_exposure(
            current_positions,
            current_position_meta,
        )
        sleeve_new_positions = defaultdict(int)
        cluster_new_positions = defaultdict(int)

        short_candidates = sum(1 for signal in ranked if signal.side < 0)
        short_selected = 0

        for signal in sorted(
            ranked,
            key=lambda signal: self._effective_score(
                signal,
                sleeve_new_positions,
                cluster_new_positions,
                sleeve_open_positions,
                cluster_open_positions,
                sleeve_notional,
                cluster_notional,
                equity,
            ),
            reverse=True,
        ):
            tier = self._tier(signal.confidence)
            if tier == "C":
                rejections.append({"symbol": signal.symbol, "sleeve": signal.sleeve, "reason": "tier_c"})
                continue

            current = current_positions.get(signal.symbol, 0.0)
            is_new_position = abs(current) < 1.0
            if is_new_position and sleeve_new_positions[signal.sleeve] >= self.config.max_new_positions_per_sleeve:
                rejections.append({"symbol": signal.symbol, "sleeve": signal.sleeve, "reason": "sleeve_new_position_cap"})
                continue
            if is_new_position and cluster_new_positions[signal.cluster] >= self.config.max_new_positions_per_cluster:
                rejections.append({"symbol": signal.symbol, "sleeve": signal.sleeve, "reason": "cluster_new_position_cap"})
                continue

            target_pct = self.config.tier_a_notional_pct if tier == "A" else self.config.tier_b_notional_pct
            target_pct *= max(float(signal.metadata.get("portfolio_weight", 1.0) or 1.0), 0.0)
            target_notional = equity * target_pct * signal.side

            if abs(target_notional) > equity * self.config.max_per_symbol_pct:
                target_notional = equity * self.config.max_per_symbol_pct * signal.side

            if sleeve_notional[signal.sleeve] + abs(target_notional) > equity * self.config.max_per_sleeve_pct:
                rejections.append({"symbol": signal.symbol, "sleeve": signal.sleeve, "reason": "sleeve_cap"})
                continue

            if cluster_notional[signal.cluster] + abs(target_notional) > equity * self.config.max_per_cluster_pct:
                rejections.append({"symbol": signal.symbol, "sleeve": signal.sleeve, "reason": "cluster_cap"})
                continue

            if gross_used + abs(target_notional) > gross_limit:
                rejections.append({"symbol": signal.symbol, "sleeve": signal.sleeve, "reason": "gross_cap"})
                continue

            if abs(net_used + target_notional) > net_limit:
                rejections.append({"symbol": signal.symbol, "sleeve": signal.sleeve, "reason": "net_cap"})
                continue

            signal.target_notional = float(target_notional)
            signal.metadata["allocation_tier"] = tier
            signal.metadata["allocator_score"] = float(self._score(signal))
            signal.metadata["allocator_effective_score"] = float(
                self._effective_score(
                    signal,
                    sleeve_new_positions,
                    cluster_new_positions,
                    sleeve_open_positions,
                    cluster_open_positions,
                    sleeve_notional,
                    cluster_notional,
                    equity,
                )
            )
            allocated.append(signal)
            gross_used += abs(target_notional)
            net_used += target_notional
            sleeve_notional[signal.sleeve] += abs(target_notional)
            cluster_notional[signal.cluster] += abs(target_notional)
            if is_new_position:
                sleeve_new_positions[signal.sleeve] += 1
                cluster_new_positions[signal.cluster] += 1
                sleeve_open_positions[signal.sleeve] += 1
                cluster_open_positions[signal.cluster] += 1
            if signal.side < 0:
                short_selected += 1

        if short_candidates and allocated:
            realized_short_share = short_selected / len(allocated)
            if realized_short_share < self.config.short_min_share:
                rejections.append(
                    {
                        "symbol": "*portfolio*",
                        "sleeve": "short_share",
                        "reason": f"short_share_below_floor:{realized_short_share:.3f}",
                    }
                )

        return allocated, rejections

    def _tier(self, confidence: float) -> str:
        if confidence >= self.config.tier_a_confidence:
            return "A"
        if confidence >= self.config.tier_b_confidence:
            return "B"
        return "C"

    @staticmethod
    def _score(signal: SleeveSignal) -> float:
        edge = max(signal.expected_edge_bps, 0.0)
        weight = max(float(signal.metadata.get("portfolio_weight", 1.0) or 1.0), 0.0)
        return edge * max(signal.confidence, 0.0) * weight

    @staticmethod
    def _seed_current_exposure(
        current_positions: dict[str, float],
        current_position_meta: dict[str, dict],
    ) -> tuple[defaultdict[str, float], defaultdict[str, float], defaultdict[str, int], defaultdict[str, int]]:
        sleeve_notional = defaultdict(float)
        cluster_notional = defaultdict(float)
        sleeve_open_positions = defaultdict(int)
        cluster_open_positions = defaultdict(int)
        for symbol, notional in current_positions.items():
            if abs(notional) < 1.0:
                continue
            meta = current_position_meta.get(symbol, {}) or {}
            sleeve = str(meta.get("sleeve", "") or "")
            cluster = str(meta.get("cluster", "") or "")
            if sleeve:
                sleeve_notional[sleeve] += abs(notional)
                sleeve_open_positions[sleeve] += 1
            if cluster:
                cluster_notional[cluster] += abs(notional)
                cluster_open_positions[cluster] += 1
        return sleeve_notional, cluster_notional, sleeve_open_positions, cluster_open_positions

    def _effective_score(
        self,
        signal: SleeveSignal,
        sleeve_new_positions: dict[str, int],
        cluster_new_positions: dict[str, int],
        sleeve_open_positions: dict[str, int],
        cluster_open_positions: dict[str, int],
        sleeve_notional: dict[str, float],
        cluster_notional: dict[str, float],
        equity: float,
    ) -> float:
        base = self._score(signal)
        penalty = 1.0
        penalty += sleeve_new_positions[signal.sleeve] * self.config.diversification_penalty
        penalty += cluster_new_positions[signal.cluster] * (self.config.diversification_penalty * 0.5)
        penalty += sleeve_open_positions[signal.sleeve] * (self.config.diversification_penalty * 0.75)
        penalty += cluster_open_positions[signal.cluster] * (self.config.diversification_penalty * 0.35)
        if equity > 0:
            penalty += (sleeve_notional[signal.sleeve] / equity) * 0.50
            penalty += (cluster_notional[signal.cluster] / equity) * 0.25
        return base / penalty
