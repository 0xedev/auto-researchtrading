from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


SideMode = Literal["long", "short", "both"]
SleeveStatus = Literal["candidate", "paper-live", "live-champion", "retired"]


@dataclass(frozen=True)
class TimeframeBundle:
    name: str
    fast_tf: str
    base_tf: str
    slow_tf: str
    execution_role: str = "base"
    label_horizon_hours: float = 24.0
    max_hold_hours: float = 48.0
    feature_window_hours: dict[str, float] = field(default_factory=dict)

    def role_timeframes(self) -> dict[str, str]:
        return {"fast": self.fast_tf, "base": self.base_tf, "slow": self.slow_tf}


@dataclass(frozen=True)
class SleeveManifest:
    name: str
    family: str
    side_mode: SideMode
    model_family: str = "xgboost"
    feature_profile: str = "price_context_plus"
    label_horizon_hours: float = 24.0
    max_hold_hours: float = 48.0
    target_sigma: float = 0.75
    min_confidence: float = 0.58
    stop_atr_mult: float = 1.5
    edge_scale_bps: float = 150.0
    cluster: str = "core"
    activation_rule: str = ""
    portfolio_weight: float = 1.0
    reentry_cooldown_hours: float = 0.0
    reentry_cooldown_scope: str = "symbol"


@dataclass
class SleeveSignal:
    bundle: str
    sleeve: str
    symbol: str
    side: int
    confidence: float
    expected_edge_bps: float
    holding_horizon_hours: float
    stop_distance: float
    target_notional: float
    regime_context: str
    reason_tag: str
    cluster: str
    timestamp: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortfolioConfig:
    max_gross_leverage: float = 2.0
    max_net_exposure_pct: float = 1.0
    max_per_symbol_pct: float = 0.18
    max_per_sleeve_pct: float = 0.30
    max_per_cluster_pct: float = 0.35
    sleeve_cap_overrides: dict[str, float] = field(default_factory=dict)
    cluster_cap_overrides: dict[str, float] = field(default_factory=dict)
    max_new_positions_per_sleeve: int = 3
    max_new_positions_per_cluster: int = 4
    diversification_penalty: float = 0.15
    sleeve_weight_overrides: dict[str, float] = field(default_factory=dict)
    tier_a_confidence: float = 0.68
    tier_b_confidence: float = 0.58
    tier_a_notional_pct: float = 0.12
    tier_b_notional_pct: float = 0.06
    short_min_share: float = 0.10
    max_participation_rate: float = 0.03
    slippage_bps: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PortfolioConfig":
        payload = payload or {}
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(**filtered)


@dataclass
class SleeveRegistryEntry:
    sleeve: str
    bundle: str
    model_set: str
    status: SleeveStatus = "candidate"
    validation_metric: float = 0.0
    oos_metric: float = 0.0
    stress_metric: float = 0.0
    concentration_pct: float = 0.0
    trades_per_day: float = 0.0
    short_share: float = 0.0
    shadow_ready: bool = False
    last_updated: str = ""
    notes: str = ""


@dataclass
class SleeveRegistry:
    entries: list[SleeveRegistryEntry] = field(default_factory=list)
