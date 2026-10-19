"""Trading Intelligence Platform - Microstructural Alpha Signal Generator.

Produces discrete signals based on order book depth imbalance, spread expansion,
and liquidity vacuum risk states to augment macro/technical trade decisions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from models.base import SignalAction


@dataclass
class MicrostructureSignal:
    """Discrete trade intelligence setup derived from L2 order book microstructure."""

    signal_name: str
    action: SignalAction
    confidence: float  # 0.0 to 1.0
    imbalance: float
    spread_bps: float
    is_vacuum: bool = False
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class MicrostructureSignalGenerator:
    """Evaluates engineered microstructure feature rows to identify order book imbalances."""

    def __init__(
        self,
        imbalance_threshold: float = 0.35,
        max_acceptable_spread_bps: float = 20.0,
    ) -> None:
        self.imbalance_threshold = imbalance_threshold
        self.max_acceptable_spread_bps = max_acceptable_spread_bps

    def evaluate_bar(self, row: pd.Series) -> Optional[MicrostructureSignal]:
        """Evaluate order book features and generate microstructure signals."""
        obi = float(row.get("obi_top5", 0.0))
        spread_bps = float(row.get("spread_bps", 5.0))
        vacuum = int(row.get("liquidity_vacuum", 0))

        # 1. Circuit Breaker / Liquidity Vacuum
        if vacuum == 1 or spread_bps > self.max_acceptable_spread_bps:
            return MicrostructureSignal(
                signal_name="LIQUIDITY_VACUUM_HALT",
                action=SignalAction.NO_TRADE,
                confidence=0.0,
                imbalance=obi,
                spread_bps=spread_bps,
                is_vacuum=True,
                reasons=[f"Liquidity vacuum active: Spread {spread_bps:.1f} bps breaches threshold"],
                metrics={"obi": obi, "spread_bps": spread_bps, "vacuum": True},
            )

        # 2. Bullish Order Book Imbalance (Bid Dominance)
        if obi >= self.imbalance_threshold:
            conf = min(0.90, 0.50 + (obi * 0.40))
            return MicrostructureSignal(
                signal_name="ORDER_BOOK_IMBALANCE_BUY",
                action=SignalAction.BUY,
                confidence=conf,
                imbalance=obi,
                spread_bps=spread_bps,
                is_vacuum=False,
                reasons=[f"Strong bid-side depth imbalance (+{obi*100:.1f}%) with tight spread ({spread_bps:.1f} bps)"],
                metrics={"obi": obi, "spread_bps": spread_bps},
            )

        # 3. Bearish Order Book Imbalance (Ask Dominance)
        if obi <= -self.imbalance_threshold:
            conf = min(0.90, 0.50 + (abs(obi) * 0.40))
            return MicrostructureSignal(
                signal_name="ORDER_BOOK_IMBALANCE_SELL",
                action=SignalAction.SELL,
                confidence=conf,
                imbalance=obi,
                spread_bps=spread_bps,
                is_vacuum=False,
                reasons=[f"Strong ask-side depth imbalance ({obi*100:.1f}%) with tight spread ({spread_bps:.1f} bps)"],
                metrics={"obi": obi, "spread_bps": spread_bps},
            )

        return None
