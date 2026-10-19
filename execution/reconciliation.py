"""Trading Intelligence Platform - Position & Balance Reconciler.

Audits and synchronizes internal state with external broker / exchange balances,
detecting ghost orders, missed fills, and quantity drift.
"""

from typing import Any

from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger(__name__)


class PositionDiscrepancy(BaseModel):
    """Specific discrepancy detected between local and remote broker state."""

    symbol: str
    issue_type: str  # MISSING_LOCAL, MISSING_REMOTE, QUANTITY_MISMATCH, SIDE_MISMATCH
    local_qty: float
    remote_qty: float
    description: str


class ReconciliationReport(BaseModel):
    """Result of an automated portfolio reconciliation audit."""

    is_synchronized: bool
    discrepancies: list[PositionDiscrepancy] = Field(default_factory=list)
    local_position_count: int
    remote_position_count: int
    summary: str


class PositionReconciler:
    """Detects state desynchronization between system memory and exchange balances."""

    @staticmethod
    def reconcile(
        local_positions: dict[str, Any],  # symbol -> position dict or model
        remote_positions: list[dict[str, Any]],  # list of exchange positions
        tolerance: float = 1e-5,
    ) -> ReconciliationReport:
        """Compare local portfolio positions against exchange-reported positions."""
        discrepancies: list[PositionDiscrepancy] = []

        remote_map: dict[str, dict[str, Any]] = {p["symbol"].upper(): p for p in remote_positions}
        local_map: dict[str, Any] = {s.upper(): p for s, p in local_positions.items()}

        # 1. Check all local positions exist remotely
        for sym, loc in local_map.items():
            loc_qty = float(getattr(loc, "quantity", loc.get("quantity", 0.0) if isinstance(loc, dict) else 0.0))
            raw_side = getattr(loc, "side", loc.get("side", "BUY") if isinstance(loc, dict) else "BUY")
            loc_side = raw_side.value if hasattr(raw_side, "value") else str(raw_side)
            loc_side = str(loc_side).upper()

            if sym not in remote_map:
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=sym,
                        issue_type="MISSING_REMOTE",
                        local_qty=loc_qty,
                        remote_qty=0.0,
                        description=f"Position in {sym} exists in local state but not on remote broker.",
                    )
                )
            else:
                rem = remote_map[sym]
                rem_qty = float(rem.get("quantity", 0.0))
                rem_side = str(rem.get("side", "BUY")).upper()

                if abs(loc_qty - rem_qty) > tolerance:
                    discrepancies.append(
                        PositionDiscrepancy(
                            symbol=sym,
                            issue_type="QUANTITY_MISMATCH",
                            local_qty=loc_qty,
                            remote_qty=rem_qty,
                            description=f"Quantity mismatch in {sym}: Local={loc_qty}, Remote={rem_qty}",
                        )
                    )

                if loc_side != rem_side:
                    discrepancies.append(
                        PositionDiscrepancy(
                            symbol=sym,
                            issue_type="SIDE_MISMATCH",
                            local_qty=loc_qty,
                            remote_qty=rem_qty,
                            description=f"Direction mismatch in {sym}: Local={loc_side}, Remote={rem_side}",
                        )
                    )

        # 2. Check for ghost positions on remote broker that local state missed
        for sym, rem in remote_map.items():
            if sym not in local_map:
                rem_qty = float(rem.get("quantity", 0.0))
                discrepancies.append(
                    PositionDiscrepancy(
                        symbol=sym,
                        issue_type="MISSING_LOCAL",
                        local_qty=0.0,
                        remote_qty=rem_qty,
                        description=f"Ghost position in {sym} on remote broker not tracked in local system.",
                    )
                )

        is_sync = len(discrepancies) == 0
        summary = "Portfolio is 100% synchronized with exchange." if is_sync else f"Detected {len(discrepancies)} position discrepancies!"

        if not is_sync:
            logger.warning("Position Reconciliation Alert", discrepancies=len(discrepancies), summary=summary)

        return ReconciliationReport(
            is_synchronized=is_sync,
            discrepancies=discrepancies,
            local_position_count=len(local_map),
            remote_position_count=len(remote_map),
            summary=summary,
        )
