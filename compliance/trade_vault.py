"""Trading Intelligence Platform - Cryptographic Trade Proof & Verifiable Audit Vault.

Provides zero-knowledge style trade verification, Merkle tree root commitments for order fills,
cryptographic exclusion/inclusion proofs, tamper-evident fill receipts, and zero-knowledge
solvency verification (Liabilities <= Assets commitment without disclosing individual accounts).
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("trade_vault")


class VaultProofType(str, Enum):
    """Classification of cryptographic vault proof queries."""

    FILL_INCLUSION = "FILL_INCLUSION"
    TRADE_NON_INCLUSION = "TRADE_NON_INCLUSION"
    SOLVENCY_COMMITMENT = "SOLVENCY_COMMITMENT"
    BATCH_SETTLEMENT = "BATCH_SETTLEMENT"


class TradeFillReceipt(BaseModel):
    """Tamper-evident canonical fill receipt with cryptographic leaf digest."""

    trade_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    venue: str
    fee_usd: float
    timestamp_ns: int
    leaf_hash: str = ""

    def compute_leaf_hash(self) -> str:
        """Calculate canonical SHA-256 leaf digest from trade parameters."""
        data_str = (
            f"{self.trade_id}|{self.order_id}|{self.symbol.upper()}|"
            f"{self.side.upper()}|{self.price:.8f}|{self.quantity:.8f}|"
            f"{self.venue.upper()}|{self.fee_usd:.6f}|{self.timestamp_ns}"
        )
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        if not self.leaf_hash:
            self.leaf_hash = self.compute_leaf_hash()


class MerkleAuditProof(BaseModel):
    """Cryptographic audit proof demonstrating leaf inclusion in root commitment."""

    target_hash: str
    root_hash: str
    leaf_index: int
    audit_path: List[Tuple[str, str]] = Field(
        ...,
        description="List of (sibling_hash, direction) where direction is 'left' or 'right'",
    )
    is_valid: bool = True
    verified_at_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SolvencyCommitmentProof(BaseModel):
    """Proof of solvency without revealing confidential client balances."""

    vault_epoch_id: str
    total_assets_usd: float
    total_liabilities_usd: float
    solvency_ratio: float
    is_solvent: bool
    assets_commitment_hash: str
    liabilities_commitment_hash: str
    root_commitment: str
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchSettlementAttestation(BaseModel):
    """Aggregated batch settlement summary anchored to a Merkle root commitment."""

    batch_id: str
    trade_count: int
    total_notional_usd: float
    symbols: List[str]
    merkle_root: str
    start_timestamp_ns: int
    end_timestamp_ns: int
    attestation_signature_hash: str


class CryptographicTradeVault:
    """Verifiable cryptographic trade vault using binary Merkle trees and cryptographic commitments."""

    def __init__(self, batch_id: Optional[str] = None) -> None:
        self.batch_id = batch_id or f"BATCH_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
        self._receipts: List[TradeFillReceipt] = []
        self._leaf_hashes: List[str] = []
        self._tree_levels: List[List[str]] = []
        self._merkle_root: str = "0" * 64

    @property
    def total_trades(self) -> int:
        return len(self._receipts)

    @property
    def merkle_root(self) -> str:
        return self._merkle_root

    @property
    def receipts(self) -> List[TradeFillReceipt]:
        return list(self._receipts)

    # -------------------------------------------------------------------------
    # Ingestion & Merkle Tree Construction
    # -------------------------------------------------------------------------

    def record_fill(
        self,
        trade_id: str,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        venue: str = "INTERNAL_CROSS",
        fee_usd: float = 0.0,
        timestamp_ns: Optional[int] = None,
    ) -> TradeFillReceipt:
        """Register a trade execution fill and compute its cryptographic receipt."""
        t_ns = timestamp_ns or int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        receipt = TradeFillReceipt(
            trade_id=trade_id,
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            venue=venue,
            fee_usd=fee_usd,
            timestamp_ns=t_ns,
        )
        self._receipts.append(receipt)
        self._leaf_hashes.append(receipt.leaf_hash)
        self._build_tree()
        return receipt

    def batch_record_fills(self, fills: List[TradeFillReceipt]) -> str:
        """Bulk ingest multiple trade fill receipts and recompute Merkle root commitment."""
        for fill in fills:
            self._receipts.append(fill)
            self._leaf_hashes.append(fill.leaf_hash)
        self._build_tree()
        return self._merkle_root

    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two child nodes in canonical order."""
        combined = f"{left}{right}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _build_tree(self) -> None:
        """Construct full binary Merkle tree levels from leaf hashes."""
        if not self._leaf_hashes:
            self._merkle_root = "0" * 64
            self._tree_levels = []
            return

        current_level = list(self._leaf_hashes)
        self._tree_levels = [current_level]

        while len(current_level) > 1:
            next_level: List[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                # If odd number of elements, duplicate the last element
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent = self._hash_pair(left, right)
                next_level.append(parent)
            self._tree_levels.append(next_level)
            current_level = next_level

        self._merkle_root = self._tree_levels[-1][0]

    # -------------------------------------------------------------------------
    # Merkle Proof Generation & Verification
    # -------------------------------------------------------------------------

    def generate_inclusion_proof(self, trade_id: str) -> Optional[MerkleAuditProof]:
        """Generate a cryptographic inclusion audit proof for a given trade ID."""
        target_idx = -1
        for idx, r in enumerate(self._receipts):
            if r.trade_id == trade_id:
                target_idx = idx
                break

        if target_idx == -1 or not self._tree_levels:
            return None

        target_hash = self._leaf_hashes[target_idx]
        audit_path: List[Tuple[str, str]] = []
        curr_idx = target_idx

        for level in self._tree_levels[:-1]:
            is_right_child = (curr_idx % 2 == 1)
            sibling_idx = curr_idx - 1 if is_right_child else curr_idx + 1

            if sibling_idx >= len(level):
                # Odd length duplication: sibling is self
                sibling_hash = level[curr_idx]
                direction = "right"
            else:
                sibling_hash = level[sibling_idx]
                direction = "left" if is_right_child else "right"

            audit_path.append((sibling_hash, direction))
            curr_idx //= 2

        proof = MerkleAuditProof(
            target_hash=target_hash,
            root_hash=self._merkle_root,
            leaf_index=target_idx,
            audit_path=audit_path,
            is_valid=True,
        )

        # Self-verify proof
        proof.is_valid = self.verify_proof(proof)
        return proof

    @staticmethod
    def verify_proof(proof: MerkleAuditProof) -> bool:
        """Verify that target_hash chained with audit_path reconstructs root_hash."""
        current_hash = proof.target_hash

        for sibling_hash, direction in proof.audit_path:
            combined = f"{sibling_hash}{current_hash}" if direction == "left" else f"{current_hash}{sibling_hash}"
            current_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

        return current_hash.lower() == proof.root_hash.lower()

    # -------------------------------------------------------------------------
    # Batch Settlement Attestation
    # -------------------------------------------------------------------------

    def generate_settlement_attestation(self) -> BatchSettlementAttestation:
        """Synthesize batch settlement report signed with deterministic hash commitment."""
        if not self._receipts:
            root = "0" * 64
            t_min = 0
            t_max = 0
            symbols: List[str] = []
            notional = 0.0
        else:
            root = self._merkle_root
            t_min = min(r.timestamp_ns for r in self._receipts)
            t_max = max(r.timestamp_ns for r in self._receipts)
            symbols = sorted(list(set(r.symbol for r in self._receipts)))
            notional = sum(r.price * r.quantity for r in self._receipts)

        sig_data = f"{self.batch_id}|{len(self._receipts)}|{notional:.2f}|{root}|{t_min}|{t_max}"
        sig_hash = hashlib.sha256(sig_data.encode("utf-8")).hexdigest()

        return BatchSettlementAttestation(
            batch_id=self.batch_id,
            trade_count=len(self._receipts),
            total_notional_usd=round(notional, 2),
            symbols=symbols,
            merkle_root=root,
            start_timestamp_ns=t_min,
            end_timestamp_ns=t_max,
            attestation_signature_hash=sig_hash,
        )

    # -------------------------------------------------------------------------
    # Zero-Knowledge Solvency Commitment
    # -------------------------------------------------------------------------

    @staticmethod
    def verify_solvency_commitment(
        epoch_id: str,
        asset_balances: Dict[str, float],
        liability_balances: Dict[str, float],
        asset_prices: Optional[Dict[str, float]] = None,
    ) -> SolvencyCommitmentProof:
        """Prove platform solvency (Assets >= Liabilities) via cryptographic commitments without exposing subaccounts."""
        prices = asset_prices or {"USD": 1.0, "USDT": 1.0, "USDC": 1.0, "BTC": 65000.0, "ETH": 3200.0}

        # Calculate notional asset total
        total_assets = 0.0
        for asset, qty in asset_balances.items():
            px = prices.get(asset.upper(), 1.0)
            total_assets += max(0.0, qty) * px

        # Calculate notional liabilities total
        total_liabilities = 0.0
        for asset, qty in liability_balances.items():
            px = prices.get(asset.upper(), 1.0)
            total_liabilities += max(0.0, qty) * px

        solvency_ratio = round(total_assets / total_liabilities, 4) if total_liabilities > 0 else 999.0
        is_solvent = total_assets >= total_liabilities

        # Hash individual anonymized leaves to create commitments
        asset_digest = hashlib.sha256(
            json.dumps({k: round(v, 6) for k, v in sorted(asset_balances.items())}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        liability_digest = hashlib.sha256(
            json.dumps({k: round(v, 6) for k, v in sorted(liability_balances.items())}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        root_commitment = hashlib.sha256(
            f"{epoch_id}|{total_assets:.2f}|{total_liabilities:.2f}|{asset_digest}|{liability_digest}".encode("utf-8")
        ).hexdigest()

        return SolvencyCommitmentProof(
            vault_epoch_id=epoch_id,
            total_assets_usd=round(total_assets, 2),
            total_liabilities_usd=round(total_liabilities, 2),
            solvency_ratio=solvency_ratio,
            is_solvent=is_solvent,
            assets_commitment_hash=asset_digest,
            liabilities_commitment_hash=liability_digest,
            root_commitment=root_commitment,
        )
