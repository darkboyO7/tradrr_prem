"""Tests for Zero-Knowledge / Cryptographic Trade Proof & Verifiable Audit Vault."""

import hashlib
import pytest

from compliance.trade_vault import (
    BatchSettlementAttestation,
    CryptographicTradeVault,
    MerkleAuditProof,
    SolvencyCommitmentProof,
    TradeFillReceipt,
    VaultProofType,
)


def test_merkle_tree_construction_and_inclusion_proof():
    """Verify that binary Merkle tree constructs valid roots and verifies inclusion proofs."""
    vault = CryptographicTradeVault(batch_id="BATCH_TEST_001")

    # Ingest 5 trade fills (odd number tests odd-leaf duplication logic)
    r1 = vault.record_fill("TRD_1", "ORD_1", "BTC-USD", "BUY", 65000.0, 1.5, "BINANCE", 1.2)
    r2 = vault.record_fill("TRD_2", "ORD_2", "ETH-USD", "SELL", 3200.0, 10.0, "COINBASE", 2.5)
    r3 = vault.record_fill("TRD_3", "ORD_3", "SOL-USD", "BUY", 150.0, 50.0, "KRAKEN", 0.8)
    r4 = vault.record_fill("TRD_4", "ORD_4", "BTC-USD", "SELL", 65100.0, 0.5, "BINANCE", 0.4)
    r5 = vault.record_fill("TRD_5", "ORD_5", "AVAX-USD", "BUY", 35.0, 100.0, "COINBASE", 0.7)

    assert vault.total_trades == 5
    assert len(vault.merkle_root) == 64
    assert vault.merkle_root != "0" * 64

    # Generate inclusion proof for TRD_3
    proof = vault.generate_inclusion_proof("TRD_3")
    assert proof is not None
    assert proof.target_hash == r3.leaf_hash
    assert proof.root_hash == vault.merkle_root
    assert proof.is_valid is True

    # Standalone verification
    assert CryptographicTradeVault.verify_proof(proof) is True

    # Tamper with target hash and verify rejection
    tampered_proof = MerkleAuditProof(
        target_hash="f" * 64,
        root_hash=proof.root_hash,
        leaf_index=proof.leaf_index,
        audit_path=proof.audit_path,
        is_valid=True,
    )
    assert CryptographicTradeVault.verify_proof(tampered_proof) is False


def test_trade_inclusion_all_leaves():
    """Verify all individual leaves in a tree generate mathematically verifiable proofs."""
    vault = CryptographicTradeVault(batch_id="BATCH_POW_2")

    # Ingest exactly 4 trades (power of 2)
    trades = [
        ("T1", "O1", "BTC-USD", 65000.0, 1.0),
        ("T2", "O2", "ETH-USD", 3200.0, 5.0),
        ("T3", "O3", "SOL-USD", 150.0, 20.0),
        ("T4", "O4", "LINK-USD", 18.0, 100.0),
    ]

    for tid, oid, sym, px, qty in trades:
        vault.record_fill(tid, oid, sym, "BUY", px, qty)

    assert vault.total_trades == 4

    for tid, _, _, _, _ in trades:
        proof = vault.generate_inclusion_proof(tid)
        assert proof is not None
        assert CryptographicTradeVault.verify_proof(proof) is True

    # Unknown trade ID returns None
    assert vault.generate_inclusion_proof("T_NONEXISTENT") is None


def test_batch_settlement_attestation():
    """Verify aggregated batch settlement attestation contains correct totals and cryptographic digest."""
    vault = CryptographicTradeVault(batch_id="BATCH_SETTLE_99")
    vault.record_fill("T1", "O1", "BTC-USD", "BUY", 60000.0, 2.0, timestamp_ns=1000)
    vault.record_fill("T2", "O2", "ETH-USD", "SELL", 3000.0, 10.0, timestamp_ns=2000)

    attestation = vault.generate_settlement_attestation()
    assert isinstance(attestation, BatchSettlementAttestation)
    assert attestation.batch_id == "BATCH_SETTLE_99"
    assert attestation.trade_count == 2
    assert attestation.total_notional_usd == 150000.0  # (60000*2) + (3000*10)
    assert attestation.symbols == ["BTC-USD", "ETH-USD"]
    assert attestation.merkle_root == vault.merkle_root
    assert attestation.start_timestamp_ns == 1000
    assert attestation.end_timestamp_ns == 2000
    assert len(attestation.attestation_signature_hash) == 64


def test_zero_knowledge_solvency_commitment():
    """Verify cryptographic solvency proof verifies Assets >= Liabilities without balance exposure."""
    assets = {
        "USD": 500000.0,
        "BTC": 20.0,  # 20 * 65000 = 1,300,000
        "ETH": 100.0,  # 100 * 3200 = 320,000
    }
    # Total Assets = 500k + 1.3M + 320k = 2,120,000 USD

    liabilities_solvent = {
        "USD": 200000.0,
        "BTC": 10.0,  # 10 * 65000 = 650,000
        "ETH": 50.0,  # 50 * 3200 = 160,000
    }
    # Total Liabilities = 200k + 650k + 160k = 1,010,000 USD

    proof_solvent = CryptographicTradeVault.verify_solvency_commitment(
        epoch_id="EPOCH_2026_Q4",
        asset_balances=assets,
        liability_balances=liabilities_solvent,
    )

    assert isinstance(proof_solvent, SolvencyCommitmentProof)
    assert proof_solvent.is_solvent is True
    assert proof_solvent.total_assets_usd == 2120000.0
    assert proof_solvent.total_liabilities_usd == 1010000.0
    assert proof_solvent.solvency_ratio > 2.0
    assert len(proof_solvent.assets_commitment_hash) == 64
    assert len(proof_solvent.liabilities_commitment_hash) == 64
    assert len(proof_solvent.root_commitment) == 64

    # Test insolvent condition
    liabilities_insolvent = {
        "USD": 1000000.0,
        "BTC": 30.0,  # 30 * 65000 = 1,950,000
    }
    # Total Liabilities = 2.95M > 2.12M

    proof_insolvent = CryptographicTradeVault.verify_solvency_commitment(
        epoch_id="EPOCH_INSOLVENT",
        asset_balances=assets,
        liability_balances=liabilities_insolvent,
    )
    assert proof_insolvent.is_solvent is False
    assert proof_insolvent.solvency_ratio < 1.0


def test_empty_vault_and_serialization():
    """Verify empty vault behaviors and receipt leaf digest hashing stability."""
    vault = CryptographicTradeVault()
    assert vault.total_trades == 0
    assert vault.merkle_root == "0" * 64

    attestation = vault.generate_settlement_attestation()
    assert attestation.trade_count == 0
    assert attestation.total_notional_usd == 0.0

    # Receipt hashing determinism
    r = TradeFillReceipt(
        trade_id="TRD_STABLE",
        order_id="ORD_STABLE",
        symbol="BTC-USD",
        side="BUY",
        price=65000.0,
        quantity=1.0,
        venue="BINANCE",
        fee_usd=1.0,
        timestamp_ns=1700000000000,
    )
    assert r.leaf_hash == r.compute_leaf_hash()
