"""Trading Intelligence Platform - FastAPI Application Entrypoint.

Exposes REST APIs for system health, live stream control, candle retrieval,
regime classification, multi-source signal fusion, pre-trade risk evaluation,
portfolio tracking, and backtesting execution.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.streaming import ws_manager

from backtesting.multi_agent_lob import (
    AgentStrategyType,
    LOBEquilibriumMetrics,
    MultiAgentLOBSandbox,
    MultiAgentSimulationReport,
    SandboxAgentMetrics,
)
from backtesting.orderbook_simulator import (
    MicrostructureSimulationReport,
    OrderBookMicrostructureSimulator,
    QueuePriorityModel,
    SimOrderSide,
    SimOrderType,
)
from analysis.alpha_research import (
    AlphaResearchPipeline,
    AlphaResearchReport,
    FactorEvaluationMetrics,
    NeutralizedFactorScore,
)
from analysis.dark_pool_detection import (
    DarkPoolDetectionEngine,
    DarkPoolDetectionReport,
    DarkPrintRecord,
    HiddenLiquidityType,
    IcebergSignature,
)
from analysis.ensemble_allocator import (
    AllocationDecision,
    BanditAlgorithmType,
    MultiArmedBanditAllocator,
    RewardUpdateResult,
    StrategyArmState,
)
from analysis.arbitrage import ArbitrageStats, CrossExchangeArbitrageScanner
from analysis.flow_toxicity import (
    FlowToxicityRegime,
    OrderFlowToxicityEngine,
    ToxicityMetricsSnapshot,
    VolumeBucket,
)
from analysis.lead_lag_network import (
    InstrumentCentralityProfile,
    InstrumentNetworkRole,
    LeadLagArbitrageSignal,
    LeadLagNetworkEngine,
    LeadLagNetworkReport,
    LeadLagSignalAction,
    PairwiseLeadLagLink,
)
from analysis.yield_curve import (
    CalibratedCurveParameters,
    CalibratedYieldCurveReport,
    CurveModelType,
    TermStructureCalibrator,
    TermStructureRollDownReport,
    YieldTenorPoint,
)
from analysis.triangular_arbitrage import (
    RateGraphData,
    RateGraphEdge,
    TriangularArbitrageEngine,
    TriangularArbitrageOpportunity,
    TriangularLeg,
    TriangularScanReport,
)



from analysis.basis_arbitrage import (
    BasisArbitrageScanner,
    BasisOpportunity,
    BasisPosition,
    BasisPositionTracker,
)
from analysis.derivatives import (
    BlackScholesEngine,
    DerivativePosition,
    DerivativePortfolioReport,
    DerivativesRiskEngine,
    ImpliedVolatilitySolver,
    MertonJumpDiffusionEngine,
    OptionGreeks,
    OptionType,
    VolatilityPoint,
    VolatilitySurface,
)
from analysis.regime import MarketRegimeDetector, MarketRegimeState
from analysis.stat_arb import (
    CointegrationAnalyzer,
    CointegrationResult,
    PairsSpreadMonitor,
    PairTradingSignal,
)
from compliance import (
    AuditRecordType,
    BatchSettlementAttestation,
    BridgeProtocol,
    BridgeTransferRecord,
    BridgeTransferStatus,
    ChainFinalityProfile,
    CrossChainBridgeMonitor,
    CrossChainMonitorReport,
    CrossChainRouteQuote,
    CryptographicTradeVault,
    ExecutionRelayType,
    FIXNewOrderSingle,
    FlashbotsBundleSubmission,
    ImmutableAuditLedger,
    MEVDefenseAssessment,
    MEVDefenseGateway,
    MEVRiskLevel,
    MerkleAuditProof,
    PendingMempoolTx,
    PreFlightRejectReason,
    PreFlightRiskLimits,
    PreFlightValidationResult,
    PreFlightValidator,
    SolvencyCommitmentProof,
    SupportedChain,
    TradeFillReceipt,
    TradeSurveillanceEngine,
    VaultProofType,
)

from core.cache import cache
from core.config import settings
from core.database import close_db, get_db_session, init_db
from core.events import Candle
from core.gateway_heartbeat import (
    GatewayConnectionState,
    GatewayHealthMetrics,
    GatewayHeartbeatManager,
    GatewayHeartbeatReport,
)
from core.logging import get_logger, setup_logging
from core.recovery import (
    FailoverOrchestrator,
    NodeRole,
    PlatformStatePayload,
    SnapshotIntegrityError,
    SnapshotManager,
    SnapshotMetadata,
)

from data.candle.engine import CandleEngine
from data.ingestion.coinbase_ws import CoinbaseWebSocketProvider
from data.ingestion.pipeline import LiveIngestionPipeline
from data.normalization.normalizer import EventNormalizer
from data.orderbook import (
    ArbitrageOpportunity,
    BBOState,
    ConsolidatedOrderBook,
    DepthHeatmapGrid,
    OFIMetric,
    OrderBookHeatmapEngine,
    OrderFlowImbalanceCalculator,
    SyntheticBBO,
)
from data.storage.repository import MarketDataRepository
from data.validation.validator import DataValidator
from execution.algo import AlgoEngine, AlgoExecutionPlan, AlgoExecutionReport, AlgoOrderType
from execution.cost_model import AlmgrenChrissCostModel, ExecutionCostEstimate
from execution.cost_surface_calibrator import (
    CalibratedCostParameters,
    CostSurfaceCalibrator,
    CostSurfaceGrid,
    CostSurfacePoint,
    SlippagePredictionRequest,
    SlippagePredictionResult,
)
from execution.crossing import (

    CrossingPoolMetrics,
    CrossingResult,
    CrossingSide,
    CrossingTimeInForce,
    InternalCrossingEngine,
    InternalCrossingOrder,
)
from execution.hedging import SmartHedgeConfig, SmartHedgePlan, SmartHedgingEngine
from execution.adaptive_market_maker import (
    ASQuotingRegime,
    AdaptiveMarketMakingOptimizer,
    AdaptiveMMConfig,
    AdaptiveMMQuotingReport,
    AdaptiveQuoteTier,
)
from execution.market_maker import (
    DMMConfig,
    DMMQuoteLevel,
    DMMQuoteSurface,
    DynamicMarketMakerEngine,
    InventoryState,
)
from execution.router import ChildOrderSlice, RoutingPlan, RoutingPolicy, SmartOrderRouter
from execution.synthetic_sor import (
    DynamicLiquidityRoutingPlan,
    SOROptimizationObjective,
    SyntheticChildAllocation,
    SyntheticSmartOrderRouter,
    VenueBookDepth,
    VenueLiquidityType,
)
from execution.tca import (
    ImplementationShortfallDecomposition,
    TCAPortfolioReport,
    TradeExecutionRecord,
    TransactionCostAnalysisEngine,
    VenueExecutionScorecard,
)
from execution.volume_profile import VolumeProfiler
from monitoring.health import check_system_health
from monitoring.metrics import get_latest_metrics_text
from monitoring.stress_harness import (
    PipelineThroughputBenchmark,
    StressBenchmarkResult,
    SyntheticTickStormGenerator,
)
from paper_trading.engine import PaperTradingEngine
from risk.collateral_rehypothecation import (
    CollateralRebalancePlan,
    CollateralRehypothecationEngine,
    CollateralTransferLeg,
    TransferStatus,
    TransferUrgency,
    VenueCollateralAsset,
    VenueCollateralState,
)
from risk.covariance_shrinkage import (
    CovarianceShrinkageEngine,
    MinimumVarianceWeights,
    ShrinkageResult,
    ShrinkageTargetType,
)
from risk.engine import RiskDecision, RiskEngine
from risk.factor_model import PortfolioFactorAttribution, SystematicFactorModel
from risk.liquidation_cascades import (
    DynamicCollateralAsset,
    DynamicMarginHealthEngine,
    LiquidationLadderPlan,
    SolvencyState,
    SolvencyStressSimulationResult,
)
from risk.portfolio_margin import (
    CollateralAsset,
    MarginPosition,
    PortfolioMarginEngine,
    PortfolioMarginReport,
)
from risk.quantum_rebalancer import (
    CardinalityOptimizationConfig,
    CardinalityOptimizationReport,
    OptimizationSolverMethod,
    QuantumCardinalityRebalancer,
    RebalanceAllocationItem,
)
from risk.rebalancing import PortfolioRebalancer, RebalancePlan
from risk.stress_testing import PortfolioStressTester, StressScenarioReport, StressScenarioType
from risk.var_model import PortfolioRiskModel, VaRMetrics
from risk.volatility_targeting import VolatilityTargetConfig, VolatilityTargetingEngine
from signals.fusion import FinalSignalDecision, SignalFusionEngine
from signals.technical import TechnicalSignalGenerator

setup_logging(level=settings.LOG_LEVEL, json_format=settings.LOG_JSON)
logger = get_logger(__name__)

# Global Singletons
_pipeline: LiveIngestionPipeline | None = None
_paper_engine: PaperTradingEngine | None = None
_risk_engine: RiskEngine | None = None


def get_pipeline() -> LiveIngestionPipeline:
    """Singleton getter for the live data pipeline."""
    global _pipeline
    if _pipeline is None:
        provider = CoinbaseWebSocketProvider(
            symbols=settings.DEFAULT_SYMBOLS,
            channels=["matches", "heartbeat"],
            stale_threshold_seconds=settings.STALE_DATA_THRESHOLD_SECONDS,
        )
        validator = DataValidator()
        normalizer = EventNormalizer()
        candle_engine = CandleEngine(timeframe=settings.PRIMARY_TIMEFRAME)
        _pipeline = LiveIngestionPipeline(
            provider=provider,
            validator=validator,
            normalizer=normalizer,
            candle_engine=candle_engine,
        )
    return _pipeline


def get_paper_engine() -> PaperTradingEngine:
    global _paper_engine
    if _paper_engine is None:
        _paper_engine = PaperTradingEngine(initial_cash=100000.0)
    return _paper_engine


def get_risk_engine() -> RiskEngine:
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine(initial_capital=100000.0)
    return _risk_engine


_arbitrage_scanner: CrossExchangeArbitrageScanner | None = None


def get_arbitrage_scanner() -> CrossExchangeArbitrageScanner:
    global _arbitrage_scanner
    if _arbitrage_scanner is None:
        _arbitrage_scanner = CrossExchangeArbitrageScanner()
        book = ConsolidatedOrderBook("BTC-USD")
        book.update_venue_snapshot(
            "binance",
            bids=[(65000.0, 2.5), (64990.0, 4.0)],
            asks=[(65010.0, 1.8), (65020.0, 3.2)],
        )
        book.update_venue_snapshot(
            "kraken",
            bids=[(64995.0, 1.2), (64985.0, 2.0)],
            asks=[(65005.0, 2.0), (65015.0, 4.5)],
        )
        book.update_venue_snapshot(
            "coinbase",
            bids=[(65002.0, 3.0), (64992.0, 5.0)],
            asks=[(65012.0, 2.5), (65025.0, 3.0)],
        )
        _arbitrage_scanner.register_book("BTC-USD", book)
    return _arbitrage_scanner


_basis_scanner: BasisArbitrageScanner | None = None
_basis_tracker: BasisPositionTracker | None = None


def get_basis_scanner() -> BasisArbitrageScanner:
    global _basis_scanner
    if _basis_scanner is None:
        _basis_scanner = BasisArbitrageScanner()
        _basis_scanner.register_spot_quote("BTC", "coinbase", "BTC-USD", bid=65000.0, ask=65010.0)
        _basis_scanner.register_spot_quote("BTC", "kraken", "BTC-USD", bid=64995.0, ask=65005.0)
        _basis_scanner.register_perp_rate(
            "BTC", "binance", "BTCUSDT", rate_per_interval=0.0003, mark_price=65180.0, index_price=65005.0
        )
        _basis_scanner.register_perp_rate(
            "BTC", "bybit", "BTCUSD", rate_per_interval=0.0001, mark_price=65040.0, index_price=65005.0
        )
        _basis_scanner.register_spot_quote("ETH", "coinbase", "ETH-USD", bid=3500.0, ask=3502.0)
        _basis_scanner.register_perp_rate(
            "ETH", "binance", "ETHUSDT", rate_per_interval=0.0002, mark_price=3512.0, index_price=3501.0
        )
    return _basis_scanner


def get_basis_tracker() -> BasisPositionTracker:
    global _basis_tracker
    if _basis_tracker is None:
        _basis_tracker = BasisPositionTracker()
    return _basis_tracker


_crossing_engine: InternalCrossingEngine | None = None


def get_crossing_engine() -> InternalCrossingEngine:
    global _crossing_engine
    if _crossing_engine is None:
        _crossing_engine = InternalCrossingEngine()
        _crossing_engine._ensure_symbol_storage("BTC-USD")
        _crossing_engine._resting_sells["BTC-USD"].append(
            InternalCrossingOrder(
                symbol="BTC-USD",
                side=CrossingSide.SELL,
                quantity=1.5,
                remaining_quantity=1.5,
                limit_price=65000.0,
                time_in_force=CrossingTimeInForce.RESTING,
                strategy_id="SEED_MM_DESK",
            )
        )
        _crossing_engine._resting_buys["BTC-USD"].append(
            InternalCrossingOrder(
                symbol="BTC-USD",
                side=CrossingSide.BUY,
                quantity=2.0,
                remaining_quantity=2.0,
                limit_price=65010.0,
                time_in_force=CrossingTimeInForce.RESTING,
                strategy_id="SEED_MM_DESK",
            )
        )
    return _crossing_engine


_dmm_engine: DynamicMarketMakerEngine | None = None


def get_dmm_engine() -> DynamicMarketMakerEngine:
    global _dmm_engine
    if _dmm_engine is None:
        _dmm_engine = DynamicMarketMakerEngine()
    return _dmm_engine


_adaptive_mm_optimizer: AdaptiveMarketMakingOptimizer | None = None


def get_adaptive_market_maker() -> AdaptiveMarketMakingOptimizer:
    global _adaptive_mm_optimizer
    if _adaptive_mm_optimizer is None:
        _adaptive_mm_optimizer = AdaptiveMarketMakingOptimizer()
        _adaptive_mm_optimizer.set_inventory("BTC-USD", 0.5)
        _adaptive_mm_optimizer.set_inventory("ETH-USD", -2.0)
    return _adaptive_mm_optimizer


_audit_ledger: ImmutableAuditLedger | None = None


def get_audit_ledger() -> ImmutableAuditLedger:
    global _audit_ledger
    if _audit_ledger is None:
        _audit_ledger = ImmutableAuditLedger()
        _audit_ledger.record_event(
            record_type=AuditRecordType.SYSTEM_CONFIG_CHANGE,
            action="SYSTEM_INITIALIZED",
            payload={"environment": settings.ENVIRONMENT, "log_level": settings.LOG_LEVEL},
            actor="SYSTEM_INIT",
        )
    return _audit_ledger


_snapshot_manager: SnapshotManager | None = None
_failover_orchestrator: FailoverOrchestrator | None = None


def get_snapshot_manager() -> SnapshotManager:
    global _snapshot_manager
    if _snapshot_manager is None:
        _snapshot_manager = SnapshotManager()
    return _snapshot_manager


def get_failover_orchestrator() -> FailoverOrchestrator:
    global _failover_orchestrator
    if _failover_orchestrator is None:
        _failover_orchestrator = FailoverOrchestrator(initial_role=NodeRole.PRIMARY)
    return _failover_orchestrator


_margin_engine: PortfolioMarginEngine | None = None
_collateral_engine: CollateralRehypothecationEngine | None = None


def get_margin_engine() -> PortfolioMarginEngine:
    global _margin_engine
    if _margin_engine is None:
        _margin_engine = PortfolioMarginEngine()
    return _margin_engine


def get_collateral_engine() -> CollateralRehypothecationEngine:
    global _collateral_engine
    if _collateral_engine is None:
        v_coinbase = VenueCollateralState(
            venue_id="coinbase",
            venue_type="EXCHANGE",
            cash_usd=250000.0,
            assets={
                "BTC": VenueCollateralAsset(asset="BTC", quantity=5.0, mark_price=65000.0, haircut_pct=0.10),
            },
            margin_required_usd=40000.0,
            target_utilization_pct=65.0,
        )
        v_binance = VenueCollateralState(
            venue_id="binance",
            venue_type="EXCHANGE",
            cash_usd=80000.0,
            assets={
                "ETH": VenueCollateralAsset(asset="ETH", quantity=30.0, mark_price=3500.0, haircut_pct=0.15),
            },
            margin_required_usd=120000.0,
            target_utilization_pct=65.0,
        )
        v_fireblocks = VenueCollateralState(
            venue_id="vault_fireblocks",
            venue_type="CUSTODIAN_VAULT",
            cash_usd=500000.0,
            margin_required_usd=0.0,
            target_utilization_pct=50.0,
            transfer_fee_flat_usd=25.0,
        )
        _collateral_engine = CollateralRehypothecationEngine(
            venues={"coinbase": v_coinbase, "binance": v_binance, "vault_fireblocks": v_fireblocks}
        )
    return _collateral_engine


_ensemble_allocator: MultiArmedBanditAllocator | None = None


def get_ensemble_allocator() -> MultiArmedBanditAllocator:
    global _ensemble_allocator
    if _ensemble_allocator is None:
        _ensemble_allocator = MultiArmedBanditAllocator()
    return _ensemble_allocator


_dark_pool_engine: DarkPoolDetectionEngine | None = None


def get_dark_pool_engine() -> DarkPoolDetectionEngine:
    global _dark_pool_engine
    if _dark_pool_engine is None:
        _dark_pool_engine = DarkPoolDetectionEngine()
    return _dark_pool_engine


_preflight_validator: PreFlightValidator | None = None


def get_preflight_validator() -> PreFlightValidator:
    global _preflight_validator
    if _preflight_validator is None:
        _preflight_validator = PreFlightValidator()
    return _preflight_validator


_gateway_heartbeat_manager: GatewayHeartbeatManager | None = None


def get_gateway_heartbeat_manager() -> GatewayHeartbeatManager:
    global _gateway_heartbeat_manager
    if _gateway_heartbeat_manager is None:
        _gateway_heartbeat_manager = GatewayHeartbeatManager()
    return _gateway_heartbeat_manager


_cost_calibrator: CostSurfaceCalibrator | None = None


def get_cost_calibrator() -> CostSurfaceCalibrator:
    global _cost_calibrator
    if _cost_calibrator is None:
        _cost_calibrator = CostSurfaceCalibrator()
    return _cost_calibrator


_triangular_engine: TriangularArbitrageEngine | None = None


def get_triangular_engine() -> TriangularArbitrageEngine:
    global _triangular_engine
    if _triangular_engine is None:
        _triangular_engine = TriangularArbitrageEngine()
        _triangular_engine.update_quote("binance", "BTC-USD", bid=65000.0, ask=65005.0, bid_qty=5.0, ask_qty=5.0)
        _triangular_engine.update_quote("binance", "ETH-BTC", bid=0.048, ask=0.049, bid_qty=80.0, ask_qty=80.0)
        _triangular_engine.update_quote("binance", "ETH-USD", bid=3200.0, ask=3202.0, bid_qty=40.0, ask_qty=40.0)
        _triangular_engine.update_quote("coinbase", "BTC-USD", bid=65002.0, ask=65008.0, bid_qty=4.0, ask_qty=4.0)
        _triangular_engine.update_quote("coinbase", "ETH-USD", bid=3201.0, ask=3203.0, bid_qty=35.0, ask_qty=35.0)
        _triangular_engine.update_quote("coinbase", "SOL-USD", bid=150.0, ask=150.2, bid_qty=200.0, ask_qty=200.0)
    return _triangular_engine


_covariance_shrinkage_engine: CovarianceShrinkageEngine | None = None


def get_covariance_shrinkage_engine() -> CovarianceShrinkageEngine:
    global _covariance_shrinkage_engine
    if _covariance_shrinkage_engine is None:
        _covariance_shrinkage_engine = CovarianceShrinkageEngine()
    return _covariance_shrinkage_engine


_quantum_rebalancer: QuantumCardinalityRebalancer | None = None


def get_quantum_rebalancer() -> QuantumCardinalityRebalancer:
    global _quantum_rebalancer
    if _quantum_rebalancer is None:
        _quantum_rebalancer = QuantumCardinalityRebalancer()
    return _quantum_rebalancer


_orderbook_simulator: OrderBookMicrostructureSimulator | None = None


def get_orderbook_simulator(symbol: str = "BTC-USD") -> OrderBookMicrostructureSimulator:
    global _orderbook_simulator
    if _orderbook_simulator is None or _orderbook_simulator.symbol != symbol.upper():
        _orderbook_simulator = OrderBookMicrostructureSimulator(symbol=symbol)
        # Pre-seed initial top-of-book
        _orderbook_simulator.apply_snapshot(
            timestamp_ns=1_000_000_000,
            bids=[(65000.0, 5.0), (64990.0, 10.0), (64980.0, 15.0)],
            asks=[(65005.0, 4.0), (65015.0, 8.0), (65025.0, 12.0)],
        )
    return _orderbook_simulator


_synthetic_sor: SyntheticSmartOrderRouter | None = None


def get_synthetic_sor() -> SyntheticSmartOrderRouter:
    global _synthetic_sor
    if _synthetic_sor is None:
        _synthetic_sor = SyntheticSmartOrderRouter(step_granularity=0.1)
        _synthetic_sor.register_venue_depth(
            VenueBookDepth(
                venue="Binance",
                symbol="BTC-USD",
                bids=[(64995.0, 2.0), (64970.0, 3.0), (64950.0, 5.0)],
                asks=[(65000.0, 2.0), (65020.0, 3.0), (65030.0, 5.0)],
                maker_fee_bps=-1.0,
                taker_fee_bps=6.0,
                latency_ms=4.0,
            )
        )
        _synthetic_sor.register_venue_depth(
            VenueBookDepth(
                venue="Coinbase",
                symbol="BTC-USD",
                bids=[(64991.0, 4.0), (64985.0, 6.0), (64980.0, 10.0)],
                asks=[(65004.0, 4.0), (65008.0, 6.0), (65015.0, 10.0)],
                maker_fee_bps=-0.5,
                taker_fee_bps=12.0,
                latency_ms=15.0,
            )
        )
        _synthetic_sor.register_venue_depth(
            VenueBookDepth(
                venue="Kraken",
                symbol="BTC-USD",
                bids=[(64994.0, 1.5), (64988.0, 4.0), (64975.0, 8.0)],
                asks=[(65001.0, 1.5), (65005.0, 4.0), (65020.0, 8.0)],
                maker_fee_bps=-1.2,
                taker_fee_bps=8.0,
                latency_ms=10.0,
            )
        )
    return _synthetic_sor


_margin_health_engine: DynamicMarginHealthEngine | None = None


def get_margin_health_engine() -> DynamicMarginHealthEngine:
    global _margin_health_engine
    if _margin_health_engine is None:
        _margin_health_engine = DynamicMarginHealthEngine()
    return _margin_health_engine


_flow_toxicity_engine: OrderFlowToxicityEngine | None = None


def get_flow_toxicity_engine(symbol: str = "BTC-USD") -> OrderFlowToxicityEngine:
    global _flow_toxicity_engine
    if _flow_toxicity_engine is None or _flow_toxicity_engine.symbol != symbol.upper():
        _flow_toxicity_engine = OrderFlowToxicityEngine(symbol=symbol)
        # Pre-seed BBO and synthetic flow
        _flow_toxicity_engine.update_bbo(bid=65000.0, ask=65005.0)
        t = 1_000_000_000
        for i in range(25):
            t += 100_000_000
            side = "BUY" if i % 2 == 0 else "SELL"
            p = 65005.0 if side == "BUY" else 65000.0
            _flow_toxicity_engine.process_trade(price=p, quantity=2.0, timestamp_ns=t, side=side)
    return _flow_toxicity_engine


_term_structure_calibrator: TermStructureCalibrator | None = None


def get_term_structure_calibrator() -> TermStructureCalibrator:
    global _term_structure_calibrator
    if _term_structure_calibrator is None:
        _term_structure_calibrator = TermStructureCalibrator()
    return _term_structure_calibrator


_lead_lag_engine: LeadLagNetworkEngine | None = None


def get_lead_lag_engine() -> LeadLagNetworkEngine:
    global _lead_lag_engine
    if _lead_lag_engine is None:
        _lead_lag_engine = LeadLagNetworkEngine(
            symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
            max_lag_steps=4,
            history_buffer_length=200,
        )
        # Pre-seed initial synchronous price trajectory
        np.random.seed(42)
        T = 40
        eps = np.random.randn(T) * 0.004
        ret_btc = eps.copy()
        ret_eth = np.zeros(T)
        ret_eth[2:] = ret_btc[:-2] * 0.8 + np.random.randn(T - 2) * 0.001
        ret_sol = np.zeros(T)
        ret_sol[3:] = ret_btc[:-3] * 0.7 + np.random.randn(T - 3) * 0.001

        p_btc = 65000.0 * np.exp(np.cumsum(ret_btc))
        p_eth = 3200.0 * np.exp(np.cumsum(ret_eth))
        p_sol = 150.0 * np.exp(np.cumsum(ret_sol))

        _lead_lag_engine.ingest_price_matrix({
            "BTC-USD": p_btc.tolist(),
            "ETH-USD": p_eth.tolist(),
            "SOL-USD": p_sol.tolist(),
        })
    return _lead_lag_engine


_mev_defense_gateway: MEVDefenseGateway | None = None


def get_mev_defense_gateway() -> MEVDefenseGateway:
    global _mev_defense_gateway
    if _mev_defense_gateway is None:
        _mev_defense_gateway = MEVDefenseGateway()
    return _mev_defense_gateway


_trade_vault: CryptographicTradeVault | None = None


def get_trade_vault() -> CryptographicTradeVault:
    global _trade_vault
    if _trade_vault is None:
        _trade_vault = CryptographicTradeVault(batch_id="GENESIS_VAULT_BATCH_01")
        _trade_vault.record_fill("TRD_GEN_01", "ORD_01", "BTC-USD", "BUY", 65000.0, 1.5, "COINBASE", 1.5)
        _trade_vault.record_fill("TRD_GEN_02", "ORD_02", "ETH-USD", "SELL", 3200.0, 10.0, "BINANCE", 2.0)
        _trade_vault.record_fill("TRD_GEN_03", "ORD_03", "SOL-USD", "BUY", 150.0, 50.0, "KRAKEN", 0.5)
    return _trade_vault


_cross_chain_monitor: CrossChainBridgeMonitor | None = None


def get_cross_chain_monitor() -> CrossChainBridgeMonitor:
    global _cross_chain_monitor
    if _cross_chain_monitor is None:
        _cross_chain_monitor = CrossChainBridgeMonitor()
        _cross_chain_monitor.update_token_price(SupportedChain.ETHEREUM, "ETH", 3220.0)
        _cross_chain_monitor.update_token_price(SupportedChain.ARBITRUM, "ETH", 3200.0)
        _cross_chain_monitor.update_token_price(SupportedChain.OPTIMISM, "ETH", 3205.0)
        _cross_chain_monitor.update_token_price(SupportedChain.ARBITRUM, "USDC", 1.0)
        _cross_chain_monitor.update_token_price(SupportedChain.BASE, "USDC", 1.002)
    return _cross_chain_monitor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and graceful shutdown lifecycle."""
    logger.info("Starting Trading Intelligence Platform API server...", env=settings.ENVIRONMENT)
    try:
        await init_db()
        await cache.initialize()
        from learning.ledger import PredictionLedgerService
        PredictionLedgerService.get_instance().start_background_resolver()
    except Exception as exc:
        logger.error("Initialization error during startup", error=str(exc))

    yield

    logger.info("Shutting down Trading Intelligence Platform API server...")
    try:
        from learning.ledger import PredictionLedgerService
        PredictionLedgerService.get_instance().stop_background_resolver()
    except Exception:
        pass
    global _pipeline
    if _pipeline and _pipeline._is_running:
        await _pipeline.stop()
    await cache.close()
    await close_db()


app = FastAPI(
    title="Trading Intelligence Platform",
    description="Professional modular quantitative trading and market intelligence backend.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Observability"])
async def health_endpoint() -> dict[str, Any]:
    """Holistic system health check across DB, cache, and provider streams."""
    pipeline = get_pipeline()
    pipeline_metrics = pipeline.get_metrics() if pipeline else None
    return await check_system_health(pipeline_metrics=pipeline_metrics)


@app.get("/metrics", tags=["Observability"])
async def metrics_endpoint() -> dict[str, Any]:
    """Operational latency profiles (p50, p95, p99) and event volume."""
    pipeline = get_pipeline()
    return pipeline.get_metrics()


class GatewayPingRequest(BaseModel):
    venue: str = "coinbase"
    rtt_ms: float = 24.5


class GatewayReconnectRequest(BaseModel):
    venue: str = "coinbase"


@app.get("/monitoring/gateways/status", response_model=GatewayHeartbeatReport, tags=["Observability"])
async def get_gateway_heartbeat_status_endpoint() -> GatewayHeartbeatReport:
    """Retrieve live multi-venue gateway connectivity status, latency percentiles, and failover posture."""
    mgr = get_gateway_heartbeat_manager()
    return mgr.evaluate_gateways()


@app.post("/monitoring/gateways/ping", tags=["Observability"])
async def record_gateway_ping_endpoint(request: GatewayPingRequest) -> dict[str, Any]:
    """Ingest gateway ping/pong latency measurement and update rolling latency percentiles."""
    mgr = get_gateway_heartbeat_manager()
    mgr.record_ping(venue=request.venue, rtt_ms=request.rtt_ms)
    gw = mgr.gateways.get(request.venue)
    return {
        "venue": request.venue,
        "status": "RECORDED",
        "current_rtt_ms": request.rtt_ms,
        "p95_latency_ms": gw.p95_latency_ms if gw else 0.0,
        "connection_state": gw.connection_state if gw else "UNKNOWN",
    }


@app.post("/monitoring/gateways/reconnect", tags=["Observability"])
async def trigger_gateway_reconnect_endpoint(request: GatewayReconnectRequest) -> dict[str, Any]:
    """Manually or programmatically trigger self-healing reconnection attempt for an exchange gateway."""
    mgr = get_gateway_heartbeat_manager()
    success = mgr.trigger_reconnect(request.venue)
    gw = mgr.gateways.get(request.venue)
    return {
        "venue": request.venue,
        "reconnect_scheduled": success,
        "connection_state": gw.connection_state if gw else "UNKNOWN",
        "consecutive_attempts": gw.consecutive_reconnect_attempts if gw else 0,
        "status_message": gw.status_message if gw else "",
    }



@app.post("/pipeline/start", tags=["Market Data Ingestion"])
async def start_pipeline() -> dict[str, str]:
    """Connect to live public crypto feed and begin candle aggregation."""
    pipeline = get_pipeline()
    if pipeline._is_running:
        return {"status": "already_running"}
    await pipeline.start()
    return {"status": "started", "provider": pipeline.provider.name}


@app.post("/pipeline/stop", tags=["Market Data Ingestion"])
async def stop_pipeline() -> dict[str, str]:
    """Gracefully terminate live ingestion and flush bars to database."""
    pipeline = get_pipeline()
    if not pipeline._is_running:
        return {"status": "already_stopped"}
    await pipeline.stop()
    return {"status": "stopped", "provider": pipeline.provider.name}


@app.get("/candles/{symbol:path}", response_model=list[Candle], tags=["Market Data"])
async def get_candles_endpoint(
    symbol: str,
    timeframe: str = Query(default="5m", description="Candle timeframe, e.g. 1m, 5m, 15m, 1h, 1d"),
    limit: int = Query(default=100, ge=1, le=1000),
    session: AsyncSession = Depends(get_db_session),
) -> list[Candle]:
    """Query chronological OHLCV candles for any market symbol (DB + live MarketStateService fallback)."""
    repo = MarketDataRepository(session)
    candles = await repo.get_candles(symbol=symbol.upper(), timeframe=timeframe, limit=limit)
    if candles:
        return candles

    from data.market_state import MarketStateService

    state_svc = MarketStateService.get_instance()
    state = await asyncio.to_thread(state_svc.ensure_instrument_state, symbol, timeframe)
    if not state:
        return []
    raw_list = state_svc.get_candles_list(state, timeframe)[-limit:]
    out: list[Candle] = []
    for c in raw_list:
        ts_val = c.get("timestamp")
        ts_dt = (
            datetime.fromisoformat(str(ts_val))
            if isinstance(ts_val, str)
            else (ts_val if isinstance(ts_val, datetime) else datetime.now(timezone.utc))
        )
        o_p = float(c.get("open", state.last_price))
        h_p = float(c.get("high", state.last_price))
        l_p = float(c.get("low", state.last_price))
        c_p = float(c.get("close", state.last_price))
        vol = float(c.get("volume", 0.0))
        out.append(
            Candle(
                symbol=state.canonical_symbol,
                timeframe=timeframe,
                timestamp=ts_dt,
                open=o_p,
                high=max(o_p, h_p, c_p),
                low=min(o_p, l_p, c_p),
                close=c_p,
                volume=max(0.0, vol),
            )
        )
    return out


@app.get("/regime/{symbol:path}", response_model=MarketRegimeState, tags=["Intelligence & Analysis"])
async def get_market_regime(
    symbol: str,
    timeframe: str = Query(default="5m"),
    session: AsyncSession = Depends(get_db_session),
) -> MarketRegimeState:
    """Analyze recent market history and classify the active 3D market regime."""
    repo = MarketDataRepository(session)
    candles = await repo.get_candles(symbol=symbol.upper(), timeframe=timeframe, limit=60)
    if candles:
        df = pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low, "close": c.close,
            "volume": c.volume, "quote_volume": c.quote_volume
        } for c in candles])
    else:
        from data.market_state import MarketStateService
        state_svc = MarketStateService.get_instance()
        state = await asyncio.to_thread(state_svc.ensure_instrument_state, symbol, timeframe)
        c_list = state_svc.get_candles_list(state, timeframe) if state else []
        if not c_list or len(c_list) < 15:
            return MarketRegimeState(regime_code="DATA_UNAVAILABLE")
        df = pd.DataFrame(c_list)

    detector = MarketRegimeDetector()
    classified = detector.classify_dataframe(df)
    latest_row = classified.iloc[-1]
    return detector.classify_bar(latest_row)


class SignalEvaluateRequest(BaseModel):
    symbol: str = "NSE:TCS"
    current_price: float = 0.0
    atr: float = 0.0
    timeframe: str = "5m"
    horizon: str = "1h"


@app.post("/signals/evaluate", response_model=FinalSignalDecision, tags=["Intelligence & Analysis"])
async def evaluate_signals_endpoint(request: SignalEvaluateRequest) -> FinalSignalDecision:
    """Evaluate full multi-source signal fusion (Features -> Regime -> Historical -> ML -> Technical -> Fusion)."""
    from learning.ledger import PredictionLedgerService
    from signals.fusion import run_live_intelligence_pipeline

    res = await asyncio.to_thread(
        run_live_intelligence_pipeline,
        symbol_or_query=request.symbol,
        timeframe=request.timeframe,
        horizon=request.horizon,
        override_price=request.current_price if request.current_price > 0 else None,
        override_atr=request.atr if request.atr > 0 else None,
    )
    decision: FinalSignalDecision = res["decision"]
    fc = res.get("forecast")
    st = res.get("market_state")
    inst = res.get("instrument")
    if fc is not None and res.get("status") == "OK":
        ledger = PredictionLedgerService.get_instance()
        pred_id = ledger.record_forecast(
            fc,
            model_id=ledger.champion_model_id,
            market=inst.market.value if inst else None,
            features_used=res.get("features"),
            historical_context_used={
                "pattern_consensus": {
                    k: v for k, v in (res.get("pattern_consensus") or {}).items() if k != "all_returns"
                },
                "developing_day": res.get("developing_day"),
            },
        )
        decision.evidence["prediction_id"] = pred_id
        decision.evidence["forecast"] = fc.model_dump(mode="json")
        if st:
            intraday_bars = (st.rolling_candles_5m or st.rolling_candles_1m)[-60:]
            decision.evidence["candles_intraday"] = intraday_bars

    return decision


@app.post("/risk/evaluate", response_model=RiskDecision, tags=["Risk Engine"])
async def evaluate_risk_endpoint(signal: FinalSignalDecision) -> RiskDecision:
    """Gatekeeper evaluation of a proposed trade setup against firm-wide risk limits."""
    risk_eng = get_risk_engine()
    return risk_eng.evaluate_trade(signal)


def _generate_synthetic_prices(symbols: list[str], bars: int = 100, periods: int | None = None) -> pd.DataFrame:
    """Generate realistic price path for risk modeling when live DB history is limited."""
    n_bars = periods if periods is not None else bars
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n_bars, freq="1D")
    data = {}
    base_prices = {"BTC-USD": 65000.0, "ETH-USD": 3500.0, "SOL-USD": 150.0}
    for sym in symbols:
        p0 = base_prices.get(sym.upper(), 100.0)
        ret = rng.normal(0.0005, 0.03, size=n_bars)
        data[sym.upper()] = p0 * np.exp(np.cumsum(ret))
    return pd.DataFrame(data, index=dates)



class VaRRequest(BaseModel):
    weights: dict[str, float] = {"BTC-USD": 0.6, "ETH-USD": 0.4}
    portfolio_value: float = 100000.0
    confidence_level: float = 0.95
    horizon_days: int = 1


@app.post("/risk/var", response_model=VaRMetrics, tags=["Risk Engine"])
async def calculate_var_endpoint(request: VaRRequest) -> VaRMetrics:
    """Compute Parametric, Historical, Monte Carlo VaR and Component Risk Attribution."""
    prices_df = _generate_synthetic_prices(list(request.weights.keys()))
    risk_model = PortfolioRiskModel()
    return risk_model.calculate_var(
        prices_df=prices_df,
        weights=request.weights,
        portfolio_value=request.portfolio_value,
        confidence_level=request.confidence_level,
        horizon_days=request.horizon_days,
    )


class StressTestRequest(BaseModel):
    portfolio_equity: float = 100000.0
    allocations_usd: dict[str, float] = {"BTC-USD": 60000.0, "ETH-USD": 40000.0}
    stop_losses: dict[str, float] | None = None
    current_prices: dict[str, float] | None = None
    scenario: str | None = None


@app.post("/risk/stress-test", tags=["Risk Engine"])
async def run_stress_test_endpoint(request: StressTestRequest) -> list[dict[str, Any]]:
    """Simulate macro crash scenarios, liquidity vacuums, and circuit breaker breaches."""
    tester = PortfolioStressTester()
    if request.scenario:
        scen_enum = getattr(StressScenarioType, request.scenario.upper(), StressScenarioType.COVID_LIQUIDITY_CRUNCH_2020)
        report = tester.run_scenario(
            scenario_type=scen_enum,
            portfolio_equity=request.portfolio_equity,
            allocations_usd=request.allocations_usd,
            stop_losses=request.stop_losses,
            current_prices=request.current_prices,
        )
        return [report.model_dump()]
    reports = tester.run_all_scenarios(
        portfolio_equity=request.portfolio_equity,
        allocations_usd=request.allocations_usd,
        stop_losses=request.stop_losses,
        current_prices=request.current_prices,
    )
    return [r.model_dump() for r in reports]


class FactorAttributionRequest(BaseModel):
    weights: dict[str, float] = {"BTC-USD": 0.6, "ETH-USD": 0.4}


@app.post("/risk/factors", response_model=PortfolioFactorAttribution, tags=["Risk Engine"])
async def calculate_factors_endpoint(request: FactorAttributionRequest) -> PortfolioFactorAttribution:
    """Decompose portfolio returns into systematic factors (Market, Momentum, Volatility, Liquidity)."""
    prices_df = _generate_synthetic_prices(list(request.weights.keys()))
    factor_model = SystematicFactorModel()
    return factor_model.attribute_portfolio(prices_df=prices_df, weights=request.weights)


class MarginEvaluationRequest(BaseModel):
    cash_balance: float = 100000.0
    collateral_assets: list[CollateralAsset] = []
    positions: list[MarginPosition] = []


@app.post("/risk/margin/evaluate", response_model=PortfolioMarginReport, tags=["Risk Engine"])
async def evaluate_portfolio_margin_endpoint(request: MarginEvaluationRequest) -> PortfolioMarginReport:
    """Evaluate institutional portfolio margin, multi-tier collateral haircuts, and cross-asset offsets."""
    engine = get_margin_engine()
    return engine.evaluate_margin(
        collateral_assets=request.collateral_assets,
        positions=request.positions,
        cash_balance=request.cash_balance,
    )


@app.get("/risk/collateral/venues/status", tags=["Risk Engine"])
async def get_collateral_venues_status_endpoint() -> dict[str, dict[str, Any]]:
    """Get real-time collateral margin utilization, haircut discounts, and health across all venues."""
    engine = get_collateral_engine()
    return engine.evaluate_all_venues()


class CollateralRebalancePlanRequest(BaseModel):
    venues: dict[str, VenueCollateralState] | None = None


@app.post("/risk/collateral/rebalance/plan", response_model=CollateralRebalancePlan, tags=["Risk Engine"])
async def generate_collateral_rebalance_plan_endpoint(
    request: CollateralRebalancePlanRequest = CollateralRebalancePlanRequest(),
) -> CollateralRebalancePlan:
    """Synthesize optimal collateral rehypothecation and minimum-cost liquidity transfer plan."""
    engine = get_collateral_engine()
    if request.venues:
        custom_engine = CollateralRehypothecationEngine(venues=request.venues)
        return custom_engine.generate_rebalance_plan()
    return engine.generate_rebalance_plan()


class CollateralExecutionRequest(BaseModel):
    plan: CollateralRebalancePlan
    dry_run: bool = False


@app.post("/risk/collateral/rebalance/execute", tags=["Risk Engine"])
async def execute_collateral_rebalance_endpoint(request: CollateralExecutionRequest) -> dict[str, Any]:
    """Execute or simulate inter-venue collateral rehypothecation transfers and state transitions."""
    engine = get_collateral_engine()
    executed_legs = engine.execute_transfer_plan(request.plan, dry_run=request.dry_run)
    post_venues = engine.evaluate_all_venues()
    return {
        "status": "COMPLETED",
        "dry_run": request.dry_run,
        "executed_transfers_count": len(executed_legs),
        "executed_legs": [leg.model_dump() for leg in executed_legs],
        "post_rebalance_venues": post_venues,
    }



class VolatilityTargetRequest(BaseModel):
    base_weights: dict[str, float] = {"BTC-USD": 0.6, "ETH-USD": 0.4}
    target_volatility_annualized: float = 0.15
    min_leverage: float = 0.10
    max_leverage: float = 1.50


@app.post("/portfolio/volatility-target", tags=["Risk Engine"])
async def calculate_volatility_target_endpoint(request: VolatilityTargetRequest) -> dict[str, Any]:
    """Calculate dynamic volatility targeting leverage scaling and target weights."""
    prices_df = _generate_synthetic_prices(list(request.base_weights.keys()))
    config = VolatilityTargetConfig(
        target_volatility_annualized=request.target_volatility_annualized,
        min_leverage=request.min_leverage,
        max_leverage=request.max_leverage,
    )
    engine = VolatilityTargetingEngine(config=config)
    res = engine.calculate_target_weights(prices_df=prices_df, base_weights=request.base_weights)
    return res.model_dump()


class RebalancePlanRequest(BaseModel):
    current_holdings: dict[str, float] = {"BTC-USD": 0.5, "ETH-USD": 5.0}
    current_cash: float = 30000.0
    current_prices: dict[str, float] = {"BTC-USD": 65000.0, "ETH-USD": 3500.0}
    target_weights: dict[str, float] = {"BTC-USD": 0.60, "ETH-USD": 0.30}
    deadband_drift_pct: float = 0.02


@app.post("/portfolio/rebalance/plan", response_model=RebalancePlan, tags=["Portfolio & Execution"])
async def generate_rebalance_plan_endpoint(request: RebalancePlanRequest) -> RebalancePlan:
    """Construct sequenced rebalancing plan prioritizing capital-freeing sells with Almgren-Chriss drag estimates."""
    rebalancer = PortfolioRebalancer(deadband_drift_pct=request.deadband_drift_pct)
    return rebalancer.generate_plan(
        current_holdings=request.current_holdings,
        current_cash=request.current_cash,
        current_prices=request.current_prices,
        target_weights=request.target_weights,
    )


class CostEstimateRequest(BaseModel):
    symbol: str = "BTC-USD"
    side: str = "BUY"
    quantity: float = 2.0
    current_price: float = 65000.0
    bid_ask_spread: float | None = None
    average_daily_volume: float = 5000.0
    daily_volatility: float = 0.03
    execution_horizon_hours: float = 0.25
    venue_fee_rate: float | None = None


@app.post("/execution/cost-estimate", response_model=ExecutionCostEstimate, tags=["Portfolio & Execution"])
async def estimate_execution_cost_endpoint(request: CostEstimateRequest) -> ExecutionCostEstimate:
    """Estimate Almgren-Chriss half-spread, temporary/permanent market impact, and shortfall drag."""
    model = AlmgrenChrissCostModel()
    return model.estimate_cost(
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        current_price=request.current_price,
        bid_ask_spread=request.bid_ask_spread,
        average_daily_volume=request.average_daily_volume,
        daily_volatility=request.daily_volatility,
        execution_horizon_hours=request.execution_horizon_hours,
        venue_fee_rate=request.venue_fee_rate,
    )


class AlgoScheduleRequest(BaseModel):
    symbol: str = "BTC-USD"
    side: str = "BUY"
    quantity: float = 2.0
    algo_type: str = "TWAP"
    duration_minutes: float = 30.0
    num_slices: int = 6
    arrival_price: float = 65000.0


@app.get("/execution/algo/profile/{symbol}", tags=["Portfolio & Execution"])
async def get_volume_profile_endpoint(symbol: str) -> dict[str, Any]:
    """Retrieve 24-hour intra-day volume distribution curve for an asset."""
    profiler = VolumeProfiler()
    profile = profiler.get_canonical_profile(symbol=symbol)
    return profile.model_dump()


@app.post("/execution/algo/schedule", response_model=AlgoExecutionPlan, tags=["Portfolio & Execution"])
async def schedule_algo_order_endpoint(request: AlgoScheduleRequest) -> AlgoExecutionPlan:
    """Generate randomized TWAP or volume-weighted VWAP child slice schedule."""
    engine = AlgoEngine()
    algo_type_upper = request.algo_type.upper()
    if algo_type_upper == "VWAP":
        return engine.create_vwap_plan(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            duration_minutes=request.duration_minutes,
            num_slices=request.num_slices,
            arrival_price=request.arrival_price,
        )
    return engine.create_twap_plan(
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        duration_minutes=request.duration_minutes,
        num_slices=request.num_slices,
        arrival_price=request.arrival_price,
    )


@app.post("/execution/algo/execute", response_model=AlgoExecutionReport, tags=["Portfolio & Execution"])
async def execute_algo_order_endpoint(plan: AlgoExecutionPlan) -> AlgoExecutionReport:
    """Execute algorithmic order schedule and calculate implementation shortfall."""
    from execution.simulated import SimulatedExecutionClient
    client = SimulatedExecutionClient()
    await client.connect()
    engine = AlgoEngine()
    return await engine.execute_algo_plan(plan=plan, client=client, fast_simulation=True)


class SmartHedgePlanRequest(BaseModel):
    portfolio_greeks: DerivativePortfolioReport
    spot_symbol: str = "BTC-USD"
    spot_price: float = 65000.0
    perp_symbol: str | None = "BTC-PERP"
    perp_price: float | None = 65020.0
    delta_tolerance: float = 0.50
    average_daily_volume: float = 10000.0
    daily_volatility: float = 0.04


@app.post("/execution/hedge/plan", response_model=SmartHedgePlan, tags=["Portfolio & Execution"])
async def generate_smart_hedge_plan_endpoint(request: SmartHedgePlanRequest) -> SmartHedgePlan:
    """Synthesize cost-optimized delta-neutral smart hedging plan with deadband filtering and Almgren-Chriss impact."""
    config = SmartHedgeConfig(delta_tolerance=request.delta_tolerance)
    engine = SmartHedgingEngine(config=config)
    return engine.generate_hedge_plan(
        portfolio_greeks=request.portfolio_greeks,
        spot_symbol=request.spot_symbol,
        spot_price=request.spot_price,
        perp_symbol=request.perp_symbol,
        perp_price=request.perp_price,
        average_daily_volume=request.average_daily_volume,
        daily_volatility=request.daily_volatility,
    )


@app.post("/execution/hedge/execute", tags=["Portfolio & Execution"])
async def execute_smart_hedge_plan_endpoint(plan: SmartHedgePlan) -> list[dict[str, Any]]:
    """Dispatch and execute smart hedge orders against simulated or live exchange execution venue."""
    from execution.simulated import SimulatedExecutionClient
    client = SimulatedExecutionClient()
    await client.connect()
    engine = SmartHedgingEngine()
    return await engine.execute_hedge_plan(plan=plan, client=client)


class CrossingSubmitRequest(BaseModel):
    symbol: str = "BTC-USD"
    side: str = "BUY"
    quantity: float = 1.0
    limit_price: float | None = None
    time_in_force: str = "IOC"
    strategy_id: str = "DESK_EXEC"
    current_bid: float = 65000.0
    current_ask: float = 65010.0
    route_residual_external: bool = False


@app.post("/execution/crossing/submit", response_model=CrossingResult, tags=["Portfolio & Execution"])
async def submit_crossing_order_endpoint(request: CrossingSubmitRequest) -> CrossingResult:
    """Submit order to internal crossing pool for mid-price execution and fee elimination."""
    engine = get_crossing_engine()
    tif = getattr(CrossingTimeInForce, request.time_in_force.upper(), CrossingTimeInForce.IOC)
    side = CrossingSide.BUY if request.side.upper() == "BUY" else CrossingSide.SELL
    order = InternalCrossingOrder(
        symbol=request.symbol,
        side=side,
        quantity=request.quantity,
        remaining_quantity=request.quantity,
        limit_price=request.limit_price,
        time_in_force=tif,
        strategy_id=request.strategy_id,
    )
    return engine.cross_order(
        order=order,
        current_bid=request.current_bid,
        current_ask=request.current_ask,
        route_residual_external=request.route_residual_external,
    )


@app.get("/execution/crossing/{symbol}/book", tags=["Portfolio & Execution"])
async def get_crossing_book_endpoint(symbol: str) -> dict[str, Any]:
    """Inspect active resting internal crossing liquidity."""
    engine = get_crossing_engine()
    return engine.get_resting_book(symbol)


@app.get("/execution/crossing/{symbol}/metrics", response_model=CrossingPoolMetrics, tags=["Portfolio & Execution"])
async def get_crossing_metrics_endpoint(symbol: str) -> CrossingPoolMetrics:
    """Retrieve crossing volume, resting liquidity, and cumulative spread/fee savings."""
    engine = get_crossing_engine()
    return engine.get_metrics(symbol)


@app.delete("/execution/crossing/{symbol}/order/{order_id}", tags=["Portfolio & Execution"])
async def cancel_crossing_order_endpoint(symbol: str, order_id: str) -> dict[str, Any]:
    """Cancel an active resting order in the internal crossing book."""
    engine = get_crossing_engine()
    cancelled = engine.cancel_order(symbol=symbol, order_id=order_id)
    return {"symbol": symbol, "order_id": order_id, "cancelled": cancelled}


class DMMQuoteRequest(BaseModel):
    symbol: str = "BTC-USD"
    best_bid: float = 65000.0
    best_ask: float = 65010.0
    bid_volume: float = 2.0
    ask_volume: float = 1.5
    volatility_annualized: float = 0.50
    order_flow_imbalance: float = 0.0


class DMMFillRecordRequest(BaseModel):
    symbol: str = "BTC-USD"
    side: str = "BUY"
    quantity: float = 0.5
    price: float = 65000.0


class DMMInventoryResetRequest(BaseModel):
    symbol: str = "BTC-USD"
    inventory: float = 0.0


@app.post("/execution/dmm/quotes", response_model=DMMQuoteSurface, tags=["Portfolio & Execution"])
async def calculate_dmm_quotes_endpoint(request: DMMQuoteRequest) -> DMMQuoteSurface:
    """Calculate Avellaneda-Stoikov quoting ladder with inventory skewing and OFI defense."""
    engine = get_dmm_engine()
    return engine.calculate_quotes(
        symbol=request.symbol,
        best_bid=request.best_bid,
        best_ask=request.best_ask,
        bid_volume=request.bid_volume,
        ask_volume=request.ask_volume,
        volatility_annualized=request.volatility_annualized,
        order_flow_imbalance=request.order_flow_imbalance,
    )


@app.post("/execution/dmm/fill", tags=["Portfolio & Execution"])
async def record_dmm_fill_endpoint(request: DMMFillRecordRequest) -> dict[str, Any]:
    """Record execution fill against DMM quotes, updating net inventory."""
    engine = get_dmm_engine()
    new_inv = engine.record_fill(
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        price=request.price,
    )
    return {"symbol": request.symbol, "inventory": new_inv, "status": "FILL_RECORDED"}


@app.get("/execution/dmm/{symbol}/inventory", tags=["Portfolio & Execution"])
async def get_dmm_inventory_endpoint(symbol: str) -> dict[str, Any]:
    """Retrieve current DMM inventory units and capacity utilization."""
    engine = get_dmm_engine()
    inv = engine.get_inventory(symbol)
    max_inv = engine.config.max_inventory_units
    return {
        "symbol": symbol,
        "inventory_units": inv,
        "max_inventory_units": max_inv,
        "inventory_utilization_pct": round((abs(inv) / max(1e-5, max_inv)) * 100.0, 2),
    }


@app.post("/execution/dmm/inventory/reset", tags=["Portfolio & Execution"])
async def reset_dmm_inventory_endpoint(request: DMMInventoryResetRequest) -> dict[str, Any]:
    """Override or reset DMM inventory position for an asset."""
    engine = get_dmm_engine()
    engine.set_inventory(request.symbol, request.inventory)
    return {"symbol": request.symbol, "inventory": request.inventory, "status": "INVENTORY_RESET"}


class AdaptiveMMQuoteRequest(BaseModel):
    symbol: str = "BTC-USD"
    best_bid: float = 65000.0
    best_ask: float = 65010.0
    volatility_annualized: float = 0.55
    drift_annualized: float = 0.0
    has_volatility_jump: bool = False
    time_to_terminal_seconds: float | None = None


class AdaptiveMMFillRequest(BaseModel):
    symbol: str = "BTC-USD"
    side: str = "BUY"
    quantity: float = 1.0
    price: float = 65000.0


@app.post("/execution/adaptive-mm/quotes", response_model=AdaptiveMMQuotingReport, tags=["Portfolio & Execution"])
async def calculate_adaptive_mm_quotes_endpoint(request: AdaptiveMMQuoteRequest) -> AdaptiveMMQuotingReport:
    """Calculate multi-tier adaptive Avellaneda-Stoikov / Guéant quotes with non-linear Poisson fill probabilities."""
    optimizer = get_adaptive_market_maker()
    return optimizer.generate_quote_surface(
        symbol=request.symbol,
        best_bid=request.best_bid,
        best_ask=request.best_ask,
        volatility_annualized=request.volatility_annualized,
        drift_annualized=request.drift_annualized,
        has_volatility_jump=request.has_volatility_jump,
        time_to_terminal_seconds=request.time_to_terminal_seconds,
    )


@app.post("/execution/adaptive-mm/fill", tags=["Portfolio & Execution"])
async def record_adaptive_mm_fill_endpoint(request: AdaptiveMMFillRequest) -> dict[str, Any]:
    """Record execution fill into adaptive market maker inventory tracking."""
    optimizer = get_adaptive_market_maker()
    new_inv = optimizer.record_fill(
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        price=request.price,
    )
    return {"symbol": request.symbol.upper(), "inventory": new_inv, "status": "FILL_RECORDED"}


@app.get("/execution/adaptive-mm/{symbol}/inventory", tags=["Portfolio & Execution"])
async def get_adaptive_mm_inventory_endpoint(symbol: str) -> dict[str, Any]:
    """Retrieve adaptive market maker net inventory position and capacity utilization."""
    optimizer = get_adaptive_market_maker()
    inv = optimizer.get_inventory(symbol)
    max_inv = optimizer.config.max_inventory_limit
    return {
        "symbol": symbol.upper(),
        "inventory": inv,
        "max_inventory_limit": max_inv,
        "utilization_pct": round((abs(inv) / max(1e-5, max_inv)) * 100.0, 2),
    }


class TCAPortfolioRequest(BaseModel):
    trades: list[TradeExecutionRecord] = []


@app.post("/execution/tca/decompose", response_model=ImplementationShortfallDecomposition, tags=["Portfolio & Execution"])
async def decompose_trade_shortfall_endpoint(trade: TradeExecutionRecord) -> ImplementationShortfallDecomposition:
    """Decompose order implementation shortfall into delay, spread, market impact, and opportunity cost."""
    engine = TransactionCostAnalysisEngine()
    return engine.decompose_trade(trade)


@app.post("/execution/tca/report", response_model=TCAPortfolioReport, tags=["Portfolio & Execution"])
async def generate_tca_portfolio_report_endpoint(request: TCAPortfolioRequest) -> TCAPortfolioReport:
    """Generate comprehensive post-trade Transaction Cost Analysis (TCA) report with venue scorecards."""
    engine = TransactionCostAnalysisEngine()
    return engine.generate_portfolio_tca_report(request.trades)


@app.post("/execution/tca/venues/scorecards", tags=["Portfolio & Execution"])
async def calculate_venue_scorecards_endpoint(request: TCAPortfolioRequest) -> dict[str, VenueExecutionScorecard]:
    """Calculate institutional broker/venue execution scorecards (EFQ, price improvement, latency)."""
    engine = TransactionCostAnalysisEngine()
    return engine.calculate_venue_scorecards(request.trades)


class CostSurfaceCalibrateRequest(BaseModel):
    symbol: str = "BTC-USD"
    venue: str = "coinbase"


@app.post("/execution/cost-surface/calibrate", response_model=CostSurfaceGrid, tags=["Portfolio & Execution"])
async def calibrate_cost_surface_endpoint(request: CostSurfaceCalibrateRequest) -> CostSurfaceGrid:
    """Empirically calibrate Almgren-Chriss impact parameters and generate 2D liquidity cost surface."""
    calibrator = get_cost_calibrator()
    return calibrator.generate_cost_surface(symbol=request.symbol, venue=request.venue)


@app.post("/execution/cost-surface/predict", response_model=SlippagePredictionResult, tags=["Portfolio & Execution"])
async def predict_slippage_endpoint(request: SlippagePredictionRequest) -> SlippagePredictionResult:
    """Forecast expected basis point and dollar slippage breakdown with asymmetric OFI adjustment."""
    calibrator = get_cost_calibrator()
    return calibrator.predict_slippage(request)


@app.get("/execution/cost-surface/parameters/{symbol}", response_model=CalibratedCostParameters, tags=["Portfolio & Execution"])
async def get_cost_parameters_endpoint(symbol: str, venue: str = "coinbase") -> CalibratedCostParameters:
    """Retrieve active calibrated market impact (eta/gamma) coefficients for a symbol and venue."""
    calibrator = get_cost_calibrator()
    return calibrator.get_parameters(symbol=symbol, venue=venue)



class AuditRecordRequest(BaseModel):
    record_type: str = "ORDER_SUBMISSION"
    action: str = "SUBMIT_ORDER"
    payload: dict[str, Any] = {"symbol": "BTC-USD", "quantity": 1.0, "side": "BUY"}
    actor: str = "API_USER"


@app.get("/compliance/audit/verify", tags=["Regulatory & Compliance"])
async def verify_audit_ledger() -> dict[str, Any]:
    """Verify cryptographic SHA-256 hash-chain integrity of the institutional audit ledger."""
    ledger = get_audit_ledger()
    is_valid, broken_seq = ledger.verify_integrity()
    return {
        "is_valid": is_valid,
        "broken_sequence": broken_seq,
        "total_audited_records": ledger.total_entries,
    }


@app.post("/compliance/audit/record", tags=["Regulatory & Compliance"])
async def record_audit_event_endpoint(request: AuditRecordRequest) -> dict[str, Any]:
    """Append a new tamper-evident event to the immutable compliance ledger."""
    ledger = get_audit_ledger()
    rec_type_enum = getattr(AuditRecordType, request.record_type.upper(), AuditRecordType.ORDER_SUBMISSION)
    entry = ledger.record_event(
        record_type=rec_type_enum,
        action=request.action,
        payload=request.payload,
        actor=request.actor,
    )
    return entry.model_dump()


@app.get("/compliance/export/mifid", tags=["Regulatory & Compliance"])
async def export_mifid_compliance_report() -> list[dict[str, Any]]:
    """Export regulatory transaction and audit extract conforming to MiFID II RTS 22/25."""
    ledger = get_audit_ledger()
    return ledger.export_mifid_report()


class SurveillanceScanRequest(BaseModel):
    orders: list[dict[str, Any]] = []


@app.post("/compliance/surveillance/scan", tags=["Regulatory & Compliance"])
async def scan_trade_surveillance_endpoint(request: SurveillanceScanRequest) -> list[dict[str, Any]]:
    """Scan order flow for prohibited market abuse patterns (wash trading, spoofing, quote stuffing)."""
    engine = TradeSurveillanceEngine()
    alerts = engine.scan_all(request.orders)
    return [a.model_dump() for a in alerts]


class PreFlightValidationRequest(BaseModel):
    order: FIXNewOrderSingle
    reference_mid_price: float = 65000.0
    best_bid: float | None = None
    best_ask: float | None = None


@app.get("/compliance/preflight/limits", response_model=PreFlightRiskLimits, tags=["Regulatory & Compliance"])
async def get_preflight_limits_endpoint() -> PreFlightRiskLimits:
    """Retrieve active SEC Rule 15c3-5 / MiFID II pre-trade risk thresholds and collar parameters."""
    validator = get_preflight_validator()
    return validator.limits


@app.post("/compliance/preflight/validate", response_model=PreFlightValidationResult, tags=["Regulatory & Compliance"])
async def validate_preflight_order_endpoint(request: PreFlightValidationRequest) -> PreFlightValidationResult:
    """Validate an outbound FIX order against pre-trade risk controls and price collar bands."""
    validator = get_preflight_validator()
    return validator.validate_order(
        order=request.order,
        reference_mid_price=request.reference_mid_price,
        best_bid=request.best_bid,
        best_ask=request.best_ask,
    )


@app.post("/compliance/preflight/reset", tags=["Regulatory & Compliance"])
async def reset_preflight_counters_endpoint() -> dict[str, str]:
    """Reset daily gross notional utilization counters and seen client order ID cache."""
    validator = get_preflight_validator()
    validator.reset_daily_limits()
    return {"status": "DAILY_LIMITS_RESET"}



@app.get("/portfolio/state", tags=["Portfolio & Execution"])
async def get_portfolio_state() -> dict[str, Any]:
    """Retrieve virtual cash, active positions, unrealized PnL, drawdown, and total equity."""
    engine = get_paper_engine()
    risk_eng = get_risk_engine()
    peak = max(risk_eng.peak_equity, engine.equity)
    dd_pct = ((peak - engine.equity) / peak * 100.0) if peak > 0 else 0.0
    return {
        "cash": round(engine.cash, 2),
        "equity": round(engine.equity, 2),
        "peak_equity": round(peak, 2),
        "max_drawdown_pct": round(max(0.0, dd_pct), 2),
        "max_drawdown_limit_pct": round(risk_eng.limits.max_drawdown_limit_pct * 100.0, 2),
        "active_positions_count": len(engine.positions),
        "positions": [p.model_dump() for p in engine.positions.values()],
    }


class SmartRouteRequest(BaseModel):
    symbol: str = "BTC-USD"
    side: str = "BUY"
    quantity: float = 0.5
    policy: str = "BEST_PRICE"
    execute: bool = False


@app.get("/orderbook/{symbol}/consolidated", tags=["Order Book & Microstructure"])
async def get_consolidated_orderbook(symbol: str) -> dict[str, Any]:
    """Retrieve consolidated synthetic BBO and multi-venue depth aggregation."""
    scanner = get_arbitrage_scanner()
    book = scanner.get_book(symbol)
    if not book:
        return {"error": f"No consolidated book registered for symbol {symbol}"}
    bbo = book.get_consolidated_bbo()
    depth = book.get_aggregated_depth(levels=5)
    return {
        "bbo": bbo.model_dump(),
        "depth": depth,
        "venues": book.venues,
    }


class OFIComputeRequest(BaseModel):
    bbo_updates: list[BBOState]
    rolling_window: int = 50


@app.post("/orderbook/ofi/compute", response_model=list[OFIMetric], tags=["Order Book & Microstructure"])
async def compute_ofi_endpoint(request: OFIComputeRequest) -> list[OFIMetric]:
    """Calculate instantaneous and normalized Order Flow Imbalance across top-of-book updates."""
    calc = OrderFlowImbalanceCalculator(rolling_window=request.rolling_window)
    return [calc.compute_step(bbo) for bbo in request.bbo_updates]


_heatmap_engines: dict[str, OrderBookHeatmapEngine] = {}


@app.get("/orderbook/{symbol}/heatmap", response_model=DepthHeatmapGrid, tags=["Order Book & Microstructure"])
async def get_orderbook_heatmap(symbol: str, levels: int = Query(default=20, ge=5, le=50)) -> DepthHeatmapGrid:
    """Retrieve spatial price-time volume intensity heatmap and Gini liquidity concentration."""
    global _heatmap_engines
    sym = symbol.upper()
    if sym not in _heatmap_engines:
        _heatmap_engines[sym] = OrderBookHeatmapEngine(symbol=sym)
        scanner = get_arbitrage_scanner()
        book = scanner.get_book(sym)
        if book:
            bids = book.get_aggregated_depth(levels=10).get("bids", [])
            asks = book.get_aggregated_depth(levels=10).get("asks", [])
            _heatmap_engines[sym].record_snapshot(bids=bids, asks=asks)

    engine = _heatmap_engines[sym]
    return engine.generate_heatmap(num_price_levels=levels)


@app.post("/execution/smart-route", tags=["Portfolio & Execution"])
async def execute_smart_route(request: SmartRouteRequest) -> dict[str, Any]:
    """Calculate and optionally execute multi-venue smart child routing plan."""
    scanner = get_arbitrage_scanner()
    book = scanner.get_book(request.symbol)
    if not book:
        return {"error": f"No book registered for {request.symbol}"}

    router = SmartOrderRouter(consolidated_book=book)
    policy_enum = getattr(RoutingPolicy, request.policy.upper(), RoutingPolicy.BEST_PRICE)
    plan = router.plan_order(
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        policy=policy_enum,
    )

    if not request.execute:
        return {"plan": plan.model_dump(), "executed": False}

    execution_result = await router.execute_plan(plan)
    return {"plan": plan.model_dump(), "executed": True, "execution_result": execution_result}


@app.get("/arbitrage/scan", tags=["Intelligence & Analysis"])
async def get_arbitrage_opportunities() -> dict[str, Any]:
    """Scan all registered consolidated order books for active cross-exchange arbitrage."""
    scanner = get_arbitrage_scanner()
    opportunities = scanner.scan_all()
    stats = scanner.get_stats()
    return {
        "active_opportunities": [o.model_dump() for o in opportunities],
        "stats": stats.model_dump(),
    }


class TriangularQuoteUpdateRequest(BaseModel):
    venue: str = "binance"
    symbol: str = "BTC-USD"
    bid: float = 65000.0
    ask: float = 65005.0
    bid_qty: float = 5.0
    ask_qty: float = 5.0


@app.post("/analysis/triangular/quotes/update", tags=["Intelligence & Analysis"])
async def update_triangular_quote_endpoint(request: TriangularQuoteUpdateRequest) -> dict[str, str]:
    """Ingest top-of-book market quote for triangular cross-currency graph traversal."""
    engine = get_triangular_engine()
    engine.update_quote(
        venue=request.venue,
        symbol=request.symbol,
        bid=request.bid,
        ask=request.ask,
        bid_qty=request.bid_qty,
        ask_qty=request.ask_qty,
    )
    return {"status": "QUOTE_UPDATED", "venue": request.venue, "symbol": request.symbol}


@app.get("/analysis/triangular/scan", response_model=TriangularScanReport, tags=["Intelligence & Analysis"])
async def scan_triangular_arbitrage_endpoint(anchor_asset: str = "USD") -> TriangularScanReport:
    """Scan currency graph for cyclic triangular and synthetic latency arbitrage opportunities."""
    engine = get_triangular_engine()
    return engine.scan_triangular_arbitrage(anchor_asset=anchor_asset)


@app.get("/analysis/triangular/graph", response_model=RateGraphData, tags=["Intelligence & Analysis"])
async def get_triangular_graph_endpoint() -> RateGraphData:
    """Retrieve active multi-currency directed rate graph with friction-adjusted edge weights."""
    engine = get_triangular_engine()
    return engine.get_rate_graph()


class CovarianceShrinkageRequest(BaseModel):
    returns_matrix: list[list[float]]
    symbols: list[str] | None = None
    target_type: str = "constant_correlation"
    custom_shrinkage_delta: float | None = None
    annualize: bool = True


class MinimumVarianceRequest(BaseModel):
    returns_matrix: list[list[float]]
    symbols: list[str] | None = None
    target_type: str = "constant_correlation"
    custom_shrinkage_delta: float | None = None
    long_only: bool = True
    annualize: bool = True


@app.post("/risk/covariance/shrink", tags=["Risk Management & Governance"])
async def estimate_covariance_shrinkage_endpoint(request: CovarianceShrinkageRequest) -> dict[str, Any]:
    """Estimate regularized covariance matrix via Ledoit-Wolf analytical shrinkage."""
    engine = get_covariance_shrinkage_engine()
    try:
        target = ShrinkageTargetType(request.target_type)
    except ValueError:
        target = ShrinkageTargetType.CONSTANT_CORRELATION

    result = engine.estimate_covariance(
        returns_matrix=request.returns_matrix,
        symbols=request.symbols,
        target_type=target,
        custom_shrinkage_delta=request.custom_shrinkage_delta,
        annualize=request.annualize,
    )
    return result.to_dict()


@app.post("/risk/covariance/minimum-variance", tags=["Risk Management & Governance"])
async def compute_minimum_variance_portfolio_endpoint(request: MinimumVarianceRequest) -> dict[str, Any]:
    """Derive global minimum-variance portfolio allocation using shrunk regularized covariance."""
    engine = get_covariance_shrinkage_engine()
    try:
        target = ShrinkageTargetType(request.target_type)
    except ValueError:
        target = ShrinkageTargetType.CONSTANT_CORRELATION

    result = engine.estimate_covariance(
        returns_matrix=request.returns_matrix,
        symbols=request.symbols,
        target_type=target,
        custom_shrinkage_delta=request.custom_shrinkage_delta,
        annualize=request.annualize,
    )
    mv = engine.compute_minimum_variance_portfolio(result, long_only=request.long_only)
    return {
        "shrinkage_summary": {
            "shrinkage_intensity": result.shrinkage_intensity,
            "condition_number_shrunk": result.condition_number_shrunk,
            "effective_rank": result.effective_rank,
            "target_type": result.target_type.value,
        },
        "minimum_variance": mv.to_dict(),
    }


class QuantumRebalanceRequest(BaseModel):
    symbols: list[str] = ["BTC", "ETH", "SOL", "AVAX", "LINK"]
    expected_returns: dict[str, float] = {
        "BTC": 0.25,
        "ETH": 0.30,
        "SOL": 0.45,
        "AVAX": 0.35,
        "LINK": 0.20,
    }
    covariance_matrix: list[list[float]] = [
        [0.09, 0.04, 0.03, 0.02, 0.02],
        [0.04, 0.12, 0.05, 0.03, 0.03],
        [0.03, 0.05, 0.20, 0.06, 0.04],
        [0.02, 0.03, 0.06, 0.18, 0.04],
        [0.02, 0.03, 0.04, 0.04, 0.10],
    ]
    current_weights: dict[str, float] | None = None
    max_cardinality: int = 3
    risk_aversion_lambda: float = 2.5
    transaction_cost_bps: float = 10.0
    solver_method: str = "SIMULATED_ANNEALING_QUBO"


@app.post("/risk/rebalance/quantum-optimize", response_model=CardinalityOptimizationReport, tags=["Risk Management & Governance"])
async def optimize_portfolio_cardinality_endpoint(request: QuantumRebalanceRequest) -> CardinalityOptimizationReport:
    """Execute quantum-inspired simulated annealing / QUBO for cardinality-constrained quadratic portfolio optimization."""
    rebalancer = get_quantum_rebalancer()
    rebalancer.config.max_cardinality_assets = request.max_cardinality
    rebalancer.config.risk_aversion_lambda = request.risk_aversion_lambda
    rebalancer.config.transaction_cost_bps = request.transaction_cost_bps

    try:
        method = OptimizationSolverMethod(request.solver_method)
    except ValueError:
        method = OptimizationSolverMethod.SIMULATED_ANNEALING_QUBO

    return rebalancer.optimize_portfolio(
        symbols=request.symbols,
        expected_returns=request.expected_returns,
        covariance_matrix=request.covariance_matrix,
        current_weights=request.current_weights,
        solver_method=method,
    )


class OrderBookSnapshotReplayRequest(BaseModel):
    symbol: str = "BTC-USD"
    timestamp_ns: int = 1_000_000_000
    bids: list[tuple[float, float]] = [[65000.0, 5.0], [64990.0, 10.0]]
    asks: list[tuple[float, float]] = [[65005.0, 4.0], [65015.0, 8.0]]


class OrderBookOrderSubmitRequest(BaseModel):
    symbol: str = "BTC-USD"
    order_id: str = "SIM_ORD_001"
    side: str = "BUY"
    order_type: str = "LIMIT"
    quantity: float = 1.0
    limit_price: float | None = 65000.0
    custom_latency_ns: int | None = 0


@app.post("/backtest/orderbook/snapshot", tags=["Backtesting & Simulation"])
async def replay_orderbook_snapshot_endpoint(request: OrderBookSnapshotReplayRequest) -> dict[str, Any]:
    """Ingest deterministic order book snapshot for microstructure replay backtesting."""
    sim = get_orderbook_simulator(symbol=request.symbol)
    sim.apply_snapshot(
        timestamp_ns=request.timestamp_ns,
        bids=request.bids,
        asks=request.asks,
    )
    return {
        "status": "APPLIED",
        "symbol": sim.symbol,
        "best_bid": sim.best_bid,
        "best_ask": sim.best_ask,
        "mid_price": sim.mid_price,
    }


@app.post("/backtest/orderbook/order", tags=["Backtesting & Simulation"])
async def submit_microstructure_sim_order_endpoint(request: OrderBookOrderSubmitRequest) -> dict[str, Any]:
    """Submit simulated order into tick-accurate queue position matching engine."""
    sim = get_orderbook_simulator(symbol=request.symbol)
    side = SimOrderSide.BUY if request.side.upper() == "BUY" else SimOrderSide.SELL
    order_type = SimOrderType.MARKET if request.order_type.upper() == "MARKET" else SimOrderType.LIMIT

    order = sim.submit_order(
        order_id=request.order_id,
        side=side,
        order_type=order_type,
        quantity=request.quantity,
        limit_price=request.limit_price,
        custom_latency_ns=request.custom_latency_ns,
    )
    return order.to_dict()


@app.get("/backtest/orderbook/report", tags=["Backtesting & Simulation"])
async def get_microstructure_simulation_report_endpoint(symbol: str = "BTC-USD") -> dict[str, Any]:
    """Generate high-fidelity microstructure simulation audit report with slippage breakdown."""
    sim = get_orderbook_simulator(symbol=symbol)
    report = sim.generate_report()
    return report.to_dict()


class MultiAgentLOBSimulateRequest(BaseModel):
    symbol: str = "BTC-USD"
    initial_mid_price: float = 65000.0
    steps: int = 50
    as_mm_count: int = 2
    grid_mm_count: int = 2
    toxic_taker_count: int = 1
    noise_taker_count: int = 1
    seed: int | None = 42


@app.post("/backtest/sandbox/multi-agent/simulate", response_model=MultiAgentSimulationReport, tags=["Backtesting & Simulation"])
async def run_multi_agent_lob_simulation_endpoint(request: MultiAgentLOBSimulateRequest) -> MultiAgentSimulationReport:
    """Execute multi-agent competitive equilibrium limit order book simulation with toxic flow and PnL attribution."""
    sandbox = MultiAgentLOBSandbox(
        symbol=request.symbol,
        initial_mid_price=request.initial_mid_price,
        seed=request.seed,
    )

    # Register configured agent cohorts
    for i in range(request.as_mm_count):
        sandbox.register_agent(f"MM_AS_{i+1}", AgentStrategyType.AVELLANEDA_STOIKOV_MM, initial_cash=100000.0, latency_ms=2.0 + i)

    for i in range(request.grid_mm_count):
        sandbox.register_agent(f"MM_GRID_{i+1}", AgentStrategyType.SYMMETRIC_GRID_MM, initial_cash=100000.0, latency_ms=4.0 + i)

    for i in range(request.toxic_taker_count):
        sandbox.register_agent(f"TAKER_TOXIC_{i+1}", AgentStrategyType.ADVERSARIAL_INFORMED_TAKER, initial_cash=100000.0, latency_ms=1.0)

    for i in range(request.noise_taker_count):
        sandbox.register_agent(f"TAKER_NOISE_{i+1}", AgentStrategyType.NOISE_TRADER_TAKER, initial_cash=50000.0, latency_ms=10.0 + i)

    return sandbox.run_simulation(steps=request.steps)


class VenueDepthUpdateRequest(BaseModel):
    venue: str
    symbol: str = "BTC-USD"
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    maker_fee_bps: float = -1.0
    taker_fee_bps: float = 6.0
    latency_ms: float = 5.0


class LiquiditySplitRequest(BaseModel):
    symbol: str = "BTC-USD"
    side: str = "BUY"
    parent_quantity: float = 5.0
    objective: str = "min_total_net_cost"
    max_venues: int = 5


@app.post("/execution/sor/venues/update", tags=["Execution & Algo Routing"])
async def update_venue_depth_endpoint(request: VenueDepthUpdateRequest) -> dict[str, str]:
    """Register or update venue order book depth profile for synthetic SOR optimization."""
    sor = get_synthetic_sor()
    depth = VenueBookDepth(
        venue=request.venue,
        symbol=request.symbol,
        bids=request.bids,
        asks=request.asks,
        maker_fee_bps=request.maker_fee_bps,
        taker_fee_bps=request.taker_fee_bps,
        latency_ms=request.latency_ms,
    )
    sor.register_venue_depth(depth)
    return {"status": "UPDATED", "venue": request.venue, "symbol": request.symbol}


@app.post("/execution/sor/split", tags=["Execution & Algo Routing"])
async def plan_liquidity_split_endpoint(request: LiquiditySplitRequest) -> dict[str, Any]:
    """Compute optimal multi-venue convex child allocation plan minimizing net cost and slippage."""
    sor = get_synthetic_sor()
    try:
        obj = SOROptimizationObjective(request.objective)
    except ValueError:
        obj = SOROptimizationObjective.MIN_TOTAL_NET_COST

    plan = sor.plan_liquidity_split(
        symbol=request.symbol,
        side=request.side,
        parent_quantity=request.parent_quantity,
        objective=obj,
        max_venues=request.max_venues,
    )
    return plan.to_dict()


class CollateralAssetInput(BaseModel):
    asset: str
    quantity: float
    base_mark_price: float
    base_haircut: float = 0.05
    volatility_haircut_sensitivity: float = 1.2
    concentration_haircut_penalty: float = 0.5


class PositionMarginInput(BaseModel):
    symbol: str
    side: str
    quantity: float
    mark_price: float
    entry_price: float | None = None
    mm_rate: float = 0.05


class SolvencyEvaluationRequest(BaseModel):
    collateral_assets: list[CollateralAssetInput]
    positions: list[PositionMarginInput]
    asset_volatilities: dict[str, float] | None = None
    market_stress_multiplier: float = 1.0


class SolvencyStressRequest(BaseModel):
    collateral_assets: list[CollateralAssetInput]
    positions: list[PositionMarginInput]
    scenario_name: str = "Flash Crash 30%"
    price_shock_pct: float = -0.30
    volatility_shock_multiplier: float = 2.5


@app.post("/risk/margin/solvency/evaluate", tags=["Risk Management & Governance"])
async def evaluate_margin_solvency_endpoint(request: SolvencyEvaluationRequest) -> dict[str, Any]:
    """Evaluate continuous dynamic margin health, haircut decay, and liquidation ladder."""
    engine = get_margin_health_engine()
    collateral = [
        DynamicCollateralAsset(
            asset=c.asset,
            quantity=c.quantity,
            base_mark_price=c.base_mark_price,
            base_haircut=c.base_haircut,
            volatility_haircut_sensitivity=c.volatility_haircut_sensitivity,
            concentration_haircut_penalty=c.concentration_haircut_penalty,
        )
        for c in request.collateral_assets
    ]
    positions = [
        {
            "symbol": p.symbol,
            "side": p.side,
            "quantity": p.quantity,
            "mark_price": p.mark_price,
            "entry_price": p.entry_price or p.mark_price,
            "mm_rate": p.mm_rate,
        }
        for p in request.positions
    ]

    plan, state = engine.evaluate_solvency_and_cascade(
        collateral_assets=collateral,
        positions=positions,
        asset_volatilities=request.asset_volatilities,
        market_stress_multiplier=request.market_stress_multiplier,
    )
    return {
        "solvency_state": state.value,
        "liquidation_plan": plan.to_dict(),
    }


@app.post("/risk/margin/solvency/stress", tags=["Risk Management & Governance"])
async def simulate_margin_solvency_stress_endpoint(request: SolvencyStressRequest) -> dict[str, Any]:
    """Simulate extreme price drop and volatility shock on portfolio margin solvency."""
    engine = get_margin_health_engine()
    collateral = [
        DynamicCollateralAsset(
            asset=c.asset,
            quantity=c.quantity,
            base_mark_price=c.base_mark_price,
            base_haircut=c.base_haircut,
            volatility_haircut_sensitivity=c.volatility_haircut_sensitivity,
            concentration_haircut_penalty=c.concentration_haircut_penalty,
        )
        for c in request.collateral_assets
    ]
    positions = [
        {
            "symbol": p.symbol,
            "side": p.side,
            "quantity": p.quantity,
            "mark_price": p.mark_price,
            "entry_price": p.entry_price or p.mark_price,
            "mm_rate": p.mm_rate,
        }
        for p in request.positions
    ]

    result = engine.simulate_solvency_stress(
        collateral_assets=collateral,
        positions=positions,
        scenario_name=request.scenario_name,
        price_shock_pct=request.price_shock_pct,
        volatility_shock_multiplier=request.volatility_shock_multiplier,
    )
    return result.to_dict()


class TradeIngestionRequest(BaseModel):
    symbol: str = "BTC-USD"
    price: float = 65000.0
    quantity: float = 1.0
    timestamp_ns: int | None = None
    side: str | None = None


@app.post("/analysis/toxicity/trade", tags=["Intelligence & Analysis"])
async def ingest_trade_for_toxicity_endpoint(request: TradeIngestionRequest) -> dict[str, Any]:
    """Ingest tick-by-tick trade into VPIN volume clock and Roll spread estimator."""
    eng = get_flow_toxicity_engine(symbol=request.symbol)
    ts = request.timestamp_ns or int(datetime.now(timezone.utc).timestamp() * 1e9)
    completed_bucket = eng.process_trade(
        price=request.price,
        quantity=request.quantity,
        timestamp_ns=ts,
        side=request.side,
    )
    return {
        "status": "PROCESSED",
        "symbol": eng.symbol,
        "bucket_completed": completed_bucket is not None,
        "completed_bucket": completed_bucket.to_dict() if completed_bucket else None,
        "current_vpin": eng.compute_vpin(),
    }


@app.get("/analysis/toxicity/snapshot", response_model=ToxicityMetricsSnapshot, tags=["Intelligence & Analysis"])
async def get_flow_toxicity_snapshot_endpoint(symbol: str = "BTC-USD") -> ToxicityMetricsSnapshot:
    """Retrieve real-time VPIN toxicity, Roll effective spread, and Kyle's lambda metrics."""
    eng = get_flow_toxicity_engine(symbol=symbol)
    return eng.get_metrics_snapshot()


class YieldTenorInput(BaseModel):
    maturity_years: float
    yield_rate: float
    weight: float = 1.0


class YieldCurveFitRequest(BaseModel):
    underlying: str = "BTC"
    tenor_points: list[YieldTenorInput] = [
        YieldTenorInput(maturity_years=1.0 / 12.0, yield_rate=0.040),
        YieldTenorInput(maturity_years=3.0 / 12.0, yield_rate=0.048),
        YieldTenorInput(maturity_years=6.0 / 12.0, yield_rate=0.055),
        YieldTenorInput(maturity_years=1.0, yield_rate=0.065),
        YieldTenorInput(maturity_years=2.0, yield_rate=0.075),
        YieldTenorInput(maturity_years=5.0, yield_rate=0.090),
    ]
    model_type: str = "NELSON_SIEGEL_SVENSSON"


class SpotRateQueryRequest(BaseModel):
    parameters: dict[str, Any]
    maturity_years: float


@app.post("/analysis/yield-curve/calibrate", tags=["Intelligence & Analysis"])
async def calibrate_yield_curve_endpoint(request: YieldCurveFitRequest) -> dict[str, Any]:
    """Fit Nelson-Siegel(-Svensson) parametric yield curve and evaluate roll-down/carry attribution."""
    calibrator = get_term_structure_calibrator()
    try:
        model = CurveModelType(request.model_type)
    except ValueError:
        model = CurveModelType.NELSON_SIEGEL_SVENSSON

    points = [
        YieldTenorPoint(
            maturity_years=p.maturity_years,
            yield_rate=p.yield_rate,
            weight=p.weight,
        )
        for p in request.tenor_points
    ]

    params = calibrator.fit_curve(points, model_type=model)
    report = calibrator.generate_full_curve_report(underlying=request.underlying, params=params)
    return report.to_dict()


@app.post("/analysis/yield-curve/spot-rate", tags=["Intelligence & Analysis"])
async def evaluate_curve_spot_rate_endpoint(request: SpotRateQueryRequest) -> dict[str, Any]:
    """Evaluate calibrated spot rate, forward rate, and discount factor at specific maturity."""
    calibrator = get_term_structure_calibrator()
    p_dict = request.parameters
    model_type = CurveModelType(p_dict.get("model_type", "NELSON_SIEGEL_SVENSSON"))
    params = CalibratedCurveParameters(
        model_type=model_type,
        beta_0=float(p_dict["beta_0"]),
        beta_1=float(p_dict["beta_1"]),
        beta_2=float(p_dict["beta_2"]),
        beta_3=float(p_dict.get("beta_3", 0.0)),
        tau_1=float(p_dict["tau_1"]),
        tau_2=float(p_dict.get("tau_2", 0.0)),
        rmse_bps=float(p_dict.get("rmse_bps", 0.0)),
        max_error_bps=float(p_dict.get("max_error_bps", 0.0)),
    )

    spot = calibrator.evaluate_spot_rate(params, request.maturity_years)
    fwd = calibrator.evaluate_instantaneous_forward(params, request.maturity_years)
    df = calibrator.evaluate_discount_factor(params, request.maturity_years)

    return {
        "maturity_years": request.maturity_years,
        "spot_rate": round(spot, 6),
        "spot_rate_bps": round(spot * 10000.0, 2),
        "instantaneous_forward_rate": round(fwd, 6),
        "discount_factor": round(df, 6),
    }


class MEVAwareAssessmentRequest(BaseModel):
    order_id: str = "ORD_ONCHAIN_001"
    symbol: str = "ETH-USDC"
    notional_usd: float = 50000.0
    pool_liquidity_usd: float = 2000000.0
    requested_slippage_bps: float = 50.0
    current_gas_price_gwei: float | None = None


class FlashbotsBundleRequest(BaseModel):
    raw_signed_txs: list[str]
    current_block_number: int = 19000000
    block_lead: int = 1


@app.post("/compliance/mev/assess", response_model=MEVDefenseAssessment, tags=["Compliance & Surveillance"])
async def assess_mev_risk_endpoint(request: MEVAwareAssessmentRequest) -> MEVDefenseAssessment:
    """Assess on-chain order front-running and sandwich vulnerability, recommending private builder relay."""
    gateway = get_mev_defense_gateway()
    return gateway.assess_order_mev_risk(
        order_id=request.order_id,
        symbol=request.symbol,
        notional_usd=request.notional_usd,
        pool_liquidity_usd=request.pool_liquidity_usd,
        requested_slippage_bps=request.requested_slippage_bps,
        current_gas_price_gwei=request.current_gas_price_gwei,
    )


@app.post("/compliance/mev/bundle", tags=["Compliance & Surveillance"])
async def build_flashbots_bundle_endpoint(request: FlashbotsBundleRequest) -> dict[str, Any]:
    """Assemble atomic private bundle for Flashbots Protect / MEV-Share direct block builder relay."""
    gateway = get_mev_defense_gateway()
    bundle = gateway.build_flashbots_bundle(
        raw_signed_txs=request.raw_signed_txs,
        current_block_number=request.current_block_number,
        block_lead=request.block_lead,
    )
    return bundle.to_dict()


class RecordFillVaultRequest(BaseModel):
    trade_id: str = "TRD_001"
    order_id: str = "ORD_001"
    symbol: str = "BTC-USD"
    side: str = "BUY"
    price: float = 65000.0
    quantity: float = 1.0
    venue: str = "BINANCE"
    fee_usd: float = 1.0


class SolvencyCommitmentRequest(BaseModel):
    epoch_id: str = "EPOCH_CURRENT"
    asset_balances: dict[str, float] = {"USD": 500000.0, "BTC": 15.0, "ETH": 50.0}
    liability_balances: dict[str, float] = {"USD": 300000.0, "BTC": 8.0, "ETH": 20.0}
    asset_prices: dict[str, float] | None = None


@app.post("/compliance/vault/fills/record", response_model=TradeFillReceipt, tags=["Compliance & Surveillance"])
async def record_vault_fill_endpoint(request: RecordFillVaultRequest) -> TradeFillReceipt:
    """Record execution fill into cryptographic Merkle vault and recompute root commitment."""
    vault = get_trade_vault()
    return vault.record_fill(
        trade_id=request.trade_id,
        order_id=request.order_id,
        symbol=request.symbol,
        side=request.side,
        price=request.price,
        quantity=request.quantity,
        venue=request.venue,
        fee_usd=request.fee_usd,
    )


@app.get("/compliance/vault/proof/inclusion/{trade_id}", response_model=MerkleAuditProof, tags=["Compliance & Surveillance"])
async def get_vault_inclusion_proof_endpoint(trade_id: str) -> MerkleAuditProof:
    """Generate cryptographic inclusion Merkle proof verifying trade presence in vault root."""
    vault = get_trade_vault()
    proof = vault.generate_inclusion_proof(trade_id)
    if not proof:
        raise HTTPException(status_code=404, detail=f"Trade fill {trade_id} not found in vault")
    return proof


@app.post("/compliance/vault/proof/verify", tags=["Compliance & Surveillance"])
async def verify_vault_proof_endpoint(proof: MerkleAuditProof) -> dict[str, Any]:
    """Verify validity of a cryptographic Merkle audit inclusion proof."""
    valid = CryptographicTradeVault.verify_proof(proof)
    return {"is_valid": valid, "target_hash": proof.target_hash, "root_hash": proof.root_hash}


@app.get("/compliance/vault/settlement/attestation", response_model=BatchSettlementAttestation, tags=["Compliance & Surveillance"])
async def get_vault_settlement_attestation_endpoint() -> BatchSettlementAttestation:
    """Generate verifiable batch settlement attestation anchored to vault Merkle root."""
    vault = get_trade_vault()
    return vault.generate_settlement_attestation()


@app.post("/compliance/vault/solvency/commitment", response_model=SolvencyCommitmentProof, tags=["Compliance & Surveillance"])
async def verify_solvency_commitment_endpoint(request: SolvencyCommitmentRequest) -> SolvencyCommitmentProof:
    """Generate cryptographic zero-knowledge solvency commitment proving Assets >= Liabilities."""
    return CryptographicTradeVault.verify_solvency_commitment(
        epoch_id=request.epoch_id,
        asset_balances=request.asset_balances,
        liability_balances=request.liability_balances,
        asset_prices=request.asset_prices,
    )


class CrossChainHeightUpdateRequest(BaseModel):
    chain: str = "ETHEREUM"
    current_block: int = 19500050
    finalized_block: int = 19500010


class CrossChainRouteQuoteRequest(BaseModel):
    source_chain: str = "ARBITRUM"
    destination_chain: str = "ETHEREUM"
    token_symbol: str = "ETH"
    bridge_protocol: str = "ACROSS"
    trade_amount: float = 10.0


@app.get("/compliance/cross-chain/report", response_model=CrossChainMonitorReport, tags=["Compliance & Surveillance"])
async def get_cross_chain_monitor_report_endpoint() -> CrossChainMonitorReport:
    """Retrieve multi-chain finality status, reorg safety buffers, and bridge arbitrage opportunities."""
    monitor = get_cross_chain_monitor()
    return monitor.generate_monitor_report()


@app.post("/compliance/cross-chain/height/update", response_model=ChainFinalityProfile, tags=["Compliance & Surveillance"])
async def update_cross_chain_height_endpoint(request: CrossChainHeightUpdateRequest) -> ChainFinalityProfile:
    """Update block height telemetry for a chain and evaluate reorg safety."""
    monitor = get_cross_chain_monitor()
    chain_enum = SupportedChain(request.chain.upper())
    return monitor.update_chain_height(
        chain=chain_enum,
        current_block=request.current_block,
        finalized_block=request.finalized_block,
    )


@app.post("/compliance/cross-chain/quote", response_model=CrossChainRouteQuote, tags=["Compliance & Surveillance"])
async def evaluate_cross_chain_route_endpoint(request: CrossChainRouteQuoteRequest) -> CrossChainRouteQuote:
    """Scan cross-chain route pricing discrepancy, fees, and latency to evaluate net arbitrage yield."""
    monitor = get_cross_chain_monitor()
    src = SupportedChain(request.source_chain.upper())
    dst = SupportedChain(request.destination_chain.upper())
    proto = BridgeProtocol(request.bridge_protocol.upper())
    return monitor.evaluate_route_quote(
        source_chain=src,
        destination_chain=dst,
        token_symbol=request.token_symbol,
        bridge_protocol=proto,
        trade_amount=request.trade_amount,
    )


class LeadLagTickIngestRequest(BaseModel):
    symbol: str = "BTC-USD"
    price: float = 65000.0


class LeadLagMatrixIngestRequest(BaseModel):
    prices: dict[str, list[float]]


@app.post("/analysis/lead-lag/ticks/ingest", tags=["Intelligence & Analysis"])
async def ingest_lead_lag_tick_endpoint(request: LeadLagTickIngestRequest) -> dict[str, Any]:
    """Ingest synchronous price tick update into rolling lead-lag return buffers."""
    engine = get_lead_lag_engine()
    engine.update_price_tick(symbol=request.symbol, price=request.price)
    return {"status": "INGESTED", "symbol": request.symbol.upper(), "price": request.price}


@app.post("/analysis/lead-lag/matrix/ingest", tags=["Intelligence & Analysis"])
async def ingest_lead_lag_matrix_endpoint(request: LeadLagMatrixIngestRequest) -> dict[str, Any]:
    """Batch ingest synchronous price matrix trajectory across multiple assets."""
    engine = get_lead_lag_engine()
    engine.ingest_price_matrix(request.prices)
    return {
        "status": "BATCH_INGESTED",
        "symbols": [s.upper() for s in request.prices.keys()],
        "ticks_per_symbol": {s: len(p) for s, p in request.prices.items()},
    }


@app.get("/analysis/lead-lag/network", response_model=LeadLagNetworkReport, tags=["Intelligence & Analysis"])
async def get_lead_lag_network_report_endpoint() -> LeadLagNetworkReport:
    """Calibrate cross-asset lead-lag graph topology, Perron-Frobenius centrality, and predictive signals."""
    engine = get_lead_lag_engine()
    return engine.calibrate_lead_lag_network()


class SpotQuoteUpdateRequest(BaseModel):
    underlying: str = "BTC"
    venue: str = "coinbase"
    symbol: str = "BTC-USD"
    bid: float = 65000.0
    ask: float = 65010.0


class PerpRateUpdateRequest(BaseModel):
    underlying: str = "BTC"
    venue: str = "binance"
    symbol: str = "BTCUSDT"
    rate_per_interval: float = 0.0003
    mark_price: float = 65180.0
    index_price: float = 65005.0
    interval_hours: int = 8


class BasisOpenPositionRequest(BaseModel):
    opportunity: BasisOpportunity
    quantity: float = 1.0


@app.get("/arbitrage/basis/scan", response_model=list[BasisOpportunity], tags=["Intelligence & Analysis"])
async def scan_basis_arbitrage_endpoint(
    min_net_apr_pct: float = Query(default=4.0, ge=0.0),
    holding_days: float = Query(default=30.0, ge=1.0),
) -> list[BasisOpportunity]:
    """Scan cross-venue perpetual funding rates and spot quotes for delta-neutral basis carry trades."""
    scanner = get_basis_scanner()
    return scanner.scan_opportunities(min_net_apr_pct=min_net_apr_pct, target_holding_days=holding_days)


@app.post("/arbitrage/basis/spot/update", tags=["Intelligence & Analysis"])
async def update_spot_quote_endpoint(request: SpotQuoteUpdateRequest) -> dict[str, str]:
    """Update top-of-book spot quote for funding basis evaluation."""
    scanner = get_basis_scanner()
    scanner.register_spot_quote(
        underlying=request.underlying,
        venue=request.venue,
        symbol=request.symbol,
        bid=request.bid,
        ask=request.ask,
    )
    return {"status": "SPOT_QUOTE_UPDATED", "venue": request.venue, "symbol": request.symbol}


@app.post("/arbitrage/basis/perp/update", tags=["Intelligence & Analysis"])
async def update_perp_rate_endpoint(request: PerpRateUpdateRequest) -> dict[str, str]:
    """Update perpetual funding rate telemetry and mark price."""
    scanner = get_basis_scanner()
    scanner.register_perp_rate(
        underlying=request.underlying,
        venue=request.venue,
        symbol=request.symbol,
        rate_per_interval=request.rate_per_interval,
        mark_price=request.mark_price,
        index_price=request.index_price,
        interval_hours=request.interval_hours,
    )
    return {"status": "PERP_RATE_UPDATED", "venue": request.venue, "symbol": request.symbol}


@app.post("/arbitrage/basis/position/open", response_model=BasisPosition, tags=["Intelligence & Analysis"])
async def open_basis_position_endpoint(request: BasisOpenPositionRequest) -> BasisPosition:
    """Open and track active delta-neutral cash-and-carry basis trade."""
    tracker = get_basis_tracker()
    return tracker.open_position(opportunity=request.opportunity, quantity=request.quantity)


@app.get("/arbitrage/basis/positions", response_model=list[BasisPosition], tags=["Intelligence & Analysis"])
async def list_basis_positions_endpoint() -> list[BasisPosition]:
    """List currently tracked delta-neutral carry trades and accumulated funding yield."""
    tracker = get_basis_tracker()
    return tracker.list_active_positions()


class AlphaPipelineRequest(BaseModel):
    symbols: list[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"]
    periods: int = 50
    forward_horizon: int = 1


class FactorNeutralizeRequest(BaseModel):
    factor_scores: dict[str, float] = {"BTC-USD": 1.2, "ETH-USD": -0.5, "SOL-USD": 0.8, "AVAX-USD": -1.1}
    risk_exposures: dict[str, dict[str, float]] = {
        "BTC-USD": {"MarketBeta": 1.0, "LogPrice": 11.0},
        "ETH-USD": {"MarketBeta": 1.1, "LogPrice": 8.1},
        "SOL-USD": {"MarketBeta": 1.4, "LogPrice": 5.0},
        "AVAX-USD": {"MarketBeta": 1.3, "LogPrice": 3.4},
    }


@app.post("/analysis/alpha/pipeline/run", response_model=AlphaResearchReport, tags=["Intelligence & Analysis"])
async def run_alpha_pipeline_endpoint(request: AlphaPipelineRequest = AlphaPipelineRequest()) -> AlphaResearchReport:
    """Execute quantitative multi-factor construction, risk neutralization, and IC evaluation pipeline."""
    prices_df = _generate_synthetic_prices(request.symbols, periods=max(30, request.periods))
    pipeline = AlphaResearchPipeline()
    return pipeline.run_alpha_pipeline(prices_df=prices_df, forward_horizon_periods=request.forward_horizon)


@app.post("/analysis/alpha/factor/neutralize", tags=["Intelligence & Analysis"])
async def neutralize_factor_endpoint(request: FactorNeutralizeRequest) -> dict[str, Any]:
    """Orthogonalize alpha factor scores against market beta and risk factors via OLS residualization."""
    pipeline = AlphaResearchPipeline()
    s = pd.Series(request.factor_scores)
    exp_df = pd.DataFrame.from_dict(request.risk_exposures, orient="index")
    neut = pipeline.neutralize_factor(s, exp_df)
    return {
        "raw_scores": request.factor_scores,
        "neutralized_scores": {k: round(float(v), 4) for k, v in neut.items()},
    }


class EnsembleAllocateRequest(BaseModel):
    total_capital_usd: float = 100000.0
    current_regime: str | None = None
    algorithm: BanditAlgorithmType | None = None


class EnsembleRewardRequest(BaseModel):
    strategy_id: str
    realized_pnl_usd: float
    notional_traded_usd: float = 10000.0
    sharpe_equivalent: float | None = None


@app.get("/analysis/ensemble/status", tags=["Intelligence & Analysis"])
async def get_ensemble_status_endpoint() -> dict[str, Any]:
    """Get active multi-armed bandit strategy arm weights, performance diagnostics, and rankings."""
    allocator = get_ensemble_allocator()
    rankings = allocator.get_strategy_rankings()
    return {
        "algorithm": allocator.algorithm,
        "total_rounds": allocator.total_rounds,
        "strategy_rankings": rankings,
        "arms": {
            sid: {
                "pull_count": arm.pull_count,
                "mean_reward": round(arm.mean_reward, 4),
                "current_weight": round(arm.current_weight, 4),
                "allocated_capital_usd": arm.allocated_capital_usd,
            }
            for sid, arm in allocator.arms.items()
        },
    }


@app.post("/analysis/ensemble/allocate", response_model=AllocationDecision, tags=["Intelligence & Analysis"])
async def allocate_ensemble_capital_endpoint(
    request: EnsembleAllocateRequest = EnsembleAllocateRequest(),
) -> AllocationDecision:
    """Dynamically allocate risk capital across strategy ensemble via Multi-Armed Bandit conditioning."""
    allocator = get_ensemble_allocator()
    if request.algorithm:
        allocator.algorithm = request.algorithm
    return allocator.allocate_capital(
        total_capital_usd=request.total_capital_usd,
        current_regime=request.current_regime,
    )


@app.post("/analysis/ensemble/reward/update", response_model=RewardUpdateResult, tags=["Intelligence & Analysis"])
async def update_ensemble_reward_endpoint(request: EnsembleRewardRequest) -> RewardUpdateResult:
    """Feed back realized strategy PnL to update bandit posteriors and exploration/exploitation weights."""
    allocator = get_ensemble_allocator()
    return allocator.update_reward(
        strategy_id=request.strategy_id,
        realized_pnl_usd=request.realized_pnl_usd,
        notional_traded_usd=request.notional_traded_usd,
        sharpe_equivalent=request.sharpe_equivalent,
    )


class DarkPoolTradeRequest(BaseModel):
    symbol: str = "BTC-USD"
    venue: str = "coinbase"
    price: float = 65010.0
    quantity: float = 2.0
    best_bid: float = 65000.0
    best_ask: float = 65020.0


class DarkPoolDepthRequest(BaseModel):
    symbol: str = "BTC-USD"
    venue: str = "binance"
    side: str = "BUY"
    price: float = 65000.0
    quantity: float = 5.0


@app.post("/analysis/darkpool/trades/process", response_model=DarkPrintRecord | None, tags=["Intelligence & Analysis"])
async def process_dark_trade_endpoint(request: DarkPoolTradeRequest) -> DarkPrintRecord | None:
    """Analyze incoming trade print for dark cross, midpoint execution, or internalization."""
    engine = get_dark_pool_engine()
    return engine.process_trade_print(
        symbol=request.symbol,
        venue=request.venue,
        price=request.price,
        quantity=request.quantity,
        best_bid=request.best_bid,
        best_ask=request.best_ask,
    )


@app.post("/analysis/darkpool/icebergs/update", response_model=IcebergSignature | None, tags=["Intelligence & Analysis"])
async def update_book_iceberg_endpoint(request: DarkPoolDepthRequest) -> IcebergSignature | None:
    """Process incremental order book depth update and detect active iceberg replenishments."""
    engine = get_dark_pool_engine()
    return engine.process_book_depth_update(
        symbol=request.symbol,
        venue=request.venue,
        side=request.side,
        price=request.price,
        new_quantity=request.quantity,
    )


@app.get("/analysis/darkpool/report/{symbol}", response_model=DarkPoolDetectionReport, tags=["Intelligence & Analysis"])
async def get_darkpool_report_endpoint(symbol: str) -> DarkPoolDetectionReport:
    """Synthesize real-time hidden liquidity, active iceberg orders, and dark volume share report."""
    engine = get_dark_pool_engine()
    return engine.generate_report(symbol)




class CointegrationAnalysisRequest(BaseModel):
    asset_y: str = "ETH-USD"
    asset_x: str = "BTC-USD"
    prices_y: list[float]
    prices_x: list[float]


class PairSpreadSignalRequest(BaseModel):
    pair_name: str = "ETH/BTC"
    asset_y: str = "ETH-USD"
    asset_x: str = "BTC-USD"
    price_y: float
    price_x: float
    z_entry: float = 2.0
    z_exit: float = 0.5


@app.post("/analysis/pairs/cointegration", response_model=CointegrationResult, tags=["Intelligence & Analysis"])
async def analyze_pair_cointegration_endpoint(request: CointegrationAnalysisRequest) -> CointegrationResult:
    """Perform Engle-Granger cointegration test and calculate Ornstein-Uhlenbeck half-life."""
    analyzer = CointegrationAnalyzer()
    sy = pd.Series(request.prices_y)
    sx = pd.Series(request.prices_x)
    return analyzer.test_cointegration(sy, sx, asset_y=request.asset_y, asset_x=request.asset_x)


_pair_monitors: dict[str, PairsSpreadMonitor] = {}


@app.post("/analysis/pairs/spread-signal", response_model=PairTradingSignal, tags=["Intelligence & Analysis"])
async def evaluate_pair_spread_signal_endpoint(request: PairSpreadSignalRequest) -> PairTradingSignal:
    """Online Kalman Filter hedge ratio adaptation and Z-score spread signal evaluation."""
    global _pair_monitors
    if request.pair_name not in _pair_monitors:
        _pair_monitors[request.pair_name] = PairsSpreadMonitor(
            pair_name=request.pair_name,
            asset_y=request.asset_y,
            asset_x=request.asset_x,
            z_entry=request.z_entry,
            z_exit=request.z_exit,
        )

    monitor = _pair_monitors[request.pair_name]
    return monitor.process_observation(price_y=request.price_y, price_x=request.price_x)


class BlackScholesPricingRequest(BaseModel):
    spot: float = 65000.0
    strike: float = 65000.0
    time_to_expiry: float = 0.25
    volatility: float = 0.55
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    option_type: str = "CALL"


class MertonPricingRequest(BaseModel):
    spot: float = 65000.0
    strike: float = 65000.0
    time_to_expiry: float = 0.25
    volatility: float = 0.50
    jump_intensity: float = 1.0
    jump_mean: float = -0.05
    jump_volatility: float = 0.15
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    option_type: str = "CALL"


class IVSolveRequest(BaseModel):
    market_price: float = 4500.0
    spot: float = 65000.0
    strike: float = 65000.0
    time_to_expiry: float = 0.25
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    option_type: str = "CALL"


class VolatilitySkewRequest(BaseModel):
    spot: float = 65000.0
    quotes: list[VolatilityPoint] = []
    time_to_expiry: float = 0.082


class PortfolioGreeksRequest(BaseModel):
    positions: list[DerivativePosition] = []


@app.post("/derivatives/pricing/black-scholes", response_model=OptionGreeks, tags=["Derivatives & Greeks Engine"])
async def price_black_scholes_endpoint(request: BlackScholesPricingRequest) -> OptionGreeks:
    """Calculate institutional Black-Scholes-Merton option price and 1st/2nd order Greeks."""
    op_type = OptionType.CALL if request.option_type.upper() == "CALL" else OptionType.PUT
    return BlackScholesEngine.calculate_greeks(
        spot=request.spot,
        strike=request.strike,
        time_to_expiry=request.time_to_expiry,
        volatility=request.volatility,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
        option_type=op_type,
    )


@app.post("/derivatives/pricing/merton", tags=["Derivatives & Greeks Engine"])
async def price_merton_jump_diffusion_endpoint(request: MertonPricingRequest) -> dict[str, Any]:
    """Calculate Merton (1976) jump-diffusion European option price via Poisson series expansion."""
    op_type = OptionType.CALL if request.option_type.upper() == "CALL" else OptionType.PUT
    price = MertonJumpDiffusionEngine.calculate_price(
        spot=request.spot,
        strike=request.strike,
        time_to_expiry=request.time_to_expiry,
        volatility=request.volatility,
        jump_intensity=request.jump_intensity,
        jump_mean=request.jump_mean,
        jump_volatility=request.jump_volatility,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
        option_type=op_type,
    )
    return {"price": round(price, 4), "spot": request.spot, "strike": request.strike, "option_type": request.option_type}


@app.post("/derivatives/volatility/solve-iv", tags=["Derivatives & Greeks Engine"])
async def solve_implied_volatility_endpoint(request: IVSolveRequest) -> dict[str, Any]:
    """Solve for implied volatility matching observed market price via Newton-Raphson/Brent."""
    op_type = OptionType.CALL if request.option_type.upper() == "CALL" else OptionType.PUT
    iv = ImpliedVolatilitySolver.solve_iv(
        market_price=request.market_price,
        spot=request.spot,
        strike=request.strike,
        time_to_expiry=request.time_to_expiry,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
        option_type=op_type,
    )
    return {
        "market_price": request.market_price,
        "implied_volatility": round(iv, 6) if iv is not None else None,
        "is_valid": iv is not None,
    }


@app.post("/derivatives/volatility/surface/skew", tags=["Derivatives & Greeks Engine"])
async def compute_volatility_skew_endpoint(request: VolatilitySkewRequest) -> dict[str, Any]:
    """Evaluate calibrated volatility surface skew and smile convexity metrics."""
    surface = VolatilitySurface(spot=request.spot, quotes=request.quotes)
    metrics = surface.compute_skew_metrics(time_to_expiry=request.time_to_expiry)
    return {"spot": request.spot, "metrics": metrics}


@app.post("/derivatives/portfolio/greeks", response_model=DerivativePortfolioReport, tags=["Derivatives & Greeks Engine"])
async def evaluate_derivatives_portfolio_greeks_endpoint(request: PortfolioGreeksRequest) -> DerivativePortfolioReport:
    """Aggregate multi-contract option positions into portfolio Greeks and delta-neutral hedging guidance."""
    engine = DerivativesRiskEngine()
    return engine.evaluate_portfolio_greeks(request.positions)


class SnapshotCreateRequest(BaseModel):
    snapshot_id: str | None = None


class SnapshotRestoreRequest(BaseModel):
    snapshot_id: str


@app.post("/recovery/snapshot/create", tags=["Disaster Recovery & High Availability"])
async def create_snapshot_endpoint(request: SnapshotCreateRequest = SnapshotCreateRequest()) -> dict[str, Any]:
    """Capture point-in-time state of the trading engine with cryptographic SHA-256 verification."""
    manager = get_snapshot_manager()
    engine = get_paper_engine()
    metadata = manager.create_snapshot(engine=engine, snapshot_id=request.snapshot_id)
    return metadata.model_dump()


@app.get("/recovery/snapshot/list", tags=["Disaster Recovery & High Availability"])
async def list_snapshots_endpoint() -> dict[str, Any]:
    """Enumerate all available point-in-time recovery snapshots sorted newest first."""
    manager = get_snapshot_manager()
    snapshots = manager.list_snapshots()
    return {"snapshots": [s.model_dump() for s in snapshots], "total_count": len(snapshots)}


@app.post("/recovery/snapshot/restore", tags=["Disaster Recovery & High Availability"])
async def restore_snapshot_endpoint(request: SnapshotRestoreRequest) -> dict[str, Any]:
    """Validate cryptographic integrity and rehydrate platform state from snapshot."""
    manager = get_snapshot_manager()
    engine = get_paper_engine()
    try:
        metadata = manager.restore_from_snapshot(snapshot_id=request.snapshot_id, engine=engine)
        return {
            "status": "RESTORED",
            "snapshot_id": metadata.snapshot_id,
            "equity": engine.equity,
            "cash": engine.cash,
            "positions_count": len(engine.positions),
        }
    except SnapshotIntegrityError as err:
        return {"status": "INTEGRITY_ERROR", "error": str(err)}
    except FileNotFoundError as err:
        return {"status": "NOT_FOUND", "error": str(err)}


@app.get("/recovery/ha/status", tags=["Disaster Recovery & High Availability"])
async def ha_cluster_status_endpoint() -> dict[str, Any]:
    """Retrieve high-availability cluster telemetry, active node role, and heartbeat status."""
    orchestrator = get_failover_orchestrator()
    return orchestrator.get_cluster_status()


class StressBenchmarkRequest(BaseModel):
    tick_count: int = 1000
    symbol: str = "BTC-USD"


@app.post("/system/benchmark/stress", response_model=StressBenchmarkResult, tags=["Monitoring & Diagnostics"])
async def run_stress_benchmark_endpoint(request: StressBenchmarkRequest = StressBenchmarkRequest()) -> StressBenchmarkResult:
    """Execute high-throughput orderbook stress benchmark and latency profiling."""
    gen = SyntheticTickStormGenerator(symbol=request.symbol, seed=42)
    ticks = gen.generate_ticks(count=min(request.tick_count, 10000))
    benchmark = PipelineThroughputBenchmark()
    return benchmark.benchmark_orderbook_ingestion(ticks=ticks, symbol=request.symbol)


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket) -> None:
    """Bi-directional WebSocket streaming gateway for live signals, ticks, and real-time dashboard updates."""
    from signals.fusion import run_live_intelligence_pipeline

    await ws_manager.connect(websocket)
    live_sub: dict[str, str] = {"symbol": "", "timeframe": "5m", "horizon": "1h"}

    async def _push_live_snapshot(sym: str, tf: str, hz: str) -> None:
        if not sym:
            return
        res = await asyncio.to_thread(
            run_live_intelligence_pipeline,
            symbol_or_query=sym,
            timeframe=tf,
            horizon=hz,
        )
        decision = res["decision"]
        st = res.get("market_state")
        inst = res.get("instrument")
        fc = res.get("forecast")
        candles_intraday = (st.rolling_candles_5m or st.rolling_candles_1m)[-60:] if st else []
        payload = {
            "topic": "live_prediction",
            "ws_send_timestamp": datetime.now(timezone.utc).isoformat(),
            "status": res.get("status", "OK"),
            "symbol": inst.symbol if inst else sym,
            "canonical_symbol": inst.canonical_symbol if inst else sym,
            "market": inst.market.value if inst else "NSE",
            "currency": inst.quote_currency if inst else "INR",
            "company_name": (inst.metadata or {}).get("name", inst.symbol if inst else sym) if inst else sym,
            "is_market_open": st.session_open if st else False,
            "is_stale": st.is_stale if st else False,
            "last_price": st.last_price if st else decision.entry_price,
            "decision": decision.model_dump(mode="json"),
            "forecast": fc.model_dump(mode="json") if fc else {},
            "developing_day": res.get("developing_day") or {},
            "mtf_context": res.get("mtf_context") or {},
            "latency_telemetry": res.get("latency_telemetry") or {},
            "candles": candles_intraday,
        }
        await websocket.send_text(json.dumps(payload))

    async def _periodic_live_pusher() -> None:
        while True:
            await asyncio.sleep(5.0)
            if live_sub["symbol"]:
                try:
                    await _push_live_snapshot(live_sub["symbol"], live_sub["timeframe"], live_sub["horizon"])
                except Exception:
                    break

    pusher_task = asyncio.create_task(_periodic_live_pusher())
    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
                action = msg.get("action")
                topic = msg.get("topic", "all")
                if action == "subscribe":
                    await ws_manager.subscribe(websocket, topic)
                    await websocket.send_text(json.dumps({"status": "subscribed", "topic": topic}))
                elif action == "unsubscribe":
                    await ws_manager.unsubscribe(websocket, topic)
                    await websocket.send_text(json.dumps({"status": "unsubscribed", "topic": topic}))
                elif action == "live_subscribe":
                    live_sub["symbol"] = str(msg.get("symbol") or "NSE:TCS")
                    live_sub["timeframe"] = str(msg.get("timeframe") or "5m")
                    live_sub["horizon"] = str(msg.get("horizon") or "1h")
                    await _push_live_snapshot(live_sub["symbol"], live_sub["timeframe"], live_sub["horizon"])
                elif action == "ping":
                    await websocket.send_text(
                        json.dumps({"status": "pong", "time": datetime.now(timezone.utc).isoformat()})
                    )
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
    except WebSocketDisconnect:
        pusher_task.cancel()
        await ws_manager.disconnect(websocket)
    except Exception as exc:
        pusher_task.cancel()
        logger.error("WebSocket streaming error", error=str(exc))
        await ws_manager.disconnect(websocket)


@app.get("/instruments", tags=["Market Data"])
async def list_instruments_endpoint(market: str | None = None) -> list[dict[str, Any]]:
    """List supported instruments across NSE, BSE, Forex, Quotex, and Crypto."""
    from core.instrument import MarketType
    from core.symbol_resolver import SymbolResolver

    m_enum = getattr(MarketType, market.upper(), None) if market else None
    instruments = SymbolResolver.get_all_instruments(m_enum)
    return [
        {
            "canonical_symbol": inst.canonical_symbol,
            "symbol": inst.symbol,
            "market": inst.market.value,
            "exchange": inst.exchange,
            "asset_class": inst.asset_class.value,
            "base_currency": inst.base_currency,
            "quote_currency": inst.quote_currency,
            "tick_size": inst.tick_size,
            "lot_size": inst.lot_size,
            "price_precision": inst.price_precision,
            "metadata": inst.metadata,
        }
        for inst in instruments
    ]


@app.get("/instruments/resolve/{query:path}", tags=["Market Data"])
async def resolve_instrument_endpoint(query: str) -> dict[str, Any]:
    """Resolve symbol alias or arbitrary stock (e.g. 'EUR/CHF', 'IRCTC', 'DIXON', '500325') to canonical instrument."""
    from core.symbol_resolver import SymbolResolver
    from core.universe import UniverseService

    inst = UniverseService.resolve_arbitrary_stock(query)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Instrument '{query}' not found in registry.")

    session = SymbolResolver.get_session(inst.market)
    return {
        "canonical_symbol": inst.canonical_symbol,
        "symbol": inst.symbol,
        "market": inst.market.value,
        "exchange": inst.exchange,
        "asset_class": inst.asset_class.value,
        "base_currency": inst.base_currency,
        "quote_currency": inst.quote_currency,
        "tick_size": inst.tick_size,
        "lot_size": inst.lot_size,
        "price_precision": inst.price_precision,
        "is_market_open": session.is_open() if session else True,
        "metadata": inst.metadata,
    }


@app.get("/market-state/{symbol:path}", tags=["Market Data"])
async def get_live_market_state_endpoint(
    symbol: str,
    timeframe: str = Query("5m", description="Candle timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
) -> dict[str, Any]:
    """Fetch real-time market state, bid/ask, ATR14, developing-day, MTF context, and candle count."""
    from core.symbol_resolver import SymbolResolver
    from core.universe import UniverseService
    from data.market_state import MarketStateService

    inst = UniverseService.resolve_arbitrary_stock(symbol)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Instrument '{symbol}' not found in universe.")

    state_svc = MarketStateService.get_instance()
    state = await asyncio.to_thread(state_svc.ensure_instrument_state, inst.canonical_symbol, timeframe)
    if not state:
        return {
            "status": "DATA_UNAVAILABLE",
            "symbol": inst.symbol,
            "canonical_symbol": inst.canonical_symbol,
            "market": inst.market.value,
        }

    c_list = state_svc.get_candles_list(state, timeframe)
    session = SymbolResolver.get_session(inst.market)
    return {
        "status": "OK",
        "symbol": state.symbol,
        "canonical_symbol": state.canonical_symbol,
        "market": state.market.value,
        "currency": inst.quote_currency,
        "provider": state.provider,
        "connection_state": state.connection_state,
        "is_market_open": session.is_open() if session else state.session_open,
        "last_price": state.last_price,
        "bid_price": state.bid_price,
        "ask_price": state.ask_price,
        "spread": state.spread,
        "spread_pct": state.spread_pct,
        "atr_14": state.atr_14,
        "volatility_pct": state.volatility_pct,
        "volume_24h": state.volume_24h,
        "last_tick_time": state.last_tick_time.isoformat(),
        "last_candle_time": state.last_candle_time.isoformat() if state.last_candle_time else None,
        "data_age_seconds": round(state.data_age_seconds, 2),
        "is_stale": state.is_stale,
        "candle_count": len(c_list),
        "developing_day": state.developing_day,
        "mtf_context": state.mtf_context,
        "tick_size": inst.tick_size,
        "lot_size": inst.lot_size,
        "metadata": inst.metadata,
    }


@app.get("/universe/search", tags=["Market Universe"])
async def universe_search_endpoint(
    q: str = Query(..., description="Stock symbol, name, ISIN, or BSE scrip code to search"),
    market: str | None = None,
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Search active 9,100+ market universe across NSE, BSE, Quotex, Forex, and Crypto."""
    from core.instrument import MarketType
    from core.universe import UniverseService

    m_enum = getattr(MarketType, market.upper(), None) if market else None
    return UniverseService.search(query=q, market=m_enum, limit=limit)


@app.get("/universe/summary", tags=["Market Universe"])
async def universe_summary_endpoint() -> dict[str, Any]:
    """Return complete dynamic universe summary across NSE, BSE, Quotex, Forex, and Crypto."""
    from core.universe import UniverseService

    return UniverseService.summary()


@app.get("/universe/audit", tags=["Market Universe"])
async def universe_audit_endpoint() -> dict[str, Any]:
    """Run complete Universe Health Check reporting discovered, active, data available, predictable, and missing instruments."""
    from core.universe import UniverseService

    return await asyncio.to_thread(UniverseService.run_universe_audit)


@app.post("/universe/sync", tags=["Market Universe"])
async def universe_sync_endpoint(refresh_remote: bool = Query(False)) -> dict[str, Any]:
    """Trigger UniverseSyncService synchronization and return sync audit logs."""
    from core.universe import UniverseService, UniverseSyncService

    total = await asyncio.to_thread(UniverseService.sync_universe, refresh_remote)
    return {
        "status": "SYNCHRONIZED",
        "total_instruments": total,
        "summary": UniverseService.summary(),
        "sync_logs": UniverseSyncService.get_sync_history(limit=10),
    }


@app.get("/scanner/opportunities", tags=["Market Intelligence"])
async def market_scanner_endpoint(
    market: str | None = None,
    min_confidence: float = 0.50,
    horizon: str = "1h",
    limit: int = 10,
    q: str | None = None,
    signal: str | None = None,
    sort_by: str = "rank",
    include_no_trade: bool = False,
) -> dict[str, Any]:
    """Scan and rank market opportunities across the complete NSE, BSE, Quotex, Forex, and Crypto universe."""
    from core.instrument import MarketType
    from signals.scanner import MarketScannerService

    m_enum = getattr(MarketType, market.upper(), None) if market and market.upper() != "ALL" else None
    scanner = MarketScannerService()
    report = await asyncio.to_thread(
        scanner.scan_market,
        market=m_enum,
        min_confidence=min_confidence,
        horizon=horizon,
        limit=limit,
        search_query=q,
        signal_filter=signal,
        sort_by=sort_by,
        include_no_trade=include_no_trade,
    )
    return report.model_dump()


@app.get("/forecast/{symbol:path}", tags=["Market Intelligence"])
async def get_probabilistic_forecast_endpoint(
    symbol: str,
    horizon: str = Query("1h", description="Forecast horizon: 5m, 15m, 1h, 4h, 1d"),
    timeframe: str = Query("5m", description="Candle timeframe: 1m, 5m, 15m, 1h, 1d"),
    all_horizons: bool = Query(False, description="Return all 5 horizons (5m, 15m, 1h, 4h, 1d)"),
) -> dict[str, Any]:
    """Generate multi-horizon probabilistic forecast with P(up), P(down), P(flat), and target probabilities."""
    from signals.forecast import ProbabilisticForecastingEngine
    from signals.fusion import run_live_intelligence_pipeline

    res = await asyncio.to_thread(
        run_live_intelligence_pipeline,
        symbol_or_query=symbol,
        timeframe=timeframe,
        horizon=horizon,
    )
    fc = res["forecast"]
    out = fc.model_dump()
    if all_horizons and res.get("market_state") is not None:
        state = res["market_state"]
        ml_pred = res.get("ml_prediction")
        ml_prob_up = ml_pred.raw_scores.get("buy") if (ml_pred and ml_pred.raw_scores) else None
        multi_rep = ProbabilisticForecastingEngine().forecast_all_horizons(
            symbol=fc.symbol,
            current_price=fc.entry_reference_price,
            atr=state.atr_14,
            primary_horizon=horizon,
            regime=res.get("regime"),
            technical_signals=res.get("technical_signals"),
            candle_history=state.rolling_candles_5m or state.rolling_candles_1m,
            ml_prob_up=ml_prob_up,
            pattern_consensus=res.get("pattern_consensus"),
            latest_features=res.get("features"),
            data_is_stale=state.is_stale,
        )
        out["horizons"] = {k: v.model_dump() for k, v in multi_rep.forecasts.items()}
    return out


@app.get("/learning/ledger/status", tags=["Learning & Governance"])
async def get_ledger_status_endpoint() -> dict[str, Any]:
    """Get active prediction ledger resolution count, accuracy, Brier score, and win rates."""
    from learning.ledger import PredictionLedgerService
    ledger = PredictionLedgerService.get_instance()
    return ledger.get_metrics()


@app.post("/learning/self-development/run", tags=["Learning & Governance"])
async def run_self_development_endpoint(symbol: str = Query("NSE:TCS")) -> dict[str, Any]:
    """Trigger a controlled background Self-Development cycle with Purged Walk-Forward Validation & Governance."""
    from learning.ledger import PredictionLedgerService
    ledger = PredictionLedgerService.get_instance()
    return await asyncio.to_thread(ledger.run_self_development_cycle, symbol)


@app.get("/prometheus", response_class=Response, tags=["Monitoring"])
async def prometheus_metrics_endpoint() -> Response:
    """Prometheus metrics endpoint in standard exposition format."""
    text_data = get_latest_metrics_text()
    return Response(content=text_data, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/dashboard", response_class=HTMLResponse, tags=["Observability & UI"])
async def dashboard_endpoint() -> HTMLResponse:
    """Beginner-Simple Real-Time Trading Intelligence Dashboard with Collapsible Advanced System Details."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading Intelligence Platform</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased py-6 px-4 md:px-8">
  <div class="max-w-4xl mx-auto space-y-6">

    <!-- Minimal Top Bar: Search Any Stock/Pair + Forecast Time Horizon + Market Scanner Link -->
    <div class="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-slate-900/90 border border-slate-800 p-3 rounded-2xl shadow-lg">
      <div class="relative flex-1">
        <div class="flex items-center gap-2">
          <input
            id="symbolInput"
            type="text"
            value="NSE:TCS"
            placeholder="Search 9,100+ NSE, BSE, Quotex, Forex & Crypto instruments (symbol, name, ISIN, scrip)..."
            oninput="onSearchInput(this.value)"
            onkeydown="if(event.key==='Enter'){selectSymbol(this.value);}"
            class="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          />
          <button
            onclick="selectSymbol(document.getElementById('symbolInput').value)"
            class="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold rounded-xl transition shrink-0"
          >
            Analyze
          </button>
        </div>
        <!-- Autocomplete Search Dropdown -->
        <div id="searchResultsBox" class="hidden absolute z-30 left-0 right-0 mt-1 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-h-60 overflow-y-auto divide-y divide-slate-800"></div>
      </div>

      <div class="flex items-center gap-2">
        <label for="horizonSelect" class="text-xs text-slate-400 whitespace-nowrap pl-1">Forecast Horizon:</label>
        <select
          id="horizonSelect"
          onchange="onHorizonChange()"
          class="bg-slate-950 border border-slate-700/80 rounded-xl px-3 py-2.5 text-sm text-white font-medium focus:outline-none focus:border-indigo-500"
        >
          <option value="5m">Next 5 Minutes</option>
          <option value="15m">Next 15 Minutes</option>
          <option value="1h" selected>Next 1 Hour</option>
          <option value="4h">Next 4 Hours</option>
          <option value="1d">Next 1 Day</option>
        </select>
        <a
          href="/scanner"
          class="px-3.5 py-2.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-indigo-300 text-xs font-bold rounded-xl transition whitespace-nowrap"
        >
          Market Scanner →
        </a>
      </div>
    </div>

    <!-- Quick Instrument Pills -->
    <div class="flex flex-wrap items-center gap-2 text-xs">
      <span class="text-slate-500 mr-1">Popular:</span>
      <button onclick="selectSymbol('NSE:TCS')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">TCS • NSE</button>
      <button onclick="selectSymbol('NSE:RELIANCE')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">RELIANCE • NSE</button>
      <button onclick="selectSymbol('NSE:IRCTC')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">IRCTC • NSE</button>
      <button onclick="selectSymbol('NSE:DIXON')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">DIXON • NSE</button>
      <button onclick="selectSymbol('BSE:RELIANCE')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">RELIANCE • BSE</button>
      <button onclick="selectSymbol('QUOTEX:EURUSD')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">EUR/USD • QUOTEX</button>
      <button onclick="selectSymbol('FX:EURCHF')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">EUR/CHF • FOREX</button>
      <button onclick="selectSymbol('CRYPTO:BTC-USD')" class="px-3 py-1 rounded-full bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition">BTC-USD • CRYPTO</button>
    </div>

    <!-- ===================================================================== -->
    <!-- BEGINNER-SIMPLE MAIN DASHBOARD CARD                                   -->
    <!-- ===================================================================== -->
    <main class="bg-slate-900 border border-slate-800 rounded-3xl p-6 md:p-8 shadow-2xl space-y-8">

      <!-- 1. Instrument Header & Live Status + 2. Current Live Price -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <div class="flex items-center gap-3">
            <h1 id="mainInstrumentTitle" class="text-2xl md:text-3xl font-extrabold tracking-tight text-white">TCS • NSE</h1>
            <span id="liveStatusBadge" class="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
              <span id="liveStatusDot" class="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
              <span id="liveStatusText">LIVE</span>
            </span>
          </div>
          <p id="companyNameSubtitle" class="text-sm text-slate-400 mt-1">Tata Consultancy Services Ltd</p>
        </div>

        <div class="sm:text-right">
          <div id="mainLivePrice" class="text-3xl md:text-4xl font-black tracking-tight text-white font-mono">₹--</div>
          <div id="intradayChangeBadge" class="text-xs font-medium text-slate-400 mt-1">Synchronizing live market state...</div>
        </div>
      </div>

      <!-- 3. Primary Signal + 4. Confidence + 5. Forecast Horizon -->
      <div id="signalHeroCard" class="rounded-2xl p-6 md:p-8 text-center border border-slate-800 bg-slate-950/70 transition-all duration-300">
        <div id="mainSignalText" class="text-5xl md:text-6xl font-black tracking-wider uppercase text-slate-200">
          EVALUATING
        </div>
        <div id="mainConfidenceText" class="text-lg md:text-xl font-semibold text-slate-300 mt-3">
          --% confidence
        </div>
        <div id="mainHorizonLabel" class="inline-block mt-2 px-3.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-slate-800 text-slate-300">
          Next 1 Hour
        </div>
      </div>

      <!-- 6. Entry, Stop Loss, and Target -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div class="bg-slate-950/80 border border-slate-800/90 rounded-2xl p-4 text-center">
          <div class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Entry</div>
          <div id="mainEntryPrice" class="text-xl md:text-2xl font-bold text-white font-mono mt-1">--</div>
        </div>
        <div class="bg-slate-950/80 border border-rose-500/20 rounded-2xl p-4 text-center">
          <div class="text-xs uppercase tracking-wider text-rose-400 font-semibold">Stop Loss</div>
          <div id="mainStopPrice" class="text-xl md:text-2xl font-bold text-rose-400 font-mono mt-1">--</div>
        </div>
        <div class="bg-slate-950/80 border border-emerald-500/20 rounded-2xl p-4 text-center">
          <div class="text-xs uppercase tracking-wider text-emerald-400 font-semibold">Target</div>
          <div id="mainTargetPrice" class="text-xl md:text-2xl font-bold text-emerald-400 font-mono mt-1">--</div>
        </div>
      </div>

      <!-- 7. LIVE INTRADAY GRAPH -->
      <div class="bg-slate-950 border border-slate-800/90 rounded-2xl p-4 space-y-2">
        <div class="flex items-center justify-between text-xs text-slate-400 px-1">
          <span class="font-semibold uppercase tracking-wider text-slate-300">Live Intraday Graph</span>
          <div class="flex items-center gap-4">
            <span class="inline-flex items-center gap-1"><span class="w-2.5 h-0.5 bg-indigo-400 inline-block"></span> VWAP</span>
            <span class="inline-flex items-center gap-1"><span class="w-2.5 h-0.5 bg-emerald-400 inline-block"></span> Target</span>
            <span class="inline-flex items-center gap-1"><span class="w-2.5 h-0.5 bg-rose-400 inline-block"></span> Stop Loss</span>
          </div>
        </div>
        <div class="w-full h-64 relative">
          <canvas id="intradayChart" class="w-full h-full block"></canvas>
        </div>
      </div>

      <!-- 8. WHY? (Plain-English Beginner Explanations) -->
      <div class="bg-slate-950/70 border border-slate-800/90 rounded-2xl p-5 space-y-3">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400">WHY?</h2>
        <ul id="whyList" class="space-y-2.5 text-sm md:text-base text-slate-200">
          <li class="flex items-start gap-2.5">
            <span class="text-emerald-400 font-bold">✓</span>
            <span>Analyzing live market trend, developing-day structure, and historical situations...</span>
          </li>
        </ul>
      </div>

    </main>

    <!-- ===================================================================== -->
    <!-- OPTIONAL COLLAPSIBLE: ADVANCED / SYSTEM DETAILS                       -->
    <!-- ===================================================================== -->
    <details id="advancedSystemDetails" class="bg-slate-900/70 border border-slate-800 rounded-2xl overflow-hidden">
      <summary class="cursor-pointer select-none px-6 py-4 text-sm font-semibold text-slate-300 hover:text-white flex items-center justify-between">
        <span>Advanced / System Details</span>
        <span class="text-xs font-normal text-slate-500">Click to expand technical indicators, MTF matrix, latency, governance & scanner</span>
      </summary>

      <div class="p-6 border-t border-slate-800 space-y-6 text-xs">

        <!-- Stage-by-Stage Real-Time Latency & Freshness -->
        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
          <div class="font-bold text-slate-300 uppercase tracking-wider">1. Real-Time Pipeline Latency & Data Freshness</div>
          <div id="advLatencyRow" class="grid grid-cols-2 md:grid-cols-5 gap-3 font-mono text-slate-400">
            <div>Provider: <span id="advProvider" class="text-white">--</span></div>
            <div>Data Age: <span id="advDataAge" class="text-white">--</span></div>
            <div>Feature/Regime: <span id="advFeatMs" class="text-emerald-400">--</span></div>
            <div>Total Pipeline: <span id="advTotalMs" class="text-emerald-400">--</span></div>
            <div>Leakage Guard: <span id="advLeakageGuard" class="text-indigo-400">VERIFIED_LE_T</span></div>
          </div>
        </div>

        <!-- Developing-Day Analysis & Multi-Timeframe Synchronization -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <div class="font-bold text-slate-300 uppercase tracking-wider">2. Developing-Day State (Open → Current T)</div>
            <div id="advDevelopingDay" class="grid grid-cols-2 gap-2 font-mono text-slate-400">
              <div>Day Open: <span id="ddOpen" class="text-white">--</span></div>
              <div>Prev Close: <span id="ddPrevClose" class="text-white">--</span></div>
              <div>Session High/Low: <span id="ddHighLow" class="text-white">--</span></div>
              <div>Session VWAP: <span id="ddVwap" class="text-indigo-400">--</span></div>
              <div>Opening Range: <span id="ddOR" class="text-emerald-400">--</span></div>
              <div>Rel Volume: <span id="ddRvol" class="text-white">--</span></div>
              <div>Support / Res: <span id="ddSupRes" class="text-white">--</span></div>
              <div>Structure: <span id="ddStruct" class="text-amber-400">--</span></div>
            </div>
          </div>

          <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <div class="font-bold text-slate-300 uppercase tracking-wider">3. Multi-Timeframe Synchronization (1m → 1d)</div>
            <div id="advMtfSummary" class="text-slate-400 font-mono mb-1">Dominant Bias: --</div>
            <div id="advMtfTable" class="grid grid-cols-3 gap-2 font-mono text-slate-300"></div>
          </div>
        </div>

        <!-- Technical Indicators, Regime, ML Probabilities & Historical Similarity -->
        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
          <div class="font-bold text-slate-300 uppercase tracking-wider">4. Quantitative Features, Regime & Model Probabilities</div>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-slate-400">
            <div>Regime Code: <span id="advRegime" class="text-indigo-400">--</span></div>
            <div>RSI (14): <span id="advRsi" class="text-white">--</span></div>
            <div>ADX (14): <span id="advAdx" class="text-white">--</span></div>
            <div>ATR (14): <span id="advAtr" class="text-white">--</span></div>
            <div>MACD Hist: <span id="advMacd" class="text-white">--</span></div>
            <div>Fusion P(Buy/Sell/Flat): <span id="advProbs" class="text-emerald-400">--</span></div>
            <div>Historical Win Rate: <span id="advHistWin" class="text-white">--</span></div>
            <div>Time-Aligned Matches: <span id="advTodMatches" class="text-indigo-400">--</span></div>
          </div>
          <div class="pt-2 border-t border-slate-900 font-mono text-slate-500" id="advRawReasons">--</div>
        </div>

        <!-- Out-of-Sample Prediction Ledger & Self-Development Governance -->
        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="flex items-center justify-between">
            <div class="font-bold text-slate-300 uppercase tracking-wider">5. Prediction Outcome Ledger & Controlled Self-Development</div>
            <button onclick="triggerSelfDevelopment()" class="px-3 py-1 bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 rounded text-indigo-300 font-mono">
              Run Challenger Validation Cycle
            </button>
          </div>
          <div class="grid grid-cols-2 md:grid-cols-5 gap-3 font-mono text-slate-400">
            <div>Accuracy: <span id="advLedgerAcc" class="text-amber-400">--</span></div>
            <div>Precision / Recall: <span id="advPrecRec" class="text-white">--</span></div>
            <div>F1 / Brier Score: <span id="advF1Brier" class="text-white">--</span></div>
            <div>Expectancy / PF: <span id="advExpPf" class="text-emerald-400">--</span></div>
            <div>Sharpe / MFE / MAE: <span id="advSharpeMfe" class="text-white">--</span></div>
          </div>
          <div id="advGovernanceBox" class="p-3 rounded bg-slate-900/90 border border-slate-800 font-mono text-slate-300">
            Synchronizing Champion vs. Challenger governance status...
          </div>
        </div>

        <!-- Continuous Market Opportunity Scanner -->
        <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
          <div class="flex items-center justify-between">
            <div class="font-bold text-slate-300 uppercase tracking-wider">6. Multi-Market Opportunity Scanner</div>
            <div class="flex items-center gap-2">
              <select id="marketSelect" onchange="runScanner()" class="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-white">
                <option value="ALL">ALL MARKETS</option>
                <option value="NSE">NSE</option>
                <option value="BSE">BSE</option>
                <option value="FOREX">FOREX</option>
                <option value="QUOTEX">QUOTEX</option>
                <option value="CRYPTO">CRYPTO</option>
              </select>
              <button onclick="runScanner()" class="px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded text-white">Refresh Scan</button>
            </div>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs text-slate-300 font-mono">
              <thead class="bg-slate-900 text-slate-400 uppercase">
                <tr>
                  <th class="p-2">Rank</th>
                  <th class="p-2">Instrument</th>
                  <th class="p-2">Signal</th>
                  <th class="p-2">Confidence</th>
                  <th class="p-2">Price</th>
                  <th class="p-2">Target</th>
                  <th class="p-2">Stop Loss</th>
                </tr>
              </thead>
              <tbody id="scannerTableBody" class="divide-y divide-slate-900">
                <tr><td colspan="7" class="p-3 text-center text-slate-500">Scanning market universe...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </details>

  </div>

  <script>
    let currentSymbol = "NSE:TCS";
    let currentTimeframe = "5m";
    let currentHorizon = "1h";
    let wsConnection = null;
    let searchTimeout = null;
    let latestCandles = [];

    const HORIZON_LABELS = {
      "5m": "Next 5 Minutes",
      "15m": "Next 15 Minutes",
      "1h": "Next 1 Hour",
      "4h": "Next 4 Hours",
      "1d": "Next 1 Day"
    };

    function formatCurrencyPrice(price, market, currency) {
      if (price === null || price === undefined || isNaN(price) || Number(price) === 0) return "--";
      const num = Number(price);
      const isFX = (market === "FOREX" || market === "QUOTEX" || num < 10);
      const decimals = isFX ? 4 : 2;
      let prefix = "₹";
      if (market === "CRYPTO" || currency === "USD") prefix = "$";
      else if (market === "FOREX" || market === "QUOTEX") prefix = "";
      return prefix + num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }

    async function onSearchInput(query) {
      const box = document.getElementById("searchResultsBox");
      if (!query || query.trim().length < 2) {
        box.classList.add("hidden");
        return;
      }
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(async () => {
        try {
          const resp = await fetch("/universe/search?q=" + encodeURIComponent(query.trim()) + "&limit=8");
          if (!resp.ok) return;
          const items = await resp.json();
          if (!items || items.length === 0) {
            box.classList.add("hidden");
            return;
          }
          box.innerHTML = items.map(item => `
            <div
              onclick="selectSymbol('${item.canonical_symbol}')"
              class="px-4 py-2.5 hover:bg-slate-800 cursor-pointer flex items-center justify-between text-xs"
            >
              <div>
                <span class="font-bold text-white">${item.symbol}</span>
                <span class="text-slate-400 ml-2">${item.name || ""}</span>
              </div>
              <span class="px-2 py-0.5 rounded bg-slate-800 text-indigo-400 font-mono">${item.market}</span>
            </div>
          `).join("");
          box.classList.remove("hidden");
        } catch (e) {}
      }, 180);
    }

    function selectSymbol(sym) {
      if (!sym) return;
      currentSymbol = sym.trim().toUpperCase();
      document.getElementById("symbolInput").value = currentSymbol;
      document.getElementById("searchResultsBox").classList.add("hidden");
      refreshActiveInstrument();
      subscribeWebSocketLive();
    }

    function onHorizonChange() {
      currentHorizon = document.getElementById("horizonSelect").value || "1h";
      document.getElementById("mainHorizonLabel").innerText = HORIZON_LABELS[currentHorizon] || "Next 1 Hour";
      refreshActiveInstrument();
      subscribeWebSocketLive();
    }

    function renderMainDashboard(payload) {
      const uiRenderStart = performance.now();
      const sym = payload.symbol || currentSymbol;
      const market = payload.market || "NSE";
      const currency = payload.currency || (market === "NSE" || market === "BSE" ? "INR" : "USD");
      const price = payload.last_price || (payload.decision && payload.decision.entry_price) || 0;
      const dec = payload.decision || {};
      const fc = payload.forecast || (dec.evidence && dec.evidence.forecast) || {};
      const devDay = payload.developing_day || (dec.evidence && dec.evidence.developing_day) || {};
      const mtf = payload.mtf_context || (dec.evidence && dec.evidence.mtf_context) || {};
      const lat = payload.latency_telemetry || (dec.evidence && dec.evidence.latency_telemetry) || {};
      const feats = (dec.evidence && dec.evidence.features) || {};
      const pat = (dec.evidence && dec.evidence.pattern_consensus) || {};
      const mktState = (dec.evidence && dec.evidence.market_state) || {};

      // 1. Title & Live Status
      document.getElementById("mainInstrumentTitle").innerText = `${sym} • ${market}`;
      document.getElementById("companyNameSubtitle").innerText = payload.company_name || `${sym} (${market})`;

      const statusBadge = document.getElementById("liveStatusBadge");
      const statusDot = document.getElementById("liveStatusDot");
      const statusText = document.getElementById("liveStatusText");

      if (payload.status && payload.status !== "OK" && dec.status && dec.status !== "EVALUATED") {
        statusText.innerText = dec.status;
        statusBadge.className = "inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-rose-500/15 text-rose-400 border border-rose-500/30";
        statusDot.className = "h-2.5 w-2.5 rounded-full bg-rose-400";
      } else if (payload.is_stale) {
        statusText.innerText = "STALE DATA";
        statusBadge.className = "inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/15 text-amber-400 border border-amber-500/30";
        statusDot.className = "h-2.5 w-2.5 rounded-full bg-amber-400";
      } else if (payload.is_market_open) {
        statusText.innerText = "LIVE";
        statusBadge.className = "inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30";
        statusDot.className = "h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse";
      } else {
        statusText.innerText = "MARKET CLOSED (LAST SESSION)";
        statusBadge.className = "inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/15 text-amber-400 border border-amber-500/30";
        statusDot.className = "h-2.5 w-2.5 rounded-full bg-amber-400";
      }

      // 2. Current Live Price & Intraday Return
      document.getElementById("mainLivePrice").innerText = formatCurrencyPrice(price, market, currency);
      if (devDay && devDay.intraday_return_pct !== undefined) {
        const ret = Number(devDay.intraday_return_pct);
        const sign = ret >= 0 ? "+" : "";
        const col = ret >= 0 ? "text-emerald-400" : "text-rose-400";
        document.getElementById("intradayChangeBadge").className = `text-xs font-semibold mt-1 ${col}`;
        document.getElementById("intradayChangeBadge").innerText = `${sign}${ret.toFixed(2)}% today (Open: ${formatCurrencyPrice(devDay.day_open, market, currency)})`;
      }

      // 3. Visually Dominant Primary Signal + 4. Confidence + 5. Horizon
      const rawAction = (dec.action || "NO_TRADE").replace("_", "-");
      const confPct = Math.round(Number(dec.confidence || 0) * 100);
      const heroCard = document.getElementById("signalHeroCard");
      const sigEl = document.getElementById("mainSignalText");

      sigEl.innerText = rawAction;
      document.getElementById("mainConfidenceText").innerText = `${confPct}% confidence`;
      document.getElementById("mainHorizonLabel").innerText = HORIZON_LABELS[currentHorizon] || "Next 1 Hour";

      if (rawAction === "BUY") {
        heroCard.className = "rounded-2xl p-6 md:p-8 text-center border-2 border-emerald-500/50 bg-emerald-950/25 shadow-lg shadow-emerald-950/30";
        sigEl.className = "text-5xl md:text-6xl font-black tracking-wider uppercase text-emerald-400";
      } else if (rawAction === "SELL") {
        heroCard.className = "rounded-2xl p-6 md:p-8 text-center border-2 border-rose-500/50 bg-rose-950/25 shadow-lg shadow-rose-950/30";
        sigEl.className = "text-5xl md:text-6xl font-black tracking-wider uppercase text-rose-400";
      } else {
        heroCard.className = "rounded-2xl p-6 md:p-8 text-center border border-slate-700 bg-slate-950/80";
        sigEl.className = "text-5xl md:text-6xl font-black tracking-wider uppercase text-amber-400";
      }

      // 6. Entry, Stop Loss, and Target (use horizon forecast target/stop if available, else decision envelope)
      const entryVal = fc.entry_reference_price || dec.entry_price || price;
      const stopVal = fc.invalidation_stop_price || dec.stop_loss || price;
      const targetVal = fc.target_price || dec.take_profit || price;

      document.getElementById("mainEntryPrice").innerText = formatCurrencyPrice(entryVal, market, currency);
      document.getElementById("mainStopPrice").innerText = formatCurrencyPrice(stopVal, market, currency);
      document.getElementById("mainTargetPrice").innerText = formatCurrencyPrice(targetVal, market, currency);

      // 7. Live Intraday Graph
      const candles = payload.candles || (dec.evidence && dec.evidence.candles_intraday) || latestCandles;
      if (candles && candles.length > 0) {
        latestCandles = candles;
        drawIntradayChart(candles, entryVal, stopVal, targetVal, devDay.vwap);
      }

      // 8. WHY? (Beginner-Friendly Plain-English Explanations)
      const humanReasons = (dec.human_reasons && dec.human_reasons.length > 0)
        ? dec.human_reasons
        : ["Market structure and historical patterns are being evaluated for this instrument"];
      const checkColor = rawAction === "SELL" ? "text-rose-400" : (rawAction === "BUY" ? "text-emerald-400" : "text-amber-400");
      document.getElementById("whyList").innerHTML = humanReasons.map(r => `
        <li class="flex items-start gap-2.5">
          <span class="${checkColor} font-bold">✓</span>
          <span>${r}</span>
        </li>
      `).join("");

      // 9. Populate Optional Advanced / System Details
      const uiRenderMs = (performance.now() - uiRenderStart).toFixed(1);
      document.getElementById("advProvider").innerText = mktState.provider || "--";
      document.getElementById("advDataAge").innerText = (mktState.data_age_seconds !== undefined ? mktState.data_age_seconds + "s" : "--");
      document.getElementById("advFeatMs").innerText = (lat.feature_and_regime_ms !== undefined ? lat.feature_and_regime_ms + " ms" : "--");
      document.getElementById("advTotalMs").innerText = (lat.total_pipeline_ms !== undefined ? `${lat.total_pipeline_ms} ms (UI: ${uiRenderMs} ms)` : "--");
      document.getElementById("advLeakageGuard").innerText = lat.future_leakage_guard || "VERIFIED_LE_T";

      if (devDay && devDay.day_open) {
        document.getElementById("ddOpen").innerText = devDay.day_open;
        document.getElementById("ddPrevClose").innerText = devDay.previous_close;
        document.getElementById("ddHighLow").innerText = `${devDay.current_high} / ${devDay.current_low}`;
        document.getElementById("ddVwap").innerText = `${devDay.vwap} (${devDay.price_vs_vwap_pct >= 0 ? '+' : ''}${devDay.price_vs_vwap_pct}%)`;
        document.getElementById("ddOR").innerText = devDay.opening_range_breakout;
        document.getElementById("ddRvol").innerText = devDay.relative_volume + "x";
        document.getElementById("ddSupRes").innerText = `${devDay.support_level} / ${devDay.resistance_level}`;
        document.getElementById("ddStruct").innerText = devDay.trend_structure;
      }

      if (mtf && mtf.timeframes) {
        document.getElementById("advMtfSummary").innerText =
          `Dominant Bias: ${mtf.dominant_bias} (Alignment Score: ${mtf.alignment_score}, Bull TFs: ${mtf.bullish_timeframes}, Bear TFs: ${mtf.bearish_timeframes})`;
        document.getElementById("advMtfTable").innerHTML = Object.entries(mtf.timeframes).map(([tfKey, info]) => {
          const badgeCol = info.trend === "BULLISH" ? "text-emerald-400" : (info.trend === "BEARISH" ? "text-rose-400" : "text-slate-400");
          return `<div class="p-2 rounded bg-slate-900 border border-slate-800">
            <div class="font-bold text-white">${tfKey} <span class="${badgeCol}">${info.trend}</span></div>
            <div class="text-slate-400">${info.return_pct >= 0 ? '+' : ''}${info.return_pct}% (${info.bars}b)</div>
          </div>`;
        }).join("");
      }

      document.getElementById("advRegime").innerText = dec.regime_code || "--";
      document.getElementById("advRsi").innerText = feats.rsi_14 ?? "--";
      document.getElementById("advAdx").innerText = feats.adx_14 ?? "--";
      document.getElementById("advAtr").innerText = feats.atr_14 ?? "--";
      document.getElementById("advMacd").innerText = feats.macd_hist ?? "--";
      if (dec.buy_probability !== undefined && dec.buy_probability !== null) {
        document.getElementById("advProbs").innerText =
          `${(dec.buy_probability * 100).toFixed(1)}% / ${(dec.sell_probability * 100).toFixed(1)}% / ${(dec.no_trade_probability * 100).toFixed(1)}%`;
      }
      if (pat && pat.sample_size !== undefined) {
        document.getElementById("advHistWin").innerText = `${(Number(pat.win_rate || 0) * 100).toFixed(1)}% (${pat.sample_size} matches)`;
        document.getElementById("advTodMatches").innerText = `${pat.time_aligned_matches ?? 0} same-time-of-day`;
      }
      if (dec.reasons) {
        document.getElementById("advRawReasons").innerText = dec.reasons.join(" | ");
      }
    }

    function drawIntradayChart(candles, entryPrice, stopPrice, targetPrice, vwapPrice) {
      const canvas = document.getElementById("intradayChart");
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);

      const W = rect.width;
      const H = rect.height;
      ctx.clearRect(0, 0, W, H);

      const slice = candles.slice(-48);
      if (slice.length === 0) return;

      let minP = Infinity;
      let maxP = -Infinity;
      slice.forEach(c => {
        if (c.low < minP) minP = Number(c.low);
        if (c.high > maxP) maxP = Number(c.high);
      });
      [entryPrice, stopPrice, targetPrice, vwapPrice].forEach(v => {
        if (v && v > 0) {
          if (v < minP) minP = Number(v);
          if (v > maxP) maxP = Number(v);
        }
      });

      const pad = (maxP - minP) * 0.08 || (maxP * 0.005) || 1;
      minP -= pad;
      maxP += pad;

      const padLeft = 12;
      const padRight = 68;
      const padTop = 12;
      const padBottom = 20;
      const plotW = W - padLeft - padRight;
      const plotH = H - padTop - padBottom;

      const yOf = (p) => padTop + plotH * (1 - (p - minP) / Math.max(1e-8, maxP - minP));

      // Subtle horizontal grid lines
      ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const gy = padTop + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(padLeft, gy);
        ctx.lineTo(W - padRight, gy);
        ctx.stroke();
      }

      // Draw Candlesticks & Running Session VWAP
      const step = plotW / Math.max(1, slice.length);
      const barW = Math.max(2, Math.min(10, step * 0.65));
      let cumPV = 0;
      let cumV = 0;
      const vwapPoints = [];

      slice.forEach((c, idx) => {
        const x = padLeft + idx * step + step / 2;
        const o = Number(c.open);
        const h = Number(c.high);
        const l = Number(c.low);
        const cl = Number(c.close);
        const v = Math.max(1, Number(c.volume || 1));

        cumPV += ((h + l + cl) / 3) * v;
        cumV += v;
        vwapPoints.push({ x, y: yOf(cumPV / cumV) });

        const isBull = cl >= o;
        ctx.strokeStyle = isBull ? "#34d399" : "#fb7185";
        ctx.fillStyle = isBull ? "#10b981" : "#f43f5e";

        // Wick
        ctx.beginPath();
        ctx.moveTo(x, yOf(h));
        ctx.lineTo(x, yOf(l));
        ctx.stroke();

        // Body
        const yOpen = yOf(o);
        const yClose = yOf(cl);
        const topY = Math.min(yOpen, yClose);
        const bodyH = Math.max(1.5, Math.abs(yClose - yOpen));
        ctx.fillRect(x - barW / 2, topY, barW, bodyH);
      });

      // Draw VWAP Line
      if (vwapPoints.length > 1) {
        ctx.strokeStyle = "#818cf8";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        vwapPoints.forEach((pt, idx) => {
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
      }

      // Helper for horizontal reference lines (Entry, Stop Loss, Target)
      function drawLevelLine(priceVal, color, label) {
        if (!priceVal || priceVal <= 0) return;
        const y = yOf(priceVal);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(W - padRight, y);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = color;
        ctx.font = "bold 10px monospace";
        const decimals = priceVal < 10 ? 4 : 1;
        ctx.fillText(`${label} ${Number(priceVal).toFixed(decimals)}`, W - padRight + 4, y + 3);
      }

      drawLevelLine(targetPrice, "#34d399", "TGT");
      drawLevelLine(entryPrice, "#e2e8f0", "NOW");
      drawLevelLine(stopPrice, "#fb7185", "STP");
    }

    async function refreshActiveInstrument() {
      try {
        const stResp = await fetch(`/market-state/${encodeURIComponent(currentSymbol)}?timeframe=${encodeURIComponent(currentTimeframe)}`);
        const stData = stResp.ok ? await stResp.json() : {};

        const sigResp = await fetch("/signals/evaluate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbol: currentSymbol,
            current_price: stData.last_price || 0.0,
            atr: stData.atr_14 || 0.0,
            timeframe: currentTimeframe,
            horizon: currentHorizon
          })
        });
        const sigData = await sigResp.json();

        renderMainDashboard({
          status: stData.status || "OK",
          symbol: stData.symbol || currentSymbol.split(":").pop(),
          canonical_symbol: stData.canonical_symbol || currentSymbol,
          market: stData.market || currentSymbol.split(":")[0] || "NSE",
          currency: stData.currency || "INR",
          company_name: (stData.metadata && stData.metadata.name) || stData.symbol || currentSymbol,
          is_market_open: stData.is_market_open ?? true,
          is_stale: stData.is_stale ?? false,
          last_price: stData.last_price || sigData.entry_price,
          decision: sigData,
          developing_day: stData.developing_day || (sigData.evidence && sigData.evidence.developing_day),
          mtf_context: stData.mtf_context || (sigData.evidence && sigData.evidence.mtf_context),
          candles: (sigData.evidence && sigData.evidence.candles_intraday) || []
        });
      } catch (e) {}
    }

    function connectWebSocket() {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      wsConnection = new WebSocket(`${proto}//${window.location.host}/ws/stream`);
      wsConnection.onopen = () => {
        subscribeWebSocketLive();
      };
      wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.topic === "live_prediction") {
            renderMainDashboard(data);
          }
        } catch (e) {}
      };
      wsConnection.onclose = () => {
        setTimeout(connectWebSocket, 3000);
      };
    }

    function subscribeWebSocketLive() {
      if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({
          action: "live_subscribe",
          symbol: currentSymbol,
          timeframe: currentTimeframe,
          horizon: currentHorizon
        }));
      }
    }

    async function updateLedgerAndGovernance() {
      try {
        const resp = await fetch("/learning/ledger/status");
        if (!resp.ok) return;
        const l = await resp.json();
        document.getElementById("advLedgerAcc").innerText = l.accuracy_display || "N/A (0 Resolved)";
        document.getElementById("advPrecRec").innerText =
          `${l.precision_pct !== null ? l.precision_pct + '%' : 'N/A'} / ${l.recall_pct !== null ? l.recall_pct + '%' : 'N/A'}`;
        document.getElementById("advF1Brier").innerText =
          `${l.f1_score_pct !== null ? l.f1_score_pct + '%' : 'N/A'} / ${l.mean_brier_score !== null ? l.mean_brier_score : 'N/A'}`;
        document.getElementById("advExpPf").innerText =
          `${l.expectancy_pct >= 0 ? '+' : ''}${l.expectancy_pct}% / PF ${l.profit_factor ?? 'N/A'}`;
        document.getElementById("advSharpeMfe").innerText =
          `${l.sharpe_ratio ?? 'N/A'} / +${l.mean_mfe_pct}% / -${l.mean_mae_pct}%`;

        if (l.self_development && l.self_development.status === "COMPLETED") {
          const sd = l.self_development;
          const vp = sd.validation_protocol || {};
          document.getElementById("advGovernanceBox").innerHTML =
            `<div><span class="text-indigo-400 font-bold">Active Champion:</span> ${sd.active_production_model} | <span class="text-emerald-400 font-bold">Verdict:</span> ${sd.governance_verdict}</div>` +
            `<div class="text-slate-400 mt-1">Protocol: ${vp.method} (Train: ${vp.train_bars}b, Embargo: ${vp.embargo_purge_bars}b, OOS: ${vp.out_of_sample_bars}b) — ${sd.governance_reason}</div>`;
        }
      } catch (e) {}
    }

    async function triggerSelfDevelopment() {
      try {
        document.getElementById("advGovernanceBox").innerText = "Running Purged Walk-Forward Challenger Validation...";
        await fetch("/learning/self-development/run?symbol=" + encodeURIComponent(currentSymbol), { method: "POST" });
        await updateLedgerAndGovernance();
      } catch (e) {}
    }

    async function runScanner() {
      const tbody = document.getElementById("scannerTableBody");
      const mkt = document.getElementById("marketSelect").value;
      const qParam = (mkt && mkt !== "ALL") ? `&market=${encodeURIComponent(mkt)}` : "";
      try {
        const resp = await fetch(`/scanner/opportunities?limit=6${qParam}`);
        const data = await resp.json();
        if (data.top_opportunities && data.top_opportunities.length > 0) {
          tbody.innerHTML = data.top_opportunities.map(opp => `
            <tr class="hover:bg-slate-900 cursor-pointer" onclick="selectSymbol('${opp.canonical_symbol}')">
              <td class="p-2 text-slate-400">#${opp.rank}</td>
              <td class="p-2 font-bold text-white">${opp.symbol} <span class="text-slate-500 font-normal">(${opp.market})</span></td>
              <td class="p-2 font-bold ${opp.action === 'BUY' ? 'text-emerald-400' : (opp.action === 'SELL' ? 'text-rose-400' : 'text-amber-400')}">${opp.action}</td>
              <td class="p-2 text-indigo-400">${Math.round(opp.confidence * 100)}%</td>
              <td class="p-2 text-white">${opp.current_price.toLocaleString()}</td>
              <td class="p-2 text-emerald-400">${opp.target_price.toLocaleString()}</td>
              <td class="p-2 text-rose-400">${opp.stop_price.toLocaleString()}</td>
            </tr>
          `).join("");
        } else {
          tbody.innerHTML = `<tr><td colspan="7" class="p-3 text-center text-slate-500">No high-conviction setups above threshold right now</td></tr>`;
        }
      } catch (e) {}
    }

    window.addEventListener("resize", () => {
      if (latestCandles && latestCandles.length > 0) {
        refreshActiveInstrument();
      }
    });

    window.onload = function() {
      const params = new URLSearchParams(window.location.search);
      if (params.get("symbol")) {
        currentSymbol = params.get("symbol").trim().toUpperCase();
        document.getElementById("symbolInput").value = currentSymbol;
      }
      if (params.get("horizon")) {
        currentHorizon = params.get("horizon").trim();
        const hs = document.getElementById("horizonSelect");
        if (hs) hs.value = currentHorizon;
      }
      refreshActiveInstrument();
      connectWebSocket();
      updateLedgerAndGovernance();
      runScanner();
    };
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.get("/scanner", response_class=HTMLResponse, tags=["Observability & UI"])
async def market_scanner_page_endpoint() -> HTMLResponse:
    """Dedicated Multi-Market Scanner Page covering NSE, BSE, Quotex, Forex, and Crypto (Section 16)."""
    scanner_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Market Scanner — Trading Intelligence Platform</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased py-6 px-4 md:px-8">
  <div class="max-w-6xl mx-auto space-y-6">

    <!-- Header & Navigation Back to Beginner Dashboard -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-900 border border-slate-800 p-5 rounded-2xl shadow-lg">
      <div>
        <h1 class="text-2xl font-extrabold text-white tracking-tight">Multi-Market Universe Scanner</h1>
        <p id="universeSummaryBar" class="text-xs text-slate-400 mt-1 font-mono">
          Synchronizing complete NSE, BSE, Quotex, Forex & Crypto universe...
        </p>
      </div>
      <div class="flex items-center gap-3">
        <button onclick="executeFullScan()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl transition">
          Refresh Live Scan
        </button>
        <a href="/dashboard" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-bold rounded-xl transition">
          ← Back to Simple Dashboard
        </a>
      </div>
    </div>

    <!-- Search, Exchange Filter, Signal Filter, Confidence Filter, Horizon Filter & Sort Controls -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3 bg-slate-900/90 border border-slate-800 p-4 rounded-2xl">
      <div class="lg:col-span-2">
        <label class="block text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Search Instrument / ISIN / Scrip</label>
        <input
          id="scanSearchInput"
          type="text"
          placeholder="e.g. TCS, RELIANCE, 500325, EURUSD, MSFT_OTC..."
          onkeydown="if(event.key==='Enter'){executeFullScan();}"
          class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
        />
      </div>

      <div>
        <label class="block text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Exchange Filter</label>
        <select id="scanExchangeFilter" onchange="executeFullScan()" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white">
          <option value="ALL">ALL EXCHANGES</option>
          <option value="NSE">NSE (India)</option>
          <option value="BSE">BSE (India)</option>
          <option value="QUOTEX">QUOTEX</option>
          <option value="FOREX">FOREX</option>
          <option value="CRYPTO">CRYPTO</option>
        </select>
      </div>

      <div>
        <label class="block text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Signal Filter</label>
        <select id="scanSignalFilter" onchange="executeFullScan()" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white">
          <option value="ALL">ALL SIGNALS (BUY / SELL / NO-TRADE)</option>
          <option value="BUY">BUY ONLY</option>
          <option value="SELL">SELL ONLY</option>
          <option value="NO_TRADE">NO-TRADE ONLY</option>
        </select>
      </div>

      <div>
        <label class="block text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Min Confidence</label>
        <select id="scanConfFilter" onchange="executeFullScan()" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white">
          <option value="0.0" selected>All Confidence (0%+)</option>
          <option value="0.50">50%+ Confidence</option>
          <option value="0.60">60%+ Confidence</option>
          <option value="0.70">70%+ Confidence</option>
          <option value="0.80">80%+ Confidence</option>
        </select>
      </div>

      <div class="grid grid-cols-2 gap-2">
        <div>
          <label class="block text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Horizon</label>
          <select id="scanHorizonFilter" onchange="executeFullScan()" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-2 py-2 text-xs text-white">
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="1h" selected>1h</option>
            <option value="4h">4h</option>
            <option value="1d">1d</option>
          </select>
        </div>
        <div>
          <label class="block text-[11px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Sort By</label>
          <select id="scanSortBy" onchange="executeFullScan()" class="w-full bg-slate-950 border border-slate-700 rounded-xl px-2 py-2 text-xs text-white">
            <option value="rank">Rank</option>
            <option value="confidence">Confidence</option>
            <option value="expected_return">Exp. Move</option>
            <option value="price">Price</option>
            <option value="symbol">Symbol</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Scanner Results Table: Exchange | Instrument | Price | Signal | Confidence | Horizon -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs font-mono">
          <thead class="bg-slate-950 text-slate-400 uppercase border-b border-slate-800">
            <tr>
              <th class="py-3.5 px-4">Exchange</th>
              <th class="py-3.5 px-4">Instrument</th>
              <th class="py-3.5 px-4">Price</th>
              <th class="py-3.5 px-4">Signal</th>
              <th class="py-3.5 px-4">Confidence</th>
              <th class="py-3.5 px-4">Horizon</th>
              <th class="py-3.5 px-4">Expected Move</th>
              <th class="py-3.5 px-4">Data / Model Status</th>
            </tr>
          </thead>
          <tbody id="fullScannerTableBody" class="divide-y divide-slate-800/70 text-slate-200">
            <tr>
              <td colspan="8" class="p-8 text-center text-slate-500">Scanning live multi-market universe...</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    async function loadUniverseSummary() {
      try {
        const resp = await fetch("/universe/summary");
        if (!resp.ok) return;
        const s = await resp.json();
        const m = s.by_market || {};
        document.getElementById("universeSummaryBar").innerText =
          `Total Universe: ${(s.total_instruments || 0).toLocaleString()} Instruments | ` +
          `NSE: ${(m.NSE || 0).toLocaleString()} | BSE: ${(m.BSE || 0).toLocaleString()} | ` +
          `Quotex: ${m.QUOTEX || 0} | Forex: ${m.FOREX || 0} | Crypto: ${m.CRYPTO || 0} | ` +
          `Cross-Listed ISINs: ${(s.cross_listed_companies || 0).toLocaleString()}`;
      } catch (e) {}
    }

    async function executeFullScan() {
      const tbody = document.getElementById("fullScannerTableBody");
      const q = document.getElementById("scanSearchInput").value.trim();
      const exch = document.getElementById("scanExchangeFilter").value;
      const sig = document.getElementById("scanSignalFilter").value;
      const conf = document.getElementById("scanConfFilter").value;
      const hor = document.getElementById("scanHorizonFilter").value;
      const sort = document.getElementById("scanSortBy").value;

      const params = new URLSearchParams({
        market: exch,
        min_confidence: conf,
        horizon: hor,
        limit: "100",
        sort_by: sort,
        include_no_trade: "true"
      });
      if (q) params.set("q", q);
      if (sig && sig !== "ALL") params.set("signal", sig);

      try {
        const resp = await fetch("/scanner/opportunities?" + params.toString());
        const data = await resp.json();
        const rows = (data.top_opportunities && data.top_opportunities.length > 0)
          ? data.top_opportunities
          : (data.all_evaluated || []);

        if (!rows || rows.length === 0) {
          tbody.innerHTML = `<tr><td colspan="8" class="p-8 text-center text-slate-500">No matching instruments found for current filters.</td></tr>`;
          return;
        }

        tbody.innerHTML = rows.map(item => {
          const act = (item.action || "NO_TRADE").replace("_", "-");
          const badgeClass = act === "BUY"
            ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
            : (act === "SELL" ? "bg-rose-500/15 text-rose-400 border-rose-500/30" : "bg-amber-500/15 text-amber-400 border-amber-500/30");
          const priceStr = item.current_price > 0
            ? item.current_price.toLocaleString(undefined, { minimumFractionDigits: item.current_price < 10 ? 4 : 2, maximumFractionDigits: item.current_price < 10 ? 4 : 2 })
            : "LIVE DATA UNAVAILABLE";
          const confPct = Math.round(Number(item.confidence || 0) * 100);
          const expRet = Number(item.expected_return_pct || 0);
          const expStr = item.current_price > 0 ? `${expRet >= 0 ? '+' : ''}${expRet.toFixed(2)}%` : "--";
          const dqText = item.data_quality === "QUOTEX_LIVE_DATA_UNAVAILABLE"
            ? "Quotex live data unavailable"
            : `${item.data_quality || 'PASSED'} • ${item.model_quality || 'ENSEMBLE'}`;

          return `
            <tr
              onclick="window.location.href='/dashboard?symbol=${encodeURIComponent(item.canonical_symbol)}&horizon=${encodeURIComponent(item.forecast_horizon)}'"
              class="hover:bg-slate-800/70 cursor-pointer transition"
            >
              <td class="py-3 px-4 font-bold text-indigo-400">${item.market}</td>
              <td class="py-3 px-4">
                <div class="font-bold text-white">${item.symbol}</div>
                <div class="text-[11px] text-slate-400 truncate max-w-xs">${item.company_name || item.canonical_symbol}</div>
              </td>
              <td class="py-3 px-4 font-bold ${item.current_price > 0 ? 'text-white' : 'text-rose-400'}">${priceStr}</td>
              <td class="py-3 px-4">
                <span class="px-2.5 py-1 rounded-full text-[11px] font-bold border ${badgeClass}">${act}</span>
              </td>
              <td class="py-3 px-4 font-bold text-slate-200">${confPct}%</td>
              <td class="py-3 px-4 text-slate-300">${item.forecast_horizon}</td>
              <td class="py-3 px-4 ${expRet >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${expStr}</td>
              <td class="py-3 px-4 text-[11px] ${item.data_quality === 'QUOTEX_LIVE_DATA_UNAVAILABLE' ? 'text-rose-400 font-semibold' : 'text-slate-400'}">${dqText}</td>
            </tr>
          `;
        }).join("");
      } catch (e) {
        tbody.innerHTML = `<tr><td colspan="8" class="p-8 text-center text-rose-400">Error loading scanner results.</td></tr>`;
      }
    }

    window.onload = function() {
      loadUniverseSummary();
      executeFullScan();
    };
  </script>
</body>
</html>
"""
    return HTMLResponse(content=scanner_html)


