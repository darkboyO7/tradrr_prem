"""Trading Intelligence Platform - Execution Package."""

from execution.algo import (
    AlgoEngine,
    AlgoExecutionPlan,
    AlgoExecutionReport,
    AlgoOrderType,
    AlgoSliceSchedule,
)
from execution.base import BaseExecutionClient
from execution.binance_rest import BinanceRESTExecutionClient
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
    CrossingFill,
    CrossingPoolMetrics,
    CrossingPriority,
    CrossingResult,
    CrossingSide,
    CrossingTimeInForce,
    InternalCrossingEngine,
    InternalCrossingOrder,
)
from execution.hedging import (
    HedgeInstrument,
    HedgeOrder,
    HedgeTriggerType,
    SmartHedgeConfig,
    SmartHedgePlan,
    SmartHedgingEngine,
)
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
from execution.reconciliation import (
    PositionDiscrepancy,
    PositionReconciler,
    ReconciliationReport,
)
from execution.router import (
    ChildOrderSlice,
    RoutingPlan,
    RoutingPolicy,
    SmartOrderRouter,
)
from execution.simulated import SimulatedExecutionClient
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
    OrderBenchmarkType,
    TCAPortfolioReport,
    TCAStatisticalDistribution,
    TradeExecutionRecord,
    TransactionCostAnalysisEngine,
    VenueExecutionScorecard,
)
from execution.volume_profile import IntradayVolumeProfile, VolumeProfiler

__all__ = [
    "BaseExecutionClient",
    "SimulatedExecutionClient",
    "BinanceRESTExecutionClient",
    "PositionDiscrepancy",
    "ReconciliationReport",
    "PositionReconciler",
    "SmartOrderRouter",
    "RoutingPolicy",
    "ChildOrderSlice",
    "RoutingPlan",
    "AlmgrenChrissCostModel",
    "ExecutionCostEstimate",
    "AlgoEngine",
    "AlgoOrderType",
    "AlgoExecutionPlan",
    "AlgoExecutionReport",
    "AlgoSliceSchedule",
    "VolumeProfiler",
    "IntradayVolumeProfile",
    "HedgeInstrument",
    "HedgeTriggerType",
    "HedgeOrder",
    "SmartHedgeConfig",
    "SmartHedgePlan",
    "SmartHedgingEngine",
    "CrossingPriority",
    "CrossingTimeInForce",
    "CrossingSide",
    "InternalCrossingOrder",
    "CrossingFill",
    "CrossingResult",
    "CrossingPoolMetrics",
    "InternalCrossingEngine",
    "DMMConfig",
    "DMMQuoteLevel",
    "DMMQuoteSurface",
    "DynamicMarketMakerEngine",
    "InventoryState",
    "OrderBenchmarkType",
    "TradeExecutionRecord",
    "ImplementationShortfallDecomposition",
    "VenueExecutionScorecard",
    "TCAStatisticalDistribution",
    "TCAPortfolioReport",
    "TransactionCostAnalysisEngine",
    "CalibratedCostParameters",
    "CostSurfacePoint",
    "CostSurfaceGrid",
    "SlippagePredictionRequest",
    "SlippagePredictionResult",
    "CostSurfaceCalibrator",
    "SyntheticSmartOrderRouter",
    "VenueBookDepth",
    "DynamicLiquidityRoutingPlan",
    "SyntheticChildAllocation",
    "SOROptimizationObjective",
    "VenueLiquidityType",
    "ASQuotingRegime",
    "AdaptiveMMConfig",
    "AdaptiveQuoteTier",
    "AdaptiveMMQuotingReport",
    "AdaptiveMarketMakingOptimizer",
]

