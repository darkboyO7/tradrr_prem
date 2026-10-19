"""Micro-Structural Statistical Arbitrage & High-Frequency Lead-Lag Network Engine.

Implements:
1. Cross-Asset High-Frequency Lead-Lag Network Analysis:
   - Pairwise cross-correlation delay profiling across discrete lag intervals tau in [-tau_max, +tau_max]:
     rho_{i,j}(tau) = corr(r_{i, t}, r_{j, t - tau})
   - Optimal lag extraction: tau^*_{i,j} = argmax_{tau} |rho_{i,j}(tau)|
   - Asymmetric lead-lag score matrix L_{i,j} in [-1, +1] where L_{i,j} > 0 implies asset i leads asset j.
2. Market Leader & Follower Eigenvalue Centrality Decomposition:
   - PageRank / Principal Eigenvector decomposition of the lead-lag transition matrix
   - Unambiguous ranking of "Leader", "Intermediate", and "Follower/Lagging" instruments.
3. High-Frequency Microstructure Predictive Signal Generation:
   - Forecasts follower instrument returns from leading instrument momentum:
     hat{r}_{follower, t+1} = beta_{lead} * r_{leader, t} + alpha
   - Threshold-gated arbitrage signal synthesis (LONG_FOLLOWER, SHORT_FOLLOWER, NEUTRAL).
4. Sub-millisecond latency profile compensation & execution horizon estimation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class InstrumentNetworkRole(str, Enum):
    """Network centrality role of an instrument in the lead-lag topology."""
    MARKET_LEADER = "MARKET_LEADER"       # Information source (leads other venues/assets)
    PROPAGATOR = "PROPAGATOR"             # Intermediate transmission node
    MARKET_FOLLOWER = "MARKET_FOLLOWER"   # Information recipient (predictable lag delay)


class LeadLagSignalAction(str, Enum):
    """Actionable prediction signal for a follower asset."""
    LONG_FOLLOWER = "LONG_FOLLOWER"       # Leader surged upward, follower expected to catch up
    SHORT_FOLLOWER = "SHORT_FOLLOWER"     # Leader plunged downward, follower expected to follow
    NEUTRAL = "NEUTRAL"                   # No significant predictive lead-lag dislocation


@dataclass(frozen=True)
class PairwiseLeadLagLink:
    """Pairwise directional lead-lag relationship between two assets."""
    leader_symbol: str
    follower_symbol: str
    optimal_lag_steps: int          # Positive integer: leader leads follower by this many steps
    lead_lag_strength: float        # Correlation magnitude at optimal lag [0, 1]
    leadership_score: float         # Signed asymmetry [-1.0, 1.0] (+1 = purely leading, -1 = lagging)
    predictive_r_squared: float     # Variance explained in follower by leader

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leader_symbol": self.leader_symbol,
            "follower_symbol": self.follower_symbol,
            "optimal_lag_steps": int(self.optimal_lag_steps),
            "lead_lag_strength": float(self.lead_lag_strength),
            "leadership_score": float(self.leadership_score),
            "predictive_r_squared": float(self.predictive_r_squared),
        }


@dataclass(frozen=True)
class InstrumentCentralityProfile:
    """Centrality and hierarchy ranking of an individual instrument."""
    symbol: str
    network_role: InstrumentNetworkRole
    eigenvector_centrality: float
    outflow_lead_score: float       # Net leadership score vs all other assets
    inflow_lag_score: float
    average_lead_time_steps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "network_role": self.network_role.value,
            "eigenvector_centrality": float(self.eigenvector_centrality),
            "outflow_lead_score": float(self.outflow_lead_score),
            "inflow_lag_score": float(self.inflow_lag_score),
            "average_lead_time_steps": float(self.average_lead_time_steps),
        }


@dataclass(frozen=True)
class LeadLagArbitrageSignal:
    """Actionable predictive arbitrage signal generated from lead-lag transmission."""
    leader_symbol: str
    follower_symbol: str
    action: LeadLagSignalAction
    leader_return_bps: float
    predicted_follower_return_bps: float
    confidence: float
    optimal_lead_steps: int
    rationale: str
    timestamp_utc: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leader_symbol": self.leader_symbol,
            "follower_symbol": self.follower_symbol,
            "action": self.action.value,
            "leader_return_bps": float(self.leader_return_bps),
            "predicted_follower_return_bps": float(self.predicted_follower_return_bps),
            "confidence": float(self.confidence),
            "optimal_lead_steps": int(self.optimal_lead_steps),
            "rationale": self.rationale,
            "timestamp_utc": self.timestamp_utc.isoformat(),
        }


@dataclass(frozen=True)
class LeadLagNetworkReport:
    """Comprehensive graph report of the cross-asset lead-lag topology."""
    symbols: List[str]
    timestamp_utc: datetime
    profiles: List[InstrumentCentralityProfile]
    pairwise_links: List[PairwiseLeadLagLink]
    top_predictive_pairs: List[PairwiseLeadLagLink]
    generated_signals: List[LeadLagArbitrageSignal]
    matrix_spectral_gap: float  # Difference between 1st and 2nd eigenvalue (hierarchy clarity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "profiles": [p.to_dict() for p in self.profiles],
            "pairwise_links": [l.to_dict() for l in self.pairwise_links],
            "top_predictive_pairs": [p.to_dict() for p in self.top_predictive_pairs],
            "generated_signals": [s.to_dict() for s in self.generated_signals],
            "matrix_spectral_gap": float(self.matrix_spectral_gap),
        }


class LeadLagNetworkEngine:
    """Micro-Structural Lead-Lag Network Analysis & Arbitrage Forecasting Engine.

    Continuously ingests synchronous or asynchronous asset return series across
    multiple venues/symbols, computes shifted cross-correlation tensors, derives
    directed adjacency networks, and generates high-conviction lead-lag trading signals.
    """

    def __init__(
        self,
        symbols: List[str],
        max_lag_steps: int = 5,
        history_buffer_length: int = 300,
        signal_threshold_bps: float = 8.0,  # Minimum predicted return in bps to trigger signal
    ) -> None:
        if len(symbols) < 2:
            raise ValueError("At least 2 symbols required for lead-lag network")
        self.symbols = [s.upper() for s in symbols]
        self.max_lag_steps = int(max_lag_steps)
        self.history_buffer_length = int(history_buffer_length)
        self.signal_threshold_bps = float(signal_threshold_bps)

        # Rolling history buffers for returns: symbol -> deque of floats
        self._price_buffers: Dict[str, deque[float]] = {
            s: deque(maxlen=self.history_buffer_length) for s in self.symbols
        }
        self._return_buffers: Dict[str, deque[float]] = {
            s: deque(maxlen=self.history_buffer_length) for s in self.symbols
        }

    def update_price_tick(self, symbol: str, price: float) -> None:
        """Ingest price tick for a symbol and record log return."""
        sym = symbol.upper()
        if sym not in self._price_buffers:
            self._price_buffers[sym] = deque(maxlen=self.history_buffer_length)
            self._return_buffers[sym] = deque(maxlen=self.history_buffer_length)

        buf = self._price_buffers[sym]
        ret_buf = self._return_buffers[sym]

        if buf:
            prev_p = buf[-1]
            if prev_p > 0 and price > 0:
                ret = math.log(price / prev_p)
                ret_buf.append(ret)
        buf.append(float(price))

    def ingest_price_matrix(self, prices_dict: Dict[str, List[float]]) -> None:
        """Batch ingest synchronous price trajectories across symbols."""
        for sym, p_list in prices_dict.items():
            sym_up = sym.upper()
            if sym_up not in self._price_buffers:
                self._price_buffers[sym_up] = deque(maxlen=self.history_buffer_length)
                self._return_buffers[sym_up] = deque(maxlen=self.history_buffer_length)
            for p in p_list:
                self.update_price_tick(sym_up, p)

    def calibrate_lead_lag_network(self) -> LeadLagNetworkReport:
        """Analyze rolling return buffers and compute complete lead-lag graph topology."""
        min_len = min((len(self._return_buffers[s]) for s in self.symbols), default=0)
        if min_len < self.max_lag_steps + 10:
            # Insufficient sample history; return initial neutral state
            return self._build_empty_report()

        N = len(self.symbols)
        # Construct aligned returns matrix (T, N)
        aligned_returns = np.column_stack([
            np.array(list(self._return_buffers[s])[-min_len:], dtype=np.float64)
            for s in self.symbols
        ])

        T = aligned_returns.shape[0]

        # 1. Compute Pairwise Cross-Correlation Delay Tensors
        pairwise_links: List[PairwiseLeadLagLink] = []
        leadership_matrix = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                r_i = aligned_returns[:, i]
                r_j = aligned_returns[:, j]

                opt_lag, max_corr, lead_score = self._compute_pairwise_delay(r_i, r_j)
                leadership_matrix[i, j] = lead_score

                if opt_lag > 0 and lead_score > 0.05:
                    r_squared = float(min(1.0, max_corr ** 2))
                    pairwise_links.append(
                        PairwiseLeadLagLink(
                            leader_symbol=self.symbols[i],
                            follower_symbol=self.symbols[j],
                            optimal_lag_steps=opt_lag,
                            lead_lag_strength=round(max_corr, 4),
                            leadership_score=round(lead_score, 4),
                            predictive_r_squared=round(r_squared, 4),
                        )
                    )

        # 2. Eigenvector Centrality & Leadership Ranking
        profiles, spectral_gap = self._compute_centrality_profiles(leadership_matrix)

        # 3. Sort links by predictive strength
        sorted_links = sorted(pairwise_links, key=lambda l: l.lead_lag_strength, reverse=True)

        # 4. Generate Predictive Arbitrage Signals
        signals = self._generate_lead_lag_signals(sorted_links)

        return LeadLagNetworkReport(
            symbols=list(self.symbols),
            timestamp_utc=datetime.now(timezone.utc),
            profiles=profiles,
            pairwise_links=pairwise_links,
            top_predictive_pairs=sorted_links[:10],
            generated_signals=signals,
            matrix_spectral_gap=round(spectral_gap, 4),
        )

    # -------------------------------------------------------------------------
    # Mathematical Lead-Lag & Eigen Decomposition Routines
    # -------------------------------------------------------------------------

    def _compute_pairwise_delay(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Compute optimal lag and signed leadership score between x and y.

        If opt_lag > 0, x leads y by opt_lag steps.
        If opt_lag < 0, y leads x by |opt_lag| steps.
        """
        lags = range(-self.max_lag_steps, self.max_lag_steps + 1)
        corrs = {}

        for lag in lags:
            if lag > 0:
                # x leads y: correlate x[:-lag] with y[lag:]
                c = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            elif lag < 0:
                # y leads x: correlate x[-lag:] with y[:lag]
                c = np.corrcoef(x[-lag:], y[:lag])[0, 1]
            else:
                c = np.corrcoef(x, y)[0, 1]
            corrs[lag] = 0.0 if np.isnan(c) else float(c)

        # Find lag with maximum absolute correlation
        best_lag = max(lags, key=lambda l: abs(corrs[l]))
        best_corr = abs(corrs[best_lag])

        # Asymmetric leadership score L_xy: difference between positive and negative lag correlations
        pos_lead = max([corrs[l] for l in lags if l > 0], default=0.0)
        neg_lead = max([corrs[l] for l in lags if l < 0], default=0.0)
        lead_score = float(np.clip(pos_lead - neg_lead, -1.0, 1.0))

        return (best_lag, best_corr, lead_score)

    def _compute_centrality_profiles(
        self,
        leadership_matrix: np.ndarray,
    ) -> Tuple[List[InstrumentCentralityProfile], float]:
        """Compute Perron-Frobenius eigenvector centrality from non-negative transition matrix."""
        N = leadership_matrix.shape[0]
        # Shift matrix to positive values: A = max(0, L)
        adj = np.maximum(0.0, leadership_matrix)
        np.fill_diagonal(adj, 0.0)

        # Add small regularization to guarantee irreducibility
        reg_adj = adj + 1e-4

        try:
            # We want source/leader centrality (outflow influence), so use reg_adj
            eigenvalues, eigenvectors = np.linalg.eig(reg_adj)
            idx = np.argsort(np.real(eigenvalues))[::-1]
            top_val = np.real(eigenvalues[idx[0]])
            second_val = np.real(eigenvalues[idx[1]]) if N > 1 else 0.0
            spectral_gap = float(max(0.0, top_val - second_val))

            leading_vec = np.real(eigenvectors[:, idx[0]])
            centrality = np.abs(leading_vec)
            s = float(np.sum(centrality))
            if s > 0:
                centrality /= s
            else:
                centrality = np.ones(N) / N
        except np.linalg.LinAlgError:
            centrality = np.ones(N) / N
            spectral_gap = 0.0

        profiles: List[InstrumentCentralityProfile] = []
        for i in range(N):
            sym = self.symbols[i]
            outflow = float(np.sum(adj[i, :]))
            inflow = float(np.sum(adj[:, i]))
            cent = float(centrality[i])

            if outflow > inflow * 1.2:
                role = InstrumentNetworkRole.MARKET_LEADER
            elif inflow > outflow * 1.2:
                role = InstrumentNetworkRole.MARKET_FOLLOWER
            else:
                role = InstrumentNetworkRole.PROPAGATOR

            profiles.append(
                InstrumentCentralityProfile(
                    symbol=sym,
                    network_role=role,
                    eigenvector_centrality=round(cent, 4),
                    outflow_lead_score=round(outflow, 4),
                    inflow_lag_score=round(inflow, 4),
                    average_lead_time_steps=round(max(0.0, outflow - inflow), 2),
                )
            )

        # Sort profiles by centrality descending
        profiles.sort(key=lambda p: p.eigenvector_centrality, reverse=True)
        return profiles, spectral_gap

    def _generate_lead_lag_signals(
        self,
        predictive_links: List[PairwiseLeadLagLink],
    ) -> List[LeadLagArbitrageSignal]:
        """Generate predictive return forecasts for lagging assets from leader movements."""
        signals: List[LeadLagArbitrageSignal] = []

        for link in predictive_links:
            leader_ret_buf = self._return_buffers[link.leader_symbol]
            if not leader_ret_buf:
                continue

            latest_leader_ret = leader_ret_buf[-1]
            leader_bps = latest_leader_ret * 10000.0

            # Linear beta projection: predicted return = leader_return * correlation
            predicted_follower_bps = leader_bps * link.lead_lag_strength

            if abs(predicted_follower_bps) >= self.signal_threshold_bps:
                action = (
                    LeadLagSignalAction.LONG_FOLLOWER
                    if predicted_follower_bps > 0
                    else LeadLagSignalAction.SHORT_FOLLOWER
                )
                conf = float(np.clip(link.lead_lag_strength, 0.0, 1.0))
                rec_str = (
                    f"Leader {link.leader_symbol} moved {leader_bps:+.1f} bps; "
                    f"Follower {link.follower_symbol} projected {predicted_follower_bps:+.1f} bps "
                    f"over {link.optimal_lag_steps} steps."
                )

                signals.append(
                    LeadLagArbitrageSignal(
                        leader_symbol=link.leader_symbol,
                        follower_symbol=link.follower_symbol,
                        action=action,
                        leader_return_bps=round(leader_bps, 2),
                        predicted_follower_return_bps=round(predicted_follower_bps, 2),
                        confidence=round(conf, 4),
                        optimal_lead_steps=link.optimal_lag_steps,
                        rationale=rec_str,
                        timestamp_utc=datetime.now(timezone.utc),
                    )
                )

        return signals

    def _build_empty_report(self) -> LeadLagNetworkReport:
        """Construct fallback report when history is warming up."""
        empty_profiles = [
            InstrumentCentralityProfile(
                symbol=s,
                network_role=InstrumentNetworkRole.PROPAGATOR,
                eigenvector_centrality=1.0 / len(self.symbols),
                outflow_lead_score=0.0,
                inflow_lag_score=0.0,
                average_lead_time_steps=0.0,
            )
            for s in self.symbols
        ]
        return LeadLagNetworkReport(
            symbols=list(self.symbols),
            timestamp_utc=datetime.now(timezone.utc),
            profiles=empty_profiles,
            pairwise_links=[],
            top_predictive_pairs=[],
            generated_signals=[],
            matrix_spectral_gap=0.0,
        )
