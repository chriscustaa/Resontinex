# Quorum Voting Systems - Technical Implementation Guide

## Overview

The RESONTINEX Quorum Voting System implements a sophisticated distributed decision-making framework for module consensus and conflict resolution. This system ensures reliable governance through weighted voting, multi-layered arbitration, and timeout-based emergency protocols.

## Architecture Overview

### Voting Power Distribution (v2.1.0)

```python
# Current voting power allocation
VOTING_POWER_MAP = {
    "EnergyLedger": {
        "votes": 2,
        "priority": 1,
        "description": "Budget and cost governance"
    },
    "EntropyAuditor": {
        "votes": 2,
        "priority": 2,
        "description": "Drift and quality control"
    },
    "TrustManager": {
        "votes": 2,
        "priority": 3,
        "description": "Alignment and trust scoring"
    },
    "ContinuityEngine": {
        "votes": 1,
        "priority": 4,
        "description": "State and memory preservation"
    },
    "InsightCollapser": {
        "votes": 1,
        "priority": 5,
        "description": "Reasoning compression"
    }
}

# Quorum requirement thresholds
QUORUM_REQUIREMENTS = {
    "simple_decision": 3,        # 3+ aligned votes
    "budget_override": 4,        # 4+ votes + EnergyLedger approval
    "emergency_escalation": 150  # 150ms timeout limit
}
```

## Core Quorum Implementation

### QuorumVotingEngine Class

```python
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

class VoteType(Enum):
    """Types of votes in the quorum system."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"

class DecisionType(Enum):
    """Types of decisions requiring quorum."""
    SIMPLE = "simple_decision"
    BUDGET_OVERRIDE = "budget_override"
    EMERGENCY = "emergency_escalation"

@dataclass
class Vote:
    """Individual vote cast by a module."""
    module_id: str
    vote_type: VoteType
    weight: int
    priority: int
    reasoning: str
    timestamp: float
    metadata: Dict[str, any] = None

@dataclass
class QuorumDecision:
    """Final quorum decision result."""
    decision_id: str
    decision_type: DecisionType
    outcome: VoteType
    total_votes: int
    weighted_score: float
    participating_modules: List[str]
    execution_time_ms: float
    arbitration_level: int
    reasoning: str

class QuorumVotingEngine:
    """
    Production-grade quorum voting system with weighted consensus.
    
    Handles module voting, conflict resolution, and emergency arbitration
    with comprehensive timeout and fallback mechanisms.
    """
    
    def __init__(self, voting_power_map: Dict[str, Dict], requirements: Dict[str, int]):
        self.voting_power_map = voting_power_map
        self.requirements = requirements
        self.active_decisions: Dict[str, Dict] = {}
        self.decision_history: List[QuorumDecision] = []
        self.logger = logging.getLogger("quorum_voting")
        
        # Module callback registry
        self.module_callbacks: Dict[str, Callable] = {}
        
        # Performance metrics
        self.metrics = {
            'decisions_processed': 0,
            'consensus_achieved': 0,
            'escalations': 0,
            'timeouts': 0
        }
    
    def register_module(self, module_id: str, callback: Callable):
        """Register a module for voting participation."""
        if module_id not in self.voting_power_map:
            raise ValueError(f"Module {module_id} not in voting power map")
        
        self.module_callbacks[module_id] = callback
        self.logger.info(f"Registered module {module_id} for quorum voting")
    
    def initiate_vote(self, decision_id: str, decision_type: DecisionType, 
                     context: Dict, timeout_ms: int = None) -> QuorumDecision:
        """
        Initiate a quorum vote across registered modules.
        
        Args:
            decision_id: Unique identifier for this decision
            decision_type: Type of decision (simple, budget_override, emergency)
            context: Decision context and metadata
            timeout_ms: Custom timeout override
            
        Returns:
            QuorumDecision with final outcome and reasoning
        """
        start_time = time.time()
        
        # Determine timeout based on decision type
        if timeout_ms is None:
            timeout_ms = self._get_default_timeout(decision_type)
        
        # Initialize decision tracking
        decision_state = {
            'decision_id': decision_id,
            'decision_type': decision_type,
            'context': context,
            'votes': {},
            'start_time': start_time,
            'timeout_ms': timeout_ms,
            'status': 'collecting_votes'
        }
        
        self.active_decisions[decision_id] = decision_state
        
        try:
            # Collect votes from modules
            votes = self._collect_votes(decision_id, context, timeout_ms)
            
            # Process votes through arbitration ladder
            decision = self._process_votes(decision_id, votes, decision_type)
            
            # Update metrics
            self.metrics['decisions_processed'] += 1
            if decision.outcome in [VoteType.APPROVE, VoteType.REJECT]:
                self.metrics['consensus_achieved'] += 1
            
            # Store in history
            self.decision_history.append(decision)
            
            return decision
            
        except QuorumTimeoutException as e:
            self.metrics['timeouts'] += 1
            return self._handle_timeout(decision_id, e)
        
        finally:
            # Cleanup
            if decision_id in self.active_decisions:
                del self.active_decisions[decision_id]
    
    def _collect_votes(self, decision_id: str, context: Dict, timeout_ms: int) -> List[Vote]:
        """Collect votes from all registered modules within timeout."""
        votes = []
        threads = []
        vote_lock = threading.Lock()
        
        def collect_module_vote(module_id: str):
            """Collect vote from individual module."""
            try:
                if module_id not in self.module_callbacks:
                    return
                
                callback = self.module_callbacks[module_id]
                module_config = self.voting_power_map[module_id]
                
                # Call module's voting function
                vote_response = callback(decision_id, context)
                
                vote = Vote(
                    module_id=module_id,
                    vote_type=VoteType(vote_response['vote']),
                    weight=module_config['votes'],
                    priority=module_config['priority'],
                    reasoning=vote_response.get('reasoning', ''),
                    timestamp=time.time(),
                    metadata=vote_response.get('metadata', {})
                )
                
                with vote_lock:
                    votes.append(vote)
                    
            except Exception as e:
                self.logger.error(f"Error collecting vote from {module_id}: {e}")
        
        # Start vote collection threads
        for module_id in self.module_callbacks.keys():
            thread = threading.Thread(target=collect_module_vote, args=(module_id,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for completion or timeout
        timeout_seconds = timeout_ms / 1000.0
        for thread in threads:
            thread.join(timeout_seconds)
        
        # Check if we got enough votes
        if len(votes) == 0:
            raise QuorumTimeoutException(f"No votes received within {timeout_ms}ms")
        
        return votes
    
    def _process_votes(self, decision_id: str, votes: List[Vote], 
                      decision_type: DecisionType) -> QuorumDecision:
        """Process collected votes through arbitration ladder."""
        
        start_time = time.time()
        
        # Level 1: Module Quorum (primary resolution)
        consensus_result = self._attempt_module_consensus(votes, decision_type)
        if consensus_result['resolved']:
            return QuorumDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                outcome=consensus_result['outcome'],
                total_votes=len(votes),
                weighted_score=consensus_result['weighted_score'],
                participating_modules=[v.module_id for v in votes],
                execution_time_ms=int((time.time() - start_time) * 1000),
                arbitration_level=1,
                reasoning=consensus_result['reasoning']
            )
        
        # Level 2: Weighted Consensus (tie-breaking)
        weighted_result = self._attempt_weighted_consensus(votes)
        if weighted_result['resolved']:
            return QuorumDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                outcome=weighted_result['outcome'],
                total_votes=len(votes),
                weighted_score=weighted_result['weighted_score'],
                participating_modules=[v.module_id for v in votes],
                execution_time_ms=int((time.time() - start_time) * 1000),
                arbitration_level=2,
                reasoning=weighted_result['reasoning']
            )
        
        # Level 3: Energy Arbitration (cost-factor resolution)
        if decision_type == DecisionType.BUDGET_OVERRIDE:
            energy_result = self._attempt_energy_arbitration(votes)
            if energy_result['resolved']:
                return QuorumDecision(
                    decision_id=decision_id,
                    decision_type=decision_type,
                    outcome=energy_result['outcome'],
                    total_votes=len(votes),
                    weighted_score=energy_result['weighted_score'],
                    participating_modules=[v.module_id for v in votes],
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    arbitration_level=3,
                    reasoning=energy_result['reasoning']
                )
        
        # Level 4: External GateKit (meta-resolution layer)
        self.metrics['escalations'] += 1
        return self._escalate_to_gatekit(decision_id, votes, decision_type, start_time)
    
    def _attempt_module_consensus(self, votes: List[Vote], 
                                decision_type: DecisionType) -> Dict:
        """Attempt primary module consensus resolution."""
        
        # Count votes by type
        vote_counts = {
            VoteType.APPROVE: 0,
            VoteType.REJECT: 0,
            VoteType.ABSTAIN: 0,
            VoteType.ESCALATE: 0
        }
        
        weighted_scores = {
            VoteType.APPROVE: 0.0,
            VoteType.REJECT: 0.0,
            VoteType.ESCALATE: 0.0
        }
        
        for vote in votes:
            vote_counts[vote.vote_type] += 1
            if vote.vote_type != VoteType.ABSTAIN:
                weighted_scores[vote.vote_type] += vote.weight
        
        # Check quorum requirements
        required_votes = self.requirements[decision_type.value]
        
        # Special handling for budget overrides
        if decision_type == DecisionType.BUDGET_OVERRIDE:
            energy_ledger_approved = any(
                v.module_id == "EnergyLedger" and v.vote_type == VoteType.APPROVE 
                for v in votes
            )
            if not energy_ledger_approved:
                return {
                    'resolved': True,
                    'outcome': VoteType.REJECT,
                    'weighted_score': 0.0,
                    'reasoning': 'Budget override requires EnergyLedger approval'
                }
        
        # Check for clear consensus
        if weighted_scores[VoteType.APPROVE] >= required_votes:
            return {
                'resolved': True,
                'outcome': VoteType.APPROVE,
                'weighted_score': weighted_scores[VoteType.APPROVE],
                'reasoning': f'Consensus achieved with {weighted_scores[VoteType.APPROVE]} weighted votes'
            }
        
        if weighted_scores[VoteType.REJECT] >= required_votes:
            return {
                'resolved': True,
                'outcome': VoteType.REJECT,
                'weighted_score': weighted_scores[VoteType.REJECT],
                'reasoning': f'Rejection consensus with {weighted_scores[VoteType.REJECT]} weighted votes'
            }
        
        return {'resolved': False}
    
    def _attempt_weighted_consensus(self, votes: List[Vote]) -> Dict:
        """Attempt weighted consensus based on module priorities."""
        
        # Calculate priority-weighted scores
        priority_scores = {VoteType.APPROVE: 0.0, VoteType.REJECT: 0.0}
        
        for vote in votes:
            if vote.vote_type in priority_scores:
                # Higher priority = lower number, so invert for scoring
                priority_weight = 1.0 / vote.priority
                weighted_contribution = vote.weight * priority_weight
                priority_scores[vote.vote_type] += weighted_contribution
        
        # Determine outcome based on highest priority-weighted score
        if priority_scores[VoteType.APPROVE] > priority_scores[VoteType.REJECT]:
            return {
                'resolved': True,
                'outcome': VoteType.APPROVE,
                'weighted_score': priority_scores[VoteType.APPROVE],
                'reasoning': f'Priority-weighted consensus: APPROVE ({priority_scores[VoteType.APPROVE]:.2f})'
            }
        elif priority_scores[VoteType.REJECT] > priority_scores[VoteType.APPROVE]:
            return {
                'resolved': True,
                'outcome': VoteType.REJECT,
                'weighted_score': priority_scores[VoteType.REJECT],
                'reasoning': f'Priority-weighted consensus: REJECT ({priority_scores[VoteType.REJECT]:.2f})'
            }
        
        return {'resolved': False}
    
    def _attempt_energy_arbitration(self, votes: List[Vote]) -> Dict:
        """Energy-based arbitration for budget-related decisions."""
        
        # Find EnergyLedger vote specifically
        energy_vote = None
        for vote in votes:
            if vote.module_id == "EnergyLedger":
                energy_vote = vote
                break
        
        if energy_vote is None:
            return {'resolved': False}
        
        # EnergyLedger has final say on budget matters
        return {
            'resolved': True,
            'outcome': energy_vote.vote_type,
            'weighted_score': energy_vote.weight,
            'reasoning': f'Energy arbitration: EnergyLedger decision ({energy_vote.reasoning})'
        }
    
    def _escalate_to_gatekit(self, decision_id: str, votes: List[Vote], 
                           decision_type: DecisionType, start_time: float) -> QuorumDecision:
        """Final escalation to external GateKit system."""
        
        # In production, this would call external arbitration service
        # For now, implement conservative fallback logic
        
        fallback_outcome = VoteType.REJECT  # Conservative default
        reasoning = "Escalated to GateKit: Conservative rejection due to consensus failure"
        
        # Simple fallback: if any critical module (priority 1-2) approves, allow
        critical_approvals = [
            v for v in votes 
            if v.vote_type == VoteType.APPROVE and v.priority <= 2
        ]
        
        if critical_approvals:
            fallback_outcome = VoteType.APPROVE
            reasoning = f"GateKit fallback: Critical module approval by {[v.module_id for v in critical_approvals]}"
        
        return QuorumDecision(
            decision_id=decision_id,
            decision_type=decision_type,
            outcome=fallback_outcome,
            total_votes=len(votes),
            weighted_score=sum(v.weight for v in critical_approvals),
            participating_modules=[v.module_id for v in votes],
            execution_time_ms=int((time.time() - start_time) * 1000),
            arbitration_level=4,
            reasoning=reasoning
        )
    
    def _get_default_timeout(self, decision_type: DecisionType) -> int:
        """Get default timeout based on decision type."""
        timeouts = {
            DecisionType.SIMPLE: 300,      # 300ms
            DecisionType.BUDGET_OVERRIDE: 500,  # 500ms  
            DecisionType.EMERGENCY: 150    # 150ms
        }
        return timeouts.get(decision_type, 300)
    
    def _handle_timeout(self, decision_id: str, timeout_exception) -> QuorumDecision:
        """Handle timeout scenarios with emergency protocols."""
        
        decision_state = self.active_decisions.get(decision_id, {})
        decision_type = decision_state.get('decision_type', DecisionType.SIMPLE)
        
        # Emergency timeout resolution
        return QuorumDecision(
            decision_id=decision_id,
            decision_type=decision_type,
            outcome=VoteType.REJECT,  # Conservative timeout default
            total_votes=0,
            weighted_score=0.0,
            participating_modules=[],
            execution_time_ms=decision_state.get('timeout_ms', 300),
            arbitration_level=0,
            reasoning=f"Timeout resolution: {str(timeout_exception)}"
        )
    
    def get_voting_statistics(self) -> Dict:
        """Get comprehensive voting system statistics."""
        return {
            'metrics': self.metrics.copy(),
            'recent_decisions': [
                {
                    'decision_id': d.decision_id,
                    'outcome': d.outcome.value,
                    'arbitration_level': d.arbitration_level,
                    'execution_time_ms': d.execution_time_ms
                }
                for d in self.decision_history[-10:]  # Last 10 decisions
            ],
            'module_participation': {
                module_id: sum(1 for d in self.decision_history 
                             if module_id in d.participating_modules)
                for module_id in self.voting_power_map.keys()
            }
        }


class QuorumTimeoutException(Exception):
    """Exception raised when quorum voting times out."""
    pass
```

## Module Integration Examples

### EnergyLedger Module Implementation

```python
class EnergyLedgerModule:
    """EnergyLedger module with quorum voting capabilities."""
    
    def __init__(self, budget_config: Dict):
        self.budget_config = budget_config
        self.current_budget_usage = 0.0
        
    def vote_on_decision(self, decision_id: str, context: Dict) -> Dict:
        """
        Cast vote on quorum decision based on energy governance rules.
        
        Returns:
            Vote response with decision and reasoning
        """
        decision_type = context.get('decision_type')
        requested_budget = context.get('requested_energy_budget', 0)
        
        # Budget override decisions
        if decision_type == 'budget_override':
            return self._vote_on_budget_override(requested_budget, context)
        
        # Simple decisions - energy impact assessment
        if decision_type == 'simple_decision':
            return self._vote_on_simple_decision(context)
        
        # Emergency decisions - prioritize system stability
        if decision_type == 'emergency_escalation':
            return self._vote_on_emergency(context)
        
        # Default: abstain on unknown decision types
        return {
            'vote': VoteType.ABSTAIN.value,
            'reasoning': f'Unknown decision type: {decision_type}',
            'metadata': {'module': 'EnergyLedger', 'confidence': 0.0}
        }
    
    def _vote_on_budget_override(self, requested_budget: float, context: Dict) -> Dict:
        """Vote on budget override requests."""
        
        # Check current budget utilization
        budget_utilization = self.current_budget_usage / self.budget_config['total_budget']
        
        # Calculate projected utilization
        projected_utilization = (self.current_budget_usage + requested_budget) / self.budget_config['total_budget']
        
        # Approval logic
        if projected_utilization <= 0.8:  # Under 80% utilization
            return {
                'vote': VoteType.APPROVE.value,
                'reasoning': f'Budget override approved: {projected_utilization:.1%} projected utilization',
                'metadata': {
                    'current_utilization': budget_utilization,
                    'projected_utilization': projected_utilization,
                    'confidence': 0.9
                }
            }
        elif projected_utilization <= 0.95:  # 80-95% utilization
            # Conditional approval with monitoring
            return {
                'vote': VoteType.APPROVE.value,
                'reasoning': f'Conditional approval: {projected_utilization:.1%} utilization with enhanced monitoring',
                'metadata': {
                    'current_utilization': budget_utilization,
                    'projected_utilization': projected_utilization,
                    'confidence': 0.6,
                    'requires_monitoring': True
                }
            }
        else:  # Over 95% utilization
            return {
                'vote': VoteType.REJECT.value,
                'reasoning': f'Budget override rejected: {projected_utilization:.1%} exceeds safe threshold',
                'metadata': {
                    'current_utilization': budget_utilization,
                    'projected_utilization': projected_utilization,
                    'confidence': 0.95
                }
            }
    
    def _vote_on_simple_decision(self, context: Dict) -> Dict:
        """Vote on simple decisions based on energy efficiency."""
        
        energy_impact = context.get('estimated_energy_impact', 0.0)
        
        # Low impact decisions - approve
        if energy_impact < 0.1:
            return {
                'vote': VoteType.APPROVE.value,
                'reasoning': f'Low energy impact decision ({energy_impact:.2f})',
                'metadata': {'energy_impact': energy_impact, 'confidence': 0.8}
            }
        
        # Medium impact - context-dependent
        elif energy_impact < 0.5:
            quality_benefit = context.get('expected_quality_improvement', 0.0)
            if quality_benefit > energy_impact * 2:  # Good ROI
                return {
                    'vote': VoteType.APPROVE.value,
                    'reasoning': f'Positive energy ROI: {quality_benefit:.2f} benefit vs {energy_impact:.2f} cost',
                    'metadata': {'energy_roi': quality_benefit / energy_impact, 'confidence': 0.7}
                }
            else:
                return {
                    'vote': VoteType.ABSTAIN.value,
                    'reasoning': f'Unclear energy ROI: {quality_benefit:.2f} benefit vs {energy_impact:.2f} cost',
                    'metadata': {'energy_roi': quality_benefit / energy_impact, 'confidence': 0.4}
                }
        
        # High impact decisions - reject unless critical
        else:
            is_critical = context.get('critical_decision', False)
            if is_critical:
                return {
                    'vote': VoteType.APPROVE.value,
                    'reasoning': f'High energy cost approved for critical decision ({energy_impact:.2f})',
                    'metadata': {'energy_impact': energy_impact, 'critical': True, 'confidence': 0.6}
                }
            else:
                return {
                    'vote': VoteType.REJECT.value,
                    'reasoning': f'High energy cost rejected for non-critical decision ({energy_impact:.2f})',
                    'metadata': {'energy_impact': energy_impact, 'confidence': 0.8}
                }
    
    def _vote_on_emergency(self, context: Dict) -> Dict:
        """Vote on emergency decisions prioritizing system stability."""
        
        # In emergencies, EnergyLedger prioritizes system preservation
        emergency_type = context.get('emergency_type', 'unknown')
        
        if emergency_type in ['system_failure', 'data_loss_risk']:
            return {
                'vote': VoteType.APPROVE.value,
                'reasoning': f'Emergency approval for {emergency_type}',
                'metadata': {'emergency_type': emergency_type, 'confidence': 0.9}
            }
        else:
            return {
                'vote': VoteType.ESCALATE.value,
                'reasoning': f'Emergency escalation for {emergency_type}',
                'metadata': {'emergency_type': emergency_type, 'confidence': 0.5}
            }
```

### EntropyAuditor Module Implementation

```python
class EntropyAuditorModule:
    """EntropyAuditor module with drift detection voting logic."""
    
    def __init__(self, entropy_config: Dict):
        self.entropy_config = entropy_config
        self.recent_entropy_scores = []
        
    def vote_on_decision(self, decision_id: str, context: Dict) -> Dict:
        """Cast vote based on entropy and drift analysis."""
        
        current_entropy = context.get('current_entropy_score', 0.0)
        drift_detected = context.get('drift_detected', False)
        
        # High entropy or drift - be conservative
        if current_entropy > 0.7 or drift_detected:
            return {
                'vote': VoteType.REJECT.value,
                'reasoning': f'High entropy ({current_entropy:.2f}) or drift detected',
                'metadata': {
                    'entropy_score': current_entropy,
                    'drift_detected': drift_detected,
                    'confidence': 0.85
                }
            }
        
        # Medium entropy - conditional approval
        elif current_entropy > 0.4:
            quality_trend = self._analyze_quality_trend(context)
            if quality_trend > 0:
                return {
                    'vote': VoteType.APPROVE.value,
                    'reasoning': f'Medium entropy ({current_entropy:.2f}) with positive quality trend',
                    'metadata': {
                        'entropy_score': current_entropy,
                        'quality_trend': quality_trend,
                        'confidence': 0.6
                    }
                }
            else:
                return {
                    'vote': VoteType.ABSTAIN.value,
                    'reasoning': f'Medium entropy ({current_entropy:.2f}) with uncertain quality trend',
                    'metadata': {
                        'entropy_score': current_entropy,
                        'quality_trend': quality_trend,
                        'confidence': 0.4
                    }
                }
        
        # Low entropy - approve
        else:
            return {
                'vote': VoteType.APPROVE.value,
                'reasoning': f'Low entropy ({current_entropy:.2f}) - system stable',
                'metadata': {
                    'entropy_score': current_entropy,
                    'system_stability': 'high',
                    'confidence': 0.9
                }
            }
    
    def _analyze_quality_trend(self, context: Dict) -> float:
        """Analyze quality trend from recent measurements."""
        quality_history = context.get('quality_history', [])
        if len(quality_history) < 3:
            return 0.0
        
        # Simple trend analysis: compare recent vs older scores
        recent_avg = sum(quality_history[-3:]) / 3
        older_avg = sum(quality_history[:-3]) / len(quality_history[:-3])
        
        return recent_avg - older_avg
```

## Production Integration

### Flask API Integration

```python
from flask import Flask, request, jsonify
import uuid

app = Flask(__name__)

# Initialize quorum voting system
quorum_engine = QuorumVotingEngine(VOTING_POWER_MAP, QUORUM_REQUIREMENTS)

# Register modules
energy_module = EnergyLedgerModule(ENERGY_BUDGET_CONFIG)
entropy_module = EntropyAuditorModule(ENTROPY_CONFIG)

quorum_engine.register_module("EnergyLedger", energy_module.vote_on_decision)
quorum_engine.register_module("EntropyAuditor", entropy_module.vote_on_decision)

@app.route('/fusion/quorum/vote', methods=['POST'])
def execute_quorum_vote():
    """Execute a quorum vote for system decisions."""
    
    request_data = request.get_json()
    
    # Generate unique decision ID
    decision_id = str(uuid.uuid4())
    
    # Determine decision type
    decision_type_str = request_data.get('decision_type', 'simple_decision')
    decision_type = DecisionType(decision_type_str)
    
    # Prepare context
    context = {
        'decision_type': decision_type_str,
        'requested_energy_budget': request_data.get('energy_budget', 0),
        'current_entropy_score': request_data.get('entropy_score', 0.0),
        'estimated_energy_impact': request_data.get('energy_impact', 0.0),
        'expected_quality_improvement': request_data.get('quality_improvement', 0.0),
        'critical_decision': request_data.get('critical', False),
        'emergency_type': request_data.get('emergency_type'),
        'drift_detected': request_data.get('drift_detected', False),
        'quality_history': request_data.get('quality_history', [])
    }
    
    try:
        # Execute quorum vote
        decision = quorum_engine.initiate_vote(
            decision_id=decision_id,
            decision_type=decision_type,
            context=context,
            timeout_ms=request_data.get('timeout_ms')
        )
        
        return jsonify({
            'status': 'success',
            'decision': {
                'decision_id': decision.decision_id,
                'outcome': decision.outcome.value,
                'total_votes': decision.total_votes,
                'weighted_score': decision.weighted_score,
                'participating_modules': decision.participating_modules,
                'execution_time_ms': decision.execution_time_ms,
                'arbitration_level': decision.arbitration_level,
                'reasoning': decision.reasoning
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'decision_id': decision_id
        }), 500

@app.route('/fusion/quorum/statistics', methods=['GET'])
def get_quorum_statistics():
    """Get quorum voting system statistics."""
    
    stats = quorum_engine.get_voting_statistics()
    
    return jsonify({
        'status': 'success',
        'statistics': stats,
        'voting_power_map': VOTING_POWER_MAP,
        'quorum_requirements': QUORUM_REQUIREMENTS
    })

@app.route('/fusion/quorum/health', methods=['GET'])
def get_quorum_health():
    """Get health status of quorum voting system."""
    
    # Check module connectivity
    module_health = {}
    for module_id in quorum_engine.module_callbacks.keys():
        try:
            # Test module responsiveness with minimal context
            test_response = quorum_engine.module_callbacks[module_id](
                'health_check', 
                {'decision_type': 'simple_decision', 'test': True}
            )
            module_health[module_id] = {
                'status': 'healthy',
                'response_time_ms': 1,  # Would measure actual time
                'last_vote': test_response.get('vote', 'unknown')
            }
        except Exception as e:
            module_health[module_id] = {
                'status': 'unhealthy',
                'error': str(e),
                'last_error_time': time.time()
            }
    
    # Overall health assessment
    healthy_modules = sum(1 for h in module_health.values() if h['status'] == 'healthy')
    total_modules = len(module_health)
    health_score = healthy_modules / total_modules if total_modules > 0 else 0.0
    
    return jsonify({
        'status': 'healthy' if health_score >= 0.7 else 'degraded',
        'health_score': health_score,
        'module_health': module_health,
        'total_decisions': quorum_engine.metrics['decisions_processed'],
        'consensus_rate': (quorum_engine.metrics['consensus_achieved'] / 
                          max(1, quorum_engine.metrics['decisions_processed']))
    })
```

## Advanced Quorum Patterns

### Federated Voting

```python
class FederatedQuorumEngine:
    """Extended quorum engine supporting federated multi-organization voting."""
    
    def __init__(self, local_voting_power: Dict, federation_config: Dict):
        self.local_engine = QuorumVotingEngine(local_voting_power, QUORUM_REQUIREMENTS)
        self.federation_config = federation_config
        self.peer_engines = {}  # Remote peer connections
        
    def initiate_federated_vote(self, decision_id: str, decision_type: DecisionType, 
                               context: Dict, federation_scope: List[str]) -> Dict:
        """Execute vote across federated organizations."""
        
        local_decision = self.local_engine.initiate_vote(decision_id, decision_type, context)
        
        # Collect votes from federation peers
        peer_decisions = {}
        for peer_org in federation_scope:
            if peer_org in self.peer_engines:
                try:
                    peer_decision = self.peer_engines[peer_org].request_vote(
                        decision_id, decision_type, context
                    )
                    peer_decisions[peer_org] = peer_decision
                except Exception as e:
                    logging.warning(f"Failed to get vote from peer {peer_org}: {e}")
        
        # Federated consensus logic
        federated_result = self._resolve_federated_consensus(
            local_decision, peer_decisions, context
        )
        
        return federated_result
    
    def _resolve_federated_consensus(self, local_decision: QuorumDecision, 
                                   peer_decisions: Dict, context: Dict) -> Dict:
        """Resolve consensus across federated organizations."""
        
        # Weight local vs peer decisions
        local_weight = self.federation_config.get('local_weight', 0.4)
        peer_weight = (1.0 - local_weight) / len(peer_decisions) if peer_decisions else 0.0
        
        # Calculate weighted consensus
        weighted_score = local_decision.weighted_score * local_weight
        
        for peer_org, peer_decision in peer_decisions.items():
            if peer_decision['outcome'] == local_decision.outcome.value:
                weighted_score += peer_decision['weighted_score'] * peer_weight
            else:
                weighted_score -= peer_decision['weighted_score'] * peer_weight
        
        # Determine federated outcome
        if weighted_score > 0:
            outcome = local_decision.outcome
            reasoning = f"Federated consensus: {weighted_score:.2f} weighted agreement"
        else:
            outcome = VoteType.REJECT if local_decision.outcome == VoteType.APPROVE else VoteType.APPROVE
            reasoning = f"Federated override: {weighted_score:.2f} weighted disagreement"
        
        return {
            'federated_decision': {
                'decision_id': local_decision.decision_id,
                'outcome': outcome.value,
                'local_decision': local_decision.outcome.value,
                'peer_decisions': {org: d['outcome'] for org, d in peer_decisions.items()},
                'weighted_consensus_score': weighted_score,
                'reasoning': reasoning
            }
        }
```

## Testing Framework

### Quorum Voting Test Suite

```python
import pytest
from unittest.mock import Mock, patch
import threading
import time

class TestQuorumVotingSystem:
    """Comprehensive test suite for quorum voting functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.quorum_engine = QuorumVotingEngine(VOTING_POWER_MAP, QUORUM_REQUIREMENTS)
        
        # Mock module callbacks
        self.mock_modules = {}
        for module_id in VOTING_POWER_MAP.keys():
            mock_callback = Mock()
            self.mock_modules[module_id] = mock_callback
            self.quorum_engine.register_module(module_id, mock_callback)
    
    def test_simple_consensus_approval(self):
        """Test simple decision consensus with approval outcome."""
        
        # Configure mock responses for approval
        self.mock_modules["EnergyLedger"].return_value = {
            'vote': VoteType.APPROVE.value,
            'reasoning': 'Energy budget acceptable',
            'metadata': {}
        }
        self.mock_modules["EntropyAuditor"].return_value = {
            'vote': VoteType.APPROVE.value,
            'reasoning': 'Low entropy detected',
            'metadata': {}
        }
        self.mock_modules["TrustManager"].return_value = {
            'vote': VoteType.APPROVE.value,
            'reasoning': 'High trust score',
            'metadata': {}
        }
        
        # Execute vote
        decision = self.quorum_engine.initiate_vote(
            "test_decision_1",
            DecisionType.SIMPLE,
            {'test': True}
        )
        
        # Assertions
        assert decision.outcome == VoteType.APPROVE
        assert decision.arbitration_level == 1  # Primary consensus
        assert decision.weighted_score >= 3  # Minimum required votes
        assert len(decision.participating_modules) >= 3
    
    def test_budget_override_energy_ledger_required(self):
        """Test budget override requires EnergyLedger approval."""
        
        # Configure EnergyLedger rejection
        self.mock_modules["EnergyLedger"].return_value = {
            'vote': VoteType.REJECT.value,
            'reasoning': 'Budget threshold exceeded',
            'metadata': {}
        }
        
        # Other modules approve
        for module_id in ["EntropyAuditor", "TrustManager"]:
            self.mock_modules[module_id].return_value = {
                'vote': VoteType.APPROVE.value,
                'reasoning': 'Module approval',
                'metadata': {}
            }
        
        # Execute budget override vote
        decision = self.quorum_engine.initiate_vote(
            "test_budget_override",
            DecisionType.BUDGET_OVERRIDE,
            {'requested_energy_budget': 15000}
        )
        
        # Should be rejected despite other approvals
        assert decision.outcome == VoteType.REJECT
        assert "EnergyLedger approval" in decision.reasoning
    
    def test_weighted_consensus_arbitration(self):
        """Test weighted consensus when primary consensus fails."""
        
        # Configure split vote requiring arbitration
        self.mock_modules["EnergyLedger"].return_value = {
            'vote': VoteType.APPROVE.value,
            'reasoning': 'Energy acceptable',
            'metadata': {}
        }
        self.mock_modules["EntropyAuditor"].return_value = {
            'vote': VoteType.REJECT.value,
            'reasoning': 'High entropy risk',
            'metadata': {}
        }
        self.mock_modules["TrustManager"].return_value = {
            'vote': VoteType.ABSTAIN.value,
            'reasoning': 'Uncertain trust metrics',
            'metadata': {}
        }
        
        decision = self.quorum_engine.initiate_vote(
            "test_arbitration",
            DecisionType.SIMPLE,
            {'split_vote_scenario': True}
        )
        
        # Should escalate to weighted consensus (level 2)
        assert decision.arbitration_level == 2
        # EnergyLedger has higher priority (1) than EntropyAuditor (2)
        assert decision.outcome == VoteType.APPROVE
    
    def test_timeout_handling(self):
        """Test timeout scenarios and emergency resolution."""
        
        # Configure slow module response
        def slow_callback(decision_id, context):
            time.sleep(0.2)  # 200ms delay
            return {'vote': VoteType.APPROVE.value, 'reasoning': 'Slow response'}
        
        self.mock_modules["EnergyLedger"] = slow_callback
        
        # Execute with short timeout
        decision = self.quorum_engine.initiate_vote(
            "test_timeout",
            DecisionType.EMERGENCY,  # 150ms timeout
            {'emergency_scenario': True}
        )
        
        # Should result in timeout resolution
        assert decision.arbitration_level == 0  # Timeout resolution
        assert decision.outcome == VoteType.REJECT  # Conservative default
        assert "Timeout resolution" in decision.reasoning
    
    def test_concurrent_voting_thread_safety(self):
        """Test thread safety with concurrent voting requests."""
        
        # Configure consistent responses
        for module_id in self.mock_modules:
            self.mock_modules[module_id].return_value = {
                'vote': VoteType.APPROVE.value,
                'reasoning': 'Thread test approval',
                'metadata': {}
            }
        
        decisions = {}
        threads = []
        
        def execute_vote(vote_id):
            decision = self.quorum_engine.initiate_vote(
                f"concurrent_test_{vote_id}",
                DecisionType.SIMPLE,
                {'thread_id': vote_id}
            )
            decisions[vote_id] = decision
        
        # Start multiple concurrent votes
        for i in range(5):
            thread = threading.Thread(target=execute_vote, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=2.0)
        
        # Verify all votes completed successfully
        assert len(decisions) == 5
        for decision in decisions.values():
            assert decision.outcome == VoteType.APPROVE
            assert decision.arbitration_level == 1
    
    def test_module_failure_resilience(self):
        """Test system resilience when modules fail."""
        
        # Configure one module to fail
        self.mock_modules["EnergyLedger"].side_effect = Exception("Module failure")
        
        # Other modules approve
        for module_id in ["EntropyAuditor", "TrustManager"]:
            self.mock_modules[module_id].return_value = {
                'vote': VoteType.APPROVE.value,
                'reasoning': 'Working module approval',
                'metadata': {}
            }
        
        decision = self.quorum_engine.initiate_vote(
            "test_module_failure",
            DecisionType.SIMPLE,
            {'failure_test': True}
        )
        
        # Should still reach consensus with remaining modules
        assert decision.outcome in [VoteType.APPROVE, VoteType.REJECT]
        # Should not include failed module in participants
        assert "EnergyLedger" not in decision.participating_modules
        
    def test_voting_statistics_tracking(self):
        """Test voting statistics and metrics collection."""
        
        # Configure mock responses
        for module_id in self.mock_modules:
            self.mock_modules[module_id].return_value = {
                'vote': VoteType.APPROVE.value,
                'reasoning': 'Stats test',
                'metadata': {}
            }
        
        # Execute multiple votes
        for i in range(3):
            self.quorum_engine.initiate_vote(
                f"stats_test_{i}",
                DecisionType.SIMPLE,
                {'stats_test': i}
            )
        
        # Check statistics
        stats = self.quorum_engine.get_voting_statistics()
        
        assert stats['metrics']['decisions_processed'] == 3
        assert stats['metrics']['consensus_achieved'] == 3
        assert len(stats['recent_decisions']) == 3
        
        # Check module participation tracking
        for module_id in VOTING_POWER_MAP.keys():
            if module_id != "EnergyLedger":  # May have failed in previous test
                assert module_id in stats['module_participation']
```

## Monitoring and Alerting

### Quorum Health Monitoring

```python
class QuorumHealthMonitor:
    """Continuous health monitoring for quorum voting system."""
    
    def __init__(self, quorum_engine: QuorumVotingEngine):
        self.quorum_engine = quorum_engine
        self.health_metrics = {
            'consensus_rate': 0.0,
            'average_execution_time': 0.0,
            'escalation_rate': 0.0,
            'timeout_rate': 0.0,
            'module_availability': {}
        }
    
    def collect_health_metrics(self) -> Dict:
        """Collect comprehensive health metrics."""
        stats = self.quorum_engine.get_voting_statistics()
        
        total_decisions = stats['metrics']['decisions_processed']
        if total_decisions == 0:
            return self.health_metrics
        
        # Calculate key rates
        self.health_metrics.update({
            'consensus_rate': stats['metrics']['consensus_achieved'] / total_decisions,
            'escalation_rate': stats['metrics']['escalations'] / total_decisions,
            'timeout_rate': stats['metrics']['timeouts'] / total_decisions,
            'total_decisions': total_decisions
        })
        
        # Module availability analysis
        for module_id, participation_count in stats['module_participation'].items():
            availability = participation_count / total_decisions
            self.health_metrics['module_availability'][module_id] = availability
        
        return self.health_metrics
    
    def generate_health_alerts(self) -> List[Dict]:
        """Generate alerts based on health thresholds."""
        alerts = []
        metrics = self.collect_health_metrics()
        
        # Consensus rate alert
        if metrics['consensus_rate'] < 0.8:
            alerts.append({
                'severity': 'high',
                'type': 'consensus_rate_low',
                'message': f"Consensus rate {metrics['consensus_rate']:.2%} below 80% threshold",
                'metric_value': metrics['consensus_rate'],
                'threshold': 0.8
            })
        
        # Escalation rate alert
        if metrics['escalation_rate'] > 0.2:
            alerts.append({
                'severity': 'medium',
                'type': 'high_escalation_rate',
                'message': f"Escalation rate {metrics['escalation_rate']:.2%} above 20% threshold",
                'metric_value': metrics['escalation_rate'],
                'threshold': 0.2
            })
        
        # Module availability alerts
        for module_id, availability in metrics['module_availability'].items():
            if availability < 0.9:
                alerts.append({
                    'severity': 'high' if availability < 0.7 else 'medium',
                    'type': 'module_availability_low',
                    'message': f"Module {module_id} availability {availability:.2%} below threshold",
                    'module_id': module_id,
                    'metric_value': availability,
                    'threshold': 0.9
                })
        
        return alerts
```

This comprehensive quorum voting system provides robust distributed decision-making with multi-layered arbitration, timeout handling, and comprehensive monitoring capabilities for production AI workflow orchestration.