#!/usr/bin/env python3
"""
RESONTINEX Composite Signal Module
Production-grade trust scoring with explainable risk assessment.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Any, Protocol, Optional
import math
import json
import yaml
from pathlib import Path
import logging

############################ 
# 1. CONFIG & UTILITIES
############################

DEFAULT_CFG = {
    "weights": {
        "epistemic_risk": 0.4,
        "alignment_score": 0.4,
        "confidence_band": 0.2
    },
    "risk_tiers": {
        "block": 0.30,
        "review": 0.55, 
        "monitor": 0.75,
        "pass": 1.01
    },
    "action_map": {
        "block": {"decision": "abort", "reason": "high_risk_auto_block"},
        "review": {"decision": "manual", "reason": "moderate_risk_needs_review"},
        "monitor": {"decision": "allow", "reason": "low_risk_monitored"}, 
        "pass": {"decision": "allow", "reason": "auto_pass"}
    },
    # RESONTINEX integration
    "trust_floor": 0.60,  # Matches resontinex.json runtime.trust_floor
    "entropy_threshold": 0.72,  # Matches EntropyAuditor.threshold
    "integration": {
        "scenario_manager_compat": True,
        "trust_manager_enhanced": True
    }
}

def load_cfg(path: str | None) -> dict:
    """Load configuration with RESONTINEX config directory support."""
    if path is None:
        return DEFAULT_CFG
        
    config_path = Path(path)
    if not config_path.is_absolute():
        # Check RESONTINEX standard config locations
        resontinex_configs = [
            Path("./configs/fusion") / config_path,
            Path("./config") / config_path, 
            config_path
        ]
        for candidate in resontinex_configs:
            if candidate.exists():
                config_path = candidate
                break
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix in (".yml", ".yaml"):
                return yaml.safe_load(f)
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load config from {config_path}: {e}")
        return DEFAULT_CFG

def logit(p: float) -> float:
    """Convert probability to logit space with numerical stability."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

def logistic(x: float) -> float:
    """Convert from logit space to probability."""
    return 1 / (1 + math.exp(-x))

############################
# 2. STRATEGY PATTERN FOR AGGREGATION
############################

class AggregationStrategy(Protocol):
    """Defines the interface for any scoring algorithm."""
    def __call__(self, signal: 'CompositeSignal') -> float: ...

class LogitStrategy(AggregationStrategy):
    """Default sophisticated aggregation strategy using logit transforms."""
    def __call__(self, signal: 'CompositeSignal') -> float:
        w = signal._cfg["weights"]
        logits = (
            w["alignment_score"] * logit(signal.alignment_score) -
            w["epistemic_risk"] * logit(signal.epistemic_risk) -
            w["confidence_band"] * logit(signal.confidence_band)
        )
        return logistic(logits)

class ResontinexStrategy(AggregationStrategy):
    """RESONTINEX-optimized strategy respecting trust_floor and entropy_threshold."""
    def __call__(self, signal: 'CompositeSignal') -> float:
        base_score = LogitStrategy()(signal)
        
        # Apply RESONTINEX constraints
        trust_floor = signal._cfg.get("trust_floor", 0.60)
        entropy_threshold = signal._cfg.get("entropy_threshold", 0.72)
        
        # Entropy penalty if confidence band indicates high entropy
        entropy_penalty = 0.0
        if signal.confidence_band > entropy_threshold:
            entropy_penalty = 0.1 * (signal.confidence_band - entropy_threshold)
            
        # Trust floor enforcement
        adjusted_score = max(trust_floor, base_score - entropy_penalty)
        
        return min(1.0, adjusted_score)

############################
# 3. DATA MODEL
############################

@dataclass(frozen=True, slots=True)
class CompositeSignal:
    """
    Production-grade composite signal for RESONTINEX trust assessment.
    
    Integrates with TrustManager, EntropyAuditor, and ScenarioManager
    for comprehensive risk evaluation and explainable decision making.
    """
    epistemic_risk: float
    alignment_score: float 
    confidence_band: float
    _cfg: dict = field(repr=False, default_factory=lambda: DEFAULT_CFG)
    _agg_strategy: AggregationStrategy = field(repr=False, default_factory=ResontinexStrategy)
    
    def __post_init__(self):
        """Validate inputs with RESONTINEX-standard error handling."""
        for k, v in asdict(self).items():
            if k.startswith("_"):
                continue
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"CompositeSignal.{k} must be within [0,1], got {v}")
    
    @property
    def trust_score(self) -> float:
        """Delegates scoring to the configured strategy."""
        score = self._agg_strategy(self)
        return round(float(score), 4)
    
    @property
    def risk_tier(self) -> str:
        """Determine risk tier for RESONTINEX decision routing."""
        score = self.trust_score
        for tier, ub in self._cfg["risk_tiers"].items():
            if score <= ub:
                return tier
        return "undefined"
    
    @property
    def entropy_flag(self) -> bool:
        """Check if signal exceeds EntropyAuditor threshold."""
        entropy_threshold = self._cfg.get("entropy_threshold", 0.72) 
        return self.confidence_band > entropy_threshold
        
    @property
    def trust_floor_met(self) -> bool:
        """Check if trust score meets RESONTINEX trust floor."""
        trust_floor = self._cfg.get("trust_floor", 0.60)
        return self.trust_score >= trust_floor
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for RESONTINEX module integration."""
        d = asdict(self)
        d.pop("_cfg", None)
        d.pop("_agg_strategy", None)
        d["trust_score"] = self.trust_score
        d["risk_tier"] = self.risk_tier
        d["entropy_flag"] = self.entropy_flag
        d["trust_floor_met"] = self.trust_floor_met
        return d
        
    def to_scenario_metrics(self) -> Dict[str, float]:
        """Convert to ScenarioMetrics-compatible format."""
        return {
            'success_rate': self.alignment_score,
            'reliability_index': 1.0 - self.epistemic_risk,
            'resource_efficiency': 1.0 - self.confidence_band,
            'user_satisfaction': self.trust_score,
            'avg_latency_ms': 100.0,  # Default for new scenarios
            'complexity_score': 0.5   # Neutral complexity
        }
    
    def downstream_action(self) -> Dict[str, Any]:
        """Generate action for RESONTINEX conflict resolution protocols."""
        tier = self.risk_tier
        base_action = self._cfg["action_map"][tier].copy()
        
        # Add RESONTINEX-specific context
        base_action.update({
            "tier": tier,
            "trust_score": self.trust_score,
            "entropy_flag": self.entropy_flag,
            "trust_floor_met": self.trust_floor_met,
            "voting_weight": self._calculate_voting_weight()
        })
        
        return base_action
        
    def _calculate_voting_weight(self) -> int:
        """Calculate voting weight for quorum-based conflict resolution."""
        # Map trust score to voting power (1-3 range)
        if self.trust_score >= 0.9:
            return 3
        elif self.trust_score >= 0.7:
            return 2
        else:
            return 1
    
    def explain_score(self) -> Dict[str, Any]:
        """Provide full explainability trace for RESONTINEX audit trails."""
        if not isinstance(self._agg_strategy, (LogitStrategy, ResontinexStrategy)):
            return {"explanation": "Not available for custom strategy"}
            
        w = self._cfg["weights"]
        explanation = {
            "strategy_used": self._agg_strategy.__class__.__name__,
            "resontinex_integration": {
                "trust_floor": self._cfg.get("trust_floor", 0.60),
                "entropy_threshold": self._cfg.get("entropy_threshold", 0.72),
                "trust_floor_met": self.trust_floor_met,
                "entropy_flag": self.entropy_flag
            },
            "contributions_in_logit_space": {
                "alignment_score": w["alignment_score"] * logit(self.alignment_score),
                "epistemic_risk_penalty": -w["epistemic_risk"] * logit(self.epistemic_risk), 
                "confidence_band_penalty": -w["confidence_band"] * logit(self.confidence_band),
            }
        }
        
        explanation["total_logit"] = sum(explanation["contributions_in_logit_space"].values())
        explanation["final_trust_score"] = self.trust_score
        explanation["risk_assessment"] = {
            "tier": self.risk_tier,
            "action": self._cfg["action_map"][self.risk_tier],
            "voting_weight": self._calculate_voting_weight()
        }
        
        return explanation

############################
# 4. INTEGRATION UTILITIES
############################

def create_from_scenario_metrics(metrics, config_path: str | None = None) -> CompositeSignal:
    """Create CompositeSignal from ScenarioManager ScenarioMetrics."""
    cfg = load_cfg(config_path)
    
    return CompositeSignal(
        epistemic_risk=1.0 - metrics.reliability_index,
        alignment_score=metrics.success_rate,
        confidence_band=1.0 - metrics.resource_efficiency,
        _cfg=cfg
    )

def enhance_trust_manager_scoring(alignment_score: float, inflation_delta: float, 
                                epistemic_risk: float, confidence_band: float,
                                config_path: str | None = None) -> Dict[str, Any]:
    """Enhanced TrustManager scoring using CompositeSignal."""
    cfg = load_cfg(config_path)
    
    # Incorporate existing TrustManager inputs
    adjusted_alignment = max(0.0, min(1.0, alignment_score - abs(inflation_delta) * 0.1))
    
    signal = CompositeSignal(
        epistemic_risk=epistemic_risk,
        alignment_score=adjusted_alignment,
        confidence_band=confidence_band,
        _cfg=cfg
    )
    
    return {
        "trust_signal": signal.to_dict(),
        "downstream_action": signal.downstream_action(),
        "explanation": signal.explain_score(),
        "legacy_score": alignment_score + inflation_delta  # For backward compatibility
    }

###############################################
# 5. PRODUCTION EXAMPLE USAGE
###############################################

if __name__ == "__main__":
    # High-trust scenario (typical good case)
    signal_good = CompositeSignal(
        epistemic_risk=0.1,
        alignment_score=0.9,
        confidence_band=0.05
    )
    
    print("=== High Trust Scenario ===")
    print(json.dumps(signal_good.to_dict(), indent=2))
    print("\nAction:", json.dumps(signal_good.downstream_action(), indent=2))
    print("\nExplanation:", json.dumps(signal_good.explain_score(), indent=2))
    
    # High-risk scenario (typical blocking case)
    signal_bad = CompositeSignal(
        epistemic_risk=0.8,
        alignment_score=0.4,
        confidence_band=0.7  
    )
    
    print("\n=== High Risk Scenario ===")
    print(json.dumps(signal_bad.to_dict(), indent=2))
    print("\nAction:", json.dumps(signal_bad.downstream_action(), indent=2))
    print("\nExplanation:", json.dumps(signal_bad.explain_score(), indent=2))