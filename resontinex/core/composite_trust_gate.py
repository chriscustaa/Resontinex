#!/usr/bin/env python3
"""
RESONTINEX CompositeTrustGate System
Enhanced production-ready trust scoring with mathematical rigor, calibration support, and observability.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, Protocol
import logging
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import hashlib

# Mathematical imports with fallbacks
try:
    from scipy.special import expit, logit as scipy_logit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# RESONTINEX integration
from .composite_signal import DEFAULT_CFG as LEGACY_DEFAULT_CFG
from ..obs.middleware import measure
from ..fusion_resilience import get_fusion_loader

##################################
# 1. ENHANCED CONFIGURATION
##################################

EPSILON = 1e-6

TRUST_GATE_CONFIG = {
    "weights": {
        "epistemic_risk": 0.35,
        "alignment_score": 0.45,
        "confidence_band": 0.20
    },
    "risk_tiers": {
        "block": 0.25,
        "review": 0.50,
        "monitor": 0.75,
        "pass": 1.01
    },
    "tier_fallbacks": {
        "undefined": "review",
        "review": "manual",
        "manual": "block"
    },
    "calibration": {
        "enabled": True,
        "method": "isotonic",  # "isotonic" or "platt"
        "min_samples": 50,
        "confidence_threshold": 0.8
    },
    "trust_floor": 0.60,
    "entropy_threshold": 0.72,
    "observability": {
        "audit_logging": True,
        "metrics_collection": True,
        "explainability_trace": True
    }
}

##################################
# 2. MATHEMATICAL RIGOR
##################################

def safe_logit(p: float, epsilon: float = EPSILON) -> float:
    """Convert probability to logit space with monotonicity guarantees."""
    p_clamped = np.clip(p, epsilon, 1.0 - epsilon)
    if HAS_SCIPY:
        return float(scipy_logit(p_clamped))
    else:
        # Fallback implementation
        return float(np.log(p_clamped / (1.0 - p_clamped)))

def safe_expit(x: float) -> float:
    """Convert from logit space with numerical stability."""
    x_clamped = np.clip(x, -500, 500)
    if HAS_SCIPY:
        return float(expit(x_clamped))
    else:
        # Fallback implementation
        return float(1.0 / (1.0 + np.exp(-x_clamped)))

def logit_aggregation(weights: Dict[str, float], 
                     epistemic_risk: float, 
                     alignment_score: float, 
                     confidence_band: float) -> float:
    """
    Logit aggregation with monotonicity guarantees.
    
    Mathematical property: Increasing alignment_score increases output.
    Mathematical property: Increasing epistemic_risk decreases output.
    Mathematical property: Increasing confidence_band decreases output.
    """
    # Ensure monotonicity through directional transforms
    alignment_logit = safe_logit(alignment_score)
    risk_logit = safe_logit(1.0 - epistemic_risk)  # Invert for monotonicity
    confidence_logit = safe_logit(1.0 - confidence_band)  # Invert for monotonicity
    
    # Weighted aggregation in logit space
    combined_logit = (
        weights["alignment_score"] * alignment_logit +
        weights["epistemic_risk"] * risk_logit +
        weights["confidence_band"] * confidence_logit
    )
    
    return safe_expit(combined_logit)

def validate_config_and_normalize(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration with weight normalization and tier sorting."""
    config = config.copy()
    
    # Validate required keys
    required_keys = ["weights", "risk_tiers", "trust_floor", "entropy_threshold"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    # Normalize weights
    weights = config["weights"]
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > EPSILON:
        config["weights"] = {k: v / total_weight for k, v in weights.items()}
    
    # Sort risk tiers by threshold
    tiers = config["risk_tiers"]
    config["risk_tiers"] = dict(sorted(tiers.items(), key=lambda x: x[1]))
    
    return config

##################################
# 3. CALIBRATION SYSTEM
##################################

class CalibrationModel(Protocol):
    """Protocol for calibration models."""
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None: ...
    def predict_proba(self, scores: np.ndarray) -> np.ndarray: ...

class IsotonicCalibrationModel:
    """Isotonic regression calibration model."""
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self.model.fit(scores, labels)
        
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores).reshape(-1, 1) if scores.ndim == 1 else scores
        return self.model.predict(scores.flatten())

class PlattCalibrationModel:
    """Platt scaling calibration model."""
    def __init__(self):
        self.model = LogisticRegression()
        
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        scores = scores.reshape(-1, 1)
        self.model.fit(scores, labels)
        
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores).reshape(-1, 1) if scores.ndim == 1 else scores
        return self.model.predict_proba(scores)[:, 1]

def fit_calibration_model(scores: List[float], 
                         labels: List[bool],
                         method: str = "isotonic") -> Optional[CalibrationModel]:
    """
    Fit calibration model supporting isotonic/Platt scaling.
    
    Args:
        scores: Raw trust scores from CompositeTrustGate
        labels: Ground truth binary labels (True = trustworthy)
        method: "isotonic" or "platt"
        
    Returns:
        Fitted calibration model or None if insufficient data
    """
    if len(scores) < TRUST_GATE_CONFIG["calibration"]["min_samples"]:
        return None
        
    scores_array = np.array(scores)
    labels_array = np.array(labels, dtype=float)
    
    if method == "isotonic":
        model = IsotonicCalibrationModel()
    elif method == "platt":
        model = PlattCalibrationModel()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
        
    model.fit(scores_array, labels_array)
    return model

def derive_tier_thresholds(calibrated_scores: List[float], 
                          confidence_threshold: float = 0.8) -> Dict[str, float]:
    """Derive tier thresholds from calibrated score distributions."""
    scores = np.array(calibrated_scores)
    
    # Calculate percentiles for tier boundaries
    percentiles = {
        "block": 25,    # Bottom 25% -> block
        "review": 50,   # 25-50% -> review  
        "monitor": 75,  # 50-75% -> monitor
        "pass": 100     # Top 25% -> pass
    }
    
    return {
        tier: np.percentile(scores, p) / 100.0 
        for tier, p in percentiles.items()
    }

##################################
# 4. CORE COMPOSITRUSTGATE CLASS
##################################

@dataclass(frozen=True, slots=True)
class CompositeTrustGate:
    """
    Production-ready composite trust gate with mathematical rigor and calibration support.
    
    Critical fixes applied:
    - Tier ordering using sorted thresholds with raw scores
    - Config validation with required keys and weight normalization
    - Safe fallbacks for undefined tiers (tier → review → manual)
    - Monotonicity mathematical guarantees via derivative properties
    - Calibration integration with historical data fitting
    """
    epistemic_risk: float
    alignment_score: float
    confidence_band: float
    
    # Configuration and strategy
    _config: Dict[str, Any] = field(repr=False, default_factory=lambda: TRUST_GATE_CONFIG)
    _calibration_model: Optional[CalibrationModel] = field(repr=False, default=None)
    _raw_score: float = field(repr=False, default=0.0, init=False)
    _display_score: float = field(repr=False, default=0.0, init=False)
    
    def __post_init__(self):
        """Input validation with epsilon tolerance."""
        # Validate core inputs
        for attr_name in ["epistemic_risk", "alignment_score", "confidence_band"]:
            value = getattr(self, attr_name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{attr_name} must be numeric, got {type(value)}")
            if not (0.0 - EPSILON <= value <= 1.0 + EPSILON):
                raise ValueError(f"{attr_name} must be within [0,1], got {value}")
        
        # Validate and normalize configuration
        try:
            validated_config = validate_config_and_normalize(self._config)
            object.__setattr__(self, '_config', validated_config)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")
        
        # Calculate raw and display scores
        raw_score = logit_aggregation(
            self._config["weights"],
            self.epistemic_risk,
            self.alignment_score, 
            self.confidence_band
        )
        object.__setattr__(self, '_raw_score', raw_score)
        object.__setattr__(self, '_display_score', round(raw_score, 4))

    @property
    def trust_score(self) -> float:
        """Primary trust score (rounded for display)."""
        return self._display_score

    @property
    def raw_trust_score(self) -> float:
        """Raw trust score (full precision for internal calculations)."""
        return self._raw_score

    @property
    def calibrated_score(self) -> Optional[float]:
        """Calibrated trust score if calibration model available."""
        if self._calibration_model is None:
            return None
        scores = np.array([self._raw_score])
        calibrated = self._calibration_model.predict_proba(scores)
        return float(calibrated[0]) if len(calibrated) > 0 else None

    @property
    def risk_tier(self) -> str:
        """Determine risk tier with safe fallback for undefined tiers."""
        # Use raw score for tier determination (more precise)
        score = self.calibrated_score or self._raw_score
        
        # Check tiers in sorted order
        for tier, threshold in self._config["risk_tiers"].items():
            if score <= threshold:
                return tier
        
        # Safe fallback for undefined tiers
        return self._apply_tier_fallback("undefined")

    def _apply_tier_fallback(self, original_tier: str) -> str:
        """Apply safe fallback chain for undefined tiers."""
        fallbacks = self._config.get("tier_fallbacks", {})
        current_tier = original_tier
        
        # Follow fallback chain up to 3 levels deep
        for _ in range(3):
            if current_tier in fallbacks:
                current_tier = fallbacks[current_tier]
            else:
                break
        
        # Ultimate fallback
        if current_tier not in self._config["risk_tiers"]:
            current_tier = "review"
            
        return current_tier

    @property
    def entropy_flag(self) -> bool:
        """Check if signal exceeds entropy threshold."""
        return self.confidence_band > self._config["entropy_threshold"]

    @property
    def trust_floor_met(self) -> bool:
        """Check if trust score meets minimum threshold."""
        score = self.calibrated_score or self.trust_score
        return score >= self._config["trust_floor"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all key metrics."""
        result = asdict(self)
        
        # Remove private fields
        result = {k: v for k, v in result.items() if not k.startswith('_')}
        
        # Add computed properties
        result.update({
            "trust_score": self.trust_score,
            "raw_trust_score": self._raw_score,
            "calibrated_score": self.calibrated_score,
            "risk_tier": self.risk_tier,
            "entropy_flag": self.entropy_flag,
            "trust_floor_met": self.trust_floor_met,
            "voting_weight": self._calculate_voting_weight()
        })
        
        return result

    def downstream_action(self) -> Dict[str, Any]:
        """Generate comprehensive action for decision routing."""
        tier = self.risk_tier
        
        action_map = {
            "block": {"decision": "abort", "reason": "high_risk_block", "requires_override": True},
            "review": {"decision": "manual_review", "reason": "moderate_risk", "requires_override": False},
            "monitor": {"decision": "proceed_monitored", "reason": "low_risk_monitored", "requires_override": False},
            "pass": {"decision": "auto_approve", "reason": "high_trust", "requires_override": False}
        }
        
        base_action = action_map.get(tier, action_map["review"])
        
        return {
            **base_action,
            "tier": tier,
            "trust_score": self.trust_score,
            "calibrated_score": self.calibrated_score,
            "entropy_flag": self.entropy_flag,
            "trust_floor_met": self.trust_floor_met,
            "voting_weight": self._calculate_voting_weight(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _calculate_voting_weight(self) -> int:
        """Calculate voting weight for consensus mechanisms."""
        score = self.calibrated_score or self.trust_score
        
        if score >= 0.9:
            return 3
        elif score >= 0.75:
            return 2
        elif score >= 0.6:
            return 1
        else:
            return 0

    def explain_score(self) -> Dict[str, Any]:
        """Full explainability payload for audit trails."""
        weights = self._config["weights"]
        
        # Calculate individual contributions in logit space
        alignment_contribution = weights["alignment_score"] * safe_logit(self.alignment_score)
        risk_contribution = weights["epistemic_risk"] * safe_logit(1.0 - self.epistemic_risk)
        confidence_contribution = weights["confidence_band"] * safe_logit(1.0 - self.confidence_band)
        
        explanation = {
            "computation_method": "logit_aggregation_with_monotonicity_guarantees",
            "mathematical_properties": {
                "monotonic_alignment": "increasing alignment_score increases trust",
                "monotonic_risk": "increasing epistemic_risk decreases trust", 
                "monotonic_confidence": "increasing confidence_band decreases trust"
            },
            "input_validation": {
                "epsilon_tolerance": EPSILON,
                "bounds_checked": True,
                "config_validated": True
            },
            "logit_space_contributions": {
                "alignment_score": {
                    "input": self.alignment_score,
                    "weight": weights["alignment_score"],
                    "contribution": alignment_contribution
                },
                "epistemic_risk": {
                    "input": self.epistemic_risk,
                    "transformed_input": 1.0 - self.epistemic_risk,
                    "weight": weights["epistemic_risk"], 
                    "contribution": risk_contribution
                },
                "confidence_band": {
                    "input": self.confidence_band,
                    "transformed_input": 1.0 - self.confidence_band,
                    "weight": weights["confidence_band"],
                    "contribution": confidence_contribution
                }
            },
            "aggregation": {
                "total_logit": alignment_contribution + risk_contribution + confidence_contribution,
                "raw_trust_score": self._raw_score,
                "display_trust_score": self.trust_score,
                "calibrated_score": self.calibrated_score
            },
            "risk_assessment": {
                "tier": self.risk_tier,
                "tier_thresholds": self._config["risk_tiers"],
                "tier_fallback_applied": self.risk_tier != self._get_original_tier(),
                "entropy_flag": self.entropy_flag,
                "trust_floor_met": self.trust_floor_met
            },
            "decision_routing": self.downstream_action(),
            "observability": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config_hash": self._get_config_hash(),
                "calibration_enabled": self._calibration_model is not None
            }
        }
        
        return explanation

    def _get_original_tier(self) -> str:
        """Get tier without fallback logic for explanation."""
        score = self.calibrated_score or self._raw_score
        for tier, threshold in self._config["risk_tiers"].items():
            if score <= threshold:
                return tier
        return "undefined"

    def _get_config_hash(self) -> str:
        """Generate hash of current configuration for audit trails."""
        config_str = json.dumps(self._config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

##################################
# 5. TRUSTMONITOR INTEGRATION
##################################

class TrustMonitor:
    """Integration class with calibration model support and observability."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.calibration_model: Optional[CalibrationModel] = None
        self.logger = logging.getLogger("resontinex.trust_monitor")
        self.metrics_sink = MetricsSink()
        self.audit_log = AuditLog()
        
        # Initialize from RESONTINEX fusion system
        self.fusion_loader = get_fusion_loader()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with RESONTINEX integration."""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith(('.yml', '.yaml')):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)
                return validate_config_and_normalize(config)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return TRUST_GATE_CONFIG.copy()

    @measure("trust_gate_evaluation")
    def evaluate(self, epistemic_risk: float, alignment_score: float, 
                confidence_band: float) -> CompositeTrustGate:
        """Evaluate trust with full observability."""
        gate = CompositeTrustGate(
            epistemic_risk=epistemic_risk,
            alignment_score=alignment_score,
            confidence_band=confidence_band,
            _config=self.config,
            _calibration_model=self.calibration_model
        )
        
        # Collect metrics
        self.metrics_sink.record_evaluation(gate)
        
        # Audit logging
        if self.config.get("observability", {}).get("audit_logging", True):
            self.audit_log.log_evaluation(gate)
        
        return gate

    def fit_calibration(self, scores: List[float], labels: List[bool], 
                       method: str = "isotonic") -> bool:
        """Fit calibration model and update configuration."""
        try:
            self.calibration_model = fit_calibration_model(scores, labels, method)
            if self.calibration_model:
                self.logger.info(f"Calibration model fitted using {method} method")
                return True
            else:
                self.logger.warning("Insufficient data for calibration")
                return False
        except Exception as e:
            self.logger.error(f"Calibration fitting failed: {e}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "status": "healthy",
            "calibration_enabled": self.calibration_model is not None,
            "config_valid": True,
            "metrics_collected": self.metrics_sink.get_count(),
            "last_evaluation": self.metrics_sink.get_last_timestamp(),
            "fusion_loader_status": self.fusion_loader.get_health_status()
        }

class MetricsSink:
    """Metrics collection for observability."""
    
    def __init__(self):
        self.evaluation_count = 0
        self.last_timestamp = None
        self.score_distribution = []
    
    def record_evaluation(self, gate: CompositeTrustGate):
        """Record evaluation metrics."""
        self.evaluation_count += 1
        self.last_timestamp = datetime.now(timezone.utc).isoformat()
        self.score_distribution.append(gate.trust_score)
        
        # Keep last 1000 scores for distribution analysis
        if len(self.score_distribution) > 1000:
            self.score_distribution = self.score_distribution[-1000:]
    
    def get_count(self) -> int:
        return self.evaluation_count
    
    def get_last_timestamp(self) -> Optional[str]:
        return self.last_timestamp

class AuditLog:
    """Structured audit logging."""
    
    def __init__(self):
        self.logger = logging.getLogger("resontinex.audit.trust_gate")
    
    def log_evaluation(self, gate: CompositeTrustGate):
        """Log evaluation with full context."""
        audit_entry = {
            "event": "trust_gate_evaluation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trust_gate": gate.to_dict(),
            "explanation": gate.explain_score(),
            "decision": gate.downstream_action()
        }
        
        self.logger.info(json.dumps(audit_entry))

##################################
# 6. INTEGRATION UTILITIES
##################################

def create_from_scenario_metrics(metrics: Dict[str, float], 
                                config_path: Optional[str] = None) -> CompositeTrustGate:
    """Create CompositeTrustGate from ScenarioManager metrics."""
    config = TRUST_GATE_CONFIG if config_path is None else validate_config_and_normalize(
        json.load(open(config_path)) if config_path else TRUST_GATE_CONFIG
    )
    
    return CompositeTrustGate(
        epistemic_risk=1.0 - metrics.get("reliability_index", 0.5),
        alignment_score=metrics.get("success_rate", 0.5),
        confidence_band=1.0 - metrics.get("resource_efficiency", 0.5),
        _config=config
    )

def enhance_trust_manager_scoring(alignment_score: float, inflation_delta: float,
                                epistemic_risk: float, confidence_band: float,
                                config_path: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced TrustManager scoring with CompositeTrustGate."""
    # Incorporate inflation delta into alignment adjustment
    adjusted_alignment = np.clip(alignment_score - abs(inflation_delta) * 0.1, 0.0, 1.0)
    
    gate = CompositeTrustGate(
        epistemic_risk=epistemic_risk,
        alignment_score=adjusted_alignment,
        confidence_band=confidence_band,
        _config=validate_config_and_normalize(
            json.load(open(config_path)) if config_path else TRUST_GATE_CONFIG
        )
    )
    
    return {
        "trust_gate": gate.to_dict(),
        "downstream_action": gate.downstream_action(),
        "explanation": gate.explain_score(),
        "legacy_score": alignment_score + inflation_delta  # Backward compatibility
    }

##################################
# 7. PRODUCTION DEPLOYMENT SUPPORT
##################################

def load_production_config(environment: str = "production") -> Dict[str, Any]:
    """Load environment-specific configuration."""
    config_paths = [
        f"./configs/fusion/trust_gate_{environment}.yaml",
        f"./config/trust_gate_{environment}.yaml",
        "./configs/fusion/trust_gate.yaml"
    ]
    
    for path in config_paths:
        if Path(path).exists():
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            return validate_config_and_normalize(config)
    
    return TRUST_GATE_CONFIG.copy()

def break_glass_override(gate: CompositeTrustGate, 
                        override_reason: str,
                        operator_id: str) -> Dict[str, Any]:
    """Break-glass override procedure for critical situations."""
    override_entry = {
        "event": "break_glass_override",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_gate": gate.to_dict(),
        "override_reason": override_reason,
        "operator_id": operator_id,
        "original_decision": gate.downstream_action()["decision"],
        "overridden_decision": "manual_approval"
    }
    
    # Log to audit trail
    logger = logging.getLogger("resontinex.audit.break_glass")
    logger.critical(json.dumps(override_entry))
    
    return override_entry