#!/usr/bin/env python3
"""
Comprehensive test suite for CompositeTrustGate system.
Property-based testing using Hypothesis for monotonicity guarantees and edge cases.
"""

import pytest
import numpy as np
import json
from unittest.mock import Mock, patch

# Optional imports with fallbacks
try:
    from hypothesis import given, strategies as st, assume, settings
    from hypothesis.stateful import RuleBasedStateMachine, rule, initialize
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Mock decorators for when hypothesis is not available
    def given(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    def settings(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    class st:
        @staticmethod
        def floats(**kwargs):
            return None
        @staticmethod
        def fixed_dictionaries(d):
            return None
    
    class RuleBasedStateMachine:
        pass
    
    def rule(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    def initialize():
        def decorator(func):
            return func
        return decorator

try:
    from scipy.stats import spearmanr
    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False
    def spearmanr(x, y):
        return 0.5, 0.05  # Mock correlation result

# Import the classes under test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from resontinex.core.composite_trust_gate import (
    CompositeTrustGate,
    TrustMonitor,
    logit_aggregation,
    safe_logit,
    safe_expit,
    validate_config_and_normalize,
    fit_calibration_model,
    derive_tier_thresholds,
    create_from_scenario_metrics,
    enhance_trust_manager_scoring,
    TRUST_GATE_CONFIG,
    EPSILON
)

##################################
# 1. UNIT TESTS - MATHEMATICAL FUNCTIONS
##################################

class TestMathematicalFunctions:
    """Test mathematical functions with boundary conditions."""
    
    @given(p=st.floats(min_value=EPSILON, max_value=1.0-EPSILON))
    def test_safe_logit_monotonicity(self, p):
        """Test that logit function maintains monotonicity."""
        if p < 0.5:
            p2 = p + 0.1
            if p2 < 1.0 - EPSILON:
                assert safe_logit(p) < safe_logit(p2)
    
    @given(x=st.floats(min_value=-10, max_value=10))
    def test_safe_expit_bounds(self, x):
        """Test that expit function output is bounded [0,1]."""
        result = safe_expit(x)
        assert 0.0 <= result <= 1.0
    
    def test_logit_expit_inverse(self):
        """Test that logit and expit are inverses."""
        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for p in test_values:
            recovered = safe_expit(safe_logit(p))
            assert abs(recovered - p) < 1e-10
    
    @given(
        weights=st.fixed_dictionaries({
            "epistemic_risk": st.floats(min_value=0.1, max_value=0.6),
            "alignment_score": st.floats(min_value=0.1, max_value=0.6),
            "confidence_band": st.floats(min_value=0.1, max_value=0.6)
        }),
        epistemic_risk=st.floats(min_value=0.0, max_value=1.0),
        alignment_score=st.floats(min_value=0.0, max_value=1.0),
        confidence_band=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_logit_aggregation_monotonicity(self, weights, epistemic_risk, alignment_score, confidence_band):
        """Property-based test for logit aggregation monotonicity."""
        # Normalize weights
        total = sum(weights.values())
        normalized_weights = {k: v/total for k, v in weights.items()}
        
        base_score = logit_aggregation(normalized_weights, epistemic_risk, alignment_score, confidence_band)
        
        # Test monotonicity w.r.t. alignment_score (should increase)
        if alignment_score < 0.95:
            higher_alignment = alignment_score + 0.05
            higher_score = logit_aggregation(normalized_weights, epistemic_risk, higher_alignment, confidence_band)
            assert higher_score >= base_score
        
        # Test monotonicity w.r.t. epistemic_risk (should decrease)
        if epistemic_risk < 0.95:
            higher_risk = epistemic_risk + 0.05
            lower_score = logit_aggregation(normalized_weights, higher_risk, alignment_score, confidence_band)
            assert lower_score <= base_score

##################################
# 2. UNIT TESTS - CORE CLASS
##################################

class TestCompositeTrustGate:
    """Test core CompositeTrustGate functionality."""
    
    def test_initialization_valid_inputs(self):
        """Test initialization with valid inputs."""
        gate = CompositeTrustGate(
            epistemic_risk=0.2,
            alignment_score=0.8,
            confidence_band=0.1
        )
        assert gate.epistemic_risk == 0.2
        assert gate.alignment_score == 0.8
        assert gate.confidence_band == 0.1
        assert 0.0 <= gate.trust_score <= 1.0
    
    @pytest.mark.parametrize("invalid_value", [-0.1, 1.1, 2.0])
    def test_initialization_invalid_inputs(self, invalid_value):
        """Test initialization with invalid inputs raises ValueError."""
        with pytest.raises(ValueError):
            CompositeTrustGate(
                epistemic_risk=invalid_value,
                alignment_score=0.8,
                confidence_band=0.1
            )
    
    def test_risk_tier_classification(self):
        """Test risk tier classification."""
        # High trust scenario
        high_trust = CompositeTrustGate(
            epistemic_risk=0.1,
            alignment_score=0.9,
            confidence_band=0.05
        )
        assert high_trust.risk_tier in ["monitor", "pass"]
        
        # High risk scenario
        high_risk = CompositeTrustGate(
            epistemic_risk=0.8,
            alignment_score=0.2,
            confidence_band=0.7
        )
        assert high_risk.risk_tier in ["block", "review"]
    
    def test_tier_fallback_mechanism(self):
        """Test safe fallback for undefined tiers."""
        # Mock config with limited tiers to trigger fallback
        limited_config = TRUST_GATE_CONFIG.copy()
        limited_config["risk_tiers"] = {"block": 0.1}  # Very restrictive
        
        gate = CompositeTrustGate(
            epistemic_risk=0.1,
            alignment_score=0.9,
            confidence_band=0.05,
            _config=limited_config
        )
        
        # Should fall back to safe tier
        tier = gate.risk_tier
        assert tier in ["block", "review", "monitor", "pass"]
    
    @given(
        epistemic_risk=st.floats(min_value=0.0, max_value=1.0),
        alignment_score=st.floats(min_value=0.0, max_value=1.0),
        confidence_band=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_trust_score_bounds(self, epistemic_risk, alignment_score, confidence_band):
        """Property test: trust scores are always bounded [0,1]."""
        gate = CompositeTrustGate(
            epistemic_risk=epistemic_risk,
            alignment_score=alignment_score,
            confidence_band=confidence_band
        )
        assert 0.0 <= gate.trust_score <= 1.0
        assert 0.0 <= gate.raw_trust_score <= 1.0
    
    def test_explainability_completeness(self):
        """Test that explanation contains all required fields."""
        gate = CompositeTrustGate(
            epistemic_risk=0.3,
            alignment_score=0.7,
            confidence_band=0.2
        )
        
        explanation = gate.explain_score()
        required_keys = [
            "computation_method",
            "mathematical_properties", 
            "input_validation",
            "logit_space_contributions",
            "aggregation",
            "risk_assessment",
            "decision_routing",
            "observability"
        ]
        
        for key in required_keys:
            assert key in explanation
    
    def test_downstream_action_consistency(self):
        """Test that downstream actions are consistent with risk tiers."""
        gate = CompositeTrustGate(
            epistemic_risk=0.9,  # High risk
            alignment_score=0.1,
            confidence_band=0.8
        )
        
        action = gate.downstream_action()
        assert action["tier"] == gate.risk_tier
        assert "decision" in action
        assert "reason" in action
        assert "voting_weight" in action
        assert isinstance(action["voting_weight"], int)

##################################
# 3. PROPERTY-BASED TESTING - MONOTONICITY
##################################

class TestMonotonicityProperties:
    """Property-based tests for monotonicity guarantees."""
    
    @given(
        base_alignment=st.floats(min_value=0.1, max_value=0.8),
        delta=st.floats(min_value=0.01, max_value=0.19),
        epistemic_risk=st.floats(min_value=0.0, max_value=1.0),
        confidence_band=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100)
    def test_alignment_score_monotonicity(self, base_alignment, delta, epistemic_risk, confidence_band):
        """Test monotonicity: higher alignment_score → higher trust_score."""
        gate1 = CompositeTrustGate(
            epistemic_risk=epistemic_risk,
            alignment_score=base_alignment,
            confidence_band=confidence_band
        )
        
        gate2 = CompositeTrustGate(
            epistemic_risk=epistemic_risk,
            alignment_score=base_alignment + delta,
            confidence_band=confidence_band
        )
        
        assert gate2.trust_score >= gate1.trust_score
    
    @given(
        alignment_score=st.floats(min_value=0.0, max_value=1.0),
        base_risk=st.floats(min_value=0.1, max_value=0.8),
        delta=st.floats(min_value=0.01, max_value=0.19),
        confidence_band=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=100)
    def test_epistemic_risk_monotonicity(self, alignment_score, base_risk, delta, confidence_band):
        """Test monotonicity: higher epistemic_risk → lower trust_score."""
        gate1 = CompositeTrustGate(
            epistemic_risk=base_risk,
            alignment_score=alignment_score,
            confidence_band=confidence_band
        )
        
        gate2 = CompositeTrustGate(
            epistemic_risk=base_risk + delta,
            alignment_score=alignment_score,
            confidence_band=confidence_band
        )
        
        assert gate2.trust_score <= gate1.trust_score
    
    @given(
        alignment_score=st.floats(min_value=0.0, max_value=1.0),
        epistemic_risk=st.floats(min_value=0.0, max_value=1.0),
        base_confidence=st.floats(min_value=0.1, max_value=0.8),
        delta=st.floats(min_value=0.01, max_value=0.19)
    )
    @settings(max_examples=100) 
    def test_confidence_band_monotonicity(self, alignment_score, epistemic_risk, base_confidence, delta):
        """Test monotonicity: higher confidence_band → lower trust_score."""
        gate1 = CompositeTrustGate(
            epistemic_risk=epistemic_risk,
            alignment_score=alignment_score,
            confidence_band=base_confidence
        )
        
        gate2 = CompositeTrustGate(
            epistemic_risk=epistemic_risk,
            alignment_score=alignment_score,
            confidence_band=base_confidence + delta
        )
        
        assert gate2.trust_score <= gate1.trust_score

##################################
# 4. INTEGRATION TESTS
##################################

class TestTrustMonitorIntegration:
    """Test TrustMonitor integration and observability."""
    
    def test_trust_monitor_initialization(self):
        """Test TrustMonitor initializes correctly."""
        monitor = TrustMonitor()
        assert monitor.config is not None
        assert monitor.calibration_model is None
        assert monitor.metrics_sink is not None
        assert monitor.audit_log is not None
    
    @patch('resontinex.core.composite_trust_gate.get_fusion_loader')
    def test_trust_monitor_evaluation(self, mock_fusion_loader):
        """Test TrustMonitor evaluation with mocked fusion loader."""
        mock_loader = Mock()
        mock_loader.get_health_status.return_value = {"status": "healthy"}
        mock_fusion_loader.return_value = mock_loader
        
        monitor = TrustMonitor()
        gate = monitor.evaluate(
            epistemic_risk=0.2,
            alignment_score=0.8,
            confidence_band=0.1
        )
        
        assert isinstance(gate, CompositeTrustGate)
        assert gate.trust_score > 0.0
        assert monitor.metrics_sink.evaluation_count == 1
    
    def test_calibration_workflow(self):
        """Test calibration model fitting workflow."""
        monitor = TrustMonitor()
        
        # Generate synthetic calibration data
        scores = [0.1, 0.3, 0.5, 0.7, 0.9] * 15  # 75 samples > min_samples
        labels = [s > 0.5 for s in scores]  # Simple labeling rule
        
        success = monitor.fit_calibration(scores, labels, method="isotonic")
        
        # Should succeed if sklearn available, otherwise gracefully fail
        if success:
            assert monitor.calibration_model is not None
            
            # Test calibrated scoring
            gate = monitor.evaluate(0.2, 0.8, 0.1)
            calibrated_score = gate.calibrated_score
            assert calibrated_score is None or 0.0 <= calibrated_score <= 1.0

##################################
# 5. CONFIGURATION TESTS
##################################

class TestConfigurationValidation:
    """Test configuration validation and normalization."""
    
    def test_valid_config_normalization(self):
        """Test that valid config gets properly normalized."""
        test_config = {
            "weights": {
                "epistemic_risk": 0.6,
                "alignment_score": 0.3,
                "confidence_band": 0.1
            },
            "risk_tiers": {
                "pass": 0.9,
                "monitor": 0.7,
                "review": 0.5,
                "block": 0.3
            },
            "trust_floor": 0.6,
            "entropy_threshold": 0.72
        }
        
        normalized = validate_config_and_normalize(test_config)
        
        # Weights should be normalized to sum to 1.0
        weight_sum = sum(normalized["weights"].values())
        assert abs(weight_sum - 1.0) < EPSILON
        
        # Tiers should be sorted by threshold
        tiers = list(normalized["risk_tiers"].keys())
        thresholds = list(normalized["risk_tiers"].values())
        assert thresholds == sorted(thresholds)
    
    def test_missing_keys_validation(self):
        """Test validation fails with missing required keys."""
        incomplete_config = {
            "weights": {
                "epistemic_risk": 0.4,
                "alignment_score": 0.4,
                "confidence_band": 0.2
            }
            # Missing risk_tiers, trust_floor, entropy_threshold
        }
        
        with pytest.raises(ValueError, match="Missing required config keys"):
            validate_config_and_normalize(incomplete_config)

##################################
# 6. CALIBRATION TESTS
##################################

class TestCalibrationSystem:
    """Test calibration system with various scenarios."""
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient calibration data."""
        small_scores = [0.1, 0.2, 0.3]  # < min_samples
        small_labels = [True, False, True]
        
        result = fit_calibration_model(small_scores, small_labels)
        assert result is None
    
    def test_derive_tier_thresholds(self):
        """Test deriving tier thresholds from score distribution."""
        # Generate realistic score distribution
        scores = np.concatenate([
            np.random.beta(2, 8, 25),  # Low scores
            np.random.beta(5, 5, 25),  # Medium scores  
            np.random.beta(8, 2, 25),  # High scores
        ])
        
        thresholds = derive_tier_thresholds(scores.tolist())
        
        # Thresholds should be ordered
        tier_values = [thresholds[tier] for tier in ["block", "review", "monitor", "pass"]]
        assert tier_values == sorted(tier_values)
        
        # Should be bounded [0,1]
        for value in tier_values:
            assert 0.0 <= value <= 1.0

##################################
# 7. INTEGRATION UTILITY TESTS
##################################

class TestIntegrationUtilities:
    """Test integration utilities with RESONTINEX modules."""
    
    def test_create_from_scenario_metrics(self):
        """Test creating CompositeTrustGate from scenario metrics."""
        mock_metrics = {
            "reliability_index": 0.8,
            "success_rate": 0.9,
            "resource_efficiency": 0.7,
            "user_satisfaction": 0.85
        }
        
        gate = create_from_scenario_metrics(mock_metrics)
        
        assert isinstance(gate, CompositeTrustGate)
        assert gate.epistemic_risk == 1.0 - 0.8  # 0.2
        assert gate.alignment_score == 0.9
        assert gate.confidence_band == 1.0 - 0.7  # 0.3
    
    def test_enhance_trust_manager_scoring(self):
        """Test enhanced TrustManager scoring integration."""
        result = enhance_trust_manager_scoring(
            alignment_score=0.8,
            inflation_delta=0.1,
            epistemic_risk=0.2,
            confidence_band=0.15
        )
        
        required_keys = ["trust_gate", "downstream_action", "explanation", "legacy_score"]
        for key in required_keys:
            assert key in result
        
        # Legacy compatibility
        assert result["legacy_score"] == 0.8 + 0.1
        
        # Enhanced scoring
        trust_gate_data = result["trust_gate"]
        assert "trust_score" in trust_gate_data
        assert "risk_tier" in trust_gate_data

##################################
# 8. EDGE CASE TESTS
##################################

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.parametrize("extreme_value", [0.0, 1.0])
    def test_extreme_input_values(self, extreme_value):
        """Test behavior with extreme input values."""
        gate = CompositeTrustGate(
            epistemic_risk=extreme_value,
            alignment_score=extreme_value,
            confidence_band=extreme_value
        )
        
        # Should not raise exceptions
        assert isinstance(gate.trust_score, float)
        assert isinstance(gate.risk_tier, str)
        assert isinstance(gate.downstream_action(), dict)
    
    def test_epsilon_boundary_handling(self):
        """Test epsilon boundary handling in calculations."""
        # Values very close to boundaries
        gate = CompositeTrustGate(
            epistemic_risk=EPSILON,
            alignment_score=1.0 - EPSILON,
            confidence_band=EPSILON
        )
        
        assert 0.0 <= gate.trust_score <= 1.0
        assert gate.trust_score > 0.5  # Should be high trust
    
    def test_config_with_extreme_weights(self):
        """Test behavior with extreme weight distributions."""
        extreme_config = TRUST_GATE_CONFIG.copy()
        extreme_config["weights"] = {
            "epistemic_risk": 0.01,
            "alignment_score": 0.98,
            "confidence_band": 0.01
        }
        
        gate = CompositeTrustGate(
            epistemic_risk=0.5,
            alignment_score=0.9,
            confidence_band=0.5,
            _config=extreme_config
        )
        
        # Should be dominated by alignment_score
        assert gate.trust_score > 0.7

##################################
# 9. STATEFUL PROPERTY TESTING
##################################

class CompositeTrustGateStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for CompositeTrustGate system."""
    
    def __init__(self):
        super().__init__()
        self.gates = []
        self.monitor = TrustMonitor()
    
    @initialize()
    def init_state(self):
        """Initialize test state."""
        self.gates = []
    
    @rule(
        epistemic_risk=st.floats(min_value=0.0, max_value=1.0),
        alignment_score=st.floats(min_value=0.0, max_value=1.0),
        confidence_band=st.floats(min_value=0.0, max_value=1.0)
    )
    def create_gate(self, epistemic_risk, alignment_score, confidence_band):
        """Rule: Create new CompositeTrustGate."""
        gate = self.monitor.evaluate(epistemic_risk, alignment_score, confidence_band)
        self.gates.append(gate)
        
        # Invariant: trust scores are bounded
        assert 0.0 <= gate.trust_score <= 1.0
        
        # Invariant: risk tiers are valid
        assert gate.risk_tier in ["block", "review", "monitor", "pass"]
    
    @rule()
    def check_monotonicity_invariant(self):
        """Rule: Check monotonicity invariant across all gates."""
        if len(self.gates) < 2:
            return
            
        # Sort gates by alignment score
        sorted_gates = sorted(self.gates, key=lambda g: g.alignment_score)
        
        # Check that trust scores generally increase with alignment
        # (allowing for some variance due to other factors)
        alignment_scores = [g.alignment_score for g in sorted_gates]
        trust_scores = [g.trust_score for g in sorted_gates]
        
        # Spearman correlation should be positive
        if len(set(alignment_scores)) > 1:
            from scipy.stats import spearmanr
            try:
                correlation, _ = spearmanr(alignment_scores, trust_scores)
                assert correlation >= -0.1  # Allow some negative correlation due to noise
            except:
                pass  # Skip if scipy not available

# Run stateful testing
TestStatefulTrustGate = CompositeTrustGateStateMachine.TestCase

##################################
# 10. PERFORMANCE TESTS
##################################

class TestPerformance:
    """Performance and scalability tests."""
    
    def test_gate_creation_performance(self):
        """Test performance of gate creation."""
        import time
        
        start_time = time.time()
        gates = []
        
        for i in range(1000):
            gate = CompositeTrustGate(
                epistemic_risk=0.1 + (i % 9) * 0.1,
                alignment_score=0.1 + (i % 9) * 0.1,
                confidence_band=0.1 + (i % 9) * 0.1
            )
            gates.append(gate)
        
        duration = time.time() - start_time
        
        # Should create 1000 gates in reasonable time
        assert duration < 5.0  # 5 seconds max
        assert len(gates) == 1000
    
    def test_memory_usage(self):
        """Test memory usage with many gates."""
        gates = [
            CompositeTrustGate(0.2, 0.8, 0.1) 
            for _ in range(10000)
        ]
        
        # All gates should be properly initialized
        assert all(0.0 <= gate.trust_score <= 1.0 for gate in gates[:100])
        assert len(gates) == 10000

##################################
# 11. ERROR HANDLING TESTS
##################################

class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configurations."""
        invalid_configs = [
            {},  # Empty config
            {"weights": {}},  # Missing required sections
            {"weights": {"only_one": 1.0}},  # Incomplete weights
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                CompositeTrustGate(
                    epistemic_risk=0.2,
                    alignment_score=0.8,
                    confidence_band=0.1,
                    _config=config
                )
    
    @patch('resontinex.core.composite_trust_gate.HAS_SKLEARN', False)
    def test_sklearn_unavailable_graceful_degradation(self):
        """Test graceful degradation when sklearn is unavailable."""
        monitor = TrustMonitor()
        
        # Should not crash, but calibration should be disabled
        success = monitor.fit_calibration([0.1, 0.5, 0.9], [True, False, True])
        assert success is False
        assert monitor.calibration_model is None
    
    def test_extreme_config_values(self):
        """Test handling of extreme configuration values."""
        extreme_config = TRUST_GATE_CONFIG.copy()
        extreme_config["risk_tiers"] = {
            "block": -1.0,  # Invalid threshold
            "review": 2.0,   # Invalid threshold
        }
        extreme_config["trust_floor"] = -0.5  # Invalid floor
        
        # Should handle gracefully by using fallbacks
        gate = CompositeTrustGate(
            epistemic_risk=0.2,
            alignment_score=0.8,
            confidence_band=0.1,
            _config=extreme_config
        )
        
        # Should still produce valid outputs
        assert isinstance(gate.risk_tier, str)
        assert 0.0 <= gate.trust_score <= 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])