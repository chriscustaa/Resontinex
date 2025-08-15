"""
Comprehensive test suite for CompositeTrustGate system
Tests functionality, performance, and integration with RESONTINEX patterns.
"""

import pytest
import time
import json
import copy
from typing import Dict, Any

from resontinex.trust.composite_trust_gate import (
    CompositeTrustGate,
    TrustMonitor,
    NullMetricsClient,
    MetricsClient,
    RiskTier,
    Decision,
    InputVector,
    route_to_resontinex,
    CalibrationAdapter,
    DEFAULT_CONFIG,
    validate_and_normalize_config,
    logit_aggregation,
    _sigmoid,
    _safe_log,
    _logit_transform,
)


class TestInputVector:
    """Test input vector validation and boundary conditions."""
    
    def test_valid_inputs(self):
        """Test valid input creation."""
        vector = InputVector(
            alignment_score=0.8,
            epistemic_risk=0.3,
            confidence_band=0.2
        )
        assert vector.alignment_score == 0.8
        assert vector.epistemic_risk == 0.3
        assert vector.confidence_band == 0.2
    
    def test_boundary_values(self):
        """Test exact boundary values [0,1]."""
        # Test minimum boundaries
        vector_min = InputVector(
            alignment_score=0.0,
            epistemic_risk=0.0,
            confidence_band=0.0
        )
        assert vector_min.alignment_score == 0.0
        
        # Test maximum boundaries  
        vector_max = InputVector(
            alignment_score=1.0,
            epistemic_risk=1.0,
            confidence_band=1.0
        )
        assert vector_max.alignment_score == 1.0
    
    def test_invalid_type_validation(self):
        """Test type validation for inputs."""
        with pytest.raises(TypeError, match="alignment_score must be numeric"):
            InputVector(
                alignment_score="invalid",  # type: ignore  # Intentional invalid type for testing
                epistemic_risk=0.5,
                confidence_band=0.5
            )
    
    def test_out_of_bounds_validation(self):
        """Test bounds validation [0,1]."""
        with pytest.raises(ValueError, match="alignment_score must be in \\[0,1\\]"):
            InputVector(
                alignment_score=1.5,
                epistemic_risk=0.5,
                confidence_band=0.5
            )
        
        with pytest.raises(ValueError, match="epistemic_risk must be in \\[0,1\\]"):
            InputVector(
                alignment_score=0.5,
                epistemic_risk=-0.1,
                confidence_band=0.5
            )


class TestMathematicalOperations:
    """Test zero-dependency mathematical operations."""
    
    def test_sigmoid_monotonicity(self):
        """Test sigmoid monotonicity property."""
        x_values = [-10, -1, 0, 1, 10]
        y_values = [_sigmoid(x) for x in x_values]
        
        # Verify monotonically increasing
        for i in range(len(y_values) - 1):
            assert y_values[i] < y_values[i + 1], "Sigmoid must be monotonically increasing"
        
        # Verify bounds [0,1]
        for y in y_values:
            assert 0.0 <= y <= 1.0, "Sigmoid output must be in [0,1]"
    
    def test_sigmoid_numerical_stability(self):
        """Test sigmoid numerical stability for extreme values."""
        # Large positive values should approach 1.0
        assert _sigmoid(500) == pytest.approx(1.0, abs=1e-6)
        
        # Large negative values should approach 0.0  
        assert _sigmoid(-500) == pytest.approx(0.0, abs=1e-6)
    
    def test_logit_transform_monotonicity(self):
        """Test logit transform monotonicity."""
        p_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        logit_values = [_logit_transform(p) for p in p_values]
        
        # Verify monotonically increasing
        for i in range(len(logit_values) - 1):
            assert logit_values[i] < logit_values[i + 1], "Logit transform must be monotonically increasing"
    
    def test_logit_aggregation_properties(self):
        """Test logit aggregation mathematical guarantees."""
        weights = {"alignment_score": 0.5, "epistemic_risk": 0.3, "confidence_band": 0.2}
        
        # Test monotonicity with alignment_score
        inputs_low = InputVector(0.2, 0.5, 0.5)
        inputs_high = InputVector(0.8, 0.5, 0.5)
        
        score_low = logit_aggregation(weights, inputs_low)
        score_high = logit_aggregation(weights, inputs_high)
        
        assert score_high > score_low, "Higher alignment should increase trust score"
        
        # Test anti-monotonicity with epistemic_risk
        inputs_low_risk = InputVector(0.5, 0.2, 0.5)
        inputs_high_risk = InputVector(0.5, 0.8, 0.5)
        
        score_low_risk = logit_aggregation(weights, inputs_low_risk)
        score_high_risk = logit_aggregation(weights, inputs_high_risk)
        
        assert score_low_risk > score_high_risk, "Higher epistemic risk should decrease trust score"


class TestConfigValidation:
    """Test configuration validation and normalization."""
    
    def test_valid_config_normalization(self):
        """Test valid config is normalized correctly."""
        config = {
            "weights": {"alignment_score": 0.9, "epistemic_risk": 0.6, "confidence_band": 0.5},
            "risk_tiers": {"block": 0.3, "review": 0.6, "monitor": 0.8, "pass": 1.0},
            "trust_floor": 0.5,
            "entropy_threshold": 0.7
        }
        
        normalized = validate_and_normalize_config(config)
        
        # Check weights are normalized to sum to 1.0
        total_weight = sum(normalized["weights"].values())
        assert total_weight == pytest.approx(1.0, abs=1e-10)
        
        # Check tiers are sorted by threshold
        tier_items = list(normalized["risk_tiers"].items())
        for i in range(len(tier_items) - 1):
            assert tier_items[i][1] <= tier_items[i + 1][1], "Tiers must be sorted by threshold"
    
    def test_missing_required_keys(self):
        """Test validation fails for missing required keys."""
        config = {"weights": {"alignment_score": 1.0}}
        
        with pytest.raises(ValueError, match="Missing required config keys"):
            validate_and_normalize_config(config)
    
    def test_invalid_weights(self):
        """Test validation fails for invalid weight configuration."""
        config = {
            "weights": {"alignment_score": 0.5},  # Missing required weight keys
            "risk_tiers": {"block": 0.5},
            "trust_floor": 0.5,
            "entropy_threshold": 0.7
        }
        
        with pytest.raises(ValueError, match="Missing required weight keys"):
            validate_and_normalize_config(config)
    
    def test_zero_weight_sum(self):
        """Test validation fails for zero weight sum."""
        config = {
            "weights": {"alignment_score": 0.0, "epistemic_risk": 0.0, "confidence_band": 0.0},
            "risk_tiers": {"block": 0.5},
            "trust_floor": 0.5,
            "entropy_threshold": 0.7
        }
        
        with pytest.raises(ValueError, match="Weight sum must be positive"):
            validate_and_normalize_config(config)


class TestCompositeTrustGate:
    """Test core CompositeTrustGate functionality."""
    
    def test_basic_evaluation(self):
        """Test basic trust gate evaluation."""
        inputs = InputVector(
            alignment_score=0.8,
            epistemic_risk=0.3,
            confidence_band=0.2
        )
        
        gate = CompositeTrustGate(inputs=inputs)
        
        # Verify properties are accessible
        assert isinstance(gate.trust_score, float)
        assert isinstance(gate.raw_trust_score, float)
        assert isinstance(gate.risk_tier, RiskTier)
        assert isinstance(gate.decision, Decision)
        assert isinstance(gate.entropy_flag, bool)
        assert isinstance(gate.trust_floor_met, bool)
    
    def test_deterministic_behavior(self):
        """Test deterministic behavior with same inputs."""
        inputs = InputVector(0.7, 0.4, 0.3)
        
        gate1 = CompositeTrustGate(inputs=inputs)
        gate2 = CompositeTrustGate(inputs=inputs)
        
        # Same inputs should produce identical outputs
        assert gate1.trust_score == gate2.trust_score
        assert gate1.raw_trust_score == gate2.raw_trust_score
        assert gate1.risk_tier == gate2.risk_tier
        assert gate1.decision == gate2.decision
    
    def test_risk_tier_assignment(self):
        """Test risk tier assignment logic."""
        # Test block tier
        low_trust_inputs = InputVector(0.1, 0.9, 0.8)
        low_gate = CompositeTrustGate(inputs=low_trust_inputs)
        assert low_gate.risk_tier in [RiskTier.BLOCK, RiskTier.REVIEW]
        
        # Test pass tier  
        high_trust_inputs = InputVector(0.9, 0.1, 0.1)
        high_gate = CompositeTrustGate(inputs=high_trust_inputs)
        assert high_gate.risk_tier in [RiskTier.MONITOR, RiskTier.PASS]
    
    def test_explain_completeness(self):
        """Test explanation provides complete audit trail."""
        inputs = InputVector(0.8, 0.3, 0.2)
        gate = CompositeTrustGate(inputs=inputs)
        
        explanation = gate.explain()
        
        # Verify key sections present
        required_sections = [
            "computation_method",
            "performance_characteristics", 
            "mathematical_guarantees",
            "inputs",
            "logit_contributions",
            "aggregation",
            "risk_assessment",
            "observability"
        ]
        
        for section in required_sections:
            assert section in explanation, f"Missing explanation section: {section}"
        
        # Verify mathematical guarantees documented
        guarantees = explanation["mathematical_guarantees"]
        assert "monotonic_alignment" in guarantees
        assert "monotonic_risk" in guarantees
        assert "deterministic" in guarantees
    
    def test_custom_config(self):
        """Test custom configuration handling."""
        custom_config = copy.deepcopy(DEFAULT_CONFIG)
        custom_config["trust_floor"] = 0.8
        custom_config["entropy_threshold"] = 0.6
        
        inputs = InputVector(0.7, 0.3, 0.4)
        gate = CompositeTrustGate(inputs=inputs, config=custom_config)
        
        # Verify custom config is applied
        assert gate.config["trust_floor"] == 0.8
        assert gate.config["entropy_threshold"] == 0.6
    
    def test_immutability(self):
        """Test dataclass immutability."""
        inputs = InputVector(0.5, 0.5, 0.5)
        gate = CompositeTrustGate(inputs=inputs)
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(AttributeError):
            gate.trust_score = 0.9  # type: ignore  # Intentional invalid assignment for testing


class TestTrustMonitor:
    """Test TrustMonitor observability system."""
    
    def test_evaluation_with_metrics(self):
        """Test evaluation with metrics collection."""
        metrics_client = NullMetricsClient()
        monitor = TrustMonitor(metrics_client=metrics_client)
        
        initial_call_count = metrics_client.get_call_count()
        
        gate = monitor.evaluate(0.8, 0.3, 0.2)
        
        # Verify gate created successfully
        assert isinstance(gate, CompositeTrustGate)
        
        # Verify metrics were collected
        final_call_count = metrics_client.get_call_count()
        assert final_call_count > initial_call_count, "Metrics should be collected"
    
    def test_health_status(self):
        """Test health status reporting."""
        monitor = TrustMonitor()
        status = monitor.get_health_status()
        
        required_fields = [
            "status",
            "uptime_seconds", 
            "evaluation_count",
            "metrics_client_calls",
            "config_hash",
            "performance_target"
        ]
        
        for field in required_fields:
            assert field in status, f"Missing health status field: {field}"
        
        assert status["status"] == "healthy"
        assert status["performance_target"] == "500ns_hot_path"
    
    def test_error_handling(self):
        """Test error handling in evaluation."""
        monitor = TrustMonitor()
        
        with pytest.raises(ValueError):
            # Invalid input should raise error
            monitor.evaluate(1.5, 0.5, 0.5)  # Out of bounds alignment_score


class TestIntegrationComponents:
    """Test integration components."""
    
    def test_route_to_resontinex(self):
        """Test decision routing to RESONTINEX verbs."""
        # Test all decision mappings
        assert route_to_resontinex(Decision.EXECUTE) == "execute"
        assert route_to_resontinex(Decision.REVIEW) == "review"
        assert route_to_resontinex(Decision.DEFER) == "defer"
        assert route_to_resontinex(Decision.ABORT) == "abort"
    
    def test_calibration_adapter(self):
        """Test calibration adapter interface."""
        adapter = CalibrationAdapter()
        
        # Test initial state
        assert not adapter.calibration_enabled
        status = adapter.get_status()
        assert not status["enabled"]
        
        # Test insufficient data
        result = adapter.fit([0.1, 0.2], [True, False])
        assert not result, "Should fail with insufficient data"
        
        # Test successful fitting
        scores = [0.1 + i * 0.02 for i in range(50)]
        labels = [score > 0.5 for score in scores]
        result = adapter.fit(scores, labels)
        assert result, "Should succeed with sufficient data"
        assert adapter.calibration_enabled
        
        # Test prediction
        calibrated = adapter.predict(0.7)
        assert calibrated is not None
        assert isinstance(calibrated, float)


class TestPerformanceBenchmarks:
    """Performance benchmarks targeting ~500ns execution time."""
    
    def test_hot_path_performance(self):
        """Test hot path execution time benchmark."""
        inputs = InputVector(0.8, 0.3, 0.2)
        
        # Warm up JIT/interpreter
        for _ in range(100):
            CompositeTrustGate(inputs=inputs)
        
        # Measure hot path performance
        iterations = 1000
        start_time = time.perf_counter_ns()
        
        for _ in range(iterations):
            CompositeTrustGate(inputs=inputs)
        
        end_time = time.perf_counter_ns()
        avg_time_ns = (end_time - start_time) / iterations
        
        # Target: ~500ns per evaluation
        # Allow some variance for test environments
        assert avg_time_ns < 10_000, f"Performance target missed: {avg_time_ns}ns > 10,000ns"
        
        print(f"Performance benchmark: {avg_time_ns:.0f}ns per evaluation")
    
    def test_monitor_evaluation_performance(self):
        """Test TrustMonitor evaluation performance."""
        monitor = TrustMonitor()
        
        # Warm up
        for _ in range(100):
            monitor.evaluate(0.8, 0.3, 0.2)
        
        # Measure performance with observability
        iterations = 100
        start_time = time.perf_counter_ns()
        
        for _ in range(iterations):
            monitor.evaluate(0.8, 0.3, 0.2)
        
        end_time = time.perf_counter_ns()
        avg_time_ns = (end_time - start_time) / iterations
        
        # Monitor adds observability overhead, should still be fast
        assert avg_time_ns < 50_000, f"Monitor performance target missed: {avg_time_ns}ns > 50,000ns"
        
        print(f"Monitor performance benchmark: {avg_time_ns:.0f}ns per evaluation")
    
    def test_memory_footprint(self):
        """Test memory footprint of frozen dataclass."""
        import sys
        
        inputs = InputVector(0.8, 0.3, 0.2)
        gate = CompositeTrustGate(inputs=inputs)
        
        # Check object size (frozen dataclass should be minimal)
        size_bytes = sys.getsizeof(gate)
        
        # Should be small due to __slots__ and frozen design
        assert size_bytes < 1024, f"Memory footprint too large: {size_bytes} bytes"
        
        print(f"Memory footprint: {size_bytes} bytes")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_extreme_input_combinations(self):
        """Test extreme input combinations."""
        # All minimum values
        min_inputs = InputVector(0.0, 0.0, 0.0)
        min_gate = CompositeTrustGate(inputs=min_inputs)
        assert 0.0 <= min_gate.trust_score <= 1.0
        
        # All maximum values
        max_inputs = InputVector(1.0, 1.0, 1.0)
        max_gate = CompositeTrustGate(inputs=max_inputs)
        assert 0.0 <= max_gate.trust_score <= 1.0
        
        # Mixed extreme values
        mixed_inputs = InputVector(1.0, 0.0, 1.0)
        mixed_gate = CompositeTrustGate(inputs=mixed_inputs)
        assert 0.0 <= mixed_gate.trust_score <= 1.0
    
    def test_config_edge_cases(self):
        """Test configuration edge cases."""
        # Single non-zero weight
        edge_config = copy.deepcopy(DEFAULT_CONFIG)
        edge_config["weights"] = {
            "alignment_score": 1.0,
            "epistemic_risk": 0.0,
            "confidence_band": 0.0
        }
        
        inputs = InputVector(0.8, 0.5, 0.5)
        gate = CompositeTrustGate(inputs=inputs, config=edge_config)
        
        # Should handle gracefully
        assert isinstance(gate.trust_score, float)
        assert 0.0 <= gate.trust_score <= 1.0
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability at boundaries."""
        # Values very close to boundaries
        edge_inputs = InputVector(1e-15, 1.0 - 1e-15, 1e-15)
        edge_gate = CompositeTrustGate(inputs=edge_inputs)
        
        # Should not produce NaN or infinite values
        assert not (edge_gate.trust_score != edge_gate.trust_score)  # Not NaN
        assert abs(edge_gate.trust_score) != float('inf')  # Not infinite
        assert 0.0 <= edge_gate.trust_score <= 1.0


class TestRESO:
    """Test RESONTINEX integration patterns."""
    
    def test_integration_workflow(self):
        """Test complete integration workflow."""
        # Create monitor with RESONTINEX-compatible logging
        monitor = TrustMonitor(logger_name="resontinex.trust.integration_test")
        
        # Evaluate trust gate
        gate = monitor.evaluate(0.8, 0.3, 0.2)
        
        # Route decision to RESONTINEX
        verb = route_to_resontinex(gate.decision)
        
        # Verify integration workflow
        assert verb in ["execute", "review", "defer", "abort"]
        
        # Get explanation for audit trail
        explanation = gate.explain()
        assert "computation_method" in explanation
        assert "observability" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])