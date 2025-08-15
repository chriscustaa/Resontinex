#!/usr/bin/env python3
"""
Test suite for RESONTINEX CompositeSignal module.
Comprehensive testing for trust scoring, risk assessment, and integration points.
"""

import pytest
import json
import math
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from resontinex.core.composite_signal import (
    CompositeSignal, 
    LogitStrategy, 
    ResontinexStrategy,
    load_cfg,
    logit,
    logistic,
    create_from_scenario_metrics,
    enhance_trust_manager_scoring,
    DEFAULT_CFG
)

class TestCompositeSignalCore:
    """Test core CompositeSignal functionality."""
    
    def test_composite_signal_creation_valid_inputs(self):
        """Test CompositeSignal creation with valid inputs."""
        signal = CompositeSignal(
            epistemic_risk=0.2,
            alignment_score=0.8,
            confidence_band=0.1
        )
        
        assert signal.epistemic_risk == 0.2
        assert signal.alignment_score == 0.8 
        assert signal.confidence_band == 0.1
        assert isinstance(signal.trust_score, float)
        assert 0.0 <= signal.trust_score <= 1.0
        
    def test_composite_signal_invalid_inputs(self):
        """Test CompositeSignal rejects invalid inputs."""
        with pytest.raises(ValueError, match="epistemic_risk must be within"):
            CompositeSignal(
                epistemic_risk=1.5,  # Invalid: > 1.0
                alignment_score=0.8,
                confidence_band=0.1
            )
            
        with pytest.raises(ValueError, match="alignment_score must be within"):
            CompositeSignal(
                epistemic_risk=0.2,
                alignment_score=-0.1,  # Invalid: < 0.0
                confidence_band=0.1
            )
            
    def test_trust_score_calculation(self):
        """Test trust score calculation with known inputs."""
        # High trust case
        signal_good = CompositeSignal(
            epistemic_risk=0.1,
            alignment_score=0.9,
            confidence_band=0.05
        )
        assert signal_good.trust_score > 0.8
        
        # Low trust case  
        signal_bad = CompositeSignal(
            epistemic_risk=0.8,
            alignment_score=0.3,
            confidence_band=0.7
        )
        assert signal_bad.trust_score < 0.5
        
    def test_risk_tier_classification(self):
        """Test risk tier classification logic."""
        # Block case
        signal_block = CompositeSignal(
            epistemic_risk=0.9,
            alignment_score=0.2,
            confidence_band=0.8
        )
        assert signal_block.risk_tier == "block"
        
        # Pass case
        signal_pass = CompositeSignal(
            epistemic_risk=0.05,
            alignment_score=0.95,
            confidence_band=0.02
        )
        assert signal_pass.risk_tier == "pass"
        
    def test_resontinex_integration_flags(self):
        """Test RESONTINEX-specific integration flags."""
        signal = CompositeSignal(
            epistemic_risk=0.1,
            alignment_score=0.9,
            confidence_band=0.8  # High entropy
        )
        
        assert signal.entropy_flag is True  # confidence_band > 0.72
        assert isinstance(signal.trust_floor_met, bool)
        
    def test_downstream_action_generation(self):
        """Test downstream action generation for decision routing."""
        signal = CompositeSignal(
            epistemic_risk=0.3,
            alignment_score=0.6,
            confidence_band=0.4
        )
        
        action = signal.downstream_action()
        
        required_keys = ["decision", "reason", "tier", "trust_score", "entropy_flag", "voting_weight"]
        for key in required_keys:
            assert key in action
            
        assert action["voting_weight"] in [1, 2, 3]
        assert action["tier"] in ["block", "review", "monitor", "pass"]


class TestAggregationStrategies:
    """Test aggregation strategy implementations."""
    
    def test_logit_strategy(self):
        """Test LogitStrategy aggregation."""
        signal = CompositeSignal(
            epistemic_risk=0.2,
            alignment_score=0.8,
            confidence_band=0.1,
            _agg_strategy=LogitStrategy()
        )
        
        score = signal.trust_score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
    def test_resontinex_strategy(self):
        """Test ResontinexStrategy with trust floor enforcement."""
        signal = CompositeSignal(
            epistemic_risk=0.7,
            alignment_score=0.4,
            confidence_band=0.9,  # High entropy for penalty
            _agg_strategy=ResontinexStrategy()
        )
        
        # Should respect trust floor even with poor inputs
        assert signal.trust_score >= DEFAULT_CFG["trust_floor"]
        
    def test_strategy_consistency(self):
        """Test strategy produces consistent results."""
        signal = CompositeSignal(
            epistemic_risk=0.3,
            alignment_score=0.7,
            confidence_band=0.2
        )
        
        # Multiple calls should return same score
        score1 = signal.trust_score
        score2 = signal.trust_score
        assert score1 == score2


class TestExplainabilityAndAudit:
    """Test explainability and audit trail functionality."""
    
    def test_explain_score_structure(self):
        """Test explain_score returns proper structure."""
        signal = CompositeSignal(
            epistemic_risk=0.2,
            alignment_score=0.8,
            confidence_band=0.3
        )
        
        explanation = signal.explain_score()
        
        required_keys = [
            "strategy_used", 
            "resontinex_integration",
            "contributions_in_logit_space",
            "final_trust_score",
            "risk_assessment"
        ]
        
        for key in required_keys:
            assert key in explanation
            
        assert "trust_floor_met" in explanation["resontinex_integration"]
        assert "entropy_flag" in explanation["resontinex_integration"]
        
    def test_to_dict_completeness(self):
        """Test to_dict includes all necessary fields."""
        signal = CompositeSignal(
            epistemic_risk=0.4,
            alignment_score=0.6,
            confidence_band=0.3
        )
        
        signal_dict = signal.to_dict()
        
        expected_keys = [
            "epistemic_risk", "alignment_score", "confidence_band",
            "trust_score", "risk_tier", "entropy_flag", "trust_floor_met"
        ]
        
        for key in expected_keys:
            assert key in signal_dict
            
        # Should not include private fields
        assert "_cfg" not in signal_dict
        assert "_agg_strategy" not in signal_dict


class TestIntegrationFunctions:
    """Test integration utility functions."""
    
    def test_create_from_scenario_metrics(self):
        """Test creation from ScenarioManager metrics."""
        # Mock ScenarioMetrics structure
        mock_metrics = MagicMock()
        mock_metrics.success_rate = 0.85
        mock_metrics.reliability_index = 0.9
        mock_metrics.resource_efficiency = 0.7
        
        signal = create_from_scenario_metrics(mock_metrics)
        
        assert isinstance(signal, CompositeSignal)
        assert signal.alignment_score == 0.85
        assert signal.epistemic_risk == 0.1  # 1.0 - 0.9
        assert signal.confidence_band == 0.3  # 1.0 - 0.7
        
    def test_enhance_trust_manager_scoring(self):
        """Test enhanced TrustManager integration."""
        result = enhance_trust_manager_scoring(
            alignment_score=0.8,
            inflation_delta=0.1,
            epistemic_risk=0.2,
            confidence_band=0.15
        )
        
        required_keys = ["trust_signal", "downstream_action", "explanation", "legacy_score"]
        for key in required_keys:
            assert key in result
            
        assert result["legacy_score"] == 0.9  # 0.8 + 0.1
        assert isinstance(result["trust_signal"], dict)
        
    def test_to_scenario_metrics_conversion(self):
        """Test conversion to ScenarioMetrics format."""
        signal = CompositeSignal(
            epistemic_risk=0.2,
            alignment_score=0.8,
            confidence_band=0.1
        )
        
        metrics = signal.to_scenario_metrics()
        
        expected_keys = [
            'success_rate', 'reliability_index', 'resource_efficiency',
            'user_satisfaction', 'avg_latency_ms', 'complexity_score'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))


class TestConfigurationHandling:
    """Test configuration loading and handling."""
    
    def test_load_cfg_default(self):
        """Test loading default configuration."""
        cfg = load_cfg(None)
        assert cfg == DEFAULT_CFG
        
    def test_load_cfg_missing_file(self):
        """Test loading non-existent config file."""
        cfg = load_cfg("nonexistent.yaml")
        assert cfg == DEFAULT_CFG
        
    @patch('builtins.open', create=True)
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_cfg_yaml_success(self, mock_exists, mock_open):
        """Test successful YAML config loading."""
        mock_config = {"weights": {"epistemic_risk": 0.5}}
        mock_open.return_value.__enter__.return_value.read.return_value = "weights:\n  epistemic_risk: 0.5"
        
        with patch('yaml.safe_load', return_value=mock_config):
            cfg = load_cfg("test.yaml")
            assert cfg == mock_config


class TestMathUtilities:
    """Test mathematical utility functions."""
    
    def test_logit_function(self):
        """Test logit transformation."""
        assert logit(0.5) == 0.0
        assert logit(0.75) > 0
        assert logit(0.25) < 0
        
        # Test boundary handling
        assert not math.isinf(logit(0.0))
        assert not math.isinf(logit(1.0))
        
    def test_logistic_function(self):
        """Test logistic transformation."""
        assert logistic(0.0) == 0.5
        assert logistic(100) == pytest.approx(1.0, rel=1e-10)
        assert logistic(-100) == pytest.approx(0.0, rel=1e-10)
        
    def test_logit_logistic_inverse(self):
        """Test logit and logistic are inverse functions."""
        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for p in test_values:
            assert logistic(logit(p)) == pytest.approx(p, rel=1e-10)


class TestProductionScenarios:
    """Test production-representative scenarios."""
    
    def test_high_confidence_scenario(self):
        """Test high-confidence, well-aligned scenario."""
        signal = CompositeSignal(
            epistemic_risk=0.05,
            alignment_score=0.95,
            confidence_band=0.02
        )
        
        assert signal.trust_score > 0.9
        assert signal.risk_tier == "pass"
        assert signal.trust_floor_met is True
        assert signal.entropy_flag is False
        
    def test_moderate_risk_scenario(self):
        """Test moderate risk requiring review."""
        signal = CompositeSignal(
            epistemic_risk=0.4,
            alignment_score=0.6,
            confidence_band=0.3
        )
        
        action = signal.downstream_action()
        assert action["decision"] in ["manual", "allow"]
        assert signal.risk_tier in ["review", "monitor"]
        
    def test_high_risk_block_scenario(self):
        """Test high-risk scenario requiring blocking."""
        signal = CompositeSignal(
            epistemic_risk=0.9,
            alignment_score=0.2,
            confidence_band=0.8
        )
        
        assert signal.risk_tier == "block"
        action = signal.downstream_action()
        assert action["decision"] == "abort"
        assert action["voting_weight"] == 1  # Low trust = low voting weight
        
    def test_entropy_boundary_conditions(self):
        """Test behavior at entropy threshold boundaries."""
        # Just below threshold
        signal_low = CompositeSignal(
            epistemic_risk=0.3,
            alignment_score=0.7,
            confidence_band=0.71  # Just below 0.72
        )
        assert signal_low.entropy_flag is False
        
        # Just above threshold
        signal_high = CompositeSignal(
            epistemic_risk=0.3,
            alignment_score=0.7,
            confidence_band=0.73  # Just above 0.72
        )
        assert signal_high.entropy_flag is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])