#!/usr/bin/env python3
"""Comprehensive integration tests for Resontinex system."""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from resontinex.config import RuntimeConfig
from resontinex.plugins import load_plugins
from resontinex.runtime.api import OverlayRouter, ScenarioManager, ProductionSafetyManager
from resontinex.obs.middleware import measure, record_circuit_trip, record_overlay_fallback, update_trust_score
from resontinex_governance.energy import EnergyLedger
from resontinex_governance.quorum import QuorumVoter, MultiStageQuorum, Vote

class TestRuntimeConfigIntegration:
    """Test configuration system integration."""
    
    def test_config_creation_and_validation(self):
        """Test basic config creation and validation."""
        config = RuntimeConfig()
        
        # Verify basic structure
        assert config.router.strategy in ["auto", "cost", "quality", "failover"]
        assert config.slo.p95_latency_ms > 0
        assert config.governance.energy_budget_limit > 0
        assert config.observability.enable_metrics is True
        
    def test_config_serialization(self):
        """Test config can be serialized/deserialized."""
        config = RuntimeConfig()
        config_dict = config.dict()
        
        # Verify serialization contains expected keys
        assert "router" in config_dict
        assert "slo" in config_dict
        assert "governance" in config_dict
        assert "observability" in config_dict
        
        # Verify reconstruction
        config2 = RuntimeConfig.parse_obj(config_dict)
        assert config2.router.strategy == config.router.strategy
        
    def test_production_validation(self):
        """Test production readiness validation."""
        config = RuntimeConfig()
        config.debug_mode = True
        config.environment = "production"
        
        issues = config.validate_production_ready()
        assert len(issues) > 0
        assert any("debug mode" in issue.lower() for issue in issues)

class TestGovernanceIntegration:
    """Test governance components integration."""
    
    def test_energy_ledger_workflow(self):
        """Test complete energy budget workflow."""
        ledger = EnergyLedger(budget=1000.0, review_threshold=0.8)
        
        # Test normal allocation
        assert ledger.allocate("tx1", 100.0, {"model": "gpt-4"})
        assert ledger.available == 900.0
        assert not ledger.needs_review()
        
        # Test large allocation triggering review
        assert ledger.allocate("tx2", 700.0, {"model": "claude-3"})
        assert ledger.needs_review()
        
        # Test budget exhaustion
        assert not ledger.allocate("tx3", 300.0)
        
        # Test deallocation
        assert ledger.deallocate("tx1")
        assert ledger.available == 200.0
        
    def test_quorum_decision_workflow(self):
        """Test complete quorum voting workflow."""
        voter = QuorumVoter(threshold=0.6, veto_power=True)
        
        # Test normal approval
        votes: list[Vote] = [True, True, True, False, False]  # 60% approval
        assert voter.decide(votes, "decision-1")
        
        # Test veto blocking
        votes_with_veto: list[Vote] = [True, True, "VETO", False]
        assert not voter.decide(votes_with_veto, "decision-2")
        
        # Test vote summary
        summary = voter.get_vote_summary(votes)
        assert summary["approval_ratio"] == 0.6
        assert summary["total_votes"] == 5
        assert summary["would_pass"] is True
        
    def test_multi_stage_quorum(self):
        """Test multi-stage governance workflow."""
        stage1 = QuorumVoter(threshold=0.5)
        stage2 = QuorumVoter(threshold=0.75)
        multi_quorum = MultiStageQuorum([stage1, stage2])
        
        # Test passing both stages
        votes_stage1: list[Vote] = [True, True, False]  # 66% - passes stage 1
        votes_stage2: list[Vote] = [True, True, True, False]  # 75% - passes stage 2
        assert multi_quorum.decide([votes_stage1, votes_stage2], "multi-decision")
        
        # Test failing second stage
        votes_stage2_fail: list[Vote] = [True, False, False, False]  # 25% - fails stage 2
        assert not multi_quorum.decide([votes_stage1, votes_stage2_fail], "multi-decision-fail")

class TestObservabilityIntegration:
    """Test observability system integration."""
    
    def test_measurement_context_manager(self):
        """Test the measurement context manager."""
        with patch('sys.stdout') as mock_stdout:
            with measure("test_operation", {"scenario": "integration"}):
                time.sleep(0.01)  # Small delay to measure
            
            # Verify logging output
            mock_stdout.write.assert_called()
            call_args = str(mock_stdout.write.call_args)
            assert "test_operation" in call_args
            
    def test_error_handling_in_measurement(self):
        """Test error handling in measurement context."""
        with patch('sys.stderr') as mock_stderr:
            try:
                with measure("failing_operation"):
                    raise ValueError("Test error")
            except ValueError:
                pass  # Expected
            
            # Verify error logging
            mock_stderr.write.assert_called()
            
    def test_metrics_recording(self):
        """Test various metrics recording functions."""
        with patch('builtins.print') as mock_print:
            record_circuit_trip("test-circuit")
            record_overlay_fallback("primary", "fallback")
            update_trust_score("component1", 0.85)
            
            # Verify all metrics were recorded
            assert mock_print.call_count >= 2

class TestPluginSystemIntegration:
    """Test plugin system integration."""
    
    def test_plugin_loading_without_governance(self):
        """Test plugin loading when governance is disabled."""
        config = {"enable_governance": False}
        
        with patch('builtins.print') as mock_print:
            load_plugins(config)
            # Should print info about no plugins loaded
            
    def test_plugin_loading_with_governance(self):
        """Test plugin loading when governance is enabled."""
        config = {"enable_governance": True}
        
        with patch('builtins.print') as mock_print:
            load_plugins(config)
            # Should attempt to load plugins

class TestRuntimeAPIIntegration:
    """Test runtime API components integration."""
    
    def test_overlay_router_basic_workflow(self):
        """Test basic overlay router workflow."""
        router = OverlayRouter.from_default()
        manager = ScenarioManager.load("non-existent-path.yaml")  # Graceful handling
        
        response = router.route("Test prompt", manager)
        assert response.text.startswith("ROUTE:")
        assert "route" in response.meta
        
    def test_production_safety_manager(self):
        """Test production safety manager workflow."""
        safety = ProductionSafetyManager.from_config("non-existent-config.yaml")
        
        def test_operation():
            return "success"
        
        result = safety.execute(test_operation)
        assert result == "success"

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_golden_scenario_components(self):
        """Test that all components work together like in golden scenario."""
        # Initialize configuration
        config = RuntimeConfig()
        config.governance.enable_governance = True
        
        # Initialize components
        router = OverlayRouter.from_default()
        manager = ScenarioManager.load("test-scenario")
        safety = ProductionSafetyManager.from_config("test-config")
        
        # Initialize governance
        ledger = EnergyLedger(budget=1000.0)
        voter = QuorumVoter(threshold=0.6)
        
        # Execute workflow with measurement
        with patch('builtins.print'):  # Suppress output during test
            with measure("end_to_end_test"):
                # Allocate energy budget
                assert ledger.allocate("e2e_tx", 50.0, {"operation": "route"})
                
                # Execute routing
                response = safety.execute(lambda: router.route("Test end-to-end", manager))
                assert response is not None
                
                # Record trust score
                update_trust_score("end_to_end", 0.85)
                
                # Make governance decision
                votes: list[Vote] = [True, True, False]  # 66% approval
                decision = voter.decide(votes, "e2e_approval")
                assert decision is True
                
                # Clean up
                ledger.deallocate("e2e_tx")

@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and error handling."""
    
    def test_configuration_fallbacks(self):
        """Test graceful handling of configuration issues."""
        # Test with missing file - should use defaults
        try:
            config = RuntimeConfig.parse_file("non-existent-config.yaml")
            pytest.fail("Should have raised FileNotFoundError")
        except FileNotFoundError:
            # Expected behavior
            pass
        
        # Test with empty config
        config = RuntimeConfig()
        assert config.version == "2.1.0"
        
    def test_plugin_loading_resilience(self):
        """Test plugin system handles errors gracefully."""
        with patch('builtins.print') as mock_print:
            load_plugins({"enable_governance": True})
            # Should not crash even if plugins fail to load
            
    def test_measurement_with_failures(self):
        """Test measurement system handles various failure modes."""
        with patch('builtins.print'):
            # Test with exception in measured code
            try:
                with measure("failing_test"):
                    raise RuntimeError("Simulated failure")
            except RuntimeError:
                pass  # Expected
            
            # Test with None metadata
            with measure("test_with_none", None):
                pass

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])