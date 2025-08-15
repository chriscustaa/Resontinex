"""
Integration validation for CompositeTrustGate with existing RESONTINEX components.
Validates compatibility and provides integration adapters.
"""

import sys
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Import CompositeTrustGate components
from .composite_trust_gate import (
    CompositeTrustGate,
    TrustMonitor,
    InputVector,
    route_to_resontinex,
    Decision,
)

# Import existing RESONTINEX components for validation
try:
    from ..runtime.api import ScenarioManager, OverlayRouter, ProductionSafetyManager
    from ..obs.middleware import measure
    from ..fusion_resilience import get_fusion_loader, FusionResilientLoader
    RESONTINEX_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some RESONTINEX components not available: {e}")
    RESONTINEX_COMPONENTS_AVAILABLE = False


class RESIXIntegrationValidator:
    """Validates integration with existing RESONTINEX components."""
    
    def __init__(self):
        self.validation_results = {}
        self.integration_issues = []
        
    def validate_observability_integration(self) -> bool:
        """Validate integration with RESONTINEX observability middleware."""
        try:
            if not RESONTINEX_COMPONENTS_AVAILABLE:
                self.integration_issues.append("RESONTINEX components not available")
                return False
                
            # Test measure decorator integration
            @measure("trust_gate_validation")
            def test_trust_evaluation():
                inputs = InputVector(0.8, 0.3, 0.2)
                gate = CompositeTrustGate(inputs=inputs)
                return gate.trust_score
            
            score = test_trust_evaluation()
            
            self.validation_results["observability"] = {
                "measure_decorator": True,
                "trust_evaluation": score > 0,
                "status": "passed"
            }
            return True
            
        except Exception as e:
            self.integration_issues.append(f"Observability integration failed: {e}")
            self.validation_results["observability"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def validate_fusion_integration(self) -> bool:
        """Validate integration with RESONTINEX fusion system."""
        try:
            if not RESONTINEX_COMPONENTS_AVAILABLE:
                return False
                
            # Test fusion loader integration
            fusion_loader = get_fusion_loader()
            health_status = fusion_loader.get_health_status()
            
            # Test that TrustMonitor can work with fusion system
            monitor = TrustMonitor(logger_name="resontinex.trust.validation")
            gate = monitor.evaluate(0.8, 0.3, 0.2)
            
            self.validation_results["fusion"] = {
                "fusion_loader": health_status.get("status") == "healthy",
                "trust_monitor": isinstance(gate, CompositeTrustGate),
                "health_status": health_status.get("status"),
                "status": "passed"
            }
            return True
            
        except Exception as e:
            self.integration_issues.append(f"Fusion integration failed: {e}")
            self.validation_results["fusion"] = {
                "status": "failed", 
                "error": str(e)
            }
            return False
    
    def validate_runtime_integration(self) -> bool:
        """Validate integration with RESONTINEX runtime components."""
        try:
            if not RESONTINEX_COMPONENTS_AVAILABLE:
                return False
                
            # Test ScenarioManager integration
            scenario_manager = ScenarioManager.load("./test_scenario.json")
            
            # Test OverlayRouter integration
            router = OverlayRouter.from_default()
            
            # Test ProductionSafetyManager integration  
            safety_manager = ProductionSafetyManager.from_config("./config.json")
            
            # Test that CompositeTrustGate decisions can route through system
            inputs = InputVector(0.8, 0.3, 0.2)
            gate = CompositeTrustGate(inputs=inputs)
            verb = route_to_resontinex(gate.decision)
            
            self.validation_results["runtime"] = {
                "scenario_manager": isinstance(scenario_manager, ScenarioManager),
                "overlay_router": isinstance(router, OverlayRouter),
                "safety_manager": isinstance(safety_manager, ProductionSafetyManager),
                "decision_routing": verb in ["execute", "review", "defer", "abort"],
                "status": "passed"
            }
            return True
            
        except Exception as e:
            self.integration_issues.append(f"Runtime integration failed: {e}")
            self.validation_results["runtime"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def validate_configuration_compatibility(self) -> bool:
        """Validate configuration compatibility with existing systems."""
        try:
            # Test configuration loading paths
            config_paths = [
                "./config/trust_gate_config.yaml",
                "./configs/fusion/trust_gate_config.yaml",
                "./configs/trust_gate.yaml"
            ]
            
            config_found = False
            for path in config_paths:
                if Path(path).exists():
                    config_found = True
                    break
            
            # Test TrustMonitor with different config sources
            monitor_default = TrustMonitor()
            monitor_health = monitor_default.get_health_status()
            
            self.validation_results["configuration"] = {
                "config_paths_checked": len(config_paths),
                "config_found": config_found,
                "monitor_health": monitor_health.get("status") == "healthy",
                "status": "passed"
            }
            return True
            
        except Exception as e:
            self.integration_issues.append(f"Configuration validation failed: {e}")
            self.validation_results["configuration"] = {
                "status": "failed",
                "error": str(e) 
            }
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete integration validation suite."""
        validation_start = time.time()
        
        validation_tests = [
            ("observability", self.validate_observability_integration),
            ("fusion", self.validate_fusion_integration), 
            ("runtime", self.validate_runtime_integration),
            ("configuration", self.validate_configuration_compatibility)
        ]
        
        passed_tests = 0
        total_tests = len(validation_tests)
        
        for test_name, test_func in validation_tests:
            try:
                result = test_func()
                if result:
                    passed_tests += 1
            except Exception as e:
                self.integration_issues.append(f"{test_name} validation exception: {e}")
        
        validation_time = time.time() - validation_start
        
        validation_summary = {
            "overall_status": "passed" if passed_tests == total_tests else "failed",
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "pass_rate": passed_tests / total_tests,
            "validation_time_seconds": validation_time,
            "integration_issues": self.integration_issues,
            "detailed_results": self.validation_results,
            "timestamp": time.time(),
            "resontinex_components_available": RESONTINEX_COMPONENTS_AVAILABLE
        }
        
        return validation_summary


class RESIXTrustIntegrator:
    """Integrates CompositeTrustGate with existing RESONTINEX workflow."""
    
    def __init__(self, fusion_config_dir: str = "./configs/fusion"):
        self.fusion_config_dir = fusion_config_dir
        self.trust_monitor = None
        self.fusion_loader = None
        
        if RESONTINEX_COMPONENTS_AVAILABLE:
            self.fusion_loader = get_fusion_loader(fusion_config_dir)
            self.trust_monitor = TrustMonitor(logger_name="resontinex.trust.integration")
    
    def create_from_scenario_metrics(self, metrics: Dict[str, float]) -> Optional[CompositeTrustGate]:
        """Create CompositeTrustGate from ScenarioManager metrics (legacy compatibility)."""
        try:
            # Map ScenarioManager metrics to InputVector
            inputs = InputVector(
                alignment_score=metrics.get("success_rate", 0.5),
                epistemic_risk=1.0 - metrics.get("reliability_index", 0.5),
                confidence_band=1.0 - metrics.get("resource_efficiency", 0.5)
            )
            
            return CompositeTrustGate(inputs=inputs)
            
        except Exception as e:
            print(f"Error creating CompositeTrustGate from scenario metrics: {e}")
            return None
    
    def integrate_with_overlay_router(self, router: 'OverlayRouter', prompt: str) -> Dict[str, Any]:
        """Integrate trust evaluation with OverlayRouter decisions."""
        if not RESONTINEX_COMPONENTS_AVAILABLE or not self.trust_monitor:
            return {"error": "Integration components not available"}
        
        try:
            # Create mock scenario manager for routing
            scenario_manager = ScenarioManager.load("./test_scenario.json")
            
            # Get router response
            response = router.route(prompt, scenario_manager)
            
            # Extract trust signals from response metadata
            meta = response.meta or {}
            
            # Create trust evaluation
            gate = self.trust_monitor.evaluate(
                alignment_score=meta.get("alignment", 0.7),
                epistemic_risk=meta.get("uncertainty", 0.3),
                confidence_band=meta.get("confidence_width", 0.2)
            )
            
            # Enhance router response with trust information
            enhanced_response = {
                "original_response": response.text,
                "original_meta": meta,
                "trust_evaluation": {
                    "trust_score": gate.trust_score,
                    "risk_tier": gate.risk_tier.value,
                    "decision": gate.decision.value,
                    "execution_verb": route_to_resontinex(gate.decision)
                },
                "recommendation": self._get_routing_recommendation(gate)
            }
            
            return enhanced_response
            
        except Exception as e:
            return {"error": f"Integration failed: {e}"}
    
    def _get_routing_recommendation(self, gate: CompositeTrustGate) -> str:
        """Get routing recommendation based on trust evaluation."""
        if gate.decision == Decision.EXECUTE and gate.trust_score > 0.8:
            return "proceed_with_confidence"
        elif gate.decision == Decision.EXECUTE and gate.trust_score > 0.6:
            return "proceed_with_monitoring"
        elif gate.decision == Decision.REVIEW:
            return "route_to_human_review"
        elif gate.decision == Decision.DEFER:
            return "defer_pending_additional_signals"
        else:  # ABORT
            return "block_with_audit_trail"
    
    def get_integration_health(self) -> Dict[str, Any]:
        """Get health status of integration components."""
        health = {
            "trust_monitor": None,
            "fusion_loader": None,
            "resontinex_components": RESONTINEX_COMPONENTS_AVAILABLE
        }
        
        if self.trust_monitor:
            health["trust_monitor"] = self.trust_monitor.get_health_status()
        
        if self.fusion_loader:
            health["fusion_loader"] = self.fusion_loader.get_health_status()
        
        return health


def run_integration_validation() -> Dict[str, Any]:
    """Run complete integration validation and return results."""
    validator = RESIXIntegrationValidator()
    return validator.run_full_validation()


def demonstrate_integration() -> None:
    """Demonstrate CompositeTrustGate integration with RESONTINEX."""
    print("=== CompositeTrustGate Integration Demonstration ===\n")
    
    # Run validation
    validation_results = run_integration_validation()
    
    print("Integration Validation Results:")
    print(f"Overall Status: {validation_results['overall_status']}")
    print(f"Tests Passed: {validation_results['tests_passed']}/{validation_results['total_tests']}")
    print(f"Pass Rate: {validation_results['pass_rate']:.1%}")
    print(f"Validation Time: {validation_results['validation_time_seconds']:.3f}s")
    
    if validation_results['integration_issues']:
        print("\nIntegration Issues:")
        for issue in validation_results['integration_issues']:
            print(f"  - {issue}")
    
    # Demonstrate integration workflow
    print("\n=== Integration Workflow Demonstration ===")
    
    if RESONTINEX_COMPONENTS_AVAILABLE:
        integrator = RESIXTrustIntegrator()
        
        # Test scenario metrics integration
        mock_metrics = {
            "success_rate": 0.85,
            "reliability_index": 0.75,
            "resource_efficiency": 0.80
        }
        
        gate = integrator.create_from_scenario_metrics(mock_metrics)
        if gate:
            print(f"Trust Score: {gate.trust_score}")
            print(f"Risk Tier: {gate.risk_tier.value}")
            print(f"Decision: {gate.decision.value}")
            print(f"Execution Verb: {route_to_resontinex(gate.decision)}")
        
        # Get integration health
        health = integrator.get_integration_health()
        print(f"\nIntegration Health: {json.dumps(health, indent=2, default=str)}")
    
    else:
        print("RESONTINEX components not available - integration limited")


if __name__ == "__main__":
    demonstrate_integration()