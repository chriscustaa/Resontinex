#!/usr/bin/env python3
"""Golden scenario test harness for Resontinex core components."""

from resontinex.runtime.api import OverlayRouter, ScenarioManager, ProductionSafetyManager, Response
from resontinex.obs.middleware import measure, record_circuit_trip, record_overlay_fallback, update_trust_score
from resontinex.config import RuntimeConfig
import json
import sys
import time

def run_golden_scenario():
    """Execute the golden scenario with full instrumentation."""
    print("=== RESONTINEX :: GOLDEN SCENARIO ===")
    
    # Load configuration (with fallback to defaults)
    try:
        config = RuntimeConfig.parse_file("configs/runtime_config.yaml")
        print(f"✅ Loaded configuration from file")
    except FileNotFoundError:
        config = RuntimeConfig()
        print(f"⚠️ Using default configuration")
    
    # Initialize components
    with measure("initialization"):
        router = OverlayRouter.from_default()
        manager = ScenarioManager.load("examples/scenarios/golden.yaml")
        safety = ProductionSafetyManager.from_config("configs/slo.yaml")
    
    # Execute test scenarios
    test_scenarios = [
        {
            "name": "standard_routing",
            "prompt": "Summarize latest system performance and identify optimization opportunities.",
            "expected_trust": 0.85
        },
        {
            "name": "fallback_handling", 
            "prompt": "Handle distributed system failure with graceful degradation.",
            "expected_trust": 0.70
        },
        {
            "name": "governance_review",
            "prompt": "Execute governance review and energy budget assessment.",
            "expected_trust": 0.90
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")
        
        with measure(f"scenario_{scenario['name']}", {"scenario": scenario["name"]}):
            try:
                # Simulate routing with safety checks
                resp: Response = safety.execute(
                    lambda: router.route(scenario["prompt"], manager)
                )
                
                # Update trust score based on response quality
                trust_score = min(0.95, len(resp.text) / 100.0)  # Simple heuristic
                update_trust_score(scenario["name"], trust_score)
                
                # Check if trust meets expectations
                if trust_score < scenario["expected_trust"]:
                    record_overlay_fallback("primary", "fallback")
                    print(f"⚠️ Trust score {trust_score:.2f} below expected {scenario['expected_trust']}")
                
                results.append({
                    "scenario": scenario["name"],
                    "status": "success",
                    "trust_score": trust_score,
                    "response_length": len(resp.text),
                    "metadata": resp.meta
                })
                
                print(f"✅ Success - Trust: {trust_score:.2f}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                record_circuit_trip(f"scenario_{scenario['name']}")
                results.append({
                    "scenario": scenario["name"],
                    "status": "error",
                    "error": str(e),
                    "trust_score": 0.0
                })
    
    # Generate summary report
    with measure("report_generation"):
        summary = {
            "total_scenarios": len(test_scenarios),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "average_trust": sum(r.get("trust_score", 0) for r in results) / len(results),
            "timestamp": time.time(),
            "config_version": config.version,
            "results": results
        }
    
    print(f"\n=== GOLDEN SCENARIO SUMMARY ===")
    print(f"Success Rate: {summary['successful']}/{summary['total_scenarios']} ({summary['successful']/summary['total_scenarios']*100:.1f}%)")
    print(f"Average Trust Score: {summary['average_trust']:.3f}")
    print(f"\nDetailed Results:")
    print(json.dumps(summary, indent=2))
    
    # Return exit code based on success rate
    success_rate = summary['successful'] / summary['total_scenarios']
    if success_rate >= 0.8:
        print("\n🎉 Golden scenario PASSED")
        return 0
    else:
        print(f"\n💥 Golden scenario FAILED (success rate: {success_rate:.1%})")
        return 1

if __name__ == "__main__":
    exit_code = run_golden_scenario()
    sys.exit(exit_code)