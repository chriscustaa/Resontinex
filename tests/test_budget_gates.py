#!/usr/bin/env python3
"""
Budget gates testing for fusion system
Validates token delta and latency performance against defined budgets
"""

import pytest
import json
import time
import sys
import os
from typing import Dict, Any, List, Tuple

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from tests.golden.golden_test_framework import GoldenTestRunner


class FusionBudgetTester:
    """Tests fusion system against performance and cost budgets."""
    
    def __init__(self):
        self.token_delta_warn_threshold = int(os.environ.get('FUSION_BUDGET_TOKEN_DELTA_WARN', 12))
        self.token_delta_block_threshold = int(os.environ.get('FUSION_BUDGET_TOKEN_DELTA_BLOCK', 25))
        self.latency_warn_threshold = int(os.environ.get('FUSION_BUDGET_LATENCY_WARN', 2000))
        self.latency_block_threshold = int(os.environ.get('FUSION_BUDGET_LATENCY_BLOCK', 5000))
        
        self.golden_runner = GoldenTestRunner()
        self.results = []
    
    def run_budget_scenarios(self, scenario_count: int = 5) -> List[Dict[str, Any]]:
        """Run budget validation scenarios."""
        # Use top scenarios for budget testing
        budget_scenarios = [
            'refund_processing_duplicate',
            'security_incident_containment', 
            'regulatory_compliance_audit_prep',
            'api_rate_limit_optimization',
            'data_migration_rollback_scenario'
        ][:scenario_count]
        
        results = []
        
        for scenario_id in budget_scenarios:
            scenario = next((s for s in self.golden_runner.scenarios if s['id'] == scenario_id), None)
            if not scenario:
                continue
                
            start_time = time.time()
            result = self.golden_runner._execute_fusion_scenario(scenario)
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate token delta percentage
            baseline_length = result.execution_metrics.get('baseline_length', 0)
            overlay_length = result.execution_metrics.get('overlay_length', 0)
            token_delta_pct = ((overlay_length - baseline_length) / baseline_length * 100) if baseline_length > 0 else 0
            
            budget_result = {
                'scenario_id': scenario_id,
                'token_delta_pct': round(token_delta_pct, 2),
                'execution_time_ms': execution_time_ms,
                'baseline_tokens': baseline_length,
                'overlay_tokens': overlay_length,
                'evaluation_scores': result.evaluation_scores,
                'rules_passed': sum(1 for passed in result.rules_validation.values() if passed),
                'total_rules': len(result.rules_validation)
            }
            
            results.append(budget_result)
        
        return results
    
    def analyze_budget_compliance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze budget compliance across all scenarios."""
        if not results:
            return {
                'status': 'error',
                'message': 'No results to analyze',
                'token_delta_avg': 0,
                'latency_p95': 0
            }
        
        # Calculate aggregate metrics
        token_deltas = [r['token_delta_pct'] for r in results]
        latencies = [r['execution_time_ms'] for r in results]
        
        avg_token_delta = sum(token_deltas) / len(token_deltas)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0
        
        # Determine budget status
        status = 'pass'
        violations = []
        
        if avg_token_delta > self.token_delta_block_threshold:
            status = 'block'
            violations.append(f'Token delta {avg_token_delta:.1f}% exceeds blocking threshold {self.token_delta_block_threshold}%')
        elif avg_token_delta > self.token_delta_warn_threshold:
            status = 'warn'
            violations.append(f'Token delta {avg_token_delta:.1f}% exceeds warning threshold {self.token_delta_warn_threshold}%')
        
        if p95_latency > self.latency_block_threshold:
            status = 'block'
            violations.append(f'P95 latency {p95_latency}ms exceeds blocking threshold {self.latency_block_threshold}ms')
        elif p95_latency > self.latency_warn_threshold:
            status = 'warn'
            violations.append(f'P95 latency {p95_latency}ms exceeds warning threshold {self.latency_warn_threshold}ms')
        
        return {
            'status': status,
            'token_delta_avg': round(avg_token_delta, 2),
            'latency_p95': p95_latency,
            'scenarios_tested': len(results),
            'violations': violations,
            'detailed_results': results
        }


@pytest.fixture
def budget_tester():
    """Fixture providing budget tester instance."""
    return FusionBudgetTester()


def test_fusion_budget_compliance(budget_tester):
    """Test fusion system compliance with budget constraints."""
    # Run budget scenarios
    scenario_count = int(os.environ.get('PYTEST_SCENARIO_COUNT', 5))
    results = budget_tester.run_budget_scenarios(scenario_count)
    
    assert len(results) > 0, "No budget scenarios executed"
    
    # Analyze compliance
    analysis = budget_tester.analyze_budget_compliance(results)
    
    # Store results for CI pipeline
    budget_report = {
        'timestamp': time.time(),
        'budget_analysis': analysis,
        'test_metadata': {
            'scenario_count': len(results),
            'token_delta_warn_threshold': budget_tester.token_delta_warn_threshold,
            'token_delta_block_threshold': budget_tester.token_delta_block_threshold,
            'latency_warn_threshold': budget_tester.latency_warn_threshold,
            'latency_block_threshold': budget_tester.latency_block_threshold
        }
    }
    
    # Save for CI consumption
    with open('budget_report.json', 'w') as f:
        json.dump(budget_report, f, indent=2)
    
    # Test assertions
    if analysis['status'] == 'block':
        pytest.fail(f"Budget blocking thresholds exceeded: {'; '.join(analysis['violations'])}")
    elif analysis['status'] == 'warn':
        import warnings
        warnings.warn(f"Budget warning thresholds exceeded: {'; '.join(analysis['violations'])}")
    
    # Verify minimum quality standards
    for result in results:
        quality_score = result['evaluation_scores'].get('overall_improvement', 0)
        assert quality_score > 0, f"Scenario {result['scenario_id']} shows no improvement"
        
        rules_pass_rate = result['rules_passed'] / result['total_rules']
        assert rules_pass_rate >= 0.8, f"Scenario {result['scenario_id']} rules pass rate {rules_pass_rate:.1%} below 80%"


def test_individual_scenario_budgets(budget_tester):
    """Test individual scenarios against budget constraints."""
    results = budget_tester.run_budget_scenarios(3)  # Test subset for individual validation
    
    for result in results:
        scenario_id = result['scenario_id']
        
        # Token delta validation
        token_delta = result['token_delta_pct']
        assert token_delta < budget_tester.token_delta_block_threshold * 1.5, \
            f"Scenario {scenario_id} token delta {token_delta:.1f}% severely exceeds budget"
        
        # Latency validation
        latency = result['execution_time_ms']
        assert latency < budget_tester.latency_block_threshold * 1.5, \
            f"Scenario {scenario_id} latency {latency}ms severely exceeds budget"
        
        # Quality validation
        quality_metrics = result['evaluation_scores']
        assert quality_metrics.get('specificity', 0) > 0.6, \
            f"Scenario {scenario_id} specificity too low"
        assert quality_metrics.get('operationality', 0) > 0.6, \
            f"Scenario {scenario_id} operationality too low"


if __name__ == '__main__':
    # Run budget tests directly
    tester = FusionBudgetTester()
    results = tester.run_budget_scenarios()
    analysis = tester.analyze_budget_compliance(results)
    
    print(f"Budget Status: {analysis['status']}")
    print(f"Token Delta: {analysis['token_delta_avg']:.1f}%")
    print(f"P95 Latency: {analysis['latency_p95']}ms")
    print(f"Scenarios Tested: {analysis['scenarios_tested']}")
    
    if analysis['violations']:
        print("Violations:")
        for violation in analysis['violations']:
            print(f"  - {violation}")
    
    # Exit with appropriate code
    sys.exit(0 if analysis['status'] in ['pass', 'warn'] else 1)