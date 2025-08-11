#!/usr/bin/env python3
"""
Golden Test Framework for RESONTINEX Fusion System
Captures and validates expected outputs for regression testing.
"""

import os
import sys
import json
import yaml
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

try:
    from resontinex.fusion_resilience import load_fusion_configuration
    from scripts.runtime_router import RuntimeRouter
    # Note: evaluate-fusion.py uses hyphens, so import differently
    import importlib.util
    
    eval_spec = importlib.util.spec_from_file_location(
        "evaluate_fusion",
        os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'evaluate-fusion.py')
    )
    if eval_spec and eval_spec.loader:
        evaluate_fusion_module = importlib.util.module_from_spec(eval_spec)
        eval_spec.loader.exec_module(evaluate_fusion_module)
        FusionEvaluator = getattr(evaluate_fusion_module, 'FusionEvaluator', None)
    else:
        FusionEvaluator = None
        
except ImportError as e:
    print(f"Warning: Could not import fusion components: {e}")
    FusionEvaluator = None
    RuntimeRouter = None


@dataclass
class GoldenTestResult:
    """Represents a golden test execution result."""
    scenario_id: str
    timestamp: str
    baseline_output: str
    overlay_output: str
    evaluation_scores: Dict[str, float]
    execution_metrics: Dict[str, Any]
    rules_validation: Dict[str, bool]
    config_hash: str


class GoldenTestRunner:
    """Executes and manages golden tests for fusion system regression testing."""
    
    def __init__(self, golden_dir: str = "./tests/golden", config_dir: str = "./configs/fusion"):
        self.golden_dir = Path(golden_dir)
        self.config_dir = Path(config_dir)
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize fusion components
        self.evaluator = None
        self.router = None
        self._initialize_fusion_components()
        
        # Load scenarios
        self.scenarios = self._load_evaluation_scenarios()
        
    def _initialize_fusion_components(self):
        """Initialize fusion system components for testing."""
        try:
            if FusionEvaluator is not None:
                self.evaluator = FusionEvaluator(str(self.config_dir))
            else:
                self.evaluator = MockFusionEvaluator()
                
            if RuntimeRouter is not None:
                self.router = RuntimeRouter(str(self.config_dir))
            else:
                self.router = MockRuntimeRouter()
        except (NameError, Exception):
            # Create mock components if real ones not available
            self.evaluator = MockFusionEvaluator()
            self.router = MockRuntimeRouter()
    
    def _load_evaluation_scenarios(self) -> List[Dict[str, Any]]:
        """Load evaluation scenarios from configuration."""
        scenarios_path = self.config_dir / "eval_scenarios.yaml"
        if not scenarios_path.exists():
            return []
        
        with open(scenarios_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('scenarios', [])
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current fusion configuration for change detection."""
        config_files = [
            "overlay_params.yaml",
            "slo.yaml", 
            "drift_policy.yaml"
        ]
        
        config_content = ""
        for file_name in config_files:
            file_path = self.config_dir / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_content += f.read()
        
        return hashlib.sha256(config_content.encode()).hexdigest()[:16]
    
    def _get_golden_file_path(self, scenario_id: str) -> Path:
        """Get path to golden file for scenario."""
        return self.golden_dir / f"{scenario_id}_golden.json"
    
    def _execute_fusion_scenario(self, scenario: Dict[str, Any]) -> GoldenTestResult:
        """Execute a single fusion scenario and capture results."""
        scenario_id = scenario['id']
        prompt = scenario['prompt']
        context = scenario.get('context', '')
        
        start_time = time.time()
        
        # Execute baseline response
        baseline_output = self._generate_baseline_response(prompt, context)
        
        # Execute overlay response  
        overlay_output = self._generate_overlay_response(prompt, context, scenario)
        
        # Evaluate responses
        evaluation_scores = self._evaluate_responses(scenario, baseline_output, overlay_output)
        
        # Validate rules
        rules_validation = self._validate_rules(overlay_output, scenario)
        
        execution_time = time.time() - start_time
        
        return GoldenTestResult(
            scenario_id=scenario_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            baseline_output=baseline_output,
            overlay_output=overlay_output,
            evaluation_scores=evaluation_scores,
            execution_metrics={
                "execution_time_ms": int(execution_time * 1000),
                "baseline_length": len(baseline_output),
                "overlay_length": len(overlay_output),
                "length_delta": len(overlay_output) - len(baseline_output)
            },
            rules_validation=rules_validation,
            config_hash=self._calculate_config_hash()
        )
    
    def _generate_baseline_response(self, prompt: str, context: str) -> str:
        """Generate baseline response without fusion overlay."""
        if self.evaluator and hasattr(self.evaluator, 'generate_baseline'):
            return self.evaluator.generate_baseline(prompt, context)
        else:
            # Mock baseline response for testing
            return f"Baseline response to: {prompt}\n\nContext: {context}\n\nThis is a deterministic baseline response for testing purposes."
    
    def _generate_overlay_response(self, prompt: str, context: str, scenario: Dict[str, Any]) -> str:
        """Generate response with fusion overlay applied."""
        if self.evaluator and hasattr(self.evaluator, 'generate_overlay'):
            # Select appropriate micro-overlay
            selected_overlay = self.router.select_overlay(scenario) if self.router else 'default'
            return self.evaluator.generate_overlay(prompt, context, selected_overlay)
        else:
            # Mock overlay response for testing
            category = scenario.get('category', 'default')
            return f"Overlay response to: {prompt}\n\nContext: {context}\n\nOverlay: {category}\n\nThis includes specific operational steps and enhanced reasoning depth."
    
    def _evaluate_responses(self, scenario: Dict[str, Any], baseline: str, overlay: str) -> Dict[str, float]:
        """Evaluate response quality using fusion metrics."""
        if self.evaluator and hasattr(self.evaluator, 'evaluate_responses'):
            return self.evaluator.evaluate_responses(scenario, baseline, overlay)
        else:
            # Mock evaluation scores
            expected_caps = scenario.get('expected_capabilities', {})
            return {
                'specificity': expected_caps.get('trust_scoring', 0.75),
                'operationality': expected_caps.get('entropy_control', 0.70),
                'rationale_density': expected_caps.get('insight_compression', 0.80),
                'overall_improvement': 0.12
            }
    
    def _validate_rules(self, response: str, scenario: Dict[str, Any]) -> Dict[str, bool]:
        """Validate response against scenario rules."""
        rules = {
            'contains_specific_steps': 'step' in response.lower() or 'action' in response.lower(),
            'includes_rationale': 'because' in response.lower() or 'reason' in response.lower(),
            'addresses_context': any(keyword in response.lower() for keyword in ['analyze', 'recommend', 'determine', 'plan']),
            'appropriate_length': len(response) > 100,  # Minimum depth requirement
            'operational_focus': any(keyword in response.lower() for keyword in ['implement', 'execute', 'process', 'configure'])
        }
        
        return rules
    
    def capture_golden_results(self, scenario_ids: Optional[List[str]] = None) -> Dict[str, str]:
        """Capture golden test results for specified scenarios."""
        if scenario_ids is None:
            # Use top 3 canonical scenarios for golden tests
            canonical_scenarios = [
                'refund_processing_duplicate',
                'security_incident_containment', 
                'regulatory_compliance_audit_prep'
            ]
            scenario_ids = canonical_scenarios
        
        results = {}
        
        for scenario_id in scenario_ids:
            scenario = next((s for s in self.scenarios if s['id'] == scenario_id), None)
            if not scenario:
                print(f"Warning: Scenario {scenario_id} not found")
                continue
            
            print(f"Capturing golden result for: {scenario_id}")
            
            # Execute scenario
            result = self._execute_fusion_scenario(scenario)
            
            # Save golden file
            golden_path = self._get_golden_file_path(scenario_id)
            with open(golden_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            
            results[scenario_id] = str(golden_path)
        
        return results
    
    def validate_against_golden(self, scenario_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Validate current results against golden files."""
        if scenario_ids is None:
            scenario_ids = [f.stem.replace('_golden', '') for f in self.golden_dir.glob('*_golden.json')]
        
        validation_results = {}
        
        for scenario_id in scenario_ids:
            golden_path = self._get_golden_file_path(scenario_id)
            
            if not golden_path.exists():
                validation_results[scenario_id] = {
                    'status': 'missing_golden',
                    'message': f'Golden file not found: {golden_path}'
                }
                continue
            
            # Load golden result
            with open(golden_path, 'r', encoding='utf-8') as f:
                golden_result = json.load(f)
            
            # Execute current scenario
            scenario = next((s for s in self.scenarios if s['id'] == scenario_id), None)
            if not scenario:
                validation_results[scenario_id] = {
                    'status': 'scenario_not_found',
                    'message': f'Scenario {scenario_id} not found in current configuration'
                }
                continue
            
            current_result = self._execute_fusion_scenario(scenario)
            current_dict = asdict(current_result)
            
            # Compare results
            comparison = self._compare_results(golden_result, current_dict)
            validation_results[scenario_id] = comparison
        
        return validation_results
    
    def _compare_results(self, golden: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current results against golden baseline."""
        comparison = {
            'status': 'pass',
            'differences': [],
            'metrics_comparison': {},
            'rules_validation_match': True
        }
        
        # Compare evaluation scores
        golden_scores = golden.get('evaluation_scores', {})
        current_scores = current.get('evaluation_scores', {})
        
        for metric, golden_value in golden_scores.items():
            current_value = current_scores.get(metric, 0)
            diff = abs(current_value - golden_value)
            
            comparison['metrics_comparison'][metric] = {
                'golden': golden_value,
                'current': current_value,
                'difference': diff,
                'threshold_exceeded': diff > 0.05  # 5% threshold
            }
            
            if diff > 0.05:
                comparison['status'] = 'fail'
                comparison['differences'].append(f'{metric}: difference {diff:.3f} exceeds threshold')
        
        # Compare rules validation
        golden_rules = golden.get('rules_validation', {})
        current_rules = current.get('rules_validation', {})
        
        for rule, golden_passed in golden_rules.items():
            current_passed = current_rules.get(rule, False)
            if golden_passed != current_passed:
                comparison['rules_validation_match'] = False
                comparison['status'] = 'fail'
                comparison['differences'].append(f'Rule {rule}: golden={golden_passed}, current={current_passed}')
        
        # Compare configuration hash
        if golden.get('config_hash') != current.get('config_hash'):
            comparison['config_changed'] = True
            comparison['differences'].append('Configuration hash changed - may require golden update')
        
        return comparison
    
    def generate_test_report(self, validation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(validation_results)
        passed_tests = sum(1 for result in validation_results.values() if result.get('status') == 'pass')
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_scenarios': total_tests,
                'passed': passed_tests,
                'failed': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'detailed_results': validation_results,
            'configuration': {
                'golden_dir': str(self.golden_dir),
                'config_hash': self._calculate_config_hash(),
                'scenarios_available': len(self.scenarios)
            }
        }
        
        return report


class MockFusionEvaluator:
    """Mock evaluator for testing when real components unavailable."""
    
    def generate_baseline(self, prompt: str, context: str) -> str:
        return f"Mock baseline: {prompt[:100]}..."
    
    def generate_overlay(self, prompt: str, context: str, overlay: str) -> str:
        return f"Mock overlay ({overlay}): {prompt[:100]}... [Enhanced with specific steps]"
    
    def evaluate_responses(self, scenario: Dict[str, Any], baseline: str, overlay: str) -> Dict[str, float]:
        return {'specificity': 0.75, 'operationality': 0.70, 'rationale_density': 0.80, 'overall_improvement': 0.12}


class MockRuntimeRouter:
    """Mock router for testing when real components unavailable."""
    
    def select_overlay(self, scenario: Dict[str, Any]) -> str:
        category_map = {
            'financial_operations': 'rollback_first',
            'security_operations': 'observability_first',
            'compliance_management': 'state_model_first'
        }
        category = scenario.get('category')
        if category is None:
            return 'default'
        return category_map.get(category, 'default')


def main():
    """CLI interface for golden test management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Golden Test Framework for Fusion System')
    parser.add_argument('--capture', action='store_true', help='Capture golden test results')
    parser.add_argument('--validate', action='store_true', help='Validate against golden results')
    parser.add_argument('--update-goldens', action='store_true', help='Update golden files with current results')
    parser.add_argument('--scenarios', nargs='*', help='Specific scenarios to test')
    parser.add_argument('--report-file', help='Save report to file')
    
    args = parser.parse_args()
    
    runner = GoldenTestRunner()
    
    if args.capture or args.update_goldens:
        print("Capturing golden test results...")
        results = runner.capture_golden_results(args.scenarios)
        
        for scenario_id, path in results.items():
            print(f"Golden result saved: {scenario_id} -> {path}")
    
    if args.validate or not (args.capture or args.update_goldens):
        print("Validating against golden results...")
        validation_results = runner.validate_against_golden(args.scenarios)
        
        report = runner.generate_test_report(validation_results)
        
        print(f"\nGolden Test Report:")
        print(f"Total scenarios: {report['summary']['total_scenarios']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success rate: {report['summary']['success_rate']:.1%}")
        
        if report['summary']['failed'] > 0:
            print("\nFailed scenarios:")
            for scenario_id, result in validation_results.items():
                if result.get('status') != 'pass':
                    print(f"  {scenario_id}: {result.get('status')} - {len(result.get('differences', []))} differences")
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.report_file}")
        
        # Exit with non-zero code if tests failed
        return 0 if report['summary']['success_rate'] == 1.0 else 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())