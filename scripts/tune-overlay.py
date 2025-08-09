#!/usr/bin/env python3
"""
RESONTINEX Fusion Parameter Auto-Tuner
Grid-search optimization for scenario-aware overlay parameters.
"""

import os
import json
import yaml
import statistics
import itertools
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import argparse


@dataclass
class ParameterSet:
    """Parameter configuration for testing."""
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    max_tokens: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TuningResult:
    """Result from parameter tuning for a scenario."""
    scenario_id: str
    parameter_set: ParameterSet
    specificity_score: float
    rationale_density_score: float
    operationality_score: float
    overall_score: float
    rule_validation_score: float
    execution_time_ms: int


class ParameterGridGenerator:
    """Generates parameter grids for optimization."""
    
    def __init__(self, constraints: Dict[str, Dict[str, float]]):
        self.constraints = constraints
        
    def generate_grid(self, scenario_category: str, grid_size: str = "small") -> List[ParameterSet]:
        """Generate parameter grid based on scenario category and size."""
        # Define grid sizes
        grid_configs = {
            "small": {"steps": 3, "coverage": 0.6},
            "medium": {"steps": 5, "coverage": 0.8}, 
            "large": {"steps": 7, "coverage": 1.0}
        }
        
        config = grid_configs.get(grid_size, grid_configs["small"])
        steps = config["steps"]
        
        # Generate parameter ranges
        temp_range = self._generate_range("temperature", steps)
        top_p_range = self._generate_range("top_p", steps) 
        presence_range = self._generate_range("presence_penalty", steps)
        frequency_range = self._generate_range("frequency_penalty", steps)
        
        # Category-specific parameter focus
        category_focus = {
            "financial_operations": {"temperature": 0.7, "frequency_penalty": 0.9},
            "security_operations": {"temperature": 0.9, "frequency_penalty": 0.8},
            "system_integration": {"temperature": 0.8, "frequency_penalty": 0.7},
            "compliance_management": {"presence_penalty": 0.6, "frequency_penalty": 0.6},
            "customer_success": {"presence_penalty": 0.8, "temperature": 0.6}
        }
        
        focus = category_focus.get(scenario_category, {})
        
        # Apply category focus by adjusting ranges
        if "temperature" in focus:
            temp_range = self._focus_range(temp_range, focus["temperature"])
        if "frequency_penalty" in focus:
            frequency_range = self._focus_range(frequency_range, focus["frequency_penalty"])
        if "presence_penalty" in focus:
            presence_range = self._focus_range(presence_range, focus["presence_penalty"])
            
        # Generate all combinations
        parameter_sets = []
        for temp, top_p, presence, frequency in itertools.product(
            temp_range, top_p_range, presence_range, frequency_range
        ):
            parameter_sets.append(ParameterSet(
                temperature=temp,
                top_p=top_p,
                presence_penalty=presence,
                frequency_penalty=frequency
            ))
            
        return parameter_sets[:25]  # Limit to 25 combinations for efficiency
    
    def _generate_range(self, param_name: str, steps: int) -> List[float]:
        """Generate parameter range based on constraints."""
        constraint = self.constraints.get(param_name, {"min": 0.0, "max": 1.0, "step": 0.1})
        min_val = constraint["min"]
        max_val = constraint["max"]
        step_size = (max_val - min_val) / (steps - 1)
        
        return [round(min_val + i * step_size, 3) for i in range(steps)]
    
    def _focus_range(self, param_range: List[float], focus_factor: float) -> List[float]:
        """Adjust parameter range to focus on certain values."""
        mid_idx = len(param_range) // 2
        if focus_factor > 0.7:
            # Focus on higher values
            return param_range[mid_idx:]
        elif focus_factor < 0.3:
            # Focus on lower values
            return param_range[:mid_idx+1]
        else:
            # Balanced range
            return param_range


class ScenarioSimulator:
    """Simulates scenario responses with different parameter sets."""
    
    def __init__(self):
        # Load existing judge_fusion for evaluation
        try:
            from judge_fusion import CrossJudgeEvaluator
            self.evaluator = CrossJudgeEvaluator()
        except ImportError:
            print("Warning: CrossJudgeEvaluator not available, using mock evaluation")
            self.evaluator = None
    
    def simulate_response(self, scenario: Dict[str, Any], params: ParameterSet) -> str:
        """Simulate response generation with given parameters."""
        # This is a simplified simulation - in production this would interface with actual models
        
        context = scenario.get('context', '')
        prompt = scenario.get('prompt', '')
        complexity = scenario.get('complexity', 0.5)
        category = scenario.get('category', 'general')
        
        # Simulate parameter effects on response characteristics
        response_parts = []
        
        # Temperature affects creativity vs precision
        if params.temperature < 0.25:
            response_parts.append("## Systematic Analysis\n")
            response_parts.append(f"Following standard protocols for {category} scenarios:\n")
        else:
            response_parts.append("## Strategic Assessment\n") 
            response_parts.append(f"Evaluating multiple approaches for {category} optimization:\n")
        
        # Presence penalty affects repetition
        if params.presence_penalty > 0.1:
            response_parts.append("Considering alternative methodologies and diverse implementation strategies.\n")
        
        # Frequency penalty affects technical depth
        if params.frequency_penalty > 0.35:
            response_parts.append("## Implementation Framework\n")
            response_parts.append("1. Execute preliminary validation protocols within 4 hours\n")
            response_parts.append("2. Deploy monitoring systems with 99.7% reliability targets\n") 
            response_parts.append("3. Establish rollback mechanisms and error handling procedures\n")
        
        # Add technical details based on complexity
        if complexity > 0.7:
            response_parts.append("## Risk Assessment\n")
            response_parts.append(f"High-complexity scenario requires specialized resources: 3 engineers, $25,000 budget allocation, 72-hour implementation window. Critical dependencies include database migration (probability of issues: 15%), API integration testing (estimated 8 hours), and compliance validation (requires legal review).\n")
        
        # Add operational steps
        response_parts.append("## Operational Timeline\n")
        response_parts.append("Phase 1: Assessment and resource allocation (0-6 hours)\n")
        response_parts.append("Phase 2: Implementation with continuous monitoring (6-18 hours)\n")
        response_parts.append("Phase 3: Validation, testing, and documentation (18-24 hours)\n")
        
        return '\n'.join(response_parts)
    
    def evaluate_response(self, response: str, scenario: Dict[str, Any]) -> TuningResult:
        """Evaluate response quality using cross-judge system."""
        start_time = datetime.now()
        
        if self.evaluator:
            # Use actual cross-judge evaluation
            result = self.evaluator.evaluate_with_cross_judges(response, scenario)
            specificity = result.final_scores['specificity']
            rationale_density = result.final_scores['rationale_density'] 
            operationality = result.final_scores['operationality']
            rule_score = result.rule_check.rule_score
        else:
            # Mock evaluation for testing
            specificity = min(len([w for w in response.split() if w.isdigit()]) / 20.0, 1.0)
            rationale_density = min(response.count('because') + response.count('due to') * 0.1, 1.0)
            operationality = min(response.count('step') + response.count('phase') * 0.15, 1.0)
            rule_score = min(response.count('implement') + response.count('execute') * 0.1, 1.0)
        
        overall_score = (specificity * 0.35 + rationale_density * 0.30 + operationality * 0.35)
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TuningResult(
            scenario_id=scenario.get('id', 'unknown'),
            parameter_set=ParameterSet(0,0,0,0), # Will be set by caller
            specificity_score=specificity,
            rationale_density_score=rationale_density,
            operationality_score=operationality,
            overall_score=overall_score,
            rule_validation_score=rule_score,
            execution_time_ms=int(execution_time)
        )


class ParameterTuner:
    """Main parameter tuning engine."""
    
    def __init__(self, config_dir: str = "./configs/fusion", build_dir: str = "./build/tuning"):
        self.config_dir = Path(config_dir)
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.overlay_params = self._load_overlay_params()
        self.scenarios = self._load_scenarios()
        
        # Initialize components
        constraints = self.overlay_params.get('constraints', {})
        self.grid_generator = ParameterGridGenerator(constraints)
        self.simulator = ScenarioSimulator()
    
    def _load_overlay_params(self) -> Dict[str, Any]:
        """Load current overlay parameters."""
        params_path = self.config_dir / "overlay_params.yaml"
        if params_path.exists():
            with open(params_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_scenarios(self) -> Dict[str, Any]:
        """Load evaluation scenarios."""
        scenarios_path = self.config_dir / "eval_scenarios.yaml"
        if scenarios_path.exists():
            with open(scenarios_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {"scenarios": []}
    
    def tune_scenario_parameters(self, scenario: Dict[str, Any], grid_size: str = "small") -> Dict[str, Any]:
        """Tune parameters for a specific scenario."""
        scenario_id = scenario.get('id', 'unknown')
        category = scenario.get('category', 'general')
        
        print(f"Tuning parameters for scenario: {scenario_id} (category: {category})")
        
        # Generate parameter grid
        parameter_sets = self.grid_generator.generate_grid(category, grid_size)
        print(f"Testing {len(parameter_sets)} parameter combinations")
        
        # Test each parameter set
        results = []
        best_result = None
        best_score = -1.0
        
        for i, param_set in enumerate(parameter_sets):
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i + 1}/{len(parameter_sets)}")
            
            # Simulate response with these parameters
            response = self.simulator.simulate_response(scenario, param_set)
            
            # Evaluate response
            result = self.simulator.evaluate_response(response, scenario)
            result.parameter_set = param_set
            
            results.append(result)
            
            # Track best result
            if result.overall_score > best_score:
                best_score = result.overall_score
                best_result = result
        
        # Ensure we have at least one result
        if not results or best_result is None:
            raise ValueError(f"No valid results generated for scenario {scenario_id}")
        
        # Calculate statistics
        scores = [r.overall_score for r in results]
        baseline_params = self._get_baseline_params(scenario_id, category)
        baseline_response = self.simulator.simulate_response(scenario, baseline_params)
        baseline_result = self.simulator.evaluate_response(baseline_response, scenario)
        
        improvement = best_result.overall_score - baseline_result.overall_score
        
        tuning_summary = {
            'scenario_id': scenario_id,
            'category': category,
            'baseline_score': baseline_result.overall_score,
            'best_score': best_result.overall_score,
            'improvement': improvement,
            'best_parameters': best_result.parameter_set.to_dict(),
            'parameter_sets_tested': len(parameter_sets),
            'score_statistics': {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'stdev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                'min': min(scores),
                'max': max(scores)
            },
            'tuned_at': datetime.now(timezone.utc).isoformat()
        }
        
        print(f"  Baseline: {baseline_result.overall_score:.3f}")
        print(f"  Best: {best_result.overall_score:.3f}")
        print(f"  Improvement: {improvement:+.3f}")
        
        return tuning_summary
    
    def _get_baseline_params(self, scenario_id: str, category: str) -> ParameterSet:
        """Get baseline parameters for comparison."""
        # Try scenario-specific override first
        overrides = self.overlay_params.get('overrides', {})
        if scenario_id in overrides:
            params = overrides[scenario_id]
        # Try category default
        elif category in self.overlay_params.get('category_defaults', {}):
            params = self.overlay_params['category_defaults'][category]
        # Use global defaults
        else:
            params = self.overlay_params.get('defaults', {})
        
        return ParameterSet(
            temperature=params.get('temperature', 0.25),
            top_p=params.get('top_p', 0.8),
            presence_penalty=params.get('presence_penalty', 0.0),
            frequency_penalty=params.get('frequency_penalty', 0.30)
        )
    
    def run_full_tuning(self, grid_size: str = "small", target_scenarios: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run parameter tuning across all scenarios."""
        scenarios = self.scenarios.get('scenarios', [])
        
        if target_scenarios:
            scenarios = [s for s in scenarios if s.get('id') in target_scenarios]
        
        print(f"Running parameter tuning on {len(scenarios)} scenarios")
        print("=" * 60)
        
        results = []
        
        for scenario in scenarios:
            try:
                result = self.tune_scenario_parameters(scenario, grid_size)
                results.append(result)
            except Exception as e:
                print(f"Error tuning scenario {scenario.get('id', 'unknown')}: {e}")
        
        # Generate summary report
        if results:
            self._generate_tuning_report(results)
            self._update_overlay_params(results)
        
        return results
    
    def _generate_tuning_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive tuning report."""
        timestamp = datetime.now(timezone.utc).isoformat().replace(':', '-')
        report_path = self.build_dir / f"parameter_tuning_report_{timestamp}.json"
        
        # Calculate summary statistics
        improvements = [r['improvement'] for r in results if r['improvement'] > 0]
        total_improvements = len(improvements)
        
        summary = {
            'tuning_metadata': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'scenarios_tuned': len(results),
                'scenarios_improved': total_improvements,
                'improvement_rate': total_improvements / len(results) if results else 0.0
            },
            'performance_summary': {
                'average_improvement': statistics.mean(improvements) if improvements else 0.0,
                'median_improvement': statistics.median(improvements) if improvements else 0.0,
                'max_improvement': max(improvements) if improvements else 0.0,
                'scenarios_meeting_threshold': sum(1 for r in results if r['improvement'] >= 0.05)
            },
            'detailed_results': results
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nTuning Report Generated: {report_path}")
        print(f"Scenarios improved: {total_improvements}/{len(results)} ({total_improvements/len(results)*100:.1f}%)")
        if improvements:
            print(f"Average improvement: {statistics.mean(improvements):+.3f}")
    
    def _update_overlay_params(self, results: List[Dict[str, Any]]):
        """Update overlay parameters with tuning results."""
        # Load current config
        config = self._load_overlay_params()
        
        # Update overrides with improvements > threshold
        threshold = config.get('quality_gates', {}).get('minimum_improvement_threshold', 0.05)
        
        updated_count = 0
        for result in results:
            if result['improvement'] >= threshold:
                scenario_id = result['scenario_id']
                best_params = result['best_parameters']
                
                if 'overrides' not in config:
                    config['overrides'] = {}
                
                config['overrides'][scenario_id] = {
                    **best_params,
                    'reasoning': f"Auto-tuned: {result['improvement']:+.3f} improvement",
                    'tuned_at': result['tuned_at']
                }
                updated_count += 1
        
        # Update tuning history
        config['tuning_history'] = {
            'last_optimization': datetime.now(timezone.utc).isoformat(),
            'iterations_completed': config.get('tuning_history', {}).get('iterations_completed', 0) + 1,
            'scenarios_updated': updated_count,
            'best_performing_params': max(results, key=lambda x: x['improvement']) if results else {}
        }
        
        # Save updated configuration
        params_path = self.config_dir / "overlay_params.yaml"
        with open(params_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Updated overlay parameters: {updated_count} scenarios")


def main():
    parser = argparse.ArgumentParser(description="RESONTINEX Parameter Auto-Tuner")
    parser.add_argument('--scenarios', nargs='*', help="Specific scenario IDs to tune")
    parser.add_argument('--grid-size', choices=['small', 'medium', 'large'], default='small', 
                       help="Grid search size")
    parser.add_argument('--config-dir', default="./configs/fusion", help="Config directory")
    parser.add_argument('--build-dir', default="./build/tuning", help="Build directory")
    
    args = parser.parse_args()
    
    try:
        tuner = ParameterTuner(args.config_dir, args.build_dir)
        results = tuner.run_full_tuning(args.grid_size, args.scenarios)
        
        print(f"\n✓ Parameter tuning completed successfully")
        print(f"Tuned {len(results)} scenarios")
        
    except Exception as e:
        print(f"✗ Parameter tuning failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())