#!/usr/bin/env python3
"""
Fusion Benchmark Runner
Executes performance benchmarks for fusion optimization scenarios
"""

import json
import time
import argparse
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import os


class FusionBenchmarkRunner:
    """Runs performance benchmarks for fusion scenarios."""
    
    def __init__(self, scenarios_dir: str = "tests/golden"):
        self.scenarios_dir = Path(scenarios_dir)
        self.results = []
        
    def discover_scenarios(self) -> List[Path]:
        """Discover available golden test scenarios."""
        scenarios = list(self.scenarios_dir.glob("*_golden.json"))
        if not scenarios:
            print(f"Warning: No golden test scenarios found in {self.scenarios_dir}")
        return scenarios
    
    def load_scenario(self, scenario_file: Path) -> Dict[str, Any]:
        """Load a scenario configuration."""
        try:
            with open(scenario_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading scenario {scenario_file}: {e}")
            return {}
    
    def simulate_fusion_execution(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate fusion execution and collect performance metrics."""
        scenario_name = scenario.get('scenario_name', 'unknown')
        
        # Simulate baseline execution
        baseline_start = time.time()
        time.sleep(0.01)  # Simulate processing time
        baseline_end = time.time()
        baseline_latency = (baseline_end - baseline_start) * 1000
        
        # Simulate overlay execution  
        overlay_start = time.time()
        time.sleep(0.015)  # Simulate slightly longer processing for overlay
        overlay_end = time.time()
        overlay_latency = (overlay_end - overlay_start) * 1000
        
        # Extract expected metrics from golden scenario
        baseline_tokens = scenario.get('baseline_output', {}).get('token_count', 150)
        overlay_tokens = scenario.get('overlay_output', {}).get('token_count', 165)
        
        # Calculate token delta
        token_delta = ((overlay_tokens - baseline_tokens) / baseline_tokens) * 100 if baseline_tokens > 0 else 0
        
        # Simulate quality metrics
        quality_score = scenario.get('evaluation', {}).get('overlay_score', 0.85)
        baseline_score = scenario.get('evaluation', {}).get('baseline_score', 0.80)
        quality_improvement = quality_score - baseline_score
        
        return {
            'scenario': scenario_name,
            'baseline_latency_ms': baseline_latency,
            'overlay_latency_ms': overlay_latency,
            'baseline_tokens': baseline_tokens,
            'overlay_tokens': overlay_tokens,
            'token_delta_pct': token_delta,
            'quality_improvement': quality_improvement,
            'baseline_score': baseline_score,
            'overlay_score': quality_score,
            'timestamp': time.time()
        }
    
    def run_benchmark(self, scenario_file: Path, iterations: int = 3) -> Dict[str, Any]:
        """Run benchmark for a single scenario with multiple iterations."""
        scenario = self.load_scenario(scenario_file)
        if not scenario:
            return {}
        
        iteration_results = []
        for i in range(iterations):
            result = self.simulate_fusion_execution(scenario)
            iteration_results.append(result)
        
        # Aggregate results
        return self.aggregate_iterations(iteration_results)
    
    def aggregate_iterations(self, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple iterations."""
        if not iterations:
            return {}
        
        # Extract metrics for aggregation
        latencies = [r['overlay_latency_ms'] for r in iterations]
        token_deltas = [r['token_delta_pct'] for r in iterations]
        quality_improvements = [r['quality_improvement'] for r in iterations]
        
        # Calculate statistics
        return {
            'scenario': iterations[0]['scenario'],
            'iterations': len(iterations),
            'latency_avg': statistics.mean(latencies),
            'latency_p50': statistics.median(latencies),
            'latency_p95': sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) > 1 else latencies[0],
            'latency_max': max(latencies),
            'token_delta_avg': statistics.mean(token_deltas),
            'token_delta_max': max(token_deltas),
            'quality_improvement_avg': statistics.mean(quality_improvements),
            'baseline_tokens_avg': statistics.mean([r['baseline_tokens'] for r in iterations]),
            'overlay_tokens_avg': statistics.mean([r['overlay_tokens'] for r in iterations]),
            'baseline_score_avg': statistics.mean([r['baseline_score'] for r in iterations]),
            'overlay_score_avg': statistics.mean([r['overlay_score'] for r in iterations]),
            'timestamp': time.time()
        }
    
    def run_all_benchmarks(self, iterations: int = 3) -> Dict[str, Any]:
        """Run benchmarks for all discovered scenarios."""
        scenarios = self.discover_scenarios()
        scenario_results = []
        
        print(f"Running benchmarks for {len(scenarios)} scenarios...")
        
        for scenario_file in scenarios:
            print(f"Benchmarking: {scenario_file.name}")
            result = self.run_benchmark(scenario_file, iterations)
            if result:
                scenario_results.append(result)
        
        return self.aggregate_all_scenarios(scenario_results)
    
    def aggregate_all_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across all scenarios."""
        if not scenarios:
            return {
                'scenarios_tested': 0,
                'budget_analysis': {
                    'token_delta_avg': 0,
                    'latency_p95': 0,
                    'scenarios_tested': 0
                }
            }
        
        # Aggregate across scenarios
        all_token_deltas = [s['token_delta_avg'] for s in scenarios]
        all_latencies = []
        for s in scenarios:
            all_latencies.append(s['latency_p95'])
        
        budget_analysis = {
            'token_delta_avg': statistics.mean(all_token_deltas),
            'token_delta_max': max(all_token_deltas),
            'latency_avg': statistics.mean([s['latency_avg'] for s in scenarios]),
            'latency_p95': max(all_latencies),  # Use worst case P95 across scenarios
            'scenarios_tested': len(scenarios),
            'quality_improvement_avg': statistics.mean([s['quality_improvement_avg'] for s in scenarios])
        }
        
        return {
            'benchmark_summary': {
                'scenarios_tested': len(scenarios),
                'total_iterations': sum(s['iterations'] for s in scenarios),
                'timestamp': time.time()
            },
            'scenario_results': scenarios,
            'budget_analysis': budget_analysis
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save benchmark results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run fusion performance benchmarks')
    parser.add_argument('--scenarios-dir', default='tests/golden', 
                        help='Directory containing golden test scenarios')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations per scenario')
    parser.add_argument('--output', default='benchmark_results.json',
                        help='Output file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = FusionBenchmarkRunner(args.scenarios_dir)
    
    try:
        # Run all benchmarks
        results = runner.run_all_benchmarks(args.iterations)
        
        if args.verbose:
            print("\nBenchmark Summary:")
            summary = results.get('benchmark_summary', {})
            print(f"Scenarios tested: {summary.get('scenarios_tested', 0)}")
            print(f"Total iterations: {summary.get('total_iterations', 0)}")
            
            budget = results.get('budget_analysis', {})
            print(f"Average token delta: {budget.get('token_delta_avg', 0):.1f}%")
            print(f"P95 latency: {budget.get('latency_p95', 0):.1f}ms")
            print(f"Quality improvement: {budget.get('quality_improvement_avg', 0):.3f}")
        
        # Save results
        runner.save_results(results, args.output)
        
        print(f"✅ Benchmark completed successfully")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()