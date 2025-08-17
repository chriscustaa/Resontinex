"""
Budget analysis module for fusion CI operations.
Comprehensive analysis with GitHub Actions integration and threshold validation.
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

@dataclass
class BudgetResult:
    """Budget validation result."""
    passed: bool
    metric: str
    actual: float
    budget: float
    threshold_pct: float
    message: str
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class BudgetAnalyzer:
    """Analyzes performance metrics against budget constraints."""
    
    def __init__(self, budget_config_path: Optional[str] = None):
        self.budget_config_path = budget_config_path or "budgets.json"
        self.results: List[BudgetResult] = []
        
    def load_budget_config(self) -> Dict[str, Any]:
        """Load budget configuration."""
        config_path = Path(self.budget_config_path)
        if not config_path.exists():
            # Default budget configuration
            return {
                "latency": {
                    "p95_ms": 500,
                    "p99_ms": 1000,
                    "threshold_pct": 10
                },
                "cost": {
                    "per_request_cents": 2.0,
                    "threshold_pct": 15
                },
                "memory": {
                    "max_mb": 512,
                    "threshold_pct": 20
                },
                "cpu": {
                    "max_utilization_pct": 80,
                    "threshold_pct": 15
                }
            }
        
        with open(config_path) as f:
            return json.load(f)
    
    def validate_latency_budget(self, metrics: Dict[str, float]) -> List[BudgetResult]:
        """Validate latency metrics against budget."""
        config = self.load_budget_config()
        latency_config = config.get("latency", {})
        results = []
        
        for percentile in ["p95_ms", "p99_ms"]:
            if percentile in metrics and percentile in latency_config:
                actual = metrics[percentile]
                budget = latency_config[percentile]
                threshold = latency_config.get("threshold_pct", 10)
                max_allowed = budget * (1 + threshold / 100)
                
                passed = actual <= max_allowed
                message = (
                    f"{percentile} latency {actual:.2f}ms within budget {budget}ms "
                    f"(threshold: +{threshold}%)" if passed else
                    f"{percentile} latency {actual:.2f}ms exceeds budget {budget}ms "
                    f"by {((actual - budget) / budget * 100):.1f}%"
                )
                
                result = BudgetResult(
                    passed=passed,
                    metric=f"latency_{percentile}",
                    actual=actual,
                    budget=budget,
                    threshold_pct=threshold,
                    message=message
                )
                results.append(result)
                
        return results
    
    def validate_cost_budget(self, metrics: Dict[str, float]) -> List[BudgetResult]:
        """Validate cost metrics against budget."""
        config = self.load_budget_config()
        cost_config = config.get("cost", {})
        results = []
        
        if "per_request_cents" in metrics and "per_request_cents" in cost_config:
            actual = metrics["per_request_cents"]
            budget = cost_config["per_request_cents"]
            threshold = cost_config.get("threshold_pct", 15)
            max_allowed = budget * (1 + threshold / 100)
            
            passed = actual <= max_allowed
            message = (
                f"Cost {actual:.2f}¢/req within budget {budget:.2f}¢/req "
                f"(threshold: +{threshold}%)" if passed else
                f"Cost {actual:.2f}¢/req exceeds budget {budget:.2f}¢/req "
                f"by {((actual - budget) / budget * 100):.1f}%"
            )
            
            result = BudgetResult(
                passed=passed,
                metric="cost_per_request",
                actual=actual,
                budget=budget,
                threshold_pct=threshold,
                message=message
            )
            results.append(result)
            
        return results
    
    def validate_resource_budget(self, metrics: Dict[str, float]) -> List[BudgetResult]:
        """Validate memory and CPU metrics against budget."""
        config = self.load_budget_config()
        results = []
        
        # Memory validation
        memory_config = config.get("memory", {})
        if "max_mb" in metrics and "max_mb" in memory_config:
            actual = metrics["max_mb"]
            budget = memory_config["max_mb"]
            threshold = memory_config.get("threshold_pct", 20)
            max_allowed = budget * (1 + threshold / 100)
            
            passed = actual <= max_allowed
            message = (
                f"Memory {actual:.1f}MB within budget {budget}MB "
                f"(threshold: +{threshold}%)" if passed else
                f"Memory {actual:.1f}MB exceeds budget {budget}MB "
                f"by {((actual - budget) / budget * 100):.1f}%"
            )
            
            results.append(BudgetResult(
                passed=passed,
                metric="memory_usage",
                actual=actual,
                budget=budget,
                threshold_pct=threshold,
                message=message
            ))
        
        # CPU validation
        cpu_config = config.get("cpu", {})
        if "max_utilization_pct" in metrics and "max_utilization_pct" in cpu_config:
            actual = metrics["max_utilization_pct"]
            budget = cpu_config["max_utilization_pct"]
            threshold = cpu_config.get("threshold_pct", 15)
            max_allowed = budget * (1 + threshold / 100)
            
            passed = actual <= max_allowed
            message = (
                f"CPU {actual:.1f}% within budget {budget}% "
                f"(threshold: +{threshold}%)" if passed else
                f"CPU {actual:.1f}% exceeds budget {budget}% "
                f"by {actual - budget:.1f} percentage points"
            )
            
            results.append(BudgetResult(
                passed=passed,
                metric="cpu_utilization",
                actual=actual,
                budget=budget,
                threshold_pct=threshold,
                message=message
            ))
            
        return results
    
    def analyze_all_budgets(self, metrics: Dict[str, float]) -> Tuple[bool, List[BudgetResult]]:
        """Analyze all budget categories."""
        all_results = []
        all_results.extend(self.validate_latency_budget(metrics))
        all_results.extend(self.validate_cost_budget(metrics))
        all_results.extend(self.validate_resource_budget(metrics))
        
        self.results.extend(all_results)
        all_passed = all(result.passed for result in all_results)
        
        return all_passed, all_results
    
    def save_results(self, output_path: str = "budget_results.json"):
        """Save analysis results to file."""
        output_data = {
            "timestamp": time.time(),
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "overall_passed": all(r.passed for r in self.results)
            },
            "results": [asdict(result) for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

def load_performance_report(report_file: str) -> Dict[str, Any]:
    """Load performance report from JSON file."""
    try:
        with open(report_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Report file not found: {report_file}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in report file: {report_file}")
        return {}

def extract_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    """Extract relevant metrics from performance report."""
    metrics = {}
    
    # Token efficiency metrics
    if 'token_stats' in report:
        token_stats = report['token_stats']
        if 'efficiency_delta' in token_stats:
            metrics['token_delta'] = token_stats['efficiency_delta']
    
    # Latency metrics
    if 'latency' in report:
        latency = report['latency']
        if 'p95_ms' in latency:
            metrics['latency_p95'] = latency['p95_ms']
        if 'p99_ms' in latency:
            metrics['latency_p99'] = latency['p99_ms']
    
    # Success/error rates
    if 'success_rate' in report:
        metrics['success_rate'] = report['success_rate']
    if 'error_rate' in report:
        metrics['error_rate'] = report['error_rate']
    
    return metrics

def evaluate_budget_status(metrics: Dict[str, float], thresholds: Dict[str, float]) -> str:
    """Evaluate budget status based on metrics and thresholds."""
    # Check blocking thresholds first
    if metrics.get('token_delta', 0) > thresholds.get('token_block', float('inf')):
        return 'block'
    if metrics.get('latency_p95', 0) > thresholds.get('latency_block', float('inf')):
        return 'block'
    
    # Check warning thresholds
    if (metrics.get('token_delta', 0) > thresholds.get('token_warn', float('inf')) or
        metrics.get('latency_p95', 0) > thresholds.get('latency_warn', float('inf'))):
        return 'warn'
    
    return 'pass'

def set_github_outputs(metrics: Dict[str, float], status: str):
    """Set GitHub Actions outputs."""
    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as f:
            f.write(f"budget_status={status}\n")
            f.write(f"token_delta={metrics.get('token_delta', 0):.2f}\n")
            f.write(f"latency_p95={metrics.get('latency_p95', 0):.0f}\n")
            f.write(f"success_rate={metrics.get('success_rate', 0):.3f}\n")
            f.write(f"error_rate={metrics.get('error_rate', 0):.3f}\n")

def save_analysis_results(metrics: Dict[str, float], status: str, thresholds: Dict[str, float]):
    """Save analysis results to file for downstream processing."""
    results = {
        "timestamp": time.time(),
        "status": status,
        "metrics": metrics,
        "thresholds": thresholds,
        "analysis": {
            "token_delta_status": "pass" if metrics.get('token_delta', 0) <= thresholds.get('token_warn', float('inf')) else "warn",
            "latency_status": "pass" if metrics.get('latency_p95', 0) <= thresholds.get('latency_warn', float('inf')) else "warn",
            "overall_recommendation": "proceed" if status == "pass" else ("review" if status == "warn" else "block")
        }
    }
    
    with open("budget_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)

def analyze_budget_metrics(
    report_file: str,
    token_warn: float = 10.0,
    token_block: float = 25.0,
    latency_warn: float = 500.0,
    latency_block: float = 1000.0,
    output_format: str = 'text',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyze budget metrics from performance report.
    
    Args:
        report_file: Path to performance report JSON file
        token_warn: Token efficiency warning threshold (%)
        token_block: Token efficiency blocking threshold (%)
        latency_warn: Latency warning threshold (ms)
        latency_block: Latency blocking threshold (ms)
        output_format: Output format ('text', 'json', 'github')
        verbose: Enable verbose output
    
    Returns:
        Dictionary with analysis results
    """
    # Load performance report
    report = load_performance_report(report_file)
    if not report:
        return {
            'status': 'error',
            'metrics': {},
            'thresholds': {}
        }
    
    # Extract metrics
    metrics = extract_metrics(report)
    
    if verbose:
        print(f"Extracted metrics: {metrics}")
    
    # Set up thresholds
    thresholds = {
        'token_warn': token_warn,
        'token_block': token_block,
        'latency_warn': latency_warn,
        'latency_block': latency_block
    }
    
    # Evaluate status
    status = evaluate_budget_status(metrics, thresholds)
    
    if verbose:
        print(f"Budget status: {status}")
        print(f"Thresholds: {thresholds}")
    
    # Return results
    return {
        'status': status,
        'metrics': metrics,
        'thresholds': thresholds
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze fusion budget metrics")
    parser.add_argument("--report-file", required=True, help="Path to performance report JSON file")
    parser.add_argument("--token-warn-threshold", type=float, default=10.0, help="Token efficiency warning threshold (%)")
    parser.add_argument("--token-block-threshold", type=float, default=25.0, help="Token efficiency blocking threshold (%)")
    parser.add_argument("--latency-warn-threshold", type=float, default=500.0, help="Latency warning threshold (ms)")
    parser.add_argument("--latency-block-threshold", type=float, default=1000.0, help="Latency blocking threshold (ms)")
    parser.add_argument("--output-format", choices=['text', 'json', 'github'], default='text', help="Output format")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--output", default="budget_results.json", help="Output file path")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        # Use the analyze function
        result = analyze_budget_metrics(
            report_file=args.report_file,
            token_warn=args.token_warn_threshold,
            token_block=args.token_block_threshold,
            latency_warn=args.latency_warn_threshold,
            latency_block=args.latency_block_threshold,
            output_format=args.output_format,
            verbose=args.verbose
        )
        
        status = result['status']
        metrics = result['metrics']
        thresholds = result['thresholds']
        
        # Create analyzer for saving results
        analyzer = BudgetAnalyzer()
        
        # Save results
        analyzer.save_results(args.output)
        print(f"Results saved to {args.output}")
        
        # Output results
        if args.output_format == 'github':
            set_github_outputs(metrics, status)
            save_analysis_results(metrics, status, thresholds)
        elif args.output_format == 'json':
            print(json.dumps(result, indent=2))
        else:  # text format
            print(f"Status: {status}")
            print(f"Token Delta: {metrics.get('token_delta', 0):.1f}%")
            print(f"Latency P95: {metrics.get('latency_p95', 0):.0f}ms")
        
        # Exit with appropriate code
        if status == 'block':
            sys.exit(1)
        elif status == 'error':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Error: Budget analysis failed: {e}")
        
        # Ensure outputs are set even on error
        if args.output_format == 'github':
            set_github_outputs({
                'token_delta': 0.0,
                'latency_p95': 0.0,
                'success_rate': 0.0,
                'error_rate': 1.0
            }, 'error')
        
        sys.exit(2)

if __name__ == "__main__":
    main()