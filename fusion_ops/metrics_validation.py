"""
Metrics validation module for fusion CI operations.
Validates metrics collection and format compliance.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ValidationResult:
    """Metrics validation result."""
    check_name: str
    passed: bool
    message: str
    severity: str  # "error", "warning", "info"
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

class MetricsValidator:
    """Validates metrics structure, completeness, and compliance."""
    
    def __init__(self, schema_path: Optional[str] = None):
        self.schema_path = schema_path
        self.results: List[ValidationResult] = []
        
        # Default required metrics schema
        self.required_metrics = {
            "latency_p95_ms": {"type": "number", "min": 0, "max": 10000},
            "latency_p99_ms": {"type": "number", "min": 0, "max": 20000},
            "throughput_rps": {"type": "number", "min": 0, "max": 100000},
            "memory_mb": {"type": "number", "min": 0, "max": 8192},
            "cpu_pct": {"type": "number", "min": 0, "max": 100},
            "error_rate_pct": {"type": "number", "min": 0, "max": 100}
        }
        
        self.optional_metrics = {
            "cost_per_request_cents": {"type": "number", "min": 0, "max": 100},
            "disk_io_mb_per_sec": {"type": "number", "min": 0, "max": 10000},
            "network_io_mb_per_sec": {"type": "number", "min": 0, "max": 1000}
        }
    
    def load_schema(self) -> Dict[str, Any]:
        """Load metrics schema if provided."""
        if not self.schema_path or not Path(self.schema_path).exists():
            return {
                "required": self.required_metrics,
                "optional": self.optional_metrics
            }
        
        with open(self.schema_path) as f:
            return json.load(f)
    
    def validate_structure(self, metrics: Dict[str, Any]) -> List[ValidationResult]:
        """Validate basic metrics structure."""
        results = []
        
        # Check if metrics is a dictionary
        if not isinstance(metrics, dict):
            results.append(ValidationResult(
                check_name="structure_type",
                passed=False,
                severity="error",
                message=f"Metrics must be a dictionary, got {type(metrics).__name__}"
            ))
            return results
        
        # Check for empty metrics
        if not metrics:
            results.append(ValidationResult(
                check_name="structure_empty",
                passed=False,
                severity="error",
                message="Metrics dictionary is empty"
            ))
            return results
        
        # Structure is valid
        results.append(ValidationResult(
            check_name="structure_valid",
            passed=True,
            severity="info",
            message=f"Metrics structure valid with {len(metrics)} entries"
        ))
        
        return results
    
    def validate_required_metrics(self, metrics: Dict[str, Any]) -> List[ValidationResult]:
        """Validate presence and format of required metrics."""
        results = []
        schema = self.load_schema()
        required = schema.get("required", self.required_metrics)
        
        for metric_name, constraints in required.items():
            if metric_name not in metrics:
                results.append(ValidationResult(
                    check_name=f"required_{metric_name}",
                    passed=False,
                    severity="error",
                    message=f"Required metric '{metric_name}' is missing"
                ))
                continue
            
            value = metrics[metric_name]
            
            # Validate type
            if constraints.get("type") == "number" and not isinstance(value, (int, float)):
                results.append(ValidationResult(
                    check_name=f"type_{metric_name}",
                    passed=False,
                    severity="error",
                    message=f"Metric '{metric_name}' must be a number, got {type(value).__name__}"
                ))
                continue
            
            # Validate range
            if isinstance(value, (int, float)):
                min_val = constraints.get("min")
                max_val = constraints.get("max")
                
                if min_val is not None and value < min_val:
                    results.append(ValidationResult(
                        check_name=f"range_{metric_name}_min",
                        passed=False,
                        severity="warning",
                        message=f"Metric '{metric_name}' value {value} below minimum {min_val}"
                    ))
                    continue
                
                if max_val is not None and value > max_val:
                    results.append(ValidationResult(
                        check_name=f"range_{metric_name}_max",
                        passed=False,
                        severity="warning",
                        message=f"Metric '{metric_name}' value {value} exceeds maximum {max_val}"
                    ))
                    continue
            
            # Metric is valid
            results.append(ValidationResult(
                check_name=f"valid_{metric_name}",
                passed=True,
                severity="info",
                message=f"Required metric '{metric_name}' = {value} is valid"
            ))
        
        return results
    
    def validate_metric_relationships(self, metrics: Dict[str, Any]) -> List[ValidationResult]:
        """Validate logical relationships between metrics."""
        results = []
        
        # P99 latency should be >= P95 latency
        if "latency_p95_ms" in metrics and "latency_p99_ms" in metrics:
            p95 = metrics["latency_p95_ms"]
            p99 = metrics["latency_p99_ms"]
            
            if isinstance(p95, (int, float)) and isinstance(p99, (int, float)):
                if p99 < p95:
                    results.append(ValidationResult(
                        check_name="latency_percentile_relationship",
                        passed=False,
                        severity="error",
                        message=f"P99 latency ({p99}ms) cannot be less than P95 latency ({p95}ms)"
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="latency_percentile_relationship",
                        passed=True,
                        severity="info",
                        message="Latency percentiles are logically consistent"
                    ))
        
        # Error rate should correlate reasonably with throughput
        if "error_rate_pct" in metrics and "throughput_rps" in metrics:
            error_rate = metrics["error_rate_pct"]
            throughput = metrics["throughput_rps"]
            
            if isinstance(error_rate, (int, float)) and isinstance(throughput, (int, float)):
                # High error rate with high throughput is suspicious
                if error_rate > 50 and throughput > 1000:
                    results.append(ValidationResult(
                        check_name="error_throughput_relationship",
                        passed=False,
                        severity="warning",
                        message=f"High error rate ({error_rate}%) with high throughput ({throughput} RPS) is unusual",
                        details={"error_rate_pct": error_rate, "throughput_rps": throughput}
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="error_throughput_relationship",
                        passed=True,
                        severity="info",
                        message="Error rate and throughput relationship is reasonable"
                    ))
        
        return results
    
    def validate_naming_conventions(self, metrics: Dict[str, Any]) -> List[ValidationResult]:
        """Validate metric naming conventions."""
        results = []
        
        # Check naming pattern (snake_case with units)
        valid_pattern = re.compile(r'^[a-z][a-z0-9_]*[a-z0-9]$')
        unit_suffixes = ['_ms', '_sec', '_mb', '_gb', '_rps', '_pct', '_count', '_rate']
        
        invalid_names = []
        missing_units = []
        
        for metric_name in metrics:
            if not valid_pattern.match(metric_name):
                invalid_names.append(metric_name)
            
            # Check for unit suffix
            has_unit = any(metric_name.endswith(suffix) for suffix in unit_suffixes)
            if not has_unit and not metric_name.endswith('_id'):  # Allow _id suffix
                missing_units.append(metric_name)
        
        if invalid_names:
            results.append(ValidationResult(
                check_name="naming_convention",
                passed=False,
                severity="warning",
                message=f"Invalid metric names (should be snake_case): {', '.join(invalid_names)}"
            ))
        
        if missing_units:
            results.append(ValidationResult(
                check_name="unit_suffixes",
                passed=False,
                severity="warning",
                message=f"Metrics missing unit suffixes: {', '.join(missing_units)}"
            ))
        
        if not invalid_names and not missing_units:
            results.append(ValidationResult(
                check_name="naming_conventions",
                passed=True,
                severity="info",
                message="All metric names follow conventions"
            ))
        
        return results
    
    def validate_all(self, metrics: Dict[str, Any]) -> tuple[bool, List[ValidationResult]]:
        """Run all validation checks."""
        all_results = []
        
        # Run all validation checks
        all_results.extend(self.validate_structure(metrics))
        all_results.extend(self.validate_required_metrics(metrics))
        all_results.extend(self.validate_metric_relationships(metrics))
        all_results.extend(self.validate_naming_conventions(metrics))
        
        self.results.extend(all_results)
        
        # Overall pass/fail based on error-level issues
        errors = [r for r in all_results if not r.passed and r.severity == "error"]
        overall_pass = len(errors) == 0
        
        return overall_pass, all_results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        if not self.results:
            return {"status": "no_data"}
        
        error_count = sum(1 for r in self.results if not r.passed and r.severity == "error")
        warning_count = sum(1 for r in self.results if not r.passed and r.severity == "warning")
        passed_count = sum(1 for r in self.results if r.passed)
        
        status = "pass"
        if error_count > 0:
            status = "fail"
        elif warning_count > 0:
            status = "warning"
        
        return {
            "status": status,
            "total_checks": len(self.results),
            "passed": passed_count,
            "errors": error_count,
            "warnings": warning_count,
            "overall_pass": error_count == 0
        }
    
    def save_results(self, output_path: str = "metrics_validation.json"):
        """Save validation results."""
        output_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.generate_summary(),
            "results": [asdict(result) for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

def main():
    """CLI entry point for metrics validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate metrics format and completeness")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    parser.add_argument("--schema", help="Path to metrics schema JSON file")
    parser.add_argument("--output", default="metrics_validation.json", help="Output path")
    
    args = parser.parse_args()
    
    # Load metrics
    try:
        with open(args.metrics) as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load metrics: {e}")
        exit(1)
    
    # Validate metrics
    validator = MetricsValidator(args.schema)
    overall_pass, results = validator.validate_all(metrics)
    
    # Print results
    print(f"Metrics Validation Results ({len(results)} checks)")
    print("=" * 50)
    
    for result in results:
        if result.severity == "error":
            emoji = "❌"
        elif result.severity == "warning":
            emoji = "⚠️"
        else:
            emoji = "✅"
        
        status = "PASS" if result.passed else result.severity.upper()
        print(f"{emoji} {result.check_name}: {status} - {result.message}")
    
    print("=" * 50)
    summary = validator.generate_summary()
    print(f"Summary: {summary['status'].upper()} - "
          f"{summary['passed']}/{summary['total_checks']} passed "
          f"({summary['errors']} errors, {summary['warnings']} warnings)")
    
    # Save results
    validator.save_results(args.output)
    print(f"Results saved to {args.output}")
    
    # Exit with appropriate code
    exit(0 if overall_pass else 1)

if __name__ == "__main__":
    main()