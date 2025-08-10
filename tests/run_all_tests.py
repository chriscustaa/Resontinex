#!/usr/bin/env python3
"""
Comprehensive test runner for the RESONTINEX fusion optimization system.
Executes all test suites and generates consolidated reports.
"""

import unittest
import sys
import os
import json
import time
from datetime import datetime
import subprocess

def discover_and_run_tests():
    """Discover and run all test modules."""
    test_dir = os.path.dirname(__file__)
    
    # Discover all test modules
    loader = unittest.TestLoader()
    start_dir = test_dir
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return result, end_time - start_time

def generate_test_report(result, duration):
    """Generate comprehensive test report."""
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_count = total_tests - failures - errors
    success_rate = success_count / total_tests if total_tests > 0 else 0
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed': success_count,
            'failed': failures,
            'errors': errors,
            'success_rate': round(success_rate, 3),
            'duration_seconds': round(duration, 2)
        },
        'test_modules': {
            'cross_judge': 'Unit tests for cross-judge evaluation system',
            'drift_detection': 'Unit tests for drift detection watchdog',
            'parameter_tuning': 'Unit tests for parameter optimization',
            'runtime_router': 'Unit tests for runtime routing system',
            'circuit_breaker': 'Unit tests for circuit breaker and SLO monitoring',
            'integration': 'End-to-end integration tests'
        },
        'system_capabilities': {
            'cross_judge_evaluation': 'Dual-evaluator system with rule-based validation',
            'parameter_auto_tuning': 'Grid-search optimization with scenario awareness',
            'drift_detection': 'Version monitoring with quality gates',
            'runtime_routing': 'Dynamic micro-overlay selection',
            'production_safety': 'Circuit breaker patterns and SLO monitoring',
            'ci_integration': 'Automated weekly parameter optimization'
        }
    }
    
    # Add failure details if any
    if failures > 0:
        report['failures'] = []
        for test, traceback in result.failures:
            report['failures'].append({
                'test': str(test),
                'traceback': traceback
            })
    
    if errors > 0:
        report['errors'] = []
        for test, traceback in result.errors:
            report['errors'].append({
                'test': str(test),
                'traceback': traceback
            })
    
    return report

def check_system_health():
    """Perform basic system health checks."""
    health_checks = {
        'scripts_directory': os.path.exists('../scripts'),
        'configs_directory': os.path.exists('../configs/fusion'),
        'overlay_files': os.path.exists('../configs/fusion/micro_overlays'),
        'github_workflow': os.path.exists('../.github/workflows/weekly-parameter-tuning.yml')
    }
    
    # Check required Python packages
    required_packages = ['yaml', 'json', 'subprocess', 'unittest']
    for package in required_packages:
        try:
            __import__(package)
            health_checks[f'package_{package}'] = True
        except ImportError:
            health_checks[f'package_{package}'] = False
    
    return health_checks

def validate_configurations():
    """Validate configuration files."""
    config_dir = '../configs/fusion'
    validation_results = {}
    
    # YAML configurations
    yaml_files = [
        'eval_scenarios.yaml',
        'overlay_params.yaml', 
        'drift_policy.yaml',
        'slo.yaml'
    ]
    
    for yaml_file in yaml_files:
        file_path = os.path.join(config_dir, yaml_file)
        try:
            if os.path.exists(file_path):
                import yaml
                with open(file_path, 'r') as f:
                    yaml.safe_load(f)
                validation_results[yaml_file] = 'valid'
            else:
                validation_results[yaml_file] = 'missing'
        except Exception as e:
            validation_results[yaml_file] = f'invalid: {str(e)}'
    
    # JSON configurations
    json_files = ['model_semantics_ledger.v0.1.0.json']
    
    for json_file in json_files:
        file_path = os.path.join(config_dir, json_file)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    json.load(f)
                validation_results[json_file] = 'valid'
            else:
                validation_results[json_file] = 'missing'
        except Exception as e:
            validation_results[json_file] = f'invalid: {str(e)}'
    
    return validation_results

def main():
    """Main test execution function."""
    print("=" * 80)
    print("RESONTINEX FUSION OPTIMIZATION SYSTEM - TEST SUITE")
    print("=" * 80)
    print()
    
    # Create test results directory
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run system health checks
    print("Running system health checks...")
    health_status = check_system_health()
    health_passed = all(health_status.values())
    
    print(f"System Health: {'HEALTHY' if health_passed else 'DEGRADED'}")
    for check, status in health_status.items():
        print(f"  {check}: {'PASS' if status else 'FAIL'}")
    print()
    
    # Validate configurations
    print("Validating configuration files...")
    config_validation = validate_configurations()
    config_valid = all(status == 'valid' for status in config_validation.values())
    
    print(f"Configuration Validation: {'PASS' if config_valid else 'FAIL'}")
    for config, status in config_validation.items():
        print(f"  {config}: {status}")
    print()
    
    # Run test suites
    print("Executing test suites...")
    print("-" * 40)
    
    try:
        test_result, duration = discover_and_run_tests()
        
        # Generate comprehensive report
        report = generate_test_report(test_result, duration)
        report['system_health'] = health_status
        report['configuration_validation'] = config_validation
        
        # Save detailed report
        report_path = os.path.join(results_dir, 'comprehensive_test_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print()
        print("=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Errors: {report['summary']['errors']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Duration: {report['summary']['duration_seconds']:.2f} seconds")
        print()
        
        # System status
        overall_status = 'OPERATIONAL'
        if report['summary']['success_rate'] < 0.8:
            overall_status = 'DEGRADED'
        elif not health_passed or not config_valid:
            overall_status = 'PARTIAL'
            
        print(f"OVERALL SYSTEM STATUS: {overall_status}")
        print(f"Detailed report saved to: {report_path}")
        print()
        
        # Return appropriate exit code
        if report['summary']['success_rate'] == 1.0 and health_passed and config_valid:
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"Test execution failed: {e}")
        return 2

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)