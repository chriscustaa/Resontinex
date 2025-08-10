# RESONTINEX Fusion Optimization System - Test Suite

Comprehensive testing and validation pipeline for the self-optimizing fusion capability.

## Overview

This test suite validates all components of the upgraded RESONTINEX fusion stack:

- **Cross-Judge Evaluation**: Dual-evaluator system with rule-based validation
- **Parameter Auto-Tuning**: Grid-search optimization with scenario awareness  
- **Drift Detection**: Version monitoring with quality gates
- **Runtime Routing**: Dynamic micro-overlay selection
- **Production Safety**: Circuit breaker patterns and SLO monitoring
- **CI Integration**: Automated weekly parameter optimization

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── test_cross_judge.py         # Cross-judge evaluation tests
├── test_drift_detection.py     # Drift detection watchdog tests  
├── test_parameter_tuning.py    # Parameter optimization tests
├── test_runtime_router.py      # Runtime routing tests
├── test_circuit_breaker.py     # Circuit breaker and SLO tests
├── test_integration.py         # End-to-end integration tests
├── run_all_tests.py           # Comprehensive test runner
└── README.md                   # This documentation
```

## Running Tests

### Individual Test Suites

Run specific test modules:

```bash
# Cross-judge evaluation tests
python -m pytest tests/test_cross_judge.py -v

# Drift detection tests  
python -m pytest tests/test_drift_detection.py -v

# Parameter tuning tests
python -m pytest tests/test_parameter_tuning.py -v

# Runtime routing tests
python -m pytest tests/test_runtime_router.py -v

# Circuit breaker tests
python -m pytest tests/test_circuit_breaker.py -v

# Integration tests
python -m pytest tests/test_integration.py -v
```

### Comprehensive Test Execution

Run all tests with detailed reporting:

```bash
# Execute complete test suite
python tests/run_all_tests.py
```

This will:
- Run system health checks
- Validate configuration files
- Execute all test modules
- Generate comprehensive reports
- Create `test_results/comprehensive_test_report.json`

### Using Standard unittest

```bash
# Run individual test file
python tests/test_cross_judge.py

# Run all tests in directory
python -m unittest discover tests/ -v
```

## Test Categories

### Unit Tests

Each component has dedicated unit tests covering:

- **Initialization and Configuration**: Proper component setup
- **Core Functionality**: Primary feature validation  
- **Edge Cases**: Boundary conditions and error handling
- **Integration Points**: Interface and dependency testing

### Integration Tests

End-to-end validation covering:

- **System Health**: Directory structure and dependencies
- **Configuration Loading**: YAML/JSON parsing validation
- **Workflow Orchestration**: Multi-component interactions
- **GitHub Actions**: CI/CD workflow validation

## Test Results

### Individual Test Results

Each test module generates results in `test_results/`:

- `cross_judge_results.json`
- `drift_detection_results.json`  
- `parameter_tuning_results.json`
- `runtime_router_results.json`
- `circuit_breaker_results.json`
- `integration_results.json`

### Comprehensive Report

The main test runner creates:

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "summary": {
    "total_tests": 85,
    "passed": 82,
    "failed": 2, 
    "errors": 1,
    "success_rate": 0.965,
    "duration_seconds": 45.32
  },
  "system_health": {
    "scripts_directory": true,
    "configs_directory": true,
    "overlay_files": true,
    "github_workflow": true
  },
  "configuration_validation": {
    "eval_scenarios.yaml": "valid",
    "overlay_params.yaml": "valid",
    "drift_policy.yaml": "valid",
    "slo.yaml": "valid"
  }
}
```

## Continuous Integration

### GitHub Actions Integration

Tests automatically run via `.github/workflows/weekly-parameter-tuning.yml`:

```yaml
- name: Run Test Suite
  run: python tests/run_all_tests.py

- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: tests/test_results/
```

### Local CI Validation

Validate CI configuration:

```bash
# Test GitHub workflow syntax
python tests/test_integration.py::TestFusionSystemIntegration::test_github_workflow_configuration
```

## Development Workflow

### Adding New Tests

1. Create test file: `tests/test_new_component.py`
2. Implement test cases following existing patterns
3. Add to import structure in `__init__.py` if needed
4. Run via `run_all_tests.py` to validate integration

### Test-Driven Development

1. Write failing tests for new functionality
2. Implement minimum code to pass tests  
3. Refactor while maintaining test coverage
4. Validate with full test suite

## System Requirements

### Python Dependencies

- `unittest` (standard library)
- `yaml` (PyYAML)
- `json` (standard library)
- `subprocess` (standard library)
- `tempfile` (standard library)

### Optional Testing Tools

- `pytest` (enhanced test runner)
- `coverage` (test coverage analysis)
- `mock` (advanced mocking, though `unittest.mock` is used)

## Troubleshooting

### Import Errors

Tests use conditional imports to handle missing dependencies:

```python
try:
    from component import Class
except ImportError as e:
    print(f"Warning: Could not import: {e}")
```

Tests will skip gracefully if components are unavailable.

### Configuration Issues

Run configuration validation separately:

```python
python -c "
import tests.test_integration as t
validator = t.TestFusionSystemIntegration()
validator.test_configuration_loading_integration()
"
```

### Path Resolution

Tests add scripts directory to Python path:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
```

Ensure relative paths work from the test directory.

## Success Metrics

### Target Coverage

- **Unit Tests**: >90% code coverage per component
- **Integration Tests**: All major workflows validated
- **Configuration Tests**: All config files parseable
- **System Health**: All dependencies verified

### Quality Gates

- **Success Rate**: >95% test passage
- **Performance**: <60 seconds total execution  
- **Reliability**: No flaky or intermittent failures
- **Maintainability**: Clear test organization and documentation

## Next Steps

### Enhanced Testing

- **Performance Benchmarking**: Add latency and throughput tests
- **Load Testing**: Validate system behavior under stress
- **Security Testing**: Add vulnerability scanning
- **Chaos Engineering**: Test resilience to random failures

### Advanced Validation

- **Property-Based Testing**: Generate test cases automatically
- **Mutation Testing**: Verify test quality by introducing bugs
- **Contract Testing**: Validate API interfaces
- **End-to-End Automation**: Full user journey validation

The testing pipeline provides comprehensive validation while maintaining the system's focus on functional minimalism and production readiness.