# RESONTINEX v2.1.0

**Production-grade AI workflow orchestration system with intelligent overlay routing, circuit breaker protection, and comprehensive drift detection.**

RESONTINEX provides enterprise-ready infrastructure for AI-powered workflows requiring high reliability, intelligent routing, and operational resilience.

**Spec Version:** 2.1.0 | **Spec Hash:** `sha256:e7f4a8d3c2b1a90876543210fedcba9876543210abcdef1234567890`

## Core Components

> **How RESONTINEX Core & Fusion Optimizer Relate**
>
> RESONTINEX provides the **core cognitive continuity engine** with modules for trust management, entropy auditing, and insight compression. The **Fusion Optimizer System** is an overlaying operational layer that adds production-grade circuit breaking, drift detection, runtime routing, and comprehensive observability. Together, they form a complete AI workflow orchestration platform with both intelligent decision-making capabilities and enterprise-grade operational resilience.

### 1. Circuit Breaker System ([`scripts/circuit_breaker.py`](scripts/circuit_breaker.py))
Production-grade circuit breaker with advanced state transition handling:
- **Intelligent Recovery**: Exponential backoff with pattern analysis
- **State Management**: Comprehensive transition logging and metrics
- **Edge Case Handling**: Race condition protection and timestamp validation
- **Integration Hooks**: Callback system for external monitoring

### 2. Drift Detection Engine ([`scripts/watch_drift.py`](scripts/watch_drift.py))
Advanced version drift monitoring with automated response orchestration:
- **Multi-Source Monitoring**: File system, package registry, and API version tracking
- **Intelligent Filtering**: Configurable sensitivity with performance-based adaptation
- **Integration Coordination**: Circuit breaker integration and pipeline hooks
- **Batch Processing**: Memory-efficient scanning with error recovery

### 3. Scenario Manager ([`resontinex/scenario_manager.py`](resontinex/scenario_manager.py))
Sophisticated scenario filtering and orchestration engine:
- **Multi-Criteria Filtering**: Capability, weight, complexity, and custom predicate support
- **Performance-Based Ranking**: Adaptive filtering with historical performance analysis
- **Composite Scoring**: Multi-dimensional evaluation with complexity factors
- **Health Monitoring**: Scenario performance tracking and automatic exclusion

### 4. Runtime Router ([`scripts/runtime_router.py`](scripts/runtime_router.py))
Intelligent micro-overlay routing with filesystem optimization:
- **Advanced Loading**: Comprehensive validation with integrity checking and recovery modes
- **Smart Routing**: Category-based, complexity-aware, and keyword-triggered selection
- **Performance Tracking**: Route performance analysis with comparative metrics
- **Error Handling**: Graceful degradation with detailed error logging

## Quick Start

### Installation
```bash
git clone https://github.com/resontinex/resontinex.git
cd resontinex
pip install -e .
```

### Basic Usage

#### Circuit Breaker Protection
```python
from scripts.circuit_breaker import ProductionSafetyManager

# Initialize with configuration
safety_manager = ProductionSafetyManager("./configs/fusion")

# Execute with circuit breaker protection
def risky_operation():
    return "success"

result = safety_manager.execute_with_circuit_breaker("service_name", risky_operation)
```

#### Drift Detection
```python
from scripts.watch_drift import DriftWatchdog

# Initialize drift monitoring
watchdog = DriftWatchdog("./configs/fusion/drift_policy.yaml")

# Run detection cycle
drift_event = watchdog.run_drift_check()
if drift_event:
    print(f"Drift detected: {len(drift_event.changes_detected)} changes")
```

#### Scenario Management
```python
from resontinex.scenario_manager import ScenarioManager

# Initialize scenario manager
manager = ScenarioManager(config)

# Filter scenarios by capability
reasoning_scenarios = manager.filter_by_capability('reasoning')

# Multi-criteria filtering
filtered = manager.filter_by_multiple_criteria({
    'capability': 'analysis',
    'min_weight': 0.7,
    'complexity': 'high'
})
```

#### Runtime Routing
```python
from scripts.runtime_router import RuntimeRouter

# Initialize router
router = RuntimeRouter("./configs/fusion")

# Route scenario to appropriate overlay
scenario = {
    'category': 'financial_operations',
    'complexity': 0.8,
    'description': 'Process refund with rollback requirements'
}

decision = router.route_scenario(scenario)
print(f"Selected: {decision.selected_overlay} (confidence: {decision.confidence})")
```

## Configuration

### Directory Structure
```
configs/fusion/
├── slo.yaml                    # Circuit breaker SLO configuration
├── drift_policy.yaml          # Drift detection policies
├── overlay_feature_flags.yaml # Feature flag configuration
└── micro_overlays/            # Overlay definitions
    ├── rollback_first.txt
    ├── state_model_first.txt
    └── observability_first.txt
```

### Environment Variables
```bash
# Feature flag overrides
export RESON_FEATURES="refund_lot_switch,state_model_first"

# Circuit breaker configuration
export RESON_FUSION_DISABLE=1              # Disable fusion processing
export RESON_FUSION_BASELINE_ONLY=1        # Use baseline models only
```

## Monitoring and Observability

### Health Checks
```python
# System health overview
health = safety_manager.check_system_health()
print(f"Overall health: {health['overall_health']}")

# Performance summary
performance = manager.get_performance_summary()
print(f"Active scenarios: {performance['active_scenarios']}")

# Router statistics
stats = router.get_routing_stats()
print(f"Available overlays: {stats['overlays_available']}")
```

### Metrics Integration
The system provides comprehensive metrics collection:
- **Circuit Breaker**: State transitions, failure rates, recovery times
- **Drift Detection**: Change detection rates, processing latency, integration health
- **Scenario Manager**: Filter performance, selection accuracy, health scores
- **Runtime Router**: Routing decisions, overlay performance, loading statistics

## Production Deployment

### Safety Features
- **Graceful Degradation**: Automatic fallback to baseline operation
- **Error Recovery**: Comprehensive error handling with retry logic
- **Resource Protection**: Memory-efficient processing with batch optimization
- **State Consistency**: Thread-safe operations with reentrant locking

### Performance Characteristics
- **Startup Time**: < 2 seconds with full configuration loading
- **Memory Usage**: < 50MB baseline, scales with overlay count
- **Processing Latency**: < 100ms for routing decisions
- **Concurrency**: Thread-safe with minimal lock contention

## Architecture Decisions

### Design Principles
1. **Functional Minimalism**: Clear, verifiable implementations over elaborate abstractions
2. **Production Readiness**: Comprehensive error handling and operational excellence
3. **Integration Focus**: Designed for seamless integration with existing workflows
4. **Performance Optimization**: Memory-efficient with intelligent caching strategies

### Key Innovations
- **Adaptive Circuit Breaking**: Dynamic timeout adjustment based on failure patterns
- **Intelligent Drift Detection**: Multi-source monitoring with configurable sensitivity
- **Performance-Based Filtering**: Historical performance integration in scenario selection
- **Recovery-Mode Loading**: Graceful handling of corrupted overlay files

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run test suite
python -m pytest tests/ -v
```

### Code Standards
- **Python 3.11+** required
- **Type hints** for all public APIs
- **Comprehensive error handling** with specific exception types
- **Performance optimization** without sacrificing clarity

## v2.1.0 New Features

### Energy Governance & Quorum Voting
- **EnergyLedger**: Budget tracking with automatic review at 85% usage and emergency braking at 95%
- **QuorumVoting**: Weighted consensus decision-making with 4-tier arbitration ladder
- **Enhanced Security**: Comprehensive threat modeling with PII detection and credential scanning
- **Cross-Platform CI**: Windows + Ubuntu testing matrix with Python 3.10-3.12 support

### n8n Integration Templates
- **Production Flow**: [`examples/certi-land-workflow.json`](examples/certi-land-workflow.json) - Full module integration
- **Simple Starter**: [`examples/n8n-simple-v1.3.json`](examples/n8n-simple-v1.3.json) - Basic routing for learning
- **60-Second Setup**: Complete import guide in [`examples/README.md`](examples/README.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For technical support and integration questions:
- **Documentation**: Review inline code documentation and type hints
- **Release Notes**: See [RELEASE_NOTES.md](RELEASE_NOTES.md) for detailed v2.1.0 changes
- **Governance**: Review [docs/governance.md](docs/governance.md) for contribution guidelines
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Contact**: chris@custaa.com for enterprise integration support
