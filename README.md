# RESONTINEX

**Production-grade AI workflow orchestration with autonomous trust scoring, energy governance, and industry-first integration patterns for mission-critical deployments.**

RESONTINEX addresses the critical gap between experimental AI systems and production deployment requirements through mathematically rigorous trust evaluation, comprehensive budget enforcement, and self-healing operational patterns.

## Industry-Missing Capabilities Being Tested

### 1. Logit Aggregation Trust Gates
**Problem**: Most AI trust systems are black boxes requiring heavy dependencies  
**Solution**: Zero-dependency mathematical trust scoring with ~500ns evaluation time
- **CompositeTrustGate**: Deterministic trust evaluation with complete auditability
- **Mathematical guarantees**: Monotonic behavior, numerically stable sigmoid implementation
- **Production metrics**: Sub-microsecond performance, minimal memory footprint
- **Regulatory compliance**: Full decision context and explanation trails

```python
from resontinex.trust import CompositeTrustGate, InputVector

inputs = InputVector(0.8, 0.3, 0.2)  # alignment, risk, confidence
gate = CompositeTrustGate(inputs=inputs)
print(f"Trust: {gate.trust_score}, Decision: {gate.decision.value}")
```

### 2. Energy Governance with Automatic Safeguards
**Problem**: AI systems lack operational budget controls leading to runaway costs  
**Solution**: Multi-layered energy governance with automatic enforcement
- **Budget tripwires**: Automatic downgrade after 3 consecutive threshold breaches
- **Cost multipliers**: Dynamic adjustment based on complexity (1.5x), trust (1.3x), entropy (1.4x)
- **Emergency braking**: Hard stop at 95% budget utilization
- **Recovery mechanisms**: Insight caching, early termination, adaptive backoff

```python
from fusion_ops.budget_analysis import analyze_budget_metrics

result = analyze_budget_metrics(
    report_file="budget_report.json",
    token_warn=12.0,    # 12% increase warning
    token_block=25.0,   # 25% increase blocking
    latency_warn=2000,  # 2s P95 latency warning
    latency_block=5000  # 5s P95 latency blocking
)
```

### 3. CI/CD Budget Gates with Performance Regression Detection
**Problem**: No automated performance validation for AI systems in deployment pipelines  
**Solution**: GitHub Actions integration with automated budget enforcement
- **Fusion Budget Gates**: Token delta and latency thresholds enforced in CI/CD
- **Performance comparison**: Baseline vs current metrics with statistical analysis
- **Automatic PR comments**: Detailed budget analysis with trending data
- **Pipeline blocking**: Automatic deployment prevention on threshold violations

See [`.github/workflows/fusion-budget-gates.yml`](.github/workflows/fusion-budget-gates.yml) for complete implementation.

### 4. Runtime Micro-Overlay Intelligence
**Problem**: Static prompt engineering lacks context-aware optimization  
**Solution**: Dynamic overlay selection with comprehensive error recovery
- **Category mapping**: Financial operations → rollback-first, compliance → state-model-first
- **Complexity analysis**: High complexity (>0.7) → state modeling, low (<0.4) → observability focus  
- **Keyword triggers**: 26+ domain-specific routing patterns with confidence scoring
- **Recovery modes**: Graceful handling of corrupted overlay files with audit trails

```python
from scripts.runtime_router import RuntimeRouter

router = RuntimeRouter("./configs/fusion")
scenario = {
    'category': 'financial_operations',
    'complexity': 0.8,
    'description': 'Process refund with rollback requirements'
}
decision = router.route_scenario(scenario)
```

### 5. Production Observability Architecture
**Problem**: AI systems lack operational visibility and debugging capabilities  
**Solution**: Comprehensive observability designed for autonomous AI workflows
- **Structured logging**: JSON-formatted events with complete decision context
- **Performance tracking**: P99 latency <1ms SLO, memory usage monitoring
- **Health endpoints**: System status validation with performance benchmarks  
- **Audit compliance**: Complete decision trails for regulatory requirements

## Quick Start

### Installation
```bash
git clone https://github.com/resontinex/resontinex.git
cd resontinex
pip install -r requirements.txt
pip install -e .
```

### Trust Gate Evaluation
```python
from resontinex.trust import TrustMonitor

monitor = TrustMonitor(config_path="config/trust_gate_config.yaml")
gate = monitor.evaluate(
    alignment_score=0.85,
    epistemic_risk=0.25,
    confidence_band=0.30
)

if gate.decision == "execute":
    proceed_with_operation()
elif gate.decision == "review":
    queue_for_human_review(gate.explain())
```

### Budget Analysis
```bash
# Run budget compliance check
python fusion_ops/budget_analysis.py \
  --report-file budget_report.json \
  --token-warn-threshold 12.0 \
  --token-block-threshold 25.0 \
  --output-format github
```

### Micro-Overlay Routing
```python
from scripts.runtime_router import RuntimeRouter

router = RuntimeRouter("./configs/fusion")
enhanced_prompt = router.apply_overlay(
    base_prompt="Analyze this scenario and provide recommendations.",
    overlay_name="rollback_first"
)
```

## Core Architecture

### Trust Evaluation Pipeline
1. **Input Validation**: Strict [0,1] bounds checking with type validation
2. **Logit Aggregation**: Mathematical combination with monotonicity guarantees  
3. **Risk Tier Assignment**: Four-tier system (block/review/monitor/pass)
4. **Decision Routing**: Deterministic mapping to execution verbs
5. **Audit Trail Generation**: Complete decision context with performance metrics

### Energy Governance Flow
1. **Budget Validation**: Multi-factor cost calculation with context multipliers
2. **Tripwire Monitoring**: Consecutive breach detection with state persistence
3. **Automatic Enforcement**: Progressive constraints from warning to emergency brake
4. **Recovery Orchestration**: Insight reuse, early termination, adaptive parameters

### Micro-Overlay Intelligence
1. **Scenario Analysis**: Category classification, complexity scoring, keyword extraction
2. **Overlay Scoring**: Multi-criteria matching with confidence calculation
3. **Dynamic Selection**: Best-fit routing with fallback option identification
4. **Content Application**: Prompt enhancement with overlay-specific optimizations

## Configuration

### Trust Gate Configuration
```yaml
# config/trust_gate_config.yaml
weights:
  alignment_score: 0.45
  epistemic_risk: 0.35
  confidence_band: 0.20

risk_tiers:
  block: 0.25      # [0.00, 0.25] → Abort
  review: 0.50     # (0.25, 0.50] → Human review  
  monitor: 0.75    # (0.50, 0.75] → Execute with monitoring
  pass: 1.01       # (0.75, 1.01] → Execute freely

trust_floor: 0.60
entropy_threshold: 0.72
```

### Energy Budget Configuration  
```yaml
# config/energy_governance.yaml
budget_limits:
  hard_limit_percentage: 95
  auto_review_threshold: 85
  approval_threshold: 10000

multipliers:
  high_complexity: 1.5
  low_trust: 1.3
  high_entropy: 1.4
  emergency_discount: 0.6

budget_tripwire:
  breach_threshold: 12.0
  consecutive_limit: 3
  auto_recovery: false
```

## Performance Characteristics

### Trust Gate Performance
- **Hot Path**: ~500ns per evaluation
- **Memory Usage**: <1KB per instance  
- **P99 Latency**: <1ms SLO
- **Throughput**: >10k evaluations/second
- **Thread Safety**: Stateless, immutable design

### Budget Analysis Performance  
- **Processing Time**: <100ms for routing decisions
- **Memory Footprint**: <50MB baseline, scales with overlay count
- **Startup Time**: <2 seconds with full configuration loading
- **Concurrency**: Thread-safe with minimal lock contention

### System Reliability
- **Circuit Breaker**: Automatic failure detection and recovery
- **Graceful Degradation**: Fallback to baseline operation on failures
- **Error Recovery**: Comprehensive retry logic with exponential backoff
- **State Consistency**: Reentrant locking for concurrent operations

## Testing Framework

### Budget Gate Validation
```bash
# CI/CD pipeline integration
pytest tests/test_budget_gates.py::test_fusion_budget_compliance -v
python fusion_ops/budget_analysis.py --report-file budget_report.json
```

### Trust Gate Testing
```bash
# Performance benchmarks
pytest tests/trust/test_composite_trust_gate.py::TestPerformanceBenchmarks -v

# Integration testing
python -c "
from resontinex.trust import CompositeTrustGate, InputVector
import time
start = time.perf_counter_ns()
gate = CompositeTrustGate(inputs=InputVector(0.8, 0.3, 0.2))
end = time.perf_counter_ns()
print(f'Performance: {end - start}ns')
"
```

### Overlay Router Testing
```bash
# Routing decision validation
python scripts/runtime_router.py  # Demo with test scenarios
pytest tests/test_runtime_router.py -v
```

## Operational Excellence

### Monitoring Integration
- **Health Endpoints**: `/fusion/health`, `/fusion/budget/status`
- **Metrics Collection**: Prometheus-compatible with cardinality controls
- **Alert Configuration**: P99 latency, error rates, budget violations
- **Audit Logging**: Complete decision trails with structured JSON events

### Production Deployment
- **Container Ready**: Docker configurations with resource limits
- **Scaling Patterns**: Horizontal scaling with no shared state
- **Config Management**: Environment-specific overrides with validation
- **Rollback Procedures**: Automated fallback to baseline on failures

## Documentation

- **Trust Gates**: [`docs/composite_trust_gate_final.md`](docs/composite_trust_gate_final.md)
- **Energy Governance**: [`docs/energy_governance_technical_guide.md`](docs/energy_governance_technical_guide.md)
- **Operational Runbook**: [`docs/operational_runbook.md`](docs/operational_runbook.md)
- **API Reference**: [`docs/api_reference_guide.md`](docs/api_reference_guide.md)

## Current Testing Focus

RESONTINEX serves as a testbed for production-grade AI workflow patterns currently missing in the industry:

1. **Mathematical rigor in AI trust evaluation** - Moving beyond heuristics to provable guarantees
2. **Operational budget controls** - Preventing runaway costs through automated enforcement  
3. **CI/CD integration for AI systems** - Treating AI performance as a first-class software metric
4. **Self-healing AI infrastructure** - Building resilient systems that gracefully handle failures
5. **Regulatory-compliant AI decision trails** - Complete auditability for compliance requirements

These integration tactics represent critical gaps between experimental AI development and production deployment requirements.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Documentation**: Comprehensive inline documentation with type hints
- **Integration Support**: [`examples/`](examples/) directory with production workflows
- **Issue Tracking**: GitHub issues for bugs and feature requests
- **Enterprise Support**: Contact for production deployment assistance
