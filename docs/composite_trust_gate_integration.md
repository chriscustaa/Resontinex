# CompositeTrustGate Integration Guide

## Overview

The CompositeTrustGate system provides production-ready trust scoring with mathematical rigor, calibration support, and comprehensive observability for RESONTINEX. This enhanced system replaces the basic CompositeSignal with critical fixes and enterprise features.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   TrustMonitor  │───▶│ CompositeTrustGate │───▶│ DecisionRouter  │
│                 │    │                    │    │                 │
│ - Config Mgmt   │    │ - Logit Aggreg.   │    │ - Risk Tiers    │
│ - Calibration   │    │ - Monotonicity    │    │ - Actions       │
│ - Observability │    │ - Explainability  │    │ - Voting Weight │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
         │                       │                          │
         ▼                       ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MetricsSink   │    │   AuditLog       │    │ BreakGlass      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **CompositeTrustGate**: Core trust scoring with frozen dataclass and mathematical guarantees
- **TrustMonitor**: Integration layer with calibration and observability
- **CalibrationModel**: Isotonic/Platt scaling for score calibration
- **MetricsSink**: Performance and usage metrics collection
- **AuditLog**: Structured logging for compliance and debugging

## Installation

### Prerequisites

```bash
# Core dependencies (always required)
pip install numpy pydantic

# Optional dependencies for enhanced features
pip install scipy scikit-learn  # For calibration models
pip install hypothesis          # For property-based testing
```

### RESONTINEX Integration

Add to your `requirements.txt`:
```
numpy>=1.21.0
pydantic>=1.9.0
scipy>=1.7.0        # Optional: enables calibration
scikit-learn>=1.0.0 # Optional: enables calibration
hypothesis>=6.0.0   # Optional: enables property-based tests
```

## Configuration

### Production Configuration

```yaml
# configs/fusion/trust_gate.yaml
weights:
  epistemic_risk: 0.35
  alignment_score: 0.45
  confidence_band: 0.20

risk_tiers:
  block: 0.25
  review: 0.50
  monitor: 0.75
  pass: 1.01

tier_fallbacks:
  undefined: "review"
  review: "manual"
  manual: "block"

trust_floor: 0.60
entropy_threshold: 0.72

calibration:
  enabled: true
  method: "isotonic"
  min_samples: 50
  auto_retrain: true
```

### Environment-Specific Configs

- `trust_gate.yaml` - Production settings
- `trust_gate_development.yaml` - Development/testing settings
- `trust_gate_staging.yaml` - Staging environment settings

## Basic Usage

### Simple Trust Scoring

```python
from resontinex.core.composite_trust_gate import CompositeTrustGate

# Create trust gate
gate = CompositeTrustGate(
    epistemic_risk=0.2,     # Model uncertainty [0-1]
    alignment_score=0.8,    # Behavioral alignment [0-1]
    confidence_band=0.1     # Statistical confidence width [0-1]
)

# Get trust score and decision
print(f"Trust Score: {gate.trust_score}")
print(f"Risk Tier: {gate.risk_tier}")
print(f"Action: {gate.downstream_action()['decision']}")
```

### Production Integration

```python
from resontinex.core.composite_trust_gate import TrustMonitor

# Initialize monitor with configuration
monitor = TrustMonitor("./configs/fusion/trust_gate.yaml")

# Evaluate trust with observability
gate = monitor.evaluate(
    epistemic_risk=0.3,
    alignment_score=0.7,
    confidence_band=0.2
)

# Get comprehensive decision information
decision = gate.downstream_action()
explanation = gate.explain_score()
```

## Integration Patterns

### 1. ScenarioManager Integration

```python
from resontinex.core.composite_trust_gate import create_from_scenario_metrics
from resontinex.runtime.api import ScenarioManager

# Load scenarios
manager = ScenarioManager.load("./scenarios/")
scenarios = manager.get_scenarios()

# Enhanced trust-based filtering
trusted_scenarios = {}
for scenario_id, scenario in scenarios.items():
    if hasattr(scenario, 'metrics'):
        gate = create_from_scenario_metrics(scenario.metrics)
        
        if gate.trust_score >= 0.8 and gate.trust_floor_met:
            trusted_scenarios[scenario_id] = scenario

print(f"Filtered {len(trusted_scenarios)} trusted scenarios")
```

### 2. Enhanced TrustManager Integration

```python
from resontinex.core.composite_trust_gate import enhance_trust_manager_scoring

# Legacy TrustManager compatibility
legacy_result = enhance_trust_manager_scoring(
    alignment_score=0.8,
    inflation_delta=0.1,
    epistemic_risk=0.2,
    confidence_band=0.15
)

# Get both enhanced and legacy scores
enhanced_trust = legacy_result["trust_gate"]["trust_score"]
legacy_trust = legacy_result["legacy_score"]  # For backward compatibility
action = legacy_result["downstream_action"]
```

### 3. Risk-Based Decision Routing

```python
def route_decision(input_data):
    """Route decisions based on CompositeTrustGate assessment."""
    monitor = TrustMonitor()
    gate = monitor.evaluate(**input_data)
    
    routing_map = {
        "block": "security_review_queue",
        "review": "human_review_queue",
        "monitor": "automated_processing", 
        "pass": "fast_track"
    }
    
    return {
        "route": routing_map[gate.risk_tier],
        "gate_data": gate.to_dict(),
        "explanation": gate.explain_score()
    }
```

## Calibration Workflows

### 1. Model Fitting

```python
from resontinex.core.composite_trust_gate import fit_calibration_model

# Collect historical data
scores = []  # Raw trust scores
labels = []  # Ground truth binary outcomes

# Load historical trust decisions and outcomes
for decision in historical_decisions:
    scores.append(decision.raw_trust_score)
    labels.append(decision.was_successful)

# Fit calibration model
calibration_model = fit_calibration_model(
    scores=scores,
    labels=labels,
    method="isotonic"  # or "platt"
)

if calibration_model:
    print("Calibration model fitted successfully")
    
    # Apply to new evaluations
    monitor.calibration_model = calibration_model
```

### 2. Threshold Derivation

```python
from resontinex.core.composite_trust_gate import derive_tier_thresholds

# Derive optimal thresholds from calibrated scores
calibrated_scores = [
    calibration_model.predict_proba([score])[0]
    for score in historical_scores
]

optimal_thresholds = derive_tier_thresholds(calibrated_scores)
print("Optimal tier thresholds:", optimal_thresholds)

# Update configuration
config["risk_tiers"].update(optimal_thresholds)
```

### 3. Continuous Calibration

```python
import schedule
import time

def retrain_calibration():
    """Periodic calibration retraining."""
    monitor = TrustMonitor()
    
    # Fetch recent data
    recent_data = fetch_recent_trust_data()
    scores = [d['trust_score'] for d in recent_data]
    labels = [d['outcome'] for d in recent_data]
    
    # Retrain if enough data
    if len(scores) >= 50:
        success = monitor.fit_calibration(scores, labels)
        if success:
            print("Calibration model updated")

# Schedule periodic retraining
schedule.every().day.at("02:00").do(retrain_calibration)
```

## Mathematical Properties

### Monotonicity Guarantees

The system provides mathematical guarantees for monotonic behavior:

- **Increasing `alignment_score`** → **Increasing `trust_score`**
- **Increasing `epistemic_risk`** → **Decreasing `trust_score`**
- **Increasing `confidence_band`** → **Decreasing `trust_score`**

These properties are enforced through:
1. Directional logit transforms
2. Proper weight normalization
3. Boundary clamping with epsilon tolerance

### Logit Aggregation Formula

```
trust_score = sigmoid(
    w_alignment * logit(alignment_score) +
    w_risk * logit(1 - epistemic_risk) +
    w_confidence * logit(1 - confidence_band)
)
```

Where weights are normalized to sum to 1.0 and inputs are clamped to [ε, 1-ε].

## Error Handling and Fallbacks

### Configuration Validation

```python
try:
    gate = CompositeTrustGate(
        epistemic_risk=1.5,  # Invalid: > 1.0
        alignment_score=0.8,
        confidence_band=0.1
    )
except ValueError as e:
    print(f"Configuration error: {e}")
    # Use safe defaults or fallback configuration
```

### Tier Fallbacks

The system provides safe fallbacks for undefined risk tiers:
1. `undefined` → `review`
2. `review` → `manual`
3. `manual` → `block` (ultimate safety)

### Graceful Degradation

```python
# When calibration fails
if not monitor.calibration_model:
    # Falls back to raw trust scores
    gate = monitor.evaluate(0.2, 0.8, 0.1)
    assert gate.calibrated_score is None
    assert gate.trust_score is not None  # Raw score still available
```

## Observability and Monitoring

### Metrics Collection

```python
# Automatic metrics collection
monitor = TrustMonitor()
gate = monitor.evaluate(0.2, 0.8, 0.1)

# Access metrics
print(f"Evaluations: {monitor.metrics_sink.evaluation_count}")
print(f"Score distribution: {monitor.metrics_sink.score_distribution}")
```

### Audit Logging

```python
# Automatic audit logging
gate = monitor.evaluate(0.3, 0.7, 0.2)

# Logs include:
# - Input parameters
# - Trust scores (raw + calibrated)
# - Risk tier and reasoning
# - Decision routing information
# - Explainability trace
```

### Health Monitoring

```python
# Check system health
health = monitor.get_health_status()
print(f"Status: {health['status']}")
print(f"Calibration enabled: {health['calibration_enabled']}")
print(f"Metrics collected: {health['metrics_collected']}")
```

## Performance Optimization

### Caching

```python
# Enable caching for high-throughput scenarios
config = {
    "performance": {
        "cache_enabled": True,
        "cache_ttl_seconds": 300,
        "max_cache_size": 10000
    }
}

monitor = TrustMonitor(config=config)
```

### Batch Processing

```python
# Process multiple evaluations efficiently
evaluations = [
    (0.1, 0.9, 0.05),
    (0.3, 0.7, 0.2),
    (0.5, 0.5, 0.4)
]

results = []
for epistemic_risk, alignment_score, confidence_band in evaluations:
    gate = monitor.evaluate(epistemic_risk, alignment_score, confidence_band)
    results.append(gate.to_dict())
```

## Security Considerations

### Input Validation

- All inputs validated with epsilon tolerance (1e-6)
- Type checking enforced at dataclass level
- Configuration validation with required key checking

### PII Protection

```python
# Automatic PII validation if enabled
config = {
    "security": {
        "pii_validation": True,
        "input_sanitization": True,
        "audit_retention_days": 90
    }
}
```

### Break-Glass Procedures

```python
from resontinex.core.composite_trust_gate import break_glass_override

# Emergency override for critical situations
override = break_glass_override(
    gate=gate,
    override_reason="Critical production incident #12345",
    operator_id="ops.admin.john.doe"
)

# Automatically logged to audit trail
print(f"Override recorded: {override['event']}")
```

## Deployment Guide

### 1. Environment Setup

```bash
# Production deployment
export RESONTINEX_ENV=production
export TRUST_GATE_CONFIG=./configs/fusion/trust_gate.yaml

# Development
export RESONTINEX_ENV=development  
export TRUST_GATE_CONFIG=./configs/fusion/trust_gate_development.yaml
```

### 2. Configuration Validation

```python
# Validate configuration before deployment
from resontinex.core.composite_trust_gate import validate_config_and_normalize
import yaml

with open('./configs/fusion/trust_gate.yaml', 'r') as f:
    config = yaml.safe_load(f)

try:
    validated_config = validate_config_and_normalize(config)
    print("Configuration validation passed")
except Exception as e:
    print(f"Configuration validation failed: {e}")
    exit(1)
```

### 3. Health Checks

```python
# Production health check endpoint
@app.route('/health/trust-gate')
def trust_gate_health():
    monitor = TrustMonitor()
    health = monitor.get_health_status()
    
    return {
        "status": health["status"],
        "timestamp": health["timestamp"],
        "calibration_enabled": health["calibration_enabled"]
    }
```

## Troubleshooting

### Common Issues

#### 1. Configuration Errors
```
ValueError: Missing required config keys: ['risk_tiers']
```
**Solution**: Ensure all required configuration sections are present.

#### 2. Invalid Input Values
```
ValueError: epistemic_risk must be within [0,1], got 1.5
```
**Solution**: Validate inputs are within [0,1] range before creating CompositeTrustGate.

#### 3. Calibration Failures
```
WARNING: sklearn not available - calibration disabled
```
**Solution**: Install scikit-learn or disable calibration in config.

#### 4. Memory Issues with Large Batches
**Solution**: Enable batch processing or reduce concurrent evaluations.

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enables detailed logging of:
# - Configuration loading
# - Score calculations  
# - Tier determinations
# - Calibration operations
```

### Performance Profiling

```python
import cProfile

def profile_trust_evaluation():
    monitor = TrustMonitor()
    for i in range(1000):
        gate = monitor.evaluate(0.2, 0.8, 0.1)

cProfile.run('profile_trust_evaluation()')
```

## Migration from CompositeSignal

### Step 1: Update Imports

```python
# Old
from resontinex.core.composite_signal import CompositeSignal

# New  
from resontinex.core.composite_trust_gate import CompositeTrustGate
```

### Step 2: Update Configuration

```python
# Old configuration format is compatible
# No changes needed for basic usage

# New features available:
config.update({
    "tier_fallbacks": {...},
    "calibration": {...},
    "observability": {...}
})
```

### Step 3: Enhanced API Usage

```python
# Old API still works
gate = CompositeTrustGate(0.2, 0.8, 0.1)
trust_score = gate.trust_score

# New enhanced features
explanation = gate.explain_score()
action = gate.downstream_action()
raw_score = gate.raw_trust_score  # Full precision
```

## Next Steps

1. **Advanced Calibration**: Implement domain-specific calibration models
2. **A/B Testing**: Framework for testing different configurations
3. **Federated Learning**: Multi-node calibration model training
4. **Quantum-Safe Algorithms**: Future-proof cryptographic validation
5. **Automated Tuning**: ML-driven configuration optimization

## Support

For implementation assistance or advanced integration scenarios, consult:
- RESONTINEX Architecture Documentation
- Performance Tuning Guide
- Security Best Practices Guide