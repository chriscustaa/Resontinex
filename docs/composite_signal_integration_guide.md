# CompositeSignal Integration Guide

## Overview

The CompositeSignal module enhances RESONTINEX with sophisticated trust scoring, risk assessment, and explainable decision-making capabilities. It integrates seamlessly with existing modules while providing advanced multi-dimensional signal analysis.

## Core Architecture

### Signal Components
- **epistemic_risk**: Uncertainty in knowledge/model confidence (0.0-1.0)
- **alignment_score**: Degree of alignment with expected behavior (0.0-1.0) 
- **confidence_band**: Statistical confidence interval width (0.0-1.0)

### Integration Points

#### 1. TrustManager Enhancement
```python
from resontinex.core.composite_signal import enhance_trust_manager_scoring

# Enhanced scoring with multi-dimensional assessment
result = enhance_trust_manager_scoring(
    alignment_score=0.8,
    inflation_delta=0.1,
    epistemic_risk=0.2,
    confidence_band=0.15
)

# Result includes:
# - trust_signal: CompositeSignal data
# - downstream_action: Decision routing info
# - explanation: Explainability trace
# - legacy_score: Backward compatibility
```

#### 2. ScenarioManager Integration
```python
from resontinex.core.composite_signal import create_from_scenario_metrics
from resontinex.scenario_manager import ScenarioManager

manager = ScenarioManager(config)
scenarios = manager.get_scenarios()

for scenario_id, scenario in scenarios.items():
    if scenario.metrics:
        # Convert to CompositeSignal for enhanced trust scoring
        signal = create_from_scenario_metrics(scenario.metrics)
        
        # Use signal for advanced filtering
        if signal.trust_score >= 0.8 and signal.risk_tier == "pass":
            # High-trust scenario eligible for automatic execution
            pass
```

#### 3. EntropyAuditor Enhancement
```python
from resontinex.core.composite_signal import CompositeSignal

signal = CompositeSignal(
    epistemic_risk=entropy_metrics.epistemic_uncertainty,
    alignment_score=trust_metrics.alignment_measure,
    confidence_band=statistical_metrics.confidence_width
)

# Enhanced entropy detection
if signal.entropy_flag:
    # Exceeds RESONTINEX entropy threshold (0.72)
    trigger_entropy_audit(signal.explain_score())
```

## Configuration

### Default Configuration
```yaml
# configs/fusion/composite_signal.yaml
weights:
  epistemic_risk: 0.4
  alignment_score: 0.4
  confidence_band: 0.2

risk_tiers:
  block: 0.30
  review: 0.55
  monitor: 0.75
  pass: 1.01

# RESONTINEX integration
trust_floor: 0.60  # Matches runtime.trust_floor
entropy_threshold: 0.72  # Matches EntropyAuditor.threshold

integration:
  scenario_manager_compat: true
  trust_manager_enhanced: true
```

### Custom Strategy Implementation
```python
from resontinex.core.composite_signal import AggregationStrategy, CompositeSignal

class CustomStrategy(AggregationStrategy):
    """Custom aggregation strategy for domain-specific requirements."""
    
    def __call__(self, signal: CompositeSignal) -> float:
        # Implement custom scoring logic
        return custom_score
        
# Use custom strategy
signal = CompositeSignal(
    epistemic_risk=0.2,
    alignment_score=0.8,
    confidence_band=0.1,
    _agg_strategy=CustomStrategy()
)
```

## Production Usage Patterns

### 1. High-Throughput Scenario Filtering
```python
from resontinex.core.composite_signal import CompositeSignal

def filter_high_trust_scenarios(scenarios):
    """Filter scenarios using CompositeSignal trust scoring."""
    high_trust = {}
    
    for scenario_id, scenario in scenarios.items():
        if scenario.metrics:
            signal = create_from_scenario_metrics(scenario.metrics)
            
            if (signal.trust_score >= 0.8 and 
                signal.trust_floor_met and 
                not signal.entropy_flag):
                high_trust[scenario_id] = scenario
                
    return high_trust
```

### 2. Risk-Based Decision Routing
```python
def route_based_on_risk(input_signal_data):
    """Route decisions based on CompositeSignal risk assessment."""
    signal = CompositeSignal(**input_signal_data)
    action = signal.downstream_action()
    
    routing_map = {
        "block": "security_review_queue",
        "review": "human_review_queue", 
        "monitor": "automated_processing_with_logging",
        "pass": "fast_track_processing"
    }
    
    return {
        "route": routing_map[action["tier"]],
        "explanation": signal.explain_score(),
        "voting_weight": action["voting_weight"]
    }
```

### 3. Explainable Audit Trails
```python
def generate_audit_entry(signal, context):
    """Generate comprehensive audit entry with explainability."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "context": context,
        "signal_data": signal.to_dict(),
        "decision": signal.downstream_action(),
        "explanation": signal.explain_score(),
        "resontinex_integration": {
            "trust_floor_met": signal.trust_floor_met,
            "entropy_flag": signal.entropy_flag,
            "risk_tier": signal.risk_tier
        }
    }
```

## Performance Characteristics

### Computational Complexity
- **Signal Creation**: O(1) - Constant time validation and initialization
- **Trust Score Calculation**: O(1) - Single logit/logistic transformation
- **Risk Tier Classification**: O(1) - Simple threshold comparison
- **Explainability Generation**: O(1) - Pre-computed explanations

### Memory Usage
- **CompositeSignal Instance**: ~200 bytes (frozen dataclass with slots)
- **Configuration Cache**: ~2KB per config file
- **Explanation Data**: ~1KB per explanation

### Integration Overhead
- **ScenarioManager Enhancement**: +5-10ms per scenario evaluation
- **TrustManager Enhancement**: +2-5ms per trust calculation
- **EntropyAuditor Integration**: +1-3ms per entropy check

## Testing Strategy

### Unit Tests
```bash
# Run comprehensive test suite
python -m pytest tests/core/test_composite_signal.py -v

# Coverage report
python -m pytest tests/core/test_composite_signal.py --cov=resontinex.core.composite_signal
```

### Integration Tests
```python
def test_end_to_end_integration():
    """Test full RESONTINEX integration pipeline."""
    # Test with ScenarioManager
    manager = ScenarioManager(test_config)
    scenarios = manager.filter_by_capability("reasoning")
    
    # Enhance with CompositeSignal
    for scenario_id, scenario in scenarios.items():
        signal = create_from_scenario_metrics(scenario.metrics)
        assert signal.trust_score >= 0.0
        assert signal.risk_tier in ["block", "review", "monitor", "pass"]
```

## Best Practices

### 1. Configuration Management
- Use environment-specific configs for different deployment stages
- Validate configurations on startup to prevent runtime errors
- Cache configurations for performance in high-throughput scenarios

### 2. Error Handling
- Always validate signal inputs before processing
- Use default configurations as fallback for missing config files
- Log configuration loading failures for debugging

### 3. Performance Optimization
- Reuse CompositeSignal instances when possible (they're frozen)
- Cache expensive calculations like explanation generation
- Use batch processing for multiple signals

### 4. Monitoring and Observability
- Track trust score distributions over time
- Monitor risk tier frequencies for system health
- Alert on unexpected entropy flag patterns

## Migration Guide

### Existing TrustManager Users
```python
# Before: Basic trust scoring
trust_score = alignment_score + inflation_delta

# After: Enhanced multi-dimensional scoring
enhanced_result = enhance_trust_manager_scoring(
    alignment_score=alignment_score,
    inflation_delta=inflation_delta,
    epistemic_risk=get_epistemic_risk(),
    confidence_band=get_confidence_band()
)
trust_score = enhanced_result["trust_signal"]["trust_score"]
```

### Existing ScenarioManager Users
```python
# Before: Basic performance filtering
filtered = manager.filter_by_weight_threshold(0.7)

# After: Trust-enhanced filtering
def enhanced_filter(scenarios):
    return {
        sid: scenario for sid, scenario in scenarios.items()
        if create_from_scenario_metrics(scenario.metrics).trust_score >= 0.7
    }
```

## Future Enhancements

### Planned Features
- **Machine Learning Integration**: Support for learned aggregation strategies
- **Temporal Dynamics**: Time-series trust scoring for drift detection
- **Multi-Agent Consensus**: Distributed trust scoring across agent networks
- **Quantum-Safe Algorithms**: Post-quantum cryptographic verification

### Extension Points
- Custom aggregation strategies via `AggregationStrategy` protocol
- Pluggable risk tier definitions via configuration
- External trust score providers via adapter pattern
- Custom explainability formatters for different audit requirements