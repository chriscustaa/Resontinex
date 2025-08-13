# Energy Governance System - Technical Implementation Guide

## Overview

The RESONTINEX Energy Governance System provides comprehensive cost control, budget enforcement, and resource optimization for AI workflow orchestration. This system implements multi-layered governance mechanisms with automatic safeguards and adaptive thresholds.

## Core Architecture

### Energy Budget Framework

```python
# Energy budget configuration structure
ENERGY_BUDGET_CONFIG = {
    "energy_budget_per_signal": "auto-calculated",  # Dynamic calculation
    "max_collapse_attempts": 3,                     # Hard limit per signal
    "adaptive_backoff_threshold_ms": 200,          # Latency trigger
    "budget_approval_threshold": 10000,            # Units requiring approval
    "hard_limit_percentage": 95,                   # Automatic termination
    "auto_review_threshold": 85,                   # Enhanced validation trigger
    "emergency_brake_enabled": True                # Circuit breaker activation
}
```

### Energy Multiplier System

The system applies dynamic cost multipliers based on operational context:

```python
def calculate_energy_cost(base_cost: float, context: dict) -> float:
    """
    Calculate final energy cost with context-aware multipliers.
    
    Args:
        base_cost: Base computational cost in energy units
        context: Operational context dictionary
        
    Returns:
        Final cost with applied multipliers
    """
    multiplier = 1.0
    
    # Complexity multiplier (1.5x for high complexity)
    if context.get('complexity_level') == 'high':
        multiplier *= 1.5
    
    # Trust multiplier (1.3x for low trust scenarios)
    if context.get('trust_score', 1.0) < 0.6:
        multiplier *= 1.3
    
    # Entropy multiplier (1.4x for high entropy)
    if context.get('entropy_score', 0.0) > 0.7:
        multiplier *= 1.4
    
    # Emergency mode discount (0.6x for reduced quality)
    if context.get('emergency_mode', False):
        multiplier *= 0.6
    
    return base_cost * multiplier
```

### Budget Enforcement Implementation

```python
from resontinex.fusion_resilience import get_fusion_loader
from fusion_ops.budget_tripwire import get_budget_tripwire

class EnergyGovernanceController:
    """Production-grade energy governance with automatic enforcement."""
    
    def __init__(self, config: dict):
        self.config = config
        self.tripwire = get_budget_tripwire()
        self.fusion_loader = get_fusion_loader()
        
    def validate_energy_budget(self, requested_units: int, context: dict) -> dict:
        """
        Validate energy budget request against governance policies.
        
        Returns:
            Validation result with approval status and constraints
        """
        # Calculate actual cost with multipliers
        actual_cost = calculate_energy_cost(requested_units, context)
        
        # Check hard limits
        if actual_cost > self.config['budget_approval_threshold']:
            return {
                'approved': False,
                'reason': 'exceeds_approval_threshold',
                'actual_cost': actual_cost,
                'requires_manual_approval': True
            }
        
        # Check budget tripwire status
        tripwire_status = self.tripwire.get_status()
        if tripwire_status['downgrade_active']:
            # Apply downgraded parameters
            downgraded_cost = actual_cost * 0.5  # 50% cost reduction
            return {
                'approved': True,
                'cost_adjustment': 'downgraded',
                'original_cost': actual_cost,
                'final_cost': downgraded_cost,
                'reason': 'budget_tripwire_active'
            }
        
        return {
            'approved': True,
            'final_cost': actual_cost,
            'multipliers_applied': self._get_applied_multipliers(context)
        }
    
    def _get_applied_multipliers(self, context: dict) -> dict:
        """Get detailed breakdown of applied cost multipliers."""
        multipliers = {}
        
        if context.get('complexity_level') == 'high':
            multipliers['complexity'] = 1.5
        if context.get('trust_score', 1.0) < 0.6:
            multipliers['trust'] = 1.3
        if context.get('entropy_score', 0.0) > 0.7:
            multipliers['entropy'] = 1.4
        if context.get('emergency_mode', False):
            multipliers['emergency_discount'] = 0.6
            
        return multipliers
```

## Budget Tripwire System

### Automatic Downgrade Mechanism

The budget tripwire system automatically downgrades system parameters when consecutive budget breaches occur:

```python
from fusion_ops.budget_tripwire import BudgetTripwire

# Initialize tripwire with production settings
tripwire = BudgetTripwire(
    state_file=".fusion_tripwire_state.json",
    breach_threshold=12.0,    # 12% token delta threshold
    consecutive_limit=3       # Trigger after 3 consecutive breaches
)

# Check for budget breach in your application
def process_fusion_result(token_delta: float, scenario_context: dict):
    """Process fusion results and check for budget violations."""
    
    # Check tripwire status
    tripwire_result = tripwire.check_budget_breach(
        token_delta=token_delta,
        context={
            'scenario_type': scenario_context.get('type'),
            'overlay_version': scenario_context.get('overlay_version'),
            'timestamp': time.time()
        }
    )
    
    if tripwire_result['downgrade_triggered']:
        print(f"⚠️  Budget tripwire activated: {tripwire_result['downgrade_reason']}")
        
        # Apply downgraded overlay parameters
        base_params = load_overlay_parameters()
        downgraded_params = tripwire.get_overlay_params(base_params)
        
        # Update system configuration
        update_fusion_overlay(downgraded_params)
        
        # Emit telemetry
        emit_metrics({
            'fusion.budget.tripwire_activated': 1,
            'fusion.budget.consecutive_breaches': tripwire_result['consecutive_breaches'],
            'fusion.budget.token_delta': token_delta
        })
    
    return tripwire_result
```

### Recovery and Reset Procedures

```python
def reset_budget_tripwire():
    """Reset budget tripwire - use with caution in production."""
    tripwire = get_budget_tripwire()
    
    reset_result = tripwire.reset_downgrade()
    
    if reset_result['reset']:
        print(f"✅ Budget tripwire reset successfully")
        print(f"   Previous state: {'active' if reset_result['was_active'] else 'inactive'}")
        
        # Reload original overlay parameters
        fusion_loader = get_fusion_loader()
        overlay_config, health = fusion_loader.load_fusion_overlay()
        
        # Emit reset event for monitoring
        emit_metrics({
            'fusion.budget.tripwire_reset': 1,
            'fusion.budget.manual_intervention': 1
        })
        
        return True
    
    return False
```

## Energy Recovery Mechanisms

### Insight Reuse and Caching

```python
class EnergyOptimizationManager:
    """Manages energy optimization through caching and reuse strategies."""
    
    def __init__(self):
        self.insight_cache = {}
        self.performance_cache = {}
        
    def get_cached_insight(self, scenario_hash: str) -> Optional[dict]:
        """
        Retrieve cached insights to avoid recomputation.
        
        Returns:
            Cached insight data or None if not found
        """
        if scenario_hash in self.insight_cache:
            cache_entry = self.insight_cache[scenario_hash]
            
            # Check cache freshness (24 hour TTL)
            if time.time() - cache_entry['timestamp'] < 86400:
                return {
                    'data': cache_entry['data'],
                    'energy_saved': cache_entry['original_cost'],
                    'cache_hit': True
                }
        
        return None
    
    def store_insight(self, scenario_hash: str, insight_data: dict, energy_cost: float):
        """Store computed insight for future reuse."""
        self.insight_cache[scenario_hash] = {
            'data': insight_data,
            'timestamp': time.time(),
            'original_cost': energy_cost,
            'reuse_count': 0
        }
    
    def early_termination_check(self, partial_results: dict, energy_consumed: float) -> dict:
        """
        Check if early termination is beneficial based on partial results.
        
        Returns:
            Termination decision with reasoning
        """
        # Quality threshold for early termination
        quality_threshold = 0.85
        energy_threshold = 0.7  # 70% of budget
        
        current_quality = partial_results.get('quality_score', 0.0)
        budget_consumed = energy_consumed / self.config.get('max_energy_budget', 1000)
        
        if current_quality >= quality_threshold and budget_consumed < energy_threshold:
            return {
                'terminate': True,
                'reason': 'quality_threshold_met',
                'energy_saved': (1.0 - budget_consumed) * 100,
                'quality_achieved': current_quality
            }
        
        return {'terminate': False}
```

## Monitoring and Observability

### Energy Audit Logging

```python
import logging
import json
from datetime import datetime, timezone

class EnergyAuditLogger:
    """Comprehensive energy usage audit logging."""
    
    def __init__(self):
        self.logger = logging.getLogger("energy_audit")
        self._configure_structured_logging()
    
    def _configure_structured_logging(self):
        """Configure structured JSON logging for energy audit trails."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "energy_audit", "event": %(message)s}'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_energy_decision(self, decision_data: dict):
        """Log energy governance decision with full context."""
        audit_event = {
            "event_type": "energy_decision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_id": decision_data.get('decision_id'),
            "requested_units": decision_data.get('requested_units'),
            "approved_units": decision_data.get('approved_units'),
            "multipliers": decision_data.get('multipliers', {}),
            "governance_flags": {
                "tripwire_active": decision_data.get('tripwire_active', False),
                "manual_approval_required": decision_data.get('manual_approval_required', False),
                "emergency_mode": decision_data.get('emergency_mode', False)
            },
            "context": decision_data.get('context', {}),
            "energy_efficiency": decision_data.get('energy_efficiency', 0.0)
        }
        
        self.logger.info(json.dumps(audit_event))
    
    def log_budget_breach(self, breach_data: dict):
        """Log budget threshold breach events."""
        breach_event = {
            "event_type": "budget_breach",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold_breached": breach_data.get('threshold'),
            "actual_consumption": breach_data.get('consumption'),
            "breach_severity": breach_data.get('severity'),
            "consecutive_breaches": breach_data.get('consecutive_count'),
            "automatic_action": breach_data.get('automatic_action'),
            "scenario_context": breach_data.get('scenario_context', {})
        }
        
        self.logger.warning(json.dumps(breach_event))
```

## Configuration Templates

### Production Energy Governance Configuration

```yaml
# config/energy_governance.yaml
energy_governance:
  # Budget thresholds
  budget_limits:
    hard_limit_percentage: 95
    auto_review_threshold: 85
    approval_threshold: 10000
    emergency_threshold: 99
  
  # Cost multipliers
  multipliers:
    high_complexity: 1.5
    low_trust: 1.3
    high_entropy: 1.4
    emergency_discount: 0.6
  
  # Tripwire configuration
  budget_tripwire:
    enabled: true
    breach_threshold: 12.0
    consecutive_limit: 3
    state_persistence: true
    auto_recovery: false
  
  # Recovery mechanisms
  optimization:
    insight_caching: true
    early_termination: true
    adaptive_backoff: true
    cache_ttl_hours: 24
  
  # Audit and monitoring
  monitoring:
    audit_logging: true
    structured_events: true
    metrics_emission: true
    alert_thresholds:
      budget_utilization: 80
      consecutive_breaches: 2
      cost_efficiency: 0.7
```

## Integration Examples

### Flask Application Integration

```python
from flask import Flask, request, jsonify
from resontinex.fusion_resilience import load_fusion_configuration
from fusion_ops.budget_tripwire import get_budget_tripwire

app = Flask(__name__)

@app.route('/fusion/execute', methods=['POST'])
def execute_fusion():
    """Execute fusion with energy governance."""
    
    request_data = request.get_json()
    scenario_context = {
        'type': request_data.get('scenario_type'),
        'complexity_level': request_data.get('complexity', 'medium'),
        'trust_score': request_data.get('trust_score', 1.0)
    }
    
    # Energy budget validation
    governance_controller = EnergyGovernanceController(ENERGY_BUDGET_CONFIG)
    budget_validation = governance_controller.validate_energy_budget(
        requested_units=request_data.get('estimated_cost', 100),
        context=scenario_context
    )
    
    if not budget_validation['approved']:
        return jsonify({
            'status': 'rejected',
            'reason': budget_validation['reason'],
            'requires_approval': budget_validation.get('requires_manual_approval', False)
        }), 403
    
    # Execute fusion with approved budget
    try:
        # Your fusion execution logic here
        result = execute_fusion_logic(request_data, budget_validation['final_cost'])
        
        # Check for budget tripwire after execution
        tripwire = get_budget_tripwire()
        tripwire_result = tripwire.check_budget_breach(
            token_delta=result.get('token_delta_pct', 0.0),
            context=scenario_context
        )
        
        return jsonify({
            'status': 'success',
            'result': result,
            'energy_governance': {
                'budget_approved': budget_validation['final_cost'],
                'tripwire_status': tripwire_result,
                'multipliers': budget_validation.get('multipliers_applied', {})
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'energy_governance': 'execution_failed'
        }), 500

@app.route('/fusion/budget/status', methods=['GET'])
def get_budget_status():
    """Get current budget and tripwire status."""
    
    tripwire = get_budget_tripwire()
    status = tripwire.get_status()
    
    fusion_loader = get_fusion_loader()
    health = fusion_loader.get_health_status()
    
    return jsonify({
        'budget_tripwire': status,
        'system_health': health,
        'timestamp': time.time()
    })
```

## Best Practices

### 1. Energy-Aware Development

```python
def energy_aware_function(data: dict) -> dict:
    """Template for energy-aware function development."""
    
    # Estimate energy cost upfront
    estimated_cost = estimate_computational_cost(data)
    
    # Validate against budget
    governance_controller = EnergyGovernanceController(ENERGY_BUDGET_CONFIG)
    validation = governance_controller.validate_energy_budget(
        estimated_cost, 
        extract_context(data)
    )
    
    if not validation['approved']:
        raise EnergyBudgetExceeded(validation['reason'])
    
    # Track actual energy consumption
    start_time = time.time()
    try:
        result = expensive_computation(data)
        actual_cost = time.time() - start_time
        
        # Update energy efficiency metrics
        efficiency = estimated_cost / actual_cost if actual_cost > 0 else 1.0
        emit_energy_metrics(efficiency, estimated_cost, actual_cost)
        
        return result
        
    except Exception as e:
        # Energy cleanup on error
        emit_energy_metrics(0.0, estimated_cost, time.time() - start_time)
        raise
```

### 2. Testing Energy Governance

```python
import pytest
from fusion_ops.budget_tripwire import BudgetTripwire
import tempfile
import os

class TestEnergyGovernance:
    """Test suite for energy governance functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.tripwire = BudgetTripwire(
            state_file=self.temp_file.name,
            breach_threshold=10.0,
            consecutive_limit=2
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        os.unlink(self.temp_file.name)
    
    def test_budget_breach_detection(self):
        """Test budget breach detection and consecutive counting."""
        
        # First breach
        result1 = self.tripwire.check_budget_breach(15.0)
        assert result1['is_breach'] is True
        assert result1['consecutive_breaches'] == 1
        assert result1['downgrade_triggered'] is False
        
        # Second breach - should trigger downgrade
        result2 = self.tripwire.check_budget_breach(12.0)
        assert result2['downgrade_triggered'] is True
        assert result2['downgrade_active'] is True
    
    def test_energy_multiplier_calculation(self):
        """Test energy cost multiplier calculations."""
        
        base_cost = 100.0
        
        # High complexity context
        high_complexity_cost = calculate_energy_cost(
            base_cost, 
            {'complexity_level': 'high'}
        )
        assert high_complexity_cost == 150.0  # 1.5x multiplier
        
        # Multiple multipliers
        complex_context_cost = calculate_energy_cost(
            base_cost,
            {
                'complexity_level': 'high',
                'trust_score': 0.5,
                'entropy_score': 0.8
            }
        )
        expected = base_cost * 1.5 * 1.3 * 1.4  # All multipliers
        assert complex_context_cost == expected
```

## Security Considerations

### 1. Energy Budget Authentication

```python
import hmac
import hashlib
from datetime import datetime, timezone

class EnergyBudgetAuthenticator:
    """Secure authentication for energy budget overrides."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
    
    def generate_budget_token(self, budget_request: dict) -> str:
        """Generate authenticated token for budget requests."""
        
        # Create canonical representation
        canonical_request = json.dumps(budget_request, sort_keys=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Generate HMAC signature
        message = f"{canonical_request}|{timestamp}".encode()
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        
        return f"{timestamp}:{signature}"
    
    def validate_budget_token(self, token: str, budget_request: dict, 
                            max_age_seconds: int = 300) -> bool:
        """Validate budget request token."""
        
        try:
            timestamp_str, signature = token.split(':', 1)
            request_time = datetime.fromisoformat(timestamp_str)
            
            # Check token age
            age = (datetime.now(timezone.utc) - request_time).total_seconds()
            if age > max_age_seconds:
                return False
            
            # Verify signature
            canonical_request = json.dumps(budget_request, sort_keys=True)
            message = f"{canonical_request}|{timestamp_str}".encode()
            expected_signature = hmac.new(
                self.secret_key, message, hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, KeyError):
            return False
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Budget Tripwire Stuck in Downgrade Mode

**Symptoms:** System remains in downgraded state despite resolved issues

**Solution:**
```bash
# Check tripwire status
python -c "
from fusion_ops.budget_tripwire import get_budget_tripwire
print(get_budget_tripwire().get_status())
"

# Reset tripwire if needed
python -c "
from fusion_ops.budget_tripwire import get_budget_tripwire
result = get_budget_tripwire().reset_downgrade()
print('Reset:', result)
"
```

#### 2. Energy Cost Multipliers Not Applied

**Check Configuration:**
```python
from resontinex.fusion_resilience import get_fusion_loader

loader = get_fusion_loader()
config = loader.get_effective_config()
print("Current overlay config:", config)

# Verify multiplier logic
test_cost = calculate_energy_cost(100, {
    'complexity_level': 'high',
    'trust_score': 0.5
})
print(f"Test cost with multipliers: {test_cost}")
```

#### 3. Budget Approval Threshold Issues

**Debug Budget Validation:**
```python
governance = EnergyGovernanceController(ENERGY_BUDGET_CONFIG)
result = governance.validate_energy_budget(
    requested_units=15000,
    context={'scenario_type': 'test'}
)
print("Validation result:", result)
```

This energy governance system provides comprehensive cost control with automatic safeguards, ensuring efficient resource utilization while maintaining system reliability and performance.