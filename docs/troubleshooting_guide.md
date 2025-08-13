# RESONTINEX Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures for the RESONTINEX AI workflow orchestration system, based on real production error patterns and recovery strategies.

## Table of Contents

1. [System Health Diagnostics](#system-health-diagnostics)
2. [Runtime Routing Issues](#runtime-routing-issues)
3. [Circuit Breaker Problems](#circuit-breaker-problems)
4. [Budget and Energy Management](#budget-and-energy-management)
5. [Configuration Issues](#configuration-issues)
6. [Drift Detection Problems](#drift-detection-problems)
7. [Performance Degradation](#performance-degradation)
8. [Recovery Procedures](#recovery-procedures)
9. [Monitoring and Alerting](#monitoring-and-alerting)

---

## System Health Diagnostics

### Quick Health Check

```python
#!/usr/bin/env python3
"""
System Health Diagnostic Script
Run this script to get comprehensive system status.
"""

from scripts.runtime_router import RuntimeRouter
from scripts.circuit_breaker import ProductionSafetyManager
from fusion_ops.budget_tripwire import get_budget_tripwire
import json
import traceback
from pathlib import Path

def run_system_diagnostics():
    """Run comprehensive system diagnostics."""
    diagnostics = {
        'timestamp': time.time(),
        'overall_status': 'unknown',
        'component_status': {},
        'issues_detected': [],
        'recommendations': []
    }
    
    # Test Runtime Router
    try:
        router = RuntimeRouter("./configs/fusion")
        routing_stats = router.get_routing_stats()
        
        diagnostics['component_status']['runtime_router'] = {
            'status': 'healthy',
            'overlays_available': routing_stats['overlays_available'],
            'overlay_names': routing_stats['overlay_names']
        }
        
        if routing_stats['overlays_available'] == 0:
            diagnostics['issues_detected'].append({
                'component': 'runtime_router',
                'issue': 'No overlays loaded',
                'severity': 'critical',
                'solution': 'Check ./configs/fusion/micro_overlays directory'
            })
            
    except Exception as e:
        diagnostics['component_status']['runtime_router'] = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        diagnostics['issues_detected'].append({
            'component': 'runtime_router',
            'issue': f'Router initialization failed: {str(e)}',
            'severity': 'critical',
            'solution': 'Check configuration directory and file permissions'
        })
    
    # Test Safety Manager
    try:
        safety_manager = ProductionSafetyManager("./configs/fusion")
        health = safety_manager.check_system_health()
        
        diagnostics['component_status']['safety_manager'] = {
            'status': health['overall_health'],
            'slo_violations': len(health['slo_violations']),
            'circuit_breaker_count': len(health['circuit_breaker_states'])
        }
        
        if health['overall_health'] != 'healthy':
            for violation in health['slo_violations']:
                diagnostics['issues_detected'].append({
                    'component': 'safety_manager',
                    'issue': f"SLO violation: {violation['slo_name']}",
                    'severity': violation.get('severity', 'warning'),
                    'solution': 'Review system performance and capacity'
                })
                
    except Exception as e:
        diagnostics['component_status']['safety_manager'] = {
            'status': 'error',
            'error': str(e)
        }
        diagnostics['issues_detected'].append({
            'component': 'safety_manager',
            'issue': f'Safety manager initialization failed: {str(e)}',
            'severity': 'critical',
            'solution': 'Check SLO configuration files'
        })
    
    # Test Budget Tripwire
    try:
        tripwire = get_budget_tripwire()
        status = tripwire.get_status()
        
        diagnostics['component_status']['budget_tripwire'] = {
            'status': 'degraded' if status['downgrade_active'] else 'healthy',
            'downgrade_active': status['downgrade_active'],
            'consecutive_breaches': status['consecutive_breaches']
        }
        
        if status['downgrade_active']:
            diagnostics['issues_detected'].append({
                'component': 'budget_tripwire',
                'issue': 'Budget downgrade active',
                'severity': 'warning',
                'solution': 'Review token usage patterns and consider resetting tripwire'
            })
            
    except Exception as e:
        diagnostics['component_status']['budget_tripwire'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Determine overall status
    if not diagnostics['issues_detected']:
        diagnostics['overall_status'] = 'healthy'
    elif any(issue['severity'] == 'critical' for issue in diagnostics['issues_detected']):
        diagnostics['overall_status'] = 'critical'
    else:
        diagnostics['overall_status'] = 'degraded'
    
    return diagnostics

if __name__ == "__main__":
    import time
    
    print("🔍 RESONTINEX System Health Diagnostics")
    print("=" * 50)
    
    diagnostics = run_system_diagnostics()
    
    print(f"Overall Status: {diagnostics['overall_status'].upper()}")
    print(f"Check Time: {time.ctime(diagnostics['timestamp'])}")
    print()
    
    print("Component Status:")
    for component, status in diagnostics['component_status'].items():
        status_icon = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'degraded' else "❌"
        print(f"  {status_icon} {component}: {status['status']}")
    
    if diagnostics['issues_detected']:
        print("\nIssues Detected:")
        for issue in diagnostics['issues_detected']:
            severity_icon = "🚨" if issue['severity'] == 'critical' else "⚠️"
            print(f"  {severity_icon} {issue['component']}: {issue['issue']}")
            print(f"      Solution: {issue['solution']}")
    
    print("\n" + "=" * 50)
    print("Diagnostics completed")
```

### Configuration Validation

```python
def validate_system_configuration():
    """Validate system configuration files."""
    validation_results = {}
    config_dir = Path("./configs/fusion")
    
    # Required configuration files
    required_files = [
        "overlay_params.yaml",
        "scenario_profiles.yaml", 
        "slo.yaml"
    ]
    
    for config_file in required_files:
        file_path = config_dir / config_file
        validation_results[config_file] = {
            'exists': file_path.exists(),
            'readable': file_path.exists() and os.access(file_path, os.R_OK),
            'size': file_path.stat().st_size if file_path.exists() else 0
        }
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if config_file.endswith('.yaml'):
                        yaml.safe_load(f)
                    elif config_file.endswith('.json'):
                        json.load(f)
                validation_results[config_file]['valid'] = True
            except Exception as e:
                validation_results[config_file]['valid'] = False
                validation_results[config_file]['error'] = str(e)
    
    # Check micro-overlays directory
    overlays_dir = config_dir / "micro_overlays"
    overlay_files = list(overlays_dir.glob("*.txt")) if overlays_dir.exists() else []
    
    validation_results['micro_overlays'] = {
        'directory_exists': overlays_dir.exists(),
        'overlay_count': len(overlay_files),
        'overlay_files': [f.name for f in overlay_files]
    }
    
    return validation_results
```

---

## Runtime Routing Issues

### Problem: No Overlays Loaded

**Symptoms:**
- `RuntimeRouter` reports 0 available overlays
- Routing decisions default to 'none'
- Enhanced prompts identical to base prompts

**Diagnosis:**
```python
from scripts.runtime_router import RuntimeRouter

router = RuntimeRouter("./configs/fusion")
stats = router.get_routing_stats()

print(f"Overlays available: {stats['overlays_available']}")
print(f"Overlay names: {stats['overlay_names']}")

if stats['overlays_available'] == 0:
    print("🚨 No overlays loaded - checking directory...")
    
    overlay_dir = Path("./configs/fusion/micro_overlays")
    if not overlay_dir.exists():
        print(f"❌ Overlay directory missing: {overlay_dir}")
    else:
        overlay_files = list(overlay_dir.glob("*.txt"))
        print(f"📁 Found {len(overlay_files)} overlay files")
        for file in overlay_files:
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
```

**Solutions:**

1. **Create Missing Directory:**
```bash
mkdir -p ./configs/fusion/micro_overlays
```

2. **Check File Permissions:**
```bash
ls -la ./configs/fusion/micro_overlays/
chmod 644 ./configs/fusion/micro_overlays/*.txt
```

3. **Validate Overlay Files:**
```python
def validate_overlay_files():
    """Validate individual overlay files."""
    overlay_dir = Path("./configs/fusion/micro_overlays")
    issues = []
    
    for overlay_file in overlay_dir.glob("*.txt"):
        try:
            with open(overlay_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) < 50:
                issues.append(f"{overlay_file.name}: Content too short")
            
            # Check for required sections
            required_sections = ["## Core Directive", "## Response Framework"]
            missing_sections = [
                section for section in required_sections 
                if section not in content
            ]
            
            if missing_sections:
                issues.append(f"{overlay_file.name}: Missing sections: {missing_sections}")
                
        except Exception as e:
            issues.append(f"{overlay_file.name}: Read error - {str(e)}")
    
    return issues
```

### Problem: Overlay Loading Errors

**Symptoms:**
- Runtime router initialization fails
- Parse errors in logs
- Partial overlay loading

**Diagnosis:**
```python
from scripts.runtime_router import RuntimeRouter, MicroOverlayParser

# Test individual overlay parsing
parser = MicroOverlayParser()
overlay_dir = Path("./configs/fusion/micro_overlays")

for overlay_file in overlay_dir.glob("*.txt"):
    try:
        overlay = parser.parse_overlay(str(overlay_file))
        print(f"✅ {overlay_file.name}: Loaded successfully")
        print(f"   Directives: {len(overlay.directives)}")
        print(f"   Patterns: {len(overlay.patterns)}")
    except Exception as e:
        print(f"❌ {overlay_file.name}: Parse error - {str(e)}")
```

**Solutions:**

1. **Fix UTF-8 Encoding Issues:**
```python
def fix_encoding_issues(overlay_file):
    """Fix common encoding issues in overlay files."""
    try:
        # Try reading with different encodings
        encodings = ['utf-8', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(overlay_file, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # Re-save with UTF-8 encoding
                with open(overlay_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"Fixed encoding for {overlay_file}")
                break
                
            except UnicodeDecodeError:
                continue
                
    except Exception as e:
        print(f"Could not fix encoding for {overlay_file}: {e}")
```

2. **Repair Malformed Overlay Files:**
```python
def repair_overlay_structure(overlay_file):
    """Repair common overlay structure issues."""
    with open(overlay_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ensure required sections exist
    required_sections = {
        "## Core Directive": "Default core directive content",
        "## Response Framework": "Default response framework", 
        "## Quality Gates": "- Ensure accuracy\n- Validate completeness"
    }
    
    for section_header, default_content in required_sections.items():
        if section_header not in content:
            content += f"\n\n{section_header}\n{default_content}\n"
    
    # Save repaired content
    with open(overlay_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Repaired overlay structure: {overlay_file}")
```

### Problem: Poor Routing Decisions

**Symptoms:**
- Consistently selects suboptimal overlays
- Low confidence scores in routing decisions
- Performance degradation

**Diagnosis:**
```python
def analyze_routing_performance():
    """Analyze routing decision quality."""
    router = RuntimeRouter("./configs/fusion")
    
    # Test scenarios with known optimal overlays
    test_scenarios = [
        {
            'id': 'financial_test',
            'category': 'financial_operations',
            'complexity': 0.8,
            'description': 'Customer refund with rollback requirements',
            'expected_overlay': 'rollback_first'
        },
        {
            'id': 'workflow_test', 
            'category': 'compliance_management',
            'complexity': 0.9,
            'description': 'Complex approval workflow with state transitions',
            'expected_overlay': 'state_model_first'
        }
    ]
    
    routing_accuracy = []
    
    for scenario in test_scenarios:
        decision = router.route_scenario(scenario)
        is_correct = decision.selected_overlay == scenario['expected_overlay']
        
        routing_accuracy.append({
            'scenario_id': scenario['id'],
            'expected': scenario['expected_overlay'],
            'actual': decision.selected_overlay,
            'confidence': decision.confidence,
            'correct': is_correct,
            'reasoning': decision.reasoning
        })
        
        print(f"Scenario {scenario['id']}:")
        print(f"  Expected: {scenario['expected_overlay']}")
        print(f"  Actual: {decision.selected_overlay} {'✅' if is_correct else '❌'}")
        print(f"  Confidence: {decision.confidence:.2f}")
    
    accuracy_rate = sum(r['correct'] for r in routing_accuracy) / len(routing_accuracy)
    print(f"\nRouting accuracy: {accuracy_rate:.1%}")
    
    return routing_accuracy
```

**Solutions:**

1. **Tune Routing Rules:**
```python
def optimize_routing_rules():
    """Optimize routing engine rules based on performance data."""
    from scripts.runtime_router import RoutingEngine
    
    # Create optimized routing engine
    engine = RoutingEngine()
    
    # Adjust category mappings based on success patterns
    enhanced_mappings = {
        'financial_operations': ['rollback_first', 'observability_first'],
        'security_operations': ['rollback_first', 'observability_first'], 
        'system_integration': ['state_model_first', 'rollback_first'],
        'compliance_management': ['state_model_first', 'observability_first'],
        'data_operations': ['rollback_first', 'state_model_first'],
        'infrastructure_management': ['observability_first', 'rollback_first']
    }
    
    # Update keyword triggers with more specific terms
    enhanced_keywords = {
        'rollback_first': [
            'refund', 'rollback', 'undo', 'revert', 'cancel', 'abort',
            'transaction', 'payment', 'financial', 'billing', 'charge',
            'migration', 'database', 'data', 'backup', 'restore',
            'reverse', 'compensation', 'chargeback'
        ],
        'state_model_first': [
            'workflow', 'process', 'state', 'transition', 'approval',
            'compliance', 'audit', 'regulation', 'legal', 'policy',
            'architecture', 'design', 'model', 'system', 'integration',
            'orchestration', 'coordination', 'sequence'
        ],
        'observability_first': [
            'monitor', 'alert', 'metric', 'dashboard', 'log', 'trace',
            'performance', 'availability', 'reliability', 'sla', 'slo',
            'capacity', 'scale', 'optimization', 'efficiency',
            'diagnostic', 'health', 'status', 'visibility'
        ]
    }
    
    return enhanced_mappings, enhanced_keywords
```

---

## Circuit Breaker Problems

### Problem: Circuit Breaker Stuck Open

**Symptoms:**
- Services consistently fail with "Circuit breaker open" errors
- No automatic recovery despite timeout expiration
- System performance severely degraded

**Diagnosis:**
```python
from scripts.circuit_breaker import ProductionSafetyManager

def diagnose_circuit_breaker_issues():
    """Diagnose circuit breaker problems."""
    safety_manager = ProductionSafetyManager("./configs/fusion")
    
    for breaker_name, breaker in safety_manager.circuit_breakers.items():
        state = breaker.get_state()
        
        print(f"Circuit Breaker: {breaker_name}")
        print(f"  State: {state['state']}")
        print(f"  Failure Count: {state['failure_count']}")
        print(f"  Last Failure: {state['last_failure_time']}")
        print(f"  State Change Time: {state['state_change_time']}")
        
        if state['state'] == 'open':
            # Check if recovery timeout has passed
            current_time = time.time()
            
            try:
                if isinstance(state['last_failure_time'], str):
                    if 'T' in state['last_failure_time']:
                        failure_dt = datetime.fromisoformat(state['last_failure_time'].replace('Z', '+00:00'))
                        failure_timestamp = failure_dt.timestamp()
                    else:
                        failure_timestamp = float(state['last_failure_time'])
                else:
                    failure_timestamp = float(state['last_failure_time'])
                
                time_since_failure = current_time - failure_timestamp
                recovery_timeout = breaker.recovery_timeout
                
                print(f"  Time since failure: {time_since_failure:.1f}s")
                print(f"  Recovery timeout: {recovery_timeout}s")
                print(f"  Should attempt reset: {time_since_failure >= recovery_timeout}")
                
            except Exception as e:
                print(f"  ❌ Timestamp parsing error: {e}")
```

**Solutions:**

1. **Force Circuit Breaker Reset:**
```python
def force_circuit_breaker_reset(breaker_name):
    """Manually reset a stuck circuit breaker."""
    safety_manager = ProductionSafetyManager("./configs/fusion")
    
    if breaker_name in safety_manager.circuit_breakers:
        breaker = safety_manager.circuit_breakers[breaker_name]
        
        # Force transition to closed state
        with breaker.lock:
            breaker.state = 'closed'
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.half_open_calls = 0
            breaker.state_change_time = datetime.now(timezone.utc).isoformat()
            breaker.consecutive_failures = 0
            
        print(f"✅ Circuit breaker {breaker_name} manually reset")
        return True
    else:
        print(f"❌ Circuit breaker {breaker_name} not found")
        return False
```

2. **Adjust Recovery Parameters:**
```python
def adjust_recovery_parameters(breaker_name, new_params):
    """Adjust circuit breaker recovery parameters."""
    safety_manager = ProductionSafetyManager("./configs/fusion")
    
    if breaker_name in safety_manager.circuit_breakers:
        breaker = safety_manager.circuit_breakers[breaker_name]
        
        # Update recovery parameters
        if 'recovery_timeout' in new_params:
            breaker.recovery_timeout = new_params['recovery_timeout']
        if 'failure_threshold' in new_params:
            breaker.failure_threshold = new_params['failure_threshold']
        if 'success_threshold' in new_params:
            breaker.success_threshold = new_params['success_threshold']
            
        print(f"✅ Updated parameters for {breaker_name}: {new_params}")
```

### Problem: False Positive Circuit Trips

**Symptoms:**
- Circuit breakers trip on minor issues
- Excessive sensitivity to temporary failures
- System availability reduced unnecessarily

**Solutions:**

1. **Tune Trip Conditions:**
```python
def optimize_trip_conditions():
    """Optimize circuit breaker trip conditions."""
    
    # More resilient trip conditions
    optimized_conditions = {
        'fusion_primary': {
            'failure_threshold': 8,  # Increased from 5
            'recovery_timeout': 180,  # Reduced from 300
            'consecutive_success_threshold': 3,  # Increased from 2
            'trip_conditions': [
                {
                    'metric': 'fusion_overlay_latency_p95',
                    'threshold_ms': 8000,  # Increased from 5000
                    'consecutive_violations': 5  # Increased from 3
                },
                {
                    'metric': 'fusion_success_rate',
                    'threshold': 0.7,  # Decreased from 0.95
                    'consecutive_violations': 3
                }
            ]
        }
    }
    
    return optimized_conditions
```

---

## Budget and Energy Management

### Problem: Budget Tripwire Stuck Active

**Symptoms:**
- Continuous budget downgrade despite normal usage
- Token consumption appears normal but tripwire remains active
- System performance consistently degraded

**Diagnosis:**
```python
from fusion_ops.budget_tripwire import get_budget_tripwire

def diagnose_budget_tripwire():
    """Diagnose budget tripwire issues."""
    tripwire = get_budget_tripwire()
    status = tripwire.get_status()
    
    print("Budget Tripwire Status:")
    print(f"  Downgrade Active: {status['downgrade_active']}")
    print(f"  Consecutive Breaches: {status['consecutive_breaches']}")
    print(f"  Breach Threshold: {status['breach_threshold']}%")
    print(f"  Last Breach: {status['last_breach_timestamp']}")
    print(f"  Activated At: {status['downgrade_activated_at']}")
    
    if status['recent_breaches']:
        print("Recent Breaches:")
        for breach in status['recent_breaches']:
            print(f"    {breach['timestamp']}: {breach['token_delta']:.1f}% delta")
    
    # Check if tripwire state file is corrupted
    state_file = Path(".fusion_tripwire_state.json")
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            print(f"  State File Valid: ✅")
            print(f"  State File Size: {state_file.stat().st_size} bytes")
        except Exception as e:
            print(f"  State File Corrupted: ❌ {str(e)}")
```

**Solutions:**

1. **Reset Budget Tripwire:**
```python
def reset_budget_tripwire_safe():
    """Safely reset budget tripwire with validation."""
    tripwire = get_budget_tripwire()
    
    # Get current status for logging
    old_status = tripwire.get_status()
    
    # Perform reset
    result = tripwire.reset_downgrade()
    
    # Validate reset
    new_status = tripwire.get_status()
    
    print(f"Budget Tripwire Reset:")
    print(f"  Was Active: {result['was_active']}")
    print(f"  Reset Successful: {result['reset']}")
    print(f"  New Status - Active: {new_status['downgrade_active']}")
    print(f"  New Status - Breaches: {new_status['consecutive_breaches']}")
    
    return result['reset']
```

2. **Repair Corrupted State:**
```python
def repair_tripwire_state():
    """Repair corrupted tripwire state file."""
    state_file = Path(".fusion_tripwire_state.json")
    backup_file = Path(".fusion_tripwire_state.json.backup")
    
    # Create backup of current state
    if state_file.exists():
        try:
            shutil.copy2(state_file, backup_file)
            print(f"✅ Created backup: {backup_file}")
        except Exception as e:
            print(f"⚠️  Could not create backup: {e}")
    
    # Reset to clean state
    clean_state = {
        "consecutive_breaches": 0,
        "last_breach_timestamp": None,
        "downgrade_active": False,
        "downgrade_activated_at": None,
        "breach_history": []
    }
    
    try:
        with open(state_file, 'w') as f:
            json.dump(clean_state, f, indent=2)
        print(f"✅ Created clean state file")
        
        # Re-initialize tripwire
        from fusion_ops.budget_tripwire import reset_tripwire_instance
        reset_tripwire_instance()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to repair state: {e}")
        return False
```

### Problem: Excessive Token Consumption

**Symptoms:**
- Token usage significantly above baseline
- Frequent budget breach warnings
- Cost escalation without quality improvements

**Diagnosis:**
```python
def analyze_token_consumption_patterns():
    """Analyze token consumption patterns for optimization."""
    
    # Collect token usage metrics from recent operations
    usage_patterns = {
        'overlay_types': {},
        'scenario_categories': {},
        'complexity_ranges': {}
    }
    
    # Simulate analysis of recent token usage
    sample_operations = [
        {'overlay': 'rollback_first', 'tokens': 2200, 'category': 'financial_operations', 'complexity': 0.6},
        {'overlay': 'state_model_first', 'tokens': 3100, 'category': 'compliance_management', 'complexity': 0.8},
        {'overlay': 'observability_first', 'tokens': 1800, 'category': 'infrastructure_management', 'complexity': 0.4}
    ]
    
    for op in sample_operations:
        # Analyze by overlay type
        overlay = op['overlay']
        if overlay not in usage_patterns['overlay_types']:
            usage_patterns['overlay_types'][overlay] = []
        usage_patterns['overlay_types'][overlay].append(op['tokens'])
        
        # Analyze by scenario category
        category = op['category']
        if category not in usage_patterns['scenario_categories']:
            usage_patterns['scenario_categories'][category] = []
        usage_patterns['scenario_categories'][category].append(op['tokens'])
    
    # Calculate averages and identify high-usage patterns
    for pattern_type, patterns in usage_patterns.items():
        print(f"\nToken Usage by {pattern_type.replace('_', ' ').title()}:")
        for pattern_name, token_usage in patterns.items():
            if token_usage:
                avg_usage = sum(token_usage) / len(token_usage)
                max_usage = max(token_usage)
                print(f"  {pattern_name}: Avg {avg_usage:.0f}, Max {max_usage:.0f} tokens")
```

**Solutions:**

1. **Implement Token Usage Optimization:**
```python
def optimize_token_usage():
    """Implement token usage optimization strategies."""
    
    optimization_strategies = {
        'overlay_selection': {
            'description': 'Use lighter overlays for simple scenarios',
            'implementation': 'Adjust complexity thresholds',
            'expected_savings': '15-25%'
        },
        'context_pruning': {
            'description': 'Remove unnecessary context from prompts',
            'implementation': 'Implement smart context filtering',
            'expected_savings': '10-20%'
        },
        'caching_responses': {
            'description': 'Cache responses for similar scenarios',
            'implementation': 'Implement response caching layer', 
            'expected_savings': '20-40%'
        }
    }
    
    return optimization_strategies
```

---

## Configuration Issues

### Problem: Configuration File Corruption

**Symptoms:**
- YAML/JSON parse errors during startup
- Missing configuration sections
- Default values used unexpectedly

**Diagnosis:**
```python
def validate_configuration_files():
    """Validate all configuration files."""
    config_dir = Path("./configs/fusion")
    validation_results = {}
    
    config_files = {
        'overlay_params.yaml': 'yaml',
        'scenario_profiles.yaml': 'yaml',
        'slo.yaml': 'yaml',
        'energy_governance.yaml': 'yaml',
        'drift_policy.yaml': 'yaml'
    }
    
    for config_file, file_type in config_files.items():
        file_path = config_dir / config_file
        result = {
            'exists': file_path.exists(),
            'readable': False,
            'valid': False,
            'size': 0,
            'error': None
        }
        
        if file_path.exists():
            try:
                result['size'] = file_path.stat().st_size
                result['readable'] = os.access(file_path, os.R_OK)
                
                if result['readable']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_type == 'yaml':
                            yaml.safe_load(f)
                        else:
                            json.load(f)
                    result['valid'] = True
                    
            except Exception as e:
                result['error'] = str(e)
        
        validation_results[config_file] = result
        
        # Print validation status
        status = "✅" if result['valid'] else "❌" if result['exists'] else "⚠️"
        print(f"{status} {config_file}: {'Valid' if result['valid'] else result.get('error', 'Missing')}")
    
    return validation_results
```

**Solutions:**

1. **Repair Configuration Files:**
```python
def repair_configuration_files():
    """Repair or recreate corrupted configuration files."""
    
    # Default configurations for recovery
    default_configs = {
        'overlay_params.yaml': {
            'ENTROPY_REDUCTION_TARGET': "0.75",
            'CONTINUITY_ENFORCEMENT': "strict_memory",
            'TRUST_SCORING_MODEL': "comprehensive_validation",
            'FUSION_MODE': "enhanced_overlay",
            'FUSION_OVERLAY_VERSION': "v2.1.0"
        },
        'slo.yaml': {
            'service_level_objectives': {
                'fusion_success_rate': {
                    'target': 0.95,
                    'measurement_window_minutes': 60
                },
                'fusion_latency_p95': {
                    'target_ms': 2000,
                    'measurement_window_minutes': 60
                }
            }
        }
    }
    
    config_dir = Path("./configs/fusion")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for config_file, default_content in default_configs.items():
        file_path = config_dir / config_file
        backup_path = config_dir / f"{config_file}.backup"
        
        # Backup existing file if it exists
        if file_path.exists():
            try:
                shutil.copy2(file_path, backup_path)
                print(f"📋 Backed up {config_file}")
            except Exception as e:
                print(f"⚠️  Could not backup {config_file}: {e}")
        
        # Write default configuration
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_content, f, default_flow_style=False, indent=2)
            print(f"✅ Recreated {config_file}")
        except Exception as e:
            print(f"❌ Failed to recreate {config_file}: {e}")
```

---

## Performance Degradation

### Problem: Slow Response Times

**Symptoms:**
- Increased latency in fusion operations
- Timeout errors
- Poor user experience

**Diagnosis:**
```python
import time
import statistics

def benchmark_system_performance():
    """Benchmark current system performance."""
    
    # Test scenarios with different complexity levels
    test_scenarios = [
        {'complexity': 0.3, 'description': 'Simple scenario'},
        {'complexity': 0.6, 'description': 'Medium scenario'}, 
        {'complexity': 0.9, 'description': 'Complex scenario'}
    ]
    
    performance_results = []
    
    for scenario in test_scenarios:
        latencies = []
        
        # Run multiple iterations
        for i in range(5):
            start_time = time.time()
            
            try:
                # Simulate fusion operation
                router = RuntimeRouter("./configs/fusion")
                decision = router.route_scenario({
                    'id': f'perf_test_{i}',
                    'category': 'system_integration',
                    'complexity': scenario['complexity'],
                    'description': scenario['description']
                })
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
            except Exception as e:
                print(f"❌ Performance test failed: {e}")
                latencies.append(float('inf'))
        
        # Calculate statistics
        valid_latencies = [l for l in latencies if l != float('inf')]
        if valid_latencies:
            result = {
                'complexity': scenario['complexity'],
                'description': scenario['description'],
                'avg_latency': statistics.mean(valid_latencies),
                'p95_latency': statistics.quantiles(valid_latencies, n=20)[18] if len(valid_latencies) >= 5 else max(valid_latencies),
                'min_latency': min(valid_latencies),
                'max_latency': max(valid_latencies),
                'success_rate': len(valid_latencies) / len(latencies)
            }
        else:
            result = {
                'complexity': scenario['complexity'],
                'description': scenario['description'],
                'avg_latency': float('inf'),
                'success_rate': 0
            }
        
        performance_results.append(result)
        
        print(f"Performance - {result['description']}:")
        print(f"  Average: {result['avg_latency']:.1f}ms")
        print(f"  P95: {result['p95_latency']:.1f}ms") 
        print(f"  Success Rate: {result['success_rate']:.1%}")
    
    return performance_results
```

**Solutions:**

1. **Performance Optimization:**
```python
def implement_performance_optimizations():
    """Implement performance optimization strategies."""
    
    optimizations = {
        'overlay_caching': {
            'description': 'Cache parsed overlays in memory',
            'expected_improvement': '30-50% faster overlay loading'
        },
        'lazy_initialization': {
            'description': 'Initialize components only when needed',
            'expected_improvement': '20-30% faster startup'
        },
        'async_processing': {
            'description': 'Use async/await for I/O operations',
            'expected_improvement': '40-60% better concurrency'
        },
        'connection_pooling': {
            'description': 'Pool database/API connections',
            'expected_improvement': '25-35% reduced latency'
        }
    }
    
    # Implementation example for overlay caching
    class CachedRuntimeRouter(RuntimeRouter):
        """Enhanced runtime router with performance optimizations."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._routing_cache = {}
            self._cache_size_limit = 1000
        
        def route_scenario(self, scenario):
            # Create cache key from scenario
            cache_key = hash(json.dumps(scenario, sort_keys=True))
            
            if cache_key in self._routing_cache:
                return self._routing_cache[cache_key]
            
            # Perform routing
            decision = super().route_scenario(scenario)
            
            # Cache result (with size limit)
            if len(self._routing_cache) < self._cache_size_limit:
                self._routing_cache[cache_key] = decision
            
            return decision
    
    return optimizations
```

---

## Recovery Procedures

### Complete System Recovery

```python
def perform_complete_system_recovery():
    """Perform complete system recovery procedure."""
    
    print("🚨 Starting RESONTINEX System Recovery")
    print("=" * 50)
    
    recovery_steps = [
        ("Backup Current State", backup_current_state),
        ("Validate Configuration", validate_configuration_files),
        ("Reset Circuit Breakers", reset_all_circuit_breakers),
        ("Clear Budget Tripwire", reset_budget_tripwire_safe),
        ("Rebuild Overlays", rebuild_overlay_cache),
        ("Test System Health", run_system_diagnostics),
        ("Verify Operations", verify_system_operations)
    ]
    
    recovery_success = True
    failed_steps = []
    
    for step_name, step_function in recovery_steps:
        print(f"\n🔧 {step_name}...")
        try:
            result = step_function()
            if result:
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"⚠️  {step_name} completed with warnings")
        except Exception as e:
            print(f"❌ {step_name} failed: {str(e)}")
            failed_steps.append(step_name)
            recovery_success = False
    
    print(f"\n{'='*50}")
    if recovery_success:
        print("✅ System recovery completed successfully")
    else:
        print(f"⚠️  System recovery completed with {len(failed_steps)} failed steps:")
        for failed_step in failed_steps:
            print(f"   - {failed_step}")
    
    return recovery_success, failed_steps

def backup_current_state():
    """Backup current system state."""
    backup_dir = Path("./backups") / f"recovery_{int(time.time())}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup configuration files
    config_files = [
        ".fusion_tripwire_state.json",
        "./configs/fusion/overlay_params.yaml",
        "./configs/fusion/slo.yaml"
    ]
    
    for config_file in config_files:
        src_path = Path(config_file)
        if src_path.exists():
            dst_path = backup_dir / src_path.name
            shutil.copy2(src_path, dst_path)
    
    print(f"📦 System state backed up to {backup_dir}")
    return True

def reset_all_circuit_breakers():
    """Reset all circuit breakers to closed state."""
    try:
        safety_manager = ProductionSafetyManager("./configs/fusion")
        reset_count = 0
        
        for breaker_name, breaker in safety_manager.circuit_breakers.items():
            state = breaker.get_state()
            if state['state'] != 'closed':
                force_circuit_breaker_reset(breaker_name)
                reset_count += 1
        
        print(f"🔄 Reset {reset_count} circuit breakers")
        return True
        
    except Exception as e:
        print(f"Failed to reset circuit breakers: {e}")
        return False

def rebuild_overlay_cache():
    """Rebuild overlay cache."""
    try:
        # Clear any cached overlays and reload
        router = RuntimeRouter("./configs/fusion")
        overlay_count = len(router.overlays)
        
        print(f"🔄 Rebuilt overlay cache with {overlay_count} overlays")
        return overlay_count > 0
        
    except Exception as e:
        print(f"Failed to rebuild overlay cache: {e}")
        return False

def verify_system_operations():
    """Verify system operations are working."""
    try:
        # Test basic operations
        test_scenario = {
            'id': 'recovery_verification',
            'category': 'system_integration', 
            'complexity': 0.5,
            'description': 'Recovery verification test'
        }
        
        router = RuntimeRouter("./configs/fusion")
        decision = router.route_scenario(test_scenario)
        
        if decision.selected_overlay != 'none':
            print(f"✅ System operations verified - selected {decision.selected_overlay}")
            return True
        else:
            print(f"⚠️  System operations working but no overlay selected")
            return False
            
    except Exception as e:
        print(f"System operations verification failed: {e}")
        return False
```

This comprehensive troubleshooting guide provides real-world error scenarios and solutions based on the actual RESONTINEX codebase, with production-ready diagnostic tools and recovery procedures.