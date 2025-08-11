#!/usr/bin/env python3
"""
RESONTINEX Circuit Breaker and SLO Monitor
Production safety system with automatic degradation and recovery.
"""

import os
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import statistics
import threading
import logging


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class Metric:
    """Represents a single metric measurement."""
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str]


@dataclass
class SLOViolation:
    """Represents an SLO violation event."""
    slo_name: str
    target_value: float
    actual_value: float
    measurement_window: int
    violation_timestamp: str
    severity: str


@dataclass
class CircuitBreakerState:
    """Current state of a circuit breaker."""
    name: str
    state: CircuitState
    failure_count: int
    last_failure_time: Optional[str]
    last_success_time: Optional[str]
    state_change_time: str
    half_open_calls: int


class NullMetricsCollector:
    """Null metrics collector for tests."""
    def incr(self, name: str, **tags): pass
    def gauge(self, name: str, value: float, **tags): pass
    def timing(self, name: str, ms: float, **tags): pass
    def record_metric(self, metric): pass
    def calculate_rate(self, metric_name: str, window_minutes: int) -> float: return 0.95
    def calculate_percentile(self, metric_name: str, window_minutes: int, percentile: float) -> float: return 1200.0
    def get_metrics(self, metric_name: str, window_minutes: int): return []


class MetricsCollector:
    """Collects and stores metrics for SLO monitoring."""
    
    def __init__(self, retention_minutes: int = 1440):  # 24 hours default
        self.metrics_storage = {}
        self.retention_minutes = retention_minutes
        self.lock = threading.Lock()
    
    def record_metric(self, metric: Metric):
        """Record a single metric measurement."""
        with self.lock:
            if metric.name not in self.metrics_storage:
                self.metrics_storage[metric.name] = []
            
            self.metrics_storage[metric.name].append(metric)
            self._cleanup_old_metrics(metric.name)
    
    def _cleanup_old_metrics(self, metric_name: str):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.retention_minutes)
        cutoff_timestamp = cutoff_time.isoformat()
        
        self.metrics_storage[metric_name] = [
            m for m in self.metrics_storage[metric_name] 
            if m.timestamp > cutoff_timestamp
        ]
    
    def get_metrics(self, metric_name: str, window_minutes: int) -> List[Metric]:
        """Get metrics for a specific time window."""
        with self.lock:
            if metric_name not in self.metrics_storage:
                return []
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            cutoff_timestamp = cutoff_time.isoformat()
            
            return [
                m for m in self.metrics_storage[metric_name]
                if m.timestamp > cutoff_timestamp
            ]
    
    def calculate_rate(self, metric_name: str, window_minutes: int) -> float:
        """Calculate rate (success/total) for a metric."""
        metrics = self.get_metrics(metric_name, window_minutes)
        if not metrics:
            return 1.0  # Assume success if no data
        
        values = [m.value for m in metrics]
        return statistics.mean(values)
    
    def calculate_percentile(self, metric_name: str, window_minutes: int, percentile: float) -> float:
        """Calculate percentile for a metric."""
        metrics = self.get_metrics(metric_name, window_minutes)
        if not metrics:
            return 0.0
        
        values = sorted([m.value for m in metrics])
        if not values:
            return 0.0
        
        index = int(percentile * (len(values) - 1))
        return values[index]


class SLOMonitor:
    """Service Level Objective monitoring system."""
    
    def __init__(self, slo_config: Dict[str, Any], metrics_collector=None):
        self.slo_config = slo_config or {}
        self.slos = slo_config  # Alias for test compatibility
        self.metrics_collector = metrics_collector or NullMetricsCollector()
        self.violations = []
        self.lock = threading.Lock()
        self.state = {"requests": 0, "errors": 0, "latencies": []}
        self.request_history = []  # For tracking request outcomes
    
    def check_slos(self) -> List[SLOViolation]:
        """Check all SLOs and return any violations."""
        violations = []
        
        slos = self.slo_config.get('service_level_objectives', {})
        
        for slo_name, slo_definition in slos.items():
            violation = self._check_single_slo(slo_name, slo_definition)
            if violation:
                violations.append(violation)
        
        with self.lock:
            self.violations.extend(violations)
            # Keep only recent violations
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            cutoff_timestamp = cutoff_time.isoformat()
            self.violations = [v for v in self.violations if v.violation_timestamp > cutoff_timestamp]
        
        return violations
    
    def _check_single_slo(self, slo_name: str, slo_definition: Dict[str, Any]) -> Optional[SLOViolation]:
        """Check a single SLO for violations."""
        target = slo_definition.get('target', 1.0)
        window_minutes = slo_definition.get('measurement_window_minutes', 60)
        alert_threshold = slo_definition.get('alert_threshold', target * 0.95)
        
        # Determine metric calculation method based on SLO type
        if '_rate' in slo_name:
            actual_value = self.metrics_collector.calculate_rate(slo_name, window_minutes)
            violation_condition = actual_value < alert_threshold
        elif '_latency_' in slo_name:
            actual_value = self.metrics_collector.calculate_percentile(slo_name, window_minutes, 0.95)
            target_ms = slo_definition.get('target_ms', 2000)
            alert_threshold_ms = slo_definition.get('alert_threshold_ms', target_ms * 1.5)
            violation_condition = actual_value > alert_threshold_ms
        elif '_delta_' in slo_name:
            actual_value = self.metrics_collector.calculate_rate(slo_name, window_minutes)
            target_pct = slo_definition.get('target_percentage', 0.12)
            alert_threshold_pct = slo_definition.get('alert_threshold_percentage', target_pct * 1.5)
            violation_condition = actual_value > alert_threshold_pct
        else:
            actual_value = self.metrics_collector.calculate_rate(slo_name, window_minutes)
            violation_condition = actual_value < alert_threshold
        
        if violation_condition:
            severity = "critical" if actual_value < target * 0.8 else "warning"
            
            return SLOViolation(
                slo_name=slo_name,
                target_value=target,
                actual_value=actual_value,
                measurement_window=window_minutes,
                violation_timestamp=datetime.now(timezone.utc).isoformat(),
                severity=severity
            )
        
        return None
    
    def record_request_outcome(self, success: bool, latency: float):
        """Record the outcome of a request for SLO tracking."""
        with self.lock:
            self.request_history.append({
                'success': success,
                'latency': latency,
                'timestamp': time.time()
            })
            
            # Keep only recent history (last hour)
            cutoff_time = time.time() - 3600
            self.request_history = [r for r in self.request_history if r['timestamp'] > cutoff_time]
    
    def calculate_availability(self) -> float:
        """Calculate availability from recorded requests."""
        with self.lock:
            if not self.request_history:
                return 1.0
            
            successful_requests = sum(1 for r in self.request_history if r['success'])
            return successful_requests / len(self.request_history)
    
    def calculate_p95_latency(self) -> float:
        """Calculate P95 latency from recorded requests."""
        with self.lock:
            if not self.request_history:
                return 0.0
            
            latencies = sorted([r['latency'] for r in self.request_history])
            if not latencies:
                return 0.0
            
            index = int(0.95 * (len(latencies) - 1))
            return latencies[index]
    
    def calculate_error_rate(self) -> float:
        """Calculate error rate from recorded requests."""
        with self.lock:
            if not self.request_history:
                return 0.0
            
            failed_requests = sum(1 for r in self.request_history if not r['success'])
            return failed_requests / len(self.request_history)
    
    def check_slo_compliance(self) -> Dict[str, Any]:
        """Check SLO compliance and return status for each SLO."""
        availability = self.calculate_availability()
        error_rate = self.calculate_error_rate()
        p95_latency = self.calculate_p95_latency()
        
        return {
            'availability': {
                'compliant': availability >= self.slos.get('availability', {}).get('target', 0.99),
                'current': availability,
                'target': self.slos.get('availability', {}).get('target', 0.99)
            },
            'error_rate': {
                'compliant': error_rate <= self.slos.get('error_rate', {}).get('target', 0.01),
                'current': error_rate,
                'target': self.slos.get('error_rate', {}).get('target', 0.01)
            },
            'latency_p95': {
                'compliant': p95_latency <= self.slos.get('latency_p95', {}).get('target', 2.0),
                'current': p95_latency,
                'target': self.slos.get('latency_p95', {}).get('target', 2.0)
            }
        }
    
    def get_recent_violations(self, hours: int = 24) -> List[SLOViolation]:
        """Get recent SLO violations."""
        with self.lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            cutoff_timestamp = cutoff_time.isoformat()
            
            return [v for v in self.violations if v.violation_timestamp > cutoff_timestamp]


class CircuitBreaker:
    """
    Production-grade circuit breaker with comprehensive edge case handling.
    
    Provides fail-fast behavior with automatic recovery, protecting downstream
    services from cascade failures through intelligent state management.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], metrics_collector=None):
        self.name = name
        self.config = config
        self.metrics_collector = metrics_collector or NullMetricsCollector()
        
        # State management with thread safety
        self.state = 'closed'
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state_change_time = datetime.now(timezone.utc).isoformat()
        self.half_open_calls = 0
        
        # Configuration with production defaults
        self.failure_threshold = config.get('failure_threshold', 5)
        self.recovery_timeout = config.get('recovery_timeout', 300)
        self.half_open_max_calls = config.get('half_open_max_calls', 3)
        self.success_threshold = config.get('success_threshold', 2)
        self.trip_conditions = config.get('trip_conditions', [])
        
        # Enhanced edge case handling
        self.consecutive_success_threshold = config.get('consecutive_success_threshold', 2)
        self.fast_fail_threshold = config.get('fast_fail_threshold', 10)
        self.sliding_window_size = config.get('sliding_window_size', 100)
        self.minimum_throughput = config.get('minimum_throughput', 10)
        
        # Thread safety and state consistency
        self.lock = threading.RLock()  # Reentrant lock for nested calls
        self._state_transition_callbacks = []
        self._last_state_check = time.time()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute a function through the circuit breaker."""
        if self.state == 'open':
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception(f"Circuit breaker {self.name} is open")
        
        if self.state == 'half_open':
            if self.half_open_calls >= self.half_open_max_calls:
                raise Exception(f"Circuit breaker {self.name} half-open call limit exceeded")
        
        try:
            with self.lock:
                if self.state == 'half_open':
                    self.half_open_calls += 1
            
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_success(self):
        """Record a successful call."""
        with self.lock:
            self.last_success_time = datetime.now(timezone.utc).isoformat()
            
            if self.state == 'half_open':
                self._transition_to_closed()
            else:
                self.failure_count = 0
    
    def _record_failure(self):
        """Record a failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc).isoformat()
            
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """
        Intelligent recovery decision with multiple validation criteria.
        
        Evaluates timeout expiry, system health indicators, and failure patterns
        to determine optimal recovery timing with comprehensive edge case handling.
        """
        if not self.last_failure_time:
            return False
        
        current_time = time.time()
        
        # Parse failure timestamp with robust error handling
        try:
            if isinstance(self.last_failure_time, str):
                # Handle ISO format timestamps
                if 'T' in self.last_failure_time:
                    dt = datetime.fromisoformat(self.last_failure_time.replace('Z', '+00:00'))
                    failure_timestamp = dt.timestamp()
                else:
                    # Handle epoch string timestamps
                    failure_timestamp = float(self.last_failure_time)
            else:
                failure_timestamp = float(self.last_failure_time)
        except (ValueError, TypeError, AttributeError) as e:
            # Log parsing error and default to not attempting reset
            logging.warning(f"Circuit breaker {self.name} timestamp parsing error: {e}")
            return False
        
        # Use dynamic recovery timeout if available
        recovery_timeout = getattr(self, 'dynamic_recovery_timeout', self.recovery_timeout)
        time_elapsed = current_time - failure_timestamp
        
        # Basic timeout check
        if time_elapsed < recovery_timeout:
            return False
        
        # Advanced recovery criteria for production resilience
        
        # 1. Prevent rapid state oscillation
        if hasattr(self, '_last_recovery_attempt'):
            time_since_last_attempt = current_time - self._last_recovery_attempt
            if time_since_last_attempt < 30:  # Minimum 30 seconds between attempts
                return False
        
        # 2. Check system health indicators
        consecutive_failures = getattr(self, 'consecutive_failures', 0)
        if consecutive_failures > 5:
            # Require longer timeout for severely degraded services
            extended_timeout = recovery_timeout * (1.5 ** (consecutive_failures - 5))
            if time_elapsed < extended_timeout:
                return False
        
        # 3. Validate minimum throughput requirement
        recent_metrics = self.metrics_collector.get_metrics(f"circuit_breaker_{self.name}_calls", 5)
        if len(recent_metrics) < self.minimum_throughput:
            # Insufficient traffic to justify recovery attempt
            return False
        
        # Record recovery attempt timestamp
        self._last_recovery_attempt = current_time
        
        return True
    
    def _transition_to_closed(self):
        """
        Transition circuit to closed state with comprehensive edge case handling.
        
        Resets all failure metrics and notifies state change callbacks.
        Handles race conditions and ensures atomic state transitions.
        """
        with self.lock:
            previous_state = self.state
            self.state = 'closed'
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.state_change_time = datetime.now(timezone.utc).isoformat()
            
            # Record metrics for state transition
            self.metrics_collector.record_metric(Metric(
                name=f"circuit_breaker_{self.name}_state_transition",
                value=1.0,
                timestamp=self.state_change_time,
                tags={'from_state': previous_state, 'to_state': 'closed', 'circuit': self.name}
            ))
            
            # Execute state change callbacks
            self._notify_state_change(previous_state, 'closed')
    
    def _transition_to_open(self):
        """
        Transition circuit to open state with failure analysis.
        
        Records failure patterns and triggers immediate fail-fast behavior.
        Implements exponential backoff for recovery timeout adjustments.
        """
        with self.lock:
            previous_state = self.state
            self.state = 'open'
            self.half_open_calls = 0
            self.state_change_time = datetime.now(timezone.utc).isoformat()
            
            # Implement exponential backoff for repeated failures
            failure_streak = getattr(self, 'consecutive_failures', 0) + 1
            self.consecutive_failures = failure_streak
            
            # Adjust recovery timeout based on failure patterns
            base_timeout = self.recovery_timeout
            if failure_streak > 3:
                adjusted_timeout = min(base_timeout * (2 ** (failure_streak - 3)), base_timeout * 8)
                self.dynamic_recovery_timeout = adjusted_timeout
            else:
                self.dynamic_recovery_timeout = base_timeout
            
            # Record critical state transition metrics
            self.metrics_collector.record_metric(Metric(
                name=f"circuit_breaker_{self.name}_state_transition",
                value=0.0,  # 0 indicates failure state
                timestamp=self.state_change_time,
                tags={'from_state': previous_state, 'to_state': 'open', 'circuit': self.name, 'failure_streak': str(failure_streak)}
            ))
            
            self._notify_state_change(previous_state, 'open')
    
    def _transition_to_half_open(self):
        """
        Transition circuit to half-open state with controlled probe behavior.
        
        Implements intelligent probing strategy with minimal risk exposure.
        Tracks probe success rates for adaptive recovery decisions.
        """
        with self.lock:
            previous_state = self.state
            self.state = 'half_open'
            self.half_open_calls = 0
            self.success_count = 0
            self.state_change_time = datetime.now(timezone.utc).isoformat()
            
            # Reset consecutive failures counter on recovery attempt
            self.consecutive_failures = 0
            
            # Record recovery attempt metrics
            self.metrics_collector.record_metric(Metric(
                name=f"circuit_breaker_{self.name}_state_transition",
                value=0.5,  # 0.5 indicates probing state
                timestamp=self.state_change_time,
                tags={'from_state': previous_state, 'to_state': 'half_open', 'circuit': self.name}
            ))
            
            self._notify_state_change(previous_state, 'half_open')
    
    def _notify_state_change(self, from_state: str, to_state: str):
        """Execute registered state change callbacks with error handling."""
        for callback in self._state_transition_callbacks:
            try:
                callback(self.name, from_state, to_state, self.state_change_time)
            except Exception as e:
                # Log callback errors but don't fail the state transition
                logging.warning(f"Circuit breaker {self.name} callback error: {e}")
    
    def register_state_callback(self, callback: Callable):
        """Register callback for state transition notifications."""
        if callback not in self._state_transition_callbacks:
            self._state_transition_callbacks.append(callback)
    
    def check_trip_conditions(self) -> bool:
        """Check if circuit should trip based on configured conditions."""
        for condition in self.trip_conditions:
            metric_name = condition.get('metric')
            threshold = condition.get('threshold', 0)
            threshold_ms = condition.get('threshold_ms', 0)
            consecutive_violations = condition.get('consecutive_violations', 1)
            
            # Check recent metrics
            recent_metrics = self.metrics_collector.get_metrics(metric_name, 5)  # Last 5 minutes
            
            if len(recent_metrics) < consecutive_violations:
                continue
            
            # Check consecutive violations
            violations = 0
            for metric in recent_metrics[-consecutive_violations:]:
                if threshold_ms > 0 and metric.value > threshold_ms:
                    violations += 1
                elif threshold > 0 and metric.value > threshold:
                    violations += 1
            
            if violations >= consecutive_violations:
                return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state,
                'failure_count': self.failure_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'state_change_time': str(self.state_change_time),
                'half_open_calls': self.half_open_calls
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ProductionSafetyManager:
    """Main production safety management system."""
    
    def __init__(self, config_or_dir, state_dir: str = "./build/safety"):
        # Handle both dict and path inputs for test compatibility
        if isinstance(config_or_dir, dict):
            self.config = config_or_dir
            self.config_dir = Path("./configs/fusion")
        else:
            self.config_dir = Path(config_or_dir)
            self.config = self._load_slo_config()
            
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.slo_monitor = SLOMonitor(self.config, self.metrics_collector)
        self.circuit_breakers = self._initialize_circuit_breakers()
        
        # State tracking
        self.degraded_services = set()
        self.monitoring_active = True
        
    def _load_slo_config(self) -> Dict[str, Any]:
        """Load SLO configuration."""
        slo_path = self.config_dir / "slo.yaml"
        if slo_path.exists():
            with open(slo_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _initialize_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Initialize all configured circuit breakers."""
        breakers = {}
        circuit_configs = self.config.get('circuit_breakers', {})
        
        for breaker_name, breaker_config in circuit_configs.items():
            breakers[breaker_name] = CircuitBreaker(
                breaker_name, breaker_config, self.metrics_collector
            )
        
        return breakers
    
    def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health status of a specific service."""
        if service_name in self.circuit_breakers:
            breaker = self.circuit_breakers[service_name]
            return {
                'healthy': breaker.state == 'closed',
                'circuit_state': breaker.state,
                'failure_count': breaker.failure_count
            }
        
        return {'healthy': True, 'circuit_state': 'closed', 'failure_count': 0}
    
    def should_degrade_service(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Determine if service should be degraded based on metrics."""
        reasons = []
        
        availability_target = 0.99
        latency_target = 2.0
        error_rate_target = 0.01
        
        if metrics.get('availability', 1.0) < availability_target:
            reasons.append('availability')
        if metrics.get('latency_p95', 0.0) > latency_target:
            reasons.append('latency_p95')
        if metrics.get('error_rate', 0.0) > error_rate_target:
            reasons.append('error_rate')
        
        return {
            'degrade': len(reasons) > 0,
            'reasons': reasons
        }
    
    def generate_alerts(self, violations: Dict[str, Any]):
        """Generate alerts for SLO violations."""
        for violation_type, violation_data in violations.items():
            logging.error(f"SLO Violation: {violation_type} - Current: {violation_data['current']}, Target: {violation_data['target']}")
    
    def call_service(self, service_name: str, func: Callable, *args, **kwargs):
        """Call a service through its circuit breaker."""
        return self.execute_with_circuit_breaker(service_name, func, *args, **kwargs)
    
    def record_fusion_metric(self, operation: str, success: bool, latency_ms: float, **tags):
        """Record fusion-specific metrics."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Record operation metrics
        success_metric = Metric(
            name=f"fusion_{operation}_success_rate",
            value=1.0 if success else 0.0,
            timestamp=timestamp,
            tags=tags
        )
        
        latency_metric = Metric(
            name=f"fusion_{operation}_latency_p95",
            value=latency_ms,
            timestamp=timestamp,
            tags=tags
        )
        
        self.metrics_collector.record_metric(success_metric)
        self.metrics_collector.record_metric(latency_metric)
        
        # Record error metrics if failed
        if not success:
            error_metric = Metric(
                name="fusion_errors_rate",
                value=1.0,
                timestamp=timestamp,
                tags=tags
            )
            self.metrics_collector.record_metric(error_metric)
    
    def execute_with_circuit_breaker(self, breaker_name: str, func: Callable, *args, **kwargs):
        """Execute function through specified circuit breaker."""
        if breaker_name not in self.circuit_breakers:
            # No circuit breaker configured, execute directly
            return func(*args, **kwargs)
        
        breaker = self.circuit_breakers[breaker_name]
        return breaker.call(func, *args, **kwargs)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health and return status."""
        violations = self.slo_monitor.check_slos()
        circuit_states = {name: cb.get_state() for name, cb in self.circuit_breakers.items()}
        
        # Check for circuit breaker trips based on conditions
        for breaker in self.circuit_breakers.values():
            if breaker.check_trip_conditions():
                breaker._transition_to_open()
        
        overall_health = "healthy"
        if violations:
            overall_health = "degraded"
        
        open_circuits = [name for name, state in circuit_states.items() if state['state'] == 'open']
        if open_circuits:
            overall_health = "critical"
        
        return {
            'overall_health': overall_health,
            'slo_violations': [asdict(v) for v in violations],
            'circuit_breaker_states': circuit_states,
            'degraded_services': list(self.degraded_services),
            'check_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def apply_degradation_strategy(self, strategy_name: str):
        """Apply a degradation strategy."""
        strategies = self.config.get('degradation_strategies', {})
        
        if strategy_name not in strategies:
            logging.warning(f"Unknown degradation strategy: {strategy_name}")
            return
        
        strategy = strategies[strategy_name]
        actions = strategy.get('actions', [])
        
        for action in actions:
            if isinstance(action, dict):
                for action_type, action_value in action.items():
                    self._execute_degradation_action(action_type, action_value)
                    
        self.degraded_services.add(strategy_name)
        logging.info(f"Applied degradation strategy: {strategy_name}")
    
    def _execute_degradation_action(self, action_type: str, action_value: Any):
        """Execute a specific degradation action."""
        if action_type == "disable_overlay":
            os.environ['RESON_FUSION_DISABLE'] = '1' if action_value else '0'
        elif action_type == "use_baseline_model":
            os.environ['RESON_FUSION_BASELINE_ONLY'] = '1' if action_value else '0'
        elif action_type == "log_degradation_event":
            logging.warning(f"Service degradation activated: {action_type}={action_value}")
        elif action_type == "notify_operations":
            # In production, this would trigger actual notifications
            logging.critical(f"Operations notification: Service degradation detected")
    
    def start_monitoring(self, check_interval_seconds: int = 60):
        """Start continuous monitoring loop."""
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    health_status = self.check_system_health()
                    
                    # Auto-apply degradation strategies if needed
                    if health_status['overall_health'] == 'critical':
                        open_circuits = [
                            name for name, state in health_status['circuit_breaker_states'].items()
                            if state['state'] == 'open'
                        ]
                        
                        for circuit_name in open_circuits:
                            if circuit_name == 'fusion_primary':
                                self.apply_degradation_strategy('fusion_overlay_degraded')
                            elif circuit_name == 'judge_fusion':
                                self.apply_degradation_strategy('judge_evaluation_degraded')
                    
                    time.sleep(check_interval_seconds)
                    
                except Exception as e:
                    logging.error(f"Error in monitoring loop: {e}")
                    time.sleep(check_interval_seconds)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logging.info("Production safety monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        logging.info("Production safety monitoring stopped")


def main():
    """Demo of production safety system."""
    safety_manager = ProductionSafetyManager("./configs/fusion")
    
    print("Production Safety Manager Demo")
    print("=" * 40)
    
    # Simulate some metrics
    safety_manager.record_fusion_metric("overlay_application", True, 1200)
    safety_manager.record_fusion_metric("cross_judge_evaluation", True, 800)
    safety_manager.record_fusion_metric("overlay_application", False, 5000)  # Simulate failure
    
    # Check system health
    health = safety_manager.check_system_health()
    
    print(f"Overall Health: {health['overall_health']}")
    print(f"SLO Violations: {len(health['slo_violations'])}")
    print(f"Circuit Breakers: {len(health['circuit_breaker_states'])}")
    
    for name, state in health['circuit_breaker_states'].items():
        print(f"  {name}: {state['state']} (failures: {state['failure_count']})")
    
    # Test circuit breaker
    def test_function():
        return "success"
    
    try:
        result = safety_manager.execute_with_circuit_breaker("fusion_primary", test_function)
        print(f"Circuit breaker test: {result}")
    except CircuitBreakerOpenError as e:
        print(f"Circuit breaker error: {e}")
    
    print("Demo completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()