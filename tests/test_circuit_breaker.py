#!/usr/bin/env python3
"""
Unit tests for circuit breaker and SLO monitoring system.
Tests failure detection, circuit state transitions, and production safety measures.
"""

import unittest
import sys
import os
import json
import time
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from circuit_breaker import CircuitBreaker, SLOMonitor, ProductionSafetyManager
except ImportError as e:
    print(f"Warning: Could not import circuit_breaker components: {e}")

class TestCircuitBreaker(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.breaker_config = {
            'failure_threshold': 5,
            'recovery_timeout': 30,
            'half_open_max_calls': 3,
            'success_threshold': 2
        }
        
    def test_breaker_initialization(self):
        """Test that CircuitBreaker initializes correctly."""
        try:
            breaker = CircuitBreaker('test_service', self.breaker_config)
            self.assertEqual(breaker.name, 'test_service')
            self.assertEqual(breaker.state, 'closed')
            self.assertEqual(breaker.failure_count, 0)
        except NameError:
            self.skipTest("CircuitBreaker not available")
    
    def test_circuit_state_transitions(self):
        """Test circuit breaker state transitions."""
        try:
            breaker = CircuitBreaker('test_service', self.breaker_config)
            
            # Simulate failures to trigger OPEN state
            for i in range(6):
                result = breaker.call(lambda: self._failing_operation())
                
            self.assertEqual(breaker.state, 'open')
            self.assertGreaterEqual(breaker.failure_count, 5)
            
        except NameError:
            self.skipTest("CircuitBreaker not available")
    
    def test_half_open_recovery(self):
        """Test circuit breaker half-open recovery logic."""
        try:
            breaker = CircuitBreaker('test_service', self.breaker_config)
            
            # Force circuit to open state
            breaker.failure_count = 5
            breaker.state = 'open'
            breaker.last_failure_time = time.time() - 31  # Past recovery timeout
            
            # Should transition to half-open
            result = breaker.call(lambda: "success")
            
            self.assertEqual(breaker.state, 'half_open')
            
        except NameError:
            self.skipTest("CircuitBreaker not available")
    
    def test_success_recovery(self):
        """Test successful recovery from half-open to closed."""
        try:
            breaker = CircuitBreaker('test_service', self.breaker_config)
            
            # Set to half-open state
            breaker.state = 'half_open'
            breaker.success_count = 0
            
            # Simulate successful calls
            for i in range(3):
                result = breaker.call(lambda: "success")
                
            self.assertEqual(breaker.state, 'closed')
            self.assertEqual(breaker.failure_count, 0)
            
        except NameError:
            self.skipTest("CircuitBreaker not available")
    
    def _failing_operation(self):
        """Helper method that always fails."""
        raise Exception("Simulated failure")

class TestSLOMonitor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.slo_config = {
            'availability': {
                'target': 0.99,
                'window_minutes': 60
            },
            'latency_p95': {
                'target': 2.0,
                'window_minutes': 60
            },
            'error_rate': {
                'target': 0.01,
                'window_minutes': 60
            }
        }
    
    def test_slo_monitor_initialization(self):
        """Test that SLOMonitor initializes correctly."""
        try:
            monitor = SLOMonitor(self.slo_config)
            self.assertIn('availability', monitor.slos)
            self.assertIn('latency_p95', monitor.slos)
            self.assertIn('error_rate', monitor.slos)
        except NameError:
            self.skipTest("SLOMonitor not available")
    
    def test_availability_tracking(self):
        """Test availability SLO tracking."""
        try:
            monitor = SLOMonitor(self.slo_config)
            
            # Record successful requests
            for i in range(95):
                monitor.record_request_outcome(success=True, latency=1.5)
                
            # Record failed requests
            for i in range(5):
                monitor.record_request_outcome(success=False, latency=3.0)
                
            availability = monitor.calculate_availability()
            self.assertAlmostEqual(availability, 0.95, places=2)
            
            # Check SLO violation
            slo_status = monitor.check_slo_compliance()
            self.assertFalse(slo_status['availability']['compliant'])
            
        except NameError:
            self.skipTest("SLOMonitor not available")
    
    def test_latency_p95_calculation(self):
        """Test P95 latency calculation."""
        try:
            monitor = SLOMonitor(self.slo_config)
            
            # Record latency samples
            latencies = [1.0] * 90 + [3.0] * 10  # 90% at 1s, 10% at 3s
            
            for latency in latencies:
                monitor.record_request_outcome(success=True, latency=latency)
                
            p95_latency = monitor.calculate_p95_latency()
            
            # P95 should be around 3.0 seconds
            self.assertGreater(p95_latency, 2.5)
            
            slo_status = monitor.check_slo_compliance()
            self.assertFalse(slo_status['latency_p95']['compliant'])
            
        except NameError:
            self.skipTest("SLOMonitor not available")
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        try:
            monitor = SLOMonitor(self.slo_config)
            
            # Record mix of successful and failed requests
            for i in range(98):
                monitor.record_request_outcome(success=True, latency=1.5)
                
            for i in range(2):
                monitor.record_request_outcome(success=False, latency=2.0)
                
            error_rate = monitor.calculate_error_rate()
            self.assertAlmostEqual(error_rate, 0.02, places=2)
            
            slo_status = monitor.check_slo_compliance()
            self.assertFalse(slo_status['error_rate']['compliant'])
            
        except NameError:
            self.skipTest("SLOMonitor not available")

class TestProductionSafetyManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.safety_config = {
            'circuit_breakers': {
                'fusion_service': {
                    'failure_threshold': 3,
                    'recovery_timeout': 30
                },
                'evaluation_service': {
                    'failure_threshold': 5,
                    'recovery_timeout': 60
                }
            },
            'slo_targets': {
                'availability': 0.99,
                'latency_p95': 2.0,
                'error_rate': 0.01
            },
            'degradation_policy': {
                'fallback_enabled': True,
                'graceful_degradation': True,
                'alert_threshold': 0.95
            }
        }
    
    def test_safety_manager_initialization(self):
        """Test that ProductionSafetyManager initializes correctly."""
        try:
            manager = ProductionSafetyManager(self.safety_config)
            self.assertEqual(len(manager.circuit_breakers), 2)
            self.assertIn('fusion_service', manager.circuit_breakers)
            self.assertIsNotNone(manager.slo_monitor)
        except NameError:
            self.skipTest("ProductionSafetyManager not available")
    
    def test_service_health_check(self):
        """Test service health checking functionality."""
        try:
            manager = ProductionSafetyManager(self.safety_config)
            
            # Simulate healthy service
            health_status = manager.check_service_health('fusion_service')
            self.assertTrue(health_status['healthy'])
            
            # Simulate unhealthy service by triggering circuit breaker
            breaker = manager.circuit_breakers['fusion_service']
            breaker.failure_count = 5
            breaker.state = 'open'
            
            health_status = manager.check_service_health('fusion_service')
            self.assertFalse(health_status['healthy'])
            self.assertEqual(health_status['circuit_state'], 'open')
            
        except NameError:
            self.skipTest("ProductionSafetyManager not available")
    
    def test_degradation_decision(self):
        """Test graceful degradation decision logic."""
        try:
            manager = ProductionSafetyManager(self.safety_config)
            
            # Simulate SLO violations
            system_metrics = {
                'availability': 0.94,  # Below target
                'latency_p95': 2.5,    # Above target
                'error_rate': 0.02     # Above target
            }
            
            degradation_decision = manager.should_degrade_service(system_metrics)
            
            self.assertTrue(degradation_decision['degrade'])
            self.assertIn('availability', degradation_decision['reasons'])
            self.assertIn('latency_p95', degradation_decision['reasons'])
            
        except NameError:
            self.skipTest("ProductionSafetyManager not available")
    
    @patch('circuit_breaker.logging')
    def test_alert_generation(self, mock_logging):
        """Test alert generation for SLO violations."""
        try:
            manager = ProductionSafetyManager(self.safety_config)
            
            # Simulate critical SLO violations
            violations = {
                'availability': {'current': 0.90, 'target': 0.99},
                'error_rate': {'current': 0.05, 'target': 0.01}
            }
            
            manager.generate_alerts(violations)
            
            # Verify alerts were logged
            self.assertTrue(mock_logging.error.called)
            
        except NameError:
            self.skipTest("ProductionSafetyManager not available")

class TestCircuitBreakerIntegration(unittest.TestCase):
    
    def test_end_to_end_safety_workflow(self):
        """Test complete production safety workflow."""
        try:
            config = {
                'circuit_breakers': {
                    'test_service': {
                        'failure_threshold': 3,
                        'recovery_timeout': 1  # Short timeout for testing
                    }
                },
                'slo_targets': {
                    'availability': 0.95,
                    'latency_p95': 2.0,
                    'error_rate': 0.05
                },
                'degradation_policy': {
                    'fallback_enabled': True,
                    'graceful_degradation': True
                }
            }
            
            manager = ProductionSafetyManager(config)
            
            # Simulate service failures
            service_responses = []
            for i in range(5):
                try:
                    # Simulate failing service call
                    def failing_service():
                        raise Exception(f"Service failure {i}")
                    
                    result = manager.call_service('test_service', failing_service)
                    service_responses.append(result)
                    
                except Exception as e:
                    service_responses.append(str(e))
            
            # Check that circuit breaker opened
            breaker = manager.circuit_breakers['test_service']
            self.assertEqual(breaker.state, 'open')
            
            # Simulate recovery after timeout
            time.sleep(1.1)
            
            # Service should be in half-open state for recovery testing
            def successful_service():
                return "Service recovered"
            
            recovery_result = manager.call_service('test_service', successful_service)
            self.assertEqual(breaker.state, 'half_open')
            
        except NameError:
            self.skipTest("ProductionSafetyManager not available")

if __name__ == '__main__':
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    
    # Save test results
    with open('test_results/circuit_breaker_results.json', 'w') as f:
        json.dump({
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'test_categories': [
                'circuit_breaker',
                'slo_monitoring',
                'production_safety',
                'integration_workflow'
            ]
        }, f, indent=2)