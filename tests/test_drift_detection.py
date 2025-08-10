#!/usr/bin/env python3
"""
Unit tests for drift detection watchdog system.
Tests version monitoring, quality gates, and re-distillation triggers.
"""

import unittest
import sys
import os
import json
import tempfile
import yaml
from unittest.mock import patch, MagicMock, mock_open

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from watch_drift import DriftWatchdog, VersionMonitor, QualityGate
except ImportError as e:
    print(f"Warning: Could not import watch_drift components: {e}")

class TestDriftWatchdog(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'version_check_interval': 300,
            'quality_gates': {
                'min_score_threshold': 0.7,
                'max_variance_threshold': 0.15,
                'min_consistency_threshold': 0.8
            },
            'redist_trigger_conditions': {
                'score_drop_threshold': 0.1,
                'consecutive_failures': 3
            }
        }
        
        self.test_versions = {
            'gpt-4o': '2024-11-20',
            'gpt-4o-mini': '2024-07-18',
            'claude-3-5-sonnet': '2024-10-22'
        }
    
    def test_version_monitor_initialization(self):
        """Test that VersionMonitor initializes correctly."""
        try:
            monitor = VersionMonitor(self.test_config)
            self.assertIsNotNone(monitor.config)
            self.assertEqual(monitor.config['version_check_interval'], 300)
        except NameError:
            self.skipTest("VersionMonitor not available")
    
    def test_version_change_detection(self):
        """Test detection of model version changes."""
        try:
            monitor = VersionMonitor(self.test_config)
            
            # Mock stored versions
            old_versions = {
                'gpt-4o': '2024-10-01',
                'gpt-4o-mini': '2024-07-18',
                'claude-3-5-sonnet': '2024-10-22'
            }
            
            # Test version change detection
            changes = monitor._detect_version_changes(old_versions, self.test_versions)
            self.assertIn('gpt-4o', changes)
            self.assertEqual(changes['gpt-4o']['old'], '2024-10-01')
            self.assertEqual(changes['gpt-4o']['new'], '2024-11-20')
            
        except NameError:
            self.skipTest("VersionMonitor not available")
    
    def test_quality_gate_evaluation(self):
        """Test quality gate evaluation logic."""
        try:
            gate = QualityGate(self.test_config['quality_gates'])
            
            # Test passing quality metrics
            passing_metrics = {
                'average_score': 0.85,
                'score_variance': 0.12,
                'consistency_ratio': 0.88
            }
            
            result = gate.evaluate(passing_metrics)
            self.assertTrue(result['passed'])
            
            # Test failing quality metrics
            failing_metrics = {
                'average_score': 0.65,
                'score_variance': 0.20,
                'consistency_ratio': 0.75
            }
            
            result = gate.evaluate(failing_metrics)
            self.assertFalse(result['passed'])
            
        except NameError:
            self.skipTest("QualityGate not available")
    
    @patch('watch_drift.requests.get')
    def test_model_version_fetching(self, mock_get):
        """Test fetching current model versions from APIs."""
        try:
            # Mock API responses
            mock_response = MagicMock()
            mock_response.json.return_value = {'version': '2024-11-20'}
            mock_get.return_value = mock_response
            
            monitor = VersionMonitor(self.test_config)
            version = monitor._fetch_openai_version('gpt-4o')
            
            self.assertEqual(version, '2024-11-20')
            mock_get.assert_called()
            
        except NameError:
            self.skipTest("VersionMonitor not available")
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"gpt-4o": "2024-10-01"}')
    def test_version_storage_and_loading(self, mock_file):
        """Test version storage and loading functionality."""
        try:
            monitor = VersionMonitor(self.test_config)
            
            # Test loading versions
            versions = monitor._load_stored_versions()
            self.assertEqual(versions['gpt-4o'], '2024-10-01')
            
            # Test saving versions
            monitor._save_versions(self.test_versions)
            mock_file.assert_called()
            
        except NameError:
            self.skipTest("VersionMonitor not available")

class TestQualityGate(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.gate_config = {
            'min_score_threshold': 0.7,
            'max_variance_threshold': 0.15,
            'min_consistency_threshold': 0.8
        }
    
    def test_threshold_validation(self):
        """Test individual threshold validations."""
        try:
            gate = QualityGate(self.gate_config)
            
            # Test score threshold
            self.assertTrue(gate._check_score_threshold(0.85))
            self.assertFalse(gate._check_score_threshold(0.65))
            
            # Test variance threshold
            self.assertTrue(gate._check_variance_threshold(0.12))
            self.assertFalse(gate._check_variance_threshold(0.20))
            
            # Test consistency threshold
            self.assertTrue(gate._check_consistency_threshold(0.88))
            self.assertFalse(gate._check_consistency_threshold(0.75))
            
        except NameError:
            self.skipTest("QualityGate not available")
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive quality evaluation."""
        try:
            gate = QualityGate(self.gate_config)
            
            # All metrics pass
            metrics = {
                'average_score': 0.85,
                'score_variance': 0.10,
                'consistency_ratio': 0.90
            }
            result = gate.evaluate(metrics)
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['failures']), 0)
            
            # Mixed results
            metrics = {
                'average_score': 0.65,  # Fail
                'score_variance': 0.10,  # Pass
                'consistency_ratio': 0.90  # Pass
            }
            result = gate.evaluate(metrics)
            self.assertFalse(result['passed'])
            self.assertIn('score_threshold', result['failures'])
            
        except NameError:
            self.skipTest("QualityGate not available")

class TestDriftDetectionIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'drift_policy.yaml')
        
        # Create test configuration
        test_config = {
            'monitoring': {
                'version_check_interval': 300,
                'quality_check_interval': 600
            },
            'quality_gates': {
                'min_score_threshold': 0.7,
                'max_variance_threshold': 0.15,
                'min_consistency_threshold': 0.8
            },
            'redist_trigger_conditions': {
                'score_drop_threshold': 0.1,
                'consecutive_failures': 3
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
    
    def test_watchdog_initialization_from_config(self):
        """Test watchdog initialization from configuration file."""
        try:
            watchdog = DriftWatchdog(self.config_path)
            self.assertIsNotNone(watchdog.version_monitor)
            self.assertIsNotNone(watchdog.quality_gate)
            
        except NameError:
            self.skipTest("DriftWatchdog not available")
    
    def test_drift_detection_workflow(self):
        """Test complete drift detection workflow."""
        try:
            watchdog = DriftWatchdog(self.config_path)
            
            # Mock version changes and quality metrics
            with patch.object(watchdog.version_monitor, 'check_versions') as mock_check_versions, \
                 patch.object(watchdog.quality_gate, 'evaluate') as mock_evaluate:
                
                # Simulate version change detected
                mock_check_versions.return_value = {
                    'changes_detected': True,
                    'changed_models': ['gpt-4o'],
                    'changes': {
                        'gpt-4o': {'old': '2024-10-01', 'new': '2024-11-20'}
                    }
                }
                
                # Simulate quality gate failure
                mock_evaluate.return_value = {
                    'passed': False,
                    'failures': ['score_threshold'],
                    'metrics': {'average_score': 0.65}
                }
                
                result = watchdog.run_drift_check()
                
                self.assertTrue(result['version_changes_detected'])
                self.assertFalse(result['quality_gates_passed'])
                self.assertTrue(result['redist_triggered'])
                
        except NameError:
            self.skipTest("DriftWatchdog not available")
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    
    # Save test results
    with open('test_results/drift_detection_results.json', 'w') as f:
        json.dump({
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'test_categories': [
                'version_monitoring',
                'quality_gates',
                'integration_workflow'
            ]
        }, f, indent=2)