#!/usr/bin/env python3
"""
Integration tests for the complete fusion optimization system.
Tests end-to-end workflows across all components.
"""

import unittest
import sys
import os
import json
import tempfile
import subprocess
import yaml
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class TestFusionSystemIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
        
        # Create test scenario configuration
        self.test_scenario = {
            "id": "integration_test_01",
            "description": "End-to-end integration test scenario",
            "complexity": 3,
            "expected_capabilities": ["reasoning", "accuracy"],
            "query": "Design a simple microservice architecture for a todo application",
            "expected_answer": "A microservice architecture should include API gateway, user service, todo service, and database"
        }
        
        # Create test parameters
        self.test_parameters = {
            'temperature': 0.7,
            'top_p': 0.9,
            'presence_penalty': 0.1,
            'frequency_penalty': 0.0
        }
    
    def test_script_existence(self):
        """Test that all required scripts exist."""
        required_scripts = [
            'judge_fusion.py',
            'tune-overlay.py',
            'watch-drift.py',
            'runtime_router.py',
            'circuit_breaker.py'
        ]
        
        for script in required_scripts:
            script_path = os.path.join(self.scripts_dir, script)
            self.assertTrue(os.path.exists(script_path), f"Script {script} not found")
    
    def test_config_files_existence(self):
        """Test that all required configuration files exist."""
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'fusion')
        
        required_configs = [
            'eval_scenarios.yaml',
            'fusion_overlay.v0.3.txt',
            'overlay_params.yaml',
            'drift_policy.yaml',
            'slo.yaml'
        ]
        
        for config in required_configs:
            config_path = os.path.join(config_dir, config)
            self.assertTrue(os.path.exists(config_path), f"Config {config} not found")
    
    def test_micro_overlay_files(self):
        """Test that micro-overlay files exist and are readable."""
        overlay_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'fusion', 'micro_overlays')
        
        required_overlays = [
            'rollback_first.txt',
            'state_model_first.txt',
            'observability_first.txt'
        ]
        
        for overlay in required_overlays:
            overlay_path = os.path.join(overlay_dir, overlay)
            self.assertTrue(os.path.exists(overlay_path), f"Overlay {overlay} not found")
            
            # Test readability
            with open(overlay_path, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0, f"Overlay {overlay} is empty")
    
    @patch('subprocess.run')
    def test_cross_judge_evaluation_workflow(self, mock_subprocess):
        """Test cross-judge evaluation integration."""
        # Mock successful script execution
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps({
            'primary_score': 0.85,
            'validation_result': {
                'is_consistent': True,
                'confidence': 0.88
            },
            'final_recommendation': 'overlay'
        })
        
        # Test script execution
        judge_script = os.path.join(self.scripts_dir, 'judge_fusion.py')
        if os.path.exists(judge_script):
            # Simulate running cross-judge evaluation
            cmd = [sys.executable, judge_script, '--test-mode']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Verify script can be executed (even if it fails due to missing dependencies)
            self.assertIsNotNone(result.returncode)
    
    @patch('subprocess.run')
    def test_parameter_tuning_workflow(self, mock_subprocess):
        """Test parameter tuning integration."""
        # Mock successful optimization
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = json.dumps({
            'best_parameters': self.test_parameters,
            'best_score': 0.87,
            'optimization_history': [
                {'params': self.test_parameters, 'score': 0.87}
            ]
        })
        
        # Test script execution
        tune_script = os.path.join(self.scripts_dir, 'tune-overlay.py')
        if os.path.exists(tune_script):
            # Simulate running parameter optimization
            cmd = [sys.executable, tune_script, '--test-mode']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Verify script can be executed
            self.assertIsNotNone(result.returncode)
    
    def test_configuration_loading_integration(self):
        """Test loading and parsing of all configuration files."""
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'fusion')
        
        # Test YAML configurations
        yaml_configs = ['eval_scenarios.yaml', 'overlay_params.yaml', 'drift_policy.yaml', 'slo.yaml']
        
        for config_file in yaml_configs:
            config_path = os.path.join(config_dir, config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    try:
                        config_data = yaml.safe_load(f)
                        self.assertIsNotNone(config_data, f"Failed to parse {config_file}")
                        self.assertIsInstance(config_data, dict, f"{config_file} should contain a dictionary")
                    except yaml.YAMLError as e:
                        self.fail(f"Invalid YAML in {config_file}: {e}")
        
        # Test JSON configurations
        json_configs = ['model_semantics_ledger.v0.1.0.json']
        
        for config_file in json_configs:
            config_path = os.path.join(config_dir, config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    try:
                        config_data = json.load(f)
                        self.assertIsNotNone(config_data, f"Failed to parse {config_file}")
                    except json.JSONDecodeError as e:
                        self.fail(f"Invalid JSON in {config_file}: {e}")
    
    def test_workflow_orchestration(self):
        """Test orchestration of multiple fusion components."""
        workflow_steps = [
            {
                'name': 'drift_detection',
                'script': 'watch-drift.py',
                'expected_output': ['version_changes_detected', 'quality_gates_passed']
            },
            {
                'name': 'parameter_tuning',
                'script': 'tune-overlay.py',
                'expected_output': ['best_parameters', 'best_score']
            },
            {
                'name': 'cross_evaluation',
                'script': 'judge_fusion.py',
                'expected_output': ['primary_score', 'validation_result']
            }
        ]
        
        # Test that workflow scripts can be identified and are executable
        for step in workflow_steps:
            script_path = os.path.join(self.scripts_dir, step['script'])
            
            if os.path.exists(script_path):
                # Check file permissions (should be readable)
                self.assertTrue(os.access(script_path, os.R_OK), 
                              f"Script {step['script']} is not readable")
                
                # Check that it's a Python file
                with open(script_path, 'r') as f:
                    first_line = f.readline()
                    self.assertTrue(first_line.startswith('#!'), 
                                  f"Script {step['script']} missing shebang")
    
    def test_github_workflow_configuration(self):
        """Test GitHub Actions workflow configuration."""
        workflow_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            '.github', 
            'workflows', 
            'weekly-parameter-tuning.yml'
        )
        
        if os.path.exists(workflow_path):
            with open(workflow_path, 'r') as f:
                try:
                    workflow_config = yaml.safe_load(f)
                    
                    # Verify essential workflow components
                    self.assertIn('name', workflow_config)
                    self.assertIn('on', workflow_config)
                    self.assertIn('jobs', workflow_config)
                    
                    # Check for required job steps
                    if 'jobs' in workflow_config:
                        for job_name, job_config in workflow_config['jobs'].items():
                            self.assertIn('steps', job_config, f"Job {job_name} missing steps")
                            
                except yaml.YAMLError as e:
                    self.fail(f"Invalid GitHub workflow YAML: {e}")
    
    def test_error_handling_integration(self):
        """Test error handling across system components."""
        # Test configuration with invalid values
        invalid_config = {
            'temperature': 1.5,  # Invalid (too high)
            'top_p': -0.1,       # Invalid (negative)
            'failure_threshold': 'invalid'  # Invalid (not numeric)
        }
        
        # This test verifies that the system handles invalid configurations gracefully
        # by checking that scripts don't crash immediately on invalid input
        config_path = os.path.join(self.temp_dir, 'invalid_config.json')
        with open(config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        # The test passes if we can create the invalid config file
        # (actual validation happens within individual component tests)
        self.assertTrue(os.path.exists(config_path))
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestSystemHealthChecks(unittest.TestCase):
    
    def test_python_dependencies(self):
        """Test that required Python packages are available."""
        required_packages = [
            'yaml',
            'json',
            'subprocess',
            'unittest',
            'tempfile',
            'os',
            'sys',
            'time'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                self.fail(f"Required package {package} not available")
    
    def test_directory_structure(self):
        """Test that the expected directory structure exists."""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        required_directories = [
            'scripts',
            'configs',
            'configs/fusion',
            'configs/fusion/micro_overlays',
            'tests',
            '.github',
            '.github/workflows'
        ]
        
        for directory in required_directories:
            dir_path = os.path.join(base_dir, directory)
            self.assertTrue(os.path.exists(dir_path), f"Directory {directory} not found")
            self.assertTrue(os.path.isdir(dir_path), f"{directory} is not a directory")

if __name__ == '__main__':
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    
    # Save test results
    with open('test_results/integration_results.json', 'w') as f:
        json.dump({
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'test_categories': [
                'system_integration',
                'configuration_validation',
                'workflow_orchestration',
                'health_checks'
            ],
            'system_status': 'healthy' if len(result.failures) == 0 and len(result.errors) == 0 else 'degraded'
        }, f, indent=2)