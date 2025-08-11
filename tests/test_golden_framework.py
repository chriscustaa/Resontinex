#!/usr/bin/env python3
"""
Unit tests for Golden Test Framework
Validates golden test functionality and regression detection.
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from tests.golden.golden_test_framework import GoldenTestRunner, GoldenTestResult
except ImportError as e:
    print(f"Warning: Could not import golden test framework: {e}")


class TestGoldenFramework(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.golden_dir = Path(self.temp_dir) / "golden"
        self.config_dir = Path(self.temp_dir) / "config"
        
        # Create test directories
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock config file
        self.mock_scenarios = {
            'scenarios': [
                {
                    'id': 'test_scenario',
                    'category': 'financial_operations',
                    'prompt': 'Test prompt',
                    'context': 'Test context',
                    'expected_capabilities': {
                        'trust_scoring': 0.8,
                        'entropy_control': 0.75,
                        'insight_compression': 0.85
                    }
                }
            ]
        }
        
        with open(self.config_dir / "eval_scenarios.yaml", 'w') as f:
            import yaml
            yaml.dump(self.mock_scenarios, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_golden_runner_initialization(self):
        """Test that GoldenTestRunner initializes correctly."""
        try:
            runner = GoldenTestRunner(str(self.golden_dir), str(self.config_dir))
            self.assertEqual(len(runner.scenarios), 1)
            self.assertEqual(runner.scenarios[0]['id'], 'test_scenario')
        except NameError:
            self.skipTest("GoldenTestRunner not available")
    
    def test_config_hash_calculation(self):
        """Test configuration hash calculation."""
        try:
            runner = GoldenTestRunner(str(self.golden_dir), str(self.config_dir))
            hash1 = runner._calculate_config_hash()
            self.assertIsInstance(hash1, str)
            self.assertEqual(len(hash1), 16)  # Should be 16 character hash
        except NameError:
            self.skipTest("GoldenTestRunner not available")
    
    def test_golden_file_path_generation(self):
        """Test golden file path generation."""
        try:
            runner = GoldenTestRunner(str(self.golden_dir), str(self.config_dir))
            path = runner._get_golden_file_path('test_scenario')
            expected_path = self.golden_dir / "test_scenario_golden.json"
            self.assertEqual(path, expected_path)
        except NameError:
            self.skipTest("GoldenTestRunner not available")
    
    def test_scenario_execution(self):
        """Test scenario execution and result generation."""
        try:
            runner = GoldenTestRunner(str(self.golden_dir), str(self.config_dir))
            scenario = runner.scenarios[0]
            result = runner._execute_fusion_scenario(scenario)
            
            self.assertIsInstance(result, GoldenTestResult)
            self.assertEqual(result.scenario_id, 'test_scenario')
            self.assertIn('specificity', result.evaluation_scores)
            self.assertIn('execution_time_ms', result.execution_metrics)
            self.assertIn('contains_specific_steps', result.rules_validation)
            
        except NameError:
            self.skipTest("GoldenTestRunner not available")
    
    def test_golden_capture_and_validation(self):
        """Test golden test capture and validation workflow."""
        try:
            runner = GoldenTestRunner(str(self.golden_dir), str(self.config_dir))
            
            # Capture golden result
            results = runner.capture_golden_results(['test_scenario'])
            self.assertIn('test_scenario', results)
            
            golden_file = Path(results['test_scenario'])
            self.assertTrue(golden_file.exists())
            
            # Validate golden result data
            with open(golden_file, 'r') as f:
                golden_data = json.load(f)
            
            self.assertEqual(golden_data['scenario_id'], 'test_scenario')
            self.assertIn('evaluation_scores', golden_data)
            self.assertIn('rules_validation', golden_data)
            
            # Test validation against golden
            validation_results = runner.validate_against_golden(['test_scenario'])
            self.assertIn('test_scenario', validation_results)
            
            # Should pass validation since we just captured it
            result = validation_results['test_scenario']
            self.assertEqual(result['status'], 'pass')
            
        except NameError:
            self.skipTest("GoldenTestRunner not available")
    
    def test_rules_validation_logic(self):
        """Test rule validation logic."""
        try:
            runner = GoldenTestRunner(str(self.golden_dir), str(self.config_dir))
            
            # Test positive case
            good_response = "Here are the specific steps to implement: 1) First, analyze the situation because we need to understand the context. 2) Then, execute the process to achieve the desired outcome."
            rules = runner._validate_rules(good_response, {})
            
            self.assertTrue(rules['contains_specific_steps'])
            self.assertTrue(rules['includes_rationale'])
            self.assertTrue(rules['addresses_context'])
            self.assertTrue(rules['appropriate_length'])
            self.assertTrue(rules['operational_focus'])
            
            # Test negative case
            bad_response = "Yes."
            rules = runner._validate_rules(bad_response, {})
            
            self.assertFalse(rules['appropriate_length'])
            
        except NameError:
            self.skipTest("GoldenTestRunner not available")


class TestGoldenTestIntegration(unittest.TestCase):
    """Integration tests for golden test system."""
    
    def test_existing_golden_files_validation(self):
        """Test validation against existing golden files."""
        try:
            # Use actual golden directory and config
            golden_dir = "./tests/golden"
            config_dir = "./configs/fusion"
            
            if not os.path.exists(golden_dir) or not os.path.exists(config_dir):
                self.skipTest("Golden or config directories not found")
            
            runner = GoldenTestRunner(golden_dir, config_dir)
            
            # Find existing golden files
            golden_files = list(Path(golden_dir).glob("*_golden.json"))
            if not golden_files:
                self.skipTest("No existing golden files found")
            
            scenario_ids = [f.stem.replace('_golden', '') for f in golden_files]
            
            # Test validation
            validation_results = runner.validate_against_golden(scenario_ids)
            
            self.assertGreater(len(validation_results), 0)
            
            # Print results for debugging
            for scenario_id, result in validation_results.items():
                print(f"Golden validation {scenario_id}: {result.get('status', 'unknown')}")
                if result.get('differences'):
                    print(f"  Differences: {result['differences']}")
            
        except NameError:
            self.skipTest("GoldenTestRunner not available")


if __name__ == '__main__':
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    
    # Save test results
    with open('test_results/golden_framework_results.json', 'w') as f:
        json.dump({
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'test_categories': [
                'golden_framework',
                'scenario_execution', 
                'validation_logic',
                'integration_tests'
            ]
        }, f, indent=2)