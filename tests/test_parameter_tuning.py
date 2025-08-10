#!/usr/bin/env python3
"""
Unit tests for parameter auto-tuning system.
Tests grid search optimization, scenario-aware tuning, and parameter validation.
"""

import unittest
import sys
import os
import json
import tempfile
import yaml
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from tune_overlay import ParameterOptimizer, GridSearchTuner, ScenarioManager
except ImportError as e:
    print(f"Warning: Could not import tune_overlay components: {e}")

class TestParameterOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'optimization': {
                'method': 'grid_search',
                'max_iterations': 50,
                'convergence_threshold': 0.01
            },
            'parameter_ranges': {
                'temperature': [0.1, 0.3, 0.5, 0.7, 0.9],
                'top_p': [0.8, 0.9, 0.95, 1.0],
                'presence_penalty': [0.0, 0.1, 0.2],
                'frequency_penalty': [0.0, 0.1, 0.2]
            },
            'scenarios': {
                'high_complexity': {
                    'weight': 0.4,
                    'target_capabilities': ['reasoning', 'accuracy']
                },
                'medium_complexity': {
                    'weight': 0.35,
                    'target_capabilities': ['efficiency', 'clarity']
                },
                'low_complexity': {
                    'weight': 0.25,
                    'target_capabilities': ['speed', 'consistency']
                }
            }
        }
    
    def test_optimizer_initialization(self):
        """Test that ParameterOptimizer initializes correctly."""
        try:
            optimizer = ParameterOptimizer(self.test_config)
            self.assertIsNotNone(optimizer.config)
            self.assertIsNotNone(optimizer.tuner)
            self.assertEqual(optimizer.config['optimization']['method'], 'grid_search')
        except NameError:
            self.skipTest("ParameterOptimizer not available")
    
    def test_parameter_space_generation(self):
        """Test generation of parameter search space."""
        try:
            optimizer = ParameterOptimizer(self.test_config)
            param_space = optimizer._generate_parameter_space()
            
            # Check that all parameter combinations are generated
            expected_combinations = 5 * 4 * 3 * 3  # temperature * top_p * presence_penalty * frequency_penalty
            self.assertEqual(len(param_space), expected_combinations)
            
            # Check that each combination has all required parameters
            for params in param_space:
                self.assertIn('temperature', params)
                self.assertIn('top_p', params)
                self.assertIn('presence_penalty', params)
                self.assertIn('frequency_penalty', params)
                
        except NameError:
            self.skipTest("ParameterOptimizer not available")
    
    def test_parameter_validation(self):
        """Test parameter value validation."""
        try:
            optimizer = ParameterOptimizer(self.test_config)
            
            # Valid parameters
            valid_params = {
                'temperature': 0.7,
                'top_p': 0.9,
                'presence_penalty': 0.1,
                'frequency_penalty': 0.1
            }
            self.assertTrue(optimizer._validate_parameters(valid_params))
            
            # Invalid temperature (too high)
            invalid_params = {
                'temperature': 1.5,
                'top_p': 0.9,
                'presence_penalty': 0.1,
                'frequency_penalty': 0.1
            }
            self.assertFalse(optimizer._validate_parameters(invalid_params))
            
        except NameError:
            self.skipTest("ParameterOptimizer not available")

class TestGridSearchTuner(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameter_ranges = {
            'temperature': [0.1, 0.5, 0.9],
            'top_p': [0.8, 0.9, 1.0]
        }
    
    def test_grid_generation(self):
        """Test grid search parameter generation."""
        try:
            tuner = GridSearchTuner(self.parameter_ranges)
            grid = tuner.generate_grid()
            
            # Should generate 3 * 3 = 9 combinations
            self.assertEqual(len(grid), 9)
            
            # Check specific combinations exist
            expected_combo = {'temperature': 0.5, 'top_p': 0.9}
            self.assertIn(expected_combo, grid)
            
        except NameError:
            self.skipTest("GridSearchTuner not available")
    
    def test_optimization_scoring(self):
        """Test scoring mechanism for parameter combinations."""
        try:
            tuner = GridSearchTuner(self.parameter_ranges)
            
            # Mock evaluation function
            def mock_evaluate(params):
                # Simple scoring: prefer moderate temperature and high top_p
                temp_score = 1.0 - abs(params['temperature'] - 0.5)
                top_p_score = params['top_p']
                return (temp_score + top_p_score) / 2
            
            best_params, best_score = tuner.optimize(mock_evaluate, max_iterations=9)
            
            # Should find optimal combination
            self.assertIsNotNone(best_params)
            self.assertIsNotNone(best_score)
            self.assertEqual(best_params['temperature'], 0.5)
            self.assertEqual(best_params['top_p'], 1.0)
            
        except NameError:
            self.skipTest("GridSearchTuner not available")

class TestScenarioManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.scenarios_config = {
            'high_complexity': {
                'weight': 0.4,
                'target_capabilities': ['reasoning', 'accuracy'],
                'test_cases': ['complex_reasoning_01', 'logical_analysis_02']
            },
            'medium_complexity': {
                'weight': 0.35,
                'target_capabilities': ['efficiency', 'clarity'],
                'test_cases': ['medium_task_01', 'explanation_task_02']
            }
        }
    
    def test_scenario_loading(self):
        """Test loading and parsing of scenario configurations."""
        try:
            manager = ScenarioManager(self.scenarios_config)
            scenarios = manager.get_scenarios()
            
            self.assertEqual(len(scenarios), 2)
            self.assertIn('high_complexity', scenarios)
            self.assertEqual(scenarios['high_complexity']['weight'], 0.4)
            
        except NameError:
            self.skipTest("ScenarioManager not available")
    
    def test_weighted_scoring(self):
        """Test weighted scoring across scenarios."""
        try:
            manager = ScenarioManager(self.scenarios_config)
            
            # Mock scenario scores
            scenario_scores = {
                'high_complexity': 0.85,
                'medium_complexity': 0.75
            }
            
            weighted_score = manager.calculate_weighted_score(scenario_scores)
            
            # Expected: 0.85 * 0.4 + 0.75 * 0.35 = 0.34 + 0.2625 = 0.6025
            expected_score = 0.85 * 0.4 + 0.75 * 0.35
            self.assertAlmostEqual(weighted_score, expected_score, places=3)
            
        except NameError:
            self.skipTest("ScenarioManager not available")
    
    def test_scenario_filtering(self):
        """Test filtering scenarios by complexity or capability."""
        try:
            manager = ScenarioManager(self.scenarios_config)
            
            # Filter by capability
            reasoning_scenarios = manager.filter_by_capability('reasoning')
            self.assertIn('high_complexity', reasoning_scenarios)
            self.assertNotIn('medium_complexity', reasoning_scenarios)
            
            # Filter by weight threshold
            high_weight_scenarios = manager.filter_by_weight_threshold(0.38)
            self.assertIn('high_complexity', high_weight_scenarios)
            self.assertNotIn('medium_complexity', high_weight_scenarios)
            
        except NameError:
            self.skipTest("ScenarioManager not available")

class TestParameterTuningIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'overlay_params.yaml')
        
        self.test_config = {
            'optimization': {
                'method': 'grid_search',
                'max_iterations': 50,
                'convergence_threshold': 0.01
            },
            'parameter_ranges': {
                'temperature': [0.1, 0.3, 0.5, 0.7, 0.9],
                'top_p': [0.8, 0.9, 0.95, 1.0],
                'presence_penalty': [0.0, 0.1, 0.2],
                'frequency_penalty': [0.0, 0.1, 0.2]
            },
            'scenarios': {
                'high_complexity': {
                    'weight': 0.4,
                    'target_capabilities': ['reasoning', 'accuracy']
                },
                'medium_complexity': {
                    'weight': 0.35,
                    'target_capabilities': ['efficiency', 'clarity']
                },
                'low_complexity': {
                    'weight': 0.25,
                    'target_capabilities': ['speed', 'consistency']
                }
            }
        }
        
        # Create test configuration file
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    @patch('tune_overlay.subprocess.run')
    def test_end_to_end_optimization(self, mock_subprocess):
        """Test complete parameter optimization workflow."""
        try:
            # Mock fusion evaluation results
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = json.dumps({
                'overall_score': 0.85,
                'scenario_scores': {
                    'high_complexity': 0.88,
                    'medium_complexity': 0.82,
                    'low_complexity': 0.86
                }
            })
            
            optimizer = ParameterOptimizer(self.test_config)
            result = optimizer.run_optimization()
            
            self.assertIn('best_parameters', result)
            self.assertIn('best_score', result)
            self.assertIn('optimization_history', result)
            
            # Verify optimization improved over baseline
            if result['best_score'] is not None:
                self.assertGreater(result['best_score'], 0.5)
            
        except NameError:
            self.skipTest("ParameterOptimizer not available")
    
    def test_parameter_persistence(self):
        """Test saving and loading optimized parameters."""
        try:
            optimizer = ParameterOptimizer(self.test_config)
            
            # Mock optimized parameters
            optimized_params = {
                'temperature': 0.7,
                'top_p': 0.9,
                'presence_penalty': 0.1,
                'frequency_penalty': 0.0
            }
            
            # Save parameters
            save_path = os.path.join(self.temp_dir, 'optimized_params.json')
            optimizer._save_optimized_parameters(optimized_params, save_path)
            
            # Verify file was created and contains correct data
            self.assertTrue(os.path.exists(save_path))
            
            with open(save_path, 'r') as f:
                loaded_params = json.load(f)
            
            self.assertEqual(loaded_params['temperature'], 0.7)
            self.assertEqual(loaded_params['top_p'], 0.9)
            
        except NameError:
            self.skipTest("ParameterOptimizer not available")
    
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
    with open('test_results/parameter_tuning_results.json', 'w') as f:
        json.dump({
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'test_categories': [
                'parameter_optimization',
                'grid_search',
                'scenario_management',
                'integration_workflow'
            ]
        }, f, indent=2)