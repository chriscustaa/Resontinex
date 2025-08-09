#!/usr/bin/env python3
"""
Unit tests for cross-judge evaluation system.
Tests the dual-evaluator setup with rule-based validation.
"""

import unittest
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from judge_fusion import CrossJudgeEvaluator, RuleBasedValidator, Prompt3Evaluator
except ImportError as e:
    print(f"Warning: Could not import judge_fusion components: {e}")

class TestCrossJudgeEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_scenario = {
            "id": "test_scenario_01",
            "description": "Test scenario for unit testing",
            "complexity": 3,
            "expected_capabilities": ["reasoning", "accuracy"],
            "query": "What is 2+2?",
            "expected_answer": "4"
        }
        
    def test_evaluator_initialization(self):
        """Test that CrossJudgeEvaluator initializes correctly."""
        try:
            evaluator = CrossJudgeEvaluator()
            self.assertIsNotNone(evaluator.primary_judge)
            self.assertIsNotNone(evaluator.validator)
            self.assertIsInstance(evaluator.primary_judge, Prompt3Evaluator)
            self.assertIsInstance(evaluator.validator, RuleBasedValidator)
        except NameError:
            self.skipTest("CrossJudgeEvaluator not available")
    
    def test_rule_based_validator(self):
        """Test rule-based validation logic."""
        try:
            validator = RuleBasedValidator()
            
            # Test consistent high scores
            scores = [0.9, 0.85, 0.88]
            result = validator.validate_consistency(scores)
            self.assertTrue(result['is_consistent'])
            
            # Test inconsistent scores
            scores = [0.9, 0.2, 0.85]
            result = validator.validate_consistency(scores)
            self.assertFalse(result['is_consistent'])
            
        except NameError:
            self.skipTest("RuleBasedValidator not available")
    
    @patch('judge_fusion.openai.ChatCompletion.create')
    def test_cross_evaluation(self, mock_openai):
        """Test cross-evaluation process."""
        try:
            # Mock OpenAI responses
            mock_openai.return_value.choices = [
                MagicMock(message=MagicMock(content="Score: 0.85\nReasoning: Good response"))
            ]
            
            evaluator = CrossJudgeEvaluator()
            
            baseline_response = "The answer is 4."
            overlay_response = "2 + 2 equals 4."
            
            result = evaluator.evaluate_cross_judge(
                self.test_scenario,
                baseline_response,
                overlay_response
            )
            
            self.assertIn('primary_score', result)
            self.assertIn('validation_result', result)
            self.assertIn('final_recommendation', result)
            
        except (NameError, ImportError):
            self.skipTest("CrossJudgeEvaluator or dependencies not available")

class TestRuleBasedValidator(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            self.validator = RuleBasedValidator()
        except NameError:
            self.validator = None
    
    def test_score_variance_calculation(self):
        """Test score variance calculation."""
        if self.validator is None:
            self.skipTest("RuleBasedValidator not available")
            
        scores = [0.8, 0.85, 0.9, 0.75]
        variance = self.validator._calculate_variance(scores)
        self.assertIsInstance(variance, float)
        self.assertGreaterEqual(variance, 0)
    
    def test_outlier_detection(self):
        """Test outlier detection in score sets."""
        if self.validator is None:
            self.skipTest("RuleBasedValidator not available")
            
        # Normal scores
        scores = [0.8, 0.85, 0.82, 0.88]
        outliers = self.validator._detect_outliers(scores)
        self.assertEqual(len(outliers), 0)
        
        # With outlier
        scores = [0.8, 0.85, 0.2, 0.88]
        outliers = self.validator._detect_outliers(scores)
        self.assertGreater(len(outliers), 0)
        self.assertIn(0.2, outliers)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        if self.validator is None:
            self.skipTest("RuleBasedValidator not available")
            
        # High consistency should yield high confidence
        scores = [0.85, 0.87, 0.86, 0.84]
        confidence = self.validator._calculate_confidence(scores)
        self.assertGreater(confidence, 0.7)
        
        # Low consistency should yield low confidence
        scores = [0.9, 0.1, 0.8, 0.3]
        confidence = self.validator._calculate_confidence(scores)
        self.assertLess(confidence, 0.5)

if __name__ == '__main__':
    # Create test results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    
    # Save test results
    with open('test_results/cross_judge_results.json', 'w') as f:
        json.dump({
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
        }, f, indent=2)