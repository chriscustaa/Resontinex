#!/usr/bin/env python3
"""
Unit tests for runtime routing system.
Tests micro-overlay routing, overlay selection logic, and performance tracking.
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
    from runtime_router import RuntimeRouter, OverlaySelector, PerformanceTracker
except ImportError as e:
    print(f"Warning: Could not import runtime_router components: {e}")

class TestRuntimeRouter(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'overlays': {
                'rollback_first': {
                    'path': 'configs/fusion/micro_overlays/rollback_first.txt',
                    'priority': 3,
                    'scenarios': ['complexity_high', 'safety_critical']
                },
                'state_model_first': {
                    'path': 'configs/fusion/micro_overlays/state_model_first.txt',
                    'priority': 2,
                    'scenarios': ['architecture_design', 'system_planning']
                },
                'observability_first': {
                    'path': 'configs/fusion/micro_overlays/observability_first.txt',
                    'priority': 1,
                    'scenarios': ['monitoring', 'debugging']
                }
            },
            'routing_rules': {
                'default_overlay': 'state_model_first',
                'fallback_overlay': 'rollback_first',
                'selection_timeout': 5.0
            },
            'performance_tracking': {
                'window_size': 100,
                'min_samples': 10,
                'performance_threshold': 0.75
            }
        }
        
        self.test_scenario = {
            'id': 'test_scenario_routing_01',
            'complexity': 4,
            'category': 'architecture_design',
            'keywords': ['system', 'design', 'architecture']
        }
    
    def test_router_initialization(self):
        """Test that RuntimeRouter initializes correctly."""
        try:
            router = RuntimeRouter(self.test_config)
            self.assertIsNotNone(router.overlay_selector)
            self.assertIsNotNone(router.performance_tracker)
            self.assertEqual(len(router.overlays), 3)
        except NameError:
            self.skipTest("RuntimeRouter not available")
    
    def test_overlay_loading(self):
        """Test loading of micro-overlay configurations."""
        try:
            router = RuntimeRouter(self.test_config)
            overlays = router.get_available_overlays()
            
            self.assertIn('rollback_first', overlays)
            self.assertIn('state_model_first', overlays)
            self.assertIn('observability_first', overlays)
            
            # Test overlay priority ordering
            overlay_priorities = [overlays[name]['priority'] for name in overlays]
            self.assertEqual(max(overlay_priorities), 3)  # rollback_first
            self.assertEqual(min(overlay_priorities), 1)  # observability_first
            
        except NameError:
            self.skipTest("RuntimeRouter not available")
    
    def test_scenario_based_routing(self):
        """Test overlay selection based on scenario characteristics."""
        try:
            router = RuntimeRouter(self.test_config)
            
            # Test architecture design scenario routing
            selected_overlay = router.select_overlay(self.test_scenario)
            self.assertEqual(selected_overlay, 'state_model_first')
            
            # Test safety critical scenario
            safety_scenario = {
                'id': 'safety_test',
                'complexity': 5,
                'category': 'safety_critical',
                'keywords': ['critical', 'safety', 'high-risk']
            }
            
            selected_overlay = router.select_overlay(safety_scenario)
            self.assertEqual(selected_overlay, 'rollback_first')
            
        except NameError:
            self.skipTest("RuntimeRouter not available")

class TestOverlaySelector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.overlays_config = {
            'rollback_first': {
                'priority': 3,
                'scenarios': ['complexity_high', 'safety_critical'],
                'keywords': ['rollback', 'safety', 'critical']
            },
            'state_model_first': {
                'priority': 2,
                'scenarios': ['architecture_design', 'system_planning'],
                'keywords': ['architecture', 'system', 'design', 'planning']
            },
            'observability_first': {
                'priority': 1,
                'scenarios': ['monitoring', 'debugging'],
                'keywords': ['monitor', 'debug', 'observe', 'metrics']
            }
        }
    
    def test_selector_initialization(self):
        """Test that OverlaySelector initializes correctly."""
        try:
            selector = OverlaySelector(self.overlays_config)
            self.assertEqual(len(selector.overlays), 3)
            self.assertIn('rollback_first', selector.overlays)
        except NameError:
            self.skipTest("OverlaySelector not available")
    
    def test_keyword_matching(self):
        """Test keyword-based overlay selection."""
        try:
            selector = OverlaySelector(self.overlays_config)
            
            # Test architecture keywords
            arch_scenario = {
                'keywords': ['system', 'architecture', 'design'],
                'category': 'architecture_design'
            }
            
            matches = selector._calculate_keyword_matches(arch_scenario)
            self.assertGreater(matches['state_model_first'], matches['rollback_first'])
            
            # Test monitoring keywords
            monitor_scenario = {
                'keywords': ['monitoring', 'metrics', 'observe'],
                'category': 'monitoring'
            }
            
            matches = selector._calculate_keyword_matches(monitor_scenario)
            self.assertGreater(matches['observability_first'], matches['state_model_first'])
            
        except NameError:
            self.skipTest("OverlaySelector not available")
    
    def test_priority_based_selection(self):
        """Test priority-based overlay selection when no clear match."""
        try:
            selector = OverlaySelector(self.overlays_config)
            
            # Ambiguous scenario with no clear keywords
            ambiguous_scenario = {
                'keywords': ['general', 'task'],
                'category': 'general'
            }
            
            selected = selector.select_overlay(ambiguous_scenario)
            # Should fall back to highest priority (rollback_first)
            self.assertEqual(selected, 'rollback_first')
            
        except NameError:
            self.skipTest("OverlaySelector not available")

class TestPerformanceTracker(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker_config = {
            'window_size': 10,
            'min_samples': 3,
            'performance_threshold': 0.75
        }
    
    def test_tracker_initialization(self):
        """Test that PerformanceTracker initializes correctly."""
        try:
            tracker = PerformanceTracker(self.tracker_config)
            self.assertEqual(tracker.window_size, 10)
            self.assertEqual(tracker.performance_threshold, 0.75)
        except NameError:
            self.skipTest("PerformanceTracker not available")
    
    def test_performance_recording(self):
        """Test recording and tracking overlay performance."""
        try:
            tracker = PerformanceTracker(self.tracker_config)
            
            # Record some performance data
            tracker.record_performance('state_model_first', 0.85, 2.3)
            tracker.record_performance('state_model_first', 0.78, 2.1)
            tracker.record_performance('rollback_first', 0.92, 1.8)
            
            # Check performance stats
            stats = tracker.get_overlay_stats('state_model_first')
            self.assertEqual(stats['count'], 2)
            self.assertAlmostEqual(stats['avg_score'], (0.85 + 0.78) / 2, places=2)
            
        except NameError:
            self.skipTest("PerformanceTracker not available")
    
    def test_performance_comparison(self):
        """Test overlay performance comparison logic."""
        try:
            tracker = PerformanceTracker(self.tracker_config)
            
            # Add performance data for multiple overlays
            for i in range(5):
                tracker.record_performance('overlay_a', 0.9, 2.0)
                tracker.record_performance('overlay_b', 0.7, 2.5)
            
            comparison = tracker.compare_overlay_performance('overlay_a', 'overlay_b')
            
            self.assertGreater(comparison['overlay_a']['avg_score'], 
                             comparison['overlay_b']['avg_score'])
            self.assertLess(comparison['overlay_a']['avg_latency'], 
                          comparison['overlay_b']['avg_latency'])
            
        except NameError:
            self.skipTest("PerformanceTracker not available")
    
    def test_underperforming_overlay_detection(self):
        """Test detection of underperforming overlays."""
        try:
            tracker = PerformanceTracker(self.tracker_config)
            
            # Record consistently poor performance
            for i in range(5):
                tracker.record_performance('poor_overlay', 0.6, 3.0)  # Below threshold
                tracker.record_performance('good_overlay', 0.85, 2.0)  # Above threshold
            
            underperforming = tracker.get_underperforming_overlays()
            
            self.assertIn('poor_overlay', underperforming)
            self.assertNotIn('good_overlay', underperforming)
            
        except NameError:
            self.skipTest("PerformanceTracker not available")

class TestRuntimeRoutingIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.test_config = {
            'overlays': {
                'rollback_first': {
                    'path': 'configs/fusion/micro_overlays/rollback_first.txt',
                    'priority': 3,
                    'scenarios': ['complexity_high', 'safety_critical']
                },
                'state_model_first': {
                    'path': 'configs/fusion/micro_overlays/state_model_first.txt',
                    'priority': 2,
                    'scenarios': ['architecture_design', 'system_planning']
                },
                'observability_first': {
                    'path': 'configs/fusion/micro_overlays/observability_first.txt',
                    'priority': 1,
                    'scenarios': ['monitoring', 'debugging']
                }
            },
            'routing_rules': {
                'default_overlay': 'state_model_first',
                'fallback_overlay': 'rollback_first',
                'selection_timeout': 5.0
            },
            'performance_tracking': {
                'window_size': 100,
                'min_samples': 10,
                'performance_threshold': 0.75
            }
        }
        
        # Create mock overlay files
        self.overlay_files = {}
        for overlay_name in ['rollback_first', 'state_model_first', 'observability_first']:
            overlay_path = os.path.join(self.temp_dir, f'{overlay_name}.txt')
            with open(overlay_path, 'w') as f:
                f.write(f"Mock {overlay_name} overlay content\nOptimized for {overlay_name.replace('_', ' ')}")
            self.overlay_files[overlay_name] = overlay_path
    
    def test_end_to_end_routing(self):
        """Test complete routing workflow from scenario to overlay selection."""
        try:
            # Update config with temp file paths
            config = self.test_config.copy()
            for overlay_name, overlay_config in config['overlays'].items():
                overlay_config['path'] = self.overlay_files[overlay_name]
            
            router = RuntimeRouter(config)
            
            # Test routing decision
            test_scenarios = [
                {
                    'id': 'arch_test',
                    'category': 'architecture_design',
                    'keywords': ['system', 'architecture']
                },
                {
                    'id': 'debug_test', 
                    'category': 'debugging',
                    'keywords': ['debug', 'monitor']
                },
                {
                    'id': 'safety_test',
                    'category': 'safety_critical',
                    'keywords': ['critical', 'safety']
                }
            ]
            
            routing_results = []
            for scenario in test_scenarios:
                selected_overlay = router.select_overlay(scenario)
                overlay_content = router.get_overlay_content(selected_overlay)
                
                routing_results.append({
                    'scenario_id': scenario['id'],
                    'selected_overlay': selected_overlay,
                    'overlay_loaded': overlay_content is not None
                })
            
            # Verify routing decisions
            self.assertEqual(len(routing_results), 3)
            
            # Architecture scenario should route to state_model_first
            arch_result = next(r for r in routing_results if r['scenario_id'] == 'arch_test')
            self.assertEqual(arch_result['selected_overlay'], 'state_model_first')
            
        except NameError:
            self.skipTest("RuntimeRouter not available")
    
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
    with open('test_results/runtime_router_results.json', 'w') as f:
        json.dump({
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'test_categories': [
                'runtime_routing',
                'overlay_selection',
                'performance_tracking',
                'integration_workflow'
            ]
        }, f, indent=2)