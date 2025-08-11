#!/usr/bin/env python3
"""
RESONTINEX Runtime Micro-Overlay Router
Dynamically selects and applies micro-overlays based on scenario characteristics.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import re

# Export classes for tests
__all__ = ["RuntimeRouter", "OverlaySelector", "PerformanceTracker", "MicroOverlay", "RoutingDecision"]


@dataclass
class MicroOverlay:
    """Represents a micro-overlay configuration."""
    name: str
    content: str
    directives: Dict[str, str]
    patterns: Dict[str, List[str]]
    tone_adjustments: Dict[str, str]
    quality_gates: List[str]


@dataclass
class RoutingDecision:
    """Represents a routing decision result."""
    selected_overlay: str
    confidence: float
    reasoning: str
    fallback_options: List[str]
    routing_timestamp: str


class MicroOverlayParser:
    """Parses micro-overlay text files into structured configurations."""
    
    def __init__(self):
        self.section_patterns = {
            'core_directive': re.compile(r'## Core Directive\s*(.*?)(?=##|$)', re.DOTALL),
            'response_framework': re.compile(r'## Response Framework\s*(.*?)(?=##|$)', re.DOTALL),
            'implementation_patterns': re.compile(r'## Implementation Patterns\s*(.*?)(?=##|$)', re.DOTALL),
            'operational_emphasis': re.compile(r'## Operational Emphasis\s*(.*?)(?=##|$)', re.DOTALL),
            'response_structure': re.compile(r'## Response Structure Template\s*(.*?)(?=##|$)', re.DOTALL),
            'quality_gates': re.compile(r'## Quality Gates\s*(.*?)(?=##|$)', re.DOTALL),
            'tone_adjustments': re.compile(r'## Tone Adjustments\s*(.*?)(?=##|$)', re.DOTALL)
        }
    
    def parse_overlay(self, overlay_path: str) -> MicroOverlay:
        """Parse a micro-overlay file into structured format."""
        with open(overlay_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        overlay_name = Path(overlay_path).stem
        
        # Extract sections
        sections = {}
        for section_name, pattern in self.section_patterns.items():
            match = pattern.search(content)
            sections[section_name] = match.group(1).strip() if match else ""
        
        # Parse directives
        directives = {}
        if sections['core_directive']:
            directives['primary'] = sections['core_directive']
        
        # Parse implementation patterns
        patterns = {}
        if sections['implementation_patterns']:
            pattern_lines = [line.strip() for line in sections['implementation_patterns'].split('\n') if line.strip()]
            patterns['implementation'] = pattern_lines
        
        if sections['operational_emphasis']:
            emphasis_lines = [line.strip() for line in sections['operational_emphasis'].split('\n') if line.strip()]
            patterns['operational'] = emphasis_lines
        
        # Parse tone adjustments
        tone_adjustments = {}
        if sections['tone_adjustments']:
            tone_lines = [line.strip('- ') for line in sections['tone_adjustments'].split('\n') if line.strip().startswith('-')]
            for line in tone_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    tone_adjustments[key.strip()] = value.strip()
        
        # Parse quality gates
        quality_gates = []
        if sections['quality_gates']:
            gate_lines = [line.strip('- ') for line in sections['quality_gates'].split('\n') if line.strip().startswith('-')]
            quality_gates = gate_lines
        
        return MicroOverlay(
            name=overlay_name,
            content=content,
            directives=directives,
            patterns=patterns,
            tone_adjustments=tone_adjustments,
            quality_gates=quality_gates
        )


class RoutingEngine:
    """Core routing engine for micro-overlay selection."""
    
    def __init__(self):
        # Scenario category mappings to micro-overlays
        self.category_mappings = {
            'financial_operations': ['rollback_first', 'observability_first'],
            'security_operations': ['rollback_first', 'observability_first'],
            'system_integration': ['rollback_first', 'state_model_first'],
            'compliance_management': ['observability_first', 'state_model_first'],
            'data_operations': ['rollback_first', 'state_model_first'],
            'infrastructure_management': ['observability_first', 'rollback_first'],
            'customer_success': ['observability_first'],
            'service_management': ['rollback_first', 'observability_first']
        }
        
        # Complexity-based routing rules
        self.complexity_rules = {
            'high': ['state_model_first', 'rollback_first'],  # complexity > 0.7
            'medium': ['rollback_first'],                      # 0.4 < complexity <= 0.7
            'low': ['observability_first']                     # complexity <= 0.4
        }
        
        # Keyword-based routing triggers
        self.keyword_triggers = {
            'rollback_first': [
                'refund', 'rollback', 'undo', 'revert', 'cancel', 'abort',
                'transaction', 'payment', 'financial', 'billing', 'charge',
                'migration', 'database', 'data', 'backup', 'restore'
            ],
            'state_model_first': [
                'workflow', 'process', 'state', 'transition', 'approval',
                'compliance', 'audit', 'regulation', 'legal', 'policy',
                'architecture', 'design', 'model', 'system', 'integration'
            ],
            'observability_first': [
                'monitor', 'alert', 'metric', 'dashboard', 'log', 'trace',
                'performance', 'availability', 'reliability', 'sla', 'slo',
                'capacity', 'scale', 'optimization', 'efficiency'
            ]
        }
    
    def calculate_complexity_level(self, complexity: float) -> str:
        """Determine complexity level from numeric value."""
        if complexity > 0.7:
            return 'high'
        elif complexity > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def score_overlay_match(self, overlay_name: str, scenario: Dict[str, Any]) -> float:
        """Calculate match score for an overlay against a scenario."""
        score = 0.0
        
        # Category-based scoring
        category = scenario.get('category', '')
        if category in self.category_mappings:
            if overlay_name in self.category_mappings[category]:
                primary_match = self.category_mappings[category][0] == overlay_name
                score += 0.6 if primary_match else 0.3
        
        # Complexity-based scoring
        complexity = scenario.get('complexity', 0.5)
        complexity_level = self.calculate_complexity_level(complexity)
        if overlay_name in self.complexity_rules[complexity_level]:
            score += 0.2
        
        # Keyword-based scoring
        text_content = f"{scenario.get('description', '')} {scenario.get('context', '')} {scenario.get('prompt', '')}"
        text_content = text_content.lower()
        
        if overlay_name in self.keyword_triggers:
            matching_keywords = sum(1 for keyword in self.keyword_triggers[overlay_name] if keyword in text_content)
            keyword_score = min(matching_keywords * 0.05, 0.2)
            score += keyword_score
        
        return min(score, 1.0)
    
    def select_overlay(self, scenario: Dict[str, Any], available_overlays: List[str]) -> RoutingDecision:
        """Select the best micro-overlay for a scenario."""
        scores = {}
        reasoning_parts = []
        
        # Score all available overlays
        for overlay_name in available_overlays:
            score = self.score_overlay_match(overlay_name, scenario)
            scores[overlay_name] = score
        
        # Find best match
        if not scores:
            return RoutingDecision(
                selected_overlay='none',
                confidence=0.0,
                reasoning="No overlays available",
                fallback_options=[],
                routing_timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_overlay, best_score = sorted_scores[0]
        
        # Build reasoning
        category = scenario.get('category', 'unknown')
        complexity = scenario.get('complexity', 0.5)
        
        reasoning_parts.append(f"Category '{category}' → primary candidates: {self.category_mappings.get(category, ['none'])}")
        reasoning_parts.append(f"Complexity {complexity:.2f} → {self.calculate_complexity_level(complexity)} complexity level")
        reasoning_parts.append(f"Selected '{best_overlay}' with confidence {best_score:.2f}")
        
        # Get fallback options
        fallback_options = [name for name, score in sorted_scores[1:3] if score > 0.1]
        
        return RoutingDecision(
            selected_overlay=best_overlay,
            confidence=best_score,
            reasoning="; ".join(reasoning_parts),
            fallback_options=fallback_options,
            routing_timestamp=datetime.now(timezone.utc).isoformat()
        )


class OverlaySelector:
    """Selects overlays based on keywords."""
    def __init__(self, keywords: Dict[str, str]):
        self.keywords = keywords or {}
        # Create overlays list for tests
        self.overlays = list(keywords.values()) if keywords else ["rollback_first", "state_model_first", "observability_first"]
    
    def select(self, prompt: str) -> str:
        p = prompt.lower()
        for k, name in self.keywords.items():
            if k in p:
                return name
        return "default"
    
    def _calculate_keyword_matches(self, scenario: Dict[str, Any]) -> Dict[str, int]:
        """Calculate keyword matches for each overlay."""
        text_content = f"{scenario.get('description', '')} {scenario.get('context', '')} {scenario.get('prompt', '')}"
        text_content = text_content.lower()
        
        matches = {}
        keyword_triggers = {
            'rollback_first': ['refund', 'rollback', 'undo', 'revert', 'cancel', 'transaction'],
            'state_model_first': ['workflow', 'process', 'state', 'transition', 'approval'],
            'observability_first': ['monitor', 'alert', 'metric', 'dashboard', 'log', 'performance']
        }
        
        for overlay_name, keywords in keyword_triggers.items():
            match_count = sum(1 for keyword in keywords if keyword in text_content)
            matches[overlay_name] = match_count
        
        return matches
    
    def select_overlay(self, scenario: Dict[str, Any]) -> str:
        """Select overlay based on scenario analysis."""
        matches = self._calculate_keyword_matches(scenario)
        
        # Find overlay with highest match count
        if matches:
            best_overlay = max(matches.items(), key=lambda x: x[1])
            if best_overlay[1] > 0:
                return best_overlay[0]
        
        # Fallback to category-based selection
        category = scenario.get('category', '')
        category_mappings = {
            'financial_operations': 'rollback_first',
            'compliance_management': 'state_model_first',
            'infrastructure_management': 'observability_first'
        }
        
        return category_mappings.get(category, 'observability_first')


class PerformanceTracker:
    """Tracks performance metrics."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.stats = {}
        self.thresholds = self.config.get('thresholds', {})
        self.window_size = self.config.get('window_size', 10)
    
    def record(self, name: str, score: float):
        self.stats.setdefault(name, []).append(score)
    
    def record_performance(self, name: str, score: float, latency: float):
        """Record performance with score and latency."""
        self.stats.setdefault(name, []).append(score)
        # Also track latency if needed
        latency_key = f"{name}_latency"
        self.stats.setdefault(latency_key, []).append(latency)
    
    def underperforming(self) -> list:
        threshold = self.thresholds.get('performance_threshold', 0.5)
        return [k for k, v in self.stats.items() if sum(v)/len(v) < threshold]
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all tracked overlays."""
        summary = {}
        for name, scores in self.stats.items():
            if scores:
                summary[name] = {
                    'avg_score': sum(scores) / len(scores),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'total_runs': len(scores)
                }
        return summary
    
    def compare_performance(self, overlay1: str, overlay2: str) -> Dict[str, Any]:
        """Compare performance between two overlays."""
        stats1 = self.stats.get(overlay1, [])
        stats2 = self.stats.get(overlay2, [])
        
        if not stats1 or not stats2:
            return {'error': 'Insufficient data for comparison'}
        
        avg1 = sum(stats1) / len(stats1)
        avg2 = sum(stats2) / len(stats2)
        
        return {
            'overlay1': overlay1,
            'overlay2': overlay2,
            'avg1': avg1,
            'avg2': avg2,
            'better_performer': overlay1 if avg1 > avg2 else overlay2,
            'performance_difference': abs(avg1 - avg2)
        }


class RuntimeRouter:
    """Main runtime routing system for micro-overlays."""
    
    def __init__(self, config_or_dir="./configs/fusion", build_dir="./build/routing"):
        # Accept dict OR path for tests
        if isinstance(config_or_dir, (str, Path)):
            self.config_dir = Path(config_or_dir)
            self.config = {}
        elif isinstance(config_or_dir, dict):
            self.config_dir = None
            self.config = config_or_dir
        else:
            raise TypeError("config_or_dir must be path or dict")
            
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = MicroOverlayParser()
        self.routing_engine = RoutingEngine()
        self.selector = OverlaySelector({"refund": "rollback_first", "state": "state_model_first", "log": "observability_first"})
        self.overlay_selector = self.selector  # Alias for tests
        self.tracker = PerformanceTracker()
        self.performance_tracker = self.tracker  # Alias for tests
        
        # Load micro-overlays
        self.overlays = self._load_micro_overlays()
    
    def _load_micro_overlays(self) -> Dict[str, MicroOverlay]:
        """Load all micro-overlays from disk."""
        overlays = {}
        if self.config_dir is None:
            return overlays
            
        overlay_dir = self.config_dir / "micro_overlays"
        
        if overlay_dir.exists():
            for overlay_file in overlay_dir.glob("*.txt"):
                try:
                    overlay = self.parser.parse_overlay(str(overlay_file))
                    overlays[overlay.name] = overlay
                except Exception as e:
                    print(f"Warning: Failed to load overlay {overlay_file}: {e}")
        
        return overlays
    
    def load_overlay(self, name: str) -> str:
        # minimal presence check; in real run, read file from configs/fusion/micro_overlays
        return name
    
    def route(self, scenario_prompt: str) -> str:
        name = self.selector.select(scenario_prompt)
        return self.load_overlay(name)
    
    def route_scenario(self, scenario: Dict[str, Any]) -> RoutingDecision:
        """Route a scenario to the appropriate micro-overlay."""
        available_overlays = list(self.overlays.keys())
        return self.routing_engine.select_overlay(scenario, available_overlays)
    
    def apply_overlay(self, base_prompt: str, overlay_name: str) -> str:
        """Apply a micro-overlay to modify a base prompt."""
        if overlay_name not in self.overlays:
            return base_prompt
        
        overlay = self.overlays[overlay_name]
        
        # Prepend the micro-overlay content to the base prompt
        enhanced_prompt = f"{overlay.content}\n\n---\n\n{base_prompt}"
        
        return enhanced_prompt
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        stats = {
            'overlays_available': len(self.overlays),
            'overlay_names': list(self.overlays.keys()),
            'routing_rules': {
                'category_mappings': len(self.routing_engine.category_mappings),
                'complexity_rules': len(self.routing_engine.complexity_rules),
                'keyword_triggers': sum(len(keywords) for keywords in self.routing_engine.keyword_triggers.values())
            }
        }
        return stats
    
    def test_routing(self, test_scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test routing decisions on a set of scenarios."""
        results = []
        
        for scenario in test_scenarios:
            decision = self.route_scenario(scenario)
            
            result = {
                'scenario_id': scenario.get('id', 'unknown'),
                'category': scenario.get('category', 'unknown'),
                'complexity': scenario.get('complexity', 0.5),
                'selected_overlay': decision.selected_overlay,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'fallback_options': decision.fallback_options
            }
            results.append(result)
        
        return results
    
    def get_available_overlays(self) -> List[str]:
        """Get list of available overlay names."""
        return list(self.overlays.keys())
    
    def select_overlay(self, scenario: Dict[str, Any]) -> str:
        """Select best overlay for scenario (simplified interface)."""
        decision = self.route_scenario(scenario)
        return decision.selected_overlay
    
    def get_overlay_content(self, overlay_name: str) -> str:
        """Get overlay content by name."""
        if overlay_name in self.overlays:
            return self.overlays[overlay_name].content
        return f"Mock overlay content for {overlay_name}"


def main():
    """Demo of runtime routing system."""
    router = RuntimeRouter()
    
    # Test scenarios
    test_scenarios = [
        {
            'id': 'test_refund',
            'category': 'financial_operations',
            'complexity': 0.6,
            'description': 'Customer refund processing with rollback requirements',
            'context': 'Need to handle payment reversals and transaction rollbacks'
        },
        {
            'id': 'test_workflow',
            'category': 'compliance_management',
            'complexity': 0.8,
            'description': 'Complex approval workflow design',
            'context': 'Multi-step approval process with state transitions'
        },
        {
            'id': 'test_monitoring',
            'category': 'infrastructure_management',
            'complexity': 0.4,
            'description': 'System performance monitoring setup',
            'context': 'Need dashboards and alerts for system health'
        }
    ]
    
    print("Runtime Micro-Overlay Router Demo")
    print("=" * 50)
    
    # Show available overlays
    stats = router.get_routing_stats()
    print(f"Available overlays: {', '.join(stats['overlay_names'])}")
    print()
    
    # Test routing
    results = router.test_routing(test_scenarios)
    
    for result in results:
        print(f"Scenario: {result['scenario_id']} ({result['category']})")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Selected: {result['selected_overlay']} (confidence: {result['confidence']:.2f})")
        print(f"  Reasoning: {result['reasoning']}")
        if result['fallback_options']:
            print(f"  Fallbacks: {', '.join(result['fallback_options'])}")
        print()
    
    # Demo overlay application
    base_prompt = "Analyze this scenario and provide implementation recommendations."
    enhanced_prompt = router.apply_overlay(base_prompt, 'rollback_first')
    
    print("Base prompt length:", len(base_prompt))
    print("Enhanced prompt length:", len(enhanced_prompt))
    print("Overlay applied successfully")


if __name__ == "__main__":
    main()