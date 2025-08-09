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


class RuntimeRouter:
    """Main runtime routing system for micro-overlays."""
    
    def __init__(self, config_dir: str = "./configs/fusion", build_dir: str = "./build/routing"):
        self.config_dir = Path(config_dir)
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = MicroOverlayParser()
        self.routing_engine = RoutingEngine()
        
        # Load micro-overlays
        self.overlays = self._load_micro_overlays()
    
    def _load_micro_overlays(self) -> Dict[str, MicroOverlay]:
        """Load all micro-overlays from disk."""
        overlays = {}
        overlay_dir = self.config_dir / "micro_overlays"
        
        if overlay_dir.exists():
            for overlay_file in overlay_dir.glob("*.txt"):
                try:
                    overlay = self.parser.parse_overlay(str(overlay_file))
                    overlays[overlay.name] = overlay
                except Exception as e:
                    print(f"Warning: Failed to load overlay {overlay_file}: {e}")
        
        return overlays
    
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