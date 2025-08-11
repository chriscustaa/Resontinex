#!/usr/bin/env python3
"""
RESONTINEX Scenario Manager
Advanced scenario filtering and orchestration with complex decision logic.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import statistics
from enum import Enum


class ScenarioCapability(Enum):
    """Enumeration of scenario capability types."""
    REASONING = "reasoning"
    ANALYSIS = "analysis" 
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    CREATIVITY = "creativity"
    CLASSIFICATION = "classification"


@dataclass
class ScenarioMetrics:
    """Comprehensive metrics for scenario performance tracking."""
    success_rate: float
    avg_latency_ms: float
    complexity_score: float
    reliability_index: float
    resource_efficiency: float
    user_satisfaction: float
    
    @property
    def composite_score(self) -> float:
        """Calculate weighted composite performance score."""
        weights = {
            'success_rate': 0.25,
            'avg_latency_ms': -0.15,  # Negative because lower is better
            'complexity_score': 0.20,
            'reliability_index': 0.20,
            'resource_efficiency': 0.15,
            'user_satisfaction': 0.15
        }
        
        normalized_latency = max(0, 1 - (self.avg_latency_ms / 10000))  # Normalize to 0-1 range
        
        return (
            weights['success_rate'] * self.success_rate +
            weights['avg_latency_ms'] * normalized_latency +
            weights['complexity_score'] * self.complexity_score +
            weights['reliability_index'] * self.reliability_index +
            weights['resource_efficiency'] * self.resource_efficiency +
            weights['user_satisfaction'] * self.user_satisfaction
        )


@dataclass
class Scenario:
    """Professional scenario representation with comprehensive metadata."""
    id: str
    name: str
    description: str
    capabilities: List[str]
    weight: float
    priority: int
    complexity_level: str
    prerequisites: List[str]
    success_criteria: Dict[str, float]
    resource_requirements: Dict[str, Any]
    tags: List[str]
    metrics: Optional[ScenarioMetrics] = None
    enabled: bool = True
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields after object creation."""
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc).isoformat()
    
    @property
    def capability_types(self) -> Set[ScenarioCapability]:
        """Get strongly-typed capability enumeration."""
        capabilities = set()
        for cap in self.capabilities:
            try:
                capabilities.add(ScenarioCapability(cap.lower()))
            except ValueError:
                logging.warning(f"Unknown capability type: {cap}")
        return capabilities
    
    def matches_filters(self, filters: Dict[str, Any]) -> bool:
        """Advanced filter matching with multiple criteria support."""
        for filter_key, filter_value in filters.items():
            if filter_key == 'capability' and filter_value not in self.capabilities:
                return False
            elif filter_key == 'min_weight' and self.weight < filter_value:
                return False
            elif filter_key == 'max_weight' and self.weight > filter_value:
                return False
            elif filter_key == 'complexity' and self.complexity_level != filter_value:
                return False
            elif filter_key == 'tags' and not any(tag in self.tags for tag in filter_value):
                return False
            elif filter_key == 'min_priority' and self.priority < filter_value:
                return False
            elif filter_key == 'enabled' and self.enabled != filter_value:
                return False
        
        return True


class ScenarioManager:
    """
    Enterprise-grade scenario management with advanced filtering and orchestration.
    
    Provides intelligent scenario selection, performance-based filtering,
    and comprehensive orchestration capabilities for complex workloads.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize scenario manager with configuration validation."""
        self.config = config or {}
        self.scenarios = self._load_scenarios()
        self.filter_cache = {}
        self.performance_history = {}
        
        # Advanced filtering configuration
        self.default_weights = self.config.get('default_scenario_weights', {})
        self.performance_threshold = self.config.get('performance_threshold', 0.7)
        self.adaptive_filtering = self.config.get('enable_adaptive_filtering', True)
        
        # Scenario lifecycle management
        self._scenario_health_scores = {}
        self._last_performance_update = None
        
        logging.info(f"ScenarioManager initialized with {len(self.scenarios)} scenarios")
    
    def _load_scenarios(self) -> Dict[str, Scenario]:
        """Load and validate scenarios from configuration."""
        scenarios = {}
        
        scenario_configs = self.config.get('scenarios', {})
        
        for scenario_id, scenario_config in scenario_configs.items():
            try:
                # Create scenario with comprehensive validation
                scenario = Scenario(
                    id=scenario_id,
                    name=scenario_config.get('name', scenario_id),
                    description=scenario_config.get('description', ''),
                    capabilities=scenario_config.get('capabilities', []),
                    weight=float(scenario_config.get('weight', 1.0)),
                    priority=int(scenario_config.get('priority', 1)),
                    complexity_level=scenario_config.get('complexity_level', 'medium'),
                    prerequisites=scenario_config.get('prerequisites', []),
                    success_criteria=scenario_config.get('success_criteria', {}),
                    resource_requirements=scenario_config.get('resource_requirements', {}),
                    tags=scenario_config.get('tags', []),
                    enabled=scenario_config.get('enabled', True)
                )
                
                scenarios[scenario_id] = scenario
                
            except Exception as e:
                logging.error(f"Failed to load scenario {scenario_id}: {e}")
        
        return scenarios
    
    def get_scenarios(self) -> Dict[str, Scenario]:
        """Get all loaded scenarios with health validation."""
        # Filter out unhealthy scenarios if adaptive filtering is enabled
        if self.adaptive_filtering:
            return {
                scenario_id: scenario 
                for scenario_id, scenario in self.scenarios.items()
                if scenario.enabled and self._is_scenario_healthy(scenario_id)
            }
        
        return {
            scenario_id: scenario 
            for scenario_id, scenario in self.scenarios.items()
            if scenario.enabled
        }
    
    def filter_by_capability(self, capability: str) -> Dict[str, Scenario]:
        """
        Advanced capability-based filtering with fuzzy matching.
        
        Supports exact matches and intelligent partial matching for
        related capabilities with performance-aware selection.
        """
        cache_key = f"capability_{capability}"
        
        if cache_key in self.filter_cache:
            cached_result, cache_time = self.filter_cache[cache_key]
            if (datetime.now(timezone.utc) - cache_time).seconds < 300:  # 5 minute cache
                return cached_result
        
        # Primary capability filtering
        primary_matches = {}
        secondary_matches = {}
        
        for scenario_id, scenario in self.get_scenarios().items():
            if capability in scenario.capabilities:
                primary_matches[scenario_id] = scenario
            elif self._has_related_capability(scenario, capability):
                secondary_matches[scenario_id] = scenario
        
        # Combine results with primary matches prioritized
        filtered_scenarios = primary_matches.copy()
        
        # Add secondary matches if primary results are insufficient
        if len(primary_matches) < self.config.get('min_capability_matches', 3):
            filtered_scenarios.update(secondary_matches)
        
        # Apply performance-based ranking
        if self.adaptive_filtering:
            filtered_scenarios = self._rank_by_performance(filtered_scenarios, capability)
        
        # Cache results
        self.filter_cache[cache_key] = (
            filtered_scenarios, 
            datetime.now(timezone.utc)
        )
        
        return filtered_scenarios
    
    def filter_by_weight_threshold(self, threshold: float) -> Dict[str, Scenario]:
        """
        Intelligent weight-based filtering with dynamic threshold adaptation.
        
        Applies statistical analysis to optimize threshold selection and
        incorporates performance history for enhanced accuracy.
        """
        cache_key = f"weight_{threshold}"
        
        if cache_key in self.filter_cache:
            cached_result, cache_time = self.filter_cache[cache_key]
            if (datetime.now(timezone.utc) - cache_time).seconds < 300:
                return cached_result
        
        scenarios = self.get_scenarios()
        
        # Calculate adaptive threshold based on scenario distribution
        weights = [s.weight for s in scenarios.values()]
        if weights:
            mean_weight = statistics.mean(weights)
            std_weight = statistics.stdev(weights) if len(weights) > 1 else 0.1
            
            # Adjust threshold based on distribution characteristics
            if self.adaptive_filtering:
                if threshold > mean_weight + std_weight:
                    # High threshold - use performance-weighted selection
                    adjusted_threshold = self._calculate_performance_threshold(threshold, weights)
                else:
                    adjusted_threshold = threshold
            else:
                adjusted_threshold = threshold
        else:
            adjusted_threshold = threshold
        
        # Apply primary weight filtering
        weight_filtered = {
            scenario_id: scenario
            for scenario_id, scenario in scenarios.items()
            if scenario.weight >= adjusted_threshold
        }
        
        # Enhanced filtering with composite scoring
        if self.adaptive_filtering and len(weight_filtered) > 10:
            weight_filtered = self._apply_composite_scoring(weight_filtered)
        
        # Cache results
        self.filter_cache[cache_key] = (
            weight_filtered,
            datetime.now(timezone.utc)
        )
        
        return weight_filtered
    
    def calculate_weighted_score(self, scenario_scores: Dict[str, float]) -> float:
        """
        Advanced weighted scoring with multi-dimensional performance analysis.
        
        Incorporates scenario weights, performance history, complexity factors,
        and adaptive learning for optimal scoring accuracy.
        """
        if not scenario_scores:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        performance_bonus = 0.0
        
        for scenario_id, score in scenario_scores.items():
            if scenario_id not in self.scenarios:
                logging.warning(f"Unknown scenario in scoring: {scenario_id}")
                continue
            
            scenario = self.scenarios[scenario_id]
            base_weight = scenario.weight
            
            # Apply adaptive weight adjustments
            if self.adaptive_filtering:
                performance_factor = self._get_performance_factor(scenario_id)
                complexity_factor = self._get_complexity_factor(scenario)
                recency_factor = self._get_recency_factor(scenario_id)
                
                adjusted_weight = base_weight * performance_factor * complexity_factor * recency_factor
            else:
                adjusted_weight = base_weight
            
            # Calculate contribution
            weighted_contribution = score * adjusted_weight
            total_weighted_score += weighted_contribution
            total_weight += adjusted_weight
            
            # Performance bonus for high-performing scenarios
            if score > 0.9 and adjusted_weight > self.performance_threshold:
                performance_bonus += 0.05 * adjusted_weight
        
        if total_weight == 0:
            return 0.0
        
        base_score = total_weighted_score / total_weight
        final_score = min(1.0, base_score + performance_bonus)
        
        # Update performance history
        self._update_performance_history(scenario_scores, final_score)
        
        return final_score
    
    def filter_by_multiple_criteria(self, filters: Dict[str, Any]) -> Dict[str, Scenario]:
        """
        Advanced multi-criteria filtering with complex logic support.
        
        Supports combination of capability, weight, complexity, priority,
        and custom filter predicates with logical operators.
        """
        scenarios = self.get_scenarios()
        
        # Apply basic filters
        filtered_scenarios = {
            scenario_id: scenario
            for scenario_id, scenario in scenarios.items()
            if scenario.matches_filters(filters)
        }
        
        # Apply advanced filters
        if 'custom_predicate' in filters:
            predicate = filters['custom_predicate']
            if callable(predicate):
                filtered_scenarios = {
                    scenario_id: scenario
                    for scenario_id, scenario in filtered_scenarios.items()
                    if predicate(scenario)
                }
        
        # Apply performance-based ranking if enabled
        if self.adaptive_filtering and len(filtered_scenarios) > 1:
            filtered_scenarios = self._rank_by_composite_performance(filtered_scenarios)
        
        return filtered_scenarios
    
    def _has_related_capability(self, scenario: Scenario, target_capability: str) -> bool:
        """Check for related or similar capabilities using semantic matching."""
        capability_relationships = {
            'reasoning': ['analysis', 'evaluation', 'synthesis'],
            'analysis': ['reasoning', 'evaluation'],
            'creativity': ['synthesis', 'generation'],
            'classification': ['analysis', 'evaluation']
        }
        
        related_caps = capability_relationships.get(target_capability.lower(), [])
        return any(cap in scenario.capabilities for cap in related_caps)
    
    def _is_scenario_healthy(self, scenario_id: str) -> bool:
        """Evaluate scenario health based on performance metrics."""
        if scenario_id not in self._scenario_health_scores:
            return True  # Default to healthy for new scenarios
        
        health_score = self._scenario_health_scores[scenario_id]
        return health_score >= self.config.get('health_threshold', 0.6)
    
    def _rank_by_performance(self, scenarios: Dict[str, Scenario], context: str) -> Dict[str, Scenario]:
        """Rank scenarios by performance metrics within given context."""
        scenario_items = list(scenarios.items())
        
        # Sort by composite performance score
        scenario_items.sort(
            key=lambda item: self._get_performance_score(item[0], context),
            reverse=True
        )
        
        return dict(scenario_items)
    
    def _calculate_performance_threshold(self, base_threshold: float, weights: List[float]) -> float:
        """Calculate adaptive threshold based on weight distribution and performance."""
        if not weights:
            return base_threshold
        
        percentile_75 = sorted(weights)[int(0.75 * len(weights))]
        mean_weight = statistics.mean(weights)
        
        # Adjust threshold based on distribution characteristics
        if base_threshold > percentile_75:
            return min(base_threshold, percentile_75 + 0.1)
        elif base_threshold < mean_weight * 0.5:
            return max(base_threshold, mean_weight * 0.6)
        
        return base_threshold
    
    def _apply_composite_scoring(self, scenarios: Dict[str, Scenario]) -> Dict[str, Scenario]:
        """Apply composite scoring to reduce scenario set to optimal candidates."""
        if len(scenarios) <= 5:
            return scenarios
        
        scenario_scores = []
        
        for scenario_id, scenario in scenarios.items():
            composite_score = (
                scenario.weight * 0.4 +
                self._get_performance_factor(scenario_id) * 0.3 +
                (scenario.priority / 10.0) * 0.2 +
                self._get_recency_factor(scenario_id) * 0.1
            )
            scenario_scores.append((scenario_id, scenario, composite_score))
        
        # Select top scenarios based on composite score
        scenario_scores.sort(key=lambda x: x[2], reverse=True)
        top_scenarios = scenario_scores[:8]  # Keep top 8 scenarios
        
        return {item[0]: item[1] for item in top_scenarios}
    
    def _get_performance_factor(self, scenario_id: str) -> float:
        """Get performance factor for scenario based on historical data."""
        if scenario_id not in self.performance_history:
            return 1.0  # Neutral factor for new scenarios
        
        history = self.performance_history[scenario_id]
        if not history:
            return 1.0
        
        # Calculate moving average of recent performance
        recent_scores = history[-10:]  # Last 10 executions
        avg_performance = statistics.mean(recent_scores)
        
        # Convert to factor (0.5 - 1.5 range)
        return 0.5 + avg_performance
    
    def _get_complexity_factor(self, scenario: Scenario) -> float:
        """Get complexity adjustment factor."""
        complexity_factors = {
            'low': 1.1,
            'medium': 1.0,
            'high': 0.9,
            'expert': 0.8
        }
        return complexity_factors.get(scenario.complexity_level, 1.0)
    
    def _get_recency_factor(self, scenario_id: str) -> float:
        """Get recency factor based on last usage."""
        if scenario_id not in self.performance_history:
            return 1.0
        
        # In a production system, this would use actual timestamp data
        # For now, return a neutral factor
        return 1.0
    
    def _get_performance_score(self, scenario_id: str, context: str) -> float:
        """Get context-specific performance score."""
        base_performance = self._get_performance_factor(scenario_id)
        
        # Context-specific adjustments
        context_bonus = 0.0
        if context in ['reasoning', 'analysis'] and 'high_complexity' in scenario_id:
            context_bonus = 0.1
        
        return base_performance + context_bonus
    
    def _rank_by_composite_performance(self, scenarios: Dict[str, Scenario]) -> Dict[str, Scenario]:
        """Rank scenarios by comprehensive composite performance."""
        scenario_items = list(scenarios.items())
        
        def composite_score(item):
            scenario_id, scenario = item
            return (
                scenario.weight * 0.3 +
                self._get_performance_factor(scenario_id) * 0.4 +
                (scenario.priority / 10.0) * 0.2 +
                self._get_complexity_factor(scenario) * 0.1
            )
        
        scenario_items.sort(key=composite_score, reverse=True)
        return dict(scenario_items)
    
    def _update_performance_history(self, scenario_scores: Dict[str, float], final_score: float):
        """Update performance history with latest execution results."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        for scenario_id, score in scenario_scores.items():
            if scenario_id not in self.performance_history:
                self.performance_history[scenario_id] = []
            
            self.performance_history[scenario_id].append(score)
            
            # Keep only recent history (last 100 executions)
            if len(self.performance_history[scenario_id]) > 100:
                self.performance_history[scenario_id] = self.performance_history[scenario_id][-100:]
        
        self._last_performance_update = timestamp
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for monitoring."""
        total_scenarios = len(self.scenarios)
        active_scenarios = len(self.get_scenarios())
        
        performance_data = []
        for scenario_id, history in self.performance_history.items():
            if history:
                performance_data.append({
                    'scenario_id': scenario_id,
                    'executions': len(history),
                    'avg_score': statistics.mean(history),
                    'latest_score': history[-1] if history else 0.0
                })
        
        return {
            'total_scenarios': total_scenarios,
            'active_scenarios': active_scenarios,
            'scenarios_with_history': len(self.performance_history),
            'cache_entries': len(self.filter_cache),
            'last_performance_update': self._last_performance_update,
            'performance_data': sorted(performance_data, key=lambda x: x['avg_score'], reverse=True)
        }