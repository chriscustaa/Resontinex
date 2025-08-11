"""
Metrics Cardinality Controls for Fusion System
Prevents metric explosion by controlling tag combinations and prefixes
"""

import re
import time
from typing import Dict, Any, List, Set, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    name: str
    value: float
    tags: Dict[str, str]
    timestamp: float


class FusionMetricsController:
    """Controls metric cardinality by enforcing tag whitelist and prefix standards."""
    
    METRIC_PREFIX = "fusion."
    ALLOWED_TAGS = {"scenario", "overlay", "result"}
    
    def __init__(self, max_cardinality: int = 1000):
        self.max_cardinality = max_cardinality
        self.metric_registry: Dict[str, Set[str]] = defaultdict(set)  # metric_name -> set of tag combinations
        self.dropped_metrics_count = 0
        self.dropped_tags_count = 0
        
    def sanitize_metric_name(self, name: str) -> str:
        """Ensure metric name has proper prefix and is valid."""
        # Remove any existing prefix variations to avoid double-prefixing
        name = re.sub(r'^(fusion[._-]?)', '', name, flags=re.IGNORECASE)
        
        # Add standard prefix
        prefixed_name = f"{self.METRIC_PREFIX}{name}"
        
        # Sanitize name - only allow alphanumeric, dots, underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._]', '_', prefixed_name)
        
        return sanitized
    
    def filter_tags(self, tags: Dict[str, str]) -> Dict[str, str]:
        """Filter tags to only include whitelisted ones."""
        if not tags:
            return {}
        
        filtered_tags = {}
        original_count = len(tags)
        
        for key, value in tags.items():
            if key.lower() in self.ALLOWED_TAGS:
                # Sanitize tag values - limit length and remove problematic characters
                sanitized_value = self._sanitize_tag_value(value)
                if sanitized_value:  # Only include non-empty values
                    filtered_tags[key.lower()] = sanitized_value
        
        # Track dropped tags
        dropped_count = original_count - len(filtered_tags)
        self.dropped_tags_count += dropped_count
        
        return filtered_tags
    
    def _sanitize_tag_value(self, value: str) -> str:
        """Sanitize tag value to prevent cardinality explosion."""
        if not isinstance(value, str):
            value = str(value)
        
        # Limit length to prevent extremely long tag values
        value = value[:50]
        
        # Remove or replace problematic characters
        value = re.sub(r'[^\w\-_.]', '_', value)
        
        # Remove multiple consecutive underscores
        value = re.sub(r'_{2,}', '_', value)
        
        # Remove leading/trailing underscores
        value = value.strip('_')
        
        return value
    
    def _generate_tag_signature(self, tags: Dict[str, str]) -> str:
        """Generate a signature for tag combination for cardinality tracking."""
        if not tags:
            return "no_tags"
        
        # Sort tags by key for consistent signature
        sorted_items = sorted(tags.items())
        return "|".join(f"{k}:{v}" for k, v in sorted_items)
    
    def should_accept_metric(self, name: str, tags: Dict[str, str]) -> bool:
        """Check if metric should be accepted based on cardinality limits."""
        tag_signature = self._generate_tag_signature(tags)
        
        # Check if this exact combination already exists
        if tag_signature in self.metric_registry[name]:
            return True  # Already exists, no cardinality impact
        
        # Check if adding this would exceed cardinality limit
        current_cardinality = len(self.metric_registry[name])
        if current_cardinality >= self.max_cardinality:
            return False
        
        # Check global cardinality across all metrics
        total_combinations = sum(len(combinations) for combinations in self.metric_registry.values())
        if total_combinations >= self.max_cardinality * 10:  # Global limit
            return False
        
        return True
    
    def process_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, timestamp: Optional[float] = None) -> Optional[MetricPoint]:
        """Process a metric through cardinality controls."""
        # Sanitize metric name
        sanitized_name = self.sanitize_metric_name(name)
        
        # Filter tags to allowed set
        filtered_tags = self.filter_tags(tags or {})
        
        # Check cardinality limits
        if not self.should_accept_metric(sanitized_name, filtered_tags):
            self.dropped_metrics_count += 1
            return None
        
        # Register the tag combination
        tag_signature = self._generate_tag_signature(filtered_tags)
        self.metric_registry[sanitized_name].add(tag_signature)
        
        # Create metric point
        metric_point = MetricPoint(
            name=sanitized_name,
            value=value,
            tags=filtered_tags,
            timestamp=timestamp or time.time()
        )
        
        return metric_point
    
    def get_cardinality_stats(self) -> Dict[str, Any]:
        """Get cardinality statistics."""
        metric_cardinalities = {
            name: len(combinations) 
            for name, combinations in self.metric_registry.items()
        }
        
        total_combinations = sum(metric_cardinalities.values())
        
        return {
            'total_metrics': len(self.metric_registry),
            'total_combinations': total_combinations,
            'max_cardinality_limit': self.max_cardinality,
            'dropped_metrics_count': self.dropped_metrics_count,
            'dropped_tags_count': self.dropped_tags_count,
            'cardinality_utilization': total_combinations / (self.max_cardinality * 10) if self.max_cardinality > 0 else 0,
            'top_metrics_by_cardinality': sorted(metric_cardinalities.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def validate_metric_standards(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of metrics against standards."""
        violations = []
        valid_count = 0
        
        for i, metric in enumerate(metrics):
            name = metric.get('name', '')
            tags = metric.get('tags', {})
            
            # Check prefix
            if not name.startswith(self.METRIC_PREFIX):
                violations.append(f"Metric {i}: Missing '{self.METRIC_PREFIX}' prefix: {name}")
            
            # Check for invalid tag keys
            if tags:
                invalid_tags = set(tags.keys()) - self.ALLOWED_TAGS
                if invalid_tags:
                    violations.append(f"Metric {i}: Invalid tags {invalid_tags}. Allowed: {self.ALLOWED_TAGS}")
            
            # Check for high-cardinality tag values
            if tags:
                for key, value in tags.items():
                    if isinstance(value, str) and len(value) > 50:
                        violations.append(f"Metric {i}: High-cardinality tag value in '{key}': length {len(value)}")
            
            if not violations or len(violations) <= i:  # No new violations for this metric
                valid_count += 1
        
        return {
            'total_metrics': len(metrics),
            'valid_metrics': valid_count,
            'violations': violations,
            'compliance_rate': valid_count / len(metrics) if metrics else 1.0
        }


class FusionMetricsCollector:
    """High-level metrics collection interface with cardinality controls."""
    
    def __init__(self, controller: Optional[FusionMetricsController] = None):
        self.controller = controller or FusionMetricsController()
        self.buffer: List[MetricPoint] = []
        self.buffer_size = 100
    
    def record_fusion_latency(self, latency_ms: float, scenario: str, overlay: str, result: str) -> bool:
        """Record fusion execution latency."""
        return self._record_metric(
            "execution.latency_ms",
            latency_ms,
            {"scenario": scenario, "overlay": overlay, "result": result}
        )
    
    def record_token_delta(self, delta_pct: float, scenario: str, overlay: str) -> bool:
        """Record token usage delta percentage."""
        return self._record_metric(
            "tokens.delta_pct",
            delta_pct,
            {"scenario": scenario, "overlay": overlay}
        )
    
    def record_quality_score(self, score: float, scenario: str, overlay: str) -> bool:
        """Record quality evaluation score."""
        return self._record_metric(
            "quality.score",
            score,
            {"scenario": scenario, "overlay": overlay}
        )
    
    def record_overlay_selection(self, scenario: str, selected_overlay: str) -> bool:
        """Record overlay selection decision."""
        return self._record_metric(
            "routing.overlay_selected",
            1.0,
            {"scenario": scenario, "overlay": selected_overlay, "result": "selected"}
        )
    
    def record_circuit_breaker_event(self, event_type: str, overlay: str) -> bool:
        """Record circuit breaker events."""
        return self._record_metric(
            "circuit_breaker.event",
            1.0,
            {"overlay": overlay, "result": event_type}
        )
    
    def _record_metric(self, name: str, value: float, tags: Dict[str, str]) -> bool:
        """Internal method to record metrics through controller."""
        metric_point = self.controller.process_metric(name, value, tags)
        if metric_point:
            self.buffer.append(metric_point)
            
            # Flush buffer if it gets too large
            if len(self.buffer) >= self.buffer_size:
                self.flush()
            
            return True
        return False
    
    def flush(self) -> List[MetricPoint]:
        """Flush buffered metrics and return them."""
        flushed_metrics = self.buffer.copy()
        self.buffer.clear()
        return flushed_metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        controller_stats = self.controller.get_cardinality_stats()
        
        return {
            'buffered_metrics': len(self.buffer),
            'cardinality_control': controller_stats,
            'prefix_standard': self.controller.METRIC_PREFIX,
            'allowed_tags': list(self.controller.ALLOWED_TAGS)
        }


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> FusionMetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = FusionMetricsCollector()
    return _metrics_collector


def reset_metrics_collector():
    """Reset metrics collector (for testing)."""
    global _metrics_collector
    _metrics_collector = None