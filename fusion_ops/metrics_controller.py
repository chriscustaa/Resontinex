"""
Metrics Cardinality Controller
Prevents metrics cardinality explosion by normalizing scenario names and controlling tag values.
"""

import hashlib
import re
from typing import Dict, Any, Set, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CardinalityLimits:
    """Cardinality limits for different metric dimensions."""
    max_scenario_types: int = 20
    max_model_names: int = 10
    max_overlay_versions: int = 50
    max_feature_flags: int = 100


class MetricsCardinalityController:
    """Controls metrics cardinality to prevent explosion in telemetry systems."""
    
    # Predefined scenario types to normalize against
    SCENARIO_TYPE_WHITELIST = {
        'fusion_overlay_test',
        'benchmark_validation', 
        'performance_comparison',
        'budget_analysis',
        'golden_test',
        'smoke_test',
        'integration_test',
        'load_test',
        'chaos_test',
        'security_test',
        'compatibility_test',
        'regression_test',
        'canary_test',
        'synthetic_test',
        'user_acceptance_test',
        'end_to_end_test',
        'unit_test',
        'component_test',
        'contract_test',
        'exploratory_test'
    }
    
    # Model name normalization
    MODEL_NAME_MAPPING = {
        'gpt-4o': 'gpt4o',
        'gpt-4o-mini': 'gpt4o_mini', 
        'claude-3.5-sonnet': 'claude35_sonnet',
        'claude-3-haiku': 'claude3_haiku',
        'gemini-pro': 'gemini_pro',
        'gemini-flash': 'gemini_flash',
        'grok-beta': 'grok',
        'local-gguf': 'local_gguf',
        'llama-70b': 'llama70b',
        'mixtral-8x7b': 'mixtral8x7b'
    }
    
    def __init__(self, limits: Optional[CardinalityLimits] = None):
        self.limits = limits or CardinalityLimits()
        self.scenario_id_cache: Dict[str, str] = {}
        self.seen_values: Dict[str, Set[str]] = {
            'scenario_types': set(),
            'model_names': set(),
            'overlay_versions': set(),
            'feature_flags': set()
        }
        
    def normalize_scenario_id(self, scenario_name: str) -> str:
        """
        Normalize scenario names to prevent cardinality explosion.
        
        Maps arbitrary scenario names to a limited set of normalized IDs.
        """
        if scenario_name in self.scenario_id_cache:
            return self.scenario_id_cache[scenario_name]
        
        # Try to extract scenario type from name
        scenario_lower = scenario_name.lower()
        
        # Check against whitelist patterns
        for scenario_type in self.SCENARIO_TYPE_WHITELIST:
            if scenario_type.replace('_', '') in scenario_lower.replace('_', ''):
                normalized_id = scenario_type
                break
        else:
            # Generate deterministic hash-based ID for unknown scenarios
            scenario_hash = hashlib.md5(scenario_name.encode()).hexdigest()[:8]
            normalized_id = f"custom_{scenario_hash}"
        
        # Cache the result
        self.scenario_id_cache[scenario_name] = normalized_id
        self.seen_values['scenario_types'].add(normalized_id)
        
        return normalized_id
    
    def normalize_model_name(self, model_name: str) -> str:
        """Normalize model names to prevent cardinality explosion."""
        model_lower = model_name.lower().strip()
        
        # Direct mapping
        if model_lower in self.MODEL_NAME_MAPPING:
            normalized = self.MODEL_NAME_MAPPING[model_lower]
        else:
            # Pattern-based normalization
            if 'gpt' in model_lower and '4' in model_lower:
                normalized = 'gpt4o' if 'mini' not in model_lower else 'gpt4o_mini'
            elif 'claude' in model_lower:
                if '3.5' in model_lower or '35' in model_lower:
                    normalized = 'claude35_sonnet'
                else:
                    normalized = 'claude3_haiku'
            elif 'gemini' in model_lower:
                normalized = 'gemini_pro' if 'pro' in model_lower else 'gemini_flash'
            elif 'grok' in model_lower:
                normalized = 'grok'
            elif any(x in model_lower for x in ['local', 'gguf', 'llama']):
                normalized = 'local_gguf'
            else:
                # Fallback: use hash for unknown models
                model_hash = hashlib.md5(model_name.encode()).hexdigest()[:6]
                normalized = f"model_{model_hash}"
        
        self.seen_values['model_names'].add(normalized)
        return normalized
    
    def normalize_overlay_version(self, version: str) -> str:
        """Normalize overlay versions to control cardinality."""
        if not version or version == "unknown":
            return "unknown"
        
        # Extract semantic version pattern
        version_match = re.match(r'(\d+\.\d+)', str(version))
        if version_match:
            normalized = version_match.group(1)
        else:
            # Handle special versions
            if 'fallback' in str(version).lower():
                normalized = 'fallback'
            elif 'downgrade' in str(version).lower():
                normalized = 'downgrade'
            else:
                normalized = 'custom'
        
        self.seen_values['overlay_versions'].add(normalized)
        return normalized
    
    def sanitize_metrics_tags(self, tags: Dict[str, Any]) -> Dict[str, str]:
        """
        Sanitize metrics tags to prevent cardinality explosion.
        
        Applies normalization and cardinality limits to all tag values.
        """
        sanitized = {}
        
        for key, value in tags.items():
            str_value = str(value)
            
            if key in ['scenario_id', 'scenario_name', 'scenario_type']:
                sanitized[key] = self.normalize_scenario_id(str_value)
            elif key in ['model_name', 'primary_model', 'secondary_model']:
                sanitized[key] = self.normalize_model_name(str_value)
            elif key in ['overlay_version', 'fusion_version']:
                sanitized[key] = self.normalize_overlay_version(str_value)
            elif key.startswith('feature_'):
                # Limit feature flag cardinality
                if len(self.seen_values['feature_flags']) < self.limits.max_feature_flags:
                    self.seen_values['feature_flags'].add(str_value)
                    sanitized[key] = str_value
                else:
                    sanitized[key] = 'other'
            else:
                # Apply general string sanitization
                sanitized[key] = self._sanitize_string_value(str_value)
        
        return sanitized
    
    def _sanitize_string_value(self, value: str, max_length: int = 64) -> str:
        """Sanitize individual string values."""
        # Remove special characters that can cause issues
        sanitized = re.sub(r'[^\w\-\.]', '_', value)
        
        # Truncate if too long
        if len(sanitized) > max_length:
            # Keep meaningful prefix and add hash suffix
            hash_suffix = hashlib.md5(value.encode()).hexdigest()[:6]
            sanitized = sanitized[:max_length-7] + f"_{hash_suffix}"
        
        return sanitized.lower()
    
    def get_cardinality_report(self) -> Dict[str, Any]:
        """Get current cardinality usage report."""
        return {
            'cardinality_limits': {
                'scenario_types': self.limits.max_scenario_types,
                'model_names': self.limits.max_model_names, 
                'overlay_versions': self.limits.max_overlay_versions,
                'feature_flags': self.limits.max_feature_flags
            },
            'current_usage': {
                'scenario_types': len(self.seen_values['scenario_types']),
                'model_names': len(self.seen_values['model_names']),
                'overlay_versions': len(self.seen_values['overlay_versions']),
                'feature_flags': len(self.seen_values['feature_flags'])
            },
            'utilization_pct': {
                'scenario_types': (len(self.seen_values['scenario_types']) / self.limits.max_scenario_types) * 100,
                'model_names': (len(self.seen_values['model_names']) / self.limits.max_model_names) * 100,
                'overlay_versions': (len(self.seen_values['overlay_versions']) / self.limits.max_overlay_versions) * 100,
                'feature_flags': (len(self.seen_values['feature_flags']) / self.limits.max_feature_flags) * 100,
            },
            'scenario_id_cache_size': len(self.scenario_id_cache)
        }
    
    def reset_cache(self) -> None:
        """Reset internal caches (for testing)."""
        self.scenario_id_cache.clear()
        for key in self.seen_values:
            self.seen_values[key].clear()


# Global instance
_cardinality_controller = None


def get_metrics_controller() -> MetricsCardinalityController:
    """Get global metrics cardinality controller."""
    global _cardinality_controller
    if _cardinality_controller is None:
        _cardinality_controller = MetricsCardinalityController()
    return _cardinality_controller


def reset_metrics_controller() -> None:
    """Reset metrics controller (for testing)."""
    global _cardinality_controller
    _cardinality_controller = None