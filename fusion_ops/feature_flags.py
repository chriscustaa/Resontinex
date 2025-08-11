"""
Feature Flags Management for Fusion Overlay System
Handles feature flag configuration loading and evaluation with environment overrides
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path


class FusionFeatureFlags:
    """Manages feature flag configuration for fusion overlay system."""
    
    def __init__(self, config_file: str = "config/overlay_feature_flags.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.env_overrides = self._parse_env_overrides()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load feature flags configuration from YAML file."""
        try:
            if not self.config_file.exists():
                return self._default_config()
            
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load feature flags config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration when file is not available."""
        return {
            "features": {},
            "global": {
                "default_strategy": "baseline_only",
                "circuit_breaker": {"enabled": True}
            }
        }
    
    def _parse_env_overrides(self) -> List[str]:
        """Parse environment variable overrides for feature flags."""
        env_features = os.getenv('RESON_FEATURES', '')
        if not env_features:
            return []
        
        # Parse comma-separated feature list
        features = [f.strip() for f in env_features.split(',') if f.strip()]
        return features
    
    def is_feature_enabled(self, feature_name: str, scenario_context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled for the given context."""
        # Environment override takes highest priority
        if feature_name in self.env_overrides:
            return True
        
        # Check feature configuration
        features_config = self.config.get('features', {})
        feature_config = features_config.get(feature_name, {})
        
        if not feature_config:
            return False
        
        # Check rollout percentage
        rollout_pct = feature_config.get('rollout_percentage', 0)
        if rollout_pct <= 0:
            return False
        
        # For now, simple rollout - in production this would use consistent hashing
        if rollout_pct < 100:
            # Simple hash-based rollout (deterministic for same inputs)
            import hashlib
            hash_input = f"{feature_name}_{scenario_context.get('scenario_name', '')}" if scenario_context else feature_name
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
            if (hash_value % 100) >= rollout_pct:
                return False
        
        # Check scenario conditions if provided
        if scenario_context:
            conditions = feature_config.get('conditions', {})
            
            # Check scenario type matching
            required_types = conditions.get('scenario_types', [])
            if required_types:
                scenario_type = scenario_context.get('scenario_type', '')
                if not any(req_type in scenario_type for req_type in required_types):
                    return False
            
            # Check confidence threshold
            confidence_threshold = conditions.get('confidence_threshold', 0.0)
            scenario_confidence = scenario_context.get('confidence', 1.0)
            if scenario_confidence < confidence_threshold:
                return False
        
        return True
    
    def get_enabled_overlays(self, feature_name: str, scenario_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get list of enabled overlays for a feature."""
        if not self.is_feature_enabled(feature_name, scenario_context):
            return []
        
        features_config = self.config.get('features', {})
        feature_config = features_config.get(feature_name, {})
        
        return feature_config.get('enabled_overlays', [])
    
    def get_fallback_strategy(self, feature_name: str) -> str:
        """Get fallback strategy for a feature."""
        features_config = self.config.get('features', {})
        feature_config = features_config.get(feature_name, {})
        
        fallback = feature_config.get('fallback_strategy', 'baseline_only')
        return fallback
    
    def get_routing_decision(self, scenario_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get routing decision for a scenario based on all applicable feature flags."""
        scenario_type = scenario_context.get('scenario_type', '')
        enabled_features = []
        overlay_candidates = []
        
        # Check all features for applicability
        features_config = self.config.get('features', {})
        for feature_name, feature_config in features_config.items():
            if self.is_feature_enabled(feature_name, scenario_context):
                enabled_features.append(feature_name)
                overlays = self.get_enabled_overlays(feature_name, scenario_context)
                overlay_candidates.extend(overlays)
        
        # Environment overrides
        for env_feature in self.env_overrides:
            if env_feature in features_config:
                if env_feature not in enabled_features:
                    enabled_features.append(env_feature)
                overlays = features_config[env_feature].get('enabled_overlays', [])
                overlay_candidates.extend(overlays)
        
        # Remove duplicates while preserving order
        unique_overlays = []
        for overlay in overlay_candidates:
            if overlay not in unique_overlays:
                unique_overlays.append(overlay)
        
        return {
            'scenario_type': scenario_type,
            'enabled_features': enabled_features,
            'overlay_candidates': unique_overlays,
            'primary_overlay': unique_overlays[0] if unique_overlays else None,
            'fallback_strategy': self._get_global_fallback(),
            'env_overrides_active': bool(self.env_overrides)
        }
    
    def _get_global_fallback(self) -> str:
        """Get global fallback strategy."""
        global_config = self.config.get('global', {})
        return global_config.get('default_strategy', 'baseline_only')
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics collection configuration."""
        global_config = self.config.get('global', {})
        metrics_config = global_config.get('metrics', {})
        
        return {
            'enabled': metrics_config.get('enabled', True),
            'track_latency': metrics_config.get('track_latency', True),
            'track_quality': metrics_config.get('track_quality', True),
            'track_token_usage': metrics_config.get('track_token_usage', True)
        }
    
    def get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration."""
        global_config = self.config.get('global', {})
        cb_config = global_config.get('circuit_breaker', {})
        
        return {
            'enabled': cb_config.get('enabled', True),
            'failure_threshold': cb_config.get('failure_threshold', 5),
            'recovery_timeout': cb_config.get('recovery_timeout', 300)
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate feature flags configuration."""
        issues = []
        warnings = []
        
        # Check if config file exists
        if not self.config_file.exists():
            warnings.append(f"Config file not found: {self.config_file}")
        
        # Validate feature definitions
        features_config = self.config.get('features', {})
        for feature_name, feature_config in features_config.items():
            # Check required fields
            if 'enabled_overlays' not in feature_config:
                issues.append(f"Feature '{feature_name}' missing 'enabled_overlays'")
            
            # Check rollout percentage
            rollout_pct = feature_config.get('rollout_percentage', 0)
            if not 0 <= rollout_pct <= 100:
                issues.append(f"Feature '{feature_name}' has invalid rollout_percentage: {rollout_pct}")
        
        # Check environment overrides
        for env_feature in self.env_overrides:
            if env_feature not in features_config:
                warnings.append(f"Environment override '{env_feature}' not found in config")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'env_overrides': self.env_overrides,
            'features_count': len(features_config)
        }


# Singleton instance for global access
_feature_flags_instance = None


def get_feature_flags() -> FusionFeatureFlags:
    """Get global feature flags instance."""
    global _feature_flags_instance
    if _feature_flags_instance is None:
        _feature_flags_instance = FusionFeatureFlags()
    return _feature_flags_instance


def reset_feature_flags():
    """Reset feature flags instance (for testing)."""
    global _feature_flags_instance
    _feature_flags_instance = None