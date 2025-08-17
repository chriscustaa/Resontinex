"""
RESONTINEX - Production-grade cognitive continuity engine

A comprehensive AI workflow orchestration system with intelligent overlay routing,
circuit breaker protection, and comprehensive drift detection.
"""

__version__ = "2.1.0"
__author__ = "RESONTINEX Team"
__email__ = "chris@custaa.com"
__license__ = "MIT"
__description__ = "Production-grade AI workflow orchestration system with intelligent overlay routing, circuit breaker protection, and comprehensive drift detection."

# Core module imports for easy access
from .scenario_manager import ScenarioManager
from .fusion_resilience import FusionResilientLoader, FusionSecurityValidator

# Version information for programmatic access
VERSION_INFO = {
    "version": __version__,
    "spec_version": "2.1.0",
    "compatibility": {
        "minimum_python": "3.10",
        "breaking_changes_from": ["1.x.x"],
        "deprecated_features": []
    },
    "features": {
        "energy_governance": True,
        "quorum_voting": True,
        "adaptive_thresholds": True,
        "circuit_breaking": True,
        "performance_monitoring": True
    }
}

def get_version_info():
    """
    Returns comprehensive version and feature information.
    
    Returns:
        dict: Version metadata including compatibility and feature flags
    """
    return VERSION_INFO.copy()

def check_compatibility(required_version: str) -> bool:
    """
    Check if current version meets minimum requirements.
    
    Args:
        required_version: Minimum required version string (e.g., "2.1.0")
        
    Returns:
        bool: True if current version is compatible
    """
    from packaging import version
    current = version.parse(__version__)
    required = version.parse(required_version)
    return current >= required

# Module-level configuration
DEFAULT_CONFIG = {
    "entropy_floor": 0.25,
    "trust_floor": 0.60,
    "energy_floor": 0.05,
    "timeout_ms": 150,
    "max_collapse_attempts": 3
}

__all__ = [
    "__version__",
    "VERSION_INFO",
    "ScenarioManager",
    "FusionResilientLoader",
    "FusionSecurityValidator",
    "get_version_info",
    "check_compatibility",
    "DEFAULT_CONFIG"
]