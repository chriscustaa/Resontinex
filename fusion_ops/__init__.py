"""
Fusion Operations Module
Utilities for managing fusion overlay system operations
"""

from .feature_flags import FusionFeatureFlags, get_feature_flags, reset_feature_flags
from .metrics import (
    FusionMetricsController,
    FusionMetricsCollector,
    get_metrics_collector,
    reset_metrics_collector
)
from .budget_analysis import analyze_budget_metrics
from .benchmark import FusionBenchmarkRunner, run_fusion_benchmark
from .budget_tripwire import BudgetTripwire, get_budget_tripwire, reset_tripwire_instance
from .performance_comparison import PerformanceComparator, compare_performance
from .metrics_controller import (
    MetricsCardinalityController,
    get_metrics_controller,
    reset_metrics_controller
)

__version__ = "0.1.1"
__all__ = [
    "FusionFeatureFlags",
    "get_feature_flags",
    "reset_feature_flags",
    "FusionMetricsController",
    "FusionMetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
    "analyze_budget_metrics",
    "FusionBenchmarkRunner",
    "run_fusion_benchmark",
    "BudgetTripwire",
    "get_budget_tripwire",
    "reset_tripwire_instance",
    "PerformanceComparator",
    "compare_performance",
    "MetricsCardinalityController",
    "get_metrics_controller",
    "reset_metrics_controller"
]