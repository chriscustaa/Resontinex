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
from .performance_comparison import PerformanceComparator, compare_performance

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
    "PerformanceComparator",
    "compare_performance"
]