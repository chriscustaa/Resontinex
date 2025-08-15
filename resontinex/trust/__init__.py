"""
RESONTINEX Trust Gate System
Production-ready trust scoring with zero dependencies and operational excellence.
"""

from .composite_trust_gate import (
    CompositeTrustGate,
    TrustMonitor,
    NullMetricsClient,
    RiskTier,
    Decision,
    InputVector,
    route_to_resontinex,
    CalibrationAdapter,
)

__version__ = "1.0.0"
__all__ = [
    "CompositeTrustGate",
    "TrustMonitor", 
    "NullMetricsClient",
    "RiskTier",
    "Decision",
    "InputVector",
    "route_to_resontinex",
    "CalibrationAdapter",
]