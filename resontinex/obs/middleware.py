"""Production-grade observability middleware with Prometheus metrics."""
import time, json, sys
from contextlib import contextmanager
from typing import Dict, Any, Optional

try:
    from prometheus_client import Counter, Histogram, Gauge
    # Core metrics for production monitoring
    LATENCY = Histogram("rnx_latency_seconds", "Operation latency", ["op", "status"])
    ERRORS = Counter("rnx_errors_total", "Total errors", ["op", "error_type"])
    CIRCUIT_TRIPS = Counter("rnx_circuit_trips_total", "Circuit breaker trips", ["circuit"])
    OVERLAY_FALLBACKS = Counter("rnx_overlay_fallbacks_total", "Overlay fallback events", ["from_overlay", "to_overlay"])
    ACTIVE_OPERATIONS = Gauge("rnx_active_operations", "Active operations", ["op"])
    TRUST_SCORE = Gauge("rnx_trust_score", "Current trust score", ["component"])
except Exception:
    LATENCY=ERRORS=CIRCUIT_TRIPS=OVERLAY_FALLBACKS=ACTIVE_OPERATIONS=TRUST_SCORE=None

@contextmanager
def measure(op: str = "route", metadata: Optional[Dict[str, Any]] = None):
    """Enhanced measurement context with comprehensive metrics capture."""
    start_time = time.perf_counter()
    if ACTIVE_OPERATIONS: ACTIVE_OPERATIONS.labels(op=op).inc()
    
    status = "success"
    error_type = None
    try:
        yield
    except Exception as e:
        status = "error"
        error_type = type(e).__name__
        if ERRORS: ERRORS.labels(op=op, error_type=error_type).inc()
        print(json.dumps({"event":"error","op":op,"error":str(e),"error_type":error_type}), file=sys.stderr)
        raise
    finally:
        duration = time.perf_counter() - start_time
        if LATENCY: LATENCY.labels(op=op, status=status).observe(duration)
        if ACTIVE_OPERATIONS: ACTIVE_OPERATIONS.labels(op=op).dec()
        
        log_data = {"event":"timing","op":op,"seconds":duration,"status":status}
        if metadata: log_data.update(metadata)
        print(json.dumps(log_data), file=sys.stdout)

def record_circuit_trip(circuit_name: str):
    """Record circuit breaker trip event."""
    if CIRCUIT_TRIPS: CIRCUIT_TRIPS.labels(circuit=circuit_name).inc()
    print(json.dumps({"event":"circuit_trip","circuit":circuit_name,"timestamp":time.time()}))

def record_overlay_fallback(from_overlay: str, to_overlay: str):
    """Record overlay fallback event."""
    if OVERLAY_FALLBACKS: OVERLAY_FALLBACKS.labels(from_overlay=from_overlay, to_overlay=to_overlay).inc()
    print(json.dumps({"event":"overlay_fallback","from":from_overlay,"to":to_overlay,"timestamp":time.time()}))

def update_trust_score(component: str, score: float):
    """Update trust score gauge."""
    if TRUST_SCORE: TRUST_SCORE.labels(component=component).set(score)