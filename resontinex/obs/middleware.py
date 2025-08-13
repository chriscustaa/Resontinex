# resontinex/obs/middleware.py
import time, json, sys
from contextlib import contextmanager

try:
  from prometheus_client import Counter, Histogram
  LAT = Histogram("rnx_latency_seconds","Resontinex op latency",{ "op":"" })
  ERR = Counter("rnx_errors_total","Resontinex errors",{ "op":"" })
except Exception:
    LAT=ERR=None

@contextmanager
def measure(op="route"):
    t=time.perf_counter()
    try:
        yield
    except Exception as e:
        if ERR:
            ERR.labels(op=op).inc()
        print(json.dumps({"event":"error","op":op,"error":str(e)}), file=sys.stderr)
        raise
    finally:
        d=time.perf_counter()-t
        if LAT:
            LAT.labels(op=op).observe(d)
        print(json.dumps({"event":"timing","op":op,"seconds":d}), file=sys.stdout)