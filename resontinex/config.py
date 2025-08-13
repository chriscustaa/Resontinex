# resontinex/config.py
from pydantic import BaseModel, Field
from typing import Dict, Any

class SLOConfig(BaseModel):
    latency_ms: int = 500
    error_rate: float = 0.01

class DriftConfig(BaseModel):
    threshold: float = 0.1

class OverlayConfig(BaseModel):
    default_overlay: str = "default"
    overlays: Dict[str, Any] = Field(default_factory=dict)

class RouterConfig(BaseModel):
    default_route: str = "default"

class RuntimeConfig(BaseModel):
    router: RouterConfig = Field(default_factory=RouterConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    enable_governance: bool = False