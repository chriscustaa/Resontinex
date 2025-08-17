from __future__ import annotations
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Dict, Any, Optional
import json, pathlib
try:
    import yaml  # type: ignore
except Exception:  # optional dependency
    yaml = None

class RouterConfig(BaseModel):
    """Router configuration for AI model selection and routing strategies."""
    strategy: Literal["auto", "cost", "quality", "failover"] = "auto"
    models: List[str] = ["gpt-4o", "claude-3.5-sonnet", "local:llama3-8b"]
    default_route: str = "gpt-4o"
    timeout_seconds: float = 30.0
    max_retries: int = 3

class SLOConfig(BaseModel):
    """Service Level Objectives for production reliability."""
    p95_latency_ms: int = 1500
    p99_latency_ms: int = 3000
    max_errors_per_min: int = 5
    min_trust_score: float = 0.70
    circuit_breaker_threshold: int = 10
    circuit_breaker_timeout_seconds: int = 60

class OverlayConfig(BaseModel):
    """Overlay management for feature flags and A/B testing."""
    default_overlay: str = "production"
    overlays: Dict[str, Any] = Field(default_factory=dict)
    rollout_percentage: float = 0.0
    sticky_sessions: bool = True
    health_check_interval_seconds: int = 30

class DriftConfig(BaseModel):
    """Configuration for drift detection and monitoring."""
    threshold: float = 0.1
    window_size: int = 100
    detection_method: Literal["statistical", "semantic", "hybrid"] = "hybrid"
    alert_threshold: float = 0.15
    auto_rollback: bool = False

class GovernanceConfig(BaseModel):
    """Governance and compliance configuration."""
    enable_governance: bool = False
    energy_budget_limit: float = 1000.0
    quorum_threshold: float = 0.5
    review_threshold: float = 0.80
    audit_logging: bool = True
    compliance_mode: Literal["strict", "relaxed", "disabled"] = "relaxed"

class ObservabilityConfig(BaseModel):
    """Observability and monitoring configuration."""
    enable_metrics: bool = True
    enable_tracing: bool = False
    prometheus_port: int = 9090
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"
    structured_logging: bool = True

class RuntimeConfig(BaseModel):
    """Unified runtime configuration for Resontinex system."""
    router: RouterConfig = Field(default_factory=RouterConfig)
    slo: SLOConfig = Field(default_factory=SLOConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    # System-wide settings
    environment: Literal["development", "staging", "production"] = "production"
    version: str = "2.1.0"
    debug_mode: bool = False

    @classmethod
    def sample(cls) -> dict:
        """Generate a sample configuration dictionary."""
        return cls().dict()

    @classmethod
    def parse_file(cls, path: str) -> "RuntimeConfig":
        """Parse configuration from YAML or JSON file."""
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        txt = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            if not yaml:
                raise RuntimeError("PyYAML not installed; use .json or install `pyyaml`.")
            data = yaml.safe_load(txt) or {}
        else:
            data = json.loads(txt)
        return cls.parse_obj(data)

    def to_file(self, path: str, format: Literal["json", "yaml"] = "yaml") -> None:
        """Save configuration to file."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.dict()
        if format == "yaml" and yaml:
            p.write_text(yaml.safe_dump(data, indent=2), encoding="utf-8")
        else:
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def validate_production_ready(self) -> List[str]:
        """Validate configuration for production deployment."""
        issues = []
        
        if self.debug_mode and self.environment == "production":
            issues.append("Debug mode should not be enabled in production")
        
        if self.slo.p95_latency_ms > 5000:
            issues.append("P95 latency target too high for production (>5s)")
            
        if not self.observability.enable_metrics:
            issues.append("Metrics should be enabled for production monitoring")
            
        if self.governance.enable_governance and self.governance.energy_budget_limit <= 0:
            issues.append("Energy budget must be positive when governance is enabled")
            
        return issues