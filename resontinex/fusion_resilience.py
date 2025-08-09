#!/usr/bin/env python3
"""
RESONTINEX Fusion Runtime Resilience
Production-grade error handling, fallback mechanisms, and observability for fusion overlay system.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib


@dataclass
class FusionMetrics:
    """Fusion system metrics for observability."""
    decisions_made: int = 0
    overlay_applied: bool = False
    delta_tokens: int = 0
    errors_count: int = 0
    last_error_timestamp: Optional[str] = None
    overlay_load_success: bool = False
    schema_validation_ok: bool = False
    fallback_activations: int = 0
    
    
class FusionResilientLoader:
    """Production-grade resilient loader for fusion configuration with comprehensive fallback."""
    
    def __init__(self, config_dir: str = "./configs/fusion"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger("fusion.resilience")
        self.metrics = FusionMetrics()
        
        # Embedded minimal overlay as ultimate fallback
        self.embedded_minimal_overlay = {
            "ENTROPY_REDUCTION_TARGET": "0.50",
            "CONTINUITY_ENFORCEMENT": "basic_thread",
            "TRUST_SCORING_MODEL": "simple_alignment",
            "PRIMARY_MODEL_SELECTION": "single_model",
            "FUSION_MODE": "baseline_only",
            "VOTING_POWER_MAP": "single:1",
            "ARBITRATION_TIMEOUT_MS": "300",
            "FUSION_OVERLAY_VERSION": "fallback",
            "TRUST_FLOOR": "0.40",
            "ENTROPY_FLOOR": "0.30"
        }
        
        self._configure_logging()

    def _configure_logging(self):
        """Configure structured logging for fusion operations."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"component": "fusion.resilience", "message": "%(message)s", '
                '"metrics": %(metrics)s}'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _log_with_metrics(self, level: str, message: str, extra_data: Optional[Dict] = None):
        """Log message with current metrics data."""
        metrics_data = asdict(self.metrics)
        if extra_data:
            metrics_data.update(extra_data)
        
        getattr(self.logger, level)(message, extra={'metrics': json.dumps(metrics_data)})

    def _validate_overlay_schema(self, overlay_data: Dict[str, str]) -> bool:
        """Validate overlay data against required schema."""
        required_keys = [
            "ENTROPY_REDUCTION_TARGET",
            "CONTINUITY_ENFORCEMENT",
            "TRUST_SCORING_MODEL",
            "PRIMARY_MODEL_SELECTION",
            "FUSION_MODE",
            "VOTING_POWER_MAP",
            "ARBITRATION_TIMEOUT_MS",
            "FUSION_OVERLAY_VERSION"
        ]
        
        try:
            missing_keys = [key for key in required_keys if key not in overlay_data]
            if missing_keys:
                self._log_with_metrics('error', f"Schema validation failed: missing keys {missing_keys}")
                return False
            
            # Validate specific value formats
            entropy_target = float(overlay_data.get("ENTROPY_REDUCTION_TARGET", "0"))
            if not 0.0 <= entropy_target <= 1.0:
                self._log_with_metrics('error', f"Invalid entropy target: {entropy_target}")
                return False
            
            timeout_ms = int(overlay_data.get("ARBITRATION_TIMEOUT_MS", "0"))
            if timeout_ms <= 0:
                self._log_with_metrics('error', f"Invalid timeout: {timeout_ms}")
                return False
            
            self.metrics.schema_validation_ok = True
            return True
            
        except (ValueError, TypeError) as e:
            self._log_with_metrics('error', f"Schema validation error: {e}")
            return False

    def _load_from_env(self) -> Optional[Dict[str, str]]:
        """Load fusion overlay from environment variables."""
        try:
            env_overlay = {}
            fusion_env_prefix = "RESONTINEX_FUSION_"
            
            for key, value in os.environ.items():
                if key.startswith(fusion_env_prefix):
                    config_key = key[len(fusion_env_prefix):]
                    env_overlay[config_key] = value
            
            if env_overlay:
                if self._validate_overlay_schema(env_overlay):
                    self._log_with_metrics('info', f"Loaded fusion overlay from environment: {len(env_overlay)} keys")
                    return env_overlay
                else:
                    self._log_with_metrics('warning', "Environment overlay failed schema validation")
            
            return None
            
        except Exception as e:
            self._log_with_metrics('error', f"Environment overlay load failed: {e}")
            self.metrics.errors_count += 1
            self.metrics.last_error_timestamp = datetime.now(timezone.utc).isoformat()
            return None

    def _load_from_config(self) -> Optional[Dict[str, str]]:
        """Load fusion overlay from configuration files."""
        try:
            overlay_files = list(self.config_dir.glob("fusion_overlay.*.txt"))
            if not overlay_files:
                self._log_with_metrics('warning', "No fusion overlay files found in config directory")
                return None
            
            # Use the most recent overlay file
            latest_overlay = max(overlay_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_overlay, 'r', encoding='utf-8') as f:
                content = f.read()
            
            overlay_data = {}
            for line in content.split('\n'):
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    overlay_data[key.strip()] = value.strip()
            
            if self._validate_overlay_schema(overlay_data):
                self._log_with_metrics('info', f"Loaded fusion overlay from config: {latest_overlay.name}")
                return overlay_data
            else:
                self._log_with_metrics('error', f"Config overlay failed schema validation: {latest_overlay.name}")
                return None
                
        except Exception as e:
            self._log_with_metrics('error', f"Config overlay load failed: {e}")
            self.metrics.errors_count += 1
            self.metrics.last_error_timestamp = datetime.now(timezone.utc).isoformat()
            return None

    def _load_embedded_minimal(self) -> Dict[str, str]:
        """Load embedded minimal overlay as ultimate fallback."""
        self._log_with_metrics('warning', "Using embedded minimal overlay (ultimate fallback)")
        self.metrics.fallback_activations += 1
        return self.embedded_minimal_overlay.copy()

    def load_fusion_overlay(self) -> Tuple[Dict[str, str], str]:
        """
        Load fusion overlay with comprehensive fallback strategy.
        
        Fallback order: ENV → configs/fusion → embedded_minimal_overlay
        
        Returns:
            Tuple[Dict[str, str], str]: (overlay_data, source_description)
        """
        start_time = time.time()
        
        # Try environment variables first
        overlay_data = self._load_from_env()
        if overlay_data:
            self.metrics.overlay_applied = True
            self.metrics.overlay_load_success = True
            load_time_ms = int((time.time() - start_time) * 1000)
            self._log_with_metrics('info', f"Fusion overlay loaded from ENV in {load_time_ms}ms")
            return overlay_data, "environment_variables"
        
        # Try configuration files
        overlay_data = self._load_from_config()
        if overlay_data:
            self.metrics.overlay_applied = True
            self.metrics.overlay_load_success = True
            load_time_ms = int((time.time() - start_time) * 1000)
            self._log_with_metrics('info', f"Fusion overlay loaded from config in {load_time_ms}ms")
            return overlay_data, "configuration_files"
        
        # Ultimate fallback: embedded minimal overlay
        overlay_data = self._load_embedded_minimal()
        self.metrics.overlay_applied = False  # This is fallback, not intended overlay
        load_time_ms = int((time.time() - start_time) * 1000)
        self._log_with_metrics('warning', f"Fusion overlay loaded from embedded fallback in {load_time_ms}ms")
        return overlay_data, "embedded_minimal_fallback"

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        return {
            "status": "healthy" if self.metrics.overlay_load_success and self.metrics.schema_validation_ok else "degraded",
            "overlay_loaded": self.metrics.overlay_load_success,
            "schema_ok": self.metrics.schema_validation_ok,
            "decisions_made": self.metrics.decisions_made,
            "overlay_applied": self.metrics.overlay_applied,
            "errors_count": self.metrics.errors_count,
            "last_error_timestamp": self.metrics.last_error_timestamp,
            "fallback_activations": self.metrics.fallback_activations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def emit_metrics(self) -> Dict[str, Any]:
        """Emit metrics for external monitoring systems."""
        return {
            "fusion.decisions": self.metrics.decisions_made,
            "fusion.overlay_applied": int(self.metrics.overlay_applied),
            "fusion.delta_tokens": self.metrics.delta_tokens,
            "fusion.errors": self.metrics.errors_count,
            "fusion.fallback_activations": self.metrics.fallback_activations,
            "fusion.schema_validation_ok": int(self.metrics.schema_validation_ok),
            "fusion.overlay_load_success": int(self.metrics.overlay_load_success),
            "timestamp": time.time()
        }

    def record_decision(self, token_delta: int = 0):
        """Record a fusion decision for metrics tracking."""
        self.metrics.decisions_made += 1
        self.metrics.delta_tokens += token_delta
        
        if self.metrics.decisions_made % 100 == 0:  # Log every 100 decisions
            self._log_with_metrics('info', f"Fusion decisions milestone: {self.metrics.decisions_made}")

    def handle_parse_error(self, error: Exception, context: str = "unknown"):
        """Handle parsing errors with graceful degradation."""
        self.metrics.errors_count += 1
        self.metrics.last_error_timestamp = datetime.now(timezone.utc).isoformat()
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self._log_with_metrics('error', f"Parse error in {context}: {error}", error_details)
        
        # If this is a critical error, disable overlay and continue with baseline
        if isinstance(error, (json.JSONDecodeError, KeyError, ValueError)):
            self.metrics.overlay_applied = False
            self._log_with_metrics('warning', "Disabling overlay due to parse error, continuing with baseline")
            return self._load_embedded_minimal(), "error_fallback"
        
        return None


class FusionSecurityValidator:
    """Security validation for fusion configuration data."""
    
    def __init__(self):
        self.logger = logging.getLogger("fusion.security")
        
        # PII detection patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b'
        }

    def validate_no_pii(self, data: Any, context: str = "unknown") -> Tuple[bool, List[str]]:
        """Validate that data contains no PII."""
        violations = []
        data_str = json.dumps(data) if not isinstance(data, str) else data
        
        for pii_type, pattern in self.pii_patterns.items():
            import re
            matches = re.findall(pattern, data_str, re.IGNORECASE)
            if matches:
                # Filter out obvious false positives
                actual_violations = []
                for match in matches:
                    if pii_type == 'email' and any(domain in match for domain in ['example.com', 'test.com']):
                        continue
                    if pii_type == 'api_key' and len(match) < 24:
                        continue
                    actual_violations.append(match)
                
                if actual_violations:
                    violations.append(f"{pii_type}: {len(actual_violations)} instances in {context}")
        
        if violations:
            self.logger.error(f"PII validation failed for {context}: {violations}")
        
        return len(violations) == 0, violations

    def add_provenance_note(self, ledger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add provenance and license information to ledger data."""
        if 'metadata' not in ledger_data:
            ledger_data['metadata'] = {}
        
        ledger_data['metadata'].update({
            'provenance': {
                'generated_by': 'RESONTINEX Fusion Engine',
                'license': 'Proprietary - RESONTINEX AI Systems',
                'data_classification': 'Internal Use Only',
                'last_security_scan': datetime.now(timezone.utc).isoformat(),
                'pii_validated': True,
                'schema_version': ledger_data.get('schema_version', 'unknown')
            }
        })
        
        return ledger_data


# Global singleton instances for production use
_fusion_loader = None
_security_validator = None


def get_fusion_loader(config_dir: str = "./configs/fusion") -> FusionResilientLoader:
    """Get singleton fusion loader instance."""
    global _fusion_loader
    if _fusion_loader is None:
        _fusion_loader = FusionResilientLoader(config_dir)
    return _fusion_loader


def get_security_validator() -> FusionSecurityValidator:
    """Get singleton security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = FusionSecurityValidator()
    return _security_validator


def load_fusion_configuration(config_dir: str = "./configs/fusion") -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Load fusion configuration with full resilience and observability.
    
    Returns:
        Tuple[Dict[str, str], Dict[str, Any]]: (overlay_config, health_status)
    """
    loader = get_fusion_loader(config_dir)
    overlay_config, source = loader.load_fusion_overlay()
    health_status = loader.get_health_status()
    
    # Security validation
    validator = get_security_validator()
    pii_valid, violations = validator.validate_no_pii(overlay_config, f"fusion_overlay_{source}")
    
    if not pii_valid:
        loader.handle_parse_error(
            ValueError(f"PII violations detected: {violations}"), 
            "security_validation"
        )
        # Use minimal embedded overlay for security
        overlay_config, source = loader._load_embedded_minimal(), "security_fallback"
    
    return overlay_config, health_status


if __name__ == "__main__":
    # Demo/test usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    config_dir = sys.argv[1] if len(sys.argv) > 1 else "./configs/fusion"
    
    try:
        overlay_config, health = load_fusion_configuration(config_dir)
        
        print("Fusion configuration loaded successfully:")
        print(f"  Source: {health.get('status', 'unknown')}")
        print(f"  Keys loaded: {len(overlay_config)}")
        print(f"  Health status: {health.get('status', 'unknown')}")
        
        # Emit metrics
        loader = get_fusion_loader(config_dir)
        metrics = loader.emit_metrics()
        print(f"  Metrics: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        print(f"Error loading fusion configuration: {e}")
        sys.exit(1)