"""
Budget Tripwire Auto-Downgrade Module
Monitors token delta metrics and automatically downgrades to default overlay params
when budget thresholds are breached for consecutive runs.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field


from dataclasses import field

@dataclass
class TripwireState:
    """State tracking for budget tripwire mechanism."""
    consecutive_breaches: int = 0
    last_breach_timestamp: Optional[float] = None
    downgrade_active: bool = False
    downgrade_activated_at: Optional[float] = None
    breach_history: List[Dict[str, Any]] = field(default_factory=list)


class BudgetTripwire:
    """Manages automatic downgrade when budget thresholds are breached."""
    
    def __init__(self, 
                 state_file: str = ".fusion_tripwire_state.json",
                 breach_threshold: float = 12.0,
                 consecutive_limit: int = 3):
        self.state_file = Path(state_file)
        self.breach_threshold = breach_threshold
        self.consecutive_limit = consecutive_limit
        self.state = self._load_state()
        
    def _load_state(self) -> TripwireState:
        """Load tripwire state from disk."""
        if not self.state_file.exists():
            return TripwireState()
            
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            return TripwireState(
                consecutive_breaches=data.get('consecutive_breaches', 0),
                last_breach_timestamp=data.get('last_breach_timestamp'),
                downgrade_active=data.get('downgrade_active', False),
                downgrade_activated_at=data.get('downgrade_activated_at'),
                breach_history=data.get('breach_history', [])
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Reset state on corruption
            return TripwireState()
    
    def _save_state(self) -> None:
        """Save tripwire state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception:
            # Fail silently to avoid disrupting main workflow
            pass
    
    def check_budget_breach(self, token_delta: float, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if budget threshold is breached and update tripwire state.
        
        Returns:
            Dict with tripwire decision and metadata
        """
        current_time = time.time()
        is_breach = token_delta > self.breach_threshold
        
        result = {
            'is_breach': is_breach,
            'token_delta': token_delta,
            'threshold': self.breach_threshold,
            'consecutive_breaches': self.state.consecutive_breaches,
            'downgrade_triggered': False,
            'downgrade_active': self.state.downgrade_active,
            'timestamp': current_time
        }
        
        if is_breach:
            # Record breach
            breach_record = {
                'timestamp': current_time,
                'token_delta': token_delta,
                'context': context or {}
            }
            
            self.state.breach_history.append(breach_record)
            self.state.consecutive_breaches += 1
            self.state.last_breach_timestamp = current_time
            
            # Trim breach history to last 10 entries
            if len(self.state.breach_history) > 10:
                self.state.breach_history = self.state.breach_history[-10:]
            
            # Check if we need to trigger downgrade
            if (self.state.consecutive_breaches >= self.consecutive_limit and 
                not self.state.downgrade_active):
                
                self.state.downgrade_active = True
                self.state.downgrade_activated_at = current_time
                result['downgrade_triggered'] = True
                result['downgrade_reason'] = (
                    f"Token delta exceeded {self.breach_threshold}% for "
                    f"{self.state.consecutive_breaches} consecutive runs"
                )
                
        else:
            # No breach - reset consecutive counter but keep downgrade active
            # if it was previously activated
            self.state.consecutive_breaches = 0
            
        result['consecutive_breaches'] = self.state.consecutive_breaches
        result['downgrade_active'] = self.state.downgrade_active
        
        self._save_state()
        return result
    
    def get_overlay_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get overlay parameters, applying downgrade if active.
        
        Args:
            base_params: Base overlay parameters
            
        Returns:
            Modified parameters (defaults applied if downgraded)
        """
        if not self.state.downgrade_active:
            return base_params
            
        # Apply default parameters (minimal overlay)
        downgraded_params = {
            "ENTROPY_REDUCTION_TARGET": "0.50",
            "CONTINUITY_ENFORCEMENT": "basic_thread", 
            "TRUST_SCORING_MODEL": "simple_alignment",
            "PRIMARY_MODEL_SELECTION": "single_model",
            "FUSION_MODE": "baseline_only",
            "VOTING_POWER_MAP": "single:1",
            "ARBITRATION_TIMEOUT_MS": "300",
            "FUSION_OVERLAY_VERSION": base_params.get("FUSION_OVERLAY_VERSION", "downgrade"),
            "TRUST_FLOOR": "0.40",
            "ENTROPY_FLOOR": "0.30",
            # Preserve essential metadata
            "BUDGET_DOWNGRADE_ACTIVE": "true",
            "BUDGET_DOWNGRADE_TIMESTAMP": str(self.state.downgrade_activated_at or time.time())
        }
        
        return downgraded_params
    
    def get_metrics_tags(self) -> Dict[str, Any]:
        """Get metrics tags for telemetry."""
        return {
            'fusion.budget_downgrade': self.state.downgrade_active,
            'fusion.consecutive_breaches': self.state.consecutive_breaches,
            'fusion.breach_threshold': self.breach_threshold,
            'fusion.tripwire_state': 'active' if self.state.downgrade_active else 'monitoring'
        }
    
    def reset_downgrade(self) -> Dict[str, Any]:
        """
        Manually reset the downgrade state.
        
        Returns:
            Status of reset operation
        """
        was_active = self.state.downgrade_active
        
        self.state.downgrade_active = False
        self.state.downgrade_activated_at = None
        self.state.consecutive_breaches = 0
        
        self._save_state()
        
        return {
            'reset': True,
            'was_active': was_active,
            'timestamp': time.time(),
            'message': 'Budget tripwire downgrade reset successfully'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tripwire status."""
        return {
            'downgrade_active': self.state.downgrade_active,
            'consecutive_breaches': self.state.consecutive_breaches,
            'breach_threshold': self.breach_threshold,
            'consecutive_limit': self.consecutive_limit,
            'last_breach_timestamp': self.state.last_breach_timestamp,
            'downgrade_activated_at': self.state.downgrade_activated_at,
            'recent_breaches': self.state.breach_history[-5:]
        }


# Global instance for singleton access
_tripwire_instance = None


def get_budget_tripwire() -> BudgetTripwire:
    """Get global budget tripwire instance."""
    global _tripwire_instance
    if _tripwire_instance is None:
        _tripwire_instance = BudgetTripwire()
    return _tripwire_instance


def reset_tripwire_instance():
    """Reset tripwire instance (for testing)."""
    global _tripwire_instance
    _tripwire_instance = None