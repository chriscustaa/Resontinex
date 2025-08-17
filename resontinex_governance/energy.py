"""Energy Budget Management for AI Operations."""

import logging
from typing import Dict, Any, Optional
import time

class EnergyLedger:
    """Tracks and manages energy budget for AI operations."""
    
    def __init__(self, budget: float, review_threshold: float = 0.8):
        self.budget = float(budget)
        self.review_threshold = float(review_threshold)
        self.spent = 0.0
        self.transactions: Dict[str, Dict[str, Any]] = {}
        self._log = logging.getLogger("resontinex.energy")
        
    @property
    def available(self) -> float:
        """Return available budget remaining."""
        return max(self.budget - self.spent, 0.0)
    
    @property
    def utilization_ratio(self) -> float:
        """Return budget utilization as a ratio (0.0 to 1.0+)."""
        if self.budget <= 0:
            return 0.0
        return self.spent / self.budget
    
    def allocate(self, tx_id: str, cost: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Allocate energy budget for a transaction.
        
        Args:
            tx_id: Unique transaction identifier
            cost: Energy cost to allocate
            metadata: Optional transaction metadata
            
        Returns:
            bool: True if allocation successful, False if budget exceeded
        """
        cost = float(cost)
        new_spent = self.spent + cost
        
        if new_spent > self.budget:
            self._log.warning(
                "Budget exceeded: transaction=%s cost=%.2f spent=%.2f budget=%.2f", 
                tx_id, cost, new_spent, self.budget
            )
            return False
            
        # Record successful allocation
        self.spent = new_spent
        self.transactions[tx_id] = {
            "cost": cost,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self._log.info(
            "Energy allocated: transaction=%s cost=%.2f available=%.2f", 
            tx_id, cost, self.available
        )
        
        return True
    
    def deallocate(self, tx_id: str) -> bool:
        """
        Deallocate energy from a previously allocated transaction.
        
        Args:
            tx_id: Transaction identifier to deallocate
            
        Returns:
            bool: True if deallocation successful
        """
        if tx_id not in self.transactions:
            self._log.warning("Cannot deallocate unknown transaction: %s", tx_id)
            return False
            
        cost = self.transactions[tx_id]["cost"]
        self.spent = max(0.0, self.spent - cost)
        del self.transactions[tx_id]
        
        self._log.info("Energy deallocated: transaction=%s cost=%.2f", tx_id, cost)
        return True
    
    def needs_review(self) -> bool:
        """Check if budget utilization requires human review."""
        return self.budget > 0 and self.utilization_ratio >= self.review_threshold
    
    def get_status(self) -> Dict[str, Any]:
        """Get current budget status summary."""
        return {
            "budget": self.budget,
            "spent": self.spent,
            "available": self.available,
            "utilization_ratio": self.utilization_ratio,
            "needs_review": self.needs_review(),
            "active_transactions": len(self.transactions),
            "review_threshold": self.review_threshold
        }