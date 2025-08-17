"""Quorum-based decision making for AI governance."""

from typing import List, Union, Dict, Any, Optional
import logging

Vote = Union[bool, str]  # True/False for approval, or "VETO" for veto power

class QuorumVoter:
    """Implements quorum-based voting for governance decisions."""
    
    def __init__(self, threshold: float = 0.5, tie_breaker: bool = False, veto_power: bool = False):
        """
        Initialize quorum voter.
        
        Args:
            threshold: Minimum ratio of positive votes required (0.0 to 1.0)
            tie_breaker: How to handle exact threshold matches
            veto_power: Whether any single "VETO" vote blocks the decision
        """
        self.threshold = float(threshold)
        self.tie_breaker = bool(tie_breaker)
        self.veto_power = bool(veto_power)
        self._log = logging.getLogger("resontinex.quorum")
        
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {self.threshold}")
    
    def decide(self, votes: List[Vote], decision_id: Optional[str] = None) -> bool:
        """
        Make a decision based on quorum voting.
        
        Args:
            votes: List of votes (True/False for approval/rejection, "VETO" for veto)
            decision_id: Optional identifier for logging
            
        Returns:
            bool: True if decision passes, False otherwise
        """
        if not votes:
            self._log.warning("No votes provided for decision %s", decision_id or "unknown")
            return False
        
        # Check for veto power
        if self.veto_power and any(v == "VETO" for v in votes):
            self._log.info("Decision %s vetoed", decision_id or "unknown")
            return False
        
        # Filter to boolean votes only
        bool_votes = [v for v in votes if isinstance(v, bool)]
        if not bool_votes:
            self._log.warning("No valid boolean votes for decision %s", decision_id or "unknown")
            return False
        
        # Calculate approval ratio
        approval_count = sum(bool_votes)
        total_votes = len(bool_votes)
        approval_ratio = approval_count / total_votes
        
        # Apply decision logic
        if approval_ratio > self.threshold:
            result = True
        elif approval_ratio == self.threshold:
            result = self.tie_breaker
        else:
            result = False
        
        self._log.info(
            "Quorum decision %s: %d/%d votes (%.2f%%) -> %s", 
            decision_id or "unknown",
            approval_count, 
            total_votes, 
            approval_ratio * 100,
            "PASS" if result else "FAIL"
        )
        
        return result
    
    def get_vote_summary(self, votes: List[Vote]) -> Dict[str, Any]:
        """Get detailed voting summary."""
        bool_votes = [v for v in votes if isinstance(v, bool)]
        veto_count = sum(1 for v in votes if v == "VETO")
        
        if bool_votes:
            approval_count = sum(bool_votes)
            total_votes = len(bool_votes)
            approval_ratio = approval_count / total_votes
        else:
            approval_count = 0
            total_votes = 0
            approval_ratio = 0.0
        
        return {
            "total_votes": len(votes),
            "boolean_votes": total_votes,
            "approval_count": approval_count,
            "rejection_count": total_votes - approval_count,
            "veto_count": veto_count,
            "approval_ratio": approval_ratio,
            "threshold": self.threshold,
            "would_pass": self.decide(votes),
            "invalid_votes": len(votes) - total_votes - veto_count
        }

class MultiStageQuorum:
    """Multi-stage quorum for complex governance decisions."""
    
    def __init__(self, stages: List[QuorumVoter]):
        """
        Initialize multi-stage quorum.
        
        Args:
            stages: List of QuorumVoter instances representing decision stages
        """
        self.stages = stages
        self._log = logging.getLogger("resontinex.multiquorum")
    
    def decide(self, stage_votes: List[List[Vote]], decision_id: Optional[str] = None) -> bool:
        """
        Make decision through multiple quorum stages.
        
        Args:
            stage_votes: List of vote lists, one per stage
            decision_id: Optional identifier for logging
            
        Returns:
            bool: True if all stages pass, False otherwise
        """
        if len(stage_votes) != len(self.stages):
            raise ValueError(f"Expected {len(self.stages)} vote lists, got {len(stage_votes)}")
        
        for i, (stage, votes) in enumerate(zip(self.stages, stage_votes)):
            stage_id = f"{decision_id or 'unknown'}/stage-{i+1}"
            if not stage.decide(votes, stage_id):
                self._log.info("Multi-stage decision %s failed at stage %d", decision_id or "unknown", i+1)
                return False
        
        self._log.info("Multi-stage decision %s passed all %d stages", decision_id or "unknown", len(self.stages))
        return True