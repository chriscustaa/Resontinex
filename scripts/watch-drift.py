#!/usr/bin/env python3
"""
RESONTINEX Drift Detection Watchdog
Monitors version changes and triggers fusion re-distillation when needed.
"""

import os
import re
import json
import yaml
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import argparse
import time
import glob


@dataclass
class FileChange:
    """Represents a detected file change."""
    path: str
    old_hash: Optional[str]
    new_hash: str
    change_type: str  # 'modified', 'added', 'deleted'
    version_change: Optional[Tuple[str, str]] = None  # (old_version, new_version)


@dataclass
class DriftEvent:
    """Represents a drift detection event."""
    timestamp: str
    trigger_files: List[str]
    changes_detected: List[FileChange]
    version_changes: Dict[str, Tuple[str, str]]
    quality_gates_passed: bool
    actions_executed: List[str]
    performance_impact: Dict[str, float]
    rollback_occurred: bool = False


class VersionExtractor:
    """Extracts version information from various file types."""
    
    def __init__(self):
        self.version_patterns = {
            'readme.txt': [
                re.compile(r'Version:\s*(\d+\.\d+\.\d+)', re.IGNORECASE),
                re.compile(r'Stable tag:\s*(\d+\.\d+\.\d+)', re.IGNORECASE),
            ],
            'composer.lock': [
                re.compile(r'"version":\s*"(\d+\.\d+\.\d+)"'),
            ],
            'package-lock.json': [
                re.compile(r'"version":\s*"(\d+\.\d+\.\d+)"'),
            ],
            'requirements.txt': [
                re.compile(r'==(\d+\.\d+\.\d+)'),
            ]
        }
    
    def extract_version(self, file_path: str, content: str) -> Optional[str]:
        """Extract version from file content."""
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name.lower()
        
        # Choose patterns based on file type
        patterns = []
        if file_name in self.version_patterns:
            patterns = self.version_patterns[file_name]
        elif file_ext == '.json':
            patterns = self.version_patterns.get('package-lock.json', [])
        elif file_ext == '.txt':
            patterns = self.version_patterns.get('readme.txt', [])
        
        # Try each pattern
        for pattern in patterns:
            matches = pattern.findall(content)
            if matches:
                return matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return None
    
    def compare_versions(self, old_version: str, new_version: str) -> str:
        """Compare versions and return change type."""
        try:
            old_parts = [int(x) for x in old_version.split('.')]
            new_parts = [int(x) for x in new_version.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(old_parts), len(new_parts))
            old_parts.extend([0] * (max_len - len(old_parts)))
            new_parts.extend([0] * (max_len - len(new_parts)))
            
            if new_parts[0] > old_parts[0]:
                return 'major'
            elif new_parts[1] > old_parts[1]:
                return 'minor'
            elif new_parts[2] > old_parts[2]:
                return 'patch'
            else:
                return 'none'
        except (ValueError, IndexError):
            return 'unknown'


class DriftDetector:
    """Main drift detection engine."""
    
    def __init__(self, config_dir: str = "./configs/fusion", state_dir: str = "./build/drift"):
        self.config_dir = Path(config_dir)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.policy = self._load_drift_policy()
        self.version_extractor = VersionExtractor()
        
        # State management
        self.state_file = self.state_dir / "drift_state.json"
        self.history_file = self.state_dir / "drift_history.json"
        
    def _load_drift_policy(self) -> Dict[str, Any]:
        """Load drift detection policy."""
        policy_path = self.config_dir / "drift_policy.yaml"
        if policy_path.exists():
            with open(policy_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_state(self) -> Dict[str, Any]:
        """Load current state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'file_hashes': {},
            'version_cache': {},
            'last_check': None,
            'consecutive_failures': 0
        }
    
    def _save_state(self, state: Dict[str, Any]):
        """Save state to disk."""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            return ""
    
    def _expand_watch_patterns(self) -> List[str]:
        """Expand glob patterns into actual file paths."""
        watch_patterns = self.policy.get('watch_patterns', {})
        file_paths = []
        
        for pattern_name, pattern in watch_patterns.items():
            try:
                # Handle different pattern types
                if isinstance(pattern, str):
                    matches = glob.glob(pattern, recursive=True)
                    file_paths.extend(matches)
                elif isinstance(pattern, list):
                    for p in pattern:
                        matches = glob.glob(p, recursive=True)
                        file_paths.extend(matches)
            except Exception as e:
                print(f"Warning: Failed to expand pattern {pattern}: {e}")
        
        return list(set(file_paths))  # Remove duplicates
    
    def scan_for_changes(self) -> List[FileChange]:
        """Scan monitored files for changes."""
        state = self._load_state()
        current_hashes = state.get('file_hashes', {})
        current_versions = state.get('version_cache', {})
        
        changes = []
        new_hashes = {}
        new_versions = {}
        
        file_paths = self._expand_watch_patterns()
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                # File was deleted
                if file_path in current_hashes:
                    changes.append(FileChange(
                        path=file_path,
                        old_hash=current_hashes[file_path],
                        new_hash="",
                        change_type='deleted'
                    ))
                continue
            
            # Calculate current hash
            new_hash = self._calculate_file_hash(file_path)
            new_hashes[file_path] = new_hash
            
            old_hash = current_hashes.get(file_path)
            
            if old_hash is None:
                # New file
                change_type = 'added'
            elif old_hash != new_hash:
                # Modified file
                change_type = 'modified'
            else:
                # No change
                continue
            
            # Check for version changes
            version_change = None
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    new_version = self.version_extractor.extract_version(file_path, content)
                    
                    if new_version:
                        new_versions[file_path] = new_version
                        old_version = current_versions.get(file_path)
                        
                        if old_version and old_version != new_version:
                            version_change = (old_version, new_version)
            except Exception:
                pass
            
            changes.append(FileChange(
                path=file_path,
                old_hash=old_hash,
                new_hash=new_hash,
                change_type=change_type,
                version_change=version_change
            ))
        
        # Update state
        state['file_hashes'] = new_hashes
        state['version_cache'] = new_versions
        state['last_check'] = datetime.now(timezone.utc).isoformat()
        self._save_state(state)
        
        return changes
    
    def should_trigger_drift_action(self, changes: List[FileChange]) -> bool:
        """Determine if changes should trigger drift actions."""
        if not changes:
            return False
        
        version_change_triggers = self.policy.get('quality_gates', {}).get('version_change_triggers', {})
        
        # Check version changes
        for change in changes:
            if change.version_change:
                old_ver, new_ver = change.version_change
                change_type = self.version_extractor.compare_versions(old_ver, new_ver)
                
                if change_type == 'major' and version_change_triggers.get('major', True):
                    return True
                elif change_type == 'minor' and version_change_triggers.get('minor', True):
                    return True
                elif change_type == 'patch' and version_change_triggers.get('patch', False):
                    return True
        
        # Check for accumulated patch changes
        patch_changes = sum(1 for c in changes if c.version_change and 
                           self.version_extractor.compare_versions(*c.version_change) == 'patch')
        
        max_patch_changes = self.policy.get('detection_settings', {}).get('max_accumulated_patch_changes', 5)
        if patch_changes >= max_patch_changes:
            return True
        
        # Check for critical file changes
        critical_patterns = ['configs/fusion/', 'scripts/']
        for change in changes:
            if any(pattern in change.path for pattern in critical_patterns):
                return True
        
        return False
    
    def execute_drift_actions(self, changes: List[FileChange]) -> Tuple[List[str], Dict[str, float]]:
        """Execute drift response actions."""
        actions = self.policy.get('on_change', {})
        executed_actions = []
        performance_impact = {}
        
        for action_name, action_config in actions.items():
            try:
                print(f"Executing action: {action_name}")
                
                script = action_config.get('script')
                args = action_config.get('args', [])
                timeout_minutes = action_config.get('timeout_minutes', 30)
                depends_on = action_config.get('depends_on', [])
                
                # Check dependencies
                if depends_on and not all(dep in executed_actions for dep in depends_on):
                    print(f"Skipping {action_name}: dependencies not met")
                    continue
                
                # Execute script
                cmd = ['python', script] + args
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_minutes * 60,
                    cwd=os.getcwd()
                )
                
                if result.returncode == 0:
                    executed_actions.append(action_name)
                    print(f"Action {action_name} completed successfully")
                    
                    # Parse performance metrics from output if available
                    if 'improvement' in result.stdout.lower():
                        # Try to extract improvement metrics
                        improvement_match = re.search(r'improvement[:\s]+([+-]?\d+\.\d+)', result.stdout, re.IGNORECASE)
                        if improvement_match:
                            performance_impact[action_name] = float(improvement_match.group(1))
                else:
                    print(f"Action {action_name} failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"Action {action_name} timed out")
            except Exception as e:
                print(f"Error executing {action_name}: {e}")
        
        return executed_actions, performance_impact
    
    def check_quality_gates(self, performance_impact: Dict[str, float]) -> bool:
        """Check if quality gates are satisfied."""
        gates = self.policy.get('quality_gates', {})
        
        # Calculate overall improvement
        overall_improvement = sum(performance_impact.values()) / len(performance_impact) if performance_impact else 0.0
        
        min_specificity = gates.get('min_specificity_gain', 0.10)
        min_operationality = gates.get('min_operationality_gain', 0.08)
        max_degradation = gates.get('max_performance_degradation', -0.05)
        
        # Simple check - in production this would be more sophisticated
        if overall_improvement < max_degradation:
            return False
        
        if overall_improvement < min_specificity and overall_improvement < min_operationality:
            return False
        
        return True
    
    def run_drift_check(self) -> Optional[DriftEvent]:
        """Run a complete drift detection cycle."""
        print(f"Running drift detection at {datetime.now(timezone.utc).isoformat()}")
        
        # Check for disabled state
        state = self._load_state()
        if state.get('consecutive_failures', 0) >= self.policy.get('error_handling', {}).get('max_consecutive_failures', 3):
            print("Drift detection disabled due to consecutive failures")
            return None
        
        try:
            # Scan for changes
            changes = self.scan_for_changes()
            
            if not changes:
                print("No changes detected")
                return None
            
            print(f"Detected {len(changes)} file changes")
            for change in changes:
                print(f"  {change.change_type}: {change.path}")
                if change.version_change:
                    old_ver, new_ver = change.version_change
                    print(f"    Version: {old_ver} → {new_ver}")
            
            # Check if changes should trigger actions
            if not self.should_trigger_drift_action(changes):
                print("Changes do not meet trigger criteria")
                return None
            
            # Execute drift actions
            executed_actions, performance_impact = self.execute_drift_actions(changes)
            
            # Check quality gates
            quality_gates_passed = self.check_quality_gates(performance_impact)
            
            # Create drift event
            drift_event = DriftEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger_files=[c.path for c in changes],
                changes_detected=changes,
                version_changes={c.path: c.version_change for c in changes if c.version_change},
                quality_gates_passed=quality_gates_passed,
                actions_executed=executed_actions,
                performance_impact=performance_impact
            )
            
            # Save to history
            self._save_drift_event(drift_event)
            
            # Reset failure counter on success
            state['consecutive_failures'] = 0
            self._save_state(state)
            
            print(f"Drift detection completed: {len(executed_actions)} actions executed")
            print(f"Quality gates passed: {quality_gates_passed}")
            
            return drift_event
            
        except Exception as e:
            print(f"Drift detection failed: {e}")
            
            # Increment failure counter
            state['consecutive_failures'] = state.get('consecutive_failures', 0) + 1
            self._save_state(state)
            
            return None
    
    def _save_drift_event(self, event: DriftEvent):
        """Save drift event to history."""
        history = []
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # Add new event
        history.append(asdict(event))
        
        # Trim history
        max_events = self.policy.get('history', {}).get('max_drift_events_stored', 100)
        history = history[-max_events:]
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def run_continuous_monitoring(self, check_interval_hours: int = 6):
        """Run continuous drift monitoring."""
        print(f"Starting continuous drift monitoring (checking every {check_interval_hours} hours)")
        
        while True:
            try:
                self.run_drift_check()
                time.sleep(check_interval_hours * 3600)
            except KeyboardInterrupt:
                print("Stopping drift monitoring")
                break
            except Exception as e:
                print(f"Error in continuous monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes before retry


def main():
    parser = argparse.ArgumentParser(description="RESONTINEX Drift Detection Watchdog")
    parser.add_argument('--once', action='store_true', help="Run once and exit")
    parser.add_argument('--interval', type=int, default=6, help="Check interval in hours")
    parser.add_argument('--config-dir', default="./configs/fusion", help="Config directory")
    parser.add_argument('--state-dir', default="./build/drift", help="State directory")
    
    args = parser.parse_args()
    
    try:
        detector = DriftDetector(args.config_dir, args.state_dir)
        
        if args.once:
            event = detector.run_drift_check()
            if event:
                print("✓ Drift detection completed with changes")
            else:
                print("✓ Drift detection completed - no changes")
        else:
            detector.run_continuous_monitoring(args.interval)
            
    except Exception as e:
        print(f"✗ Drift detection failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())