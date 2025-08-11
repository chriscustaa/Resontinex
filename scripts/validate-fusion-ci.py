#!/usr/bin/env python3
"""
RESONTINEX Fusion CI Validation
Comprehensive validation for fusion configuration integrity and security compliance.
"""

import os
import json
import sys
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import argparse
import jsonschema
import subprocess
from datetime import datetime


class FusionCIValidator:
    """CI validation engine for fusion system integrity and compliance."""
    
    def __init__(self, config_dir: str = "./configs/fusion", cache_dir: str = "./.ci_cache"):
        self.config_dir = Path(config_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # PII detection patterns (production-grade denylist)
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'api_key': re.compile(r'\b[A-Za-z0-9]{20,}\b'),
            'password_field': re.compile(r'\b(password|passwd|pwd)\s*[:=]\s*["\']?[^"\'\s]+["\']?', re.IGNORECASE),
            'token': re.compile(r'\b(token|secret|key)\s*[:=]\s*["\']?[A-Za-z0-9+/=]{16,}["\']?', re.IGNORECASE)
        }
        
        self.validation_errors = []
        self.validation_warnings = []

    def log_error(self, message: str):
        """Log validation error."""
        self.validation_errors.append(message)
        print(f"ERROR: {message}")

    def log_warning(self, message: str):
        """Log validation warning."""
        self.validation_warnings.append(message)
        print(f"WARNING: {message}")

    def load_json_schema(self) -> Optional[Dict[str, Any]]:
        """Load the capability profile JSON schema."""
        schema_path = self.config_dir / "capability_profile.schema.json"
        
        if not schema_path.exists():
            self.log_error(f"Schema file not found: {schema_path}")
            return None
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            return schema
        except Exception as e:
            self.log_error(f"Failed to load schema: {e}")
            return None

    def validate_json_files(self) -> bool:
        """Validate all JSON files in fusion config directory against schema."""
        schema = self.load_json_schema()
        if not schema:
            return False
        
        json_files = list(self.config_dir.glob("*.json"))
        if not json_files:
            self.log_warning("No JSON files found for validation")
            return True
        
        validation_success = True
        
        for json_file in json_files:
            # Skip schema file itself
            if json_file.name.endswith('.schema.json'):
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Only validate capability profile files
                if 'capability_profile' in json_file.name or 'ledger' in json_file.name:
                    jsonschema.validate(data, schema)
                    print(f"[PASS] Schema validation passed: {json_file.name}")
                else:
                    print(f"[SKIP] Schema validation: {json_file.name}")
                    
            except jsonschema.ValidationError as e:
                self.log_error(f"Schema validation failed for {json_file.name}: {e.message}")
                validation_success = False
            except json.JSONDecodeError as e:
                self.log_error(f"Invalid JSON in {json_file.name}: {e}")
                validation_success = False
            except Exception as e:
                self.log_error(f"Error validating {json_file.name}: {e}")
                validation_success = False
        
        return validation_success

    def validate_fusion_overlay_keys(self) -> bool:
        """Validate fusion overlay file has all required configuration keys."""
        overlay_files = list(self.config_dir.glob("fusion_overlay.*.txt"))
        
        if not overlay_files:
            self.log_error("No fusion overlay files found (fusion_overlay.*.txt)")
            return False
        
        # Required keys for fusion overlay
        required_keys = [
            'ENTROPY_REDUCTION_TARGET',
            'CONTINUITY_ENFORCEMENT',
            'TRUST_SCORING_MODEL',
            'PRIMARY_MODEL_SELECTION',
            'FUSION_MODE',
            'VOTING_POWER_MAP',
            'ARBITRATION_TIMEOUT_MS',
            'FUSION_OVERLAY_VERSION'
        ]
        
        validation_success = True
        
        for overlay_file in overlay_files:
            try:
                with open(overlay_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                found_keys = set()
                for line in content.split('\n'):
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key = line.split('=', 1)[0].strip()
                        found_keys.add(key)
                
                missing_keys = set(required_keys) - found_keys
                if missing_keys:
                    self.log_error(f"Missing required keys in {overlay_file.name}: {', '.join(missing_keys)}")
                    validation_success = False
                else:
                    print(f"[PASS] Required keys validation passed: {overlay_file.name}")
                    
            except Exception as e:
                self.log_error(f"Error validating overlay file {overlay_file.name}: {e}")
                validation_success = False
        
        return validation_success

    def validate_ledger_required_fields(self) -> bool:
        """Validate model semantics ledger has required fields."""
        ledger_files = list(self.config_dir.glob("model_semantics_ledger.*.json"))
        
        if not ledger_files:
            self.log_error("No model semantics ledger files found")
            return False
        
        required_fields = [
            'schema_version',
            'generated_at', 
            'ledger_id',
            'metadata',
            'cross_model',
            'fusion_recommendations',
            'validation_metrics'
        ]
        
        validation_success = True
        
        for ledger_file in ledger_files:
            try:
                with open(ledger_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                missing_fields = []
                for field in required_fields:
                    if field not in data:
                        missing_fields.append(field)
                
                if missing_fields:
                    self.log_error(f"Missing required fields in {ledger_file.name}: {', '.join(missing_fields)}")
                    validation_success = False
                else:
                    print(f"[PASS] Required fields validation passed: {ledger_file.name}")
                    
            except Exception as e:
                self.log_error(f"Error validating ledger file {ledger_file.name}: {e}")
                validation_success = False
        
        return validation_success

    def check_pii_content(self) -> bool:
        """Check for PII content in configuration files."""
        config_files = []
        config_files.extend(self.config_dir.glob("*.json"))
        config_files.extend(self.config_dir.glob("*.txt"))
        config_files.extend(self.config_dir.glob("*.yaml"))
        config_files.extend(self.config_dir.glob("*.yml"))
        
        pii_found = False
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pii_type, pattern in self.pii_patterns.items():
                    matches = pattern.findall(content)
                    if matches:
                        # Filter out obvious false positives
                        actual_matches = []
                        for match in matches:
                            # Skip common false positives
                            if pii_type == 'email' and any(domain in match for domain in ['example.com', 'test.com', 'localhost']):
                                continue
                            if pii_type == 'ip_address' and any(ip in match for ip in ['127.0.0.1', '0.0.0.0', '255.255.255.255']):
                                continue
                            if pii_type == 'api_key' and len(match) < 24:  # Too short to be real API key
                                continue
                            actual_matches.append(match)
                        
                        if actual_matches:
                            self.log_error(f"Potential PII ({pii_type}) found in {config_file.name}: {len(actual_matches)} instances")
                            pii_found = True
                            
            except Exception as e:
                self.log_warning(f"Could not scan {config_file.name} for PII: {e}")
        
        if not pii_found:
            print("[PASS] PII security check passed")
        
        return not pii_found

    def calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get_cached_ledger_hash(self) -> Optional[str]:
        """Get cached ledger content hash."""
        cache_file = self.cache_dir / "ledger_content_hash.txt"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                return None
        return None

    def update_cached_ledger_hash(self, content_hash: str):
        """Update cached ledger content hash."""
        cache_file = self.cache_dir / "ledger_content_hash.txt"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content_hash)
        except Exception as e:
            self.log_warning(f"Could not update cache: {e}")

    def get_current_ledger_version(self) -> Optional[str]:
        """Get current ledger version from file."""
        ledger_files = list(self.config_dir.glob("model_semantics_ledger.*.json"))
        
        if not ledger_files:
            return None
        
        try:
            with open(ledger_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('schema_version')
        except Exception:
            return None

    def increment_semver(self, version: str) -> str:
        """Increment patch version in semver format."""
        try:
            parts = version.split('.')
            if len(parts) == 3:
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
        except Exception:
            pass
        return "0.1.1"  # Default fallback

    def validate_semver_bump(self) -> bool:
        """Validate semantic version bump on content changes."""
        ledger_files = list(self.config_dir.glob("model_semantics_ledger.*.json"))
        
        if not ledger_files:
            self.log_warning("No ledger files found for version validation")
            return True
        
        try:
            with open(ledger_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                data = json.load(open(ledger_files[0], 'r', encoding='utf-8'))
            
            # Remove timestamp fields for content comparison
            content_for_hash = re.sub(r'"(generated_at|last_validated|last_benchmark)"\s*:\s*"[^"]*"', '', content)
            current_hash = self.calculate_content_hash(content_for_hash)
            cached_hash = self.get_cached_ledger_hash()
            
            current_version = data.get('schema_version', '0.1.0')
            
            if cached_hash and cached_hash != current_hash:
                # Content changed, version should have been bumped
                cache_version_file = self.cache_dir / "ledger_version.txt"
                
                if cache_version_file.exists():
                    try:
                        with open(cache_version_file, 'r', encoding='utf-8') as f:
                            cached_version = f.read().strip()
                        
                        if cached_version == current_version:
                            expected_version = self.increment_semver(cached_version)
                            self.log_error(f"Ledger content changed but version not bumped. Expected: {expected_version}, Found: {current_version}")
                            return False
                    except Exception:
                        pass
            
            # Update cache
            self.update_cached_ledger_hash(current_hash)
            cache_version_file = self.cache_dir / "ledger_version.txt"
            try:
                with open(cache_version_file, 'w', encoding='utf-8') as f:
                    f.write(current_version)
            except Exception:
                pass
            
            print(f"[PASS] Version validation passed: {current_version}")
            return True
            
        except Exception as e:
            self.log_error(f"Error during version validation: {e}")
            return False

    def validate_license_headers(self) -> bool:
        """Validate license headers and provenance notes in ledger files."""
        ledger_files = list(self.config_dir.glob("model_semantics_ledger.*.json"))
        
        validation_success = True
        
        for ledger_file in ledger_files:
            try:
                with open(ledger_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for provenance information in metadata
                metadata = data.get('metadata', {})
                
                required_provenance_fields = ['description', 'engine_version', 'last_validated']
                missing_fields = [field for field in required_provenance_fields if field not in metadata]
                
                if missing_fields:
                    self.log_error(f"Missing provenance fields in {ledger_file.name}: {', '.join(missing_fields)}")
                    validation_success = False
                
                # Check for appropriate description indicating source
                description = metadata.get('description', '')
                if 'RESONTINEX' not in description:
                    self.log_warning(f"Ledger description should include RESONTINEX attribution: {ledger_file.name}")
                
                print(f"[PASS] Provenance validation passed: {ledger_file.name}")
                    
            except Exception as e:
                self.log_error(f"Error validating provenance for {ledger_file.name}: {e}")
                validation_success = False
        
        return validation_success

    def run_fuse_ledger_validation(self) -> bool:
        """Run fuse-ledger.py script to validate ledger generation."""
        fuse_script = Path("scripts/fuse-ledger.py")
        
        if not fuse_script.exists():
            self.log_error("fuse-ledger.py script not found")
            return False
        
        try:
            # Run fuse-ledger script with validation flag
            result = subprocess.run([
                sys.executable, str(fuse_script), '--validate'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("[PASS] Fuse-ledger validation passed")
                return True
            else:
                self.log_error(f"Fuse-ledger validation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_error("Fuse-ledger validation timed out")
            return False
        except Exception as e:
            self.log_error(f"Error running fuse-ledger validation: {e}")
            return False

    def validate_configuration_completeness(self) -> bool:
        """Validate that all required configuration files are present and complete."""
        required_files = [
            "capability_profile.schema.json",
            "fusion_overlay.v0.3.txt",
            "model_semantics_ledger.v0.1.0.json"
        ]
        
        validation_success = True
        
        for required_file in required_files:
            file_path = self.config_dir / required_file
            if not file_path.exists():
                self.log_error(f"Required configuration file missing: {required_file}")
                validation_success = False
            else:
                # Check file is not empty
                if file_path.stat().st_size == 0:
                    self.log_error(f"Configuration file is empty: {required_file}")
                    validation_success = False
        
        if validation_success:
            print("[PASS] Configuration completeness check passed")
        
        return validation_success

    def run_all_validations(self) -> bool:
        """Run comprehensive CI validation suite."""
        print("Running RESONTINEX Fusion CI Validation Suite")
        print("=" * 50)
        
        validations = [
            ("Configuration completeness", self.validate_configuration_completeness),
            ("JSON schema validation", self.validate_json_files),
            ("Fusion overlay keys", self.validate_fusion_overlay_keys),
            ("Ledger required fields", self.validate_ledger_required_fields),
            ("PII security check", self.check_pii_content),
            ("Semantic version validation", self.validate_semver_bump),
            ("License and provenance", self.validate_license_headers),
            ("Fuse-ledger integration", self.run_fuse_ledger_validation)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            print(f"\n{validation_name}:")
            print("-" * 30)
            
            try:
                if not validation_func():
                    all_passed = False
            except Exception as e:
                self.log_error(f"Validation {validation_name} failed with exception: {e}")
                all_passed = False
        
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        if self.validation_errors:
            print(f"[FAIL] {len(self.validation_errors)} errors found:")
            for error in self.validation_errors:
                print(f"  - {error}")
        
        if self.validation_warnings:
            print(f"\n[WARN] {len(self.validation_warnings)} warnings:")
            for warning in self.validation_warnings:
                print(f"  - {warning}")
        
        if all_passed:
            print("[PASS] ALL VALIDATIONS PASSED")
        else:
            print("[FAIL] VALIDATION FAILED - Build should not proceed")
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="RESONTINEX Fusion CI Validation")
    parser.add_argument('--config-dir', default='./configs/fusion', help='Fusion config directory')
    parser.add_argument('--cache-dir', default='./.ci_cache', help='CI cache directory')
    parser.add_argument('--fail-on-warnings', action='store_true', help='Fail validation on warnings')
    
    args = parser.parse_args()
    
    validator = FusionCIValidator(args.config_dir, args.cache_dir)
    
    validation_passed = validator.run_all_validations()
    
    # Check if we should fail on warnings
    if args.fail_on_warnings and validator.validation_warnings:
        print(f"\n[FAIL] FAILING due to {len(validator.validation_warnings)} warnings (--fail-on-warnings enabled)")
        validation_passed = False
    
    return 0 if validation_passed else 1


if __name__ == "__main__":
    exit(main())