#!/usr/bin/env python3
"""
RESONTINEX Fusion Hardening System Integration Test
Comprehensive validation of all hardening components working together.
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
from datetime import datetime
import traceback


class FusionHardeningTester:
    """Integration tester for complete fusion hardening system."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.test_results = {}
        self.test_start_time = time.time()
        self.temp_dirs = []
        
    def cleanup(self):
        """Cleanup temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {test_name}: {details}")
        self.test_results[test_name] = {
            "passed": passed,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }

    def test_ci_validation_pipeline(self) -> bool:
        """Test CI validation components."""
        print("\n=== Testing CI Validation Pipeline ===")
        
        try:
            # Test CI validation script exists and runs
            ci_script = self.project_root / "scripts" / "validate-fusion-ci.py"
            if not ci_script.exists():
                self.log_test("ci_validation_script_exists", False, "validate-fusion-ci.py not found")
                return False
            
            self.log_test("ci_validation_script_exists", True, "Script found")
            
            # Test CI script execution with dry run
            config_dir = self.project_root / "configs" / "fusion"
            if config_dir.exists():
                try:
                    result = subprocess.run([
                        sys.executable, str(ci_script), 
                        "--config-dir", str(config_dir),
                        "--cache-dir", str(tempfile.mkdtemp())
                    ], capture_output=True, text=True, timeout=30)
                    
                    success = result.returncode == 0
                    details = f"Exit code: {result.returncode}"
                    if not success and result.stderr:
                        details += f", Error: {result.stderr[:200]}"
                    
                    self.log_test("ci_validation_execution", success, details)
                    return success
                    
                except subprocess.TimeoutExpired:
                    self.log_test("ci_validation_execution", False, "Timeout after 30s")
                    return False
                except Exception as e:
                    self.log_test("ci_validation_execution", False, f"Exception: {e}")
                    return False
            else:
                self.log_test("ci_validation_execution", False, "Config directory not found")
                return False
                
        except Exception as e:
            self.log_test("ci_validation_pipeline", False, f"Unexpected error: {e}")
            return False

    def test_fusion_evaluation_system(self) -> bool:
        """Test fusion effectiveness evaluation system."""
        print("\n=== Testing Fusion Evaluation System ===")
        
        try:
            # Test evaluation script exists
            eval_script = self.project_root / "scripts" / "evaluate-fusion.py"
            if not eval_script.exists():
                self.log_test("fusion_eval_script_exists", False, "evaluate-fusion.py not found")
                return False
            
            self.log_test("fusion_eval_script_exists", True, "Script found")
            
            # Test scenarios configuration exists
            scenarios_file = self.project_root / "configs" / "fusion" / "eval_scenarios.yaml"
            if not scenarios_file.exists():
                self.log_test("scenarios_config_exists", False, "eval_scenarios.yaml not found")
                return False
            
            self.log_test("scenarios_config_exists", True, "Scenarios file found")
            
            # Test evaluation execution with minimal iterations
            temp_output = tempfile.mkdtemp()
            self.temp_dirs.append(Path(temp_output))
            
            try:
                result = subprocess.run([
                    sys.executable, str(eval_script),
                    "--scenarios", str(scenarios_file),
                    "--out", temp_output,
                    "--iterations", "1",
                    "--config-dir", str(self.project_root / "configs" / "fusion")
                ], capture_output=True, text=True, timeout=60)
                
                success = result.returncode == 0
                details = f"Exit code: {result.returncode}"
                
                # Check if reports were generated
                output_files = list(Path(temp_output).glob("*"))
                if output_files:
                    details += f", Generated {len(output_files)} output files"
                
                if not success and result.stderr:
                    details += f", Error: {result.stderr[:200]}"
                
                self.log_test("fusion_eval_execution", success, details)
                return success
                
            except subprocess.TimeoutExpired:
                self.log_test("fusion_eval_execution", False, "Timeout after 60s")
                return False
            except Exception as e:
                self.log_test("fusion_eval_execution", False, f"Exception: {e}")
                return False
                
        except Exception as e:
            self.log_test("fusion_evaluation_system", False, f"Unexpected error: {e}")
            return False

    def test_runtime_resilience(self) -> bool:
        """Test runtime resilience and fallback mechanisms."""
        print("\n=== Testing Runtime Resilience ===")
        
        try:
            # Test resilience module import
            resilience_module = self.project_root / "resontinex" / "fusion_resilience.py"
            if not resilience_module.exists():
                self.log_test("resilience_module_exists", False, "fusion_resilience.py not found")
                return False
            
            self.log_test("resilience_module_exists", True, "Module found")
            
            # Test module execution
            try:
                result = subprocess.run([
                    sys.executable, str(resilience_module),
                    str(self.project_root / "configs" / "fusion")
                ], capture_output=True, text=True, timeout=30)
                
                success = result.returncode == 0
                details = f"Exit code: {result.returncode}"
                
                if success:
                    # Check for expected output patterns
                    output = result.stdout + result.stderr
                    if "Fusion configuration loaded successfully" in output:
                        details += ", Configuration loaded"
                    if "Metrics:" in output:
                        details += ", Metrics generated"
                
                if not success:
                    details += f", Error: {result.stderr[:200]}"
                
                self.log_test("resilience_execution", success, details)
                
                # Test fallback mechanism by testing with invalid config
                temp_config_dir = tempfile.mkdtemp()
                self.temp_dirs.append(Path(temp_config_dir))
                
                # Create invalid config file
                invalid_config = Path(temp_config_dir) / "fusion_overlay.v0.test.txt"
                with open(invalid_config, 'w') as f:
                    f.write("INVALID_CONFIG=broken\n")
                
                fallback_result = subprocess.run([
                    sys.executable, str(resilience_module),
                    str(temp_config_dir)
                ], capture_output=True, text=True, timeout=30)
                
                fallback_success = fallback_result.returncode == 0
                fallback_details = "Fallback mechanism triggered" if "embedded_fallback" in fallback_result.stdout else "No fallback detected"
                
                self.log_test("fallback_mechanism", fallback_success, fallback_details)
                
                return success and fallback_success
                
            except subprocess.TimeoutExpired:
                self.log_test("resilience_execution", False, "Timeout after 30s")
                return False
            except Exception as e:
                self.log_test("resilience_execution", False, f"Exception: {e}")
                return False
                
        except Exception as e:
            self.log_test("runtime_resilience", False, f"Unexpected error: {e}")
            return False

    def test_security_validation(self) -> bool:
        """Test security checks and PII detection."""
        print("\n=== Testing Security Validation ===")
        
        try:
            # Test PII detection by creating test data
            temp_test_dir = tempfile.mkdtemp()
            self.temp_dirs.append(Path(temp_test_dir))
            
            # Create test file with PII
            test_file = Path(temp_test_dir) / "test_pii.txt"
            with open(test_file, 'w') as f:
                f.write("ENTROPY_REDUCTION_TARGET=0.75\n")
                f.write("TRUST_SCORING_MODEL=alignment\n") 
                f.write("# This should be detected: test@example.com\n")
                f.write("# This should not: test@test.com\n")
            
            ci_script = self.project_root / "scripts" / "validate-fusion-ci.py"
            if ci_script.exists():
                result = subprocess.run([
                    sys.executable, str(ci_script),
                    "--config-dir", str(temp_test_dir),
                    "--cache-dir", str(tempfile.mkdtemp())
                ], capture_output=True, text=True, timeout=30)
                
                # Security scan should detect issues (expect failure for this test)
                security_detected = result.returncode != 0 and "email" in result.stdout
                
                self.log_test("pii_detection", security_detected, "PII detection working" if security_detected else "No PII detected")
            else:
                self.log_test("pii_detection", False, "CI script not found for security test")
            
            # Test security module functionality
            resilience_module = self.project_root / "resontinex" / "fusion_resilience.py"
            if resilience_module.exists():
                # Import and test security validator
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("fusion_resilience", resilience_module)
                    if spec is None or spec.loader is None:
                        self.log_test("metrics_functionality", False, "Failed to load resilience module spec")
                        return False
                    resilience = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(resilience)
                    
                    # Test security validator
                    validator = resilience.FusionSecurityValidator()
                    
                    # Test with clean data
                    clean_data = {"ENTROPY_REDUCTION_TARGET": "0.75"}
                    is_valid, violations = validator.validate_no_pii(clean_data, "test")
                    
                    self.log_test("security_validator_clean", is_valid, "Clean data passed validation")
                    
                    # Test with PII data
                    pii_data = {"config": "user@example.com", "key": "abc123"}
                    is_valid_pii, violations_pii = validator.validate_no_pii(pii_data, "test")
                    
                    self.log_test("security_validator_pii", not is_valid_pii, f"PII detected: {len(violations_pii)} violations" if violations_pii else "No violations found")
                    
                    return True
                    
                except Exception as e:
                    self.log_test("security_validator", False, f"Module import/execution failed: {e}")
                    return False
            else:
                self.log_test("security_validator", False, "Resilience module not found")
                return False
                
        except Exception as e:
            self.log_test("security_validation", False, f"Unexpected error: {e}")
            return False

    def test_observability_metrics(self) -> bool:
        """Test observability and metrics collection."""
        print("\n=== Testing Observability Metrics ===")
        
        try:
            resilience_module = self.project_root / "resontinex" / "fusion_resilience.py"
            if not resilience_module.exists():
                self.log_test("metrics_module_exists", False, "fusion_resilience.py not found")
                return False
            
            self.log_test("metrics_module_exists", True, "Module found")
            
            # Test metrics functionality
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("fusion_resilience", resilience_module)
                if spec is None or spec.loader is None:
                    self.log_test("security_validator", False, "Failed to load resilience module spec")
                    return False
                resilience = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(resilience)
                
                # Test loader metrics
                loader = resilience.FusionResilientLoader()
                
                # Test health status
                health = loader.get_health_status()
                required_health_keys = ["status", "overlay_loaded", "schema_ok", "decisions_made", "errors_count"]
                health_keys_present = all(key in health for key in required_health_keys)
                
                self.log_test("health_status_complete", health_keys_present, f"Health status has {len(health)} fields")
                
                # Test metrics emission
                metrics = loader.emit_metrics()
                required_metric_keys = ["fusion.decisions", "fusion.overlay_applied", "fusion.errors", "timestamp"]
                metrics_keys_present = all(key in metrics for key in required_metric_keys)
                
                self.log_test("metrics_emission_complete", metrics_keys_present, f"Metrics have {len(metrics)} fields")
                
                # Test decision recording
                initial_decisions = loader.metrics.decisions_made
                loader.record_decision(token_delta=50)
                decisions_incremented = loader.metrics.decisions_made == initial_decisions + 1
                tokens_recorded = loader.metrics.delta_tokens >= 50
                
                self.log_test("decision_recording", decisions_incremented and tokens_recorded, "Decision and token metrics recorded")
                
                return health_keys_present and metrics_keys_present and decisions_incremented
                
            except Exception as e:
                self.log_test("metrics_functionality", False, f"Metrics testing failed: {e}")
                return False
                
        except Exception as e:
            self.log_test("observability_metrics", False, f"Unexpected error: {e}")
            return False

    def test_github_actions_workflow(self) -> bool:
        """Test GitHub Actions workflow configuration."""
        print("\n=== Testing GitHub Actions Workflow ===")
        
        try:
            workflow_file = self.project_root / ".github" / "workflows" / "fusion-ci.yml"
            if not workflow_file.exists():
                self.log_test("workflow_file_exists", False, "fusion-ci.yml not found")
                return False
            
            self.log_test("workflow_file_exists", True, "Workflow file found")
            
            # Validate workflow file structure
            with open(workflow_file, 'r') as f:
                workflow_content = f.read()
            
            # Check for required workflow components
            required_components = [
                "name: fusion-ci",
                "validate_configs",
                "fusion_evaluation", 
                "Security PII scan",
                "Upload fusion reports",
                "Final status check"
            ]
            
            components_found = []
            for component in required_components:
                if component in workflow_content:
                    components_found.append(component)
            
            all_components_present = len(components_found) == len(required_components)
            
            self.log_test("workflow_completeness", all_components_present, 
                         f"Found {len(components_found)}/{len(required_components)} required components")
            
            # Check workflow syntax (basic YAML validation)
            try:
                import yaml
                yaml.safe_load(workflow_content)
                self.log_test("workflow_syntax", True, "YAML syntax valid")
                yaml_valid = True
            except Exception as e:
                self.log_test("workflow_syntax", False, f"YAML syntax error: {e}")
                yaml_valid = False
            
            return all_components_present and yaml_valid
            
        except Exception as e:
            self.log_test("github_actions_workflow", False, f"Unexpected error: {e}")
            return False

    def test_end_to_end_integration(self) -> bool:
        """Test complete end-to-end integration."""
        print("\n=== Testing End-to-End Integration ===")
        
        try:
            # Test complete fusion configuration loading with resilience
            config_dir = self.project_root / "configs" / "fusion"
            if not config_dir.exists():
                self.log_test("e2e_config_dir", False, "Config directory not found")
                return False
            
            # Test full configuration load
            resilience_module = self.project_root / "resontinex" / "fusion_resilience.py"
            if resilience_module.exists():
                try:
                    result = subprocess.run([
                        sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{self.project_root}')
from resontinex.fusion_resilience import load_fusion_configuration
overlay_config, health = load_fusion_configuration('{config_dir}')
print(f'SUCCESS: Loaded {{len(overlay_config)}} config keys, status={{health.get("status")}}')
"""], capture_output=True, text=True, timeout=30)
                    
                    success = result.returncode == 0 and "SUCCESS:" in result.stdout
                    details = result.stdout.strip() if success else result.stderr[:200]
                    
                    self.log_test("e2e_config_loading", success, details)
                    
                except Exception as e:
                    self.log_test("e2e_config_loading", False, f"Exception: {e}")
                    success = False
            else:
                self.log_test("e2e_config_loading", False, "Resilience module not found")
                success = False
            
            # Test integration of multiple components
            components_working = sum([
                self.test_results.get("ci_validation_execution", {}).get("passed", False),
                self.test_results.get("fusion_eval_execution", {}).get("passed", False),
                self.test_results.get("resilience_execution", {}).get("passed", False),
                self.test_results.get("security_validator", {}).get("passed", False),
                self.test_results.get("metrics_functionality", {}).get("passed", False)
            ])
            
            integration_success = components_working >= 4  # At least 4 out of 5 major components working
            
            self.log_test("e2e_integration", integration_success, 
                         f"{components_working}/5 major components working")
            
            return success and integration_success
            
        except Exception as e:
            self.log_test("end_to_end_integration", False, f"Unexpected error: {e}")
            return False

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        test_duration = time.time() - self.test_start_time
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["passed"])
        failed_tests = total_tests - passed_tests
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "duration_seconds": round(test_duration, 2),
                "timestamp": datetime.utcnow().isoformat()
            },
            "test_results": self.test_results,
            "system_status": "HEALTHY" if passed_tests >= total_tests * 0.8 else "DEGRADED"
        }
        
        return report

    def run_all_tests(self) -> bool:
        """Run all hardening system tests."""
        print("🔬 RESONTINEX Fusion Hardening System Integration Test")
        print("=" * 60)
        
        try:
            # Run all test suites
            test_suites = [
                ("CI Validation Pipeline", self.test_ci_validation_pipeline),
                ("Fusion Evaluation System", self.test_fusion_evaluation_system),
                ("Runtime Resilience", self.test_runtime_resilience),
                ("Security Validation", self.test_security_validation),
                ("Observability Metrics", self.test_observability_metrics),
                ("GitHub Actions Workflow", self.test_github_actions_workflow),
                ("End-to-End Integration", self.test_end_to_end_integration)
            ]
            
            suite_results = []
            
            for suite_name, suite_func in test_suites:
                print(f"\n🧪 Running {suite_name} tests...")
                try:
                    suite_result = suite_func()
                    suite_results.append(suite_result)
                    print(f"{'✅' if suite_result else '❌'} {suite_name}: {'PASSED' if suite_result else 'FAILED'}")
                except Exception as e:
                    print(f"❌ {suite_name}: EXCEPTION - {e}")
                    suite_results.append(False)
            
            # Generate final report
            report = self.generate_test_report()
            
            print("\n" + "=" * 60)
            print("📊 TEST SUMMARY REPORT")
            print("=" * 60)
            
            summary = report["test_summary"]
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Passed: {summary['passed']} ✅")
            print(f"Failed: {summary['failed']} ❌")
            print(f"Success Rate: {summary['success_rate']:.1%}")
            print(f"Duration: {summary['duration_seconds']}s")
            print(f"System Status: {report['system_status']}")
            
            # List failed tests
            if summary['failed'] > 0:
                print("\n❌ Failed Tests:")
                for test_name, result in self.test_results.items():
                    if not result["passed"]:
                        print(f"  • {test_name}: {result['details']}")
            
            overall_success = summary['success_rate'] >= 0.8
            print(f"\n🎯 Overall Result: {'SUCCESS' if overall_success else 'FAILURE'}")
            
            # Save detailed report
            report_file = self.project_root / "build" / "reports" / "fusion" / "hardening_test_report.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📁 Detailed report saved: {report_file}")
            
            return overall_success
            
        except Exception as e:
            print(f"\n💥 Critical test failure: {e}")
            traceback.print_exc()
            return False
        finally:
            self.cleanup()


def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    tester = FusionHardeningTester(project_root)
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())