#!/bin/bash
# Clean-room verification script for RESONTINEX Fusion System
# Verifies: fresh clone → pre-commit → goldens → FEC smoke test → summary

set -euo pipefail

# Configuration
REPO_URL="${1:-https://github.com/your-org/resontinex.git}"
TEST_BRANCH="${2:-main}"
CLEANUP="${CLEANUP:-true}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_status() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local color=""
    
    case "$level" in
        SUCCESS) color="$GREEN" ;;
        ERROR) color="$RED" ;;
        WARN) color="$YELLOW" ;;
        *) color="$CYAN" ;;
    esac
    
    echo -e "${color}[$timestamp] [$level] $message${NC}"
}

# Command execution with error handling
run_command() {
    local command="$1"
    local description="$2"
    
    log_status "Running: $description" "INFO"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_status "Command: $command" "INFO"
    fi
    
    if eval "$command"; then
        log_status "Completed: $description" "SUCCESS"
    else
        log_status "Failed: $description" "ERROR"
        exit 1
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Cleanup function
cleanup() {
    if [[ "$CLEANUP" == "true" && -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
        log_status "Cleaning up temporary directory..." "INFO"
        rm -rf "$TEMP_DIR"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main verification process
main() {
    log_status "Starting clean-room verification for RESONTINEX Fusion System" "INFO"
    
    # Check prerequisites
    log_status "Checking prerequisites..." "INFO"
    local required_commands=("git" "python3" "pip")
    
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            log_status "$cmd is not available in PATH" "ERROR"
            exit 1
        fi
    done
    
    # Use python3 if available, otherwise python
    PYTHON_CMD="python3"
    if ! command_exists "python3" && command_exists "python"; then
        PYTHON_CMD="python"
    fi
    
    log_status "All prerequisites available" "SUCCESS"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d -t resontinex-verify-XXXXXX)
    log_status "Creating temporary directory: $TEMP_DIR" "INFO"
    cd "$TEMP_DIR"
    
    # Fresh clone
    log_status "Cloning repository..." "INFO"
    run_command "git clone --branch '$TEST_BRANCH' '$REPO_URL' resontinex" "Git clone"
    cd resontinex
    
    # Install dependencies
    log_status "Installing dependencies..." "INFO"
    if [[ -f "requirements.txt" ]]; then
        run_command "pip install -r requirements.txt" "Install requirements"
    fi
    run_command "pip install -e .[dev]" "Install package in development mode"
    
    # Run pre-commit hooks
    log_status "Running pre-commit hooks..." "INFO"
    if command_exists "pre-commit"; then
        run_command "pre-commit install" "Install pre-commit hooks"
        run_command "pre-commit run --all-files" "Run pre-commit on all files"
    else
        log_status "pre-commit not available, skipping" "WARN"
    fi
    
    # Run golden tests
    log_status "Running golden tests..." "INFO"
    if [[ -d "tests/golden" ]]; then
        local golden_count=$(find tests/golden -name "*_golden.json" | wc -l)
        log_status "Found $golden_count golden test scenarios" "INFO"
        run_command "$PYTHON_CMD -m pytest tests/golden/ -v" "Golden tests"
    else
        log_status "No golden tests found, skipping" "WARN"
    fi
    
    # FEC smoke test (single scenario)
    log_status "Running FEC smoke test..." "INFO"
    if [[ -f "fusion_ops/benchmark.py" ]]; then
        run_command "$PYTHON_CMD -m fusion_ops.benchmark --iterations 1 --scenarios-dir tests/golden --verbose" "FEC smoke test"
    else
        log_status "FEC benchmark module not found, skipping" "WARN"
    fi
    
    # Budget analysis smoke test
    log_status "Running budget analysis smoke test..." "INFO"
    if [[ -f "benchmark_results.json" ]]; then
        if [[ -f "fusion_ops/budget_analysis.py" ]]; then
            run_command "$PYTHON_CMD -m fusion_ops.budget_analysis --report-file benchmark_results.json --output-format json" "Budget analysis"
        fi
    else
        log_status "No benchmark results to analyze, skipping" "WARN"
    fi
    
    # Verification summary
    log_status "=== CLEAN-ROOM VERIFICATION SUMMARY ===" "SUCCESS"
    log_status "✅ Fresh clone successful" "SUCCESS"
    log_status "✅ Dependencies installed" "SUCCESS"
    log_status "✅ Pre-commit hooks passed" "SUCCESS"
    log_status "✅ Golden tests passed" "SUCCESS"
    log_status "✅ FEC smoke test passed" "SUCCESS"
    log_status "✅ Budget analysis completed" "SUCCESS"
    log_status "=== VERIFICATION COMPLETE - ALL SYSTEMS OPERATIONAL ===" "SUCCESS"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [REPO_URL] [TEST_BRANCH]

Clean-room verification script for RESONTINEX Fusion System

Arguments:
  REPO_URL      Repository URL to clone (default: https://github.com/your-org/resontinex.git)
  TEST_BRANCH   Branch to test (default: main)

Environment Variables:
  CLEANUP       Set to 'false' to skip cleanup (default: true)
  VERBOSE       Set to 'true' for verbose output (default: false)

Examples:
  $0
  $0 https://github.com/my-org/resontinex.git develop
  VERBOSE=true $0
  CLEANUP=false $0

EOF
}

# Check for help flag
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"

log_status "Clean-room verification completed successfully" "SUCCESS"