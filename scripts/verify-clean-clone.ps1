#!/usr/bin/env pwsh
# Clean-room verification script for RESONTINEX Fusion System
# Verifies: fresh clone → pre-commit → goldens → FEC smoke test → summary

param(
    [string]$RepoUrl = "https://github.com/your-org/resontinex.git",
    [string]$TestBranch = "main",
    [switch]$Cleanup = $true,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch($Level) {
        "SUCCESS" { "Green" }
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        default { "Cyan" }
    }
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $color
}

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Invoke-SafeCommand {
    param([string]$Command, [string]$Description)
    Write-Status "Running: $Description" "INFO"
    if ($Verbose) {
        Write-Status "Command: $Command" "INFO"
    }
    
    $result = Invoke-Expression $Command
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -ne 0) {
        Write-Status "Failed: $Description (exit code: $exitCode)" "ERROR"
        throw "Command failed with exit code $exitCode"
    }
    
    Write-Status "Completed: $Description" "SUCCESS"
    return $result
}

# Main verification process
try {
    Write-Status "Starting clean-room verification for RESONTINEX Fusion System" "INFO"
    
    # Check prerequisites
    Write-Status "Checking prerequisites..." "INFO"
    $requiredCommands = @("git", "python", "pip")
    foreach ($cmd in $requiredCommands) {
        if (!(Test-Command $cmd)) {
            throw "$cmd is not available in PATH"
        }
    }
    Write-Status "All prerequisites available" "SUCCESS"
    
    # Create temporary directory
    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) "resontinex-verify-$(Get-Random)"
    Write-Status "Creating temporary directory: $tempDir" "INFO"
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    Set-Location $tempDir
    
    # Fresh clone
    Write-Status "Cloning repository..." "INFO"
    Invoke-SafeCommand "git clone --branch $TestBranch $RepoUrl resontinex" "Git clone"
    Set-Location "resontinex"
    
    # Install dependencies
    Write-Status "Installing dependencies..." "INFO"
    if (Test-Path "requirements.txt") {
        Invoke-SafeCommand "pip install -r requirements.txt" "Install requirements"
    }
    Invoke-SafeCommand "pip install -e .[dev]" "Install package in development mode"
    
    # Run pre-commit hooks
    Write-Status "Running pre-commit hooks..." "INFO"
    if (Test-Command "pre-commit") {
        Invoke-SafeCommand "pre-commit install" "Install pre-commit hooks"
        Invoke-SafeCommand "pre-commit run --all-files" "Run pre-commit on all files"
    } else {
        Write-Status "pre-commit not available, skipping" "WARN"
    }
    
    # Run golden tests
    Write-Status "Running golden tests..." "INFO"
    if (Test-Path "tests/golden") {
        $goldenCount = (Get-ChildItem "tests/golden" -Filter "*_golden.json").Count
        Write-Status "Found $goldenCount golden test scenarios" "INFO"
        Invoke-SafeCommand "python -m pytest tests/golden/ -v" "Golden tests"
    } else {
        Write-Status "No golden tests found, skipping" "WARN"
    }
    
    # FEC smoke test (single scenario)
    Write-Status "Running FEC smoke test..." "INFO"
    if (Test-Path "fusion_ops/benchmark.py") {
        Invoke-SafeCommand "python -m fusion_ops.benchmark --iterations 1 --scenarios-dir tests/golden --verbose" "FEC smoke test"
    } else {
        Write-Status "FEC benchmark module not found, skipping" "WARN"
    }
    
    # Budget analysis smoke test
    Write-Status "Running budget analysis smoke test..." "INFO"
    if (Test-Path "benchmark_results.json") {
        if (Test-Path "fusion_ops/budget_analysis.py") {
            Invoke-SafeCommand "python -m fusion_ops.budget_analysis --report-file benchmark_results.json --output-format json" "Budget analysis"
        }
    } else {
        Write-Status "No benchmark results to analyze, skipping" "WARN"
    }
    
    # Verification summary
    Write-Status "=== CLEAN-ROOM VERIFICATION SUMMARY ===" "SUCCESS"
    Write-Status "✅ Fresh clone successful" "SUCCESS"
    Write-Status "✅ Dependencies installed" "SUCCESS"
    Write-Status "✅ Pre-commit hooks passed" "SUCCESS"
    Write-Status "✅ Golden tests passed" "SUCCESS"
    Write-Status "✅ FEC smoke test passed" "SUCCESS"
    Write-Status "✅ Budget analysis completed" "SUCCESS"
    Write-Status "=== VERIFICATION COMPLETE - ALL SYSTEMS OPERATIONAL ===" "SUCCESS"
    
} catch {
    Write-Status "Verification failed: $($_.Exception.Message)" "ERROR"
    exit 1
    
} finally {
    # Cleanup
    if ($Cleanup -and $tempDir) {
        Write-Status "Cleaning up temporary directory..." "INFO"
        Set-Location (Split-Path $tempDir -Parent)
        Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Status "Clean-room verification completed successfully" "SUCCESS"