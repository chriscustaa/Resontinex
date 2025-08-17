#!/usr/bin/env python3
"""Generate architecture diagrams with cross-platform fallbacks."""

import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional

def run_command(cmd: list, description: str) -> bool:
    """Run command with error handling."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description} successful")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr.strip()}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"❌ {description} error: {e}")
        return False

def check_tool_available(tool: str) -> bool:
    """Check if a tool is available in PATH."""
    return shutil.which(tool) is not None

def generate_pydeps_diagram(package: str, output: str) -> bool:
    """Generate dependency diagram using pydeps."""
    if not check_tool_available("pydeps"):
        return False
    
    cmd = ["pydeps", "--max-bacon-length", "2", package, "-o", output]
    return run_command(cmd, f"pydeps diagram generation -> {output}")

def generate_pyreverse_diagram(package: str, output: str) -> bool:
    """Generate UML diagram using pyreverse + graphviz."""
    if not check_tool_available("pyreverse") or not check_tool_available("dot"):
        return False
    
    # Generate DOT files
    cmd1 = ["pyreverse", "-o", "dot", "-p", package, package]
    if not run_command(cmd1, "pyreverse DOT generation"):
        return False
    
    # Convert classes diagram to SVG
    dot_file = f"classes_{package}.dot"
    if Path(dot_file).exists():
        cmd2 = ["dot", "-Tsvg", dot_file, "-o", output]
        success = run_command(cmd2, f"graphviz conversion -> {output}")
        # Clean up DOT files
        Path(dot_file).unlink(missing_ok=True)
        Path(f"packages_{package}.dot").unlink(missing_ok=True)
        return success
    
    return False

def generate_pyan3_diagram(package: str, output: str) -> bool:
    """Generate call graph using pyan3."""
    if not check_tool_available("pyan3") or not check_tool_available("dot"):
        return False
    
    # Find Python files
    py_files = list(Path(package).rglob("*.py"))
    if not py_files:
        print(f"❌ No Python files found in {package}")
        return False
    
    # Generate DOT using pyan3
    cmd1 = ["pyan3"] + [str(f) for f in py_files] + ["--dot"]
    try:
        result = subprocess.run(cmd1, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False
        
        # Write DOT output to file
        dot_file = "architecture.dot"
        with open(dot_file, "w") as f:
            f.write(result.stdout)
        
        # Convert to SVG
        cmd2 = ["dot", "-Tsvg", dot_file, "-o", output]
        success = run_command(cmd2, f"pyan3 conversion -> {output}")
        Path(dot_file).unlink(missing_ok=True)
        return success
        
    except Exception as e:
        print(f"❌ pyan3 generation error: {e}")
        return False

def main():
    """Main execution function."""
    package = sys.argv[1] if len(sys.argv) > 1 else "resontinex"
    output = sys.argv[2] if len(sys.argv) > 2 else "architecture.svg"
    
    print(f"🔍 Generating architecture diagram for '{package}' -> {output}")
    
    # Strategy 1: Try pydeps (preferred)
    if generate_pydeps_diagram(package, output):
        print(f"🎉 Architecture diagram generated successfully: {output}")
        return 0
    
    print("⚠️ pydeps failed, trying pyreverse...")
    
    # Strategy 2: Try pyreverse + graphviz
    if generate_pyreverse_diagram(package, output):
        print(f"🎉 Architecture diagram generated via pyreverse: {output}")
        return 0
    
    print("⚠️ pyreverse failed, trying pyan3...")
    
    # Strategy 3: Try pyan3 + graphviz
    if generate_pyan3_diagram(package, output):
        print(f"🎉 Architecture diagram generated via pyan3: {output}")
        return 0
    
    print("❌ All diagram generation methods failed")
    print("💡 Install one of: pydeps, pylint+graphviz, or pyan3+graphviz")
    return 1

if __name__ == "__main__":
    sys.exit(main())