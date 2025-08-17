#!/usr/bin/env python3
"""Generate living API documentation using pdoc."""

import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional

def check_pdoc_available() -> bool:
    """Check if pdoc is available."""
    return shutil.which("pdoc") is not None

def generate_pdoc_docs(package: str, output_dir: str = "docs/api", format: str = "html") -> bool:
    """Generate API documentation using pdoc."""
    if not check_pdoc_available():
        print("[ERROR] pdoc not found. Install with: pip install pdoc")
        return False
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "pdoc",
        "-o", str(output_path),
        "--show-source",
        package
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"[OK] API documentation generated: {output_path}/{package}")
            return True
        else:
            print(f"[ERROR] pdoc failed: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print("[ERROR] pdoc generation timed out")
        return False
    except Exception as e:
        print(f"[ERROR] pdoc generation error: {e}")
        return False

def generate_module_docs(modules: list, output_dir: str = "docs/api") -> int:
    """Generate documentation for multiple modules."""
    success_count = 0
    
    for module in modules:
        print(f"[INFO] Generating docs for {module}...")
        if generate_pdoc_docs(module, output_dir):
            success_count += 1
        else:
            print(f"[WARN] Failed to generate docs for {module}")
    
    return success_count

def create_docs_index(output_dir: str = "docs/api") -> None:
    """Create an index.html file for the API documentation."""
    output_path = Path(output_dir)
    index_file = output_path / "index.html"
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resontinex API Documentation</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; }
        h1 { color: #2563eb; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
        .module-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 30px; }
        .module-card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; background: #f9fafb; }
        .module-card h3 { margin-top: 0; color: #1f2937; }
        .module-card a { color: #2563eb; text-decoration: none; font-weight: 500; }
        .module-card a:hover { text-decoration: underline; }
        .description { color: #6b7280; margin: 10px 0; }
        .timestamp { color: #9ca3af; font-size: 0.875rem; text-align: center; margin-top: 40px; }
    </style>
</head>
<body>
    <h1>Resontinex API Documentation</h1>
    <p>Production-grade AI workflow orchestration system with intelligent overlay routing, circuit breaker protection, and comprehensive drift detection.</p>
    
    <div class="module-grid">
        <div class="module-card">
            <h3><a href="resontinex/index.html">Core Package</a></h3>
            <div class="description">Main Resontinex package with configuration, plugins, and core utilities.</div>
        </div>
        
        <div class="module-card">
            <h3><a href="resontinex/config.html">Configuration</a></h3>
            <div class="description">Unified RuntimeConfig models for system configuration management.</div>
        </div>
        
        <div class="module-card">
            <h3><a href="resontinex/runtime/index.html">Runtime API</a></h3>
            <div class="description">OverlayRouter, ScenarioManager, and ProductionSafetyManager components.</div>
        </div>
        
        <div class="module-card">
            <h3><a href="resontinex/obs/index.html">Observability</a></h3>
            <div class="description">Prometheus metrics middleware and monitoring instrumentation.</div>
        </div>
        
        <div class="module-card">
            <h3><a href="resontinex/trust/index.html">Trust System</a></h3>
            <div class="description">Composite trust gate and integration validation components.</div>
        </div>
        
        <div class="module-card">
            <h3><a href="resontinex_governance/index.html">Governance Plugin</a></h3>
            <div class="description">Energy budget management and quorum voting governance features.</div>
        </div>
    </div>
    
    <div class="timestamp">
        Generated automatically with pdoc. Last updated: <script>document.write(new Date().toISOString().split('T')[0])</script>
    </div>
</body>
</html>"""
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[INFO] Created documentation index: {index_file}")

def main():
    """Main execution function."""
    modules_to_document = [
        "resontinex",
        "resontinex_governance"
    ]
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "docs/api"
    
    print("[INFO] Generating Resontinex API documentation...")
    print(f"[INFO] Output directory: {output_dir}")
    
    success_count = generate_module_docs(modules_to_document, output_dir)
    
    if success_count > 0:
        create_docs_index(output_dir)
        print(f"\n[SUCCESS] Generated documentation for {success_count}/{len(modules_to_document)} modules")
        print(f"[INFO] Open {output_dir}/index.html to view the documentation")
        return 0
    else:
        print("\n[ERROR] Failed to generate any documentation")
        print("[TIP] Make sure pdoc is installed: pip install resontinex[docs]")
        return 1

if __name__ == "__main__":
    sys.exit(main())