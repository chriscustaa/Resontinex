"""
Budget Analysis Module for Fusion System
Analyzes test results against budget thresholds and generates reports for CI/CD
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_budget_report(report_file: str) -> Dict[str, Any]:
    """Load budget report from test execution."""
    try:
        with open(report_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Budget report file not found: {report_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in budget report: {e}")
        sys.exit(1)


def analyze_thresholds(analysis: Dict[str, Any], thresholds: Dict[str, int]) -> Dict[str, Any]:
    """Analyze budget metrics against thresholds."""
    token_delta = analysis.get('token_delta_avg', 0)
    latency_p95 = analysis.get('latency_p95', 0)
    
    violations = []
    status = 'pass'
    
    # Token delta analysis
    if token_delta > thresholds['token_block']:
        status = 'block'
        violations.append({
            'type': 'token_delta_block',
            'metric': 'token_delta',
            'value': token_delta,
            'threshold': thresholds['token_block'],
            'severity': 'blocking'
        })
    elif token_delta > thresholds['token_warn']:
        if status == 'pass':
            status = 'warn'
        violations.append({
            'type': 'token_delta_warn',
            'metric': 'token_delta',
            'value': token_delta,
            'threshold': thresholds['token_warn'],
            'severity': 'warning'
        })
    
    # Latency analysis
    if latency_p95 > thresholds['latency_block']:
        status = 'block'
        violations.append({
            'type': 'latency_p95_block',
            'metric': 'latency_p95',
            'value': latency_p95,
            'threshold': thresholds['latency_block'],
            'severity': 'blocking'
        })
    elif latency_p95 > thresholds['latency_warn']:
        if status == 'pass':
            status = 'warn'
        violations.append({
            'type': 'latency_p95_warn',
            'metric': 'latency_p95',
            'value': latency_p95,
            'threshold': thresholds['latency_warn'],
            'severity': 'warning'
        })
    
    return {
        'status': status,
        'violations': violations,
        'token_delta': token_delta,
        'latency_p95': latency_p95,
        'scenarios_analyzed': analysis.get('scenarios_tested', 0)
    }


def generate_github_outputs(result: Dict[str, Any]) -> None:
    """Generate GitHub Actions outputs."""
    # Set outputs for workflow
    print(f"::set-output name=status::{result['status']}")
    print(f"::set-output name=token_delta::{result['token_delta']:.1f}")
    print(f"::set-output name=latency_p95::{result['latency_p95']}")
    
    # Set environment variables
    with open('budget_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)


def generate_summary_markdown(result: Dict[str, Any], thresholds: Dict[str, int]) -> str:
    """Generate markdown summary for PR comments."""
    status_emoji = {
        'pass': '✅',
        'warn': '⚠️',
        'block': '❌',
        'error': '🔥'
    }
    
    emoji = status_emoji.get(result['status'], '❓')
    status_text = result['status'].upper()
    
    md = f"""## {emoji} Budget Analysis: {status_text}

**Performance Metrics:**
- **Token Delta**: {result['token_delta']:.1f}% (Warning: {thresholds['token_warn']}%, Block: {thresholds['token_block']}%)
- **P95 Latency**: {result['latency_p95']}ms (Warning: {thresholds['latency_warn']}ms, Block: {thresholds['latency_block']}ms)
- **Scenarios Tested**: {result['scenarios_analyzed']}

"""
    
    if result['violations']:
        md += "**Budget Violations:**\n"
        for violation in result['violations']:
            severity_emoji = '🚫' if violation['severity'] == 'blocking' else '⚠️'
            md += f"- {severity_emoji} **{violation['metric']}**: {violation['value']} exceeds {violation['severity']} threshold {violation['threshold']}\n"
    else:
        md += "**✅ No budget violations detected**\n"
    
    md += f"\n**Budget Status**: {result['status']}"
    
    if result['status'] == 'pass':
        md += " - All budget constraints satisfied ✅"
    elif result['status'] == 'warn':
        md += " - Warning thresholds exceeded, review recommended ⚠️"
    elif result['status'] == 'block':
        md += " - Blocking thresholds exceeded, changes required ❌"
    
    return md


def analyze_budget_metrics(report_file: str, thresholds: Dict[str, int], output_format: str = 'json') -> Dict[str, Any]:
    """Main function to analyze budget metrics."""
    # Load budget report
    try:
        report = load_budget_report(report_file)
    except Exception as e:
        print(f"Error loading budget report: {e}")
        if output_format == 'github':
            print("::set-output name=status::error")
        raise
    
    # Extract analysis from report
    if 'budget_analysis' not in report:
        error_msg = "Error: No budget analysis found in report"
        print(error_msg)
        if output_format == 'github':
            print("::set-output name=status::error")
        raise ValueError(error_msg)
    
    analysis = report['budget_analysis']
    
    # Analyze against thresholds
    result = analyze_thresholds(analysis, thresholds)
    
    # Generate output based on format
    if output_format == 'github':
        generate_github_outputs(result)
    elif output_format == 'markdown':
        summary = generate_summary_markdown(result, thresholds)
        with open('budget_summary.md', 'w') as f:
            f.write(summary)
        return {'result': result, 'summary': summary}
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze fusion budget metrics')
    parser.add_argument('--report-file', required=True, help='Budget report JSON file')
    parser.add_argument('--token-warn-threshold', type=int, default=12, help='Token delta warning threshold (%)')
    parser.add_argument('--token-block-threshold', type=int, default=25, help='Token delta blocking threshold (%)')
    parser.add_argument('--latency-warn-threshold', type=int, default=2000, help='Latency warning threshold (ms)')
    parser.add_argument('--latency-block-threshold', type=int, default=5000, help='Latency blocking threshold (ms)')
    parser.add_argument('--output-format', choices=['json', 'github', 'markdown'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    # Set up thresholds
    thresholds = {
        'token_warn': args.token_warn_threshold,
        'token_block': args.token_block_threshold,
        'latency_warn': args.latency_warn_threshold,
        'latency_block': args.latency_block_threshold
    }
    
    try:
        result = analyze_budget_metrics(args.report_file, thresholds, args.output_format)
        
        if args.output_format == 'json':
            print(json.dumps(result, indent=2))
        elif args.output_format == 'markdown':
            print(result['summary'])
        
        # Exit with appropriate code
        exit_codes = {'pass': 0, 'warn': 0, 'block': 1, 'error': 2}
        status = result.get('status', 'error') if isinstance(result, dict) else 'error'
        sys.exit(exit_codes.get(status, 2))
        
    except Exception as e:
        print(f"❌ Budget analysis failed: {e}")
        sys.exit(2)


if __name__ == '__main__':
    main()