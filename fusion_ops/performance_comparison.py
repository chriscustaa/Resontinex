"""
Performance Comparison Tool for Fusion System
Compares current performance metrics against baseline to detect regressions
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class PerformanceComparator:
    """Compares performance metrics between baseline and current runs."""
    
    def __init__(self, regression_thresholds: Optional[Dict[str, float]] = None):
        self.regression_thresholds = regression_thresholds or {
            'token_delta': 5.0,  # 5% increase is considered regression
            'latency_p95': 500.0,  # 500ms increase is considered regression
            'quality_improvement': -0.05  # 0.05 decrease is considered regression
        }
    
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load performance results from file."""
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Results file not found: {results_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in results file {results_file}: {e}")
    
    def extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance metrics from results."""
        budget_analysis = results.get('budget_analysis', {})
        
        return {
            'token_delta_avg': budget_analysis.get('token_delta_avg', 0.0),
            'latency_p95': budget_analysis.get('latency_p95', 0.0),
            'latency_avg': budget_analysis.get('latency_avg', 0.0),
            'quality_improvement_avg': budget_analysis.get('quality_improvement_avg', 0.0),
            'scenarios_tested': budget_analysis.get('scenarios_tested', 0)
        }
    
    def calculate_metric_changes(self, baseline: Dict[str, float], current: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate changes between baseline and current metrics."""
        changes = {}
        
        for metric, baseline_value in baseline.items():
            if metric in current:
                current_value = current[metric]
                
                # Calculate absolute and percentage changes
                abs_change = current_value - baseline_value
                pct_change = (abs_change / baseline_value * 100) if baseline_value != 0 else 0.0
                
                changes[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'absolute_change': abs_change,
                    'percentage_change': pct_change
                }
        
        return changes
    
    def detect_regressions(self, changes: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Detect performance regressions based on thresholds."""
        regressions = []
        
        # Token delta regression (increase is bad)
        if 'token_delta_avg' in changes:
            token_change = changes['token_delta_avg']['absolute_change']
            if token_change > self.regression_thresholds['token_delta']:
                regressions.append({
                    'metric': 'token_delta_avg',
                    'type': 'increase',
                    'severity': 'high' if token_change > self.regression_thresholds['token_delta'] * 2 else 'medium',
                    'baseline': changes['token_delta_avg']['baseline'],
                    'current': changes['token_delta_avg']['current'],
                    'change': token_change,
                    'threshold': self.regression_thresholds['token_delta'],
                    'description': f"Token usage increased by {token_change:.1f}%"
                })
        
        # Latency regression (increase is bad)
        if 'latency_p95' in changes:
            latency_change = changes['latency_p95']['absolute_change']
            if latency_change > self.regression_thresholds['latency_p95']:
                regressions.append({
                    'metric': 'latency_p95',
                    'type': 'increase',
                    'severity': 'high' if latency_change > self.regression_thresholds['latency_p95'] * 2 else 'medium',
                    'baseline': changes['latency_p95']['baseline'],
                    'current': changes['latency_p95']['current'],
                    'change': latency_change,
                    'threshold': self.regression_thresholds['latency_p95'],
                    'description': f"P95 latency increased by {latency_change:.1f}ms"
                })
        
        # Quality regression (decrease is bad)
        if 'quality_improvement_avg' in changes:
            quality_change = changes['quality_improvement_avg']['absolute_change']
            if quality_change < self.regression_thresholds['quality_improvement']:
                regressions.append({
                    'metric': 'quality_improvement_avg',
                    'type': 'decrease',
                    'severity': 'high' if quality_change < self.regression_thresholds['quality_improvement'] * 2 else 'medium',
                    'baseline': changes['quality_improvement_avg']['baseline'],
                    'current': changes['quality_improvement_avg']['current'],
                    'change': quality_change,
                    'threshold': self.regression_thresholds['quality_improvement'],
                    'description': f"Quality improvement decreased by {abs(quality_change):.3f}"
                })
        
        return regressions
    
    def detect_improvements(self, changes: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Detect performance improvements."""
        improvements = []
        
        # Token delta improvement (decrease is good)
        if 'token_delta_avg' in changes:
            token_change = changes['token_delta_avg']['absolute_change']
            if token_change < -1.0:  # 1% decrease is noteworthy
                improvements.append({
                    'metric': 'token_delta_avg',
                    'type': 'decrease',
                    'baseline': changes['token_delta_avg']['baseline'],
                    'current': changes['token_delta_avg']['current'],
                    'change': token_change,
                    'description': f"Token usage improved by {abs(token_change):.1f}%"
                })
        
        # Latency improvement (decrease is good)
        if 'latency_p95' in changes:
            latency_change = changes['latency_p95']['absolute_change']
            if latency_change < -100.0:  # 100ms improvement is noteworthy
                improvements.append({
                    'metric': 'latency_p95',
                    'type': 'decrease',
                    'baseline': changes['latency_p95']['baseline'],
                    'current': changes['latency_p95']['current'],
                    'change': latency_change,
                    'description': f"P95 latency improved by {abs(latency_change):.1f}ms"
                })
        
        # Quality improvement (increase is good)
        if 'quality_improvement_avg' in changes:
            quality_change = changes['quality_improvement_avg']['absolute_change']
            if quality_change > 0.01:  # 0.01 increase is noteworthy
                improvements.append({
                    'metric': 'quality_improvement_avg',
                    'type': 'increase',
                    'baseline': changes['quality_improvement_avg']['baseline'],
                    'current': changes['quality_improvement_avg']['current'],
                    'change': quality_change,
                    'description': f"Quality improvement increased by {quality_change:.3f}"
                })
        
        return improvements
    
    def compare(self, baseline_file: str, current_file: str) -> Dict[str, Any]:
        """Compare baseline and current performance results."""
        # Load results
        baseline_results = self.load_results(baseline_file)
        current_results = self.load_results(current_file)
        
        # Extract key metrics
        baseline_metrics = self.extract_key_metrics(baseline_results)
        current_metrics = self.extract_key_metrics(current_results)
        
        # Calculate changes
        changes = self.calculate_metric_changes(baseline_metrics, current_metrics)
        
        # Detect regressions and improvements
        regressions = self.detect_regressions(changes)
        improvements = self.detect_improvements(changes)
        
        # Determine overall status
        if regressions:
            high_severity_regressions = [r for r in regressions if r['severity'] == 'high']
            status = 'regression' if high_severity_regressions else 'minor_regression'
        elif improvements:
            status = 'improvement'
        else:
            status = 'stable'
        
        return {
            'comparison_summary': {
                'status': status,
                'baseline_file': baseline_file,
                'current_file': current_file,
                'scenarios_compared': min(baseline_metrics.get('scenarios_tested', 0), 
                                        current_metrics.get('scenarios_tested', 0)),
                'regressions_detected': len(regressions),
                'improvements_detected': len(improvements)
            },
            'metric_changes': changes,
            'regressions': regressions,
            'improvements': improvements,
            'regression_thresholds': self.regression_thresholds
        }
    
    def generate_summary_markdown(self, comparison: Dict[str, Any]) -> str:
        """Generate markdown summary of comparison."""
        summary = comparison['comparison_summary']
        status = summary['status']
        
        status_emojis = {
            'regression': '🔴',
            'minor_regression': '🟡', 
            'improvement': '🟢',
            'stable': '🔵'
        }
        
        emoji = status_emojis.get(status, '❓')
        
        md = f"""## {emoji} Performance Comparison: {status.upper().replace('_', ' ')}

**Comparison Summary:**
- **Status**: {status.replace('_', ' ').title()}
- **Scenarios Compared**: {summary['scenarios_compared']}
- **Regressions**: {summary['regressions_detected']}
- **Improvements**: {summary['improvements_detected']}

"""
        
        # Add regressions section
        if comparison['regressions']:
            md += "### 🔴 Performance Regressions\n"
            for regression in comparison['regressions']:
                severity_emoji = '🚨' if regression['severity'] == 'high' else '⚠️'
                md += f"- {severity_emoji} **{regression['metric']}**: {regression['description']}\n"
                md += f"  - Baseline: {regression['baseline']:.2f}, Current: {regression['current']:.2f}\n"
            md += "\n"
        
        # Add improvements section
        if comparison['improvements']:
            md += "### 🟢 Performance Improvements\n"
            for improvement in comparison['improvements']:
                md += f"- ✨ **{improvement['metric']}**: {improvement['description']}\n"
                md += f"  - Baseline: {improvement['baseline']:.2f}, Current: {improvement['current']:.2f}\n"
            md += "\n"
        
        # Add detailed metrics
        md += "### 📊 Detailed Metrics\n"
        for metric, change in comparison['metric_changes'].items():
            trend = "📈" if change['absolute_change'] > 0 else "📉" if change['absolute_change'] < 0 else "➡️"
            md += f"- {trend} **{metric}**: {change['baseline']:.2f} → {change['current']:.2f} "
            md += f"({change['percentage_change']:+.1f}%)\n"
        
        return md


def compare_performance(baseline_file: str, current_file: str, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """High-level function to compare performance results."""
    comparator = PerformanceComparator(thresholds)
    return comparator.compare(baseline_file, current_file)


def main():
    parser = argparse.ArgumentParser(description='Compare fusion performance results')
    parser.add_argument('--baseline', required=True, help='Baseline performance results file')
    parser.add_argument('--current', required=True, help='Current performance results file')
    parser.add_argument('--output', default='performance_comparison.json', help='Output comparison file')
    parser.add_argument('--markdown-output', help='Generate markdown summary file')
    parser.add_argument('--token-regression-threshold', type=float, default=5.0,
                        help='Token delta regression threshold (%)')
    parser.add_argument('--latency-regression-threshold', type=float, default=500.0,
                        help='Latency regression threshold (ms)')
    parser.add_argument('--quality-regression-threshold', type=float, default=-0.05,
                        help='Quality regression threshold')
    
    args = parser.parse_args()
    
    # Set up thresholds
    thresholds = {
        'token_delta': args.token_regression_threshold,
        'latency_p95': args.latency_regression_threshold,
        'quality_improvement': args.quality_regression_threshold
    }
    
    try:
        comparison = compare_performance(args.baseline, args.current, thresholds)
        
        # Save JSON results
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Performance comparison saved to: {args.output}")
        
        # Generate markdown summary if requested
        if args.markdown_output:
            comparator = PerformanceComparator(thresholds)
            summary_md = comparator.generate_summary_markdown(comparison)
            with open(args.markdown_output, 'w') as f:
                f.write(summary_md)
            print(f"Markdown summary saved to: {args.markdown_output}")
        
        # Print summary to console
        summary = comparison['comparison_summary']
        status = summary['status']
        
        if status == 'regression':
            print(f"❌ Performance regression detected ({summary['regressions_detected']} issues)")
            sys.exit(1)
        elif status == 'minor_regression':
            print(f"⚠️ Minor performance regression detected ({summary['regressions_detected']} issues)")
            sys.exit(0)  # Don't fail build for minor regressions
        elif status == 'improvement':
            print(f"✅ Performance improvements detected ({summary['improvements_detected']} improvements)")
        else:
            print("✅ Performance is stable")
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()