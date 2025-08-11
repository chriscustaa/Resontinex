# RESONTINEX Fusion Quickstart

5-minute validation of the fusion overlay system with expected FEC scores.

## Quick Start

```bash
# Run the quickstart scenario
make quickstart

# Or manually:
python -m fusion_ops.benchmark --scenarios-dir examples/quickstart --iterations 1 --verbose
```

## Expected Results

This quickstart validates:
- ✅ Fusion overlay loading and parsing
- ✅ Budget analysis within thresholds
- ✅ FEC score calculation
- ✅ Provenance stamp generation

**Expected FEC Metrics:**
- Token Delta: 8-15% (within 25% budget)
- P95 Latency: < 2000ms
- Quality Improvement: +0.05 to +0.15
- Overlay Score: 0.82-0.89

## Files

- [`quickstart_scenario.json`](quickstart_scenario.json) - Single test scenario
- [`expected_output.json`](expected_output.json) - Reference output for validation
- [`config_override.yaml`](config_override.yaml) - Minimal overlay configuration

## Troubleshooting

**Common Issues:**

1. **Import errors**: Ensure RESONTINEX is installed with `pip install -e .[dev]`
2. **Config not found**: Run from project root directory
3. **Budget thresholds exceeded**: Check if tripwire mechanism activated

**Debug Commands:**
```bash
# Check fusion system health
python -c "from resontinex.fusion_resilience import load_fusion_configuration; print(load_fusion_configuration())"

# Validate golden scenario format
python -m json.tool examples/quickstart/quickstart_scenario.json

# Test budget analysis
python -m fusion_ops.budget_analysis --report-file benchmark_results.json --output-format markdown