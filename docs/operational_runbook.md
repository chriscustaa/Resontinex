# Fusion System Operational Runbook

## System Overview

The RESONTINEX Fusion Optimization System is an AI model overlay system that enhances baseline models with specialized overlays for improved performance in specific scenarios.

### Key Components
- **Fusion Core**: Main processing engine for overlay selection and execution
- **Golden Tests**: Regression testing framework with canonical scenarios
- **Feature Flags**: Runtime configuration for overlay routing decisions
- **Metrics System**: Performance monitoring with cardinality controls
- **Budget Gates**: Cost and latency enforcement system

## Quick Reference

### Emergency Contacts
- **Primary On-Call**: fusion-oncall@resontinex.com
- **Secondary**: ops-team@resontinex.com  
- **Escalation**: engineering-leads@resontinex.com

### Critical Metrics Dashboard
- **SLO Dashboard**: `https://monitoring.resontinex.com/fusion-slo`
- **Performance Metrics**: `https://metrics.resontinex.com/fusion`
- **Alert Manager**: `https://alerts.resontinex.com/fusion`

## Common Operations

### 1. Health Check Procedures

#### System Health Check
```bash
# Check core service health
curl -f https://api.resontinex.com/fusion/health

# Validate feature flags configuration
python -m fusion_ops.feature_flags validate

# Run critical golden tests
python -m tests.golden.golden_test_framework --validate --critical-only
```

#### Performance Health Check
```bash
# Run budget analysis
python scripts/benchmark_fusion.py --iterations 1 --output health_check.json
python scripts/analyze_budget_metrics.py --report-file health_check.json
```

### 2. Feature Flag Management

#### Enable Emergency Rollback
```bash
# Set environment override to force baseline-only
export RESON_FEATURES=""
# Restart services to pick up change
systemctl restart fusion-service
```

#### Enable Specific Feature
```bash
# Enable specific overlay via environment override
export RESON_FEATURES="refund_lot_switch,state_model_first"
systemctl restart fusion-service
```

#### Validate Feature Flag Changes
```bash
python -c "
from fusion_ops import get_feature_flags
ff = get_feature_flags()
print('Validation:', ff.validate_config())
"
```

### 3. Monitoring and Alerts

#### Key Metrics to Monitor
- `fusion.execution.latency_ms` - P95 < 2000ms (warn), < 5000ms (critical)
- `fusion.tokens.delta_pct` - Average < 12% (warn), < 25% (critical)  
- `fusion.quality.score` - Average > 0.8
- `fusion.circuit_breaker.event` - Monitor for excessive failures

#### Alert Response Times
- **Critical**: 15 minutes (system down, data loss risk)
- **High**: 1 hour (performance degradation, SLO at risk)
- **Medium**: 4 hours (quality issues, budget exceeded)
- **Low**: Next business day (minor issues, optimization needed)

## Incident Response Procedures

### Severity Levels

**Critical (P1)**: System unavailable, data corruption, security breach
**High (P2)**: Major feature broken, SLO violation > 1 hour
**Medium (P3)**: Performance degradation, non-critical features affected  
**Low (P4)**: Minor issues, cosmetic problems

### Response Procedures

#### 1. High Latency / Performance Issues

**Symptoms**: P95 latency > 5000ms, timeouts, user complaints

**Immediate Actions** (0-15 minutes):
1. Check system load and resource utilization
2. Review recent deployments/configuration changes
3. Enable emergency baseline-only mode if needed:
   ```bash
   export RESON_FEATURES=""
   systemctl restart fusion-service
   ```

**Investigation** (15-60 minutes):
1. Run performance comparison against baseline:
   ```bash
   python scripts/compare_performance.py \
     --baseline benchmark_baseline.json \
     --current benchmark_current.json
   ```
2. Check for specific overlay causing issues
3. Review circuit breaker metrics for failing overlays

**Recovery Actions**:
1. Disable problematic overlays via feature flags
2. Scale resources if load-related
3. Revert recent changes if identified as cause

#### 2. Quality Degradation

**Symptoms**: Quality scores dropping, user feedback negative, golden tests failing

**Immediate Actions**:
1. Run golden test validation:
   ```bash
   python -m tests.golden.golden_test_framework --validate
   ```
2. Check for model drift or training data issues
3. Compare current vs baseline quality metrics

**Investigation**:
1. Analyze failing scenarios and overlays
2. Review recent model updates or training changes
3. Check for data quality issues in input scenarios

**Recovery Actions**:
1. Rollback to previous model version if available
2. Disable specific overlays showing quality issues
3. Re-run model validation and testing

#### 3. Budget Violations

**Symptoms**: Token usage spiking, cost alerts firing, budget gates blocking

**Immediate Actions**:
1. Check budget analysis:
   ```bash
   python scripts/analyze_budget_metrics.py \
     --report-file current_budget.json \
     --output-format markdown
   ```
2. Identify scenarios causing high token usage
3. Enable stricter budget controls if needed

**Investigation**:
1. Review overlay selection patterns
2. Check for runaway processes or loops
3. Analyze token usage by scenario type

**Recovery Actions**:
1. Adjust budget thresholds in CI configuration
2. Disable high-cost overlays temporarily
3. Optimize prompts and overlay logic

#### 4. Circuit Breaker Activation

**Symptoms**: High error rates, circuit breaker events, fallback to baseline

**Immediate Actions**:
1. Check circuit breaker status and recent events
2. Review error logs for specific failure patterns
3. Validate system dependencies are available

**Investigation**:
1. Identify root cause of failures (network, API, model issues)
2. Check for resource exhaustion or rate limiting
3. Review recent changes that might cause instability

**Recovery Actions**:
1. Fix underlying issue causing failures
2. Reset circuit breaker state once issue resolved
3. Gradually re-enable affected overlays

### 5. Golden Test Failures

**Symptoms**: Regression tests failing, quality gates blocked, deployment issues

**Immediate Actions**:
1. Run specific failing tests in isolation:
   ```bash
   python -m tests.golden.golden_test_framework \
     --scenario refund_processing_duplicate \
     --validate
   ```
2. Compare current outputs with golden standards
3. Check if failure is due to legitimate model improvement

**Investigation**:
1. Analyze nature of differences (quality, format, content)
2. Review recent model or overlay changes
3. Validate against business requirements

**Recovery Actions**:
1. If legitimate improvement, update golden files:
   ```bash
   python -m tests.golden.golden_test_framework \
     --scenario refund_processing_duplicate \
     --update-goldens
   ```
2. If regression, identify and fix root cause
3. Re-run full golden test suite to validate fix

## Recovery Procedures

### System Recovery Checklist

#### Full System Recovery
1. **Stop Traffic**: Route traffic to baseline system
2. **Assess Damage**: Determine scope and impact
3. **Fix Root Cause**: Address underlying issue
4. **Validate Fix**: Run health checks and tests
5. **Gradual Rollout**: Slowly re-enable fusion system
6. **Monitor**: Watch key metrics during recovery

#### Database Recovery
1. **Stop writes** to prevent further corruption
2. **Restore from backup** to last known good state
3. **Replay transactions** if possible and safe
4. **Validate data integrity** before resuming operations

#### Configuration Recovery
```bash
# Restore from version control
git checkout HEAD~1 config/overlay_feature_flags.yaml
systemctl restart fusion-service

# Or use emergency baseline configuration
cp config/emergency_baseline.yaml config/overlay_feature_flags.yaml
systemctl restart fusion-service
```

### Data Recovery

#### Golden Test Data
```bash
# Restore from backup
cp backups/golden_tests_backup_YYYYMMDD.tar.gz .
tar -xzf golden_tests_backup_YYYYMMDD.tar.gz

# Validate integrity
python -m tests.golden.golden_test_framework --validate-all
```

#### Performance Baselines
```bash
# Restore baseline performance data
cp backups/performance_baseline_YYYYMMDD.json benchmark_baseline.json

# Re-establish baselines if needed
python scripts/benchmark_fusion.py --baseline-mode --output new_baseline.json
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily (Automated)
- Health check execution
- Metric collection and analysis  
- Budget analysis and reporting
- Golden test validation

#### Weekly
- Performance trend analysis
- Feature flag usage review
- Circuit breaker statistics review
- Capacity planning analysis

#### Monthly
- Full system performance review
- Golden test suite expansion/cleanup
- Alert threshold tuning
- Operational procedure updates

### Preventive Measures

#### Capacity Management
- Monitor resource utilization trends
- Plan capacity increases before reaching limits
- Review and optimize resource-intensive overlays

#### Performance Optimization
- Regular performance testing and benchmarking
- Identify and optimize slow overlays
- Review and tune budget thresholds

#### Quality Assurance
- Expand golden test coverage for new scenarios
- Regular model validation and testing
- Monitor quality trends and patterns

## Escalation Procedures

### Internal Escalation Path
1. **L1 Support** → Check runbook, basic troubleshooting
2. **L2 Operations** → Advanced troubleshooting, system recovery
3. **L3 Engineering** → Code changes, architectural decisions
4. **Management** → Resource decisions, external communications

### External Escalation
- **Vendor Issues**: Contact respective vendor support
- **Infrastructure**: Engage infrastructure team
- **Security**: Involve security team for any security-related issues

### Communication Protocols
- **Status Page**: Update public status for customer-facing issues
- **Slack**: Use #fusion-incidents for real-time coordination
- **Email**: Send updates to stakeholders for P1/P2 incidents

## Testing and Validation

### Post-Incident Testing
1. Reproduce the incident in test environment
2. Validate fix resolves the issue
3. Run full regression test suite
4. Perform load testing if performance-related

### Rollback Testing
```bash
# Test rollback procedures in staging
export RESON_FEATURES=""
python scripts/benchmark_fusion.py --output rollback_test.json
python -m tests.golden.golden_test_framework --validate
```

### Recovery Time Testing
- Regularly test and measure recovery procedures
- Document actual recovery times vs. targets
- Update procedures based on testing results

---

**Document Version**: 1.0
**Last Updated**: 2024-01-10
**Next Review**: 2024-04-10
**Owner**: Fusion Operations Team