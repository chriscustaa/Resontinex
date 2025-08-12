# RESONTINEX n8n Workflow Examples

Import these pre-built workflows to start using RESONTINEX with n8n in under 60 seconds.

## Quick Import (< 60 seconds)

### Step 1: Import Workflow (15 seconds)
1. Open n8n interface
2. Click **"Import from JSON"** 
3. Select workflow file:
   - [`certi-land-workflow.json`](certi-land-workflow.json) - **Production example** with full RESONTINEX module integration
   - [`n8n-simple-v1.3.json`](n8n-simple-v1.3.json) - **Simple starter** with basic routing logic

### Step 2: Configure Trigger (20 seconds)
1. Click the **Manual Trigger** node
2. Set trigger parameters (if any)
3. Save workflow

### Step 3: Test Execution (25 seconds)
1. Click **"Execute Workflow"**
2. Monitor execution in real-time
3. Review output data and flow progression

## Workflow Comparison

| Feature | Simple v1.3 | Certi-Land Production |
|---------|-------------|----------------------|
| **Complexity** | Basic routing | Full RESONTINEX modules |
| **Use Case** | Learning/testing | Production scenarios |
| **Nodes** | 6 nodes | 8 nodes |
| **Trust Scoring** | Binary (high/standard) | Weighted trust analysis |
| **Entropy Detection** | Threshold-based | Multi-factor auditing |
| **Energy Tracking** | Basic calculation | Not implemented |

## Configuration Notes

### Simple v1.3 Workflow
- **Input**: Business scenario text with complexity score
- **Processing**: Route based on complexity threshold (0.7)
- **Output**: Formatted result with energy consumption estimate
- **Modifications**: Adjust `complexity_score` threshold in Complexity Gate node

### Certi-Land Production Workflow  
- **Input**: Land parcel data with trust and risk factors
- **Processing**: Multi-stage validation (Entropy → Trust → Insight)
- **Output**: Approval status with trust score and processing summary
- **Modifications**: Update validation criteria in Entropy Auditor and Trust Manager nodes

## Advanced Integration

### Custom Data Sources
Replace Manual Trigger with:
- **HTTP Request** for API integration
- **Schedule Trigger** for batch processing  
- **Webhook** for event-driven activation

### RESONTINEX Module Mapping
n8n nodes map to RESONTINEX modules as follows:

```
Parse Metadata     → ContinuityEngine (state_persistence)
Entropy Auditor    → EntropyAuditor (threshold validation)
Trust Manager      → TrustManager (scoring + risk assessment)  
Insight Collapser  → InsightCollapser (output formatting)
```

### Error Handling
Both workflows include:
- **Error Workflow**: Automatic activation on node failures
- **Timeout Protection**: 300-second execution limit
- **Execution Logging**: Manual execution preservation

## Production Deployment

### Required Environment Variables
```bash
# Optional: Override RESONTINEX defaults
export RESON_FEATURES="energy_ledger,quorum_voting"
export RESON_TRUST_FLOOR="0.65"
export RESON_ENTROPY_THRESHOLD="0.68"
```

### Performance Expectations
- **Simple v1.3**: < 500ms execution time
- **Certi-Land**: < 1200ms execution time  
- **Memory Usage**: < 25MB per workflow instance
- **Concurrent Executions**: Up to 50 per n8n instance

### Monitoring Integration
Add HTTP Request nodes to send metrics to:
- **Prometheus**: Workflow completion metrics
- **Grafana**: Real-time dashboard updates
- **Custom Webhooks**: External monitoring systems

## Troubleshooting

### Common Issues
1. **Node Execution Fails**: Check input data schema matches expected format
2. **Timeout Errors**: Reduce complexity or increase timeout in workflow settings
3. **Trust Score Issues**: Validate input data quality and trust calculation logic

### Debug Mode
Enable detailed logging by adding Set nodes with:
```json
{
  "debug_mode": true,
  "execution_id": "{{ $workflow.id }}-{{ $now.toISOString() }}"
}
```

### Support
- **Documentation**: Review [`docs/`](../docs/) for detailed RESONTINEX concepts
- **Issues**: Submit via GitHub issues with workflow export attached
- **Integration Support**: chris@custaa.com for enterprise deployments