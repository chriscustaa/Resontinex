# Dashboard Images Directory

This directory contains screenshots and visual documentation for the Fusion System monitoring dashboards.

## Required Images

### fusion-dashboard-v0.1.1.png
**Status**: 🔄 Pending Capture  
**Description**: Production monitoring dashboard showing real-time performance metrics, SLO compliance, and operational health  
**Source**: Import [`config/monitoring_dashboard.yaml`](../../config/monitoring_dashboard.yaml) into Grafana and capture screenshot  
**Dimensions**: 1920x1080 recommended  
**Content**: Performance metrics, feature flag usage, budget compliance, circuit breaker status, SLO monitoring

#### Capture Instructions
1. Import dashboard configuration into Grafana:
   ```bash
   curl -X POST \
     http://your-grafana-instance/api/dashboards/db \
     -H 'Content-Type: application/json' \
     -d @config/monitoring_dashboard.yaml
   ```

2. Navigate to Fusion System dashboard
3. Set time range to "Last 1 hour" 
4. Ensure all panels are displaying data
5. Capture full-screen screenshot as PNG
6. Save as `fusion-dashboard-v0.1.1.png` in this directory

#### Panel Coverage Required
- ✅ System Health Overview (stat panel)
- ✅ Execution Latency P95 (graph)
- ✅ Token Usage Delta (graph) 
- ✅ Quality Score Trends (graph)
- ✅ Feature Flag Usage (pie chart)
- ✅ Circuit Breaker Events (stat)
- ✅ Budget Compliance Status (stat)
- ✅ Error Rate by Overlay (graph)
- ✅ Success Rate SLO (gauge)

## Image Standards

- **Format**: PNG with transparency support
- **Quality**: High resolution, readable text
- **Content**: No sensitive data or credentials visible
- **Consistency**: Use production data, not test/demo data
- **Documentation**: Include timestamp and version in filename

## Update Process

When updating dashboard images:
1. Follow semantic versioning (e.g., `fusion-dashboard-v0.1.2.png`)
2. Update README.md image reference 
3. Archive previous version in `archive/` subdirectory
4. Update changelog with visual improvements noted