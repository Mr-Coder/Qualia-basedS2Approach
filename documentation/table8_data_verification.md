# Table 8 Raw Experimental Data Verification Report

**Generation Date**: 2025-06-23 00:50:44
**Total Experiments**: 35,000
**Configurations**: 7

## Configuration Performance Verification

| Configuration | Expected Overall | Actual Overall | Difference | Expected L2 | Actual L2 | L2 Diff | Expected L3 | Actual L3 | L3 Diff |
|---------------|------------------|----------------|------------|-------------|-----------|---------|-------------|-----------|----------|
| w/o CV | 77.6% | 77.7% | 0.1% | 80.1% | 80.1% | 0.04% | 70.4% | 70.3% | 0.08% |
| w/o MLR | 74.9% | 74.6% | 0.27% | 77.4% | 77.3% | 0.15% | 66.7% | 66.6% | 0.09% |
| MLR only | 68.7% | 68.5% | 0.19% | 71.3% | 71.3% | 0.03% | 59.6% | 60.0% | 0.39% |
| IRD only | 65.2% | 64.8% | 0.43% | 67.8% | 67.7% | 0.08% | 55.1% | 55.3% | 0.2% |
| COT-DIR (Full) | 80.4% | 80.4% | 0.03% | 83.4% | 83.5% | 0.05% | 73.2% | 73.3% | 0.06% |
| w/o IRD | 72.8% | 71.8% | 1.05% | 75.5% | 75.5% | 0.02% | 61.7% | 61.5% | 0.15% |
| CV only | 62.9% | 62.1% | 0.8% | 64.7% | 64.7% | 0.05% | 52.8% | 52.7% | 0.05% |

## Data Quality Summary

- **Maximum Overall Accuracy Difference**: 1.05%
- **Maximum L2 Accuracy Difference**: 0.15%
- **Maximum L3 Accuracy Difference**: 0.39%
- **Maximum Efficiency Difference**: 1.59s

## Component Analysis

### Component Contributions (Calculated)

- **IRD Contribution**: +8.6% (removing IRD drops performance by 8.6%)
- **MLR Contribution**: +5.8% (removing MLR drops performance by 5.8%)
- **CV Contribution**: +2.7% (removing CV drops performance by 2.7%)

### Individual Component Performance

- **IRD only**: 64.8%
- **MLR only**: 68.5%
- **CV only**: 62.1%
