# Table 9 Raw Experimental Data Verification Report

**Generation Date**: 2025-06-23 01:02:00
**Total Experiments**: 28,000
**Component Combinations**: 4

## Component Combination Performance Verification

| Combination | Expected Acc | Actual Acc | Acc Diff | Expected Error | Actual Error | Error Diff | Synergy Score |
|-------------|--------------|------------|----------|----------------|--------------|------------|---------------|
| IRD + CV | 78.3% | 77.1% | 1.21% | 15.8% | 22.9% | 7.11% | 0.69 |
| IRD + MLR | 78.9% | 77.4% | 1.5% | 19.2% | 22.6% | 3.4% | 0.71 |
| IRD + MLR + CV | 80.4% | 79.4% | 1.05% | 13.1% | 20.6% | 7.55% | 0.84 |
| MLR + CV | 76.4% | 75.2% | 1.24% | 17.3% | 24.8% | 7.54% | 0.66 |

## Data Quality Summary

- **Maximum Accuracy Difference**: 1.50%
- **Maximum Error Rate Difference**: 7.55%
- **Runtime Range**: 2.9s - 4.1s

## Component Interaction Analysis

### Synergy Effects

1. **IRD + MLR + CV**: 0.84 synergy score
2. **IRD + MLR**: 0.71 synergy score
3. **IRD + CV**: 0.69 synergy score
4. **MLR + CV**: 0.66 synergy score

### Performance vs Component Complexity

Analysis of how adding components affects performance:

**Two-Component Combinations:**
- IRD + MLR: 77.4% accuracy, 22.6% error rate
- IRD + CV: 77.1% accuracy, 22.9% error rate
- MLR + CV: 75.2% accuracy, 24.8% error rate

**Three-Component Combination:**
- IRD + MLR + CV: 79.4% accuracy, 20.6% error rate

### Component Importance Analysis

Based on performance when combined with other components:

- **IRD** (in combinations): 78.0% average accuracy
- **MLR** (in combinations): 77.3% average accuracy
- **CV** (in combinations): 77.2% average accuracy
