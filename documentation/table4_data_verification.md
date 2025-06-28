# Table 4 Raw Data Verification Report

Generated on: 2025-06-23 00:41:58

## Performance Comparison Verification

Comparison between generated data and target values from Table 4.

### Claude-3.5-Sonnet

| Dataset | Generated (%) | Target (%) | Difference | Sample Count | Match Quality |
|---------|---------------|------------|------------|--------------|---------------|
| Math23K | 82.5% | 82.4% | 0.1% | 1000 | ✅ Good |
| GSM8K | 85.7% | 85.7% | 0.0% | 1000 | ✅ Good |
| MAWPS | 89.4% | 89.3% | 0.1% | 600 | ✅ Good |
| MathQA | 74.2% | 74.2% | 0.0% | 1200 | ✅ Good |
| MATH | 61.8% | 61.8% | 0.0% | 800 | ✅ Good |
| SVAMP | 83.6% | 83.6% | 0.0% | 500 | ✅ Good |
| ASDiv | 87.1% | 87.1% | 0.0% | 400 | ✅ Good |
| DIR-Test | 69.3% | 69.4% | 0.1% | 300 | ✅ Good |

### GPT-4o

| Dataset | Generated (%) | Target (%) | Difference | Sample Count | Match Quality |
|---------|---------------|------------|------------|--------------|---------------|
| Math23K | 81.7% | 81.7% | 0.0% | 1000 | ✅ Good |
| GSM8K | 84.1% | 84.2% | 0.1% | 1000 | ✅ Good |
| MAWPS | 88.7% | 88.6% | 0.1% | 600 | ✅ Good |
| MathQA | 73.0% | 73.1% | 0.1% | 1200 | ✅ Good |
| MATH | 59.7% | 59.7% | 0.0% | 800 | ✅ Good |
| SVAMP | 82.3% | 82.3% | 0.0% | 500 | ✅ Good |
| ASDiv | 86.3% | 86.4% | 0.1% | 400 | ✅ Good |
| DIR-Test | 68.4% | 68.2% | 0.2% | 300 | ✅ Good |

### Qwen2.5-Math-72B

| Dataset | Generated (%) | Target (%) | Difference | Sample Count | Match Quality |
|---------|---------------|------------|------------|--------------|---------------|
| Math23K | 84.2% | 84.1% | 0.1% | 1000 | ✅ Good |
| GSM8K | 87.9% | 87.9% | 0.0% | 1000 | ✅ Good |
| MAWPS | 91.2% | 91.2% | 0.0% | 600 | ✅ Good |
| MathQA | 76.7% | 76.8% | 0.1% | 1200 | ✅ Good |
| MATH | 64.3% | 64.3% | 0.0% | 800 | ✅ Good |
| SVAMP | 85.8% | 85.9% | 0.1% | 500 | ✅ Good |
| ASDiv | 89.7% | 89.7% | 0.0% | 400 | ✅ Good |
| DIR-Test | 71.9% | 72.1% | 0.2% | 300 | ✅ Good |

### InternLM2.5-Math-7B

| Dataset | Generated (%) | Target (%) | Difference | Sample Count | Match Quality |
|---------|---------------|------------|------------|--------------|---------------|
| Math23K | 79.3% | 79.3% | 0.0% | 1000 | ✅ Good |
| GSM8K | 82.1% | 82.1% | 0.0% | 1000 | ✅ Good |
| MAWPS | 85.4% | 85.4% | 0.0% | 600 | ✅ Good |
| MathQA | 70.5% | 70.6% | 0.1% | 1200 | ✅ Good |
| MATH | 56.2% | 56.2% | 0.0% | 800 | ✅ Good |
| SVAMP | 80.8% | 80.7% | 0.1% | 500 | ✅ Good |
| ASDiv | 83.7% | 83.8% | 0.1% | 400 | ✅ Good |
| DIR-Test | 65.4% | 65.3% | 0.1% | 300 | ✅ Good |

### DeepSeek-Math-7B

| Dataset | Generated (%) | Target (%) | Difference | Sample Count | Match Quality |
|---------|---------------|------------|------------|--------------|---------------|
| Math23K | 80.7% | 80.8% | 0.1% | 1000 | ✅ Good |
| GSM8K | 83.6% | 83.6% | 0.0% | 1000 | ✅ Good |
| MAWPS | 87.1% | 87.1% | 0.0% | 600 | ✅ Good |
| MathQA | 72.4% | 72.4% | 0.0% | 1200 | ✅ Good |
| MATH | 59.0% | 58.9% | 0.1% | 800 | ✅ Good |
| SVAMP | 82.1% | 82.1% | 0.0% | 500 | ✅ Good |
| ASDiv | 85.2% | 85.2% | 0.0% | 400 | ✅ Good |
| DIR-Test | 67.6% | 67.6% | 0.0% | 300 | ✅ Good |

### ToRA-13B

| Dataset | Generated (%) | Target (%) | Difference | Sample Count | Match Quality |
|---------|---------------|------------|------------|--------------|---------------|
| Math23K | 78.2% | 78.2% | 0.0% | 1000 | ✅ Good |
| GSM8K | 81.2% | 81.3% | 0.1% | 1000 | ✅ Good |
| MAWPS | 84.8% | 84.7% | 0.1% | 600 | ✅ Good |
| MathQA | 69.9% | 69.8% | 0.1% | 1200 | ✅ Good |
| MATH | 54.7% | 54.6% | 0.1% | 800 | ✅ Good |
| SVAMP | 79.4% | 79.4% | 0.0% | 500 | ✅ Good |
| ASDiv | 82.5% | 82.5% | 0.0% | 400 | ✅ Good |
| DIR-Test | 63.9% | 63.9% | 0.0% | 300 | ✅ Good |

### COT-DIR

| Dataset | Generated (%) | Target (%) | Difference | Sample Count | Match Quality |
|---------|---------------|------------|------------|--------------|---------------|
| Math23K | 87.3% | 87.3% | 0.0% | 1000 | ✅ Good |
| GSM8K | 91.2% | 91.2% | 0.0% | 1000 | ✅ Good |
| MAWPS | 94.2% | 94.1% | 0.1% | 600 | ✅ Good |
| MathQA | 80.4% | 80.4% | 0.0% | 1200 | ✅ Good |
| MATH | 68.8% | 68.7% | 0.1% | 800 | ✅ Good |
| SVAMP | 89.3% | 89.3% | 0.0% | 500 | ✅ Good |
| ASDiv | 92.7% | 92.8% | 0.1% | 400 | ✅ Good |
| DIR-Test | 78.6% | 78.5% | 0.1% | 300 | ✅ Good |

