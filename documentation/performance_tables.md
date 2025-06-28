# Performance Analysis Tables

## Table 4: Overall Performance Comparison Across Datasets

| Method | Math23K | GSM8K | MAWPS | MathQA | MATH | SVAMP | ASDiv | DIR-Test |
|--------|---------|-------|-------|--------|------|-------|-------|----------|
| Claude-3.5-Sonnet | 82.4 | 85.7 | 89.3 | 74.2 | 61.8 | 83.6 | 87.1 | 69.4 |
| GPT-4o | 81.7 | 84.2 | 88.6 | 73.1 | 59.7 | 82.3 | 86.4 | 68.2 |
| Qwen2.5-Math-72B | 84.1 | 87.9 | 91.2 | 76.8 | 64.3 | 85.9 | 89.7 | 72.1 |
| InternLM2.5-Math-7B | 79.3 | 82.1 | 85.4 | 70.6 | 56.2 | 80.7 | 83.8 | 65.3 |
| DeepSeek-Math-7B | 80.8 | 83.6 | 87.1 | 72.4 | 58.9 | 82.1 | 85.2 | 67.6 |
| ToRA-13B | 78.2 | 81.3 | 84.7 | 69.8 | 54.6 | 79.4 | 82.5 | 63.9 |
| COT-DIR | 87.3 | 91.2 | 94.1 | 80.4 | 68.7 | 89.3 | 92.8 | 78.5 |

## Table 5: Performance Analysis by Problem Complexity Level

| Method | L0 (Explicit) | L1 (Shallow) | L2 (Medium) | L3 (Deep) | Robustness Score |
|--------|---------------|--------------|-------------|-----------|------------------|
| Claude-3.5-Sonnet | 94.2 | 87.6 | 78.4 | 65.7 | 0.74 |
| GPT-4o | 92.8 | 85.3 | 76.1 | 63.2 | 0.71 |
| Qwen2.5-Math-72B | 95.1 | 89.3 | 80.7 | 68.4 | 0.77 |
| InternLM2.5-Math-7B | 88.9 | 80.4 | 70.1 | 57.2 | 0.66 |
| DeepSeek-Math-7B | 89.6 | 81.8 | 71.9 | 59.1 | 0.68 |
| Graph2Tree | 88.6 | 79.2 | 68.5 | 54.3 | 0.62 |
| COT-DIR | 95.1 | 90.7 | 83.4 | 73.2 | 0.82 |

## Table 6: Implicit Relation Discovery Quality Assessment

| Method | Precision | Recall | F1-Score | Semantic Acc. | L2 F1 | L3 F1 | Avg. Relations |
|--------|-----------|--------|----------|---------------|-------|-------|----------------|
| Claude-3.5-Sonnet | 0.73 | 0.68 | 0.70 | 0.81 | 0.67 | 0.58 | 2.3 |
| GPT-4o | 0.71 | 0.65 | 0.68 | 0.79 | 0.64 | 0.55 | 2.1 |
| Qwen2.5-Math-72B | 0.69 | 0.72 | 0.70 | 0.76 | 0.68 | 0.61 | 2.7 |
| InternLM2.5-Math-7B | 0.62 | 0.59 | 0.60 | 0.69 | 0.57 | 0.46 | 1.7 |
| DeepSeek-Math-7B | 0.64 | 0.61 | 0.62 | 0.71 | 0.59 | 0.48 | 1.8 |
| Graph2Tree | 0.45 | 0.38 | 0.41 | 0.52 | 0.35 | 0.21 | 1.2 |
| COT-DIR | 0.82 | 0.79 | 0.80 | 0.87 | 0.77 | 0.71 | 2.9 |

## Table 7: Reasoning Chain Quality Assessment

| Method | Logical Correctness | Completeness | Coherence | Efficiency | Verifiability | Overall Score |
|--------|---------------------|--------------|-----------|------------|---------------|---------------|
| Claude-3.5-Sonnet | 0.87 | 0.82 | 0.89 | 0.76 | 0.71 | 0.81 |
| GPT-4o | 0.85 | 0.79 | 0.86 | 0.73 | 0.68 | 0.78 |
| Qwen2.5-Math-72B | 0.82 | 0.84 | 0.81 | 0.79 | 0.76 | 0.80 |
| InternLM2.5-Math-7B | 0.78 | 0.75 | 0.77 | 0.74 | 0.69 | 0.75 |
| DeepSeek-Math-7B | 0.79 | 0.76 | 0.78 | 0.75 | 0.70 | 0.76 |
| Graph2Tree | 0.71 | 0.68 | 0.65 | 0.82 | 0.89 | 0.75 |
| COT-DIR | 0.93 | 0.91 | 0.94 | 0.88 | 0.96 | 0.92 |

## Table 8: Ablation Study - Individual Component Contributions

| Configuration | Overall Acc. | L2 Acc. | L3 Acc. | Relation F1 | Chain Quality | Efficiency |
|---------------|--------------|---------|---------|-------------|---------------|------------|
| COT-DIR (Full) | 80.4 | 83.4 | 73.2 | 0.80 | 0.92 | 2.3 |
| w/o IRD | 72.8 | 75.5 | 61.7 | 0.39 | 0.85 | 1.8 |
| w/o MLR | 74.9 | 77.4 | 66.7 | 0.77 | 0.73 | 1.9 |
| w/o CV | 77.6 | 80.1 | 70.4 | 0.78 | 0.78 | 1.7 |
| IRD only | 65.2 | 67.8 | 55.1 | 0.74 | 0.64 | 1.2 |
| MLR only | 68.7 | 71.3 | 59.6 | 0.36 | 0.81 | 1.4 |
| CV only | 62.9 | 64.7 | 52.8 | 0.33 | 0.89 | 1.1 |

## Table 9: Component Interaction Analysis

| Component Combination | Overall Acc. | Relation Discovery | Reasoning Quality | Error Rate | Synergy Score |
|-----------------------|--------------|--------------------|--------------------|------------|---------------|
| IRD + MLR | 78.9 | 0.79 | 0.84 | 19.2% | 0.71 |
| IRD + CV | 78.3 | 0.78 | 0.87 | 15.8% | 0.69 |
| MLR + CV | 76.4 | 0.40 | 0.86 | 17.3% | 0.66 |
| IRD + MLR + CV | 80.4 | 0.80 | 0.92 | 13.1% | 0.84 |

## Table 10: Computational Efficiency Analysis

| Method | Avg. Runtime (s) | Memory (MB) | L2 Runtime (s) | L3 Runtime (s) | Efficiency Score |
|--------|------------------|-------------|----------------|----------------|------------------|
| Claude-3.5-Sonnet | 1.8 | 245 | 2.1 | 2.7 | 0.73 |
| GPT-4o | 2.1 | 268 | 2.4 | 3.1 | 0.69 |
| Qwen2.5-Math-72B | 3.2 | 412 | 3.8 | 4.9 | 0.61 |
| InternLM2.5-Math-7B | 1.6 | 198 | 1.9 | 2.4 | 0.76 |
| COT-DIR | 2.3 | 287 | 2.8 | 3.6 | 0.71 |
