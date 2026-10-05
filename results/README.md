# Results

This directory contains generated figures, tables, experiment histories, and experiment summaries.

All metrics below come from actual project runs.

## Metric standardization

For the final report, the **primary overall metrics** will be:

- **Corpus CER:** total character edit distance divided by total reference characters.
- **Corpus WER:** total word edit distance divided by total reference words.
- **Exact match:** proportion of samples whose complete prediction exactly matches the reference.

**Macro-CER** (mean of per-sample CER values) remains a secondary diagnostic metric for subgroup/error analysis because it weights each sample equally.

The earlier comparison mixed aggregation conventions. To avoid that inconsistency, run \`src/compare_experiments.py\` after Experiment 3 is trained. It recomputes Baseline, Experiment 1, Experiment 2, and Experiment 3 on the same common test sample indices and writes one standardized table using corpus-level CER/WER.

## Current common-test comparison

The existing comparison used the same **208 CTC-feasible Indonesian test samples**.

| Model | Input | Test macro-CER | Test WER | Exact match |
| --- | --- | ---: | ---: | ---: |
| Baseline | 128 x 1024 | 41.28% | 73.82% | 6.25% |
| Experiment 1 | 128 x 1536 | 47.18% | 82.33% | 4.33% |
| Experiment 2 | 192 x 1024 | 67.84% | 99.00% | 0.00% |

These are **legacy macro-CER comparison values** retained for traceability. Do not mix them with the final standardized corpus-level comparison.

## Experiment 2 error analysis

- 208 test samples
- Mean sample CER: 67.84%
- Exact matches: 0/208
- Severe prediction collapse: 71/208 (34.13%)
- Single-line mean sample CER: 54.75%
- Multiline mean sample CER: 93.10%
- Long mean sample CER: 84.62%
- Very long mean sample CER: 93.87%

## Files

- \`all_experiments_standardized.csv\`
- \`all_experiments_standardized_summary.txt\`
- \`all_experiments_comparison.csv\`
- \`all_experiments_comparison_summary.txt\`
- \`experiment3_history.csv\`
- \`experiment2_history.csv\`
- \`experiment2_error_analysis_summary.txt\`
- \`experiment_2.md\` in \`experiments/\`
- \`experiment_3.md\` in \`experiments/\`
