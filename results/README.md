# Results

This directory contains generated figures, tables, experiment histories, and experiment summaries.

All metrics below come from actual project runs.

## Current common-test comparison

The three models were evaluated on the same **208 CTC-feasible Indonesian test samples**.

| Model | Input | Test macro-CER | Test WER | Exact match |
| --- | --- | ---: | ---: | ---: |
| Baseline | 128 x 1024 | 41.28% | 73.82% | 6.25% |
| Experiment 1 | 128 x 1536 | 47.18% | 82.33% | 4.33% |
| Experiment 2 | 192 x 1024 | 67.84% | 99.00% | 0.00% |

### Changes relative to baseline

- Experiment 1 CER: +5.89 percentage points
- Experiment 1 WER: +8.51 percentage points
- Experiment 1 exact match: -1.92 percentage points
- Experiment 2 CER: +26.56 percentage points
- Experiment 2 WER: +25.17 percentage points
- Experiment 2 exact match: -6.25 percentage points

## Experiment 2 error analysis

- 208 test samples
- Mean sample CER: 67.84%
- Exact matches: 0/208
- Severe prediction collapse: 71/208 (34.13%)
- Single-line mean sample CER: 54.75%
- Multiline mean sample CER: 93.10%
- Long mean sample CER: 84.62%
- Very long mean sample CER: 93.87%

## Metric note

The common comparison currently reports mean per-sample (macro) CER. The earlier evaluator also produced a corpus-level CER. The final report will standardize one aggregation methodology across models and retain macro-CER for subgroup error analysis where appropriate.

## Files

- `all_experiments_comparison.csv`
- `all_experiments_comparison_summary.txt`
- `experiment2_history.csv`
- `experiment2_error_analysis_summary.txt`
- `experiment_2.md` in `experiments/`