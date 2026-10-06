# Final model designation

**Designated final model: Baseline CNN-BiLSTM-CTC (`checkpoints/baseline_best.pt`)**
Deployed as `models/final_model.pt` (created by `python src/export_final_model.py --model baseline`).

Status: **provisional until the Experiment 3 subgroup analysis is run** (see "Open check").

## Why the baseline

All comparisons use the same 208 common test samples.

| Criterion | Baseline | Exp 1 | Exp 2 | Exp 3 | Best |
|---|---:|---:|---:|---:|---|
| Best validation CER* | 69.56% | 72.96% | 82.06% | 81.67% | Baseline |
| Test corpus CER | 68.79% | 71.80% | 81.70% | 81.36% | Baseline |
| Test corpus WER | 85.08% | 89.37% | 99.07% | 98.37% | Baseline |
| Test exact match | 6.25% | 4.33% | 0.00% | 0.00% | Baseline |
| Test macro-CER | 41.28% | 47.18% | 67.84% | 70.09% | Baseline |
| Parameters | 7.30 M | see report | 9.39 M | 7.30 M | Baseline (smallest tie) |

\*Validation CERs come from each run's own validation set and are only roughly
comparable (Exp 1 uses a 384-step CTC filter, so its validation set differs slightly).

Selection rests on **validation** performance first (the baseline also has the best
validation CER), and the test set confirms it; the test set was not used to tune
anything. The baseline wins on every primary and secondary metric, is the
smallest/fastest configuration, and has the simplest preprocessing.

## What "best" does and does not mean

- Best **of four configurations trained once each for 10 epochs.** Differences of a few
  points (baseline vs Exp 1) are within what a different seed could change; the
  paired bootstrap in `results/tables/paired_bootstrap_vs_baseline.md` only measures
  test-sample uncertainty, not training variance.
- Best **on synthetic handwriting.** It has not been shown to work on real handwriting.
- The absolute quality is low (about 69% corpus CER). The model is a
  demonstrator and research object, not a production OCR tool.

## Open check (do before submission)

1. Run `python src/analyze_model.py --model experiment3` and read
   `results/analysis/experiment3_analysis_summary.txt`.
2. If Exp 3 improved the **multiline** subgroup but not overall (a trade-off), keep the
   baseline as the deployed model but report this as a finding: balanced sampling
   moved performance between groups rather than adding capability.
3. If Exp 3 is worse on *every* subgroup, report that multiline oversampling did not help
   under this budget.
4. Optional control: `python src/train_baseline_extended.py` shows whether the
   baseline keeps improving with more epochs (validation CER was still falling at epoch 10).
   If the extended baseline is clearly better on validation, re-export it
   (`--model baseline_extended`) and update this file.
