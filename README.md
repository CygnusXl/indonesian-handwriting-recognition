# Indonesian Handwritten Academic Notes Recognition

Deep learning-based recognition of Indonesian handwritten text under real-world image conditions.

## Project status

**Experimental phase — baseline established, Experiment 1 completed, Experiment 2 completed, Experiment 3 planned**

The project now has a working CNN-BiLSTM handwriting-recognition pipeline, baseline training/evaluation, two controlled experiments, and detailed error analysis.

## Problem

Students commonly write academic notes by hand, but handwriting styles differ substantially. Photos of handwritten notes can also contain blur, noise, uneven lighting, rotation, shadows, corrections, and other visual degradation.

The project investigates whether a deep learning handwriting-recognition system can convert Indonesian handwritten text into digital text and how recognition performance changes under different handwriting styles, text lengths, and image conditions.

## Target scenario

A student takes or uploads a photo of handwritten Indonesian notes. The system processes the image and produces recognized digital text.

## Core research question

> How well can a deep learning model recognize Indonesian handwritten text when handwriting style and image conditions change?

Current error analysis shows that long and multiline text is the dominant observed weakness of the baseline system.

## Current workflow

1. Define the real-world problem and target scenario.
2. Verify and analyze the dataset.
3. Prepare and preprocess the data.
4. Construct and train a CNN-BiLSTM baseline.
5. Evaluate the baseline.
6. Run controlled experiments.
7. Analyze training behavior and recognition errors.
8. Compare approaches and document trade-offs.
9. Build a working application.
10. Perform real-world/practical evaluation.

## Dataset

The current benchmark is OpenHand-Synth (to-be/OpenHand-Synth), filtered to Indonesian (ind) samples.

Current Indonesian split sizes after CTC-feasibility filtering:

- Train: 1,655 usable samples
- Validation: 206 usable samples
- Test: 208 usable samples for the common experiment comparison

The dataset includes handwriting-style metadata such as writer ID, neatness, line width, color, augmented noise, and source.

A major limitation is that the dataset is synthetic rather than a collection of real university lecture notes. This is documented as a limitation rather than being represented as real student-note data.

## Baseline model

CNN-BiLSTM with CTC loss:

- Input: 128 x 1024 grayscale image
- CNN feature extractor
- 2-layer bidirectional LSTM
- CTC output with 70 classes (69 characters + blank)
- Parameters: approximately 7.3 million

Baseline training uses AdamW, learning rate 0.0003, weight decay 0.0001, effective batch size 4, and 10 epochs.

## Experiments completed

### Experiment 1 — Increased horizontal resolution

Question: Does increasing horizontal image resolution and CTC sequence capacity improve recognition of long and multiline handwriting?

Configuration: 128 x 1536 input, 384 CTC timesteps, same CNN-BiLSTM design, physical batch size 2, gradient accumulation 2, effective batch size 4.

On the same 208 test samples, Experiment 1 performed worse than the baseline:

- Baseline macro-CER: 41.28%
- Experiment 1 macro-CER: 47.18%

### Experiment 2 — Increased vertical resolution

Question: Does increasing vertical image resolution preserve more multiline information and improve recognition?

Configuration: 192 x 1024 input, 256 CTC timesteps, same horizontal resolution as baseline, with the corresponding larger CNN feature representation.

On the same 208 test samples:

- Baseline macro-CER: 41.28%
- Experiment 2 macro-CER: 67.84%
- Baseline WER: 73.82%
- Experiment 2 WER: 99.00%
- Baseline exact match: 6.25%
- Experiment 2 exact match: 0.00%

Experiment 2 therefore did not improve recognition. Error analysis shows that performance degradation affects both single-line and multiline samples, although multiline and long text remain especially difficult.

## Error analysis

For Experiment 2:

- Mean sample CER: 67.84%
- Exact matches: 0/208
- Severe prediction collapse: 71/208 (34.13%)
- Single-line mean sample CER: 54.75%
- Multiline mean sample CER: 93.10%
- Short mean sample CER: 57.49%
- Medium mean sample CER: 55.43%
- Long mean sample CER: 84.62%
- Very long mean sample CER: 93.87%

The strongest observed failure pattern remains long/multiline sequence recognition.

## Next experiment

**Experiment 3 — Multiline-balanced training**

The next controlled experiment will keep the baseline image size and architecture unchanged while increasing the training exposure to multiline samples. This directly tests whether the severe multiline weakness is partly related to the training sample distribution.

The validation and test sets will remain unchanged for fair comparison.

## Evaluation

The project uses:

- Character Error Rate (CER)
- Word Error Rate (WER)
- Exact-match rate
- Training/validation loss
- Metadata-based error analysis
- Representative failure examples

The report will standardize the final CER/WER aggregation methodology across all models before presenting the final Results chapter.

## Application

Planned system:

Handwritten note image -> preprocessing -> trained DL model -> recognized Indonesian text

The application is the demonstration layer. The main academic contribution is the model construction, controlled experimentation, error analysis, and evaluation of practical limitations.

## Reproducibility

Training configurations, preprocessing choices, experiment settings, evaluation results, and relevant code are recorded in this repository.

No experimental metric, user feedback, or benchmark result will be reported until it has actually been obtained.

## Academic project

This repository supports the BINUS Deep Learning final project and documents the complete experimental workflow and working application.