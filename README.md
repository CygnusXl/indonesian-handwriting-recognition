# Indonesian Handwritten Academic Notes Recognition

Deep learning-based recognition of messy Indonesian handwritten academic notes.

## Project status

**Setup phase — dataset verification and problem definition**

This repository will contain the implementation, experiments, evaluation, and working application for the Deep Learning final project.

## Problem

Students commonly write academic notes by hand, but handwriting styles differ substantially. Photos of handwritten notes can also contain blur, noise, uneven lighting, rotation, shadows, corrections, and other visual degradation.

The project investigates whether a specialized deep learning handwriting-recognition system can convert Indonesian handwritten academic notes into digital text and how recognition performance changes under different handwriting styles and image-quality conditions.

## Target scenario

A student takes or uploads a photo of handwritten Indonesian academic notes. The system processes the image and produces recognized digital text.

## Research question

> How well can a deep learning model recognize Indonesian academic handwriting when handwriting style and image quality change?

## Planned workflow

1. Define the real-world problem and target scenario.
2. Verify and analyze suitable handwriting datasets.
3. Prepare and preprocess the data.
4. Construct and train a baseline deep learning model.
5. Evaluate the baseline.
6. Run at least two controlled experiments.
7. Analyze training behavior and recognition errors.
8. Compare approaches and document trade-offs.
9. Build a working application that accepts handwriting images and displays recognized text.
10. Evaluate practical suitability and limitations.

## Planned experiments

The exact experiments will be finalized after dataset and baseline-model analysis.

Initial candidates:

- **Baseline:** standard training configuration.
- **Experiment 1:** augmentation and simulated image degradation such as rotation, blur, and noise.
- **Experiment 2:** alternative model architecture or training configuration.

Experiments will be changed only when the dataset and baseline justify the choice.

## Evaluation

For handwriting transcription, the project will investigate metrics such as:

- Character Error Rate (CER)
- Word Error Rate (WER)
- Training and validation loss
- Additional task-appropriate metrics where applicable

Error analysis will examine representative recognition failures rather than relying only on a single aggregate metric.

## Application

The final system is planned as a simple interface:

**Handwritten note image → preprocessing → trained DL model → recognized Indonesian text**

The application is the working system used to demonstrate the trained model; the main research focus is the model, experimentation, evaluation, and robustness analysis.

## Repository structure

```text
indonesian-handwriting-recognition/
├── README.md
├── requirements.txt
├── data/
├── notebooks/
├── src/
├── experiments/
├── results/
├── app/
└── report/
```

More detailed subdirectories will be added as the implementation develops.

## Dataset

No dataset is committed to this repository at this stage.

Dataset selection must be verified for:

- Indonesian-language coverage
- handwriting images
- corresponding transcription/ground-truth text
- writer diversity
- suitability for academic handwriting
- licensing and permitted use
- dataset size and quality
- limitations and possible bias

Dataset sources and preparation procedures will be documented after verification.

## Reproducibility

Training configurations, preprocessing choices, experiment settings, evaluation results, and relevant code will be recorded in this repository.

No experimental metric, user feedback, or benchmark result will be reported until it has actually been obtained.

## Academic project

This repository supports the BINUS Deep Learning final project and is intended to document the complete experimental workflow and working application.
