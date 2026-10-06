# Deep Learning-Based Recognition of Indonesian Handwritten Text Under Real-World Image Conditions

BINUS University · COMP6826001 Deep Learning · Odd Semester 2026/2027

A CNN + BiLSTM + CTC system that converts an image of Indonesian handwriting
into digital text, with a controlled investigation of where and why it fails
(long and multiline handwriting) and a working Streamlit application.

> **Honest scope.** The model is trained and tested on **synthetic** handwriting
> (OpenHand-Synth), *not* on real student notes. Results on synthetic data do not
> predict accuracy on real photos; the real-world evaluation protocol in
> [`evaluation/`](evaluation/PROTOCOL.md) exists to measure that gap.

## 1. Problem and motivation

Students take handwritten notes, but searching, sharing and editing them requires
digital text. Handwriting varies by writer, and photos add noise, blur and uneven
lighting. We ask:

> **How well can a deep learning model recognise Indonesian handwritten text when
> handwriting style, text length and layout change — and what limits it?**

The baseline error analysis showed that **long and multiline text is the dominant
weakness**, so Experiments 1–3 test three different ideas for addressing it.

## 2. Pipeline

```
image → grayscale → resize to height, pad to 128×1024 white canvas
      → CNN (4 conv blocks, width kept as the sequence axis)
      → 2-layer BiLSTM (hidden 256)
      → linear classifier (70 classes = 69 characters + CTC blank)
      → CTC greedy decoding → text
```

Baseline: 7,296,710 parameters; AdamW (lr 3e-4, weight decay 1e-4), batch size 4,
10 epochs, gradient clipping 5, CTC loss. Model code: `src/model.py`.
Preprocessing, decoding and metrics are defined once in `src/htr_common.py` and
shared by training, evaluation and the app.

## 3. Dataset

[OpenHand-Synth](https://huggingface.co/datasets/to-be/OpenHand-Synth), Indonesian
(`ind`) subset: 1,665 train / 207 validation / 209 test images (after dropping
samples that cannot be emitted within the CTC length: 1,655 train and 208 test
for the 256-step models). Metadata: writer, neatness, line width, colour, augmented noise,
text source. **Limitations:** synthetic handwriting; machine-generated text
sources (Tatoeba sentences, Faker names/dates); about half the samples carry
synthetic noise; neatness is confounded with other variables and is *not*
interpreted causally. Generate the exact statistics with
`python src/dataset_stats.py`.

## 4. Experiments

All models are evaluated on the **same 208 test samples**.
Primary metrics: **corpus CER, corpus WER, exact match**. Secondary: macro-CER
(mean of per-sample CER) for subgroup diagnostics. Never mix the two CERs.

| Model | Change from baseline | Corpus CER | Corpus WER | Exact | Macro-CER |
|---|---|---:|---:|---:|---:|
| **Baseline** | – | **68.79%** | **85.08%** | **6.25%** | **41.28%** |
| Exp 1 | width 1024→1536, 256→384 CTC steps | 71.80% | 89.37% | 4.33% | 47.18% |
| Exp 2 | height 128→192 | 81.70% | 99.07% | 0.00% | 67.84% |
| Exp 3 | 50/50 multiline-balanced sampling | 81.36% | 98.37% | 0.00% | 70.09% |

Regenerate every table and figure from the saved predictions:

```bash
python src/analyze_model.py --model all      # needs checkpoints + dataset
python src/report_assets.py                  # -> results/tables, results/figures
```

**Observations** (measured, baseline): single-line macro-CER 17.1% vs multiline
88.0%; all 49 "very long" test samples are multiline; 60/208 predictions collapse
to under 25% of the reference length. **Inference, not proven:** configurations
that change the input size or sampling stopped while validation CER was still
falling, so part of their deficit may be under-training; the optional control
`src/train_baseline_extended.py` tests this. Each configuration was trained once
(no seed variance estimate). See [`FINAL_MODEL.md`](FINAL_MODEL.md) and the report
for the full analysis.

## 5. Application

```bash
python src/export_final_model.py --model baseline   # creates models/final_model.pt
streamlit run app/app.py
```

Upload an image → preview → *Recognize text* → editable recognised text, a
heuristic confidence, warnings for tall/dark/low-contrast inputs, and the
preprocessed canvas the model sees. The sidebar states the model's measured
synthetic-test numbers and that it was trained on synthetic data and is meant for
**one line per image**. An optional form logs practical-evaluation results.

To put it on a website (Streamlit Community Cloud, Hugging Face Spaces or Docker), see [`DEPLOY.md`](DEPLOY.md).

## 6. Real-world evaluation

[`evaluation/PROTOCOL.md`](evaluation/PROTOCOL.md): (A) a photo test set written by
real people under real capture conditions, scored with
`python src/evaluate_realworld.py`; (B) a usability session with the SUS
questionnaire, scored with `python src/summarize_usability.py`.
**Results appear in the report only once collected.**

## 7. Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests -q                  # 52 tests, no dataset/GPU needed
python src/train.py                        # baseline  -> checkpoints/baseline_best.pt
python src/train_experiment1.py
python src/train_experiment2.py
python src/train_experiment3.py
python src/analyze_model.py --model all
python src/report_assets.py
python src/record_environment.py           # versions + checkpoint hashes
```

Run everything from the repository root. Hardware used: RTX 3060 Ti (8 GB), Ubuntu.
Seeds: Experiment 3 uses seed 42; `train.py` now sets seed 42 too, but the reported
baseline was trained before that was added (checkpoint hash recorded in
`results/analysis/baseline_analysis.json`). Checkpoints are git-ignored.

## 8. Repository layout

```
src/htr_common.py          shared preprocessing / decoding / metrics / model loading
src/dataset.py, dataloader.py, model.py, model_experiment2.py
src/train*.py              baseline + Experiments 1-3 (+ optional extended control)
src/analyze_model.py       unified evaluation + error analysis for any model
src/report_assets.py       report tables and figures from real outputs
src/export_final_model.py  slim deployment checkpoint with provenance
src/evaluate_realworld.py, summarize_usability.py
app/                       Streamlit app (app.py, app_logic.py, evaluation_log.py)
evaluation/                real-world protocol, prompts, questionnaire, templates
experiments/               per-experiment write-ups
results/                   histories, predictions, tables, figures
report/                    report skeleton, evidence map, AI usage log
tests/                     pytest suite
```

Legacy per-experiment scripts (`error_analysis*.py`, `evaluate*.py`,
`compare_*.py`) are kept for traceability; `analyze_model.py` reproduces their
numbers (verified by tests on the committed predictions).

## 9. Current limitations

Synthetic training data; one line per image is the supported use; multiline and
long text mostly fail; single training run per configuration; models trained for
10 epochs only; greedy decoding without a language model; no line segmentation
(simple attempts were unreliable); performance on real photos not yet measured.
