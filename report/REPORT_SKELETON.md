# Technical report skeleton and evidence map

Legend: ✅ evidence exists in the repository · ▶ produced by a command you must run · ✍ needs collected data or your own writing.
Rule: **no number enters the report unless it comes from a file named here.**
Label every statement as OBSERVATION (measured), INFERENCE (plausible explanation) or HYPOTHESIS (needs more experiments).

## 1. Introduction
- Problem, motivation, research question, contributions: README §1.
- State the scope honestly: synthetic data; one line per image; multiline is the studied weakness.
- Missing: related-work paragraph (CRNN, CTC: cite Shi et al. 2015, Graves et al. 2006) ✍.

## 2. Dataset and Data Preparation
| Needed | Source |
|---|---|
| Dataset description, licence, access date | OpenHand-Synth page ✍ (add licence and date) |
| Split sizes, multiline share, noise share, writers, length histogram | ▶ `python src/dataset_stats.py` → `results/tables/dataset_split_summary.md`, `results/figures/fig_dataset_distribution.png` |
| Vocabulary (69 chars), CTC feasibility filter (why 208 test samples) | `src/dataset.py`, `notebooks/05_vocabulary_analysis.py`, `07_ctc_feasibility.py` |
| Preprocessing description | `htr_common.preprocess_image` (aspect-preserving resize, white-canvas padding) |
| Limitations: synthetic, machine-generated sentences, neatness confounded, tall images shrink | state explicitly; never call it real student handwriting |

## 3. Deep Learning Fundamentals and Model
- CNN feature extraction, BiLSTM sequence modelling, CTC loss and blank, greedy decoding: explain each in your own words ✍.
- Architecture figure with tensor shapes: [B,1,128,1024] → CNN [B,256,8,256] → 2048 features/step → BiLSTM 512 → 70 classes ✍ (draw). Parameter counts: `tests/test_htr_common.py` confirms 7,296,710 and 9,393,862 ✅.
- Justify design choices (width = time axis; why CTC: no character alignment labels).

## 4. System Design and Implementation
- App architecture: `app/app.py` (UI) → `app/app_logic.py` → `src/htr_common.py` (identical preprocessing/decoding to training) ✅.
- Deployment checkpoint with provenance: `models/final_model.json` ▶ (`export_final_model.py`).
- Error handling and user warnings (corrupt image, tall image, dark/low-contrast) ✅ tested in `tests/test_app_logic.py`.
- Screenshots of the app ✍ (take yourself).

## 5. Experimental Methodology
- Controlled comparison table of what changed per experiment ✅ README §4 and `experiments/*.md`.
- Metrics and aggregation definitions (corpus vs macro) ✅ `htr_common.py`.
- Common 208-sample test set; subgroup definitions (`length_category`, `is_multiline`, `is_collapsed`) ✅.
- Reproducibility: seeds, hyperparameters, hardware, `results/environment.json` ▶ (`record_environment.py`).
- Be explicit: one run per configuration; baseline was not seeded.

## 6. Results
| Item | Source |
|---|---|
| Final comparison table, deltas vs baseline | ▶ `results/tables/model_comparison.md`, `delta_vs_baseline.md` |
| Training/validation curves | ▶ `results/figures/fig_training_curves.png` (needs `*_history.csv` for each run; commit `experiment3_history.csv`; baseline has validation CER only unless re-trained) |
| Subgroup tables (multiline, length, noise, source) | ▶ `results/tables/subgroup_*.md`, `fig_error_by_subgroup.png` |
| Paired bootstrap | ▶ `results/tables/paired_bootstrap_vs_baseline.md` |
| Prediction examples | ▶ `results/tables/prediction_examples.md`, `fig_prediction_length.png` |
| Compute: parameters, ms/image | ▶ `meta` in `results/analysis/*_analysis.json` (needs GPU run) |
| Experiment 3 subgroup result | ▶ **must run** `python src/analyze_model.py --model experiment3` |

## 7. Analysis and Discussion
Verified facts you can build on (from committed baseline/Exp 2 predictions):
- OBSERVATION: baseline single-line macro-CER 17.1% vs multiline 88.0%.
- OBSERVATION: every "very long" (101+) test sample is multiline, so length and layout cannot be separated at the top end. Within 31–60 characters, multiline macro-CER 71.4% (n=9) vs single-line 14.9% (n=48), which suggests layout matters beyond length (small n).
- OBSERVATION: Exp 2 degrades single-line text too (17.1% → ~55% macro-CER), so its failure is not multiline-specific.
- OBSERVATION: baseline macro-CER says noisy images are slightly easier (40.0 vs 42.5) while corpus CER says harder; this is why aggregation must be named.
- INFERENCE (to discuss): all runs stopped with validation CER still falling steeply, so part of every gap may be under-training; larger inputs/changed sampling may need more epochs.
- INFERENCE: the BiLSTM has to read a 2-D page as one 1-D sequence; stacking lines vertically and shrinking them destroys the line structure the CTC alignment assumes.
- HYPOTHESIS: line segmentation + per-line recognition, or an attention/2-D model, would address multiline; not tested (a simple segmentation attempt was unreliable).
- Discuss trade-offs: size/compute vs accuracy; Exp 3 corpus vs macro CER disagreement ✍ after the subgroup run.

## 8. Real-World Evaluation
- Protocol: `evaluation/PROTOCOL.md` ✅. Results: ▶ `evaluation/results/realworld_summary.md` and `usability_summary.md` ✍ (collect first).
- Include n beside every percentage, failure examples, and who the participants were.
- If no external participants: say so; report as a limitation.

## 9. Conclusion
Answer the research question with the measured evidence only; list limitations (README §9); future work (line segmentation, real handwriting data such as SMHD or Indonesian DBABS for fine-tuning, longer training, language-model decoding).

## 10. AI Usage Declaration
Use `report/AI_USAGE_LOG.md` (complete it with your own account) ✍.

## Figure/table checklist
- [ ] fig_dataset_distribution  - [ ] fig_training_curves  - [ ] fig_model_comparison
- [ ] fig_error_by_subgroup  - [ ] fig_prediction_length  - [ ] model_comparison table
- [ ] subgroup tables  - [ ] bootstrap table  - [ ] prediction_examples  - [ ] architecture diagram
- [ ] app screenshots  - [ ] real-world table  - [ ] SUS summary
