# Source code

Run every script from the repository root, e.g. `python src/train.py`.
Modules import each other directly (`from dataset import ...`).

| Group | Files |
|---|---|
| Shared (single source of truth) | `htr_common.py` preprocessing, CTC greedy decoding, CER/WER, subgroup definitions, model registry and loading |
| Data | `dataset.py`, `dataloader.py` |
| Models | `model.py` (baseline, Exp 1, Exp 3), `model_experiment2.py` (192-px height) |
| Training | `train.py` (baseline), `train_experiment1.py`, `train_experiment2.py`, `train_experiment3.py`, optional `train_baseline_extended.py` |
| Evaluation / analysis | `analyze_model.py` (any model, common 208-sample test set), `report_assets.py` (tables + figures), `dataset_stats.py` |
| Deployment | `export_final_model.py` -> `models/final_model.pt` |
| Real-world evaluation | `evaluate_realworld.py` (line photos), `evaluate_pipeline.py` (page photos, pipelines A-D + E), `summarize_usability.py` |
| Photograph pipeline | `document_pipeline.py` (preprocessing, ruling suppression, line detection, crops), `page_recognition.py` (pipelines A-D, calls the unchanged model) |
| Synthetic component checks | `synthetic_pages.py`, `evaluate_segmentation.py` (geometry only, not real handwriting) |
| Reproducibility | `record_environment.py` |
| Legacy (kept for traceability) | `evaluate*.py`, `error_analysis*.py`, `compare_*.py`: superseded by `analyze_model.py`, which reproduces their numbers (see `tests/test_analyze_model.py`) |

Model keys used by `analyze_model.py` and `export_final_model.py`:
`baseline`, `experiment1`, `experiment2`, `experiment3`, `baseline_extended`
(configured in `htr_common.MODEL_REGISTRY`).
