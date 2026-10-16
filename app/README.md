# Application

Streamlit app: `streamlit run app/app.py` (from the repository root; needs
`models/final_model.pt`, see the main README).

Flow shown on the page:

1. Input image
2. Preprocessed image (original / normalised / processed)
3. Detected handwriting lines (numbered boxes, top to bottom)
4. Recognised text (heuristic confidence, per-line results, warnings)
5. "What the model sees": the exact crops fed to the recognizer

Look and feel: `style.css` (ruled-notebook theme), `ui_components.py` (escaped HTML pieces),
`../.streamlit/config.toml` (colours). Processing starts automatically after an upload.

Files: `app.py` (UI), `app_logic.py` (model loading, upload checks, `recognize_photo`),
`evaluation_log.py` (optional practical-evaluation CSV). The processing itself lives in
`src/document_pipeline.py` and `src/page_recognition.py`; see `../DOCUMENT_PIPELINE.md`.
Set `HTR_HOSTED=1` on a public deployment (see `../DEPLOY.md`).
