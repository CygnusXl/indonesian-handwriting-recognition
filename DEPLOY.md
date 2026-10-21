# Putting the demo on a website

The app is `app/app.py` (Streamlit). It needs one file that only you can create,
because it comes from your trained checkpoint:

```bash
python src/export_final_model.py --model baseline     # -> models/final_model.pt (+ .json)
```

Test locally first: `streamlit run app/app.py`.

## Option 1 - Streamlit Community Cloud (free, simplest)

1. Commit and push `models/final_model.pt` and `models/final_model.json`
   (about 29 MB; `.gitignore` already allows them; GitHub's limit is 100 MB per file).
2. Go to <https://share.streamlit.io>, sign in with GitHub, **Create app**.
3. Repository: `CygnusXl/indonesian-handwriting-recognition`, branch: your branch,
   main file path: `app/app.py`.
4. Advanced settings -> Secrets/Env: add `HTR_HOSTED = "1"` (as an environment
   variable if offered; otherwise the app still works, but would write evaluation
   rows to the server's temporary disk).
5. Deploy. The first build downloads PyTorch (CPU) and takes a few minutes.
   You get a public `https://<name>.streamlit.app` link to share with testers.

## Option 2 - Hugging Face Spaces (Docker)

1. Create a Space, SDK **Docker**. Clone it and copy this repository's files in
   (at least `Dockerfile`, `src/`, `app/`, `models/`, `.streamlit/`).
2. Add this header at the top of the Space's `README.md`:
   ```
   ---
   title: Indonesian Handwriting Recognition
   sdk: docker
   app_port: 8501
   ---
   ```
3. Files over 10 MB need Git LFS: `git lfs track "models/*.pt"` before committing.
4. Push; the Space builds from the `Dockerfile` and serves the app.

## Option 3 - Temporary public link from your own PC (for a quick test)

```bash
streamlit run app/app.py
# in a second terminal (needs cloudflared installed):
cloudflared tunnel --url http://localhost:8501
```

The link works only while your PC and the command are running. Anyone on your
Wi-Fi can also open `http://<your-PC-ip>:8501` if you start Streamlit with
`--server.address 0.0.0.0`.

## Notes

- CPU inference is fast enough for this 7.3 M-parameter model (see `ms/img` in
  the analysis output); no GPU is needed for hosting.
- On a hosted site, set `HTR_HOSTED=1` so testers' practical-evaluation entries are
  kept in their own session and downloaded by them as CSV, not stored on the server.
  Collect those CSV files from testers and place them in `evaluation/results/`.
- The sidebar already tells visitors that the model was trained on synthetic
  handwriting and that accuracy on real photos is unmeasured; keep that visible.
- The app now needs `opencv-python-headless` (in `app/requirements.txt`) for photo
  preprocessing; it runs on CPU. Latency on large photos on a free hosted CPU has not
  been measured; photos are limited to 2400 px on the long side internally.
- Do not publish `evaluation/photos/` (it may contain people's handwriting).
- The "Translate to Indonesian" button calls Google Translate through the
  `deep-translator` package (listed in `app/requirements.txt`). It needs
  internet access from the server, sends the recognised text (never the
  photo) to Google, and is an unofficial free endpoint that can be
  rate-limited. If it fails, the app shows a message instead of crashing.
  Not tested from the development workspace: click it once after deploying.
