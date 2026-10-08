"""
Indonesian handwriting recognition - demo application.

Run from the repository root:

    streamlit run app/app.py

Pipeline (default "D"): photograph -> orientation / perspective / deskew /
illumination / contrast / notebook-ruling suppression -> handwriting-line
detection -> one crop per line -> the (unchanged) CNN-BiLSTM line recognizer
with CTC greedy decoding -> lines joined in reading order.
"A" is the original single-line pipeline, kept for comparison.
"""

import csv
import io
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app_logic as logic  # noqa: E402
import evaluation_log as elog  # noqa: E402

# Set HTR_HOSTED=1 on a public deployment: results are then NOT written to
# the server's disk; the visitor downloads their own evaluation CSV instead.
HOSTED = os.environ.get("HTR_HOSTED", "0") == "1"

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

st.set_page_config(
    page_title="Indonesian Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model...")
def get_model(path):
    return logic.load_app_model(path)


def load_examples():
    path = os.path.join(EXAMPLES_DIR, "examples.csv")

    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("✍️ Indonesian Handwriting Recognition")
st.caption(
    "Deep learning demo (CNN + BiLSTM + CTC line recogniser) that converts "
    "a photo of handwriting into digital text. Photos are preprocessed and "
    "split into lines before recognition."
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

try:
    loaded = get_model(logic.model_path())
except FileNotFoundError as error:
    st.error("The trained model file is missing.")
    st.code(str(error))
    st.stop()
except Exception as error:  # corrupt / incompatible checkpoint
    st.error(f"The model could not be loaded: {error}")
    st.stop()

provenance = loaded.provenance

with st.sidebar:
    st.header("Model")
    st.write(
        f"**{provenance.get('label', 'Trained model')}**  \n"
        f"Input: {loaded.config['image_height']} x "
        f"{loaded.config['image_width']} px  \n"
        f"Parameters: {provenance.get('parameters', 0):,}  \n"
        f"Decoding: CTC greedy"
    )

    metrics = provenance.get("test_metrics")

    if metrics:
        st.markdown("**Measured on the OpenHand-Synth test set "
                    f"({metrics['samples']} synthetic samples):**")
        st.write(
            f"Corpus CER {metrics['corpus_cer'] * 100:.1f}%  \n"
            f"Corpus WER {metrics['corpus_wer'] * 100:.1f}%  \n"
            f"Exact match {metrics['exact_match'] * 100:.1f}%"
        )

    st.warning(
        "Trained only on SYNTHETIC handwriting (OpenHand-Synth). "
        "Accuracy on real handwriting photos can be much lower and has "
        "NOT been measured here. The model itself reads ONE line at a "
        "time; for photographed pages the app first finds and crops the "
        "lines."
    )

    st.caption(
        "Uploaded images are processed in memory for this session and are "
        "not stored by the app."
    )

    st.header("Pipeline")

    pipeline = st.selectbox(
        "Processing pipeline",
        list(logic.PIPELINE_CHOICES),
        index=list(logic.PIPELINE_CHOICES).index(logic.DEFAULT_PIPELINE),
        format_func=lambda key: logic.PIPELINE_CHOICES[key],
        help="D is the default for photographs of pages. A is the original "
        "single-line pipeline (use it for a tight crop of one line). "
        "Which pipeline reads better on YOUR photos has to be measured: "
        "see evaluation/real_world/README.md.",
    )

    rotation_label = st.selectbox(
        "Page rotation",
        ["Auto", "0 (as is)", "90 left", "180", "90 right"],
        help="Auto corrects sideways pages and leaves upright ones alone. "
        "If automatic orientation guesses wrong, choose it manually.",
    )
    rotation = {"Auto": None, "0 (as is)": 0, "90 left": 90,
                "180": 180, "90 right": 270}[rotation_label]

    force = st.checkbox(
        "Recognize anyway if no handwriting is found",
        value=False,
        help="By default recognition is skipped when the image looks blank "
        "or not like handwriting.",
    )

    auto_contrast = st.checkbox(
        "Auto-contrast (pipeline A only)",
        value=False,
        disabled=(pipeline != "A"),
        help="Stretches contrast for the original pipeline A. Pipelines "
        "B-D already normalise illumination and contrast.",
    )

# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

examples = load_examples()
reference_prefill = ""
image_bytes = None
source = "upload"

if examples:
    choice = st.radio(
        "Input", ["Upload an image", "Use an example"], horizontal=True
    )
else:
    choice = "Upload an image"

if choice == "Upload an image":
    uploaded = st.file_uploader(
        "Upload a photo or scan of handwriting",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
else:
    labels = [f"{e['file']}" for e in examples]
    picked = st.selectbox("Example", labels)
    entry = examples[labels.index(picked)]

    with open(os.path.join(EXAMPLES_DIR, entry["file"]), "rb") as file:
        image_bytes = file.read()

    reference_prefill = entry["reference"]
    source = "example"

if image_bytes is None:
    st.info("Upload an image of handwriting (one line, or a photographed notebook page) to begin.")
    st.stop()

try:
    image = logic.open_user_image(io.BytesIO(image_bytes))
except ValueError as error:
    st.error(str(error))
    st.stop()

fingerprint = logic.image_fingerprint(image_bytes)

options_key = (pipeline, rotation, force, auto_contrast if pipeline == "A" else False)

st.subheader("1. Input image")
st.image(image, width="stretch")

run = st.button("Recognize text", type="primary")

if run:
    with st.spinner("Preprocessing, finding lines and recognising..."):
        if pipeline == "A":
            value = logic.pr.recognize_legacy(
                loaded, image,
                logic.replace_spec(logic.pr.EXPERIMENTS["A"], auto_contrast),
            )
            value.warnings = logic.image_warnings(image) + value.warnings
        else:
            value = logic.recognize_photo(
                loaded, image, pipeline=pipeline, force=force, rotation=rotation
            )

        st.session_state["result"] = {
            "fingerprint": fingerprint,
            "options": options_key,
            "value": value,
        }

state = st.session_state.get("result")
result = (
    state["value"]
    if state and state["fingerprint"] == fingerprint
    and state["options"] == options_key
    else None
)

if result is None:
    st.caption("Press **Recognize text**.")
else:
    spec = result.spec

    st.caption(f"Pipeline: {spec.describe()}")

    message = logic.KIND_MESSAGES.get(result.kind)

    if message:
        (st.error if result.kind in ("no_handwriting", "not_handwriting") else st.info)(
            message + (f" {result.reason}" if result.reason and result.kind
                       in ("no_handwriting", "not_handwriting") else "")
        )

    for note in result.warnings:
        st.warning(note)

    # ---- 2. preprocessing ------------------------------------------------
    if spec.legacy:
        st.subheader("2. Preprocessing")
        st.caption("Original pipeline: grayscale + resize to the model input. "
                   "No page-level preprocessing and no line detection.")
    else:
        st.subheader("2. Preprocessed image")
        tabs = st.tabs(["Original", "Normalised", "Processed (used for detection)"])

        with tabs[0]:
            st.image(result.images["original"], width="stretch")
        with tabs[1]:
            st.image(result.images["normalized"], width="stretch")
            st.caption("After orientation / perspective / deskew, illumination "
                       "flattening, denoising and contrast normalisation.")
        with tabs[2]:
            st.image(result.images["processed"], width="stretch")
            st.caption(
                "Notebook-ruling suppression applied." if spec.preprocess.suppress_ruling
                else "No ruling suppression in this pipeline (same as Normalised)."
            )

        for note in result.notes:
            st.caption("• " + note)

    # ---- 3. detected lines ----------------------------------------------
    if "overlay" in result.images:
        st.subheader("3. Detected handwriting lines")
        st.image(result.images["overlay"], width="stretch")
        st.caption(f"{result.n_lines} line(s) found, numbered top to bottom.")

    # ---- 4. recognised text ------------------------------------------------
    st.subheader("4. Recognised text")

    if result.lines:
        st.text_area(
            "Result (select to copy)",
            value=logic.page_display_text(result),
            height=max(120, 40 * (result.n_lines + 1)),
            label_visibility="collapsed",
        )

        columns = st.columns(3)

        if result.confidence is not None:
            columns[0].metric(
                "Model confidence (heuristic)",
                f"{result.confidence * 100:.0f}%",
                help="Mean probability of the characters the model emitted, "
                "weighted by text length. A heuristic, NOT a calibrated "
                "probability of being correct; outputs can be wrong even at "
                "high confidence.",
            )

        columns[1].metric("Lines", f"{result.n_lines}")
        columns[2].metric("Time", f"{result.timings.get('total_s', 0) * 1000:.0f} ms")

        if result.n_lines > 1:
            with st.expander("Per-line results"):
                for line in result.lines:
                    confidence = (f"{line.confidence * 100:.0f}% (heuristic)"
                                  if line.confidence is not None else "n/a")
                    st.markdown(f"**Line {line.index + 1}** - confidence {confidence}")
                    st.code(line.text or "(empty)", language=None)
    else:
        st.caption("No text was produced.")

    # ---- debugging ---------------------------------------------------------
    with st.expander("What the model sees"):
        st.caption(
            "Each image below is exactly what is fed to the recognizer: a crop "
            "of one line, converted to grayscale, scaled to the model height and "
            "placed on the white model canvas. If this does not look like a "
            "clean line of dark writing, the problem is in preprocessing or "
            "line detection, not in the model."
        )

        for line in result.lines:
            for number, chunk in enumerate(line.chunks, 1):
                if chunk.canvas is None:
                    continue

                label = f"Line {line.index + 1}" + (
                    f" (part {number})" if len(line.chunks) > 1 else "")
                detail = []

                if chunk.fit_scale is not None and chunk.fit_scale < 0.99:
                    detail.append(f"shrunk to {chunk.fit_scale * 100:.0f}% to fit the width")
                if chunk.ink_height_fraction is not None:
                    detail.append(f"writing is {chunk.ink_height_fraction * 100:.0f}% "
                                  "of the model input height")

                st.markdown(f"**{label}**" + (" - " + "; ".join(detail) if detail else ""))
                st.image(chunk.canvas, width="stretch")
                st.caption(f"Model output: {chunk.text or '(empty)'}")

        t = result.timings
        st.caption(
            f"Timing: preprocessing {t.get('preprocess_s', 0) * 1000:.0f} ms, "
            f"line detection {t.get('segment_s', 0) * 1000:.0f} ms, "
            f"recognition {t.get('infer_s', 0) * 1000:.0f} ms."
        )

# ------------------------------------------------------------
# Optional practical-evaluation logging
# ------------------------------------------------------------

if result is not None:
    with st.expander("Practical evaluation: record this result (optional)"):
        st.caption(
            "Type what the handwriting REALLY says to measure the error "
            "rate. Only text and settings are recorded, never the image. "
            + (
                "On this hosted version nothing is saved on the server: "
                "download your results as a CSV below."
                if HOSTED
                else "Saved locally to evaluation/results/app_sessions.csv."
            )
        )

        with st.form("evaluation_form", clear_on_submit=False):
            participant = st.text_input("Participant ID", value="P01")
            reference = st.text_area(
                "True text (one line break per handwritten line; "
                "line breaks are scored as spaces)",
                value=reference_prefill,
            )
            line_type = st.selectbox(
                "Line type", ["single-line", "multiline"]
            )
            handwriting = st.selectbox(
                "Handwriting", ["neat", "average", "messy"]
            )
            capture = st.selectbox(
                "Capture", ["scan", "good photo", "dim or shadowed photo",
                            "blurry photo", "synthetic example"]
            )
            usefulness = st.slider(
                "How useful was the result? (1 = useless, 5 = very useful)",
                1, 5, 3,
            )
            submitted = st.form_submit_button("Save evaluation")

        if submitted:
            if not reference.strip():
                st.error("Please enter the true text first.")
            else:
                row = elog.build_row(
                    participant,
                    fingerprint,
                    line_type,
                    handwriting,
                    capture,
                    auto_contrast,
                    " ".join(reference.split()),
                    result.text_oneline,
                    usefulness,
                    result.timings.get("total_s", 0.0),
                    loaded.sha256,
                )
                if HOSTED:
                    st.session_state.setdefault("eval_rows", []).append(row)
                    st.success(
                        f"Recorded in this session. CER for this sample: "
                        f"{row['cer'] * 100:.1f}%"
                    )
                else:
                    elog.append_session(row)
                    st.success(
                        f"Saved. CER for this sample: {row['cer'] * 100:.1f}%"
                    )

        if HOSTED and st.session_state.get("eval_rows"):
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=elog.COLUMNS)
            writer.writeheader()
            writer.writerows(st.session_state["eval_rows"])

            st.download_button(
                f"Download my {len(st.session_state['eval_rows'])} "
                "recorded result(s) as CSV",
                buffer.getvalue(),
                file_name="my_evaluation_results.csv",
                mime="text/csv",
            )
