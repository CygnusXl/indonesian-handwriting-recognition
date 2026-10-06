"""
Indonesian handwriting recognition - demo application.

Run from the repository root:

    streamlit run app/app.py

Pipeline: uploaded image -> preprocessing (same function as training)
-> CNN-BiLSTM -> CTC greedy decoding -> recognised text.
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
    "Deep learning demo (CNN + BiLSTM + CTC) that converts an image of "
    "Indonesian handwriting into digital text."
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
        "Accuracy on real handwriting photos can be much lower. "
        "The model is designed for ONE line of text per image; "
        "multiline text was its main weakness."
    )

    st.caption(
        "Uploaded images are processed in memory for this session and are "
        "not stored by the app."
    )

    auto_contrast = st.checkbox(
        "Auto-contrast (not used in training)",
        value=False,
        help="Stretches image contrast before recognition. Useful for "
        "grey paper or dim photos. Not part of the trained pipeline.",
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
    st.info("Upload an image of one line of handwriting to begin.")
    st.stop()

try:
    image = logic.open_user_image(io.BytesIO(image_bytes))
except ValueError as error:
    st.error(str(error))
    st.stop()

fingerprint = logic.image_fingerprint(image_bytes)

left, right = st.columns(2)

with left:
    st.subheader("Input")
    st.image(image, width="stretch")

    run = st.button("Recognize text", type="primary")

if run:
    with st.spinner("Recognising..."):
        st.session_state["result"] = {
            "fingerprint": fingerprint,
            "auto_contrast": auto_contrast,
            "value": logic.recognize_image(
                loaded, image, auto_contrast=auto_contrast
            ),
        }

state = st.session_state.get("result")
result = (
    state["value"]
    if state and state["fingerprint"] == fingerprint
    and state["auto_contrast"] == auto_contrast
    else None
)

with right:
    st.subheader("Recognised text")

    if result is None:
        st.caption("Press **Recognize text**.")
    else:
        for message in result.warnings:
            st.warning(message)

        st.text_area(
            "Result (select to copy)",
            value=result.display_text,
            height=140,
            label_visibility="collapsed",
        )

        columns = st.columns(2)

        if result.confidence is not None:
            columns[0].metric(
                "Model confidence (heuristic)",
                f"{result.confidence * 100:.0f}%",
                help="Mean probability of the characters the model "
                "emitted. Not a calibrated probability of being correct; "
                "outputs can be wrong even at high confidence.",
            )

        columns[1].metric("Time", f"{result.seconds * 1000:.0f} ms")

        with st.expander("What the model sees (after preprocessing)"):
            st.image(result.canvas, width="stretch")
            st.caption(
                "The image is converted to grayscale, scaled to the "
                "model's height/width and placed on a white canvas. "
                "Tall (multiline) images are shrunk to fit."
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
                "True text (one line break per handwritten line)",
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
                    reference.replace("\n", "\\n"),
                    result.text,
                    usefulness,
                    result.seconds,
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
