"""
Indonesian handwriting recognition - web application.

Run from the repository root:

    streamlit run app/app.py

Pipeline (default "D"): photograph -> orientation / perspective / deskew /
illumination / contrast / notebook-ruling suppression -> handwriting-line
detection -> one crop per line -> the (unchanged) CNN-BiLSTM line recognizer
with CTC greedy decoding -> lines joined in reading order.
"A" is the original single-line pipeline, kept for comparison.

The look (ruled-paper theme) lives in app/style.css and app/ui_components.py.
"""

import csv
import io
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app_logic as logic  # noqa: E402
import evaluation_log as elog  # noqa: E402
import ui_components as ui  # noqa: E402

# Set HTR_HOSTED=1 on a public deployment: results are then NOT written to
# the server's disk; the visitor downloads their own evaluation CSV instead.
HOSTED = os.environ.get("HTR_HOSTED", "0") == "1"

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

st.set_page_config(
    page_title="Indonesian Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(ui.load_css(), unsafe_allow_html=True)

# Light / dark appearance (dark is the default, as on main).
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

with st.sidebar:
    st.subheader("Appearance")
    dark_mode = st.toggle("Dark mode", key="dark_mode")

if dark_mode:
    st.markdown(ui.load_dark_css(), unsafe_allow_html=True)


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

# ------------------------------------------------------------
# Hero + input
# ------------------------------------------------------------

st.markdown(ui.hero(provenance.get("label", "Trained model")), unsafe_allow_html=True)

examples = load_examples()
reference_prefill = ""
image_bytes = None

if examples:
    choice = st.radio("Input", ["Upload an image", "Use an example"], horizontal=True)
else:
    choice = "Upload an image"

if choice == "Upload an image":
    uploaded = st.file_uploader(
        "Upload a photo or scan of handwriting",
        type=["png", "jpg", "jpeg", "webp"],
        help="One line or a whole notebook page. The photo is processed in memory "
        "for this session and is not stored.",
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()
else:
    st.markdown("#### Test set examples (OpenHand-Synth)")
    st.caption("Synthetic handwriting images from the dataset test split, not photographs. "
               "Pick one to see how the model reads it.")

    labels = [ui.example_label(i, e["reference"]) for i, e in enumerate(examples)]
    picked = st.selectbox("Choose a test set sample", labels)
    entry = examples[labels.index(picked)]

    with open(os.path.join(EXAMPLES_DIR, entry["file"]), "rb") as file:
        image_bytes = file.read()

    reference_prefill = entry["reference"]
    st.markdown(ui.ground_truth(reference_prefill), unsafe_allow_html=True)

with st.expander("Settings"):
    columns = st.columns(2)

    pipeline = columns[0].selectbox(
        "Processing pipeline",
        list(logic.PIPELINE_CHOICES),
        index=list(logic.PIPELINE_CHOICES).index(logic.DEFAULT_PIPELINE),
        format_func=lambda key: logic.PIPELINE_CHOICES[key],
        help="D is the default for photos of pages. A is the original single-line "
        "pipeline (the whole image is one input); use it for a tight crop of one "
        "line. Which pipeline reads your photos better has to be measured: see "
        "evaluation/real_world/README.md.",
    )

    rotation_label = columns[1].selectbox(
        "Page rotation",
        ["Auto", "0 (as is)", "90 left", "180", "90 right"],
        help="Auto turns sideways pages upright and leaves upright pages alone. "
        "If the automatic guess is wrong, choose the rotation yourself.",
    )
    rotation = {"Auto": None, "0 (as is)": 0, "90 left": 90,
                "180": 180, "90 right": 270}[rotation_label]

    force = columns[0].checkbox(
        "Recognize anyway if no handwriting is found",
        value=False,
        help="By default recognition is skipped when the image looks blank "
        "or not like handwriting.",
    )

    auto_contrast = columns[1].checkbox(
        "Auto-contrast (pipeline A only)",
        value=False,
        disabled=(pipeline != "A"),
        help="Stretches contrast for the original pipeline A. Pipelines B-D "
        "already normalise illumination and contrast.",
    )

if image_bytes is None:
    st.markdown(ui.empty_state(), unsafe_allow_html=True)
    image = None
else:
    try:
        image = logic.open_user_image(io.BytesIO(image_bytes))
    except ValueError as error:
        st.markdown(ui.status_banner("stop", str(error)), unsafe_allow_html=True)
        image = None

result = None

if image is not None:
    fingerprint = logic.image_fingerprint(image_bytes)
    options_key = (pipeline, rotation, force, auto_contrast if pipeline == "A" else False)

    state = st.session_state.get("result")

    if not (state and state["fingerprint"] == fingerprint and state["options"] == options_key):
        with st.spinner("Cleaning up the photo, finding lines and reading them..."):
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

        state = {"fingerprint": fingerprint, "options": options_key, "value": value}
        st.session_state["result"] = state

    result = state["value"]

# ------------------------------------------------------------
# Results: photo -> cleaned up -> lines found -> text
# ------------------------------------------------------------

if result is not None:
    spec = result.spec

    message = logic.KIND_MESSAGES.get(result.kind)

    if message:
        stopped = result.kind in ("no_handwriting", "not_handwriting")
        extra = f" {result.reason}" if stopped and result.reason else ""
        st.markdown(ui.status_banner("stop" if stopped else "info", message + extra),
                    unsafe_allow_html=True)

    for note in result.warnings:
        st.markdown(ui.status_banner("warn", note), unsafe_allow_html=True)

    col_photo, col_clean, col_lines = st.columns(3)

    with col_photo:
        st.markdown(ui.step_header(1, "Your photo", "As uploaded."), unsafe_allow_html=True)
        st.image(image, width="stretch")

    with col_clean:
        if spec.legacy:
            st.markdown(ui.step_header(
                2, "Cleaned up", "Pipeline A only converts to grayscale and resizes."),
                unsafe_allow_html=True)
        else:
            st.markdown(ui.step_header(
                2, "Cleaned up",
                "Straightened, evenly lit, contrast normalised"
                + (", ruled lines removed." if spec.preprocess.suppress_ruling else ".")),
                unsafe_allow_html=True)

            tabs = st.tabs(["Processed", "Normalised"])

            with tabs[0]:
                st.image(result.images["processed"], width="stretch")
            with tabs[1]:
                st.image(result.images["normalized"], width="stretch")

            for note in result.notes:
                st.caption(note)

    with col_lines:
        if "overlay" in result.images:
            st.markdown(ui.step_header(
                3, "Lines found", f"{result.n_lines} line(s), numbered top to bottom."),
                unsafe_allow_html=True)
            st.image(result.images["overlay"], width="stretch")
        else:
            st.markdown(ui.step_header(
                3, "Lines found",
                "Not used in this pipeline." if not result.lines or spec.legacy
                else "No lines were detected."),
                unsafe_allow_html=True)

    st.markdown(ui.step_header(4, "Recognised text"), unsafe_allow_html=True)

    if result.lines:
        st.markdown(
            ui.paper_text((l.index, logic.hc.display_text(l.text), l.confidence)
                          for l in result.lines),
            unsafe_allow_html=True,
        )

        metrics = st.columns(3)

        if result.confidence is not None:
            metrics[0].metric(
                "Model confidence (heuristic)",
                f"{result.confidence * 100:.0f}%",
                help="Mean probability of the characters the model emitted, weighted "
                "by text length. A heuristic, NOT a calibrated probability of being "
                "correct; outputs can be wrong even at high confidence.",
            )

        metrics[1].metric("Lines", f"{result.n_lines}")
        metrics[2].metric("Time", f"{result.timings.get('total_s', 0) * 1000:.0f} ms")

        text_out = logic.page_display_text(result)

        actions = st.columns([1, 1, 3])
        actions[0].download_button(
            "Download .txt", text_out, file_name="recognised_text.txt", mime="text/plain")

        with st.expander("Copy as plain text"):
            st.code(text_out, language=None)
    else:
        st.markdown(ui.paper_empty("Nothing was read from this image. See the notes above."),
                    unsafe_allow_html=True)

    with st.expander("What the model sees"):
        st.caption(
            "Each image is exactly what is fed to the recognizer: a crop of one line, "
            "converted to grayscale, scaled to the model height and placed on a white "
            "canvas. If it does not look like a clean line of dark writing, the problem "
            "is in preprocessing or line detection, not in the model."
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
            f"Pipeline: {spec.describe()}. Timing: preprocessing "
            f"{t.get('preprocess_s', 0) * 1000:.0f} ms, line detection "
            f"{t.get('segment_s', 0) * 1000:.0f} ms, recognition "
            f"{t.get('infer_s', 0) * 1000:.0f} ms."
        )

# ------------------------------------------------------------
# Optional practical-evaluation logging
# ------------------------------------------------------------

if result is not None:
    with st.expander("Help evaluate: record this result (optional)"):
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

# ------------------------------------------------------------
# About
# ------------------------------------------------------------

st.markdown("## About this model")
st.markdown(
    ui.model_card(
        provenance.get("label", "Trained model"),
        provenance.get("parameters", 0),
        f"{loaded.config['image_height']} x {loaded.config['image_width']} px",
        provenance.get("test_metrics"),
    ),
    unsafe_allow_html=True,
)

about_left, about_right = st.columns(2)

with about_left:
    st.markdown("#### How it reads a page")
    st.markdown(
        "1. Straighten the photo and even out the lighting.\n"
        "2. Remove notebook ruling (pipeline D).\n"
        "3. Find each line of writing and crop it.\n"
        "4. Read each crop with the line recognizer.\n"
        "5. Join the lines in reading order."
    )

with about_right:
    st.markdown("#### What to keep in mind")
    st.markdown(
        "- The model was trained only on synthetic handwriting (OpenHand-Synth) and "
        "reads one line at a time.\n"
        "- The test-set numbers above are measured on synthetic data, not on real photos.\n"
        "- Confidence is a heuristic, not a probability of being correct.\n"
        "- Underlines and strike-throughs may be removed with the ruling, so "
        "crossed-out words can be read."
    )

st.markdown(
    '<div class="site-foot">Deep Learning (COMP6826001) final project, BINUS University. '
    "Photos are processed in memory for your session and are not stored.</div>",
    unsafe_allow_html=True,
)
