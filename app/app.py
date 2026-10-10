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
# Theme & Custom CSS (Light / Dark mode support)
# ------------------------------------------------------------

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

with st.sidebar:
    st.subheader("🎨 Appearance")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"], key="theme_toggle")
    st.session_state["dark_mode"] = dark_mode

if dark_mode:
    theme_css = """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .custom-card {
        background-color: #161f30;
        border: 1px solid #28354d;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    .step-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        text-align: center;
        height: 100%;
    }
    .custom-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
        margin-right: 0.4rem;
    }
    .ground-truth-box {
        background-color: #111827;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        font-family: monospace;
        color: #34d399;
    }
    .tip-box {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        color: #fbbf24;
        font-size: 0.9rem;
    }
    .result-display {
        background-color: #0d131f;
        border: 1px solid #28354d;
        border-radius: 8px;
        padding: 1.2rem;
        font-size: 1.25rem;
        font-weight: 500;
        line-height: 1.6;
        color: #f8fafc;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        word-break: break-word;
    }
    </style>
    """
else:
    theme_css = """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .custom-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    .step-card {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        text-align: center;
        height: 100%;
    }
    .custom-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: rgba(79, 70, 229, 0.1);
        color: #4338ca;
        border: 1px solid rgba(79, 70, 229, 0.25);
        margin-right: 0.4rem;
    }
    .ground-truth-box {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        font-family: monospace;
        color: #065f46;
    }
    .tip-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.6rem 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        color: #92400e;
        font-size: 0.9rem;
    }
    .result-display {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1.2rem;
        font-size: 1.25rem;
        font-weight: 500;
        line-height: 1.6;
        color: #0f172a;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        word-break: break-word;
    }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)


# ------------------------------------------------------------
# Model Loading
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
# Sidebar Details
# ------------------------------------------------------------

with st.sidebar:
    st.subheader("⚙️ Settings")
    auto_contrast = st.checkbox(
        "Auto-contrast adjustment",
        value=False,
        help="Enhances contrast for faint pencil or low-light photos.",
    )

    st.divider()

    with st.expander("ℹ️ About the Model", expanded=False):
        st.write(
            f"**Architecture**: CNN + BiLSTM + CTC  \n"
            f"**Canvas**: {loaded.config['image_height']} × {loaded.config['image_width']} px  \n"
            f"**Parameters**: {provenance.get('parameters', 0):,}  \n"
            f"**Decoding**: Greedy CTC"
        )

        metrics = provenance.get("test_metrics")
        if metrics:
            st.markdown(f"**Test Set Performance ({metrics['samples']} synthetic samples):**")
            st.write(
                f"• Corpus CER: {metrics['corpus_cer'] * 100:.1f}%  \n"
                f"• Corpus WER: {metrics['corpus_wer'] * 100:.1f}%  \n"
                f"• Exact Match: {metrics['exact_match'] * 100:.1f}%"
            )

        st.caption(
            "Trained on synthetic handwriting (OpenHand-Synth). "
            "Optimized for one line of text per image."
        )


# ------------------------------------------------------------
# Main Header & Casual-Friendly Quick Guide
# ------------------------------------------------------------

st.title("✍️ Indonesian Handwriting Recognition")
st.markdown(
    "Convert photos or scans of Indonesian handwriting into editable digital text with deep learning."
)

# 3-step friendly guide for casual visitors
step_col1, step_col2, step_col3 = st.columns(3)
with step_col1:
    st.markdown(
        """
        <div class="step-card">
            <strong>1. Choose Image</strong><br>
            <span style="font-size:0.88rem; opacity:0.85;">Pick a test set sample or upload your own.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with step_col2:
    st.markdown(
        """
        <div class="step-card">
            <strong>2. Recognize</strong><br>
            <span style="font-size:0.88rem; opacity:0.85;">Click "Recognize text" to run the model.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with step_col3:
    st.markdown(
        """
        <div class="step-card">
            <strong>3. Copy Text</strong><br>
            <span style="font-size:0.88rem; opacity:0.85;">View, edit, and copy the recognized text.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# ------------------------------------------------------------
# Input Selection: Upload an image OR Test Set Examples
# ------------------------------------------------------------

examples = load_examples()
reference_prefill = ""
image_bytes = None
source = "upload"

if examples:
    choice = st.radio(
        "Input",
        ["Upload an image", "Use an example"],
        horizontal=True,
        help="Select 'Use an example' to test instantly without uploading files.",
    )
else:
    choice = "Upload an image"

if choice == "Upload an image":
    st.markdown(
        """
        <div class="tip-box">
            💡 <strong>Tip for best results:</strong> The model is trained for <strong>one line of text</strong> per image. Crop tightly around the line on a plain background for highest accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload a photo or scan of handwriting",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded is not None:
        image_bytes = uploaded.getvalue()

else:
    st.markdown("### 📚 Test Set Examples (OpenHand-Synth)")
    st.caption("Click any sample below to see how the model reads it:")

    # Casual-friendly readable labels for the example options
    example_options = []
    for i, e in enumerate(examples):
        ref_preview = e["reference"]
        if len(ref_preview) > 36:
            ref_preview = ref_preview[:33] + "..."
        example_options.append(f"Sample {i + 1}: \"{ref_preview}\"")

    picked = st.selectbox(
        "Choose a test set sample",
        example_options,
        index=0,
    )
    picked_index = example_options.index(picked)
    entry = examples[picked_index]

    example_file_path = os.path.join(EXAMPLES_DIR, entry["file"])
    with open(example_file_path, "rb") as file:
        image_bytes = file.read()

    reference_prefill = entry["reference"]
    source = "example"


# If no image is provided yet, show prompt and stop cleanly
if image_bytes is None:
    st.info("Upload an image of one line of handwriting to begin.")
    st.stop()


# ------------------------------------------------------------
# Image Processing & Recognition
# ------------------------------------------------------------

try:
    image = logic.open_user_image(io.BytesIO(image_bytes))
except ValueError as error:
    st.error(str(error))
    st.stop()

fingerprint = logic.image_fingerprint(image_bytes)

left, right = st.columns([1, 1], gap="medium")

with left:
    st.subheader("🖼️ Handwriting Image")
    st.image(image, use_container_width=True)

    if source == "example" and reference_prefill:
        st.markdown(
            f"""
            <div class="ground-truth-box">
                <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; font-weight:700;">Ground Truth (Expected Text):</span><br>
                {reference_prefill}
            </div>
            """,
            unsafe_allow_html=True,
        )

    run = st.button("✨ Recognize text", type="primary", use_container_width=True)

if run:
    with st.spinner("Recognising handwriting..."):
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
    st.subheader("📝 Recognised Text")

    if result is None:
        st.caption("Press **✨ Recognize text** on the left to start.")
    else:
        for message in result.warnings:
            st.warning(f"⚠️ {message}")

        # Big readable result box
        display_str = result.display_text if result.display_text.strip() else "(No text detected)"
        st.markdown(
            f"""
            <div class="result-display">
                {display_str}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            "Editable text (select to copy)",
            value=result.display_text,
            height=100,
            label_visibility="visible",
        )

        # Performance & Confidence metrics
        metric_col1, metric_col2 = st.columns(2)

        if result.confidence is not None:
            metric_col1.metric(
                "Confidence Score",
                f"{result.confidence * 100:.0f}%",
                help="Mean confidence of character emissions. Note that confidence can be high even with errors.",
            )

        metric_col2.metric("Processing Time", f"{result.seconds * 1000:.0f} ms")

        # Visual inspect expander
        with st.expander("🔍 What the model sees (Preprocessed 128×1024 Canvas)"):
            st.image(result.canvas, use_container_width=True)
            st.caption(
                "The image is converted to grayscale, scaled to height 128, and padded onto a 128×1024 white canvas."
            )


# ------------------------------------------------------------
# Optional Practical-Evaluation Logging (Research / User Study)
# ------------------------------------------------------------

if result is not None:
    with st.expander("🔬 Practical Evaluation: Record This Result (Optional)"):
        st.caption(
            "Enter what the handwriting REALLY says to calculate character and word error rates. "
            + (
                "On this hosted version, your results can be downloaded as a CSV below."
                if HOSTED
                else "Saved locally to evaluation/results/app_sessions.csv."
            )
        )

        with st.form("evaluation_form", clear_on_submit=False):
            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                participant = st.text_input("Participant ID", value="P01")
                line_type = st.selectbox(
                    "Line type", ["single-line", "multiline"]
                )
                handwriting = st.selectbox(
                    "Handwriting", ["neat", "average", "messy"]
                )
            with eval_col2:
                capture = st.selectbox(
                    "Capture condition",
                    [
                        "scan",
                        "good photo",
                        "dim or shadowed photo",
                        "blurry photo",
                        "synthetic example",
                    ],
                )
                usefulness = st.slider(
                    "How useful was the result? (1 = useless, 5 = very useful)",
                    1,
                    5,
                    3,
                )

            reference = st.text_area(
                "True text (one line break per handwritten line)",
                value=reference_prefill,
            )
            submitted = st.form_submit_button("Save Evaluation")

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
                        f"Recorded in this session. Sample CER: {row['cer'] * 100:.1f}%"
                    )
                else:
                    elog.append_session(row)
                    st.success(
                        f"Saved. Sample CER: {row['cer'] * 100:.1f}%"
                    )

        if HOSTED and st.session_state.get("eval_rows"):
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=elog.COLUMNS)
            writer.writeheader()
            writer.writerows(st.session_state["eval_rows"])

            st.download_button(
                f"📥 Download my {len(st.session_state['eval_rows'])} recorded result(s) as CSV",
                buffer.getvalue(),
                file_name="my_evaluation_results.csv",
                mime="text/csv",
            )
