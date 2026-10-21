"""
Framework-independent logic of the demo application.

Keeping this separate from the Streamlit UI (app/app.py) means it can be
unit-tested without a browser (tests/test_app_logic.py).
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch  # noqa: E402

import htr_common as hc  # noqa: E402
import page_recognition as pr  # noqa: E402
import translation  # noqa: E402


DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "final_model.pt")
MAX_UPLOAD_MEGAPIXELS = 40

# UNVALIDATED heuristic: images that are not much wider than tall are
# more likely to contain several lines (or not be cropped to the text).
MULTILINE_ASPECT_WARNING = 3.0


def model_path():
    return os.environ.get("HTR_MODEL_PATH", DEFAULT_MODEL_PATH)


@dataclass
class LoadedModel:
    model: object
    vocabulary: object
    config: dict
    provenance: dict
    path: str
    sha256: str


@dataclass
class RecognitionResult:
    text: str
    display_text: str
    confidence: float = None
    canvas: Image.Image = None
    warnings: list = field(default_factory=list)
    seconds: float = 0.0


def load_app_model(path=None, device="cpu"):
    """
    Load the deployment model and its provenance file.
    Raises FileNotFoundError with an actionable message if missing.
    """

    path = path or model_path()

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Create it with:  python src/export_final_model.py "
            "--model baseline"
        )

    model, vocabulary, config, _ = hc.load_model(path, device)

    provenance = {}
    json_path = os.path.splitext(path)[0] + ".json"

    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as file:
            provenance = json.load(file)

    return LoadedModel(
        model=model,
        vocabulary=vocabulary,
        config=config,
        provenance=provenance,
        path=path,
        sha256=hc.file_sha256(path),
    )


def open_user_image(file_like):
    """Open an uploaded file safely; raises ValueError with a clear message."""

    try:
        image = Image.open(file_like)
        image.load()
    except Exception as error:
        raise ValueError(
            "This file could not be read as an image. "
            "Please upload a PNG, JPG or WEBP picture."
        ) from error

    width, height = image.size

    if width < 1 or height < 1:
        raise ValueError("The image is empty.")

    if width * height > MAX_UPLOAD_MEGAPIXELS * 1_000_000:
        raise ValueError(
            f"The image is too large (limit {MAX_UPLOAD_MEGAPIXELS} "
            "megapixels). Please downscale it first."
        )

    return image


def image_warnings(image):
    """
    Cheap, explainable checks that tell the user when the input is
    outside what the model was trained on.
    """

    warnings = []
    width, height = image.size

    if min(width, height) < 32:
        warnings.append(
            "The image is very small; handwriting may be unreadable."
        )

    if width / height < MULTILINE_ASPECT_WARNING:
        warnings.append(
            "This image is not much wider than it is tall, so it may "
            "contain several lines of text. The original pipeline (A) "
            "feeds the whole image to a model built for ONE line, which "
            "shrinks the writing; choose pipeline D (default) to detect and "
            "recognise each line separately, or crop one line per image."
        )

    gray = np.array(hc.prepare_user_image(image), dtype=np.float32)

    if gray.std() < 20:
        warnings.append(
            "The image has low contrast. Try the 'Auto-contrast' option."
        )

    if gray.mean() < 100:
        warnings.append(
            "The image looks dark. The model expects dark ink on a "
            "light background."
        )

    return warnings


def canvas_to_image(canvas_tensor):
    """[1,H,W] float tensor in [0,1] -> PIL image (what the model sees)."""

    array = (canvas_tensor.squeeze(0).numpy() * 255).astype(np.uint8)

    return Image.fromarray(array, mode="L")


def recognize_image(loaded, image, auto_contrast=False, device="cpu"):
    began = time.perf_counter()

    warnings = image_warnings(image)

    prepared = hc.prepare_user_image(image, auto_contrast=auto_contrast)

    canvas = hc.preprocess_image(
        prepared,
        target_height=loaded.config["image_height"],
        target_width=loaded.config["image_width"],
    )

    with torch.no_grad():
        logits = loaded.model(canvas.unsqueeze(0).to(device))

    text, confidence = hc.greedy_decode_with_confidence(
        logits, loaded.vocabulary
    )[0]

    if not text:
        warnings.append(
            "The model produced no text for this image. Try a cleaner "
            "crop of a single line, or toggle 'Auto-contrast'."
        )

    return RecognitionResult(
        text=text,
        display_text=hc.display_text(text),
        confidence=confidence,
        canvas=canvas_to_image(canvas),
        warnings=warnings,
        seconds=time.perf_counter() - began,
    )


# ------------------------------------------------------------
# Photograph pipeline (preprocessing -> line detection -> recognizer)
# ------------------------------------------------------------

PIPELINE_CHOICES = {key: spec.label for key, spec in pr.EXPERIMENTS.items()}
DEFAULT_PIPELINE = pr.DEFAULT_PIPELINE

KIND_MESSAGES = {
    "single_line": "Detected: one line of handwriting (not split).",
    "multi_line": "Detected: several lines of handwriting; each line was "
                  "recognised separately and joined top to bottom.",
    "no_handwriting": "No handwriting was detected.",
    "not_handwriting": "This does not look like handwriting.",
    "legacy": "Original single-line pipeline (the whole image is one input).",
    "whole_image": "The whole image was sent to the model as ONE line.",
}


def recognize_photo(loaded, image, pipeline=None, force=False, rotation=None):
    """
    Photograph -> text with the staged pipeline (see page_recognition).
    `pipeline` is a key of page_recognition.EXPERIMENTS (default "D").
    Returns a page_recognition.PageResult; never raises for ordinary bad input.
    """

    spec = pr.EXPERIMENTS[pipeline or DEFAULT_PIPELINE]

    try:
        return pr.recognize_page(loaded, image, spec, force=force, rotation=rotation)
    except Exception as error:  # last-resort guard: degrade, do not crash the UI
        result = pr.recognize_legacy(loaded, image, pr.EXPERIMENTS["A"])
        result.fallback = True
        result.warnings.append(
            f"The staged pipeline failed ({type(error).__name__}: {error}); the original "
            "single-line pipeline was used on the whole image instead."
        )
        return result


def replace_spec(spec, auto_contrast):
    from dataclasses import replace

    return replace(spec, auto_contrast=bool(auto_contrast))


def page_display_text(result):
    return hc.display_text(result.text)


def translate_to_indonesian(text):
    """
    English -> Indonesian for the app's translate button. Returns
    (translated_text, error_message_or_None); never raises, so a service
    failure shows as a message instead of crashing the page.
    """

    try:
        return translation.translate_text(text), None
    except translation.TranslationError as error:
        return "", str(error)


def image_fingerprint(image_bytes):
    return hashlib.sha1(image_bytes).hexdigest()[:12]
