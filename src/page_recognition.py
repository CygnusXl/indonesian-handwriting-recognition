"""
Page recognition: photograph -> text, using the EXISTING line recognizer.

The CNN-BiLSTM-CTC model is unchanged. It still receives exactly what it was
trained on: one line of handwriting, scaled to 128 px high (max 1024 px wide)
by `htr_common.preprocess_image`. What this module adds is everything that has
to happen BEFORE the model for a photographed page:

    photo -> document_pipeline (orientation, perspective, deskew,
    illumination, contrast, optional ruling suppression)
    -> line detection -> per-line crops (long lines split at word gaps so
    they fit the canvas) -> existing model on every crop -> text joined in
    reading order

Four pipelines are defined so they can be compared on real photographs
(`src/evaluate_pipeline.py`):

    A  original single-line pipeline applied to the whole image
    B  improved preprocessing, whole page still sent as ONE image
    C  B + line segmentation (+ chunking of long lines)
    D  C + notebook-ruling suppression

Nothing here claims any pipeline is better; that must be measured.
"""

import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from PIL import Image

import document_pipeline as dp
import htr_common as hc


# ============================================================
# PIPELINE SPECIFICATIONS
# ============================================================

@dataclass(frozen=True)
class PipelineSpec:
    key: str
    label: str
    legacy: bool = False             # exact original path (A)
    segment: bool = False            # detect lines and crop them
    preprocess: dp.PreprocessConfig = dp.PreprocessConfig()
    segmentation: dp.SegmentationConfig = dp.SegmentationConfig()
    auto_contrast: bool = False      # legacy path only (old app option)

    def describe(self):
        if self.legacy:
            return "original: grayscale + resize" + (
                " + autocontrast" if self.auto_contrast else "")

        parts = [self.preprocess.describe()]
        parts.append("line-segmentation" if self.segment else "whole-page-as-one-image")

        return "; ".join(parts)


_IMPROVED = dp.PreprocessConfig(name="improved")

EXPERIMENTS = {
    "A": PipelineSpec("A", "A: original preprocessing", legacy=True),
    "B": PipelineSpec("B", "B: improved preprocessing (no line detection)",
                      preprocess=_IMPROVED),
    "C": PipelineSpec("C", "C: improved preprocessing + line segmentation",
                      segment=True, preprocess=_IMPROVED),
    "D": PipelineSpec("D", "D: improved preprocessing + line segmentation + ruling suppression",
                      segment=True,
                      preprocess=replace(_IMPROVED, name="improved+ruling",
                                         suppress_ruling=True)),
}

DEFAULT_PIPELINE = "D"


# ============================================================
# RESULTS
# ============================================================

@dataclass
class ChunkResult:
    text: str
    confidence: float = None
    crop: Image.Image = None       # the gray crop (before model scaling)
    canvas: Image.Image = None     # exactly what the model receives (H x W)
    fit_scale: float = 1.0         # < 1: width-limited, text shrunk below the trained scale
    ink_height_fraction: float = None  # line ink height / model canvas height


@dataclass
class LineResult:
    index: int
    bbox: tuple = None             # (x0, y0, x1, y1) in processed-page coordinates
    text: str = ""
    confidence: float = None
    chunks: list = field(default_factory=list)


@dataclass
class PageResult:
    spec: PipelineSpec
    kind: str = ""                 # single_line | multi_line | no_handwriting | not_handwriting | legacy | whole_image
    reason: str = ""
    lines: list = field(default_factory=list)
    text: str = ""
    confidence: float = None       # HEURISTIC, not a calibrated probability
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    images: dict = field(default_factory=dict)    # PIL images for display/debug
    timings: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    fallback: bool = False

    @property
    def n_lines(self):
        return len(self.lines)

    @property
    def n_chunks(self):
        return sum(len(line.chunks) for line in self.lines)

    @property
    def text_oneline(self):
        # the dataset writes line breaks as a literal backslash-n; both forms
        # are scored as a space (whitespace-normalised, see evaluation docs)
        return " ".join(self.text.replace("\\n", " ").split())


# ============================================================
# MODEL INFERENCE ON CANVASES
# ============================================================

def _model_device(loaded):
    try:
        return next(loaded.model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_model(loaded, canvases, batch_size=8):
    """
    canvases: list of [1,H,W] tensors (output of hc.preprocess_image).
    Returns ([(text, heuristic_confidence)], seconds spent in the model).
    """

    device = _model_device(loaded)
    outputs, seconds = [], 0.0

    for start in range(0, len(canvases), batch_size):
        batch = torch.stack(canvases[start:start + batch_size]).to(device)

        _sync(device)
        began = time.perf_counter()

        with torch.no_grad():
            logits = loaded.model(batch)

        _sync(device)
        seconds += time.perf_counter() - began

        outputs.extend(hc.greedy_decode_with_confidence(logits, loaded.vocabulary))

    return outputs, seconds


def _canvas_image(canvas_tensor):
    array = (canvas_tensor.squeeze(0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def _weighted_confidence(items):
    """Character-count weighted mean of heuristic confidences (None if empty)."""

    total, weight = 0.0, 0

    for text, confidence in items:
        if confidence is not None and text:
            total += confidence * len(text)
            weight += len(text)

    return total / weight if weight else None


# ============================================================
# A: ORIGINAL PATH
# ============================================================

def recognize_legacy(loaded, image, spec):
    config = loaded.config
    began = time.perf_counter()

    prepared = hc.prepare_user_image(image, auto_contrast=spec.auto_contrast)
    tensor = hc.preprocess_image(prepared, config["image_height"], config["image_width"])
    prep_s = time.perf_counter() - began

    outputs, infer_s = run_model(loaded, [tensor])
    text, confidence = outputs[0]

    original_h, original_w = prepared.size[1], prepared.size[0]
    scale = min(config["image_height"] / original_h, config["image_width"] / original_w)

    chunk = ChunkResult(text=text, confidence=confidence, crop=prepared,
                        canvas=_canvas_image(tensor),
                        fit_scale=min(1.0, scale * original_h / config["image_height"]),
                        ink_height_fraction=None)

    line = LineResult(0, None, text, confidence, [chunk])

    result = PageResult(spec=spec, kind="legacy", lines=[line], text=text,
                        confidence=confidence)
    result.images = {"original": prepared, "processed": prepared}
    result.timings = {"preprocess_s": prep_s, "segment_s": 0.0, "infer_s": infer_s,
                      "total_s": time.perf_counter() - began}
    result.info = {"shrink_scale": scale}

    return result


# ============================================================
# B / C / D: DOCUMENT PIPELINE
# ============================================================

def _to_pil(array):
    return Image.fromarray(np.ascontiguousarray(array))


def _quality_warnings(info):
    quality = info.get("quality", {})
    warnings = []

    if min(quality.get("width", 9999), quality.get("height", 9999)) < 400:
        warnings.append("The photo has low resolution; small handwriting may not be legible.")
    if quality.get("ink_contrast", 255) < 40:
        warnings.append("Very low contrast: ink and paper are hard to tell apart. "
                        "Contrast was normalised, but recognition may still fail.")
    if quality.get("mean", 255) < 90:
        warnings.append("The photo is very dark. Try better lighting.")
    if quality.get("sharpness", 1e9) < 15:
        warnings.append("The photo looks blurry (heuristic check). "
                        "Hold the camera steady and refocus.")

    return warnings


def _build_line_inputs(stages, detection, spec, model_config):
    """
    Crop every detected line (splitting over-long lines at word gaps) and
    turn the crops into model canvases. Returns (specs, canvases) where
    specs[i] = (line_index, crop_array, fit_scale, ink_fraction).
    """

    height, width = model_config["image_height"], model_config["image_width"]
    max_aspect = width / height

    reference = float(np.median([line.height for line in detection.lines]))
    entries = []

    for line in detection.lines:
        crop, origin = dp.crop_line(stages.processed, line, reference, spec.segmentation)
        placed = dp.line_mask_in_crop(line, crop.shape, origin)
        margin_x = int(round(spec.segmentation.margin_x * max(line.height, reference)))

        for start, end in dp.split_line_into_chunks(placed, max_aspect, line.height, margin_x):
            piece = crop[:, start:end]

            if piece.shape[1] < 2:
                continue

            scale = min(height / piece.shape[0], width / piece.shape[1])
            fit_scale = min(1.0, scale * piece.shape[0] / height)
            ink_fraction = line.height * scale / height

            entries.append((line.index, piece, fit_scale, ink_fraction))

    canvases = [
        hc.preprocess_image(_to_pil(piece), height, width)
        for _, piece, _, _ in entries
    ]

    return entries, canvases


def _recognize_once(loaded, image, spec, rotation=None, force=False):
    began = time.perf_counter()
    model_config = loaded.config
    result = PageResult(spec=spec)

    stages = dp.preprocess_document(image, spec.preprocess, rotation=rotation)
    prep_s = time.perf_counter() - began

    result.notes.extend(stages.notes)
    result.warnings.extend(_quality_warnings(stages.info))
    result.info = dict(stages.info)
    result.images = {
        "original": _to_pil(stages.original),
        "geometry": _to_pil(stages.geometry),
        "normalized": _to_pil(stages.normalized),
        "processed": _to_pil(stages.processed),
    }

    detection = None
    segment_s = 0.0
    segment_began = time.perf_counter()

    if spec.segment:
        try:
            detection = dp.detect_lines(stages.mask, spec.segmentation)
        except Exception as error:  # never crash the app on a weird image
            result.warnings.append(
                f"Line detection failed ({type(error).__name__}: {error}); "
                "falling back to the single-line pipeline on the whole image.")
            result.fallback = True
            detection = None

        segment_s = time.perf_counter() - segment_began

    use_whole_image = (not spec.segment) or detection is None

    if detection is not None and detection.kind in ("no_handwriting", "not_handwriting"):
        result.kind = detection.kind
        result.reason = detection.reason
        result.info.update({f"seg_{k}": v for k, v in detection.info.items()})

        if not force:
            result.warnings.append(detection.reason + " Recognition was skipped; "
                                   "enable 'Recognize anyway' to force it.")
            result.timings = {"preprocess_s": prep_s, "segment_s": segment_s,
                              "infer_s": 0.0, "total_s": time.perf_counter() - began}
            return result

        result.warnings.append(detection.reason + " Recognizing anyway as requested.")
        use_whole_image = True
        result.fallback = True

    if detection is not None and detection.lines:
        result.info.update({f"seg_{k}": v for k, v in detection.info.items()})
        result.kind = detection.kind
        result.images["overlay"] = _to_pil(
            dp.draw_line_overlay(stages.processed, detection.lines))

        entries, canvases = _build_line_inputs(stages, detection, spec, model_config)

        outputs, infer_s = run_model(loaded, canvases)

        by_line = {}

        for (line_index, piece, fit_scale, ink_fraction), canvas, (text, confidence) in zip(
                entries, canvases, outputs):
            by_line.setdefault(line_index, []).append(
                ChunkResult(text, confidence, _to_pil(piece), _canvas_image(canvas),
                            fit_scale, ink_fraction))

        for line in detection.lines:
            chunks = by_line.get(line.index, [])
            text = " ".join(c.text for c in chunks if c.text)
            confidence = _weighted_confidence([(c.text, c.confidence) for c in chunks])

            result.lines.append(LineResult(
                line.index, (line.x0, line.y0, line.x1, line.y1), text, confidence, chunks))

            if line.height < 14:
                result.warnings.append(
                    f"Line {line.index + 1}: handwriting is only ~{line.height} px tall in "
                    "the image; move closer or use a higher-resolution photo.")
            if not text:
                result.warnings.append(f"Line {line.index + 1}: the model returned no characters.")
            if any(c.fit_scale < 0.58 for c in chunks):
                result.warnings.append(
                    f"Line {line.index + 1}: a very long word was shrunk to fit the model input.")

        result.text = "\n".join(line.text for line in result.lines)

    elif use_whole_image:
        height, width = model_config["image_height"], model_config["image_width"]
        gray = stages.processed
        tensor = hc.preprocess_image(_to_pil(gray), height, width)
        outputs, infer_s = run_model(loaded, [tensor])
        text, confidence = outputs[0]

        scale = min(height / gray.shape[0], width / gray.shape[1])
        chunk = ChunkResult(text, confidence, _to_pil(gray), _canvas_image(tensor),
                            min(1.0, scale * gray.shape[0] / height), None)

        result.kind = result.kind or "whole_image"
        result.lines = [LineResult(0, None, text, confidence, [chunk])]
        result.text = text

        if spec.segment:
            result.warnings.append(
                "Fell back to the single-line pipeline on the whole image; "
                "results for multi-line pages will be poor.")

        result.warnings.append(
            f"The whole image was scaled to {height} px high; if it contains several "
            f"lines or a lot of blank paper the text is shrunk to about "
            f"{scale * 100:.0f}% of its size, far from the training scale.")
    else:
        infer_s = 0.0

    result.confidence = _weighted_confidence([(l.text, l.confidence) for l in result.lines])
    result.timings = {"preprocess_s": prep_s, "segment_s": segment_s, "infer_s": infer_s,
                      "total_s": time.perf_counter() - began}

    return result


def recognize_page(loaded, image, spec=None, force=False, rotation=None):
    """
    Recognize a photograph with the pipeline described by `spec`
    (default: D). `loaded` needs `.model`, `.vocabulary` and `.config`
    (e.g. app_logic.LoadedModel).

    rotation: None = automatic; 0/90/180/270 = apply exactly this
    counter-clockwise turn (manual override, skips detection).

    Never raises for ordinary bad input: problems become warnings and the
    result may be empty with an explanation in `warnings`.
    """

    spec = spec or EXPERIMENTS[DEFAULT_PIPELINE]

    if spec.legacy:
        return recognize_legacy(loaded, image, spec)

    if rotation is not None:
        return _recognize_once(loaded, image, spec, rotation=int(rotation), force=force)

    # Orientation: if the text axis says "vertical" the direction (90 vs 270)
    # cannot be told from ink statistics; try each and keep the one the model
    # is more confident about. This confidence is a heuristic.
    probe = dp.preprocess_document(
        image, replace(spec.preprocess, deskew=False, illumination=False,
                       denoise="none", contrast=False, suppress_ruling=False))
    candidates = probe.info.get("rotation_candidates", [0])

    if len(candidates) <= 1:
        return _recognize_once(loaded, image, spec, rotation=None, force=force)

    best, best_score = None, -1.0

    for rotation in candidates:
        attempt = _recognize_once(loaded, image, spec, rotation=rotation, force=force)
        score = attempt.confidence if attempt.confidence is not None else 0.0
        score *= min(1.0, len(attempt.text.replace("\n", "")) / 10.0)

        if score > best_score:
            best, best_score = attempt, score

    best.notes.append(
        f"Rotation {best.info['rotation_applied']} degrees chosen among {candidates} by "
        "comparing the model's heuristic confidence (not validated).")
    best.timings["total_s"] = best.timings.get("total_s", 0.0)

    return best


def recognize_line_images(loaded, images, spec=None):
    """
    Experiment E (diagnostic): run the ORIGINAL pipeline on images that were
    already cropped to single lines by a human. This isolates the recognizer
    from preprocessing and segmentation.

    Returns (list of (text, confidence), model seconds).
    """

    spec = spec or EXPERIMENTS["A"]
    config = loaded.config

    canvases = [
        hc.preprocess_image(hc.prepare_user_image(image, auto_contrast=spec.auto_contrast),
                            config["image_height"], config["image_width"])
        for image in images
    ]

    return run_model(loaded, canvases)
