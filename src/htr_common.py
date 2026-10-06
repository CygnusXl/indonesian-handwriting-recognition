"""
Shared utilities for the Indonesian handwriting-recognition project.

This module is the single source of truth for everything that must be
IDENTICAL between training, evaluation, error analysis and the
application:

    * image preprocessing            (preprocess_image)
    * CTC greedy decoding            (greedy_decode)
    * edit distance / CER / WER      (edit_distance, corpus_metrics, ...)
    * subgroup definitions           (length_category, is_multiline, ...)
    * model construction / loading   (MODEL_REGISTRY, load_model)

It deliberately does NOT import `datasets` (Hugging Face), so the
application can use it without the training-time dependencies.

Convention (unchanged from the rest of the project): scripts are run from
the repository root and import sibling modules directly, e.g.

    python src/analyze_model.py --model baseline
"""

import hashlib
import os

import numpy as np
import torch
from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1024

# Each entry fully describes how a model was trained so that evaluation
# and the application rebuild it exactly.
MODEL_REGISTRY = {
    "baseline": {
        "label": "Baseline",
        "checkpoint": "checkpoints/baseline_best.pt",
        "architecture": "cnn_bilstm",
        "image_height": 128,
        "image_width": 1024,
        "ctc_timesteps": 256,
    },
    # OPTIONAL control run (src/train_baseline_extended.py): the baseline
    # configuration trained for more epochs. Skipped everywhere if its
    # checkpoint does not exist.
    "baseline_extended": {
        "label": "Baseline, extended training",
        "checkpoint": "checkpoints/baseline_extended_best.pt",
        "architecture": "cnn_bilstm",
        "image_height": 128,
        "image_width": 1024,
        "ctc_timesteps": 256,
    },
    "experiment1": {
        "label": "Experiment 1 (wider input)",
        "checkpoint": "checkpoints/experiment1_best.pt",
        "architecture": "cnn_bilstm",
        "image_height": 128,
        "image_width": 1536,
        "ctc_timesteps": 384,
    },
    "experiment2": {
        "label": "Experiment 2 (taller input)",
        "checkpoint": "checkpoints/experiment2_best.pt",
        "architecture": "cnn_bilstm_exp2",
        "image_height": 192,
        "image_width": 1024,
        "ctc_timesteps": 256,
    },
    "experiment3": {
        "label": "Experiment 3 (multiline-balanced)",
        "checkpoint": "checkpoints/experiment3_best.pt",
        "architecture": "cnn_bilstm",
        "image_height": 128,
        "image_width": 1024,
        "ctc_timesteps": 256,
    },
}

# The common test set used for every model comparison: Indonesian test
# samples whose CTC requirement fits in 256 time steps (208 samples).
COMMON_TEST_CTC_TIMESTEPS = 256


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    target_height=IMAGE_HEIGHT,
    target_width=IMAGE_WIDTH,
):
    """
    Convert a PIL image into a fixed-size grayscale tensor
    [1, target_height, target_width].

    Aspect ratio is preserved. The image is scaled to `target_height`;
    if that makes it wider than `target_width` it is instead scaled to
    `target_width` (so tall / multiline images become SMALLER than
    `target_height`). The result is placed on a white canvas, top-left
    aligned horizontally and centred vertically.

    NOTE: this is the exact function used to train and evaluate every
    reported model. Do not change it without retraining.
    """

    image = image.convert("L")

    original_width, original_height = image.size

    scale = target_height / original_height
    new_width = int(original_width * scale)

    if new_width > target_width:
        new_width = target_width
        scale = target_width / original_width
        new_height = int(original_height * scale)
    else:
        new_height = target_height

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0

    padded = np.ones(
        (target_height, target_width),
        dtype=np.float32,
    )

    y_offset = (target_height - new_height) // 2

    padded[
        y_offset:y_offset + new_height,
        :new_width,
    ] = image_array

    tensor = torch.from_numpy(padded)

    return tensor.unsqueeze(0)


def prepare_user_image(image, auto_contrast=False):
    """
    Make an arbitrary user-supplied image safe to feed to
    `preprocess_image`. Used by the application and the real-world
    evaluation only (NOT by the synthetic-data pipeline).

    - applies EXIF orientation (phone photos);
    - flattens transparency onto white;
    - optionally stretches contrast (`auto_contrast`). This is NOT part
      of the training pipeline; it is an explicit, switchable aid for
      photos with grey paper / low contrast and is evaluated as such.
    """

    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)

    image = image.convert("L")

    if auto_contrast:
        image = ImageOps.autocontrast(image, cutoff=1)

    return image


# ============================================================
# VOCABULARY (from checkpoint)
# ============================================================

class CheckpointVocabulary:
    """
    Character <-> id mapping restored from a checkpoint. Behaves like
    dataset.CharacterVocabulary for decoding and encoding, but needs no
    dataset access.
    """

    def __init__(self, char_to_id, id_to_char=None):
        self.char_to_id = dict(char_to_id)

        if id_to_char is None:
            id_to_char = {
                index: character
                for character, index in self.char_to_id.items()
            }

        self.id_to_char = {
            int(index): character
            for index, character in id_to_char.items()
        }

        self.num_characters = len(self.char_to_id)
        self.blank_id = self.num_characters
        self.num_classes = self.num_characters + 1

    def encode(self, text):
        return [self.char_to_id[character] for character in text]

    def decode(self, ids):
        result = []

        for index in ids:
            if index == self.blank_id:
                continue
            if index in self.id_to_char:
                result.append(self.id_to_char[index])

        return "".join(result)


def vocabulary_from_checkpoint(checkpoint):
    return CheckpointVocabulary(
        checkpoint["vocabulary_char_to_id"],
        checkpoint.get("vocabulary_id_to_char"),
    )


# ============================================================
# CTC GREEDY DECODING
# ============================================================

def greedy_decode(logits, vocabulary):
    """
    Greedy CTC decoding: argmax per time step, collapse consecutive
    repeats, remove blanks.

    logits: [time, batch, classes]  ->  list[str] (one per batch item)
    """

    predictions = torch.argmax(logits, dim=2)

    texts = []

    for batch_index in range(predictions.size(1)):
        sequence = predictions[:, batch_index].tolist()

        decoded_ids = []
        previous = None

        for token in sequence:
            if token == vocabulary.blank_id:
                previous = token
                continue

            if token != previous:
                decoded_ids.append(token)

            previous = token

        texts.append(vocabulary.decode(decoded_ids))

    return texts


def greedy_decode_with_confidence(logits, vocabulary):
    """
    Same decoding as `greedy_decode`, plus a heuristic confidence per
    item: the mean softmax probability of the time steps that emitted
    a character. This is NOT a calibrated probability of correctness;
    it is only a rough signal that can be shown to users.

    Returns list of (text, confidence_or_None).
    """

    probabilities = torch.softmax(logits, dim=2)
    max_probs, predictions = probabilities.max(dim=2)

    outputs = []

    for batch_index in range(predictions.size(1)):
        sequence = predictions[:, batch_index].tolist()
        scores = max_probs[:, batch_index].tolist()

        decoded_ids = []
        emitted_scores = []
        previous = None

        for token, score in zip(sequence, scores):
            if token == vocabulary.blank_id:
                previous = token
                continue

            if token != previous:
                decoded_ids.append(token)
                emitted_scores.append(score)

            previous = token

        confidence = (
            sum(emitted_scores) / len(emitted_scores)
            if emitted_scores
            else None
        )

        outputs.append((vocabulary.decode(decoded_ids), confidence))

    return outputs


# ============================================================
# EDIT DISTANCE AND METRICS
# ============================================================

def edit_distance(reference, hypothesis):
    """
    Levenshtein distance between two sequences (strings for CER,
    lists of words for WER).
    """

    previous_row = list(range(len(hypothesis) + 1))

    for i in range(1, len(reference) + 1):
        current_row = [i]

        for j in range(1, len(hypothesis) + 1):
            substitution_cost = (
                0 if reference[i - 1] == hypothesis[j - 1] else 1
            )

            current_row.append(
                min(
                    current_row[j - 1] + 1,
                    previous_row[j] + 1,
                    previous_row[j - 1] + substitution_cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]


def char_distance(reference, hypothesis):
    return edit_distance(reference, hypothesis)


def word_distance(reference, hypothesis):
    # Whitespace tokenisation, identical to the original evaluate.py.
    # NOTE: multiline references contain the two-character sequence
    # backslash + 'n' (not whitespace), so a line break does not split
    # words in the WER computation.
    return edit_distance(reference.split(), hypothesis.split())


def sample_cer(reference, hypothesis):
    if len(reference) == 0:
        return 0.0

    return char_distance(reference, hypothesis) / len(reference)


def corpus_cer(references, hypotheses):
    """total character edit distance / total reference characters"""

    total_distance = 0
    total_characters = 0

    for reference, hypothesis in zip(references, hypotheses):
        total_distance += char_distance(reference, hypothesis)
        total_characters += len(reference)

    return total_distance / total_characters if total_characters else 0.0


def corpus_wer(references, hypotheses):
    """total word edit distance / total reference words"""

    total_distance = 0
    total_words = 0

    for reference, hypothesis in zip(references, hypotheses):
        total_distance += word_distance(reference, hypothesis)
        total_words += len(reference.split())

    return total_distance / total_words if total_words else 0.0


def macro_cer(references, hypotheses):
    """mean of per-sample CER values"""

    values = [
        sample_cer(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
    ]

    return sum(values) / len(values) if values else 0.0


def exact_match_rate(references, hypotheses):
    if not references:
        return 0.0

    matches = sum(
        reference == hypothesis
        for reference, hypothesis in zip(references, hypotheses)
    )

    return matches / len(references)


def corpus_metrics(references, hypotheses):
    """All reported metrics, explicitly labelled by aggregation."""

    return {
        "samples": len(references),
        "corpus_cer": corpus_cer(references, hypotheses),
        "corpus_wer": corpus_wer(references, hypotheses),
        "macro_cer": macro_cer(references, hypotheses),
        "exact_match": exact_match_rate(references, hypotheses),
    }


# ============================================================
# SUBGROUP DEFINITIONS (identical to the original error analysis)
# ============================================================

def is_multiline(text):
    """
    OpenHand-Synth stores line breaks either as a real newline or as the
    two-character sequence backslash + 'n'. Both count as multiline.
    """

    return "\n" in text or "\\n" in text


def length_category(length):
    if length <= 30:
        return "Short (1-30)"
    if length <= 60:
        return "Medium (31-60)"
    if length <= 100:
        return "Long (61-100)"
    return "Very long (101+)"


LENGTH_CATEGORY_ORDER = [
    "Short (1-30)",
    "Medium (31-60)",
    "Long (61-100)",
    "Very long (101+)",
]


def neatness_category(neatness):
    if neatness <= 60:
        return "Very messy (50-60)"
    if neatness <= 70:
        return "Messy (61-70)"
    if neatness <= 80:
        return "Moderate (71-80)"
    if neatness <= 90:
        return "Neat (81-90)"
    return "Very neat (91-95)"


def is_collapsed(reference_length, prediction_length):
    """
    'Severe prediction collapse': the prediction is shorter than 25% of
    a reference that has at least 20 characters.
    """

    if reference_length <= 0:
        return False

    return (
        prediction_length / reference_length < 0.25
        and reference_length >= 20
    )


# ============================================================
# MODEL CONSTRUCTION AND LOADING
# ============================================================

def build_model(architecture, num_classes, image_height=IMAGE_HEIGHT):
    """Rebuild an (untrained) model of a known architecture."""

    if architecture == "cnn_bilstm":
        from model import CNNBiLSTM

        return CNNBiLSTM(num_classes=num_classes)

    if architecture == "cnn_bilstm_exp2":
        from model_experiment2 import CNNBiLSTMExperiment2

        return CNNBiLSTMExperiment2(
            num_classes=num_classes,
            image_height=image_height,
        )

    raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def file_sha256(path, chunk_size=1 << 20):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_checkpoint(path, device="cpu"):
    """
    Load a checkpoint dictionary. Tries the safe `weights_only=True`
    loader first and only falls back to full unpickling (which can
    execute code) for legacy files, with a printed warning. Only load
    checkpoints you produced or trust.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print(
            f"WARNING: {path} could not be loaded with "
            "weights_only=True; falling back to full unpickling. "
            "Only do this for files you trust."
        )
        return torch.load(path, map_location=device, weights_only=False)


def infer_config(checkpoint, path):
    """
    Work out (architecture, height, width, ctc_timesteps) for a
    checkpoint. Order of precedence:

        1. fields stored in the checkpoint (deployment checkpoints),
        2. the registry entry whose checkpoint filename matches,
        3. shapes found in the state dict (height) + defaults.
    """

    if "architecture" in checkpoint and "image_height" in checkpoint:
        return {
            "architecture": checkpoint["architecture"],
            "image_height": int(checkpoint["image_height"]),
            "image_width": int(checkpoint["image_width"]),
            "ctc_timesteps": int(
                checkpoint.get(
                    "ctc_timesteps",
                    int(checkpoint["image_width"]) // 4,
                )
            ),
        }

    basename = os.path.basename(path)

    for entry in MODEL_REGISTRY.values():
        if os.path.basename(entry["checkpoint"]) == basename:
            return {
                key: entry[key]
                for key in (
                    "architecture",
                    "image_height",
                    "image_width",
                    "ctc_timesteps",
                )
            }

    weight = checkpoint["model_state_dict"]["lstm.weight_ih_l0"]
    cnn_height = weight.shape[1] // 256
    image_height = cnn_height * 16

    return {
        "architecture": (
            "cnn_bilstm" if image_height == 128 else "cnn_bilstm_exp2"
        ),
        "image_height": image_height,
        "image_width": IMAGE_WIDTH,
        "ctc_timesteps": IMAGE_WIDTH // 4,
    }


def load_model(path, device="cpu"):
    """
    Load any project checkpoint (training or deployment) for inference.

    Returns (model_in_eval_mode, vocabulary, config, checkpoint_dict)
    where `config` has architecture / image_height / image_width /
    ctc_timesteps.
    """

    checkpoint = read_checkpoint(path, device)
    vocabulary = vocabulary_from_checkpoint(checkpoint)
    config = infer_config(checkpoint, path)

    model = build_model(
        config["architecture"],
        vocabulary.num_classes,
        config["image_height"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, vocabulary, config, checkpoint


@torch.no_grad()
def recognize(
    model,
    vocabulary,
    config,
    pil_image,
    device="cpu",
    auto_contrast=False,
):
    """
    Full inference path used by the application and the real-world
    evaluation: user image -> preprocessing -> model -> CTC greedy
    decoding.

    Returns (text, confidence_or_None).
    """

    image = prepare_user_image(pil_image, auto_contrast=auto_contrast)

    tensor = preprocess_image(
        image,
        target_height=config["image_height"],
        target_width=config["image_width"],
    ).unsqueeze(0).to(device)

    logits = model(tensor)

    return greedy_decode_with_confidence(logits, vocabulary)[0]


def display_text(text):
    """
    Turn the dataset's literal backslash-n line separators into real
    line breaks for display.
    """

    return text.replace("\\n", "\n")
