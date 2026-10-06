"""
Optional local logging of practical-evaluation sessions.

When a participant types the true text of their handwriting into the
app, the character/word error rates are computed with the SAME functions
used for the offline evaluation and appended to a CSV. Only text and
metadata are stored (never the image), and nothing leaves the machine.
"""

import csv
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import htr_common as hc  # noqa: E402

LOG_PATH = os.path.join(ROOT, "evaluation", "results", "app_sessions.csv")

COLUMNS = [
    "timestamp_utc",
    "participant_id",
    "image_id",
    "line_type",
    "handwriting_condition",
    "capture_condition",
    "auto_contrast",
    "reference_text",
    "prediction",
    "char_errors",
    "reference_chars",
    "cer",
    "word_errors",
    "reference_words",
    "wer",
    "exact_match",
    "usefulness_1_5",
    "seconds",
    "model_sha256_12",
]


def build_row(
    participant_id,
    image_id,
    line_type,
    handwriting_condition,
    capture_condition,
    auto_contrast,
    reference_text,
    prediction,
    usefulness,
    seconds,
    model_sha256,
):
    char_errors = hc.char_distance(reference_text, prediction)
    word_errors = hc.word_distance(reference_text, prediction)
    reference_chars = len(reference_text)
    reference_words = len(reference_text.split())

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "participant_id": participant_id,
        "image_id": image_id,
        "line_type": line_type,
        "handwriting_condition": handwriting_condition,
        "capture_condition": capture_condition,
        "auto_contrast": bool(auto_contrast),
        "reference_text": reference_text,
        "prediction": prediction,
        "char_errors": char_errors,
        "reference_chars": reference_chars,
        "cer": char_errors / reference_chars if reference_chars else "",
        "word_errors": word_errors,
        "reference_words": reference_words,
        "wer": word_errors / reference_words if reference_words else "",
        "exact_match": reference_text == prediction,
        "usefulness_1_5": usefulness if usefulness else "",
        "seconds": round(seconds, 3),
        "model_sha256_12": model_sha256[:12],
    }


def append_session(row, path=LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)

        if new_file:
            writer.writeheader()

        writer.writerow(row)
