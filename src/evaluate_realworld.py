"""
Score the model on a REAL photo/scan test set collected by the team.

Inputs
------
    evaluation/photos/                     the images you collected
    evaluation/ground_truth.csv            what each image really says

ground_truth.csv columns (see evaluation/ground_truth_template.csv):

    image_file, participant_id, reference_text, line_type,
    handwriting_condition, capture_condition

    line_type              single-line | multiline
    handwriting_condition  neat | average | messy
    capture_condition      scan | good photo | dim or shadowed photo |
                           blurry photo   (or your own labels)

    Multiline references use the two-character sequence \n (backslash n)
    between lines, exactly like the OpenHand-Synth dataset.

Usage (repository root):

    python src/evaluate_realworld.py

Every image is recognised twice: with the trained pipeline exactly as
trained (PRIMARY) and with the optional auto-contrast aid (SECONDARY).

Outputs (evaluation/results/):
    realworld_per_image.csv
    realworld_summary.csv / realworld_summary.md

Metrics are labelled like everywhere else: corpus CER/WER (primary),
exact match, macro-CER (secondary). With small samples these numbers
are DESCRIPTIVE; report the counts next to every percentage.
"""

import argparse
import csv
import os
from collections import defaultdict

from PIL import Image

import htr_common as hc
import analyze_model as am


DIMENSIONS = [
    ("line_type", "Line type"),
    ("handwriting_condition", "Handwriting"),
    ("capture_condition", "Capture condition"),
    ("participant_id", "Participant"),
]


def evaluate_images(model, vocabulary, config, rows, photos_dir, device):
    results = []

    for row in rows:
        path = os.path.join(photos_dir, row["image_file"])

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image listed in ground truth is missing: {path}")

        image = Image.open(path)
        image.load()

        reference = row["reference_text"]

        # Characters the model can never output (not in its vocabulary)
        # inflate the error rate; report them instead of hiding them.
        out_of_vocabulary = "".join(
            sorted({c for c in reference if c not in vocabulary.char_to_id})
        )

        for auto_contrast in (False, True):
            prediction, confidence = hc.recognize(
                model, vocabulary, config, image,
                device=device, auto_contrast=auto_contrast,
            )

            char_errors = hc.char_distance(reference, prediction)
            word_errors = hc.word_distance(reference, prediction)
            length = len(reference)

            results.append(
                {
                    **{k: row[k] for k in (
                        "image_file", "participant_id", "line_type",
                        "handwriting_condition", "capture_condition",
                    )},
                    "auto_contrast": auto_contrast,
                    "out_of_vocabulary_chars": out_of_vocabulary,
                    "reference": reference,
                    "prediction": prediction,
                    "confidence": "" if confidence is None else confidence,
                    "text_length": length,
                    "reference_words": len(reference.split()),
                    "char_errors": char_errors,
                    "word_errors": word_errors,
                    "cer": char_errors / length if length else 0.0,
                    "exact_match": reference == prediction,
                    "prediction_length": len(prediction),
                    "prediction_ratio": len(prediction) / length if length else 0.0,
                    "collapsed_prediction": hc.is_collapsed(length, len(prediction)),
                }
            )

    return results


def summary_rows(results):
    rows = []

    for auto_contrast in (False, True):
        subset = [r for r in results if r["auto_contrast"] == auto_contrast]
        mode = "auto-contrast ON" if auto_contrast else "as trained (PRIMARY)"

        rows.append(
            ["Overall", "all images", mode, am.group_stats(subset)]
        )

        for field, title in DIMENSIONS:
            groups = defaultdict(list)

            for record in subset:
                groups[record[field]].append(record)

            for name in sorted(groups):
                rows.append([title, name, mode, am.group_stats(groups[name])])

    return rows


def write_summary(output_dir, rows):
    header = [
        "dimension", "group", "pipeline", "images", "corpus_cer",
        "corpus_wer", "exact_match", "macro_cer", "collapse_count",
    ]

    flat = [
        [d, g, m, s["samples"], s["corpus_cer"], s["corpus_wer"],
         s["exact_match"], s["macro_cer"], s["collapse_count"]]
        for d, g, m, s in rows
    ]

    with open(os.path.join(output_dir, "realworld_summary.csv"),
              "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(flat)

    lines = [
        "| Dimension | Group | Pipeline | n | Corpus CER | Corpus WER | Exact match | Macro-CER |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for d, g, m, s in rows:
        lines.append(
            f"| {d} | {g} | {m} | {s['samples']} | "
            f"{s['corpus_cer'] * 100:.1f}% | {s['corpus_wer'] * 100:.1f}% | "
            f"{s['exact_match'] * 100:.1f}% | {s['macro_cer'] * 100:.1f}% |"
        )

    with open(os.path.join(output_dir, "realworld_summary.md"),
              "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join("models", "final_model.pt"))
    parser.add_argument("--photos-dir", default=os.path.join("evaluation", "photos"))
    parser.add_argument("--ground-truth", default=os.path.join("evaluation", "ground_truth.csv"))
    parser.add_argument("--output-dir", default=os.path.join("evaluation", "results"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.ground_truth, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise SystemExit("ground_truth.csv has no rows.")

    model, vocabulary, config, _ = hc.load_model(args.model, args.device)

    results = evaluate_images(
        model, vocabulary, config, rows, args.photos_dir, args.device
    )

    os.makedirs(args.output_dir, exist_ok=True)

    columns = list(results[0].keys())

    with open(os.path.join(args.output_dir, "realworld_per_image.csv"),
              "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    summary = summary_rows(results)
    write_summary(args.output_dir, summary)

    oov_images = sum(
        1 for r in results
        if not r["auto_contrast"] and r["out_of_vocabulary_chars"]
    )

    if oov_images:
        print(f"NOTE: {oov_images} reference(s) contain characters outside "
              "the model vocabulary (see out_of_vocabulary_chars).")

    overall = summary[0][3]
    print(f"{overall['samples']} images (as trained): "
          f"corpus CER {overall['corpus_cer'] * 100:.1f}%, "
          f"corpus WER {overall['corpus_wer'] * 100:.1f}%, "
          f"exact {overall['exact_match'] * 100:.1f}%")
    print(f"Written to {args.output_dir}")


if __name__ == "__main__":
    main()
