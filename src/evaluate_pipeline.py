"""
Real-world evaluation of the photograph pipeline (experiments A-D, + E).

Put your photographs in  evaluation/real_world/  and (optionally) say what
they really say. Then, from the repository root:

    python src/evaluate_pipeline.py

Ground truth (optional, per image; metrics are computed ONLY where it exists):
    evaluation/real_world/ground_truth.csv     columns: image_file, reference_text
                                               (extra columns are copied through)
    evaluation/real_world/<image name>.txt     text file next to the image
                                               (the CSV wins if both exist)
    Line breaks in the reference: a real line break, or the two characters \\n
    (dataset convention). For scoring, line breaks count as spaces.

Experiments (see src/page_recognition.py):
    A  original preprocessing, whole image as one input
    B  improved preprocessing, whole page still ONE input
    C  B + line detection (+ word-gap chunking of long lines)
    D  C + notebook-ruling suppression
    E  DIAGNOSTIC: hand-cropped single-line images (you crop them) through the
       original pipeline: isolates the recognizer from preprocessing and
       segmentation. Needs  evaluation/real_world/manual_lines/<image name>/
       01.png 02.png ...  (one file per handwritten line, top to bottom).

Outputs (evaluation/real_world/results/ by default):
    per_image.csv   one row per image x experiment (filename, detected lines,
                    preprocessing, predicted text, timings, heuristic
                    confidence, CER/WER when ground truth exists, warnings)
    summary.csv / summary.json / report.md
    debug/<image>/<experiment>/   overlay, processed page, line crops and the
                                  exact model inputs ("what the model sees")

NOTHING is estimated: a metric is empty when its ground truth is missing, and
with few images the numbers are only descriptive (the report prints the counts).
Confidence is the model's heuristic (mean emitted-character probability), not a
calibrated probability.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for folder in (os.path.join(ROOT, "src"), os.path.join(ROOT, "app")):
    if folder not in sys.path:
        sys.path.insert(0, folder)

import htr_common as hc  # noqa: E402
import page_recognition as pr  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
DEFAULT_DIR = os.path.join(ROOT, "evaluation", "real_world")
SMALL_SAMPLE = 20

PER_IMAGE_COLUMNS = [
    "filename", "experiment", "preprocessing", "input_kind",
    "detected_lines", "detected_paragraphs", "model_inputs", "true_lines",
    "predicted_text", "heuristic_confidence",
    "preprocess_s", "segment_s", "inference_s", "total_s",
    "has_ground_truth", "reference_text", "out_of_vocabulary_chars",
    "char_errors", "reference_chars", "cer",
    "word_errors", "reference_words", "wer",
    "cer_ignoring_case", "exact_match", "line_count_correct",
    "fallback", "warnings",
]


# ============================================================
# DISCOVERY
# ============================================================

def normalise_reference(text):
    """Line breaks (real or literal backslash-n) become spaces; whitespace collapsed."""

    return " ".join(text.replace("\\n", " ").split())


def count_reference_lines(text):
    """Number of handwritten lines according to the reference's line breaks."""

    parts = [p for p in text.replace("\\n", "\n").splitlines() if p.strip()]

    return max(1, len(parts))


def find_images(folder):
    names = []

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)

        if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTENSIONS):
            names.append(name)

    return names


def load_ground_truth(folder, csv_path=None):
    """{filename: reference_text (raw)} from ground_truth.csv and sidecar .txt files."""

    truth = {}
    extras = {}

    csv_path = csv_path or os.path.join(folder, "ground_truth.csv")

    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                name = (row.get("image_file") or row.get("filename") or "").strip()
                text = row.get("reference_text")

                if text is None:
                    text = row.get("text")

                if name and text is not None and text.strip():
                    truth[name] = text
                    extras[name] = {
                        k: v for k, v in row.items()
                        if k not in ("image_file", "filename", "reference_text", "text")
                    }

    for name in find_images(folder):
        sidecar = os.path.join(folder, os.path.splitext(name)[0] + ".txt")

        if name not in truth and os.path.exists(sidecar):
            with open(sidecar, encoding="utf-8") as file:
                text = file.read().strip("\n")

            if text.strip():
                truth[name] = text

    return truth, extras


def find_manual_lines(folder, image_name):
    stem = os.path.splitext(image_name)[0]
    directory = os.path.join(folder, "manual_lines", stem)

    if not os.path.isdir(directory):
        return []

    return [os.path.join(directory, n) for n in sorted(os.listdir(directory))
            if n.lower().endswith(IMAGE_EXTENSIONS)]


def open_image(path):
    image = Image.open(path)
    image.load()

    return ImageOps.exif_transpose(image)


# ============================================================
# SCORING
# ============================================================

def score(reference_raw, prediction, vocabulary=None):
    """CER/WER pieces for one image. Everything is None when there is no reference."""

    empty = {k: "" for k in ("reference_text", "out_of_vocabulary_chars", "char_errors",
                             "reference_chars", "cer", "word_errors", "reference_words",
                             "wer", "cer_ignoring_case", "exact_match", "true_lines")}

    if reference_raw is None:
        return empty, None

    reference = normalise_reference(reference_raw)
    prediction = normalise_reference(prediction)

    char_errors = hc.char_distance(reference, prediction)
    word_errors = hc.word_distance(reference, prediction)
    reference_chars = len(reference)
    reference_words = len(reference.split())
    folded = hc.char_distance(reference.casefold(), prediction.casefold())

    oov = ""

    if vocabulary is not None:
        oov = "".join(sorted({c for c in reference if c not in vocabulary.char_to_id}))

    return {
        "reference_text": reference,
        "out_of_vocabulary_chars": oov,
        "char_errors": char_errors,
        "reference_chars": reference_chars,
        "cer": char_errors / reference_chars if reference_chars else "",
        "word_errors": word_errors,
        "reference_words": reference_words,
        "wer": word_errors / reference_words if reference_words else "",
        "cer_ignoring_case": folded / reference_chars if reference_chars else "",
        "exact_match": reference == prediction,
        "true_lines": count_reference_lines(reference_raw),
    }, (reference, prediction)


# ============================================================
# RUNNING EXPERIMENTS
# ============================================================

def save_debug(directory, result):
    os.makedirs(directory, exist_ok=True)

    for name, image in result.images.items():
        image.save(os.path.join(directory, f"page_{name}.png"))

    for line in result.lines:
        for number, chunk in enumerate(line.chunks, 1):
            tag = f"line{line.index + 1:02d}_part{number}"

            if chunk.crop is not None:
                chunk.crop.save(os.path.join(directory, f"{tag}_crop.png"))
            if chunk.canvas is not None:
                chunk.canvas.save(os.path.join(directory, f"{tag}_model_input.png"))


def override_margin_x(margin_x):
    """
    Use another horizontal crop margin (x line height) for every experiment
    that detects lines (C, D).  For diagnosis only: the default lives in
    document_pipeline.SegmentationConfig.
    """

    import dataclasses

    for key, spec in list(pr.EXPERIMENTS.items()):
        if spec.segment:
            pr.EXPERIMENTS[key] = dataclasses.replace(
                spec, segmentation=dataclasses.replace(spec.segmentation, margin_x=margin_x))


def run_pipeline_experiment(loaded, image, key):
    spec = pr.EXPERIMENTS[key]
    began = time.perf_counter()
    result = pr.recognize_page(loaded, image, spec, force=True)
    wall = time.perf_counter() - began

    return result, wall


def run_manual_experiment(loaded, paths):
    began = time.perf_counter()
    images = [open_image(p) for p in paths]
    outputs, infer_s = pr.recognize_line_images(loaded, images)
    wall = time.perf_counter() - began

    text = " ".join(t for t, _ in outputs if t)
    confidence = pr._weighted_confidence(outputs)

    return text, confidence, infer_s, wall, len(images)


def warm_up(loaded):
    """One throw-away forward pass so GPU start-up cost is not billed to image 1."""

    config = loaded.config
    blank = torch.ones(1, config["image_height"], config["image_width"])
    pr.run_model(loaded, [blank])


def evaluate_folder(loaded, folder, experiments, truth, extras, out_dir,
                    save_debug_images=True, log=print):
    rows = []
    names = find_images(folder)

    for name in names:
        path = os.path.join(folder, name)

        try:
            image = open_image(path)
        except Exception as error:
            log(f"  skipped {name}: cannot be opened ({error})")
            continue

        reference_raw = truth.get(name)

        for key in experiments:
            row = {c: "" for c in PER_IMAGE_COLUMNS}
            row.update(filename=name, experiment=key,
                       has_ground_truth=reference_raw is not None)

            if key == "E":
                manual = find_manual_lines(folder, name)

                if not manual:
                    continue

                text, confidence, infer_s, wall, count = run_manual_experiment(loaded, manual)
                row.update(preprocessing="original, on hand-cropped line images",
                           input_kind="manual_lines", detected_lines=count,
                           model_inputs=count, predicted_text=text,
                           heuristic_confidence=confidence, preprocess_s="",
                           segment_s="", inference_s=infer_s, total_s=wall,
                           fallback=False, warnings="")
                result = None
            else:
                result, wall = run_pipeline_experiment(loaded, image, key)
                spec = result.spec

                row.update(
                    preprocessing=spec.describe(),
                    input_kind=result.kind,
                    detected_lines=result.n_lines if spec.segment else "",
                    detected_paragraphs=result.n_paragraphs if spec.segment else "",
                    model_inputs=result.n_chunks,
                    predicted_text=result.text_oneline,
                    heuristic_confidence=result.confidence,
                    preprocess_s=result.timings.get("preprocess_s"),
                    segment_s=result.timings.get("segment_s"),
                    inference_s=result.timings.get("infer_s"),
                    total_s=result.timings.get("total_s", wall),
                    fallback=result.fallback,
                    warnings=" | ".join(result.warnings),
                )

                if save_debug_images:
                    save_debug(os.path.join(out_dir, "debug",
                                            os.path.splitext(name)[0], key), result)

            metrics, _ = score(reference_raw, row["predicted_text"], loaded.vocabulary)
            row.update(metrics)

            if reference_raw is not None and row["detected_lines"] != "":
                row["line_count_correct"] = int(row["detected_lines"]) == metrics["true_lines"]

            rows.append(row)
            log(f"  {name} [{key}] lines={row['detected_lines']!s:>2} "
                f"CER={'n/a' if row['cer'] == '' else format(row['cer'], '.3f')} "
                f"-> {row['predicted_text'][:60]!r}")

    return rows


# ============================================================
# SUMMARY
# ============================================================

def summarise(rows, experiments):
    summary = []

    for key in experiments:
        mine = [r for r in rows if r["experiment"] == key]

        if not mine:
            continue

        scored = [r for r in mine if r["reference_chars"] not in ("", None)]
        counted = [r for r in scored if r["line_count_correct"] != ""]
        times = [float(r["total_s"]) for r in mine if r["total_s"] != ""]

        entry = {
            "experiment": key,
            "images": len(mine),
            "images_with_ground_truth": len(scored),
            "corpus_cer": "", "corpus_wer": "", "macro_cer": "",
            "cer_ignoring_case": "", "exact_match": "",
            "line_count_accuracy": "", "line_count_n": len(counted),
            "mean_heuristic_confidence": "",
            "mean_total_s": float(np.mean(times)) if times else "",
            "images_with_no_text": sum(1 for r in mine if not r["predicted_text"]),
        }

        confidences = [float(r["heuristic_confidence"]) for r in mine
                       if r["heuristic_confidence"] not in ("", None)]

        if confidences:
            entry["mean_heuristic_confidence"] = float(np.mean(confidences))

        if scored:
            chars = sum(r["reference_chars"] for r in scored)
            words = sum(r["reference_words"] for r in scored)

            entry["corpus_cer"] = sum(r["char_errors"] for r in scored) / chars
            entry["corpus_wer"] = (sum(r["word_errors"] for r in scored) / words
                                   if words else "")
            entry["macro_cer"] = float(np.mean([r["cer"] for r in scored]))
            entry["cer_ignoring_case"] = float(np.mean([r["cer_ignoring_case"] for r in scored]))
            entry["exact_match"] = sum(1 for r in scored if r["exact_match"]) / len(scored)

        if counted:
            entry["line_count_accuracy"] = (
                sum(1 for r in counted if r["line_count_correct"]) / len(counted))

        summary.append(entry)

    return summary


def pct(value):
    return "n/a" if value == "" else f"{float(value) * 100:.1f}%"


def write_report(path, summary, rows, loaded, folder, experiments, model_path):
    n_images = len({r["filename"] for r in rows})
    n_truth = len({r["filename"] for r in rows if r["has_ground_truth"] is True})
    small = 0 < n_truth < SMALL_SAMPLE

    lines = [
        "# Real-world pipeline evaluation",
        "",
        f"- Model: `{os.path.basename(model_path)}` (sha256 {loaded.sha256[:12]}...)",
        f"- Image folder: `{os.path.relpath(folder, ROOT)}`",
        f"- Images: {n_images}; with ground truth: {n_truth}",
        f"- Experiments run: {', '.join(k for k in experiments if any(r['experiment'] == k for r in rows))}",
        "- CER/WER: whitespace-normalised, line breaks scored as spaces; "
        "corpus = total errors / total reference length (primary), macro = mean "
        "per-image (secondary).",
        "- Confidence = model heuristic, not a calibrated probability.",
        "",
    ]

    if n_truth == 0:
        lines += ["**No ground truth was provided, so no CER/WER was computed.** "
                  "Add `ground_truth.csv` or `<image>.txt` files to score the results.", ""]
    elif small:
        lines += [f"**Only {n_truth} images have ground truth (< {SMALL_SAMPLE}): the "
                  "numbers below are descriptive and carry large uncertainty; do not "
                  "claim one pipeline is better from differences of a few points.**", ""]

    lines += ["| Exp | Images | With GT | Corpus CER | Corpus WER | Macro CER | CER ignoring case | "
              "Exact match | Line-count correct | No-text images | Mean time (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]

    for e in summary:
        count_text = ("n/a" if e["line_count_accuracy"] == ""
                      else f"{pct(e['line_count_accuracy'])} (n={e['line_count_n']})")
        mean_time = "n/a" if e["mean_total_s"] == "" else f"{e['mean_total_s']:.2f}"

        lines.append(
            f"| {e['experiment']} | {e['images']} | {e['images_with_ground_truth']} | "
            f"{pct(e['corpus_cer'])} | {pct(e['corpus_wer'])} | {pct(e['macro_cer'])} | "
            f"{pct(e['cer_ignoring_case'])} | {pct(e['exact_match'])} | {count_text} | "
            f"{e['images_with_no_text']} | {mean_time} |")

    oov = sorted({c for r in rows for c in str(r["out_of_vocabulary_chars"])})

    if oov:
        lines += ["", "Reference characters the model can never output "
                  f"(not in its vocabulary): {' '.join(repr(c) for c in oov)}. "
                  "They count as errors for every pipeline."]

    lines += [
        "",
        "## How to read this (hypotheses to test, not conclusions)",
        "",
        "Compare the corpus CER of the experiments on the SAME images:",
        "",
        "- A vs B: both send the whole image as one line. A gap would point to "
        "preprocessing (illumination/contrast/orientation) mattering.",
        "- B vs C: the gap is the effect of line segmentation (and of giving the model "
        "line-sized inputs).",
        "- C vs D: the gap is the effect of notebook-ruling suppression.",
        "- E (hand-cropped lines, original pipeline) vs C/D: if E is much better than "
        "C/D, automatic preprocessing or segmentation is the bottleneck. If E is "
        "itself poor, the recognizer does not generalise to your handwriting "
        "(training-data mismatch: the model saw only synthetic handwriting).",
        "- Check `debug/<image>/<experiment>/*_model_input.png`: a crop that looks "
        "clean to you but is read wrongly points at the model; a crop that is "
        "cut, tiny, striped by ruling or mostly blank points at preprocessing or "
        "segmentation.",
        "",
        "Per-image detail is in `per_image.csv`.",
    ]

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", default=DEFAULT_DIR, help="folder with the photographs")
    parser.add_argument("--ground-truth", default=None,
                        help="CSV (image_file, reference_text); default <images>/ground_truth.csv")
    parser.add_argument("--model", default=None,
                        help="checkpoint (default: models/final_model.pt or $HTR_MODEL_PATH)")
    parser.add_argument("--experiments", nargs="+", default=["A", "B", "C", "D", "E"],
                        choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--output", default=None,
                        help="output folder (default <images>/results)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-debug-images", action="store_true")
    parser.add_argument("--margin-x", type=float, default=None,
                        help="horizontal crop margin around each line, in line heights, for C/D "
                        "(diagnostic override; default = SegmentationConfig.margin_x)")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.margin_x is not None:
        override_margin_x(args.margin_x)
        print(f"Horizontal crop margin overridden: {args.margin_x} line heights (C/D)")

    import app_logic as logic

    folder = os.path.abspath(args.images)

    if not os.path.isdir(folder):
        sys.exit(f"Image folder not found: {folder}\n"
                 f"Create it and put your photographs inside "
                 f"(see evaluation/real_world/README.md).")

    names = find_images(folder)

    if not names:
        sys.exit(f"No images ({', '.join(IMAGE_EXTENSIONS)}) found in {folder}.\n"
                 "Add photographs and run again; nothing was evaluated.")

    device = args.device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model or logic.model_path()
    loaded = logic.load_app_model(model_path, device)
    loaded.model.eval()

    out_dir = os.path.abspath(args.output or os.path.join(folder, "results"))
    os.makedirs(out_dir, exist_ok=True)

    truth, extras = load_ground_truth(folder, args.ground_truth)
    missing = [n for n in truth if n not in names]

    if missing:
        print(f"Warning: ground truth given for images that are not in the folder: {missing}")

    print(f"Model: {model_path} on {device}; {len(names)} images, "
          f"{sum(1 for n in names if n in truth)} with ground truth")

    warm_up(loaded)

    rows = evaluate_folder(loaded, folder, args.experiments, truth, extras, out_dir,
                           save_debug_images=not args.no_debug_images)

    for row in rows:
        row.update({f"gt_{k}": v for k, v in extras.get(row["filename"], {}).items()})

    summary = summarise(rows, args.experiments)

    extra_columns = sorted({k for r in rows for k in r if k.startswith("gt_")})
    write_csv(os.path.join(out_dir, "per_image.csv"), PER_IMAGE_COLUMNS + extra_columns, rows)
    write_csv(os.path.join(out_dir, "summary.csv"),
              list(summary[0].keys()) if summary else ["experiment"], summary)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump({"model": os.path.basename(model_path), "model_sha256": loaded.sha256,
                   "device": device, "images": len(names), "summary": summary},
                  file, indent=2)

    write_report(os.path.join(out_dir, "report.md"), summary, rows, loaded, folder,
                 args.experiments, model_path)

    print(f"\nWrote {out_dir}/ (per_image.csv, summary.csv, summary.json, report.md)")

    return rows, summary


if __name__ == "__main__":
    main()
