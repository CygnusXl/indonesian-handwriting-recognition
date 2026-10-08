"""
Component benchmark of the DOCUMENT stages on SYNTHETIC notebook pages.

    python src/evaluate_segmentation.py [--seeds 8] [--output results/segmentation_benchmark]

WHAT THIS MEASURES: whether orientation handling, preprocessing and line
detection find the right NUMBER of lines on pages whose layout is known by
construction (src/synthetic_pages.py), under shadows, ruling, noise, skew,
perspective, rotation and strike-throughs; and that single-line crops are
not split and blank / noise pages are flagged.

WHAT THIS DOES NOT MEASURE: any recognition accuracy, and anything about REAL
handwriting. The pages are font-rendered text with wobble. The thresholds in
src/document_pipeline.py were developed while looking at pages from the same
generator, so these numbers are OPTIMISTIC for real photographs. Use
src/evaluate_pipeline.py on real photos for the real answer.

No model is loaded.
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import document_pipeline as dp  # noqa: E402
import page_recognition as pr  # noqa: E402
import synthetic_pages as sp  # noqa: E402

FIRST_SEED = 1000   # not used while developing the thresholds by hand
PIPELINES = ["C", "D"]


def detect(image, key):
    spec = pr.EXPERIMENTS[key]
    stages = dp.preprocess_document(image, spec.preprocess)
    detection = dp.detect_lines(stages.mask, spec.segmentation)

    return stages, detection


def expected_rotation(condition):
    """Counter-clockwise turn that makes the page upright (None = no turn)."""

    return 270 if condition == "rotated_90" else 0


def page_benchmark(seeds):
    rows = []

    for condition in sp.CONDITIONS:
        for offset in range(seeds):
            seed = FIRST_SEED + offset
            image, truth = sp.make_condition_page(condition, seed)

            for key in PIPELINES:
                stages, detection = detect(image, key)
                rotation = stages.info.get("rotation_applied", 0)
                candidates = stages.info.get("rotation_candidates", [0])

                rows.append({
                    "condition": condition, "seed": seed, "pipeline": key,
                    "true_lines": truth["n_lines"],
                    "detected_lines": len(detection.lines),
                    "line_count_correct": len(detection.lines) == truth["n_lines"],
                    "kind": detection.kind,
                    "orientation_ok": expected_rotation(condition) in candidates,
                    "rotation_applied": rotation,
                    "skew_degrees": round(stages.info.get("skew_degrees", 0.0), 2),
                })

    return rows


def single_line_benchmark(seeds):
    """Tight crops of ONE line (with and without ruling) must stay one line."""

    rows = []

    for ruled in (False, True):
        for offset in range(seeds):
            seed = FIRST_SEED + 500 + offset
            page, truth = sp.render_page(seed, n_lines=1, ruled=ruled)
            x0, y0, x1, y1 = truth["boxes"][0]
            pad = 25
            crop = page.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))

            for key in PIPELINES:
                _, detection = detect(crop, key)
                rows.append({
                    "case": f"single_line_crop_{'ruled' if ruled else 'plain'}",
                    "seed": seed, "pipeline": key,
                    "detected_lines": len(detection.lines), "kind": detection.kind,
                    "ok": len(detection.lines) == 1 and detection.kind == "single_line",
                })

    return rows


def empty_page_benchmark(seeds):
    """Blank paper / pure noise must be flagged, not 'read'."""

    rows = []

    for offset in range(seeds):
        seed = FIRST_SEED + 900 + offset
        rng = np.random.default_rng(seed)

        blank = sp.render_page(seed, n_lines=1, ruled=bool(offset % 2))[0]
        blank = blank.crop((0, 1100, blank.width, blank.height))   # below the text
        noise = Image.fromarray(rng.integers(0, 255, (900, 700, 3), dtype=np.uint8))

        for name, image in (("blank_paper", blank), ("random_noise", noise)):
            for key in PIPELINES:
                _, detection = detect(image, key)
                rows.append({
                    "case": name, "seed": seed, "pipeline": key,
                    "detected_lines": len(detection.lines), "kind": detection.kind,
                    "ok": detection.kind in ("no_handwriting", "not_handwriting"),
                })

    return rows


def summarise(page_rows, single_rows, empty_rows):
    lines = []

    lines.append("| Condition | Pipeline | Pages | Line count correct | Orientation axis ok |")
    lines.append("|---|---|---|---|---|")

    for condition in sp.CONDITIONS:
        for key in PIPELINES:
            mine = [r for r in page_rows if r["condition"] == condition and r["pipeline"] == key]
            correct = sum(r["line_count_correct"] for r in mine)
            oriented = sum(r["orientation_ok"] for r in mine)
            lines.append(f"| {condition} | {key} | {len(mine)} | {correct}/{len(mine)} | "
                         f"{oriented}/{len(mine)} |")

    lines += ["", "| Case | Pipeline | Cases | As expected |", "|---|---|---|---|"]

    for rows in (single_rows, empty_rows):
        for case in sorted({r["case"] for r in rows}):
            for key in PIPELINES:
                mine = [r for r in rows if r["case"] == case and r["pipeline"] == key]
                lines.append(f"| {case} | {key} | {len(mine)} | {sum(r['ok'] for r in mine)}/{len(mine)} |")

    return "\n".join(lines)


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=8, help="pages per condition")
    parser.add_argument("--output", default=os.path.join(ROOT, "results", "segmentation_benchmark"))
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)

    page_rows = page_benchmark(args.seeds)
    single_rows = single_line_benchmark(args.seeds)
    empty_rows = empty_page_benchmark(args.seeds)

    write_csv(os.path.join(args.output, "pages.csv"), page_rows)
    write_csv(os.path.join(args.output, "single_line_crops.csv"), single_rows)
    write_csv(os.path.join(args.output, "empty_inputs.csv"), empty_rows)

    table = summarise(page_rows, single_rows, empty_rows)
    header = (
        "# Synthetic segmentation benchmark\n\n"
        "SYNTHETIC pages (font-rendered text, known line counts). Tests the document "
        "stages only; says nothing about recognition or real handwriting, and is "
        "optimistic because the thresholds were developed on the same generator.\n\n"
    )

    with open(os.path.join(args.output, "report.md"), "w", encoding="utf-8") as file:
        file.write(header + table + "\n")

    print(header + table)

    return page_rows, single_rows, empty_rows


if __name__ == "__main__":
    main()
