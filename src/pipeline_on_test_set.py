"""
Run the PAGE PIPELINE (experiments A/C/D) on the 208 OpenHand-Synth TEST
images and report single-line vs multiline separately.

This answers: "what does the pipeline's line splitting do to the multiline
numbers?"  The images are SYNTHETIC dataset images, not photographs; this is
not evidence about real handwriting.

STATUS: `summarize` and the grouping are unit-tested (tests/test_pipeline_on_test_set.py)
and rely on evaluate_pipeline.py copying the `group` column of ground_truth.csv into
per_image.csv as `gt_group`.  `export` has NOT been run: it needs the dataset download,
which was not reachable when this was written.  No numbers from this script exist yet.

Step 1 - export the test images + ground truth (needs `datasets`, internet):

    python src/pipeline_on_test_set.py export

Step 2 - run the pipeline (GPU or CPU, a few minutes):

    python src/evaluate_pipeline.py --images evaluation/synthetic_test \
        --output evaluation/synthetic_test/results --experiments A C D

Step 3 - split the results into single-line / multiline:

    python src/pipeline_on_test_set.py summarize

Experiment A here is "one model call on the whole image with the
pipeline's own preprocessing"; it should land close to, but need not equal,
the 68.79% baseline from analyze_model.py (different preprocessing path).
"""

import argparse
import csv
import json
import os

DEFAULT_DIR = os.path.join("evaluation", "synthetic_test")


def export(directory):
    import htr_common as hc
    from dataset import IndonesianHandwritingDataset

    dataset = IndonesianHandwritingDataset(split="test", ctc_timesteps=hc.COMMON_TEST_CTC_TIMESTEPS)

    os.makedirs(directory, exist_ok=True)
    rows = []

    for i in dataset.indices:
        sample = dataset.dataset[i]
        group = "multiline" if hc.is_multiline(sample["text"]) else "single"
        name = f"{group}_{i}.png"
        sample["image"].convert("RGB").save(os.path.join(directory, name))
        rows.append({"image_file": name, "reference_text": sample["text"], "group": group,
                     "dataset_index": i, "writer_id": sample["writer_id"]})

    with open(os.path.join(directory, "ground_truth.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} images to {directory}")


def summarize_rows(rows):
    """
    rows: per_image.csv rows from evaluate_pipeline.py.  Uses the error counts
    it already wrote (char_errors / reference_chars / word_errors /
    reference_words / exact_match) and the copied-through gt_group column.
    Returns {(experiment, group): metrics}, with group "all" added.
    """

    totals = {}

    for row in rows:
        if str(row.get("has_ground_truth")).lower() != "true" or not row.get("reference_chars"):
            continue

        for group in (row.get("gt_group") or "all", "all"):
            t = totals.setdefault((row["experiment"], group), {"n": 0, "ce": 0, "cn": 0, "we": 0, "wn": 0, "em": 0})
            t["n"] += 1
            t["ce"] += float(row["char_errors"])
            t["cn"] += float(row["reference_chars"])
            t["we"] += float(row["word_errors"])
            t["wn"] += float(row["reference_words"])
            t["em"] += str(row.get("exact_match")).lower() == "true"

    return {k: {"n": v["n"], "corpus_cer": v["ce"] / v["cn"] if v["cn"] else None,
                "corpus_wer": v["we"] / v["wn"] if v["wn"] else None,
                "exact_match": v["em"] / v["n"]} for k, v in totals.items()}


def summarize(directory, results="results"):
    path = os.path.join(directory, results, "per_image.csv")

    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    result = summarize_rows(rows)
    lines = ["| Experiment | Group | n | Corpus CER | Corpus WER | Exact match |", "| --- | --- | ---: | ---: | ---: | ---: |"]

    for (experiment, group), m in sorted(result.items()):
        lines.append(f"| {experiment} | {group} | {m['n']} | {m['corpus_cer'] * 100:.1f}% | "
                     f"{m['corpus_wer'] * 100:.1f}% | {m['exact_match'] * 100:.1f}% |")

    text = "\n".join(lines)
    out = os.path.join(directory, results, "by_group.md")

    with open(out, "w", encoding="utf-8") as file:
        file.write("# Pipeline on synthetic test images, by group\n\nSynthetic dataset images, not photographs.\n\n" + text + "\n")

    with open(os.path.join(directory, results, "by_group.json"), "w", encoding="utf-8") as file:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in result.items()}, file, indent=2)

    print(text)
    print(f"\nWrote {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["export", "summarize"])
    parser.add_argument("--directory", default=DEFAULT_DIR)
    parser.add_argument("--results", default="results",
                        help="results sub-folder of --directory to summarize (default: results)")
    args = parser.parse_args(argv)

    export(args.directory) if args.command == "export" else summarize(args.directory, args.results)


if __name__ == "__main__":
    main()
