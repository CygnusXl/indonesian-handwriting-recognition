"""
Unified test-set evaluation + error analysis for ANY project model.

This replaces the per-experiment copies (error_analysis.py,
error_analysis_experiment2.py, evaluate*.py) with one implementation
that uses the shared code in htr_common.py, so every model is evaluated
with identical decoding, metrics and subgroup definitions.

Usage (from the repository root, GPU recommended):

    python src/analyze_model.py --model experiment3
    python src/analyze_model.py --model all

Outputs (written to results/analysis/ by default), per model key:

    <key>_error_analysis.csv      one row per test sample
    <key>_subgroups.csv           long-format subgroup table
    <key>_writers.csv             per-writer table
    <key>_analysis.json           everything the report scripts need
    <key>_analysis_summary.txt    human-readable summary

Every model is evaluated on the SAME common test set: Indonesian test
samples whose CTC requirement fits in 256 time steps (208 samples),
whatever the model's own training width.

Metric labelling (never mixed):
    corpus_cer / corpus_wer : sum of edit distances / sum of reference
                              characters (words)            [PRIMARY]
    exact_match             : fraction of identical transcripts [PRIMARY]
    macro_cer               : mean of per-sample CERs  [SECONDARY, used
                              for subgroup diagnostics]
"""

import argparse
import csv
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

import torch

import htr_common as hc


DEFAULT_OUTPUT_DIR = os.path.join("results", "analysis")

# (field in the record, human-readable dimension name)
SUBGROUP_DIMENSIONS = [
    ("multiline", "Multiline"),
    ("length_category", "Text length"),
    ("augmented_noise", "Augmented noise"),
    ("source", "Text source"),
    ("neatness_category", "Neatness (confounded)"),
    ("line_width", "Pen line width"),
    ("multiline_x_length", "Multiline x text length"),
]

CSV_COLUMNS = [
    # Same leading columns / order as the legacy error-analysis CSVs.
    "dataset_index",
    "writer_id",
    "source",
    "neatness",
    "neatness_category",
    "line_width",
    "augmented_noise",
    "multiline",
    "text_length",
    "length_category",
    "prediction_length",
    "prediction_ratio",
    "collapsed_prediction",
    "reference",
    "prediction",
    "cer",
    "exact_match",
    # New columns.
    "char_errors",
    "word_errors",
    "reference_words",
]


# ============================================================
# PURE FUNCTIONS (no dataset / GPU; unit-testable)
# ============================================================

def make_record(raw, reference, prediction):
    """
    Build one per-sample record from raw OpenHand-Synth metadata
    (`raw`: mapping with writer_id, source, neatness, line_width,
    augmented_noise, and `dataset_index`) and the model output.
    """

    reference_length = len(reference)
    prediction_length = len(prediction)

    char_errors = hc.char_distance(reference, prediction)
    word_errors = hc.word_distance(reference, prediction)

    prediction_ratio = (
        prediction_length / reference_length if reference_length else 0.0
    )

    multiline = hc.is_multiline(reference)
    category = hc.length_category(reference_length)

    return {
        "dataset_index": raw["dataset_index"],
        "writer_id": raw["writer_id"],
        "source": raw["source"],
        "neatness": raw["neatness"],
        "neatness_category": hc.neatness_category(raw["neatness"]),
        "line_width": raw["line_width"],
        "augmented_noise": raw["augmented_noise"],
        "multiline": multiline,
        "text_length": reference_length,
        "length_category": category,
        "prediction_length": prediction_length,
        "prediction_ratio": prediction_ratio,
        "collapsed_prediction": hc.is_collapsed(
            reference_length, prediction_length
        ),
        "reference": reference,
        "prediction": prediction,
        "cer": (
            char_errors / reference_length if reference_length else 0.0
        ),
        "exact_match": reference == prediction,
        "char_errors": char_errors,
        "word_errors": word_errors,
        "reference_words": len(reference.split()),
    }


def _group_value(record, field):
    if field == "multiline_x_length":
        kind = "multiline" if record["multiline"] else "single-line"
        return f"{kind} / {record['length_category']}"

    return record[field]


def group_stats(records):
    """Aggregate statistics for a list of records (one subgroup)."""

    n = len(records)
    total_chars = sum(r["text_length"] for r in records)
    total_words = sum(r["reference_words"] for r in records)

    return {
        "samples": n,
        "corpus_cer": (
            sum(r["char_errors"] for r in records) / total_chars
            if total_chars
            else 0.0
        ),
        "corpus_wer": (
            sum(r["word_errors"] for r in records) / total_words
            if total_words
            else 0.0
        ),
        "macro_cer": sum(r["cer"] for r in records) / n if n else 0.0,
        "exact_match": (
            sum(r["exact_match"] for r in records) / n if n else 0.0
        ),
        "collapse_count": sum(r["collapsed_prediction"] for r in records),
        "collapse_rate": (
            sum(r["collapsed_prediction"] for r in records) / n
            if n
            else 0.0
        ),
        "mean_prediction_ratio": (
            sum(r["prediction_ratio"] for r in records) / n if n else 0.0
        ),
    }


def subgroup_table(records, field):
    groups = defaultdict(list)

    for record in records:
        groups[_group_value(record, field)].append(record)

    rows = []

    for name, members in groups.items():
        row = {"group": name}
        row.update(group_stats(members))
        rows.append(row)

    if field == "length_category":
        order = {n: i for i, n in enumerate(hc.LENGTH_CATEGORY_ORDER)}
        rows.sort(key=lambda r: order.get(r["group"], 99))
    else:
        rows.sort(key=lambda r: str(r["group"]))

    return rows


def writer_table(records):
    groups = defaultdict(list)

    for record in records:
        groups[record["writer_id"]].append(record)

    rows = []

    for writer, members in groups.items():
        row = {"writer_id": writer}
        row.update(group_stats(members))
        rows.append(row)

    rows.sort(key=lambda r: (-r["macro_cer"], str(r["writer_id"])))

    return rows


def select_examples(records, per_group=3):
    """
    Deterministic representative examples for the report:
    best / median / worst single-line and multiline samples.
    """

    def ordered(subset):
        return sorted(
            subset, key=lambda r: (r["cer"], r["dataset_index"])
        )

    examples = {}

    for name, subset in (
        ("single_line", [r for r in records if not r["multiline"]]),
        ("multiline", [r for r in records if r["multiline"]]),
    ):
        ranked = ordered(subset)

        if not ranked:
            continue

        middle = len(ranked) // 2

        examples[f"{name}_best"] = ranked[:per_group]
        examples[f"{name}_median"] = ranked[
            max(0, middle - per_group // 2):
            max(0, middle - per_group // 2) + per_group
        ]
        examples[f"{name}_worst"] = list(reversed(ranked[-per_group:]))

    return examples


def summarize(records):
    """Overall + subgroup summary dictionary (JSON-serialisable)."""

    references = [r["reference"] for r in records]
    predictions = [r["prediction"] for r in records]

    overall = hc.corpus_metrics(references, predictions)
    overall["collapse_count"] = sum(
        r["collapsed_prediction"] for r in records
    )
    overall["collapse_rate"] = (
        overall["collapse_count"] / len(records) if records else 0.0
    )
    overall["empty_predictions"] = sum(
        1 for r in records if r["prediction_length"] == 0
    )
    overall["mean_prediction_ratio"] = (
        sum(r["prediction_ratio"] for r in records) / len(records)
        if records
        else 0.0
    )

    subgroups = {
        field: {
            "title": title,
            "rows": subgroup_table(records, field),
        }
        for field, title in SUBGROUP_DIMENSIONS
    }

    writers = writer_table(records)
    writer_macro = [w["macro_cer"] for w in writers]

    writer_summary = {
        "writers": len(writers),
        "samples_per_writer_max": max(
            (w["samples"] for w in writers), default=0
        ),
        "macro_cer_min": min(writer_macro, default=0.0),
        "macro_cer_median": (
            statistics.median(writer_macro) if writer_macro else 0.0
        ),
        "macro_cer_max": max(writer_macro, default=0.0),
    }

    return {
        "overall": overall,
        "subgroups": subgroups,
        "writers_summary": writer_summary,
        "writers": writers,
    }


# ============================================================
# OUTPUT
# ============================================================

def write_records_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({c: record[c] for c in CSV_COLUMNS})


def write_subgroups_csv(path, model_key, summary):
    columns = [
        "model",
        "dimension",
        "group",
        "samples",
        "corpus_cer",
        "corpus_wer",
        "macro_cer",
        "exact_match",
        "collapse_count",
        "collapse_rate",
        "mean_prediction_ratio",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for field, block in summary["subgroups"].items():
            for row in block["rows"]:
                writer.writerow(
                    {
                        "model": model_key,
                        "dimension": block["title"],
                        **{c: row[c] for c in columns[2:]},
                    }
                )


def write_writers_csv(path, summary):
    columns = [
        "writer_id",
        "samples",
        "corpus_cer",
        "macro_cer",
        "exact_match",
        "collapse_count",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in summary["writers"]:
            writer.writerow({c: row[c] for c in columns})


def format_summary_text(model_key, meta, summary, examples):
    overall = summary["overall"]
    lines = []

    def add(text=""):
        lines.append(text)

    add(f"ERROR ANALYSIS: {meta['label']} ({model_key})")
    add("=" * 70)
    add(f"Checkpoint:        {meta['checkpoint']}")
    add(f"Checkpoint SHA256: {meta['checkpoint_sha256']}")
    add(f"Checkpoint epoch:  {meta['checkpoint_epoch']}")
    add(
        "Best validation CER (from checkpoint): "
        f"{meta['checkpoint_validation_cer']}"
    )
    add(f"Parameters:        {meta['parameters']:,}")
    add(
        f"Input:             {meta['image_height']} x {meta['image_width']}"
        f" (CTC time steps {meta['ctc_timesteps']})"
    )
    add(
        f"Test set:          {overall['samples']} common samples "
        f"(CTC-feasible at {hc.COMMON_TEST_CTC_TIMESTEPS} time steps)"
    )
    add(
        f"Inference speed:   {meta['mean_ms_per_image']:.1f} ms/image "
        f"on {meta['device']} (batch size {meta['batch_size']})"
    )
    add()
    add("PRIMARY METRICS (corpus-level)")
    add("-" * 70)
    add(f"Corpus CER:  {overall['corpus_cer'] * 100:.2f}%")
    add(f"Corpus WER:  {overall['corpus_wer'] * 100:.2f}%")
    add(f"Exact match: {overall['exact_match'] * 100:.2f}%")
    add()
    add("SECONDARY METRICS")
    add("-" * 70)
    add(f"Macro-CER (mean of per-sample CER): {overall['macro_cer'] * 100:.2f}%")
    add(
        "Severe prediction collapse: "
        f"{overall['collapse_count']}/{overall['samples']} "
        f"({overall['collapse_rate'] * 100:.2f}%)"
    )
    add(f"Empty predictions: {overall['empty_predictions']}")
    add(
        "Mean prediction/reference length ratio: "
        f"{overall['mean_prediction_ratio']:.3f}"
    )

    for field, block in summary["subgroups"].items():
        add()
        add(block["title"].upper())
        add("-" * 70)
        add(
            f"{'group':<34}{'n':>5}{'cCER%':>8}{'mCER%':>8}"
            f"{'cWER%':>8}{'EM%':>6}{'coll':>6}"
        )

        for row in block["rows"]:
            add(
                f"{str(row['group']):<34}{row['samples']:>5}"
                f"{row['corpus_cer'] * 100:>8.2f}"
                f"{row['macro_cer'] * 100:>8.2f}"
                f"{row['corpus_wer'] * 100:>8.2f}"
                f"{row['exact_match'] * 100:>6.1f}"
                f"{row['collapse_count']:>6}"
            )

    ws = summary["writers_summary"]
    add()
    add("WRITERS")
    add("-" * 70)
    add(
        f"{ws['writers']} writers in the test set "
        f"(max {ws['samples_per_writer_max']} samples per writer)."
    )
    add(
        "Per-writer macro-CER: "
        f"min {ws['macro_cer_min'] * 100:.1f}% / "
        f"median {ws['macro_cer_median'] * 100:.1f}% / "
        f"max {ws['macro_cer_max'] * 100:.1f}%"
    )
    add(
        "Most writers have very few test samples; per-writer values are "
        "descriptive only."
    )

    add()
    add("REPRESENTATIVE EXAMPLES (reference -> prediction, CER)")
    add("-" * 70)

    for name, subset in examples.items():
        add(f"[{name}]")
        for record in subset:
            add(f"  idx={record['dataset_index']}  CER={record['cer']:.3f}")
            add(f"  REF:  {record['reference']}")
            add(f"  PRED: {record['prediction']}")
        add()

    return "\n".join(lines) + "\n"


def save_outputs(output_dir, model_key, meta, records):
    os.makedirs(output_dir, exist_ok=True)

    summary = summarize(records)
    examples = select_examples(records)

    paths = {
        "csv": os.path.join(output_dir, f"{model_key}_error_analysis.csv"),
        "subgroups": os.path.join(output_dir, f"{model_key}_subgroups.csv"),
        "writers": os.path.join(output_dir, f"{model_key}_writers.csv"),
        "json": os.path.join(output_dir, f"{model_key}_analysis.json"),
        "txt": os.path.join(
            output_dir, f"{model_key}_analysis_summary.txt"
        ),
    }

    write_records_csv(paths["csv"], records)
    write_subgroups_csv(paths["subgroups"], model_key, summary)
    write_writers_csv(paths["writers"], summary)

    payload = {
        "model": model_key,
        "meta": meta,
        "overall": summary["overall"],
        "subgroups": summary["subgroups"],
        "writers_summary": summary["writers_summary"],
        "examples": examples,
    }

    with open(paths["json"], "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    with open(paths["txt"], "w", encoding="utf-8") as file:
        file.write(format_summary_text(model_key, meta, summary, examples))

    return summary, paths


# ============================================================
# MODEL RUN (needs `datasets`, a checkpoint, ideally a GPU)
# ============================================================

def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_model(model_key, checkpoint_path, device, batch_size, split):
    # Imported lazily so the pure functions above need no `datasets`.
    from dataset import IndonesianHandwritingDataset

    registry = hc.MODEL_REGISTRY[model_key]

    print(f"\nLoading {registry['label']} from {checkpoint_path} ...")

    model, vocabulary, config, checkpoint = hc.load_model(
        checkpoint_path, device
    )

    expected = {
        k: registry[k]
        for k in ("architecture", "image_height", "image_width")
    }
    found = {k: config[k] for k in expected}

    if expected != found:
        raise RuntimeError(
            f"Checkpoint configuration {found} does not match the "
            f"registered configuration {expected} for '{model_key}'."
        )

    dataset = IndonesianHandwritingDataset(
        split=split,
        vocabulary=vocabulary,
        image_height=config["image_height"],
        image_width=config["image_width"],
        # Common test set for every model.
        ctc_timesteps=hc.COMMON_TEST_CTC_TIMESTEPS,
    )

    records = []
    inference_seconds = 0.0
    measured_images = 0

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        samples = [dataset[i] for i in range(start, end)]

        images = torch.stack([s["image"] for s in samples]).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        began = time.perf_counter()

        with torch.no_grad():
            logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - began

        # First batch is warm-up (cuDNN autotune, allocation): excluded.
        if start > 0:
            inference_seconds += elapsed
            measured_images += len(samples)

        predictions = hc.greedy_decode(logits, vocabulary)

        for offset, prediction in enumerate(predictions):
            position = start + offset
            original_index = dataset.indices[position]
            raw = dataset.dataset[original_index]

            records.append(
                make_record(
                    {
                        "dataset_index": original_index,
                        "writer_id": raw["writer_id"],
                        "source": raw["source"],
                        "neatness": raw["neatness"],
                        "line_width": raw["line_width"],
                        "augmented_noise": raw["augmented_noise"],
                    },
                    raw["text"],
                    prediction,
                )
            )

        if (start // batch_size) % 10 == 0:
            print(f"  {end}/{len(dataset)} samples", flush=True)

    meta = {
        "label": registry["label"],
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": hc.file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_cer": checkpoint.get("validation_cer"),
        "parameters": hc.count_parameters(model),
        "architecture": config["architecture"],
        "image_height": config["image_height"],
        "image_width": config["image_width"],
        "ctc_timesteps": config["ctc_timesteps"],
        "split": split,
        "common_test_ctc_timesteps": hc.COMMON_TEST_CTC_TIMESTEPS,
        "device": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "cpu"
        ),
        "batch_size": batch_size,
        "mean_ms_per_image": (
            1000.0 * inference_seconds / measured_images
            if measured_images
            else float("nan")
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }

    return meta, records


def main():
    parser = argparse.ArgumentParser(
        description="Unified test-set evaluation and error analysis."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(hc.MODEL_REGISTRY) + ["all"],
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the registered checkpoint path "
        "(single --model only).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.checkpoint and args.model == "all":
        parser.error("--checkpoint requires a single --model")

    device = resolve_device(args.device)
    keys = (
        list(hc.MODEL_REGISTRY) if args.model == "all" else [args.model]
    )

    print(f"Device: {device}")

    for key in keys:
        checkpoint_path = (
            args.checkpoint or hc.MODEL_REGISTRY[key]["checkpoint"]
        )

        if not os.path.exists(checkpoint_path):
            print(f"\nSKIPPING {key}: checkpoint not found: {checkpoint_path}")
            continue

        meta, records = run_model(
            key, checkpoint_path, device, args.batch_size, args.split
        )

        summary, paths = save_outputs(
            args.output_dir, key, meta, records
        )

        overall = summary["overall"]

        print(f"\n{meta['label']}: {overall['samples']} samples")
        print(f"  Corpus CER  {overall['corpus_cer'] * 100:.2f}%")
        print(f"  Corpus WER  {overall['corpus_wer'] * 100:.2f}%")
        print(f"  Exact match {overall['exact_match'] * 100:.2f}%")
        print(f"  Macro-CER   {overall['macro_cer'] * 100:.2f}%")
        print(f"  Written: {paths['txt']}")


if __name__ == "__main__":
    main()
