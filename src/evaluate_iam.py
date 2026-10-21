"""
Evaluate one or more checkpoints on the IAM test split (English lines).

STATUS: the evaluation logic is tested on CPU with fake data and a tiny
random model (tests/test_evaluate_iam.py). No IAM result exists until
you run this on the real data.

What it reports, per checkpoint and per border setting:

    * corpus CER (primary) and corpus WER over all test lines;
    * line exact-match rate;
    * 95 % bootstrap confidence intervals over lines (resampling whole
      lines, 10 000 draws), and, when several checkpoints are given, the
      PAIRED difference of each one against the first.

Border settings
---------------
  --pad 0      the raw IAM line image;
  --pad 0.25   the same image with a white border of 0.25 x height on
               every side. That is what the page pipeline's crop margin
               produces in the app, so this is the closer match to real
               use. Report both; do not pick the better one silently.

Limits you should state when you report results:
  * IAM test lines are clean scans of form-based writing, NOT phone
    photos. A good number here does not mean good photo accuracy.
  * No test line is dropped for being long or having rare characters;
    such lines simply count as errors, which keeps the numbers honest.
  * The old Indonesian model has a different character set; its
    characters that IAM uses but it lacks always count as errors.
    That is a real property of that model, not a bug.

Usage (repository root):

    python src/evaluate_iam.py --checkpoint checkpoints/iam_best.pt \\
        --checkpoint models/final_model.pt

Outputs: results/iam_eval/<tag>_pad<x>.csv (per line),
results/iam_eval/summary.md and summary.json.
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch

import htr_common as hc
import iam_dataset as iam
from photo_augment import pad_like_page_pipeline

OUTPUT_DIR = os.path.join("results", "iam_eval")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261021


def read_test_pairs(name=iam.IAM_DATASET_NAME, limit=None):
    """Load the test split as (loader, text) pairs; drops only empties."""

    split = iam.load_split("test", name)

    if limit:
        split = split.select(range(min(limit, len(split))))

    image_column, text_column = iam.detect_columns(split.column_names)
    texts = split[text_column]

    pairs = []

    for index, text in enumerate(texts):
        text = iam.normalize_text(text)

        if text:
            pairs.append((split[index][image_column], text))

    return pairs


@torch.no_grad()
def recognize_pairs(
    model, vocabulary, config, pairs, pad_fraction, device, batch_size=16
):
    """Return the recognised text for every (PIL image, text) pair."""

    outputs = []

    for start in range(0, len(pairs), batch_size):
        tensors = []

        for image, _ in pairs[start:start + batch_size]:
            image = hc.prepare_user_image(image)

            if pad_fraction:
                image = pad_like_page_pipeline(image, pad_fraction)

            tensors.append(
                hc.preprocess_image(
                    image, config["image_height"], config["image_width"]
                )
            )

        logits = model(torch.stack(tensors).to(device))

        outputs.extend(
            text for text, _ in hc.greedy_decode_with_confidence(
                logits, vocabulary
            )
        )

    return outputs


def line_statistics(references, hypotheses):
    """Per-line arrays used for point estimates and the bootstrap."""

    rows = []

    for reference, hypothesis in zip(references, hypotheses):
        rows.append(
            (
                hc.char_distance(reference, hypothesis),
                len(reference),
                hc.word_distance(reference, hypothesis),
                len(reference.split()),
                int(reference == hypothesis),
            )
        )

    return np.array(rows, dtype=np.float64)


def _rates(stats):
    return (
        stats[:, 0].sum() / max(stats[:, 1].sum(), 1),
        stats[:, 2].sum() / max(stats[:, 3].sum(), 1),
        stats[:, 4].mean(),
    )


def bootstrap_intervals(stats, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """95 % percentile intervals for (CER, WER, exact match) over lines."""

    rng = np.random.default_rng(seed)
    n = len(stats)
    samples = np.empty((draws, 3))

    for i in range(draws):
        samples[i] = _rates(stats[rng.integers(0, n, n)])

    return np.percentile(samples, [2.5, 97.5], axis=0)


def paired_cer_difference(stats_a, stats_b, draws=BOOTSTRAP_DRAWS,
                          seed=BOOTSTRAP_SEED):
    """
    CER(b) - CER(a) with a 95 % interval, resampling the SAME lines for
    both models. Negative means b is better.
    """

    rng = np.random.default_rng(seed)
    n = len(stats_a)
    diffs = np.empty(draws)

    for i in range(draws):
        index = rng.integers(0, n, n)
        diffs[i] = _rates(stats_b[index])[0] - _rates(stats_a[index])[0]

    point = _rates(stats_b)[0] - _rates(stats_a)[0]

    return point, np.percentile(diffs, [2.5, 97.5])


def evaluate_checkpoint(
    path, pairs, pad_fractions, device, batch_size=16, draws=BOOTSTRAP_DRAWS
):
    """Returns {pad: dict(hypotheses, stats, cer, wer, exact, intervals)}."""

    model, vocabulary, config, checkpoint = hc.load_model(path, device)
    references = [text for _, text in pairs]
    results = {}

    for pad in pad_fractions:
        hypotheses = recognize_pairs(
            model, vocabulary, config, pairs, pad, device, batch_size
        )
        stats = line_statistics(references, hypotheses)
        cer, wer, exact = _rates(stats)

        results[pad] = {
            "references": references,
            "hypotheses": hypotheses,
            "stats": stats,
            "cer": cer,
            "wer": wer,
            "exact": exact,
            "intervals": bootstrap_intervals(stats, draws),
            "config": config,
            "training_data": checkpoint.get("training_data"),
        }

    return results


def pct(x):
    return f"{100 * x:.2f}%"


def summary_markdown(evaluations, pad_fractions, lines, draws):
    """evaluations: list of (path, results-by-pad)."""

    out = [
        "# IAM test-set evaluation",
        "",
        f"{lines} test lines. 95 % intervals: {draws} bootstrap draws over "
        "lines. IAM test lines are clean scans, not phone photos.",
        "",
    ]

    for pad in pad_fractions:
        out += [
            f"## Border {pad} x height",
            "",
            "| Checkpoint | CER | WER | Exact match |",
            "|---|---|---|---|",
        ]

        first = None

        for path, results in evaluations:
            r = results[pad]
            (c_lo, w_lo, e_lo), (c_hi, w_hi, e_hi) = r["intervals"]
            out.append(
                f"| {os.path.basename(path)} | "
                f"{pct(r['cer'])} ({pct(c_lo)}-{pct(c_hi)}) | "
                f"{pct(r['wer'])} ({pct(w_lo)}-{pct(w_hi)}) | "
                f"{pct(r['exact'])} ({pct(e_lo)}-{pct(e_hi)}) |"
            )

        out.append("")

        first_path, first_results = evaluations[0]

        for path, results in evaluations[1:]:
            point, (lo, hi) = paired_cer_difference(
                first_results[pad]["stats"], results[pad]["stats"],
                draws,
            )
            out.append(
                f"Paired CER difference, {os.path.basename(path)} minus "
                f"{os.path.basename(first_path)}: {100 * point:+.2f} points "
                f"(95 % interval {100 * lo:+.2f} to {100 * hi:+.2f})."
            )

        out.append("")

    return "\n".join(out)


def summary_dict(evaluations, pad_fractions, lines, dataset, draws):
    """Machine-readable summary; export_final_model copies from it."""

    checkpoints = {}

    for path, results in evaluations:
        pads = {}

        for pad in pad_fractions:
            r = results[pad]
            (c_lo, w_lo, e_lo), (c_hi, w_hi, e_hi) = r["intervals"]
            pads[str(pad)] = {
                "samples": int(len(r["references"])),
                "corpus_cer": float(r["cer"]),
                "corpus_wer": float(r["wer"]),
                "exact_match": float(r["exact"]),
                "corpus_cer_ci95": [float(c_lo), float(c_hi)],
                "corpus_wer_ci95": [float(w_lo), float(w_hi)],
                "exact_match_ci95": [float(e_lo), float(e_hi)],
            }

        checkpoints[os.path.basename(path)] = {
            "sha256": hc.file_sha256(path),
            "training_data": results[pad_fractions[0]]["training_data"],
            "pads": pads,
        }

    return {
        "dataset": dataset,
        "test_lines": lines,
        "bootstrap_draws": draws,
        "checkpoints": checkpoints,
    }


def write_line_csv(path, results_for_pad):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["reference", "hypothesis", "char_distance"])

        for ref, hyp, stat in zip(
            results_for_pad["references"],
            results_for_pad["hypotheses"],
            results_for_pad["stats"],
        ):
            writer.writerow([ref, hyp, int(stat[0])])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="repeat for several; differences are against the first",
    )
    parser.add_argument(
        "--pad", type=float, action="append",
        help="border as a fraction of height (default: 0 and 0.25)",
    )
    parser.add_argument("--dataset", default=iam.IAM_DATASET_NAME)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    pad_fractions = args.pad or [0.0, 0.25]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        pairs = read_test_pairs(args.dataset, args.limit)
    except iam.IamUnavailableError as error:
        raise SystemExit(f"\nERROR: {error}")

    print(f"Device: {device}; test lines: {len(pairs)}")

    evaluations = []

    for path in args.checkpoint:
        print(f"Evaluating {path} ...", flush=True)
        results = evaluate_checkpoint(
            path, pairs, pad_fractions, device, args.batch_size, args.draws
        )
        evaluations.append((path, results))

        tag = os.path.splitext(os.path.basename(path))[0]

        for pad in pad_fractions:
            write_line_csv(
                os.path.join(args.output_dir, f"{tag}_pad{pad}.csv"),
                results[pad],
            )

    text = summary_markdown(evaluations, pad_fractions, len(pairs), args.draws)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(
        os.path.join(args.output_dir, "summary.md"), "w", encoding="utf-8"
    ) as file:
        file.write(text + "\n")

    with open(
        os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(
            summary_dict(
                evaluations, pad_fractions, len(pairs), args.dataset,
                args.draws,
            ),
            file,
            indent=2,
        )

    print()
    print(text)


if __name__ == "__main__":
    random.seed(0)
    main()
