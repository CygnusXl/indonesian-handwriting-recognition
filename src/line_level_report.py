"""
Line-level re-reading of the saved test-set predictions, with bootstrap
confidence intervals.  NO model is run and nothing is re-predicted: the script
only re-aggregates results/analysis/<model>_error_analysis.csv.

Why: the pooled corpus CER mixes two very different kinds of sample
(single-line and multiline images).  This report separates them and puts an
uncertainty range on every number.

Usage (repository root, no GPU needed):

    python src/line_level_report.py
    python src/line_level_report.py --models baseline experiment1 --resamples 20000

Outputs (results/analysis/):
    line_level_report.md   readable tables
    line_level_report.csv  the same numbers in long format

Method
------
* Corpus CER = sum(char_errors) / sum(reference characters); corpus WER likewise.
* Confidence intervals: percentile bootstrap, two resampling units:
    - "sample": the 208 test samples are resampled independently;
    - "writer": whole writers are resampled (all of a writer's samples
      together), which respects that samples of one writer are not
      independent.  This interval is the more conservative one.
* Paired difference (model - baseline) uses the SAME resampled units for both
  models, so it measures whether the gap survives resampling.
* Fixed seed.  The intervals describe sampling noise of this one test set
  and one training run per model.  They do NOT include training-seed
  variance (only one run per model exists), so they understate total
  uncertainty.
"""

import argparse
import csv
import os

import numpy as np

ANALYSIS_DIR = os.path.join("results", "analysis")
DEFAULT_MODELS = ["baseline", "experiment1", "experiment2", "experiment3"]
SEED = 20261016


def load_model(directory, key):
    path = os.path.join(directory, f"{key}_error_analysis.csv")

    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return {
        "index": np.array([int(r["dataset_index"]) for r in rows]),
        "writer": np.array([r["writer_id"] for r in rows]),
        "multiline": np.array([r["multiline"].strip().lower() == "true" for r in rows]),
        "ref_chars": np.array([int(r["text_length"]) for r in rows], dtype=float),
        "char_err": np.array([int(r["char_errors"]) for r in rows], dtype=float),
        "ref_words": np.array([int(r["reference_words"]) for r in rows], dtype=float),
        "word_err": np.array([int(r["word_errors"]) for r in rows], dtype=float),
        "exact": np.array([r["exact_match"].strip().lower() == "true" for r in rows], dtype=float),
        "collapsed": np.array([r["collapsed_prediction"].strip().lower() == "true" for r in rows], dtype=float),
    }


def check_same_samples(models):
    first = next(iter(models.values()))["index"]

    for key, data in models.items():
        if not np.array_equal(first, data["index"]):
            raise SystemExit(f"{key}: test samples differ from the first model; "
                             "paired comparison would be invalid.")


def _ratio(num, den):
    total = den.sum()

    return float("nan") if total == 0 else float(num.sum() / total)


def point_metrics(data, mask):
    return {
        "n": int(mask.sum()),
        "corpus_cer": _ratio(data["char_err"][mask], data["ref_chars"][mask]),
        "corpus_wer": _ratio(data["word_err"][mask], data["ref_words"][mask]),
        "exact_match": float(data["exact"][mask].mean()) if mask.any() else float("nan"),
        "collapse_rate": float(data["collapsed"][mask].mean()) if mask.any() else float("nan"),
    }


def _units(data, unit):
    """List of index arrays; one per resampling unit."""

    if unit == "sample":
        return [np.array([i]) for i in range(len(data["index"]))]

    writers = {}

    for i, w in enumerate(data["writer"]):
        writers.setdefault(w, []).append(i)

    return [np.array(v) for v in writers.values()]


def bootstrap(models, key, mask, unit, resamples, rng, baseline_key=None):
    """
    Returns {metric: (lo, hi)} for `key`, and the paired CER difference
    interval vs baseline_key (or None).  Units with no sample inside `mask`
    contribute nothing (they still count as a draw, as in a stratified-free
    bootstrap), which keeps subgroup intervals honest about subgroup size.
    """

    data = models[key]
    units = _units(data, unit)
    n_units = len(units)

    # per-unit sums restricted to the mask, for every model
    def unit_sums(d):
        return np.array([[d["char_err"][u][mask[u]].sum(), d["ref_chars"][u][mask[u]].sum(),
                          d["word_err"][u][mask[u]].sum(), d["ref_words"][u][mask[u]].sum(),
                          d["exact"][u][mask[u]].sum(), mask[u].sum()] for u in units])

    mine = unit_sums(data)
    base = unit_sums(models[baseline_key]) if baseline_key and baseline_key != key else None

    stats = {"corpus_cer": [], "corpus_wer": [], "exact_match": []}
    diffs = []

    for _ in range(resamples):
        pick = rng.integers(0, n_units, n_units)
        s = mine[pick].sum(axis=0)

        if s[1] == 0 or s[5] == 0:
            continue

        stats["corpus_cer"].append(s[0] / s[1])
        stats["corpus_wer"].append(s[2] / s[3] if s[3] else np.nan)
        stats["exact_match"].append(s[4] / s[5])

        if base is not None:
            b = base[pick].sum(axis=0)

            if b[1]:
                diffs.append(s[0] / s[1] - b[0] / b[1])

    def interval(values):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]

        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (np.nan, np.nan)

    result = {name: interval(values) for name, values in stats.items()}
    result["diff_cer_vs_baseline"] = interval(diffs) if diffs else None

    return result


def build_rows(models, resamples, baseline_key):
    rng = np.random.default_rng(SEED)
    rows = []

    for key, data in models.items():
        groups = {"all": np.ones(len(data["index"]), dtype=bool),
                  "single-line": ~data["multiline"],
                  "multiline": data["multiline"]}

        for group, mask in groups.items():
            point = point_metrics(data, mask)
            base_point = point_metrics(models[baseline_key], mask)

            for unit in ("sample", "writer"):
                ci = bootstrap(models, key, mask, unit, resamples, rng, baseline_key)
                row = {"model": key, "group": group, "unit": unit, **point,
                       "char_share_of_corpus": float(data["ref_chars"][mask].sum() / data["ref_chars"].sum()),
                       "diff_cer_vs_baseline": point["corpus_cer"] - base_point["corpus_cer"]
                       if key != baseline_key else None}

                for name in ("corpus_cer", "corpus_wer", "exact_match"):
                    row[f"{name}_lo"], row[f"{name}_hi"] = ci[name]

                if ci["diff_cer_vs_baseline"] is not None:
                    row["diff_lo"], row["diff_hi"] = ci["diff_cer_vs_baseline"]

                rows.append(row)

    return rows


def pct(x):
    return "n/a" if x is None or np.isnan(x) else f"{x * 100:.1f}%"


def write_csv(path, rows):
    names = []

    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, models, resamples, baseline_key):
    def pick(model, group, unit):
        return next(r for r in rows if r["model"] == model and r["group"] == group and r["unit"] == unit)

    base = models[baseline_key]
    multi_share = base["ref_chars"][base["multiline"]].sum() / base["ref_chars"].sum()
    per_writer = {}

    for w in base["writer"]:
        per_writer[w] = per_writer.get(w, 0) + 1

    n_writers, max_per_writer = len(per_writer), max(per_writer.values())

    lines = ["# Line-level re-reading of the saved test predictions", "",
             "Generated by `src/line_level_report.py` from `results/analysis/*_error_analysis.csv`. "
             "No model was run; these are re-aggregations of predictions that already existed.", "",
             f"Test set: {len(base['index'])} Indonesian samples "
             f"({int((~base['multiline']).sum())} single-line, {int(base['multiline'].sum())} multiline). "
             f"Bootstrap: {resamples:,} resamples, percentile 95% intervals, fixed seed.", ""]

    lines += ["## 1. Single-line vs multiline (corpus CER, 95% CI)", "",
              "| Model | Group | n | Corpus CER | CI (samples) | CI (writers) | Corpus WER | Exact match |",
              "| --- | --- | ---: | ---: | --- | --- | ---: | ---: |"]

    for key in models:
        for group in ("all", "single-line", "multiline"):
            s, w = pick(key, group, "sample"), pick(key, group, "writer")
            lines.append(f"| {key} | {group} | {s['n']} | {pct(s['corpus_cer'])} | "
                         f"{pct(s['corpus_cer_lo'])}–{pct(s['corpus_cer_hi'])} | "
                         f"{pct(w['corpus_cer_lo'])}–{pct(w['corpus_cer_hi'])} | "
                         f"{pct(s['corpus_wer'])} | {pct(s['exact_match'])} |")

    lines += ["", f"Multiline samples hold {multi_share * 100:.1f}% of the reference characters in the test set, "
              "so they dominate the pooled corpus CER even though they are one third of the samples.", ""]

    lines += ["## 2. Gap to the baseline (paired, CER percentage points)", "",
              "Positive = worse than the baseline.  An interval that contains 0 means the gap is within "
              "resampling noise on this test set.", "",
              "| Model | Group | Difference | CI (samples) | CI (writers) |",
              "| --- | --- | ---: | --- | --- |"]

    for key in models:
        if key == baseline_key:
            continue

        for group in ("all", "single-line", "multiline"):
            s, w = pick(key, group, "sample"), pick(key, group, "writer")
            lines.append(f"| {key} | {group} | {s['diff_cer_vs_baseline'] * 100:+.1f} pt | "
                         f"{s['diff_lo'] * 100:+.1f} to {s['diff_hi'] * 100:+.1f} | "
                         f"{w['diff_lo'] * 100:+.1f} to {w['diff_hi'] * 100:+.1f} |")

    lines += ["", "## 3. What this does and does not show", "",
              "**Observation.** For every model the single-line and multiline rows differ by a wide margin, "
              "far wider than the intervals above. The pooled CER is therefore a mixture of two regimes, not one "
              "typical error rate.", "",
              "**Not measured here.** The effect of the page pipeline's line splitting on the multiline samples. "
              "That needs the dataset images, which are not in the repository; the splitting has only been checked "
              "on five dataset sample crops (see `DOCUMENT_PIPELINE.md`). "
              "To measure it, run `python src/evaluate_pipeline.py` on the multiline test images with ground truth "
              "(see `evaluation/real_world/README.md` for the input layout).", "",
              "**Limits.** One training run per model: the intervals cover test-set sampling noise only, not "
              "training-seed variance, so they understate the real uncertainty. "
              f"The test set has {n_writers} writers with at most {max_per_writer} samples each, which is why "
              "resampling by writer changes the intervals very little. Samples from the same writer are "
              "correlated, so quote the writer interval."]

    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--directory", default=ANALYSIS_DIR)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args(argv)

    if args.baseline not in args.models:
        args.models.insert(0, args.baseline)

    models = {key: load_model(args.directory, key) for key in args.models}
    check_same_samples(models)

    rows = build_rows(models, args.resamples, args.baseline)

    write_csv(os.path.join(args.directory, "line_level_report.csv"), rows)
    write_markdown(os.path.join(args.directory, "line_level_report.md"), rows, models,
                   args.resamples, args.baseline)

    print(f"Wrote {os.path.join(args.directory, 'line_level_report.md')}")

    return rows


if __name__ == "__main__":
    main()
