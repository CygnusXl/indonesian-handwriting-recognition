"""
Generate report-ready tables and figures from REAL project outputs only.

Inputs (all produced by actual runs):

    results/analysis/<key>_error_analysis.csv   from src/analyze_model.py
    results/analysis/<key>_analysis.json        from src/analyze_model.py
    results/<key>_history.csv                   training histories

Outputs:

    results/tables/*.md and *.csv
    results/figures/*.png

Usage (repository root):

    python src/report_assets.py

    # Preview using the legacy per-sample files already committed for
    # the baseline and Experiment 2 (before everything is re-analysed):
    python src/report_assets.py --legacy --output-dir /tmp/preview

Models without analysis files are skipped and listed; nothing is
invented or interpolated.
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import htr_common as hc  # noqa: E402
import analyze_model as am  # noqa: E402


MODEL_ORDER = [
    "baseline",
    "experiment1",
    "experiment2",
    "experiment3",
    "baseline_extended",  # optional control run
]

SHORT_LABEL = {
    "baseline": "Baseline",
    "experiment1": "Exp 1 (width 1536)",
    "experiment2": "Exp 2 (height 192)",
    "experiment3": "Exp 3 (multiline-balanced)",
    "baseline_extended": "Baseline, longer training",
}

COLORS = {
    "baseline": "#1f77b4",
    "experiment1": "#ff7f0e",
    "experiment2": "#2ca02c",
    "experiment3": "#d62728",
    "baseline_extended": "#7f7f7f",
}

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 0


# ============================================================
# LOADING
# ============================================================

def load_records_csv(path):
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    records = [
        am.make_record(
            {
                "dataset_index": int(r["dataset_index"]),
                "writer_id": r["writer_id"],
                "source": r["source"],
                "neatness": int(r["neatness"]),
                "line_width": int(r["line_width"]),
                "augmented_noise": r["augmented_noise"] == "True",
            },
            r["reference"],
            r["prediction"],
        )
        for r in rows
    ]

    return records


def load_history(path):
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    def number(value):
        return float(value) if value not in ("", None) else None

    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "train_loss": [number(r.get("train_loss")) for r in rows],
        "validation_loss": [number(r.get("validation_loss")) for r in rows],
        "validation_cer": [number(r["validation_cer"]) for r in rows],
    }


def load_models(analysis_dir, results_dir, legacy):
    """Returns {key: {"records", "meta", "history"}} for available models."""

    models = {}
    missing = []

    for key in MODEL_ORDER:
        registry = hc.MODEL_REGISTRY[key]

        if legacy:
            csv_path = os.path.join(
                results_dir, f"{key}_error_analysis.csv"
            )
            json_path = None
        else:
            csv_path = os.path.join(
                analysis_dir, f"{key}_error_analysis.csv"
            )
            json_path = os.path.join(analysis_dir, f"{key}_analysis.json")

        if not os.path.exists(csv_path):
            missing.append(key)
            continue

        meta = {}

        if json_path and os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as file:
                meta = json.load(file)["meta"]

        history = load_history(
            os.path.join(results_dir, f"{key}_history.csv")
        )

        meta.setdefault("image_height", registry["image_height"])
        meta.setdefault("image_width", registry["image_width"])
        meta.setdefault("parameters", None)
        meta.setdefault("mean_ms_per_image", None)

        if meta.get("checkpoint_validation_cer") is None and history:
            meta["checkpoint_validation_cer"] = min(
                v for v in history["validation_cer"] if v is not None
            )

        models[key] = {
            "records": load_records_csv(csv_path),
            "meta": meta,
            "history": history,
        }

    return models, missing


def check_same_samples(models):
    reference_key = next(iter(models))
    reference = [r["dataset_index"] for r in models[reference_key]["records"]]

    for key, model in models.items():
        indices = [r["dataset_index"] for r in model["records"]]

        if indices != reference:
            raise RuntimeError(
                f"{key} was not evaluated on the same test samples as "
                f"{reference_key}; comparison would be invalid."
            )


# ============================================================
# TABLES
# ============================================================

def pct(value, digits=2):
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def write_table(path_base, header, rows, caption):
    os.makedirs(os.path.dirname(path_base), exist_ok=True)

    with open(path_base + ".csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    lines = [f"**{caption}**", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")

    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    with open(path_base + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def pp(delta):
    return f"{delta * 100:+.2f} pp"


def model_comparison_table(models, tables_dir):
    base = models.get("baseline")
    base_overall = (
        am.summarize(base["records"])["overall"] if base else None
    )

    header = [
        "Model",
        "Input (H x W)",
        "Params",
        "Best val CER",
        "Corpus CER",
        "Corpus WER",
        "Exact match",
        "Macro-CER",
        "Collapse",
        "ms/img",
    ]

    rows = []

    for key, model in models.items():
        overall = am.summarize(model["records"])["overall"]
        meta = model["meta"]

        rows.append(
            [
                SHORT_LABEL[key],
                f"{meta['image_height']} x {meta['image_width']}",
                f"{meta['parameters']:,}" if meta["parameters"] else "n/a",
                pct(meta.get("checkpoint_validation_cer")),
                pct(overall["corpus_cer"]),
                pct(overall["corpus_wer"]),
                pct(overall["exact_match"]),
                pct(overall["macro_cer"]),
                f"{overall['collapse_count']}/{overall['samples']}",
                (
                    f"{meta['mean_ms_per_image']:.1f}"
                    if meta["mean_ms_per_image"]
                    else "n/a"
                ),
            ]
        )

    write_table(
        os.path.join(tables_dir, "model_comparison"),
        header,
        rows,
        "Table: model comparison on the common test set "
        "(corpus metrics are primary; macro-CER is secondary).",
    )

    if base_overall:
        delta_rows = []

        for key, model in models.items():
            if key == "baseline":
                continue

            overall = am.summarize(model["records"])["overall"]

            delta_rows.append(
                [
                    SHORT_LABEL[key],
                    pp(overall["corpus_cer"] - base_overall["corpus_cer"]),
                    pp(overall["corpus_wer"] - base_overall["corpus_wer"]),
                    pp(overall["exact_match"] - base_overall["exact_match"]),
                    pp(overall["macro_cer"] - base_overall["macro_cer"]),
                ]
            )

        write_table(
            os.path.join(tables_dir, "delta_vs_baseline"),
            [
                "Model",
                "d Corpus CER",
                "d Corpus WER",
                "d Exact match",
                "d Macro-CER",
            ],
            delta_rows,
            "Table: change relative to the baseline, computed from unrounded "
            "values (positive CER/WER = worse; positive exact match = "
            "better; may differ by 0.01 pp from subtracting rounded "
            "table entries).",
        )


def subgroup_comparison_tables(models, tables_dir):
    for field, name, order in (
        ("multiline", "multiline", [False, True]),
        ("length_category", "length", hc.LENGTH_CATEGORY_ORDER),
        ("augmented_noise", "noise", [False, True]),
        ("source", "source", None),
        ("multiline_x_length", "multiline_x_length", None),
    ):
        tables = {
            key: {
                row["group"]: row
                for row in am.subgroup_table(model["records"], field)
            }
            for key, model in models.items()
        }

        groups = order or sorted(
            {g for table in tables.values() for g in table}, key=str
        )

        header = ["Group", "n"]

        for key in models:
            header += [
                f"{SHORT_LABEL[key]} macro-CER",
                f"{SHORT_LABEL[key]} corpus-CER",
            ]

        rows = []

        for group in groups:
            first = next(
                (t[group] for t in tables.values() if group in t), None
            )

            if first is None:
                continue

            row = [str(group), first["samples"]]

            for key in models:
                cell = tables[key].get(group)
                row += (
                    [pct(cell["macro_cer"]), pct(cell["corpus_cer"])]
                    if cell
                    else ["n/a", "n/a"]
                )

            rows.append(row)

        write_table(
            os.path.join(tables_dir, f"subgroup_{name}"),
            header,
            rows,
            f"Table: CER by {name} subgroup "
            "(macro = mean of per-sample CER; corpus = pooled).",
        )


def bootstrap_table(models, tables_dir):
    """
    Paired bootstrap over TEST SAMPLES for corpus-CER differences
    relative to the baseline. This quantifies test-set sampling
    uncertainty only; it cannot capture training randomness (each
    configuration was trained once).
    """

    if "baseline" not in models or len(models) < 2:
        return

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def arrays(key):
        records = models[key]["records"]
        return (
            np.array([r["char_errors"] for r in records], dtype=float),
            np.array([r["text_length"] for r in records], dtype=float),
            np.array([r["multiline"] for r in records]),
        )

    base_err, lengths, multiline = arrays("baseline")
    n = len(lengths)

    draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))

    def corpus(err, mask, idx):
        sel = mask[idx]
        return (err[idx] * sel).sum(axis=1) / np.maximum(
            (lengths[idx] * sel).sum(axis=1), 1
        )

    masks = {
        "all samples": np.ones(n, dtype=bool),
        "single-line": ~multiline,
        "multiline": multiline,
    }

    rows = []

    for key in models:
        if key == "baseline":
            continue

        err, _, _ = arrays(key)

        for label, mask in masks.items():
            diff = corpus(err, mask, draws) - corpus(base_err, mask, draws)
            observed = (
                (err * mask).sum() / (lengths * mask).sum()
                - (base_err * mask).sum() / (lengths * mask).sum()
            )
            low, high = np.percentile(diff, [2.5, 97.5])

            per_sample = (
                np.array([r["cer"] for r in models[key]["records"]])
                - np.array([r["cer"] for r in models["baseline"]["records"]])
            )[mask]

            rows.append(
                [
                    SHORT_LABEL[key],
                    label,
                    int(mask.sum()),
                    pp(observed),
                    f"[{low * 100:+.2f}, {high * 100:+.2f}] pp",
                    f"{int((per_sample < 0).sum())} / "
                    f"{int((per_sample > 0).sum())} / "
                    f"{int((per_sample == 0).sum())}",
                ]
            )

    write_table(
        os.path.join(tables_dir, "paired_bootstrap_vs_baseline"),
        [
            "Model",
            "Subset",
            "n",
            "d Corpus CER vs baseline",
            "95% bootstrap interval",
            "Samples better / worse / equal (per-sample CER)",
        ],
        rows,
        "Table: paired bootstrap of corpus-CER difference vs baseline "
        f"({BOOTSTRAP_DRAWS} resamples of the test samples, seed "
        f"{BOOTSTRAP_SEED}). Reflects test-set sampling uncertainty only, "
        "NOT training-seed variance (single training run per model).",
    )


def shorten(text, limit=110):
    return text if len(text) <= limit else text[: limit - 1] + "…"


def examples_table(models, tables_dir, count=2):
    if "baseline" not in models:
        return

    records = {
        key: {r["dataset_index"]: r for r in m["records"]}
        for key, m in models.items()
    }

    base = models["baseline"]["records"]
    singles = sorted(
        (r for r in base if not r["multiline"]),
        key=lambda r: (r["cer"], r["dataset_index"]),
    )
    multis = sorted(
        (r for r in base if r["multiline"]),
        key=lambda r: (r["cer"], r["dataset_index"]),
    )

    picks = (
        [("single-line, median", singles[len(singles) // 2])]
        + [("single-line, worst", singles[-1])]
        + [("multiline, best", multis[0])]
        + [("multiline, median", multis[len(multis) // 2])]
        + [("multiline, worst", multis[-1])][:count - 1]
    )

    header = ["Case", "Idx", "Reference"] + [
        f"{SHORT_LABEL[k]} (CER)" for k in models
    ]

    rows = []

    for label, record in picks:
        idx = record["dataset_index"]
        row = [label, idx, shorten(record["reference"]).replace("|", "/")]

        for key in models:
            r = records[key][idx]
            row.append(
                f"{shorten(r['prediction'])} ({r['cer'] * 100:.0f}%)".replace(
                    "|", "/"
                )
            )

        rows.append(row)

    write_table(
        os.path.join(tables_dir, "prediction_examples"),
        header,
        rows,
        "Table: representative reference vs prediction examples "
        "(cases chosen by baseline ranking; same samples for all models; "
        "literal backslash-n marks a line break).",
    )


# ============================================================
# FIGURES
# ============================================================

def save_figure(fig, figures_dir, name):
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_training_curves(models, figures_dir):
    with_history = {k: m for k, m in models.items() if m["history"]}

    if not with_history:
        return

    has_loss = any(
        any(v is not None for v in m["history"]["validation_loss"])
        for m in with_history.values()
    )

    columns = 3 if has_loss else 1
    fig, axes = plt.subplots(1, columns, figsize=(5 * columns, 4))
    axes = np.atleast_1d(axes)

    for key, model in with_history.items():
        h = model["history"]
        label, color = SHORT_LABEL[key], COLORS[key]

        axes[0].plot(
            h["epoch"],
            [v * 100 if v is not None else np.nan for v in h["validation_cer"]],
            marker="o",
            label=label,
            color=color,
        )

        if has_loss:
            axes[1].plot(
                h["epoch"],
                [v if v is not None else np.nan for v in h["train_loss"]],
                marker="o",
                label=label,
                color=color,
            )
            axes[2].plot(
                h["epoch"],
                [v if v is not None else np.nan for v in h["validation_loss"]],
                marker="o",
                label=label,
                color=color,
            )

    axes[0].set_title("Validation CER")
    axes[0].set_ylabel("CER (%)")

    if has_loss:
        axes[1].set_title("Training CTC loss")
        axes[2].set_title("Validation CTC loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=8)

    fig.suptitle(
        "Training behaviour (validation CER on each model's own "
        "validation set)",
        fontsize=10,
    )

    save_figure(fig, figures_dir, "fig_training_curves.png")


def grouped_bars(ax, categories, series, ylabel, title):
    width = 0.8 / max(len(series), 1)
    positions = np.arange(len(categories))

    for i, (key, values) in enumerate(series.items()):
        bars = ax.bar(
            positions + i * width - 0.4 + width / 2,
            [v * 100 for v in values],
            width,
            label=SHORT_LABEL[key],
            color=COLORS[key],
        )

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v * 100:.0f}",
                ha="center",
                fontsize=6,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3)


def figure_model_comparison(models, figures_dir):
    overall = {
        key: am.summarize(m["records"])["overall"]
        for key, m in models.items()
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    grouped_bars(
        ax,
        ["Corpus CER", "Corpus WER", "Exact match"],
        {
            key: [o["corpus_cer"], o["corpus_wer"], o["exact_match"]]
            for key, o in overall.items()
        },
        "%",
        "Common test set: primary metrics (lower CER/WER is better)",
    )
    ax.legend(fontsize=8)

    save_figure(fig, figures_dir, "fig_model_comparison.png")


def figure_error_by_subgroup(models, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (field, categories, title) in zip(
        axes,
        (
            (
                "length_category",
                hc.LENGTH_CATEGORY_ORDER,
                "Macro-CER by text length",
            ),
            ("multiline", [False, True], "Macro-CER: single-line vs multiline"),
        ),
    ):
        series = {}

        for key, model in models.items():
            table = {
                row["group"]: row["macro_cer"]
                for row in am.subgroup_table(model["records"], field)
            }
            series[key] = [table.get(c, 0.0) for c in categories]

        labels = (
            ["Single-line", "Multiline"]
            if field == "multiline"
            else categories
        )

        grouped_bars(ax, labels, series, "Macro-CER (%)", title)

    axes[0].legend(fontsize=8)

    save_figure(fig, figures_dir, "fig_error_by_subgroup.png")


def figure_length_ratio(models, figures_dir):
    columns = len(models)
    fig, axes = plt.subplots(
        1, columns, figsize=(3.6 * columns, 3.6), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, (key, model) in zip(axes, models.items()):
        for flag, color, label in (
            (False, "#4c72b0", "single-line"),
            (True, "#dd8452", "multiline"),
        ):
            subset = [r for r in model["records"] if r["multiline"] == flag]
            ax.scatter(
                [r["text_length"] for r in subset],
                [r["prediction_length"] for r in subset],
                s=10,
                alpha=0.6,
                color=color,
                label=label,
            )

        ax.plot([0, 260], [0, 260], "k--", linewidth=0.8)
        ax.set_title(SHORT_LABEL[key], fontsize=9)
        ax.set_xlabel("Reference length (chars)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Predicted length (chars)")
    axes[0].legend(fontsize=7)

    save_figure(fig, figures_dir, "fig_prediction_length.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", default=os.path.join("results", "analysis")
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Read results/<key>_error_analysis.csv (old per-model "
        "files) instead of results/analysis/. Preview only.",
    )
    args = parser.parse_args()

    models, missing = load_models(
        args.analysis_dir, args.results_dir, args.legacy
    )

    if not models:
        raise SystemExit(
            "No analysed models found. Run: python src/analyze_model.py"
        )

    check_same_samples(models)

    tables_dir = os.path.join(args.output_dir, "tables")
    figures_dir = os.path.join(args.output_dir, "figures")

    model_comparison_table(models, tables_dir)
    subgroup_comparison_tables(models, tables_dir)
    bootstrap_table(models, tables_dir)
    examples_table(models, tables_dir)

    figure_training_curves(models, figures_dir)
    figure_model_comparison(models, figures_dir)
    figure_error_by_subgroup(models, figures_dir)
    figure_length_ratio(models, figures_dir)

    print("Models included:", ", ".join(models))

    if missing:
        print("Models skipped (no analysis files):", ", ".join(missing))

    print(f"Tables:  {tables_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
