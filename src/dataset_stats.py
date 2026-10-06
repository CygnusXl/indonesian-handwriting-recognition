"""
Dataset characterisation for the report (Indonesian OpenHand-Synth).

Needs network access to Hugging Face (downloads/caches the dataset).

    python src/dataset_stats.py

Outputs:
    results/tables/dataset_split_summary.md / .csv
    results/figures/fig_dataset_distribution.png

Everything is computed from the actual dataset; nothing is typed in.
Statistics are computed on ALL Indonesian samples of each split and also
on the CTC-feasible subset actually used for training/testing.
"""

import argparse
import csv
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import htr_common as hc  # noqa: E402

DATASET_NAME = "to-be/OpenHand-Synth"
SPLITS = ["train", "validation", "test"]


def required_timesteps(text):
    repeats = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])
    return len(text) + repeats


def collect(split):
    from datasets import load_dataset

    data = load_dataset(DATASET_NAME, split=split)
    rows = []

    for i, language in enumerate(data["language"]):
        if language != "ind":
            continue

        text = data["text"][i]
        rows.append(
            {
                "index": i,
                "text": text,
                "length": len(text),
                "multiline": hc.is_multiline(text),
                "source": data["source"][i],
                "writer_id": data["writer_id"][i],
                "noise": bool(data["augmented_noise"][i]),
                "neatness": data["neatness"][i],
                "color": data["color"][i],
                "feasible": required_timesteps(text)
                <= hc.COMMON_TEST_CTC_TIMESTEPS,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    per_split = {split: collect(split) for split in SPLITS}

    tables = os.path.join(args.output_dir, "tables")
    figures = os.path.join(args.output_dir, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

    header = [
        "split", "ind_samples", "ctc_feasible_256", "excluded",
        "multiline", "multiline_%", "noise_%", "writers",
        "mean_chars", "max_chars",
    ]
    rows = []

    for split, data in per_split.items():
        n = len(data)
        feasible = sum(r["feasible"] for r in data)
        multi = sum(r["multiline"] for r in data)
        rows.append([
            split, n, feasible, n - feasible, multi,
            f"{100 * multi / n:.1f}", f"{100 * sum(r['noise'] for r in data) / n:.1f}",
            len({r["writer_id"] for r in data}),
            f"{sum(r['length'] for r in data) / n:.1f}",
            max(r["length"] for r in data),
        ])

    with open(os.path.join(tables, "dataset_split_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    md = ["**Table: Indonesian OpenHand-Synth splits (computed from the dataset).**", "",
          "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]

    train = per_split["train"]
    md += ["", "Text sources (train): " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(r["source"] for r in train).items()))]

    with open(os.path.join(tables, "dataset_split_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for split in SPLITS:
        axes[0].hist([r["length"] for r in per_split[split]], bins=30, alpha=0.5, label=split)
    axes[0].axvline(30, color="k", ls=":", lw=0.8)
    axes[0].axvline(60, color="k", ls=":", lw=0.8)
    axes[0].axvline(100, color="k", ls=":", lw=0.8)
    axes[0].set_title("Text length (characters); dotted = length categories")
    axes[0].set_xlabel("characters")
    axes[0].legend()

    for ax, key, title in ((axes[1], "source", "Text source (train)"),
                           (axes[2], "multiline", "Single-line vs multiline")):
        if key == "source":
            counts = Counter(r["source"] for r in train)
            ax.bar(list(counts), list(counts.values()))
            ax.tick_params(axis="x", rotation=30)
        else:
            x = range(len(SPLITS))
            single = [sum(not r["multiline"] for r in per_split[s]) for s in SPLITS]
            multi = [sum(r["multiline"] for r in per_split[s]) for s in SPLITS]
            ax.bar(x, single, label="single-line")
            ax.bar(x, multi, bottom=single, label="multiline")
            ax.set_xticks(list(x))
            ax.set_xticklabels(SPLITS)
            ax.legend()
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(os.path.join(figures, "fig_dataset_distribution.png"), dpi=200)
    print("Written dataset_split_summary and fig_dataset_distribution.png")


if __name__ == "__main__":
    main()
