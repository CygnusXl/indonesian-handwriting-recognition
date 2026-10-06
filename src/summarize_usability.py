"""
Summarise REAL usability-questionnaire responses.

Input: evaluation/usability_responses.csv (one row per participant; see
evaluation/usability_responses_template.csv). Nothing is generated or
assumed: if the file has no rows, the script says so and stops.

Computes:
    * System Usability Scale (SUS) score per participant and overall
      (Brooke, 1996): odd items contribute (response - 1), even items
      (5 - response), summed and multiplied by 2.5 -> 0..100.
    * mean / median / min / max of the custom 1-5 items.
    * task success counts.

Usage: python src/summarize_usability.py
"""

import argparse
import csv
import os
import statistics

SUS_ITEMS = [f"sus_{i}" for i in range(1, 11)]
CUSTOM_ITEMS = [
    "usefulness_1_5",
    "text_quality_1_5",
    "would_use_for_notes_1_5",
    "trust_in_output_1_5",
]


def sus_score(row):
    total = 0

    for number in range(1, 11):
        value = int(row[f"sus_{number}"])

        if not 1 <= value <= 5:
            raise ValueError(f"SUS item {number} must be 1-5, got {value}")

        total += (value - 1) if number % 2 == 1 else (5 - value)

    return total * 2.5


def describe(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join("evaluation", "usability_responses.csv"))
    parser.add_argument("--output", default=os.path.join("evaluation", "results", "usability_summary.md"))
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(
            f"{args.input} not found. Collect responses first "
            "(evaluation/PROTOCOL.md, Part B)."
        )

    with open(args.input, encoding="utf-8", newline="") as file:
        rows = [r for r in csv.DictReader(file) if r.get("participant_id")]

    if not rows:
        raise SystemExit("No participant rows found; nothing to summarise.")

    scores = [sus_score(r) for r in rows]
    lines = [
        f"Participants: {len(rows)}",
        "",
        f"SUS score: mean {statistics.mean(scores):.1f}, "
        f"median {statistics.median(scores):.1f}, "
        f"min {min(scores):.1f}, max {max(scores):.1f}",
        "",
        "Per participant SUS: "
        + ", ".join(f"{r['participant_id']}={s:.1f}" for r, s in zip(rows, scores)),
        "",
        "| Item | n | mean | median | min | max |",
        "|---|---|---|---|---|---|",
    ]

    for item in CUSTOM_ITEMS:
        values = [int(r[item]) for r in rows if r.get(item)]

        if values:
            d = describe(values)
            lines.append(
                f"| {item} | {d['n']} | {d['mean']:.2f} | {d['median']} | {d['min']} | {d['max']} |"
            )

    if "tasks_completed" in rows[0]:
        lines += ["", "Tasks completed (as recorded): "
                  + ", ".join(f"{r['participant_id']}={r['tasks_completed']}" for r in rows)]

    lines += ["", f"NOTE: n={len(rows)}. With so few participants these are "
              "descriptive observations, not statistically generalisable results."]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
