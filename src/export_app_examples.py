"""
Export a few Indonesian OpenHand-Synth TEST images (with their true
text) for the app's "Use an example" option.

    python src/export_app_examples.py --count 6

Writes app/examples/*.png and app/examples/examples.csv. Selection is
deterministic: evenly spaced single-line test samples from the common
test set. Check the dataset licence before committing the images
(app/examples/* is git-ignored by default).
"""

import argparse
import csv
import os

import htr_common as hc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output-dir", default=os.path.join("app", "examples"))
    args = parser.parse_args()

    from dataset import IndonesianHandwritingDataset

    dataset = IndonesianHandwritingDataset(
        split="test", ctc_timesteps=hc.COMMON_TEST_CTC_TIMESTEPS
    )

    singles = [
        i for i in dataset.indices
        if not hc.is_multiline(dataset.dataset[i]["text"])
    ]
    step = max(1, len(singles) // args.count)
    chosen = singles[::step][: args.count]

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    for i in chosen:
        sample = dataset.dataset[i]
        name = f"test_{i}.png"
        sample["image"].convert("RGB").save(os.path.join(args.output_dir, name))
        rows.append({"file": name, "reference": sample["text"]})

    with open(os.path.join(args.output_dir, "examples.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "reference"])
        w.writeheader()
        w.writerows(rows)

    print(f"Exported {len(rows)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
