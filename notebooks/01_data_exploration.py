"""
OpenHand-Synth Indonesian subset exploration.

This script intentionally uses the Hugging Face Datasets Server filter API
for metadata inspection first, so the complete 8+ GB dataset does not need
to be downloaded just to answer basic dataset questions.

Run:
    python notebooks/01_data_exploration.py

Requirements:
    pip install requests pandas
"""

from collections import Counter
import requests
import pandas as pd

DATASET = "to-be/openhand-synth"
CONFIG = "default"
SPLITS = ["train", "validation", "test"]
BASE = "https://datasets-server.huggingface.co"


def query_split(split: str, where: str = "language='ind'", length: int = 100):
    """Return matching metadata rows and the server's reported total."""
    url = f"{BASE}/filter"
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": split,
        "where": where,
        "offset": 0,
        "length": length,
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload.get("rows", []), payload.get("num_rows_total")


def inspect_split(split: str):
    rows, total = query_split(split)

    print(f"\n=== {split.upper()} ===")
    print(f"Indonesian rows reported by server: {total}")
    print(f"Rows sampled for metadata inspection: {len(rows)}")

    if not rows:
        return pd.DataFrame()

    records = [row["row"] for row in rows]
    df = pd.DataFrame(records)

    useful = [
        column for column in [
            "text", "writer_id", "neatness", "line_width", "color",
            "augmented_noise", "language", "source", "cer", "jaro_winkler"
        ] if column in df.columns
    ]

    print("\nColumns:")
    print(list(df.columns))

    if "text" in df:
        lengths = df["text"].fillna("").astype(str).str.len()
        print("\nSample text length statistics:")
        print(lengths.describe())

    for column in ["source", "color", "augmented_noise", "neatness", "writer_id"]:
        if column in df:
            print(f"\n{column}:")
            print(df[column].value_counts().head(20))

    if "text" in df:
        print("\nSample Indonesian transcriptions:")
        for text in df["text"].head(10):
            print("-", repr(text))

    return df


def main():
    print("OpenHand-Synth Indonesian dataset exploration")
    print("=" * 55)
    print("Dataset:", DATASET)
    print("Purpose: verify Indonesian data before model development.")
    print()

    totals = {}

    for split in SPLITS:
        df = inspect_split(split)

        # The filter endpoint gives us the authoritative count for this query.
        _, total = query_split(split)
        totals[split] = total

    print("\n" + "=" * 55)
    print("INDONESIAN SPLIT SUMMARY")
    print("=" * 55)

    for split, total in totals.items():
        print(f"{split:12s}: {total}")

    if all(isinstance(value, int) for value in totals.values()):
        total = sum(totals.values())
        print(f"{'TOTAL':12s}: {total}")

        for split, count in totals.items():
            percentage = (count / total * 100) if total else 0
            print(f"  {split:10s}: {percentage:.2f}%")

    print("\nNext questions to answer before training:")
    print("1. How many unique writer IDs are represented in each split?")
    print("2. Are writer IDs separated between train/validation/test?")
    print("3. What are the source proportions for Indonesian samples?")
    print("4. How many Indonesian samples contain augmented noise?")
    print("5. What is the text-length distribution?")
    print("6. What image dimensions occur?")
    print("7. Are there enough examples for the planned model and experiments?")

    print("\nNote:")
    print("This script does not claim that OpenHand-Synth contains Indonesian")
    print("university notes. It is synthetic handwriting data with Indonesian text.")


if __name__ == "__main__":
    main()
