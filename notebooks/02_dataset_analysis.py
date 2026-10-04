from datasets import load_dataset
from collections import Counter
import statistics


DATASET = "to-be/OpenHand-Synth"

SPLITS = [
    "train",
    "validation",
    "test",
]


def analyze_split(split):
    print("\n" + "=" * 70)
    print(f"ANALYZING {split.upper()} SPLIT")
    print("=" * 70)

    dataset = load_dataset(
        DATASET,
        split=split,
    )

    print(f"Total samples: {len(dataset)}")

    # Find Indonesian samples
    indonesian_indices = [
        i
        for i, language in enumerate(dataset["language"])
        if language == "ind"
    ]

    indonesian = dataset.select(indonesian_indices)

    print(f"Indonesian samples: {len(indonesian)}")

    if len(indonesian) == 0:
        print("No Indonesian samples found.")
        return

    # ---------------------------------------------------------
    # Writers
    # ---------------------------------------------------------

    writers = indonesian["writer_id"]
    unique_writers = set(writers)

    print(f"Unique writers: {len(unique_writers)}")

    # ---------------------------------------------------------
    # Neatness
    # ---------------------------------------------------------

    neatness = indonesian["neatness"]

    print("\nNeatness:")
    print(f"  Minimum: {min(neatness)}")
    print(f"  Maximum: {max(neatness)}")
    print(f"  Mean:    {statistics.mean(neatness):.2f}")
    print(f"  Median:  {statistics.median(neatness):.2f}")

    # ---------------------------------------------------------
    # Noise
    # ---------------------------------------------------------

    noise = indonesian["augmented_noise"]

    noise_count = sum(noise)
    no_noise_count = len(noise) - noise_count

    print("\nNoise:")
    print(f"  With noise:    {noise_count}")
    print(f"  Without noise: {no_noise_count}")
    print(f"  Noise ratio:   {noise_count / len(noise) * 100:.2f}%")

    # ---------------------------------------------------------
    # Colors
    # ---------------------------------------------------------

    colors = Counter(indonesian["color"])

    print("\nInk colors:")

    for color, count in colors.items():
        percentage = count / len(indonesian) * 100
        print(f"  {color}: {count} ({percentage:.2f}%)")

    # ---------------------------------------------------------
    # Sources
    # ---------------------------------------------------------

    sources = Counter(indonesian["source"])

    print("\nText sources:")

    for source, count in sources.items():
        percentage = count / len(indonesian) * 100
        print(f"  {source}: {count} ({percentage:.2f}%)")

    # ---------------------------------------------------------
    # Text lengths
    # ---------------------------------------------------------

    text_lengths = [
        len(text)
        for text in indonesian["text"]
    ]

    print("\nText length (characters):")
    print(f"  Minimum: {min(text_lengths)}")
    print(f"  Maximum: {max(text_lengths)}")
    print(f"  Mean:    {statistics.mean(text_lengths):.2f}")
    print(f"  Median:  {statistics.median(text_lengths):.2f}")

    # ---------------------------------------------------------
    # Image dimensions
    # ---------------------------------------------------------

    first_image = indonesian[0]["image"]

    print("\nImage dimensions:")
    print(f"  Width:  {first_image.width}")
    print(f"  Height: {first_image.height}")
    print(f"  Mode:   {first_image.mode}")

    # ---------------------------------------------------------
    # Sample texts
    # ---------------------------------------------------------

    print("\nExample Indonesian texts:")

    for i in range(min(3, len(indonesian))):
        print(f"\nExample {i + 1}:")
        print(indonesian[i]["text"])


def main():

    print("=" * 70)
    print("OPENHAND-SYNTH INDONESIAN DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nDataset: {DATASET}")

    for split in SPLITS:
        analyze_split(split)

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()