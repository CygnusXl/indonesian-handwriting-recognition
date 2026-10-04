from datasets import load_dataset
from collections import Counter


DATASET = "to-be/OpenHand-Synth"


def analyze_split(split):
    dataset = load_dataset(
        DATASET,
        split=split
    )

    writers = [
        writer
        for writer, language
        in zip(dataset["writer_id"], dataset["language"])
        if language == "ind"
    ]

    return Counter(writers)


def main():
    print("=" * 70)
    print("INDONESIAN WRITER SAMPLE DISTRIBUTION")
    print("=" * 70)

    for split in ["train", "validation", "test"]:

        counts = analyze_split(split)

        print("\n" + "=" * 70)
        print(f"{split.upper()}")
        print("=" * 70)

        print(f"Number of writers: {len(counts)}")
        print(f"Total samples:     {sum(counts.values())}")

        values = list(counts.values())

        print(f"Minimum samples/writer: {min(values)}")
        print(f"Maximum samples/writer: {max(values)}")
        print(f"Average samples/writer: {sum(values) / len(values):.2f}")

        print("\nSample distribution:")

        for writer, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True
        )[:10]:
            print(f"Writer {writer}: {count} samples")

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()