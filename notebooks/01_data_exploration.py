from datasets import load_dataset
import matplotlib.pyplot as plt

DATASET = "to-be/OpenHand-Synth"


def main():
    print("Loading OpenHand-Synth...")

    dataset = load_dataset(
        DATASET,
        split="train"
    )

    print(f"Total training samples: {len(dataset)}")

    print("\nFiltering Indonesian samples...")

    indonesian = dataset.filter(
        lambda example: example["language"] == "ind"
    )

    print(f"Indonesian training samples: {len(indonesian)}")

    print("\nFirst 5 Indonesian samples:")

    for i in range(min(5, len(indonesian))):
        sample = indonesian[i]

        print("\n" + "=" * 50)
        print(f"Sample {i}")
        print("=" * 50)

        print("Text:")
        print(sample["text"])

        print("Writer ID:", sample["writer_id"])
        print("Neatness:", sample["neatness"])
        print("Line width:", sample["line_width"])
        print("Color:", sample["color"])
        print("Noise:", sample["augmented_noise"])
        print("Source:", sample["source"])
        print("CER:", sample["cer"])

    print("\nCreating image preview...")

    number_to_show = min(5, len(indonesian))

    fig, axes = plt.subplots(
        number_to_show,
        1,
        figsize=(12, 3 * number_to_show)
    )

    if number_to_show == 1:
        axes = [axes]

    for i in range(number_to_show):
        sample = indonesian[i]

        axes[i].imshow(sample["image"])
        axes[i].axis("off")

        axes[i].set_title(
            f"Writer {sample['writer_id']} | "
            f"Neatness {sample['neatness']} | "
            f"Noise {sample['augmented_noise']}\n"
            f"{sample['text'][:100]}"
        )

    plt.tight_layout()

    output_file = "results/indonesian_samples.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nImage preview saved to:")
    print(output_file)


if __name__ == "__main__":
    main()