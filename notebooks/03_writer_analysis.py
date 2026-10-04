from datasets import load_dataset


DATASET = "to-be/OpenHand-Synth"


def get_indonesian_writers(split):
    dataset = load_dataset(
        DATASET,
        split=split
    )

    writers = {
        writer
        for writer, language
        in zip(dataset["writer_id"], dataset["language"])
        if language == "ind"
    }

    return writers


def main():
    print("=" * 70)
    print("INDONESIAN WRITER DISTRIBUTION ANALYSIS")
    print("=" * 70)

    train_writers = get_indonesian_writers("train")
    validation_writers = get_indonesian_writers("validation")
    test_writers = get_indonesian_writers("test")

    print("\nWriter counts:")
    print(f"Train:       {len(train_writers)}")
    print(f"Validation:  {len(validation_writers)}")
    print(f"Test:        {len(test_writers)}")

    train_validation = train_writers & validation_writers
    train_test = train_writers & test_writers
    validation_test = validation_writers & test_writers

    all_three = (
        train_writers
        & validation_writers
        & test_writers
    )

    print("\nWriter overlap:")
    print(
        f"Train ∩ Validation: "
        f"{len(train_validation)}"
    )

    print(
        f"Train ∩ Test: "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ Test: "
        f"{len(validation_test)}"
    )

    print(
        f"All three splits: "
        f"{len(all_three)}"
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()