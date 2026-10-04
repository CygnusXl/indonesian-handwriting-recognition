from datasets import load_dataset


DATASET_NAME = "to-be/OpenHand-Synth"


def minimum_ctc_timesteps(text):
    """
    CTC requires an extra timestep when two identical
    characters occur consecutively.

    Example:

        "ab"  -> 2
        "aa"  -> 3
        "aab" -> 4

    because repeated adjacent characters need a blank
    between them.
    """

    if not text:
        return 0

    repeats = 0

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            repeats += 1

    return len(text) + repeats


print("=" * 70)
print("CTC FEASIBILITY ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)

indonesian = dataset.filter(
    lambda example: example["language"] == "ind"
)

print(f"Total Indonesian samples: {len(indonesian)}")

CTC_TIMESTEPS = 256

feasible = 0
infeasible = 0

max_required = 0
max_text_length = 0

longest_examples = []

for example in indonesian:

    text = example["text"]

    text_length = len(text)

    required_timesteps = minimum_ctc_timesteps(text)

    max_text_length = max(
        max_text_length,
        text_length
    )

    max_required = max(
        max_required,
        required_timesteps
    )

    if required_timesteps <= CTC_TIMESTEPS:
        feasible += 1
    else:
        infeasible += 1

        longest_examples.append(
            (
                required_timesteps,
                text_length,
                text
            )
        )


total = len(indonesian)

print("\n" + "-" * 70)
print("RESULTS")
print("-" * 70)

print(f"CTC timesteps available: {CTC_TIMESTEPS}")

print(
    f"Maximum text length: "
    f"{max_text_length}"
)

print(
    f"Maximum required CTC timesteps: "
    f"{max_required}"
)

print(
    f"\nFeasible samples: "
    f"{feasible} / {total} "
    f"({feasible / total * 100:.2f}%)"
)

print(
    f"Infeasible samples: "
    f"{infeasible} / {total} "
    f"({infeasible / total * 100:.2f}%)"
)


if infeasible > 0:

    print("\n" + "-" * 70)
    print("LONGEST INFEASIBLE EXAMPLES")
    print("-" * 70)

    longest_examples.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    for required, length, text in longest_examples[:10]:

        print(
            f"\nRequired timesteps: {required}"
        )

        print(
            f"Text length: {length}"
        )

        print(
            f"Text: {text[:200]}"
        )


print("\n" + "=" * 70)
print("CTC ANALYSIS COMPLETE")
print("=" * 70)