from datasets import load_dataset
from collections import Counter


DATASET = "to-be/OpenHand-Synth"


def main():
    print("=" * 70)
    print("INDONESIAN CHARACTER VOCABULARY ANALYSIS")
    print("=" * 70)

    dataset = load_dataset(
        DATASET,
        split="train"
    )

    texts = [
        text
        for text, language
        in zip(dataset["text"], dataset["language"])
        if language == "ind"
    ]

    print(f"\nIndonesian training samples: {len(texts)}")

    character_counts = Counter()

    for text in texts:
        character_counts.update(text)

    characters = sorted(character_counts.keys())

    print(f"\nUnique characters: {len(characters)}")

    print("\nCharacters:")
    print(characters)

    print("\nCharacter frequencies:")
    
    for character, count in character_counts.most_common():
        if character == " ":
            display = "[SPACE]"
        elif character == "\n":
            display = "[NEWLINE]"
        else:
            display = character

        print(f"{display:12} {count}")

    print("\n" + "=" * 70)
    print("VOCABULARY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()