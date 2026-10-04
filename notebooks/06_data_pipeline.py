from datasets import load_dataset
from PIL import Image
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "to-be/OpenHand-Synth"

# Special value used by CTC.
# This is NOT an actual character.
CTC_BLANK = "<BLANK>"


# ============================================================
# LOAD CHARACTER VOCABULARY
# ============================================================

def build_vocabulary(dataset):
    """
    Build a character-to-ID mapping from the Indonesian
    training data.
    """

    characters = set()

    for text, language in zip(dataset["text"], dataset["language"]):
        if language == "ind":
            characters.update(text)

    # Sort so the vocabulary is deterministic.
    characters = sorted(characters)

    # Character -> integer ID
    char_to_id = {
        character: index
        for index, character in enumerate(characters)
    }

    # Integer ID -> character
    id_to_char = {
        index: character
        for character, index in char_to_id.items()
    }

    return char_to_id, id_to_char


# ============================================================
# TEXT ENCODING
# ============================================================

def encode_text(text, char_to_id):
    """
    Convert text into a list of integer character IDs.
    """

    return [
        char_to_id[character]
        for character in text
    ]


# ============================================================
# TEXT DECODING
# ============================================================

def decode_text(ids, id_to_char):
    """
    Convert integer character IDs back into text.
    """

    return "".join(
        id_to_char[index]
        for index in ids
        if index in id_to_char
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Basic image preprocessing.

    For now we:
    1. Convert the image to grayscale.
    2. Convert it to a NumPy array.
    3. Normalize pixel values to [0, 1].
    """

    image = image.convert("L")

    image_array = np.array(image, dtype=np.float32)

    image_array = image_array / 255.0

    return image_array


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATA PIPELINE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET,
        split="train"
    )

    print(f"Total training samples: {len(dataset)}")

    # --------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------

    print("\nBuilding Indonesian character vocabulary...")

    char_to_id, id_to_char = build_vocabulary(dataset)

    print(f"Number of characters: {len(char_to_id)}")

    print("\nCharacter vocabulary:")

    for character, index in char_to_id.items():

        if character == " ":
            display = "[SPACE]"
        elif character == "\n":
            display = "[NEWLINE]"
        else:
            display = character

        print(f"{index:3} -> {display}")

    # --------------------------------------------------------
    # Select Indonesian samples
    # --------------------------------------------------------

    print("\nSelecting Indonesian samples...")

    indonesian_indices = [
        index
        for index, language in enumerate(dataset["language"])
        if language == "ind"
    ]

    print(
        f"Indonesian training samples: "
        f"{len(indonesian_indices)}"
    )

    # --------------------------------------------------------
    # Test first few samples
    # --------------------------------------------------------

    print("\nTesting preprocessing...")

    for sample_number, dataset_index in enumerate(
        indonesian_indices[:5],
        start=1
    ):

        sample = dataset[dataset_index]

        image = sample["image"]
        text = sample["text"]

        # --------------------------------------------
        # Image
        # --------------------------------------------

        processed_image = preprocess_image(image)

        # --------------------------------------------
        # Text
        # --------------------------------------------

        encoded_text = encode_text(
            text,
            char_to_id
        )

        decoded_text = decode_text(
            encoded_text,
            id_to_char
        )

        # --------------------------------------------
        # Display results
        # --------------------------------------------

        print("\n" + "-" * 70)
        print(f"SAMPLE {sample_number}")
        print("-" * 70)

        print(f"Original text:")
        print(text)

        print("\nEncoded text:")
        print(encoded_text)

        print("\nDecoded text:")
        print(decoded_text)

        print("\nOriginal image:")
        print(f"Size: {image.size}")
        print(f"Mode: {image.mode}")

        print("\nProcessed image:")
        print(f"Shape: {processed_image.shape}")
        print(f"Dtype: {processed_image.dtype}")
        print(
            f"Pixel range: "
            f"{processed_image.min():.3f} "
            f"to "
            f"{processed_image.max():.3f}"
        )

        # --------------------------------------------
        # Verify text conversion
        # --------------------------------------------

        if text == decoded_text:
            print("\nText encoding test: PASS")
        else:
            print("\nText encoding test: FAIL")

    # --------------------------------------------------------
    # CTC information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CTC INFORMATION")
    print("=" * 70)

    print(f"\nCharacter classes: {len(char_to_id)}")
    print(f"CTC blank token: {CTC_BLANK}")
    print(
        f"Total model output classes: "
        f"{len(char_to_id) + 1}"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA PIPELINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()