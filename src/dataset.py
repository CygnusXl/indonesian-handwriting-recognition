import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "to-be/OpenHand-Synth"

# Height that every image will be resized to.
IMAGE_HEIGHT = 128

# Maximum width after resizing/padding.
IMAGE_WIDTH = 1024


# ============================================================
# VOCABULARY
# ============================================================

class CharacterVocabulary:
    """
    Converts characters to integer IDs and back again.
    """

    def __init__(self, dataset):

        characters = set()

        for text, language in zip(
            dataset["text"],
            dataset["language"]
        ):
            if language == "ind":
                characters.update(text)

        # Deterministic ordering.
        characters = sorted(characters)

        # Character -> ID
        self.char_to_id = {
            character: index
            for index, character in enumerate(characters)
        }

        # ID -> Character
        self.id_to_char = {
            index: character
            for character, index in self.char_to_id.items()
        }

        self.num_characters = len(self.char_to_id)

        # CTC needs one additional blank class.
        self.blank_id = self.num_characters

        self.num_classes = self.num_characters + 1

    def encode(self, text):
        """
        Convert text into integer IDs.
        """

        return [
            self.char_to_id[character]
            for character in text
        ]

    def decode(self, ids):
        """
        Convert integer IDs back into text.
        """

        result = []

        for index in ids:

            # Ignore CTC blank.
            if index == self.blank_id:
                continue

            if index in self.id_to_char:
                result.append(
                    self.id_to_char[index]
                )

        return "".join(result)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image,
    target_height=IMAGE_HEIGHT,
    target_width=IMAGE_WIDTH
):
    """
    Convert an image into a fixed-size grayscale tensor.

    The original aspect ratio is preserved.

    Steps:
        RGB
        ↓
        grayscale
        ↓
        resize while preserving aspect ratio
        ↓
        pad remaining width
        ↓
        normalize to [0, 1]
        ↓
        PyTorch tensor
    """

    # --------------------------------------------------------
    # Convert RGB → grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    original_width, original_height = image.size

    # --------------------------------------------------------
    # Calculate new dimensions
    # --------------------------------------------------------

    scale = target_height / original_height

    new_width = int(
        original_width * scale
    )

    # Prevent images from becoming wider than target width.
    if new_width > target_width:

        new_width = target_width

        scale = target_width / original_width

        new_height = int(
            original_height * scale
        )

    else:

        new_height = target_height

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixels to [0, 1].
    image_array = image_array / 255.0

    # --------------------------------------------------------
    # Create padded image
    # --------------------------------------------------------

    padded = np.ones(
        (target_height, target_width),
        dtype=np.float32
    )

    # Center vertically.
    y_offset = (
        target_height - new_height
    ) // 2

    padded[
        y_offset:y_offset + new_height,
        :new_width
    ] = image_array

    # --------------------------------------------------------
    # Convert to PyTorch tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        padded
    )

    # Add channel dimension.
    #
    # From:
    #     [height, width]
    #
    # To:
    #     [1, height, width]
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# DATASET
# ============================================================

class IndonesianHandwritingDataset(Dataset):
    """
    PyTorch Dataset for Indonesian handwriting recognition.
    """

    def __init__(
        self,
        split="train",
        vocabulary=None
    ):

        print(
            f"Loading OpenHand-Synth "
            f"split='{split}'..."
        )

        self.dataset = load_dataset(
            DATASET_NAME,
            split=split
        )

        # Keep only Indonesian samples.
        self.indices = [
            index
            for index, language
            in enumerate(self.dataset["language"])
            if language == "ind"
        ]

        print(
            f"Indonesian samples: "
            f"{len(self.indices)}"
        )

        # Build vocabulary if one wasn't supplied.
        if vocabulary is None:

            self.vocabulary = CharacterVocabulary(
                self.dataset
            )

        else:

            self.vocabulary = vocabulary

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        dataset_index = self.indices[index]

        sample = self.dataset[
            dataset_index
        ]

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = preprocess_image(
            sample["image"]
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text = sample["text"]

        label = self.vocabulary.encode(
            text
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return {
            "image": image,
            "label": label,
            "text": text,
        }


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 70)
    print("PYTORCH DATASET TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = IndonesianHandwritingDataset(
        split="train"
    )

    print("\nDataset information")
    print("-" * 70)

    print(
        f"Number of samples: "
        f"{len(dataset)}"
    )

    print(
        f"Characters: "
        f"{dataset.vocabulary.num_characters}"
    )

    print(
        f"CTC classes: "
        f"{dataset.vocabulary.num_classes}"
    )

    print(
        f"CTC blank ID: "
        f"{dataset.vocabulary.blank_id}"
    )

    # --------------------------------------------------------
    # Test samples
    # --------------------------------------------------------

    for i in range(5):

        sample = dataset[i]

        image = sample["image"]
        label = sample["label"]
        text = sample["text"]

        print("\n" + "-" * 70)
        print(f"SAMPLE {i + 1}")
        print("-" * 70)

        print(f"Text:")
        print(text)

        print(
            f"\nImage tensor shape: "
            f"{tuple(image.shape)}"
        )

        print(
            f"Image dtype: "
            f"{image.dtype}"
        )

        print(
            f"Image range: "
            f"{image.min():.3f} "
            f"to "
            f"{image.max():.3f}"
        )

        print(
            f"Label length: "
            f"{len(label)}"
        )

        print(
            f"Encoded label:"
        )

        print(
            label.tolist()
        )

        # Decode again.
        decoded = dataset.vocabulary.decode(
            label.tolist()
        )

        print(
            f"\nDecoded text:"
        )

        print(decoded)

        if decoded == text:
            print(
                "\nLabel encoding test: PASS"
            )
        else:
            print(
                "\nLabel encoding test: FAIL"
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PYTORCH DATASET TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()