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

# The CNN-BiLSTM model produces 256 CTC time steps.
CTC_TIMESTEPS = 256


# ============================================================
# VOCABULARY
# ============================================================

class CharacterVocabulary:
    """
    Converts characters to integer IDs and back again.
    """

    def __init__(self, dataset):

        characters = set()

        # Collect all characters used by Indonesian samples.
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

        # Total number of classes = characters + CTC blank.
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

            # Ignore the CTC blank token.
            if index == self.blank_id:
                continue

            if index in self.id_to_char:
                result.append(
                    self.id_to_char[index]
                )

        return "".join(result)


# ============================================================
# CTC FEASIBILITY
# ============================================================

def calculate_ctc_required_timesteps(text):
    """
    Calculate the minimum number of CTC time steps required
    for a target text.

    Normally, a target with N characters requires at least N
    time steps.

    However, when two identical characters appear next to each
    other, CTC requires an additional blank between them.

    Example:

        "ab"  -> 2 time steps minimum

        "aa"  -> 3 time steps minimum
                 a, blank, a

    Therefore:

        required_timesteps =
            text_length + adjacent_repeated_characters
    """

    repeated_characters = sum(
        1
        for i in range(1, len(text))
        if text[i] == text[i - 1]
    )

    required_timesteps = (
        len(text)
        + repeated_characters
    )

    return required_timesteps


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

    Processing steps:

        Original image
              ↓
        Convert to grayscale
              ↓
        Resize while preserving aspect ratio
              ↓
        Pad to 128 x 1024
              ↓
        Normalize pixels to [0, 1]
              ↓
        PyTorch tensor

    Output shape:

        [1, 128, 1024]
    """

    # --------------------------------------------------------
    # Convert RGB/RGBA/etc. to grayscale.
    # --------------------------------------------------------

    image = image.convert("L")

    original_width, original_height = image.size

    # --------------------------------------------------------
    # Calculate resized dimensions.
    # --------------------------------------------------------

    scale = target_height / original_height

    new_width = int(
        original_width * scale
    )

    # If the resized image is wider than our maximum width,
    # scale it down so that it fits.
    if new_width > target_width:

        new_width = target_width

        scale = target_width / original_width

        new_height = int(
            original_height * scale
        )

    else:

        new_height = target_height

    # --------------------------------------------------------
    # Resize image.
    # --------------------------------------------------------

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Convert image to NumPy array.
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values:
    #
    #   0   -> black
    #   255 -> white
    #
    # becomes:
    #
    #   0.0 -> black
    #   1.0 -> white
    image_array = image_array / 255.0

    # --------------------------------------------------------
    # Create white padded canvas.
    # --------------------------------------------------------

    padded = np.ones(
        (target_height, target_width),
        dtype=np.float32
    )

    # Center the image vertically.
    y_offset = (
        target_height - new_height
    ) // 2

    # Place resized image on the white canvas.
    padded[
        y_offset:y_offset + new_height,
        :new_width
    ] = image_array

    # --------------------------------------------------------
    # Convert to PyTorch tensor.
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        padded
    )

    # Add channel dimension:
    #
    # [128, 1024]
    #
    # becomes:
    #
    # [1, 128, 1024]
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# DATASET
# ============================================================

class IndonesianHandwritingDataset(Dataset):
    """
    PyTorch Dataset for Indonesian handwriting recognition.

    The dataset performs two filtering operations:

    1. Keep only Indonesian samples.
    2. Remove samples that require more CTC time steps
       than the model can produce.
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

        # ----------------------------------------------------
        # Load the requested dataset split.
        # ----------------------------------------------------

        self.dataset = load_dataset(
            DATASET_NAME,
            split=split
        )

        # ----------------------------------------------------
        # Filter Indonesian samples and CTC-infeasible samples.
        # ----------------------------------------------------

        self.indices = []

        total_indonesian = 0
        excluded_ctc = 0

        for index, language in enumerate(
            self.dataset["language"]
        ):

            # ------------------------------------------------
            # Only keep Indonesian samples.
            # ------------------------------------------------

            if language != "ind":
                continue

            total_indonesian += 1

            # Get target transcription.
            text = self.dataset["text"][index]

            # ------------------------------------------------
            # Determine how many CTC time steps are required.
            # ------------------------------------------------

            required_timesteps = (
                calculate_ctc_required_timesteps(text)
            )

            # ------------------------------------------------
            # Keep sample only if it fits the model output.
            # ------------------------------------------------

            if required_timesteps <= CTC_TIMESTEPS:

                self.indices.append(index)

            else:

                excluded_ctc += 1

        # ----------------------------------------------------
        # Display filtering results.
        # ----------------------------------------------------

        print(
            f"Indonesian samples before CTC filtering: "
            f"{total_indonesian}"
        )

        print(
            f"CTC-infeasible samples excluded: "
            f"{excluded_ctc}"
        )

        print(
            f"Indonesian samples after CTC filtering: "
            f"{len(self.indices)}"
        )

        # ----------------------------------------------------
        # Vocabulary
        # ----------------------------------------------------

        # Build a vocabulary if one was not provided.
        #
        # In train.py, we create the vocabulary once and pass
        # the same vocabulary to the training and validation
        # datasets.
        if vocabulary is None:

            self.vocabulary = CharacterVocabulary(
                self.dataset
            )

        else:

            self.vocabulary = vocabulary

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):

        # Convert filtered index into the original dataset index.
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

        # ----------------------------------------------------
        # Convert text to character IDs.
        # ----------------------------------------------------

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
    # Load training dataset.
    # --------------------------------------------------------

    dataset = IndonesianHandwritingDataset(
        split="train"
    )

    # --------------------------------------------------------
    # Dataset information.
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATASET INFORMATION")
    print("-" * 70)

    print(
        f"Number of usable samples: "
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

    print(
        f"CTC time steps: "
        f"{CTC_TIMESTEPS}"
    )

    # --------------------------------------------------------
    # Test up to 5 samples.
    # --------------------------------------------------------

    number_of_samples = min(
        5,
        len(dataset)
    )

    print("\n" + "-" * 70)
    print("SAMPLE TEST")
    print("-" * 70)

    for i in range(number_of_samples):

        sample = dataset[i]

        image = sample["image"]
        label = sample["label"]
        text = sample["text"]

        print()
        print(
            f"Sample {i}"
        )

        print(
            f"Image shape: "
            f"{tuple(image.shape)}"
        )

        print(
            f"Original text: "
            f"{text}"
        )

        print(
            f"Text length: "
            f"{len(text)}"
        )

        print(
            f"Required CTC time steps: "
            f"{calculate_ctc_required_timesteps(text)}"
        )

        print(
            f"Encoded label length: "
            f"{len(label)}"
        )

        # ----------------------------------------------------
        # Decode the label.
        # ----------------------------------------------------

        decoded = dataset.vocabulary.decode(
            label.tolist()
        )

        print(
            f"Decoded text: "
            f"{decoded}"
        )

        # ----------------------------------------------------
        # Verify encoding/decoding.
        # ----------------------------------------------------

        if decoded == text:

            print(
                "Verification: PASS"
            )

        else:

            print(
                "Verification: FAIL"
            )

    # --------------------------------------------------------
    # Complete.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PYTORCH DATASET TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()