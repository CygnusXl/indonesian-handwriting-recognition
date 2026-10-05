import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "to-be/OpenHand-Synth"

# Image height remains fixed for all experiments.
IMAGE_HEIGHT = 128

# ------------------------------------------------------------
# BASELINE configuration
# ------------------------------------------------------------

IMAGE_WIDTH = 1024

# The current CNN reduces width by a factor of 4:
#
# 1024
#   ↓ /2
# 512
#   ↓ /2
# 256
#
CTC_TIMESTEPS = IMAGE_WIDTH // 4


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
            for index, character in enumerate(
                characters
            )
        }

        # ID -> Character
        self.id_to_char = {
            index: character
            for character, index
            in self.char_to_id.items()
        }

        self.num_characters = len(
            self.char_to_id
        )

        # CTC blank token.
        self.blank_id = (
            self.num_characters
        )

        # Characters + blank.
        self.num_classes = (
            self.num_characters + 1
        )

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
# CTC FEASIBILITY
# ============================================================

def calculate_ctc_required_timesteps(text):
    """
    Calculate the minimum CTC sequence length required
    for a target text.

    For normal characters:

        required = text length

    For adjacent repeated characters, CTC needs an
    additional blank between them.

    Example:

        "ab" -> 2

        "aa" -> 3

    Therefore:

        required =
            text_length
            +
            adjacent_repeated_characters
    """

    repeated_characters = sum(
        1
        for i in range(
            1,
            len(text)
        )
        if text[i] == text[i - 1]
    )

    return (
        len(text)
        + repeated_characters
    )


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

    The aspect ratio of the original image is preserved.

    Output:

        [1, target_height, target_width]
    """

    # --------------------------------------------------------
    # Convert to grayscale.
    # --------------------------------------------------------

    image = image.convert("L")

    original_width, original_height = (
        image.size
    )

    # --------------------------------------------------------
    # Resize according to target height.
    # --------------------------------------------------------

    scale = (
        target_height
        / original_height
    )

    new_width = int(
        original_width * scale
    )

    # --------------------------------------------------------
    # Prevent width from exceeding target.
    # --------------------------------------------------------

    if new_width > target_width:

        new_width = target_width

        scale = (
            target_width
            / original_width
        )

        new_height = int(
            original_height * scale
        )

    else:

        new_height = target_height

    # --------------------------------------------------------
    # Resize.
    # --------------------------------------------------------

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Convert to NumPy.
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize [0,255] -> [0,1].
    image_array = (
        image_array / 255.0
    )

    # --------------------------------------------------------
    # Create white padded canvas.
    # --------------------------------------------------------

    padded = np.ones(
        (
            target_height,
            target_width
        ),
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
    # Convert to tensor.
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        padded
    )

    # [H,W] -> [1,H,W]
    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# DATASET
# ============================================================

class IndonesianHandwritingDataset(Dataset):
    """
    PyTorch Dataset for Indonesian handwriting recognition.

    The dataset can be configured with different image widths
    and corresponding CTC sequence lengths.

    Baseline:

        image_width = 1024
        ctc_timesteps = 256

    Experiment 1:

        image_width = 1536
        ctc_timesteps = 384
    """

    def __init__(
    self,
    split="train",
    vocabulary=None,
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
    ctc_timesteps=CTC_TIMESTEPS
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.ctc_timesteps = ctc_timesteps

        print(
            f"Loading OpenHand-Synth "
            f"split='{split}'..."
        )

        print(
            f"Image size: "
            f"{self.image_height} x "
            f"{self.image_width}"
        )

        print(
            f"CTC time steps: "
            f"{self.ctc_timesteps}"
        )

        # ----------------------------------------------------
        # Load dataset.
        # ----------------------------------------------------

        self.dataset = load_dataset(
            DATASET_NAME,
            split=split
        )

        # ----------------------------------------------------
        # Filter samples.
        # ----------------------------------------------------

        self.indices = []

        total_indonesian = 0
        excluded_ctc = 0

        for index, language in enumerate(
            self.dataset["language"]
        ):

            # Only Indonesian.
            if language != "ind":
                continue

            total_indonesian += 1

            text = self.dataset[
                "text"
            ][index]

            required_timesteps = (
                calculate_ctc_required_timesteps(
                    text
                )
            )

            # Keep only CTC-feasible samples.
            if (
                required_timesteps
                <= self.ctc_timesteps
            ):

                self.indices.append(
                    index
                )

            else:

                excluded_ctc += 1

        # ----------------------------------------------------
        # Filtering information.
        # ----------------------------------------------------

        print(
            f"Indonesian samples before "
            f"CTC filtering: "
            f"{total_indonesian}"
        )

        print(
            f"CTC-infeasible samples excluded: "
            f"{excluded_ctc}"
        )

        print(
            f"Indonesian samples after "
            f"CTC filtering: "
            f"{len(self.indices)}"
        )

        # ----------------------------------------------------
        # Vocabulary.
        # ----------------------------------------------------

        if vocabulary is None:

            self.vocabulary = (
                CharacterVocabulary(
                    self.dataset
                )
            )

        else:

            self.vocabulary = vocabulary

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index
    ):

        # Translate filtered index to original
        # OpenHand-Synth index.
        dataset_index = (
            self.indices[index]
        )

        sample = self.dataset[
            dataset_index
        ]

        # ----------------------------------------------------
        # Image.
        # ----------------------------------------------------

        image = preprocess_image(
            sample["image"],
            target_height=self.image_height,
            target_width=self.image_width
        )

        # ----------------------------------------------------
        # Text.
        # ----------------------------------------------------

        text = sample["text"]

        # ----------------------------------------------------
        # Encode text.
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
    # Baseline/default configuration.
    # --------------------------------------------------------

    dataset = (
        IndonesianHandwritingDataset(
            split="train"
        )
    )

    # --------------------------------------------------------
    # Dataset information.
    # --------------------------------------------------------

    print()
    print("-" * 70)
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
        f"Image width: "
        f"{dataset.image_width}"
    )

    print(
        f"CTC time steps: "
        f"{dataset.ctc_timesteps}"
    )

    # --------------------------------------------------------
    # Test samples.
    # --------------------------------------------------------

    number_of_samples = min(
        5,
        len(dataset)
    )

    print()
    print("-" * 70)
    print("SAMPLE TEST")
    print("-" * 70)

    for i in range(
        number_of_samples
    ):

        sample = dataset[i]

        image = sample[
            "image"
        ]

        label = sample[
            "label"
        ]

        text = sample[
            "text"
        ]

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

        # Decode.
        decoded = (
            dataset.vocabulary.decode(
                label.tolist()
            )
        )

        print(
            f"Decoded text: "
            f"{decoded}"
        )

        if decoded == text:

            print(
                "Verification: PASS"
            )

        else:

            print(
                "Verification: FAIL"
            )

    print()
    print("=" * 70)
    print("PYTORCH DATASET TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()