"""
IAM handwriting database (English) as line-level training data.

STATUS - READ THIS FIRST
------------------------
This loader has been tested only against small FAKE datasets built in
memory (tests/test_iam_dataset.py). It has NOT been run against the real
IAM data, because the cloud workspace used to write it cannot reach the
dataset. Facts that are therefore UNVERIFIED until you run it:

    * that the Hugging Face mirror named below exists, is reachable and
      needs no login;
    * its column names (the loader auto-detects an image column and a
      text column and fails with the list of real columns if it cannot);
    * its split names and sizes;
    * its licence terms for your use.

Verified from the official IAM page (fki.tic.heia-fr.ch/databases/iam-
handwriting-database): 657 writers, 1,539 pages, 13,353 lines, 115,320
words; "may be used for non-commercial research purpose only";
registration is required for the official download. Cite:

    U. Marti and H. Bunke. The IAM-database: an English sentence
    database for offline handwriting recognition. International Journal
    on Document Analysis and Recognition, 5:39-46, 2002.

If the mirror does not work, register at the official site, download the
line images and transcriptions, and tell Claude which files you have so
the loader can be adapted; `from_pairs()` already accepts any list of
(PIL image, text) pairs.
"""

import torch
from torch.utils.data import Dataset

import htr_common as hc
from photo_augment import augment_line


IAM_DATASET_NAME = "Teklia/IAM-line"   # UNVERIFIED mirror, see above

SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("validation", "val", "valid", "dev"),
    "test": ("test",),
}

IMAGE_COLUMN_CANDIDATES = ("image", "img", "line_image", "line")
TEXT_COLUMN_CANDIDATES = ("text", "transcription", "label", "gt", "sentence")


class IamUnavailableError(RuntimeError):
    """The IAM data could not be loaded; the message says what to do."""


def detect_columns(column_names):
    """
    Return (image_column, text_column) from a list of column names, or
    raise IamUnavailableError that lists the real columns.
    """

    names = list(column_names)
    lowered = {name.lower(): name for name in names}

    image_column = next(
        (lowered[c] for c in IMAGE_COLUMN_CANDIDATES if c in lowered), None
    )
    text_column = next(
        (lowered[c] for c in TEXT_COLUMN_CANDIDATES if c in lowered), None
    )

    if image_column is None or text_column is None:
        raise IamUnavailableError(
            "Could not find an image column and a text column. "
            f"Columns found: {names}. Tell Claude these names and the "
            "loader will be adapted."
        )

    return image_column, text_column


def load_split(split, name=IAM_DATASET_NAME):
    """
    Load one split from Hugging Face. Raises IamUnavailableError with an
    actionable message instead of a raw traceback.
    """

    from datasets import load_dataset

    if split not in SPLIT_ALIASES:
        raise ValueError(f"split must be one of {list(SPLIT_ALIASES)}")

    errors = []

    for alias in SPLIT_ALIASES[split]:
        try:
            return load_dataset(name, split=alias)
        except Exception as error:  # network, gating, missing split ...
            errors.append(f"split '{alias}': {type(error).__name__}: {error}")

    raise IamUnavailableError(
        f"Could not load '{name}' ({split}).\n  "
        + "\n  ".join(errors)
        + "\nIf the dataset is unreachable or gated, register at "
        "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database "
        "(non-commercial research use) and tell Claude which files you "
        "downloaded."
    )


def required_timesteps(text):
    """Minimum CTC length: characters plus a blank between repeats."""

    repeats = sum(1 for a, b in zip(text, text[1:]) if a == b)

    return len(text) + repeats


def normalize_text(text):
    """
    Light, documented normalisation: the Hugging Face / IAM transcripts
    may contain a different quote or whitespace convention. Only
    surrounding whitespace is stripped; characters are never rewritten
    (so the vocabulary reflects the data).
    """

    return " ".join(str(text).split())


def build_vocabulary(texts):
    """Sorted, deterministic character vocabulary from training texts."""

    characters = sorted({c for text in texts for c in text})

    return hc.CheckpointVocabulary({c: i for i, c in enumerate(characters)})


class IamLineDataset(Dataset):
    """
    Line images + transcripts as tensors for the CNN-BiLSTM-CTC model.

    Items are (PIL image, text) pairs. Samples are dropped (and counted
    in `.dropped`) when their transcript:

        * is empty,
        * needs more CTC steps than the model has (`ctc_timesteps`), or
        * contains a character that is not in `vocabulary`.

    `augment=True` applies photo_augment.augment_line (training only).
    `pad_fraction` adds a fixed white border (evaluation under the page
    pipeline's crop margin).
    """

    def __init__(
        self,
        pairs,
        vocabulary,
        image_height=128,
        image_width=1536,
        ctc_timesteps=384,
        augment=False,
        pad_fraction=0.0,
        rng=None,
    ):
        import random

        self.vocabulary = vocabulary
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment
        self.pad_fraction = pad_fraction
        self.rng = rng or random.Random(0)

        self.items = []
        self.dropped = {"empty": 0, "too_long_for_ctc": 0, "unknown_char": 0}

        for image, text in pairs:
            text = normalize_text(text)

            if not text:
                self.dropped["empty"] += 1
            elif required_timesteps(text) > ctc_timesteps:
                self.dropped["too_long_for_ctc"] += 1
            elif any(c not in vocabulary.char_to_id for c in text):
                self.dropped["unknown_char"] += 1
            else:
                self.items.append((image, text))

    @classmethod
    def from_huggingface(cls, hf_dataset, vocabulary, **kwargs):
        """
        Wrap a Hugging Face split lazily: images are decoded only when a
        sample is read, so 10k line images are not held in memory.
        """

        image_column, text_column = detect_columns(hf_dataset.column_names)
        texts = hf_dataset[text_column]
        lazy = _LazyPairs(hf_dataset, image_column, texts)

        return cls(lazy, vocabulary, **kwargs)

    @classmethod
    def from_pairs(cls, pairs, vocabulary, **kwargs):
        return cls(list(pairs), vocabulary, **kwargs)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image, text = self.items[index]

        if callable(image):
            image = image()  # lazy Hugging Face loader

        if isinstance(image, dict) and image.get("bytes"):
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image["bytes"]))

        pil = image

        pil = hc.prepare_user_image(pil)

        if self.augment:
            pil = augment_line(pil, self.rng)

        if self.pad_fraction:
            from photo_augment import pad_like_page_pipeline

            pil = pad_like_page_pipeline(pil, self.pad_fraction)

        tensor = hc.preprocess_image(
            pil, self.image_height, self.image_width
        )

        return {
            "image": tensor,
            "label": torch.tensor(
                self.vocabulary.encode(text), dtype=torch.long
            ),
            "text": text,
        }


class _LazyPairs:
    """Iterates (loader, text) where loader() decodes the image."""

    def __init__(self, hf_dataset, image_column, texts):
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.texts = texts

    def __iter__(self):
        for index, text in enumerate(self.texts):
            yield (
                lambda i=index: self.hf_dataset[i][self.image_column],
                text,
            )
