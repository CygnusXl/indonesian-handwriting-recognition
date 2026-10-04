import torch
from torch.utils.data import DataLoader

from dataset import IndonesianHandwritingDataset


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):
    """
    Combine samples into a batch.

    Images all have the same size, so they can be stacked.

    Labels have different lengths, so they are concatenated
    into one tensor. Their individual lengths are stored
    separately.
    """

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    images = torch.stack(
        [
            sample["image"]
            for sample in batch
        ]
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = [
        sample["label"]
        for sample in batch
    ]

    # Keep track of each label's length.
    label_lengths = torch.tensor(
        [
            len(label)
            for label in labels
        ],
        dtype=torch.long
    )

    # CTC can receive concatenated targets.
    labels = torch.cat(labels)

    # --------------------------------------------------------
    # Original text
    # --------------------------------------------------------

    texts = [
        sample["text"]
        for sample in batch
    ]

    return {
        "images": images,
        "labels": labels,
        "label_lengths": label_lengths,
        "texts": texts,
    }


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 70)
    print("DATALOADER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = IndonesianHandwritingDataset(
        split="train"
    )

    print(
        f"\nDataset size: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Get one batch
    # --------------------------------------------------------

    batch = next(iter(dataloader))

    images = batch["images"]
    labels = batch["labels"]
    label_lengths = batch["label_lengths"]
    texts = batch["texts"]

    # --------------------------------------------------------
    # Display batch information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("BATCH INFORMATION")
    print("-" * 70)

    print(
        f"\nImages shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Images dtype: "
        f"{images.dtype}"
    )

    print(
        f"Labels shape: "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Labels dtype: "
        f"{labels.dtype}"
    )

    print(
        f"Label lengths: "
        f"{label_lengths.tolist()}"
    )

    print(
        f"Total label values: "
        f"{len(labels)}"
    )

    print(
        f"Number of texts: "
        f"{len(texts)}"
    )

    # --------------------------------------------------------
    # Verify lengths
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("LABEL VERIFICATION")
    print("-" * 70)

    position = 0

    for i, length in enumerate(
        label_lengths.tolist()
    ):

        sample_labels = labels[
            position:position + length
        ]

        decoded = dataset.vocabulary.decode(
            sample_labels.tolist()
        )

        print(
            f"\nSample {i + 1}:"
        )

        print(
            f"Original: "
            f"{texts[i]}"
        )

        print(
            f"Decoded:  "
            f"{decoded}"
        )

        if decoded == texts[i]:

            print(
                "Verification: PASS"
            )

        else:

            print(
                "Verification: FAIL"
            )

        position += length

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATALOADER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()