import os
import random
import time
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset

from dataset import (
    CharacterVocabulary,
    IndonesianHandwritingDataset,
    DATASET_NAME,
)
from dataloader import collate_fn
from model import CNNBiLSTM
from train import DEVICE, train_one_epoch, validate


EXPERIMENT_NAME = "experiment3"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1024
CTC_TIMESTEPS = 256

BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

TARGET_MULTILINE_RATIO = 0.50
SEED = 42

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "experiment3_best.pt",
)
HISTORY_PATH = os.path.join(
    "results",
    "experiment3_history.csv",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_multiline_text(text):
    return "\n" in text or "\\n" in text


def create_multiline_balanced_sampler(
    train_dataset,
    target_multiline_ratio=0.50,
    seed=42,
):
    """
    Give approximately target_multiline_ratio of each
    training epoch to multiline samples.

    Sampling is with replacement so the minority group
    receives greater training exposure.
    """

    multiline_positions = []
    single_line_positions = []

    for position, dataset_index in enumerate(
        train_dataset.indices
    ):
        text = train_dataset.dataset["text"][dataset_index]

        if is_multiline_text(text):
            multiline_positions.append(position)
        else:
            single_line_positions.append(position)

    if not multiline_positions:
        raise RuntimeError(
            "No multiline training samples were found."
        )

    if not single_line_positions:
        raise RuntimeError(
            "No single-line training samples were found."
        )

    weights = torch.empty(
        len(train_dataset),
        dtype=torch.double,
    )

    multiline_weight = (
        target_multiline_ratio
        / len(multiline_positions)
    )

    single_line_weight = (
        (1.0 - target_multiline_ratio)
        / len(single_line_positions)
    )

    for position in multiline_positions:
        weights[position] = multiline_weight

    for position in single_line_positions:
        weights[position] = single_line_weight

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )

    print()
    print("Multiline sampling analysis:")
    print(
        f"  Single-line samples: "
        f"{len(single_line_positions)}"
    )
    print(
        f"  Multiline samples: "
        f"{len(multiline_positions)}"
    )
    print(
        f"  Natural multiline ratio: "
        f"{len(multiline_positions) / len(train_dataset):.2%}"
    )
    print(
        f"  Target multiline ratio: "
        f"{target_multiline_ratio:.2%}"
    )
    print(
        f"  Expected multiline draws/epoch: "
        f"{len(train_dataset) * target_multiline_ratio:.1f}"
    )

    return sampler


def save_history(history):
    os.makedirs(
        "results",
        exist_ok=True,
    )

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_cer",
            ]
        )

        for epoch in range(
            len(history["train_loss"])
        ):
            writer.writerow(
                [
                    epoch + 1,
                    history["train_loss"][epoch],
                    history["validation_loss"][epoch],
                    history["validation_cer"][epoch],
                ]
            )


def main():
    set_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT 3: MULTILINE-BALANCED TRAINING")
    print("=" * 70)

    print()
    print("Research question:")
    print(
        "Does increasing training exposure to multiline "
        "handwriting improve recognition without changing "
        "the baseline architecture or image representation?"
    )

    print()
    print("Only changed variable:")
    print("Training sample distribution")

    print()
    print(
        "Target: approximately 50% single-line / "
        "50% multiline training exposure"
    )

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()
    print("Loading training split for vocabulary...")

    raw_train_dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    vocabulary = CharacterVocabulary(
        raw_train_dataset
    )

    print(
        f"Characters: {vocabulary.num_characters}"
    )
    print(
        f"CTC classes: {vocabulary.num_classes}"
    )

    print()
    print("Loading training dataset...")

    train_dataset = IndonesianHandwritingDataset(
        split="train",
        vocabulary=vocabulary,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        ctc_timesteps=CTC_TIMESTEPS,
    )

    print()
    print("Loading validation dataset...")

    validation_dataset = IndonesianHandwritingDataset(
        split="validation",
        vocabulary=vocabulary,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        ctc_timesteps=CTC_TIMESTEPS,
    )

    print()
    print(
        f"Training samples: {len(train_dataset)}"
    )
    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    train_sampler = create_multiline_balanced_sampler(
        train_dataset=train_dataset,
        target_multiline_ratio=TARGET_MULTILINE_RATIO,
        seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print(
        f"Training batches: {len(train_loader)}"
    )
    print(
        f"Validation batches: {len(validation_loader)}"
    )

    print()
    print("Creating baseline CNN-BiLSTM...")

    model = CNNBiLSTM(
        num_classes=vocabulary.num_classes
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Model parameters: {parameter_count:,}"
    )

    test_batch = next(
        iter(train_loader)
    )

    test_images = test_batch[
        "images"
    ].to(DEVICE)

    with torch.no_grad():
        test_logits = model(
            test_images
        )

    expected_shape = (
        256,
        BATCH_SIZE,
        vocabulary.num_classes,
    )

    print(
        f"Input shape: {tuple(test_images.shape)}"
    )
    print(
        f"Output shape: {tuple(test_logits.shape)}"
    )

    if tuple(test_logits.shape) != expected_shape:
        raise RuntimeError(
            f"Expected output {expected_shape}, "
            f"got {tuple(test_logits.shape)}"
        )

    print("Sequence-length test: PASS")

    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_cer": [],
    }

    best_cer = float("inf")

    print()
    print("=" * 70)
    print("STARTING EXPERIMENT 3")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            vocabulary=vocabulary,
            epoch=epoch,
        )

        print()
        print("Running validation...")

        (
            validation_loss,
            validation_cer,
            references,
            hypotheses,
        ) = validate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            vocabulary=vocabulary,
        )

        history["train_loss"].append(
            train_loss
        )
        history["validation_loss"].append(
            validation_loss
        )
        history["validation_cer"].append(
            validation_cer
        )

        save_history(history)

        print()
        print(
            f"Epoch {epoch} results:"
        )
        print(
            f"  Training loss: {train_loss:.4f}"
        )
        print(
            f"  Validation loss: {validation_loss:.4f}"
        )
        print(
            f"  Validation CER: "
            f"{validation_cer:.4f} "
            f"({validation_cer * 100:.2f}%)"
        )
        print(
            f"  Epoch time: "
            f"{time.time() - epoch_start:.1f}s"
        )

        print()
        print("Example predictions:")

        for i in range(
            min(3, len(references))
        ):
            print(
                f"  {i + 1}. "
                f"Reference: {references[i]}"
            )
            print(
                f"     Prediction: {hypotheses[i]}"
            )

        if validation_cer < best_cer:
            best_cer = validation_cer

            checkpoint = {
                "experiment": EXPERIMENT_NAME,
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "validation_cer":
                    validation_cer,
                "validation_loss":
                    validation_loss,
                "image_height":
                    IMAGE_HEIGHT,
                "image_width":
                    IMAGE_WIDTH,
                "ctc_timesteps":
                    CTC_TIMESTEPS,
                "batch_size":
                    BATCH_SIZE,
                "learning_rate":
                    LEARNING_RATE,
                "weight_decay":
                    WEIGHT_DECAY,
                "multiline_target_ratio":
                    TARGET_MULTILINE_RATIO,
                "seed":
                    SEED,
                "num_classes":
                    vocabulary.num_classes,
                "vocabulary_char_to_id":
                    vocabulary.char_to_id,
                "vocabulary_id_to_char":
                    vocabulary.id_to_char,
                "history":
                    history,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH,
            )

            print()
            print(
                f"Best model saved to: "
                f"{BEST_MODEL_PATH}"
            )

    print()
    print("=" * 70)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation CER: "
        f"{best_cer:.4f}"
    )
    print(
        f"Best checkpoint: "
        f"{BEST_MODEL_PATH}"
    )
    print(
        f"History: "
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()
