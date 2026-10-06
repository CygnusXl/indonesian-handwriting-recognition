import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from dataset import (
    CharacterVocabulary,
    IndonesianHandwritingDataset,
    DATASET_NAME,
)

from dataloader import collate_fn
from model import CNNBiLSTM

# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4

# Keep this at 1 for the first pipeline test.
# We will increase it after confirming everything works.
NUM_EPOCHS = 10

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "baseline_best.pt"
)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1024

# Added after the reported baseline run (which was NOT seeded). A fixed
# seed makes re-training repeatable on the same hardware/software, but
# the reported baseline checkpoint comes from the earlier unseeded run.
SEED = 42

HISTORY_PATH = os.path.join(
    "results",
    "baseline_history.csv"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(history, path):
    """Write the per-epoch history as CSV (same format as Experiments 1-3)."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["epoch", "train_loss", "validation_loss", "validation_cer"]
        )

        for epoch in range(len(history["train_loss"])):
            writer.writerow(
                [
                    epoch + 1,
                    history["train_loss"][epoch],
                    history["validation_loss"][epoch],
                    history["validation_cer"][epoch],
                ]
            )


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# GREEDY CTC DECODING
# ============================================================

def greedy_decode(logits, vocabulary):
    """
    Convert CTC model output into text using greedy decoding.

    Input:
        logits: [time, batch, classes]

    Output:
        list of decoded strings
    """

    predictions = torch.argmax(
        logits,
        dim=2
    )

    decoded_texts = []

    for batch_index in range(
        predictions.size(1)
    ):

        sequence = predictions[
            :,
            batch_index
        ].tolist()

        decoded_ids = []

        previous = None

        for token in sequence:

            # CTC blank
            if token == vocabulary.blank_id:
                previous = token
                continue

            # Collapse repeated consecutive tokens.
            if token != previous:
                decoded_ids.append(token)

            previous = token

        text = vocabulary.decode(
            decoded_ids
        )

        decoded_texts.append(text)

    return decoded_texts


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein_distance(reference, hypothesis):
    """
    Calculate character-level Levenshtein distance.
    """

    reference_length = len(reference)
    hypothesis_length = len(hypothesis)

    previous_row = list(
        range(hypothesis_length + 1)
    )

    for i in range(1, reference_length + 1):

        current_row = [
            i
        ]

        for j in range(
            1,
            hypothesis_length + 1
        ):

            insertion = (
                current_row[j - 1] + 1
            )

            deletion = (
                previous_row[j] + 1
            )

            substitution = (
                previous_row[j - 1]
                + (
                    reference[i - 1]
                    != hypothesis[j - 1]
                )
            )

            current_row.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous_row = current_row

    return previous_row[-1]


# ============================================================
# CHARACTER ERROR RATE
# ============================================================

def calculate_cer(
    references,
    hypotheses
):
    """
    Calculate Character Error Rate.

    CER =
        total edit distance /
        total number of reference characters
    """

    total_distance = 0
    total_characters = 0

    for reference, hypothesis in zip(
        references,
        hypotheses
    ):

        total_distance += (
            levenshtein_distance(
                reference,
                hypothesis
            )
        )

        total_characters += len(
            reference
        )

    if total_characters == 0:
        return 0.0

    return (
        total_distance
        / total_characters
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    vocabulary,
    epoch
):

    model.train()

    total_loss = 0.0
    num_batches = len(dataloader)

    start_time = time.time()

    for batch_index, batch in enumerate(
        dataloader,
        start=1
    ):

        images = batch["images"].to(
            DEVICE
        )

        labels = batch["labels"].to(
            DEVICE
        )

        label_lengths = batch[
            "label_lengths"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(images)

        # ----------------------------------------------------
        # CTC expects log probabilities.
        #
        # logits:
        #     [time, batch, classes]
        #
        # log_probs:
        #     [time, batch, classes]
        # ----------------------------------------------------

        log_probs = torch.log_softmax(
            logits,
            dim=2
        )

        time_steps = logits.size(0)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=time_steps,
            dtype=torch.long,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Calculate CTC loss
        # ----------------------------------------------------

        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            label_lengths
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # Prevent extremely large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index == 1
            or batch_index % 50 == 0
            or batch_index == num_batches
        ):

            elapsed = (
                time.time()
                - start_time
            )

            average_loss = (
                total_loss
                / batch_index
            )

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index}/{num_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Avg Loss: {average_loss:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    return (
        total_loss
        / num_batches
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    dataloader,
    criterion,
    vocabulary
):

    model.eval()

    total_loss = 0.0

    references = []
    hypotheses = []

    with torch.no_grad():

        for batch in dataloader:

            images = batch[
                "images"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            label_lengths = batch[
                "label_lengths"
            ].to(DEVICE)

            texts = batch[
                "texts"
            ]

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(images)

            log_probs = torch.log_softmax(
                logits,
                dim=2
            )

            time_steps = logits.size(0)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=time_steps,
                dtype=torch.long,
                device=DEVICE
            )

            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # Decode predictions
            # ------------------------------------------------

            predictions = greedy_decode(
                logits,
                vocabulary
            )

            references.extend(
                texts
            )

            hypotheses.extend(
                predictions
            )

    average_loss = (
        total_loss
        / len(dataloader)
    )

    cer = calculate_cer(
        references,
        hypotheses
    )

    return (
        average_loss,
        cer,
        references,
        hypotheses
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print(
        "BASELINE CNN-BiLSTM TRAINING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    print()

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        gpu_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{gpu_memory:.2f} GB"
        )

    # --------------------------------------------------------
    # Load raw training dataset
    #
    # This dataset is used ONLY to construct
    # the character vocabulary.
    # --------------------------------------------------------

    print()
    print(
        "Loading training split "
        "for vocabulary..."
    )

    raw_train_dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    print(
        f"Raw training samples: "
        f"{len(raw_train_dataset)}"
    )

    # --------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------

    print()
    print(
        "Building character vocabulary..."
    )

    vocabulary = CharacterVocabulary(
        raw_train_dataset
    )

    print(
        f"Number of characters: "
        f"{vocabulary.num_characters}"
    )

    print(
        f"Number of CTC classes: "
        f"{vocabulary.num_classes}"
    )

    print(
        f"CTC blank ID: "
        f"{vocabulary.blank_id}"
    )

    # --------------------------------------------------------
    # Load Indonesian training dataset
    #
    # IMPORTANT:
    # IndonesianHandwritingDataset already knows
    # DATASET_NAME internally.
    #
    # Therefore we pass ONLY split and vocabulary.
    # --------------------------------------------------------

    print()
    print(
        "Loading training dataset..."
    )

    train_dataset = (
        IndonesianHandwritingDataset(
            split="train",
            vocabulary=vocabulary
        )
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading validation dataset..."
    )

    validation_dataset = (
        IndonesianHandwritingDataset(
            split="validation",
            vocabulary=vocabulary
        )
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    print()
    print(
        "Creating DataLoaders..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print()
    print(
        "Creating CNN-BiLSTM model..."
    )

    model = CNNBiLSTM(
        num_classes=vocabulary.num_classes
    )

    model = model.to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameter_count:,}"
    )

    # --------------------------------------------------------
    # CTC loss
    # --------------------------------------------------------

    criterion = nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print()
    print(
        "Training configuration:"
    )

    print(
        f"  Batch size: {BATCH_SIZE}"
    )

    print(
        f"  Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"  Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"  Epochs: {NUM_EPOCHS}"
    )

    print(
        f"  Image size: "
        f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}"
    )

    print(
        f"  CTC time steps: 256"
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_cer": []
    }

    best_cer = float("inf")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print("-" * 70)

        epoch_start = time.time()

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            vocabulary=vocabulary,
            epoch=epoch
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        print()
        print(
            "Running validation..."
        )

        validation_loss, validation_cer, references, hypotheses = (
            validate(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                vocabulary=vocabulary
            )
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "validation_loss"
        ].append(validation_loss)

        history[
            "validation_cer"
        ].append(validation_cer)

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch} results:"
        )

        print(
            f"  Training loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Validation loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"  Validation CER: "
            f"{validation_cer:.4f}"
        )

        print(
            f"  Epoch time: "
            f"{epoch_time:.1f} seconds"
        )

        # ----------------------------------------------------
        # Show example predictions
        # ----------------------------------------------------

        print()
        print(
            "Example predictions:"
        )

        number_of_examples = min(
            3,
            len(references)
        )

        for i in range(
            number_of_examples
        ):

            print(
                f"\nExample {i + 1}:"
            )

            print(
                f"  Reference: "
                f"{references[i]}"
            )

            print(
                f"  Prediction: "
                f"{hypotheses[i]}"
            )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if validation_cer < best_cer:

            best_cer = validation_cer

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_cer": validation_cer,
                "validation_loss": validation_loss,
                "vocabulary_char_to_id":
                    vocabulary.char_to_id,
                "vocabulary_id_to_char":
                    vocabulary.id_to_char,
                "num_classes":
                    vocabulary.num_classes,
                "architecture": "cnn_bilstm",
                "image_height": IMAGE_HEIGHT,
                "image_width": IMAGE_WIDTH,
                "ctc_timesteps": IMAGE_WIDTH // 4,
                "seed": SEED,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH
            )

            print()
            print(
                f"Best model saved to: "
                f"{BEST_MODEL_PATH}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
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

    save_history(history, HISTORY_PATH)

    print(f"History saved to: {HISTORY_PATH}")

    print()
    print("Training history:")

    for i in range(
        len(history["train_loss"])
    ):

        print(
            f"Epoch {i + 1}: "
            f"train_loss="
            f"{history['train_loss'][i]:.4f}, "
            f"val_loss="
            f"{history['validation_loss'][i]:.4f}, "
            f"val_CER="
            f"{history['validation_cer'][i]:.4f}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()