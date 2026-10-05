import os
import time
import csv

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
from model_experiment2 import CNNBiLSTMExperiment2


# ============================================================
# EXPERIMENT 2 CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "experiment2"

# ------------------------------------------------------------
# Image configuration
# ------------------------------------------------------------

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 1024

# The CNN reduces the width by a factor of 4:
#
# 1024 -> 512 -> 256
#
CTC_TIMESTEPS = IMAGE_WIDTH // 4


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

# Physical batch size.
BATCH_SIZE = 2

# Accumulate two batches:
#
# 2 × 2 = effective batch size 4
#
GRAD_ACCUMULATION_STEPS = 2

EFFECTIVE_BATCH_SIZE = (
    BATCH_SIZE
    * GRAD_ACCUMULATION_STEPS
)

# Keep the same number of epochs as baseline
# and Experiment 1.
NUM_EPOCHS = 10

# Keep learning rate unchanged.
LEARNING_RATE = 0.0003

# Keep weight decay unchanged.
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_DIR = "checkpoints"

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "experiment2_best.pt"
)

HISTORY_PATH = os.path.join(
    "results",
    "experiment2_history.csv"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CTC GREEDY DECODING
# ============================================================

def greedy_decode(
    logits,
    vocabulary
):
    """
    Decode CTC output using greedy decoding.

    Input:
        logits: [time, batch, classes]

    Output:
        list[str]
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

            # Ignore CTC blank.
            if token == vocabulary.blank_id:

                previous = token

                continue

            # Collapse consecutive repeated tokens.
            if token != previous:

                decoded_ids.append(
                    token
                )

            previous = token

        decoded_texts.append(
            vocabulary.decode(
                decoded_ids
            )
        )

    return decoded_texts


# ============================================================
# LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein_distance(
    reference,
    hypothesis
):
    """
    Calculate character-level Levenshtein distance.
    """

    reference_length = len(
        reference
    )

    hypothesis_length = len(
        hypothesis
    )

    previous_row = list(
        range(
            hypothesis_length + 1
        )
    )

    for i in range(
        1,
        reference_length + 1
    ):

        current_row = [i]

        for j in range(
            1,
            hypothesis_length + 1
        ):

            insertion = (
                current_row[j - 1]
                + 1
            )

            deletion = (
                previous_row[j]
                + 1
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
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    epoch
):

    model.train()

    total_loss = 0.0

    num_batches = len(
        dataloader
    )

    start_time = time.time()

    optimizer.zero_grad(
        set_to_none=True
    )

    for batch_index, batch in enumerate(
        dataloader,
        start=1
    ):

        images = batch[
            "images"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        label_lengths = batch[
            "label_lengths"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Forward.
        # ----------------------------------------------------

        logits = model(
            images
        )

        # ----------------------------------------------------
        # CTC log probabilities.
        # ----------------------------------------------------

        log_probs = torch.log_softmax(
            logits,
            dim=2
        )

        time_steps = logits.size(0)

        input_lengths = torch.full(
            size=(
                images.size(0),
            ),
            fill_value=time_steps,
            dtype=torch.long,
            device=DEVICE
        )

        # ----------------------------------------------------
        # CTC loss.
        # ----------------------------------------------------

        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            label_lengths
        )

        loss_value = loss.item()

        # ----------------------------------------------------
        # Gradient accumulation.
        # ----------------------------------------------------

        scaled_loss = (
            loss
            / GRAD_ACCUMULATION_STEPS
        )

        scaled_loss.backward()

        # ----------------------------------------------------
        # Optimizer step.
        # ----------------------------------------------------

        should_step = (
            batch_index
            % GRAD_ACCUMULATION_STEPS
            == 0
            or batch_index == num_batches
        )

        if should_step:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += loss_value

        # ----------------------------------------------------
        # Progress.
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
                f"Loss: {loss_value:.4f} | "
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
            # Forward.
            # ------------------------------------------------

            logits = model(
                images
            )

            log_probs = torch.log_softmax(
                logits,
                dim=2
            )

            time_steps = logits.size(0)

            input_lengths = torch.full(
                size=(
                    images.size(0),
                ),
                fill_value=time_steps,
                dtype=torch.long,
                device=DEVICE
            )

            # ------------------------------------------------
            # Validation loss.
            # ------------------------------------------------

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # Decode.
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
# SAVE HISTORY
# ============================================================

def save_history(history):

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "EXPERIMENT 2: INCREASED VERTICAL RESOLUTION"
    )
    print("=" * 70)

    print()
    print(
        "Research question:"
    )

    print(
        "Does increasing vertical image resolution "
        "preserve more information about multiline "
        "handwriting and improve recognition?"
    )

    # --------------------------------------------------------
    # Device.
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
    # Configuration.
    # --------------------------------------------------------

    print()
    print(
        "Experiment 2 configuration:"
    )

    print(
        f"  Image height: "
        f"{IMAGE_HEIGHT}"
    )

    print(
        f"  Image width: "
        f"{IMAGE_WIDTH}"
    )

    print(
        f"  CTC time steps: "
        f"{CTC_TIMESTEPS}"
    )

    print(
        f"  Physical batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"  Gradient accumulation: "
        f"{GRAD_ACCUMULATION_STEPS}"
    )

    print(
        f"  Effective batch size: "
        f"{EFFECTIVE_BATCH_SIZE}"
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
        f"  Epochs: "
        f"{NUM_EPOCHS}"
    )

    # --------------------------------------------------------
    # Vocabulary.
    # --------------------------------------------------------

    print()
    print(
        "Loading training split for vocabulary..."
    )

    raw_train_dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    vocabulary = CharacterVocabulary(
        raw_train_dataset
    )

    print(
        f"Characters: "
        f"{vocabulary.num_characters}"
    )

    print(
        f"CTC classes: "
        f"{vocabulary.num_classes}"
    )

    print(
        f"CTC blank ID: "
        f"{vocabulary.blank_id}"
    )

    # --------------------------------------------------------
    # Training dataset.
    # --------------------------------------------------------

    print()
    print(
        "Loading Experiment 2 training dataset..."
    )

    train_dataset = (
        IndonesianHandwritingDataset(
            split="train",
            vocabulary=vocabulary,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ctc_timesteps=CTC_TIMESTEPS
        )
    )

    # --------------------------------------------------------
    # Validation dataset.
    # --------------------------------------------------------

    print()
    print(
        "Loading Experiment 2 validation dataset..."
    )

    validation_dataset = (
        IndonesianHandwritingDataset(
            split="validation",
            vocabulary=vocabulary,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ctc_timesteps=CTC_TIMESTEPS
        )
    )

    print()
    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders.
    # --------------------------------------------------------

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

    print()
    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # Create model.
    # --------------------------------------------------------

    print()
    print(
        "Creating Experiment 2 model..."
    )

    model = CNNBiLSTMExperiment2(
        num_classes=vocabulary.num_classes,
        image_height=IMAGE_HEIGHT
    )

    model = model.to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Verify dimensions.
    # --------------------------------------------------------

    print()
    print(
        "Testing model output dimensions..."
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

    print(
        f"Input shape: "
        f"{tuple(test_images.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(test_logits.shape)}"
    )

    expected_shape = (
        CTC_TIMESTEPS,
        BATCH_SIZE,
        vocabulary.num_classes
    )

    if tuple(test_logits.shape) != expected_shape:

        raise RuntimeError(
            f"Expected model output "
            f"{expected_shape}, but got "
            f"{tuple(test_logits.shape)}"
        )

    print(
        "Sequence-length test: PASS"
    )

    # --------------------------------------------------------
    # CTC loss.
    # --------------------------------------------------------

    criterion = nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True
    )

    # --------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Checkpoint directory.
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # History.
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_cer": []
    }

    best_cer = float(
        "inf"
    )

    # --------------------------------------------------------
    # Start training.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STARTING EXPERIMENT 2"
    )
    print("=" * 70)

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print()
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "-" * 70
        )

        epoch_start = time.time()

        # ----------------------------------------------------
        # Training.
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch
        )

        # ----------------------------------------------------
        # Validation.
        # ----------------------------------------------------

        print()
        print(
            "Running validation..."
        )

        (
            validation_loss,
            validation_cer,
            references,
            hypotheses
        ) = validate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            vocabulary=vocabulary
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # Store history.
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "validation_loss"
        ].append(
            validation_loss
        )

        history[
            "validation_cer"
        ].append(
            validation_cer
        )

        save_history(
            history
        )

        # ----------------------------------------------------
        # Report.
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
            f"  Validation CER (%): "
            f"{validation_cer * 100:.2f}%"
        )

        print(
            f"  Epoch time: "
            f"{epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Example predictions.
        # ----------------------------------------------------

        print()
        print(
            "Example predictions:"
        )

        for i in range(
            min(
                3,
                len(references)
            )
        ):

            print()
            print(
                f"Example {i + 1}:"
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
        # Save best checkpoint.
        # ----------------------------------------------------

        if validation_cer < best_cer:

            best_cer = (
                validation_cer
            )

            checkpoint = {
                "experiment":
                    EXPERIMENT_NAME,

                "epoch":
                    epoch,

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

                "gradient_accumulation_steps":
                    GRAD_ACCUMULATION_STEPS,

                "effective_batch_size":
                    EFFECTIVE_BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

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
                BEST_MODEL_PATH
            )

            print()
            print(
                f"Best model saved to: "
                f"{BEST_MODEL_PATH}"
            )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXPERIMENT 2 COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Best validation CER: "
        f"{best_cer:.4f}"
    )

    print(
        f"Best validation CER (%): "
        f"{best_cer * 100:.2f}%"
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