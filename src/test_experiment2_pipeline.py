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
# CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 1024
CTC_TIMESTEPS = 256

BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 2 PIPELINE SANITY TEST")
    print("=" * 70)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load raw training data for vocabulary.
    # --------------------------------------------------------

    print()
    print(
        "Loading training data for vocabulary..."
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
    # Create Experiment 2 dataset.
    # --------------------------------------------------------

    print()
    print(
        "Creating Experiment 2 dataset..."
    )

    dataset = (
        IndonesianHandwritingDataset(
            split="train",
            vocabulary=vocabulary,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ctc_timesteps=CTC_TIMESTEPS
        )
    )

    print()
    print(
        f"Dataset size: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader.
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Get one batch.
    # --------------------------------------------------------

    batch = next(
        iter(dataloader)
    )

    images = batch[
        "images"
    ].to(DEVICE)

    labels = batch[
        "labels"
    ].to(DEVICE)

    label_lengths = batch[
        "label_lengths"
    ].to(DEVICE)

    print()
    print(
        f"Images shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Labels shape: "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Label lengths: "
        f"{label_lengths.tolist()}"
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
    # Forward pass.
    # --------------------------------------------------------

    print()
    print(
        "Running forward pass..."
    )

    logits = model(
        images
    )

    print(
        f"Logits shape: "
        f"{tuple(logits.shape)}"
    )

    expected_shape = (
        CTC_TIMESTEPS,
        BATCH_SIZE,
        vocabulary.num_classes
    )

    print(
        f"Expected shape: "
        f"{expected_shape}"
    )

    if tuple(logits.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected model output shape."
        )

    print(
        "FORWARD PASS: PASS"
    )

    # --------------------------------------------------------
    # Check CTC loss.
    # --------------------------------------------------------

    print()
    print(
        "Calculating CTC loss..."
    )

    criterion = nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True
    )

    log_probs = torch.log_softmax(
        logits,
        dim=2
    )

    input_lengths = torch.full(
        size=(BATCH_SIZE,),
        fill_value=logits.size(0),
        dtype=torch.long,
        device=DEVICE
    )

    loss = criterion(
        log_probs,
        labels,
        input_lengths,
        label_lengths
    )

    print(
        f"CTC loss: "
        f"{loss.item():.6f}"
    )

    if not torch.isfinite(loss):

        raise RuntimeError(
            "CTC loss is non-finite."
        )

    print(
        "CTC LOSS: PASS"
    )

    # --------------------------------------------------------
    # Backward pass.
    # --------------------------------------------------------

    print()
    print(
        "Running backward pass..."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003,
        weight_decay=1e-4
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    gradient_count = 0
    nonfinite_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradient_count += 1

            if not torch.isfinite(
                parameter.grad
            ).all():

                nonfinite_gradients += 1

    print(
        f"Parameters with gradients: "
        f"{gradient_count}"
    )

    print(
        f"Non-finite gradients: "
        f"{nonfinite_gradients}"
    )

    if nonfinite_gradients != 0:

        raise RuntimeError(
            "Non-finite gradients detected."
        )

    print(
        "BACKWARD PASS: PASS"
    )

    # --------------------------------------------------------
    # Optimizer step.
    # --------------------------------------------------------

    print()
    print(
        "Running optimizer step..."
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=5.0
    )

    optimizer.step()

    print(
        "OPTIMIZER STEP: PASS"
    )

    # --------------------------------------------------------
    # Complete.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXPERIMENT 2 PIPELINE SANITY TEST COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()