"""
OPTIONAL control experiment: the baseline trained for longer.

Why this exists
---------------
In every reported run the validation CER was still falling steeply at
epoch 10 (see results/*_history.csv). All four configurations therefore
stopped while still improving, and the 10-epoch budget may have favoured
the configuration that learns fastest rather than the best one. This run
changes ONE thing relative to the baseline - the number of epochs - so it
shows how much of the long/multiline weakness is simply under-training.

It is NOT a new idea or model: same architecture, input, optimizer,
learning rate, weight decay, batch size, loss, seed and data as the
baseline.

Usage (repository root, GPU strongly recommended; about 4x the baseline
training time at the default of 40 epochs):

    python src/train_baseline_extended.py
    python src/train_baseline_extended.py --epochs 60

Outputs:
    checkpoints/baseline_extended_best.pt
    results/baseline_extended_history.csv

Then evaluate with the same unified tool as every other model:

    python src/analyze_model.py --model baseline_extended
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

import htr_common as hc
from train import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    SEED,
    set_seed,
    save_history,
    train_one_epoch,
    validate,
)
from dataloader import collate_fn

CHECKPOINT_PATH = hc.MODEL_REGISTRY["baseline_extended"]["checkpoint"]
HISTORY_PATH = os.path.join("results", "baseline_extended_history.csv")
DEFAULT_EPOCHS = 40


def train_loop(
    model,
    train_loader,
    validation_loader,
    vocabulary,
    epochs,
    checkpoint_path,
    history_path,
):
    """
    Same loop as train.py main(): AdamW + CTC, best checkpoint by
    validation CER. Separated from main() so it can be tested without
    the dataset.
    """

    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    history = {"train_loss": [], "validation_loss": [], "validation_cer": []}
    best_cer = float("inf")

    for epoch in range(1, epochs + 1):
        began = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            vocabulary=vocabulary,
            epoch=epoch,
        )

        validation_loss, validation_cer, _, _ = validate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            vocabulary=vocabulary,
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_cer"].append(validation_cer)

        # Rewritten every epoch so an interrupted run keeps its history.
        save_history(history, history_path)

        print(
            f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_CER={validation_cer:.4f} "
            f"({time.time() - began:.0f}s)",
            flush=True,
        )

        if validation_cer < best_cer:
            best_cer = validation_cer

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_cer": validation_cer,
                    "validation_loss": validation_loss,
                    "vocabulary_char_to_id": vocabulary.char_to_id,
                    "vocabulary_id_to_char": vocabulary.id_to_char,
                    "num_classes": vocabulary.num_classes,
                    "architecture": "cnn_bilstm",
                    "image_height": 128,
                    "image_width": 1024,
                    "ctc_timesteps": 256,
                    "seed": SEED,
                    "planned_epochs": epochs,
                },
                checkpoint_path,
            )

            print(f"  best so far; saved {checkpoint_path}", flush=True)

    return history, best_cer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()

    # Imported here so the loop above is usable without `datasets`.
    from datasets import load_dataset
    from dataset import (
        CharacterVocabulary,
        IndonesianHandwritingDataset,
        DATASET_NAME,
    )

    set_seed(SEED)

    print(f"Device: {DEVICE}; epochs: {args.epochs}; seed: {SEED}")

    vocabulary = CharacterVocabulary(
        load_dataset(DATASET_NAME, split="train")
    )

    train_dataset = IndonesianHandwritingDataset(
        split="train", vocabulary=vocabulary
    )
    validation_dataset = IndonesianHandwritingDataset(
        split="validation", vocabulary=vocabulary
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    model = hc.build_model("cnn_bilstm", vocabulary.num_classes).to(DEVICE)

    _, best_cer = train_loop(
        model,
        train_loader,
        validation_loader,
        vocabulary,
        args.epochs,
        CHECKPOINT_PATH,
        HISTORY_PATH,
    )

    print(f"Best validation CER: {best_cer:.4f}")


if __name__ == "__main__":
    main()
