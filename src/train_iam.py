"""
Train the CNN-BiLSTM-CTC line recognizer on IAM (English handwriting).

STATUS: the training loop is tested on CPU with random data
(tests/test_train_iam.py). It has NOT been run on the real IAM data, and
no accuracy figure exists for the model it produces until you run it and
src/evaluate_iam.py. Hyper-parameters below are the project's earlier
defaults plus a cosine learning-rate decay; they are NOT tuned.

Why IAM
-------
The earlier model was trained only on synthetic Indonesian handwriting
and could not read a real cursive photo (one documented case, n=1, see
DOCUMENT_PIPELINE.md). The working hypothesis is that the training data,
not the architecture, is the main limit. IAM has real pen-on-paper
writing from 657 writers. That is MORE varied than synthetic data but is
still one collection method (forms, one kind of paper and scanner), so
"reads any handwriting from any photo" is NOT a claim this project can
make; only the measured results can.

What is the same as before: the architecture (CNN + BiLSTM + CTC, 7.3 M
parameters with ~70 classes), the 128-pixel input height, greedy CTC
decoding, AdamW, seed 42, the best checkpoint chosen by validation CER.
What is different: the data, the character set (built from IAM), the
default input width 1536 px / 384 CTC steps (IAM lines are long), and
photo-style augmentation (src/photo_augment.py, switch off with
--no-augment).

Usage (repository root, GPU strongly recommended):

    python src/train_iam.py --limit 200 --epochs 1      # quick check
    python src/train_iam.py                              # full run

Outputs:
    checkpoints/iam_best.pt
    checkpoints/iam_best.json       run details (sizes, dropped lines)
    results/iam_history.csv
"""

import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader

import htr_common as hc
import iam_dataset as iam
from dataloader import collate_fn
from train import (
    DEVICE,
    SEED,
    WEIGHT_DECAY,
    save_history,
    set_seed,
    train_one_epoch,
    validate,
)

CHECKPOINT_PATH = os.path.join("checkpoints", "iam_best.pt")
HISTORY_PATH = os.path.join("results", "iam_history.csv")

DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WIDTH = 1536
IMAGE_HEIGHT = 128   # fixed by the model (CNN flattens 256 x 8)


def train_loop(
    model,
    train_loader,
    validation_loader,
    vocabulary,
    epochs,
    checkpoint_path,
    history_path,
    image_width,
    learning_rate=DEFAULT_LEARNING_RATE,
    extra_checkpoint_fields=None,
):
    """
    AdamW + CTC with cosine learning-rate decay; best checkpoint by
    validation CER. Separate from main() so it can be tested without the
    dataset. The checkpoint stores everything inference needs (the
    vocabulary and the input configuration), so the app never guesses.
    """

    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id, zero_infinity=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=learning_rate * 0.03
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

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_cer"].append(validation_cer)
        save_history(history, history_path)

        print(
            f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_CER={validation_cer:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"({time.time() - began:.0f}s)",
            flush=True,
        )

        if validation_cer < best_cer:
            best_cer = validation_cer

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_cer": validation_cer,
                "validation_loss": validation_loss,
                "vocabulary_char_to_id": vocabulary.char_to_id,
                "vocabulary_id_to_char": vocabulary.id_to_char,
                "num_classes": vocabulary.num_classes,
                "architecture": "cnn_bilstm",
                "image_height": IMAGE_HEIGHT,
                "image_width": image_width,
                "ctc_timesteps": image_width // 4,
                "planned_epochs": epochs,
            }
            checkpoint.update(extra_checkpoint_fields or {})

            torch.save(checkpoint, checkpoint_path)
            print(f"  best so far; saved {checkpoint_path}", flush=True)

    return history, best_cer


def make_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        # Workers must persist, otherwise every epoch would restart each
        # worker's random generator and repeat the same augmentations.
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers > 0 else None,
    )


def _seed_worker(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(SEED * 1000 + worker_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dataset", default=iam.IAM_DATASET_NAME)
    parser.add_argument("--output", default=CHECKPOINT_PATH)
    parser.add_argument("--history", default=HISTORY_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N lines of each split (quick check)",
    )
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    if args.width % 4:
        parser.error("--width must be a multiple of 4")

    set_seed(SEED)

    print(f"Device: {DEVICE}; epochs: {args.epochs}; seed: {SEED}")

    try:
        train_hf = iam.load_split("train", args.dataset)
        validation_hf = iam.load_split("validation", args.dataset)
    except iam.IamUnavailableError as error:
        raise SystemExit(f"\nERROR: {error}")

    if args.limit:
        train_hf = train_hf.select(range(min(args.limit, len(train_hf))))
        validation_hf = validation_hf.select(
            range(min(args.limit, len(validation_hf)))
        )

    _, text_column = iam.detect_columns(train_hf.column_names)

    vocabulary = iam.build_vocabulary(
        iam.normalize_text(t) for t in train_hf[text_column]
    )

    print(
        f"Train lines: {len(train_hf)}; validation lines: "
        f"{len(validation_hf)}; characters: {vocabulary.num_characters} "
        f"(classes incl. blank: {vocabulary.num_classes})"
    )

    common = dict(
        image_height=IMAGE_HEIGHT,
        image_width=args.width,
        ctc_timesteps=args.width // 4,
    )

    train_dataset = iam.IamLineDataset.from_huggingface(
        train_hf,
        vocabulary,
        augment=not args.no_augment,
        rng=random.Random(SEED),
        **common,
    )
    validation_dataset = iam.IamLineDataset.from_huggingface(
        validation_hf, vocabulary, **common
    )

    print(f"Dropped from train:      {train_dataset.dropped}")
    print(f"Dropped from validation: {validation_dataset.dropped}")

    if not len(train_dataset) or not len(validation_dataset):
        raise SystemExit("ERROR: no usable training or validation lines.")

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.workers
    )
    validation_loader = make_loader(
        validation_dataset, args.batch_size, False, 0
    )

    model = hc.build_model("cnn_bilstm", vocabulary.num_classes).to(DEVICE)

    print(f"Parameters: {hc.count_parameters(model):,}")

    run_fields = {
        "training_data": "iam",
        "training_data_label": "IAM handwriting database (English)",
        "training_dataset_source": args.dataset,
        "augmentation": not args.no_augment,
        "seed": SEED,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_lines": len(train_dataset),
        "validation_lines": len(validation_dataset),
    }

    history, best_cer = train_loop(
        model,
        train_loader,
        validation_loader,
        vocabulary,
        args.epochs,
        args.output,
        args.history,
        image_width=args.width,
        learning_rate=args.lr,
        extra_checkpoint_fields=run_fields,
    )

    details = {
        **run_fields,
        "dropped_train": train_dataset.dropped,
        "dropped_validation": validation_dataset.dropped,
        "best_validation_cer": best_cer,
        "epochs": args.epochs,
        "image_width": args.width,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    }

    with open(
        os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8"
    ) as file:
        json.dump(details, file, indent=2)

    print(f"\nBest validation CER: {best_cer:.4f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
