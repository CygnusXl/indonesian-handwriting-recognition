"""
Compare Baseline, Experiment 1, and Experiment 2
on the exact same test samples.

Baseline:
    128 x 1024

Experiment 1:
    128 x 1536

Experiment 2:
    192 x 1024

All models are evaluated on the same 208 CTC-feasible
Indonesian test samples.
"""

import os
import csv
import torch
from torch.utils.data import DataLoader

from dataset import (
    CharacterVocabulary,
    IndonesianHandwritingDataset,
    calculate_ctc_required_timesteps,
)

from dataloader import collate_fn
from model import CNNBiLSTM
from model_experiment2 import CNNBiLSTMExperiment2


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CTC_TIMESTEPS = 256
BATCH_SIZE = 2

BASELINE_CHECKPOINT = "checkpoints/baseline_best.pt"
EXPERIMENT1_CHECKPOINT = "checkpoints/experiment1_best.pt"
EXPERIMENT2_CHECKPOINT = "checkpoints/experiment2_best.pt"

OUTPUT_CSV = "results/all_experiments_comparison.csv"
OUTPUT_SUMMARY = "results/all_experiments_comparison_summary.txt"


# ============================================================
# GREEDY CTC DECODING
# ============================================================

def greedy_decode(logits, vocabulary):
    """
    logits:
        [T, B, C]
    """

    predictions = logits.argmax(dim=2)  # [T, B]

    decoded = []

    for batch_index in range(predictions.shape[1]):

        sequence = predictions[:, batch_index].tolist()

        output = []
        previous = vocabulary.blank_id

        for token in sequence:

            # Remove repeated tokens
            if token != previous:

                # Remove CTC blank
                if token != vocabulary.blank_id:
                    output.append(token)

            previous = token

        decoded.append(vocabulary.decode(output))

    return decoded


# ============================================================
# CER
# ============================================================

def character_error_rate(reference, hypothesis):

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    previous = list(range(len(hypothesis) + 1))

    for i, ref_char in enumerate(reference, start=1):

        current = [i]

        for j, hyp_char in enumerate(hypothesis, start=1):

            if ref_char == hyp_char:
                cost = 0
            else:
                cost = 1

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1] / len(reference)


# ============================================================
# WER
# ============================================================

def word_error_rate(reference, hypothesis):

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    previous = list(range(len(hyp_words) + 1))

    for i, ref_word in enumerate(ref_words, start=1):

        current = [i]

        for j, hyp_word in enumerate(hyp_words, start=1):

            if ref_word == hyp_word:
                cost = 0
            else:
                cost = 1

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1] / len(ref_words)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(path, model):

    print(f"Loading checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        val_cer = checkpoint.get("val_cer", None)
    else:
        model.load_state_dict(checkpoint)
        epoch = "unknown"
        val_cer = None

    model.to(DEVICE)
    model.eval()

    return model, epoch, val_cer


# ============================================================
# CREATE TEST DATASET
# ============================================================

def create_dataset(vocabulary, height, width):

    dataset = IndonesianHandwritingDataset(
        split="test",
        vocabulary=vocabulary,
        image_height=height,
        image_width=width,
        ctc_timesteps=TEST_CTC_TIMESTEPS,
    )

    return dataset


# ============================================================
# VERIFY SAME TEST SAMPLES
# ============================================================

def verify_same_samples(datasets):

    print()
    print("=" * 70)
    print("VERIFYING COMMON TEST SET")
    print("=" * 70)

    names = list(datasets.keys())

    lengths = [len(datasets[name]) for name in names]

    print("Dataset sizes:")
    for name in names:
        print(f"  {name}: {len(datasets[name])}")

    if len(set(lengths)) != 1:
        raise RuntimeError(
            "The three test datasets do not contain the same number "
            "of samples."
        )

    reference_texts = []

    first_dataset = datasets[names[0]]

    for index in range(len(first_dataset)):
        sample = first_dataset[index]
        reference_texts.append(sample["text"])

    for name in names[1:]:

        dataset = datasets[name]

        for index in range(len(dataset)):

            sample = dataset[index]
            text = sample["text"]

            if text != reference_texts[index]:

                raise RuntimeError(
                    f"Test sample mismatch at index {index} "
                    f"between {names[0]} and {name}."
                )

    print("Same test samples: PASS")
    print(f"Common samples: {len(reference_texts)}")

    return reference_texts


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    dataset,
    vocabulary,
    model_name,
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True,
    )

    total_loss = 0.0
    total_batches = 0

    all_predictions = []
    all_references = []

    print()
    print("=" * 70)
    print(f"EVALUATING {model_name}")
    print("=" * 70)

    with torch.no_grad():

        for batch in loader:

            images = batch["images"]
            labels = batch["labels"]
            label_lengths = batch["label_lengths"]
            texts = batch["texts"]

            images = images.to(DEVICE)

            logits = model(images)

            input_lengths = torch.full(
                size=(images.shape[0],),
                fill_value=logits.shape[0],
                dtype=torch.long,
                device=DEVICE,
            )

            loss = criterion(
                logits.log_softmax(2),
                labels.to(DEVICE),
                input_lengths,
                label_lengths.to(DEVICE),
            )

            total_loss += loss.item()
            total_batches += 1

            predictions = greedy_decode(
                logits.cpu(),
                vocabulary,
            )

            all_predictions.extend(predictions)
            all_references.extend(texts)

    average_loss = total_loss / total_batches

    cer_values = []
    wer_values = []

    for reference, prediction in zip(
        all_references,
        all_predictions,
    ):

        cer_values.append(
            character_error_rate(
                reference,
                prediction,
            )
        )

        wer_values.append(
            word_error_rate(
                reference,
                prediction,
            )
        )

    mean_cer = sum(cer_values) / len(cer_values)
    mean_wer = sum(wer_values) / len(wer_values)

    exact_matches = sum(
        reference == prediction
        for reference, prediction
        in zip(all_references, all_predictions)
    )

    exact_match = exact_matches / len(all_references)

    print(f"Test loss:      {average_loss:.4f}")
    print(f"CER:            {mean_cer:.4f}")
    print(f"CER (%):        {mean_cer * 100:.2f}%")
    print(f"WER:            {mean_wer:.4f}")
    print(f"WER (%):        {mean_wer * 100:.2f}%")
    print(f"Exact match:    {exact_match:.4f}")
    print(f"Exact match (%): {exact_match * 100:.2f}%")

    return {
        "model": model_name,
        "test_loss": average_loss,
        "cer": mean_cer,
        "cer_percent": mean_cer * 100,
        "wer": mean_wer,
        "wer_percent": mean_wer * 100,
        "exact_match": exact_match,
        "exact_match_percent": exact_match * 100,
        "predictions": all_predictions,
        "references": all_references,
        "cer_values": cer_values,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("ALL EXPERIMENTS COMMON TEST SET COMPARISON")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    print()
    print("Creating vocabulary from training data...")

    train_raw = IndonesianHandwritingDataset(
        split="train",
    )

    vocabulary = CharacterVocabulary(
        train_raw.dataset
    )

    print(f"Characters: {vocabulary.num_characters}")
    print(f"CTC classes: {vocabulary.num_classes}")
    print(f"Blank ID: {vocabulary.blank_id}")

    # --------------------------------------------------------
    # Test datasets
    # --------------------------------------------------------

    print()
    print("Creating common test datasets...")

    datasets = {

        "Baseline": create_dataset(
            vocabulary,
            height=128,
            width=1024,
        ),

        "Experiment 1": create_dataset(
            vocabulary,
            height=128,
            width=1536,
        ),

        "Experiment 2": create_dataset(
            vocabulary,
            height=192,
            width=1024,
        ),
    }

    reference_texts = verify_same_samples(datasets)

    if len(reference_texts) != 208:

        raise RuntimeError(
            f"Expected exactly 208 common test samples, "
            f"but found {len(reference_texts)}."
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING MODELS")
    print("=" * 70)

    baseline_model = CNNBiLSTM(
        num_classes=vocabulary.num_classes
    )

    experiment1_model = CNNBiLSTM(
        num_classes=vocabulary.num_classes
    )

    experiment2_model = CNNBiLSTMExperiment2(
        num_classes=vocabulary.num_classes
    )

    # --------------------------------------------------------
    # Load checkpoints
    # --------------------------------------------------------

    baseline_model, baseline_epoch, baseline_val_cer = load_checkpoint(
        BASELINE_CHECKPOINT,
        baseline_model,
    )

    experiment1_model, experiment1_epoch, experiment1_val_cer = load_checkpoint(
        EXPERIMENT1_CHECKPOINT,
        experiment1_model,
    )

    experiment2_model, experiment2_epoch, experiment2_val_cer = load_checkpoint(
        EXPERIMENT2_CHECKPOINT,
        experiment2_model,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    results.append(
        evaluate_model(
            baseline_model,
            datasets["Baseline"],
            vocabulary,
            "Baseline",
        )
    )

    results.append(
        evaluate_model(
            experiment1_model,
            datasets["Experiment 1"],
            vocabulary,
            "Experiment 1",
        )
    )

    results.append(
        evaluate_model(
            experiment2_model,
            datasets["Experiment 2"],
            vocabulary,
            "Experiment 2",
        )
    )

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("FINAL COMMON TEST SET COMPARISON")
    print("=" * 80)

    print()

    header = (
        f"{'Model':<18}"
        f"{'Loss':>12}"
        f"{'CER':>12}"
        f"{'WER':>12}"
        f"{'Exact %':>12}"
    )

    print(header)
    print("-" * 80)

    for result in results:

        print(
            f"{result['model']:<18}"
            f"{result['test_loss']:>12.4f}"
            f"{result['cer_percent']:>11.2f}%"
            f"{result['wer_percent']:>11.2f}%"
            f"{result['exact_match_percent']:>11.2f}%"
        )

    # --------------------------------------------------------
    # Calculate changes relative to baseline
    # --------------------------------------------------------

    baseline = results[0]

    print()
    print("=" * 80)
    print("CHANGE RELATIVE TO BASELINE")
    print("=" * 80)

    for result in results[1:]:

        cer_change = (
            result["cer_percent"]
            - baseline["cer_percent"]
        )

        wer_change = (
            result["wer_percent"]
            - baseline["wer_percent"]
        )

        exact_change = (
            result["exact_match_percent"]
            - baseline["exact_match_percent"]
        )

        print()
        print(result["model"])
        print(f"  CER change:        {cer_change:+.2f} percentage points")
        print(f"  WER change:        {wer_change:+.2f} percentage points")
        print(f"  Exact-match change:{exact_change:+.2f} percentage points")

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "model",
            "test_loss",
            "cer",
            "cer_percent",
            "wer",
            "wer_percent",
            "exact_match",
            "exact_match_percent",
        ])

        for result in results:

            writer.writerow([
                result["model"],
                result["test_loss"],
                result["cer"],
                result["cer_percent"],
                result["wer"],
                result["wer_percent"],
                result["exact_match"],
                result["exact_match_percent"],
            ])

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ALL EXPERIMENTS COMMON TEST SET COMPARISON\n"
        )
        file.write("=" * 70 + "\n\n")

        file.write(
            "All models were evaluated on the same "
            "208 CTC-feasible Indonesian test samples.\n\n"
        )

        file.write(
            "Baseline:       128 x 1024\n"
        )

        file.write(
            "Experiment 1:   128 x 1536\n"
        )

        file.write(
            "Experiment 2:   192 x 1024\n\n"
        )

        for result in results:

            file.write(
                f"{result['model']}\n"
            )

            file.write(
                f"  Test loss: {result['test_loss']:.4f}\n"
            )

            file.write(
                f"  CER: {result['cer_percent']:.2f}%\n"
            )

            file.write(
                f"  WER: {result['wer_percent']:.2f}%\n"
            )

            file.write(
                f"  Exact match: "
                f"{result['exact_match_percent']:.2f}%\n\n"
            )

        file.write(
            "Changes relative to baseline\n"
        )
        file.write("-" * 70 + "\n")

        for result in results[1:]:

            cer_change = (
                result["cer_percent"]
                - baseline["cer_percent"]
            )

            wer_change = (
                result["wer_percent"]
                - baseline["wer_percent"]
            )

            exact_change = (
                result["exact_match_percent"]
                - baseline["exact_match_percent"]
            )

            file.write(
                f"\n{result['model']}\n"
            )

            file.write(
                f"  CER change: "
                f"{cer_change:+.2f} pp\n"
            )

            file.write(
                f"  WER change: "
                f"{wer_change:+.2f} pp\n"
            )

            file.write(
                f"  Exact-match change: "
                f"{exact_change:+.2f} pp\n"
            )

    print()
    print(f"CSV saved to: {OUTPUT_CSV}")
    print(f"Summary saved to: {OUTPUT_SUMMARY}")

    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()