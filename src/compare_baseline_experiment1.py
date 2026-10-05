import os
import csv

import torch
from torch.utils.data import DataLoader

from dataset import (
    CharacterVocabulary,
    IndonesianHandwritingDataset,
)
from dataloader import collate_fn
from model import CNNBiLSTM


# ============================================================
# CONFIGURATION
# ============================================================

BASELINE_CHECKPOINT = (
    "checkpoints/baseline_best.pt"
)

EXPERIMENT1_CHECKPOINT = (
    "checkpoints/experiment1_best.pt"
)

RESULTS_DIR = "results"

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "baseline_vs_experiment1_common_test.txt"
)

BATCH_SIZE = 2

# ------------------------------------------------------------
# IMPORTANT:
#
# Both models are evaluated on the SAME 208 samples.
#
# The baseline has 256 CTC time steps.
# Therefore the common test population is defined using
# the 256-step feasibility criterion.
# ------------------------------------------------------------

COMMON_CTC_TIMESTEPS = 256

BASELINE_WIDTH = 1024
EXPERIMENT1_WIDTH = 1536

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# VOCABULARY
# ============================================================

def vocabulary_from_checkpoint(
    checkpoint
):

    vocabulary = CharacterVocabulary.__new__(
        CharacterVocabulary
    )

    vocabulary.char_to_id = dict(
        checkpoint[
            "vocabulary_char_to_id"
        ]
    )

    vocabulary.id_to_char = {
        int(index): character
        for index, character
        in checkpoint[
            "vocabulary_id_to_char"
        ].items()
    }

    vocabulary.num_characters = len(
        vocabulary.char_to_id
    )

    vocabulary.blank_id = (
        vocabulary.num_characters
    )

    vocabulary.num_classes = (
        vocabulary.num_characters + 1
    )

    return vocabulary


# ============================================================
# GREEDY CTC DECODER
# ============================================================

def greedy_decode(
    logits,
    vocabulary
):

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

            # Ignore blank.
            if token == vocabulary.blank_id:

                previous = token
                continue

            # Collapse repeated predictions.
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
# LEVENSHTEIN
# ============================================================

def levenshtein_distance(
    reference,
    hypothesis
):

    previous_row = list(
        range(
            len(hypothesis) + 1
        )
    )

    for i in range(
        1,
        len(reference) + 1
    ):

        current_row = [i]

        for j in range(
            1,
            len(hypothesis) + 1
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
# WORD LEVENSHTEIN
# ============================================================

def word_levenshtein_distance(
    reference,
    hypothesis
):

    reference_words = (
        reference.split()
    )

    hypothesis_words = (
        hypothesis.split()
    )

    previous_row = list(
        range(
            len(hypothesis_words) + 1
        )
    )

    for i in range(
        1,
        len(reference_words) + 1
    ):

        current_row = [i]

        for j in range(
            1,
            len(hypothesis_words) + 1
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
                    reference_words[i - 1]
                    != hypothesis_words[j - 1]
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
# WORD ERROR RATE
# ============================================================

def calculate_wer(
    references,
    hypotheses
):

    total_distance = 0
    total_words = 0

    for reference, hypothesis in zip(
        references,
        hypotheses
    ):

        total_distance += (
            word_levenshtein_distance(
                reference,
                hypothesis
            )
        )

        total_words += len(
            reference.split()
        )

    if total_words == 0:

        return 0.0

    return (
        total_distance
        / total_words
    )


# ============================================================
# EXACT MATCH
# ============================================================

def calculate_exact_match(
    references,
    hypotheses
):

    matches = sum(
        reference == hypothesis
        for reference, hypothesis
        in zip(
            references,
            hypotheses
        )
    )

    return (
        matches
        / len(references)
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    dataloader,
    vocabulary,
    checkpoint_name
):

    model.eval()

    references = []
    hypotheses = []

    total_loss = 0.0

    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True
    )

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

            logits = model(
                images
            )

            # ------------------------------------------------
            # CTC loss
            # ------------------------------------------------

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

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # Decode
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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    average_loss = (
        total_loss
        / len(dataloader)
    )

    cer = calculate_cer(
        references,
        hypotheses
    )

    wer = calculate_wer(
        references,
        hypotheses
    )

    exact_match = (
        calculate_exact_match(
            references,
            hypotheses
        )
    )

    print()
    print(
        f"{checkpoint_name} results:"
    )

    print(
        f"  Test loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"  Test CER: "
        f"{cer:.4f} "
        f"({cer * 100:.2f}%)"
    )

    print(
        f"  Test WER: "
        f"{wer:.4f} "
        f"({wer * 100:.2f}%)"
    )

    print(
        f"  Exact match: "
        f"{exact_match:.4f} "
        f"({exact_match * 100:.2f}%)"
    )

    return {
        "name": checkpoint_name,
        "loss": average_loss,
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match,
        "references": references,
        "hypotheses": hypotheses,
    }


# ============================================================
# CREATE DATASETS WITH COMMON TEST POPULATION
# ============================================================

def create_dataset(
    image_width,
    vocabulary
):

    dataset = (
        IndonesianHandwritingDataset(
            split="test",
            vocabulary=vocabulary,
            image_width=image_width,
            ctc_timesteps=COMMON_CTC_TIMESTEPS
        )
    )

    return dataset


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BASELINE VS EXPERIMENT 1"
    )
    print(
        "COMMON TEST-SET COMPARISON"
    )
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
    # Checkpoints
    # --------------------------------------------------------

    if not os.path.exists(
        BASELINE_CHECKPOINT
    ):

        raise FileNotFoundError(
            BASELINE_CHECKPOINT
        )

    if not os.path.exists(
        EXPERIMENT1_CHECKPOINT
    ):

        raise FileNotFoundError(
            EXPERIMENT1_CHECKPOINT
        )

    print()
    print(
        "Loading checkpoints..."
    )

    baseline_checkpoint = (
        torch.load(
            BASELINE_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False
        )
    )

    experiment1_checkpoint = (
        torch.load(
            EXPERIMENT1_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False
        )
    )

    print(
        f"Baseline best epoch: "
        f"{baseline_checkpoint['epoch']}"
    )

    print(
        f"Baseline validation CER: "
        f"{baseline_checkpoint['validation_cer']:.4f}"
    )

    print(
        f"Experiment 1 best epoch: "
        f"{experiment1_checkpoint['epoch']}"
    )

    print(
        f"Experiment 1 validation CER: "
        f"{experiment1_checkpoint['validation_cer']:.4f}"
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    baseline_vocabulary = (
        vocabulary_from_checkpoint(
            baseline_checkpoint
        )
    )

    experiment1_vocabulary = (
        vocabulary_from_checkpoint(
            experiment1_checkpoint
        )
    )

    # --------------------------------------------------------
    # Verify vocabularies are identical.
    # --------------------------------------------------------

    if (
        baseline_vocabulary.char_to_id
        != experiment1_vocabulary.char_to_id
    ):

        raise RuntimeError(
            "Baseline and Experiment 1 "
            "vocabularies are different."
        )

    vocabulary = baseline_vocabulary

    print()
    print(
        "Vocabulary consistency: PASS"
    )

    # --------------------------------------------------------
    # Create the TWO datasets.
    #
    # Both use:
    #
    #     ctc_timesteps = 256
    #
    # This forces them to select exactly the same 208
    # test samples.
    #
    # Their image widths differ.
    #
    # Baseline:
    #     1024
    #
    # Experiment 1:
    #     1536
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Creating common test datasets..."
    )

    baseline_dataset = create_dataset(
        BASELINE_WIDTH,
        vocabulary
    )

    experiment1_dataset = create_dataset(
        EXPERIMENT1_WIDTH,
        vocabulary
    )

    print()

    print(
        f"Baseline test samples: "
        f"{len(baseline_dataset)}"
    )

    print(
        f"Experiment 1 test samples: "
        f"{len(experiment1_dataset)}"
    )

    # --------------------------------------------------------
    # Verify same original dataset indices.
    # --------------------------------------------------------

    if (
        baseline_dataset.indices
        != experiment1_dataset.indices
    ):

        raise RuntimeError(
            "The two test datasets do not "
            "contain the exact same samples."
        )

    print(
        "Common test-set consistency: PASS"
    )

    print(
        f"Common test samples: "
        f"{len(baseline_dataset)}"
    )

    # --------------------------------------------------------
    # Create DataLoaders.
    # --------------------------------------------------------

    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    experiment1_loader = DataLoader(
        experiment1_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Load baseline model.
    # --------------------------------------------------------

    print()
    print(
        "Loading baseline model..."
    )

    baseline_model = CNNBiLSTM(
        num_classes=vocabulary.num_classes
    )

    baseline_model = (
        baseline_model.to(DEVICE)
    )

    baseline_model.load_state_dict(
        baseline_checkpoint[
            "model_state_dict"
        ]
    )

    baseline_model.eval()

    # --------------------------------------------------------
    # Verify baseline output sequence.
    # --------------------------------------------------------

    baseline_test_batch = next(
        iter(baseline_loader)
    )

    baseline_test_images = (
        baseline_test_batch[
            "images"
        ].to(DEVICE)
    )

    with torch.no_grad():

        baseline_test_output = (
            baseline_model(
                baseline_test_images
            )
        )

    print(
        f"Baseline input shape: "
        f"{tuple(baseline_test_images.shape)}"
    )

    print(
        f"Baseline output shape: "
        f"{tuple(baseline_test_output.shape)}"
    )

    # --------------------------------------------------------
    # Load Experiment 1 model.
    # --------------------------------------------------------

    print()
    print(
        "Loading Experiment 1 model..."
    )

    experiment1_model = CNNBiLSTM(
        num_classes=vocabulary.num_classes
    )

    experiment1_model = (
        experiment1_model.to(DEVICE)
    )

    experiment1_model.load_state_dict(
        experiment1_checkpoint[
            "model_state_dict"
        ]
    )

    experiment1_model.eval()

    # --------------------------------------------------------
    # Verify Experiment 1 output sequence.
    # --------------------------------------------------------

    experiment1_test_batch = next(
        iter(experiment1_loader)
    )

    experiment1_test_images = (
        experiment1_test_batch[
            "images"
        ].to(DEVICE)
    )

    with torch.no_grad():

        experiment1_test_output = (
            experiment1_model(
                experiment1_test_images
            )
        )

    print(
        f"Experiment 1 input shape: "
        f"{tuple(experiment1_test_images.shape)}"
    )

    print(
        f"Experiment 1 output shape: "
        f"{tuple(experiment1_test_output.shape)}"
    )

    # --------------------------------------------------------
    # Evaluate both.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATING COMMON TEST SET")
    print("=" * 70)

    baseline_results = evaluate_model(
        model=baseline_model,
        dataloader=baseline_loader,
        vocabulary=vocabulary,
        checkpoint_name="BASELINE"
    )

    experiment1_results = evaluate_model(
        model=experiment1_model,
        dataloader=experiment1_loader,
        vocabulary=vocabulary,
        checkpoint_name="EXPERIMENT 1"
    )

    # --------------------------------------------------------
    # Calculate changes.
    # --------------------------------------------------------

    cer_change = (
        experiment1_results["cer"]
        - baseline_results["cer"]
    )

    wer_change = (
        experiment1_results["wer"]
        - baseline_results["wer"]
    )

    exact_change = (
        experiment1_results["exact_match"]
        - baseline_results["exact_match"]
    )

    # Negative CER change = improvement.
    cer_improvement = (
        baseline_results["cer"]
        - experiment1_results["cer"]
    )

    # --------------------------------------------------------
    # Print comparison.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMMON TEST-SET COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Metric':<20}"
        f"{'Baseline':>15}"
        f"{'Experiment 1':>18}"
        f"{'Change':>15}"
    )

    print("-" * 68)

    print(
        f"{'Test Loss':<20}"
        f"{baseline_results['loss']:>15.4f}"
        f"{experiment1_results['loss']:>18.4f}"
        f"{experiment1_results['loss'] - baseline_results['loss']:>15.4f}"
    )

    print(
        f"{'Test CER':<20}"
        f"{baseline_results['cer'] * 100:>14.2f}%"
        f"{experiment1_results['cer'] * 100:>17.2f}%"
        f"{cer_change * 100:>14.2f} pp"
    )

    print(
        f"{'Test WER':<20}"
        f"{baseline_results['wer'] * 100:>14.2f}%"
        f"{experiment1_results['wer'] * 100:>17.2f}%"
        f"{wer_change * 100:>14.2f} pp"
    )

    print(
        f"{'Exact Match':<20}"
        f"{baseline_results['exact_match'] * 100:>14.2f}%"
        f"{experiment1_results['exact_match'] * 100:>17.2f}%"
        f"{exact_change * 100:>14.2f} pp"
    )

    print()
    print(
        f"Common test-set size: "
        f"{len(baseline_dataset)}"
    )

    # --------------------------------------------------------
    # Conclusion.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT 1 CONCLUSION")
    print("=" * 70)

    if cer_improvement > 0:

        print()
        print(
            "Experiment 1 improved CER on the "
            "common test set."
        )

        print(
            f"CER improvement: "
            f"{cer_improvement * 100:.2f} "
            f"percentage points"
        )

    elif cer_improvement < 0:

        print()
        print(
            "Experiment 1 performed worse than "
            "the baseline on the common test set."
        )

        print(
            f"CER change: "
            f"{cer_change * 100:.2f} "
            f"percentage points"
        )

    else:

        print()
        print(
            "Experiment 1 produced the same CER "
            "as the baseline."
        )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "BASELINE VS EXPERIMENT 1\n"
        )

        file.write(
            "COMMON TEST-SET COMPARISON\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            f"Common test samples: "
            f"{len(baseline_dataset)}\n"
        )

        file.write(
            f"Baseline image width: "
            f"{BASELINE_WIDTH}\n"
        )

        file.write(
            f"Experiment 1 image width: "
            f"{EXPERIMENT1_WIDTH}\n"
        )

        file.write(
            f"Common CTC feasibility threshold: "
            f"{COMMON_CTC_TIMESTEPS}\n\n"
        )

        file.write(
            f"Baseline test loss: "
            f"{baseline_results['loss']:.6f}\n"
        )

        file.write(
            f"Experiment 1 test loss: "
            f"{experiment1_results['loss']:.6f}\n\n"
        )

        file.write(
            f"Baseline test CER: "
            f"{baseline_results['cer']:.6f}\n"
        )

        file.write(
            f"Experiment 1 test CER: "
            f"{experiment1_results['cer']:.6f}\n"
        )

        file.write(
            f"CER change: "
            f"{cer_change:.6f}\n\n"
        )

        file.write(
            f"Baseline test WER: "
            f"{baseline_results['wer']:.6f}\n"
        )

        file.write(
            f"Experiment 1 test WER: "
            f"{experiment1_results['wer']:.6f}\n"
        )

        file.write(
            f"WER change: "
            f"{wer_change:.6f}\n\n"
        )

        file.write(
            f"Baseline exact match: "
            f"{baseline_results['exact_match']:.6f}\n"
        )

        file.write(
            f"Experiment 1 exact match: "
            f"{experiment1_results['exact_match']:.6f}\n"
        )

        file.write(
            f"Exact match change: "
            f"{exact_change:.6f}\n"
        )

    print()
    print(
        f"Summary saved to:\n"
        f"{SUMMARY_PATH}"
    )

    print()
    print("=" * 70)
    print("COMMON TEST COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()