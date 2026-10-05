import os
import csv

import torch
from torch.utils.data import DataLoader

from dataset import IndonesianHandwritingDataset
from dataloader import collate_fn
from model import CNNBiLSTM


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/experiment1_best.pt"
)

RESULTS_DIR = "results"

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "experiment1_test_results.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "experiment1_test_summary.txt"
)

BATCH_SIZE = 2

IMAGE_WIDTH = 1536
CTC_TIMESTEPS = 384

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VOCABULARY
# ============================================================

def vocabulary_from_checkpoint(checkpoint):

    from dataset import CharacterVocabulary

    vocabulary = CharacterVocabulary.__new__(
        CharacterVocabulary
    )

    vocabulary.char_to_id = dict(
        checkpoint["vocabulary_char_to_id"]
    )

    vocabulary.id_to_char = {
        int(index): character
        for index, character
        in checkpoint["vocabulary_id_to_char"].items()
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
# GREEDY CTC DECODING
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

            if token == vocabulary.blank_id:

                previous = token

                continue

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
        range(len(hypothesis) + 1)
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
# CER
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
# WER
# ============================================================

def word_levenshtein_distance(
    reference,
    hypothesis
):

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    previous_row = list(
        range(len(hypothesis_words) + 1)
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
                current_row[j - 1] + 1
            )

            deletion = (
                previous_row[j] + 1
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
# EVALUATION
# ============================================================

def evaluate(
    model,
    dataloader,
    vocabulary,
    criterion
):

    model.eval()

    total_loss = 0.0

    references = []
    hypotheses = []

    results = []

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

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item()

            predictions = greedy_decode(
                logits,
                vocabulary
            )

            for reference, prediction in zip(
                texts,
                predictions
            ):

                distance = (
                    levenshtein_distance(
                        reference,
                        prediction
                    )
                )

                sample_cer = (
                    distance / len(reference)
                    if len(reference) > 0
                    else 0.0
                )

                results.append(
                    {
                        "reference":
                            reference,

                        "prediction":
                            prediction,

                        "cer":
                            sample_cer,

                        "exact_match":
                            reference == prediction,
                    }
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

    wer = calculate_wer(
        references,
        hypotheses
    )

    exact_matches = sum(
        result["exact_match"]
        for result in results
    )

    exact_match_rate = (
        exact_matches
        / len(results)
    )

    return (
        average_loss,
        cer,
        wer,
        exact_match_rate,
        references,
        hypotheses,
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "EXPERIMENT 1 TEST-SET EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        f"Device: {DEVICE}"
    )

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load checkpoint.
    # --------------------------------------------------------

    print()
    print(
        f"Loading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation CER: "
        f"{checkpoint['validation_cer']:.4f}"
    )

    # --------------------------------------------------------
    # Vocabulary.
    # --------------------------------------------------------

    vocabulary = vocabulary_from_checkpoint(
        checkpoint
    )

    print()
    print(
        f"Characters: "
        f"{vocabulary.num_characters}"
    )

    print(
        f"CTC classes: "
        f"{vocabulary.num_classes}"
    )

    # --------------------------------------------------------
    # Test dataset.
    # --------------------------------------------------------

    print()
    print(
        "Loading test dataset..."
    )

    test_dataset = (
        IndonesianHandwritingDataset(
            split="test",
            vocabulary=vocabulary,
            image_width=IMAGE_WIDTH,
            ctc_timesteps=CTC_TIMESTEPS
        )
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    print()
    print(
        "Creating model..."
    )

    model = CNNBiLSTM(
        num_classes=vocabulary.num_classes
    )

    model = model.to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Verify sequence length.
    # --------------------------------------------------------

    test_batch = next(
        iter(test_loader)
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

    if test_logits.size(0) != CTC_TIMESTEPS:

        raise RuntimeError(
            "Unexpected CTC sequence length."
        )

    print(
        "Sequence-length test: PASS"
    )

    # --------------------------------------------------------
    # CTC.
    # --------------------------------------------------------

    criterion = torch.nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True
    )

    # --------------------------------------------------------
    # Evaluate.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING TEST EVALUATION")
    print("=" * 70)

    (
        test_loss,
        test_cer,
        test_wer,
        exact_match_rate,
        references,
        hypotheses,
        results
    ) = evaluate(
        model=model,
        dataloader=test_loader,
        vocabulary=vocabulary,
        criterion=criterion
    )

    # --------------------------------------------------------
    # Results.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT 1 TEST RESULTS")
    print("=" * 70)

    print()
    print(
        f"Test loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test CER: "
        f"{test_cer:.4f}"
    )

    print(
        f"Test CER (%): "
        f"{test_cer * 100:.2f}%"
    )

    print(
        f"Test WER: "
        f"{test_wer:.4f}"
    )

    print(
        f"Test WER (%): "
        f"{test_wer * 100:.2f}%"
    )

    print(
        f"Exact-match rate: "
        f"{exact_match_rate * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Examples.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    for i in range(
        min(10, len(results))
    ):

        print()
        print(
            f"Example {i + 1}"
        )

        print(
            f"Reference:\n"
            f"{references[i]}"
        )

        print(
            f"\nPrediction:\n"
            f"{hypotheses[i]}"
        )

        print(
            f"\nCER: "
            f"{results[i]['cer']:.4f}"
        )

    # --------------------------------------------------------
    # Save CSV.
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "reference",
                "prediction",
                "cer",
                "exact_match"
            ]
        )

        writer.writeheader()

        for result in results:

            writer.writerow(result)

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "EXPERIMENT 1 TEST-SET EVALUATION\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        file.write(
            f"Epoch: "
            f"{checkpoint['epoch']}\n"
        )

        file.write(
            f"Validation CER: "
            f"{checkpoint['validation_cer']:.6f}\n"
        )

        file.write(
            f"Test loss: "
            f"{test_loss:.6f}\n"
        )

        file.write(
            f"Test CER: "
            f"{test_cer:.6f}\n"
        )

        file.write(
            f"Test WER: "
            f"{test_wer:.6f}\n"
        )

        file.write(
            f"Exact-match rate: "
            f"{exact_match_rate:.6f}\n"
        )

        file.write(
            f"Test samples: "
            f"{len(test_dataset)}\n"
        )

        file.write(
            f"Image width: "
            f"{IMAGE_WIDTH}\n"
        )

        file.write(
            f"CTC time steps: "
            f"{CTC_TIMESTEPS}\n"
        )

    print()
    print(
        f"Detailed results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Summary saved to: "
        f"{SUMMARY_PATH}"
    )

    print()
    print("=" * 70)
    print("EXPERIMENT 1 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()