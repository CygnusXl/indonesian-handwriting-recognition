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

BATCH_SIZE = 4

CHECKPOINT_PATH = (
    "checkpoints/baseline_best.pt"
)

RESULTS_DIR = "results"

TEST_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "baseline_test_results.csv"
)

TEST_SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "baseline_test_summary.txt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VOCABULARY FROM CHECKPOINT
# ============================================================

def vocabulary_from_checkpoint(
    checkpoint
):
    """
    Reconstruct the CharacterVocabulary from the mappings
    stored inside the trained checkpoint.

    This guarantees that test-set decoding uses exactly the
    same character-to-ID mapping used during training.
    """

    char_to_id = checkpoint[
        "vocabulary_char_to_id"
    ]

    id_to_char = checkpoint[
        "vocabulary_id_to_char"
    ]

    # Import the class only so we can recreate the object.
    from dataset import CharacterVocabulary

    vocabulary = CharacterVocabulary.__new__(
        CharacterVocabulary
    )

    vocabulary.char_to_id = char_to_id
    vocabulary.id_to_char = {
        int(index): character
        for index, character
        in id_to_char.items()
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
    """
    Decode CTC predictions using greedy decoding.

    Input:
        logits:
            [time, batch, classes]

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

            # ------------------------------------------------
            # Ignore CTC blank.
            # ------------------------------------------------

            if token == vocabulary.blank_id:

                previous = token

                continue

            # ------------------------------------------------
            # Collapse consecutive duplicate tokens.
            # ------------------------------------------------

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

def levenshtein_distance(
    reference,
    hypothesis
):
    """
    Calculate Levenshtein edit distance.
    """

    reference_length = len(
        reference
    )

    hypothesis_length = len(
        hypothesis
    )

    previous_row = list(
        range(hypothesis_length + 1)
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
# WORD LEVENSHTEIN DISTANCE
# ============================================================

def word_levenshtein_distance(
    reference,
    hypothesis
):
    """
    Calculate word-level Levenshtein distance.

    This is used to calculate WER.
    """

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    reference_length = len(
        reference_words
    )

    hypothesis_length = len(
        hypothesis_words
    )

    previous_row = list(
        range(hypothesis_length + 1)
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
# CHARACTER ERROR RATE
# ============================================================

def calculate_cer(
    references,
    hypotheses
):
    """
    Calculate overall Character Error Rate.

    CER =
        total character edit distance
        /
        total reference characters
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
# WORD ERROR RATE
# ============================================================

def calculate_wer(
    references,
    hypotheses
):
    """
    Calculate overall Word Error Rate.

    WER =
        total word edit distance
        /
        total reference words
    """

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
# TEST EVALUATION
# ============================================================

def evaluate(
    model,
    dataloader,
    vocabulary,
    criterion
):
    """
    Evaluate the trained model on the test set.
    """

    model.eval()

    total_loss = 0.0

    references = []
    hypotheses = []

    # Store detailed per-sample information.
    sample_results = []

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
            # Convert logits to log probabilities.
            # ------------------------------------------------

            log_probs = torch.log_softmax(
                logits,
                dim=2
            )

            # ------------------------------------------------
            # All images produce 256 CTC time steps.
            # ------------------------------------------------

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
            # Test loss
            # ------------------------------------------------

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths
            )

            total_loss += loss.item()

            # ------------------------------------------------
            # Decode predictions.
            # ------------------------------------------------

            predictions = greedy_decode(
                logits,
                vocabulary
            )

            # ------------------------------------------------
            # Store results.
            # ------------------------------------------------

            for reference, hypothesis in zip(
                texts,
                predictions
            ):

                distance = (
                    levenshtein_distance(
                        reference,
                        hypothesis
                    )
                )

                if len(reference) > 0:

                    sample_cer = (
                        distance
                        / len(reference)
                    )

                else:

                    sample_cer = 0.0

                exact_match = (
                    reference == hypothesis
                )

                sample_results.append(
                    {
                        "reference": reference,
                        "prediction": hypothesis,
                        "cer": sample_cer,
                        "exact_match": exact_match,
                    }
                )

            references.extend(
                texts
            )

            hypotheses.extend(
                predictions
            )

    # --------------------------------------------------------
    # Overall metrics.
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

    exact_matches = sum(
        result["exact_match"]
        for result in sample_results
    )

    exact_match_rate = (
        exact_matches
        / len(sample_results)
    )

    return (
        average_loss,
        cer,
        wer,
        exact_match_rate,
        references,
        hypotheses,
        sample_results,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    sample_results
):
    """
    Save every test-set reference and prediction to CSV.

    This will also be useful later for error analysis.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        TEST_RESULTS_PATH,
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
                "exact_match",
            ]
        )

        writer.writeheader()

        for result in sample_results:

            writer.writerow(
                {
                    "reference":
                        result["reference"],

                    "prediction":
                        result["prediction"],

                    "cer":
                        result["cer"],

                    "exact_match":
                        result["exact_match"],
                }
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BASELINE TEST-SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
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

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    print()
    print(
        f"Loading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}\n"
            f"Run the baseline training first."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation CER: "
        f"{checkpoint['validation_cer']:.4f}"
    )

    # --------------------------------------------------------
    # Reconstruct vocabulary.
    # --------------------------------------------------------

    print()
    print(
        "Loading vocabulary from checkpoint..."
    )

    vocabulary = vocabulary_from_checkpoint(
        checkpoint
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
    # Create test dataset.
    # --------------------------------------------------------

    print()
    print(
        "Loading test dataset..."
    )

    test_dataset = (
        IndonesianHandwritingDataset(
            split="test",
            vocabulary=vocabulary
        )
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Create test DataLoader.
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"Test batches: "
        f"{len(test_loader)}"
    )

    # --------------------------------------------------------
    # Create model.
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

    # --------------------------------------------------------
    # Load trained parameters.
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Model checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # CTC loss.
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
    print("EVALUATING TEST SET")
    print("=" * 70)

    (
        test_loss,
        test_cer,
        test_wer,
        exact_match_rate,
        references,
        hypotheses,
        sample_results,
    ) = evaluate(
        model=model,
        dataloader=test_loader,
        vocabulary=vocabulary,
        criterion=criterion
    )

    # --------------------------------------------------------
    # Print metrics.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST-SET RESULTS")
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
        f"{exact_match_rate:.4f}"
    )

    print(
        f"Exact-match rate (%): "
        f"{exact_match_rate * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Example predictions.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXAMPLE TEST PREDICTIONS")
    print("=" * 70)

    number_of_examples = min(
        10,
        len(references)
    )

    for i in range(
        number_of_examples
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

        sample_cer = sample_results[
            i
        ]["cer"]

        print(
            f"\nSample CER: "
            f"{sample_cer:.4f}"
        )

    # --------------------------------------------------------
    # Find worst predictions.
    # --------------------------------------------------------

    worst_results = sorted(
        sample_results,
        key=lambda result: result["cer"],
        reverse=True
    )

    print()
    print("=" * 70)
    print("WORST TEST PREDICTIONS")
    print("=" * 70)

    number_of_worst = min(
        10,
        len(worst_results)
    )

    for i in range(
        number_of_worst
    ):

        result = worst_results[i]

        print()
        print(
            f"Worst #{i + 1}"
        )

        print(
            f"CER: "
            f"{result['cer']:.4f}"
        )

        print(
            f"Reference:\n"
            f"{result['reference']}"
        )

        print(
            f"\nPrediction:\n"
            f"{result['prediction']}"
        )

    # --------------------------------------------------------
    # Save detailed CSV.
    # --------------------------------------------------------

    save_results(
        sample_results
    )

    print()
    print(
        f"Detailed test results saved to:\n"
        f"{TEST_RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # Save summary.
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        TEST_SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "BASELINE TEST-SET EVALUATION\n"
        )

        file.write(
            "========================================\n"
        )

        file.write(
            f"Checkpoint epoch: "
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
            f"Test CER (%): "
            f"{test_cer * 100:.2f}%\n"
        )

        file.write(
            f"Test WER: "
            f"{test_wer:.6f}\n"
        )

        file.write(
            f"Test WER (%): "
            f"{test_wer * 100:.2f}%\n"
        )

        file.write(
            f"Exact-match rate: "
            f"{exact_match_rate:.6f}\n"
        )

        file.write(
            f"Exact-match rate (%): "
            f"{exact_match_rate * 100:.2f}%\n"
        )

        file.write(
            f"Test samples: "
            f"{len(test_dataset)}\n"
        )

    print(
        f"Summary saved to:\n"
        f"{TEST_SUMMARY_PATH}"
    )

    # --------------------------------------------------------
    # Complete.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()