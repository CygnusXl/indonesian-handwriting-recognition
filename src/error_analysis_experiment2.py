import os
import csv
from collections import defaultdict

import torch

from dataset import (
    IndonesianHandwritingDataset,
    CharacterVocabulary,
)
from dataloader import collate_fn
from model_experiment2 import CNNBiLSTMExperiment2


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "checkpoints/experiment2_best.pt"

RESULTS_DIR = "results"

DETAILED_RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "experiment2_error_analysis.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "experiment2_error_analysis_summary.txt"
)

BATCH_SIZE = 4

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 1024
CTC_TIMESTEPS = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VOCABULARY
# ============================================================

def vocabulary_from_checkpoint(checkpoint):
    """
    Reconstruct the exact vocabulary stored in the checkpoint.
    """

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
# CTC GREEDY DECODER
# ============================================================

def greedy_decode(logits, vocabulary):
    """
    Greedy CTC decoding.

    Input:
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

            if token == vocabulary.blank_id:

                previous = token
                continue

            if token != previous:

                decoded_ids.append(token)

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
    Character-level Levenshtein distance.
    """

    reference_length = len(reference)
    hypothesis_length = len(hypothesis)

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
# SAMPLE CER
# ============================================================

def sample_cer(
    reference,
    hypothesis
):
    """
    CER for one sample.
    """

    if len(reference) == 0:

        return 0.0

    return (
        levenshtein_distance(
            reference,
            hypothesis
        )
        / len(reference)
    )


# ============================================================
# TEXT LENGTH CATEGORY
# ============================================================

def length_category(length):

    if length <= 30:
        return "Short (1-30)"

    if length <= 60:
        return "Medium (31-60)"

    if length <= 100:
        return "Long (61-100)"

    return "Very long (101+)"


# ============================================================
# NEATNESS CATEGORY
# ============================================================

def neatness_category(neatness):

    if neatness <= 60:
        return "Very messy (50-60)"

    if neatness <= 70:
        return "Messy (61-70)"

    if neatness <= 80:
        return "Moderate (71-80)"

    if neatness <= 90:
        return "Neat (81-90)"

    return "Very neat (91-95)"


# ============================================================
# GROUP ANALYSIS
# ============================================================

def analyze_group(
    results,
    field
):
    """
    Calculate sample count and mean sample CER
    for a metadata category.
    """

    groups = defaultdict(list)

    for result in results:

        groups[
            result[field]
        ].append(
            result["cer"]
        )

    analysis = []

    for group_name, cers in groups.items():

        analysis.append(
            {
                "group": group_name,
                "samples": len(cers),
                "mean_cer": (
                    sum(cers)
                    / len(cers)
                ),
            }
        )

    return sorted(
        analysis,
        key=lambda item: item["mean_cer"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 2 ERROR ANALYSIS")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

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
    # Load checkpoint
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
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation CER: "
        f"{checkpoint['validation_cer']:.4f}"
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

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
    # Test dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading Experiment 2 test dataset..."
    )

    test_dataset = (
        IndonesianHandwritingDataset(
            split="test",
            vocabulary=vocabulary,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            ctc_timesteps=CTC_TIMESTEPS
        )
    )

    print(
        f"Usable test samples: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Creating Experiment 2 model..."
    )

    model = CNNBiLSTMExperiment2(
        num_classes=vocabulary.num_classes
    )

    model = model.to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # ========================================================
    # COLLECT ERROR DATA
    # ========================================================

    print()
    print("=" * 70)
    print("COLLECTING TEST-SET ERROR DATA")
    print("=" * 70)

    results = []

    with torch.no_grad():

        for start in range(
            0,
            len(test_dataset),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(test_dataset)
            )

            # ------------------------------------------------
            # Get samples
            # ------------------------------------------------

            samples = [
                test_dataset[index]
                for index in range(
                    start,
                    end
                )
            ]

            batch = collate_fn(
                samples
            )

            images = batch[
                "images"
            ].to(DEVICE)

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            logits = model(
                images
            )

            predictions = greedy_decode(
                logits,
                vocabulary
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            for local_index, prediction in enumerate(
                predictions
            ):

                dataset_position = (
                    start + local_index
                )

                original_index = (
                    test_dataset.indices[
                        dataset_position
                    ]
                )

                raw_sample = (
                    test_dataset.dataset[
                        original_index
                    ]
                )

                reference = raw_sample[
                    "text"
                ]

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                cer = sample_cer(
                    reference,
                    prediction
                )

                reference_length = len(
                    reference
                )

                prediction_length = len(
                    prediction
                )

                if reference_length > 0:

                    prediction_ratio = (
                        prediction_length
                        / reference_length
                    )

                else:

                    prediction_ratio = 0.0

                collapsed_prediction = (
                    prediction_ratio < 0.25
                    and reference_length >= 20
                )

                # ------------------------------------------------
                # Metadata
                # ------------------------------------------------

                neatness = raw_sample[
                    "neatness"
                ]

                augmented_noise = raw_sample[
                    "augmented_noise"
                ]

                line_width = raw_sample[
                    "line_width"
                ]

                source = raw_sample[
                    "source"
                ]

                writer_id = raw_sample[
                    "writer_id"
                ]

                multiline = (
                    "\n" in reference
                    or "\\n" in reference
                )

                # ------------------------------------------------
                # Save result
                # ------------------------------------------------

                results.append(
                    {
                        "dataset_index":
                            original_index,

                        "writer_id":
                            writer_id,

                        "source":
                            source,

                        "neatness":
                            neatness,

                        "neatness_category":
                            neatness_category(
                                neatness
                            ),

                        "line_width":
                            line_width,

                        "augmented_noise":
                            augmented_noise,

                        "multiline":
                            multiline,

                        "text_length":
                            reference_length,

                        "length_category":
                            length_category(
                                reference_length
                            ),

                        "prediction_length":
                            prediction_length,

                        "prediction_ratio":
                            prediction_ratio,

                        "collapsed_prediction":
                            collapsed_prediction,

                        "reference":
                            reference,

                        "prediction":
                            prediction,

                        "cer":
                            cer,

                        "exact_match":
                            reference == prediction,
                    }
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                end == len(test_dataset)
                or end % 40 == 0
            ):

                print(
                    f"Processed "
                    f"{end}/"
                    f"{len(test_dataset)}"
                )

    # ========================================================
    # SAVE CSV
    # ========================================================

    print()
    print(
        "Saving detailed error analysis..."
    )

    fieldnames = [
        "dataset_index",
        "writer_id",
        "source",
        "neatness",
        "neatness_category",
        "line_width",
        "augmented_noise",
        "multiline",
        "text_length",
        "length_category",
        "prediction_length",
        "prediction_ratio",
        "collapsed_prediction",
        "reference",
        "prediction",
        "cer",
        "exact_match",
    ]

    with open(
        DETAILED_RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    total_samples = len(
        results
    )

    mean_cer = (
        sum(
            result["cer"]
            for result in results
        )
        / total_samples
    )

    exact_matches = sum(
        result["exact_match"]
        for result in results
    )

    exact_match_rate = (
        exact_matches
        / total_samples
    )

    collapsed_predictions = sum(
        result["collapsed_prediction"]
        for result in results
    )

    collapsed_rate = (
        collapsed_predictions
        / total_samples
    )

    # ========================================================
    # GROUP ANALYSIS
    # ========================================================

    neatness_analysis = analyze_group(
        results,
        "neatness_category"
    )

    noise_analysis = analyze_group(
        results,
        "augmented_noise"
    )

    length_analysis = analyze_group(
        results,
        "length_category"
    )

    multiline_analysis = analyze_group(
        results,
        "multiline"
    )

    line_width_analysis = analyze_group(
        results,
        "line_width"
    )

    source_analysis = analyze_group(
        results,
        "source"
    )

    # ========================================================
    # WRITER ANALYSIS
    # ========================================================

    writer_groups = defaultdict(list)

    for result in results:

        writer_groups[
            result["writer_id"]
        ].append(
            result["cer"]
        )

    writer_analysis = []

    for writer_id, cers in writer_groups.items():

        writer_analysis.append(
            {
                "writer_id":
                    writer_id,

                "samples":
                    len(cers),

                "mean_cer":
                    sum(cers) / len(cers),
            }
        )

    writer_analysis = sorted(
        writer_analysis,
        key=lambda item:
            item["mean_cer"],
        reverse=True
    )

    # ========================================================
    # PRINT GROUP HELPER
    # ========================================================

    def print_group_analysis(
        title,
        analysis
    ):

        print()
        print("=" * 70)
        print(title)
        print("=" * 70)

        for item in analysis:

            print(
                f"{str(item['group']):25s} "
                f"| samples: "
                f"{item['samples']:3d} "
                f"| mean CER: "
                f"{item['mean_cer']:.4f} "
                f"("
                f"{item['mean_cer'] * 100:.2f}%"
                f")"
            )

    # ========================================================
    # PRINT OVERALL
    # ========================================================

    print()
    print("=" * 70)
    print("OVERALL ERROR ANALYSIS")
    print("=" * 70)

    print()

    print(
        f"Test samples: "
        f"{total_samples}"
    )

    print(
        f"Mean sample CER: "
        f"{mean_cer:.4f}"
    )

    print(
        f"Mean sample CER (%): "
        f"{mean_cer * 100:.2f}%"
    )

    print(
        f"Exact matches: "
        f"{exact_matches}/"
        f"{total_samples}"
    )

    print(
        f"Exact-match rate: "
        f"{exact_match_rate * 100:.2f}%"
    )

    print(
        f"Severe prediction collapse: "
        f"{collapsed_predictions}/"
        f"{total_samples}"
    )

    print(
        f"Prediction collapse rate: "
        f"{collapsed_rate * 100:.2f}%"
    )

    # ========================================================
    # GROUP RESULTS
    # ========================================================

    print_group_analysis(
        "CER BY NEATNESS",
        neatness_analysis
    )

    print_group_analysis(
        "CER BY NOISE",
        noise_analysis
    )

    print_group_analysis(
        "CER BY TEXT LENGTH",
        length_analysis
    )

    print_group_analysis(
        "CER BY MULTILINE",
        multiline_analysis
    )

    print_group_analysis(
        "CER BY LINE WIDTH",
        line_width_analysis
    )

    print_group_analysis(
        "CER BY SOURCE",
        source_analysis
    )

    # ========================================================
    # WRITER VARIATION
    # ========================================================

    print()
    print("=" * 70)
    print("WRITER VARIATION")
    print("=" * 70)

    print(
        f"Unique writers in test set: "
        f"{len(writer_analysis)}"
    )

    print()
    print(
        "10 writers with highest mean CER:"
    )

    for item in writer_analysis[:10]:

        print(
            f"Writer {item['writer_id']:>4} "
            f"| samples: "
            f"{item['samples']:3d} "
            f"| mean CER: "
            f"{item['mean_cer']:.4f}"
        )

    print()
    print(
        "10 writers with lowest mean CER:"
    )

    for item in writer_analysis[-10:]:

        print(
            f"Writer {item['writer_id']:>4} "
            f"| samples: "
            f"{item['samples']:3d} "
            f"| mean CER: "
            f"{item['mean_cer']:.4f}"
        )

    # ========================================================
    # WORST EXAMPLES
    # ========================================================

    worst_results = sorted(
        results,
        key=lambda result:
            result["cer"],
        reverse=True
    )

    print()
    print("=" * 70)
    print("10 WORST SAMPLE ERRORS")
    print("=" * 70)

    for index, result in enumerate(
        worst_results[:10],
        start=1
    ):

        print()
        print(
            f"#{index} "
            f"CER={result['cer']:.4f}"
        )

        print(
            f"Writer: "
            f"{result['writer_id']}"
        )

        print(
            f"Neatness: "
            f"{result['neatness']} "
            f"("
            f"{result['neatness_category']}"
            f")"
        )

        print(
            f"Noise: "
            f"{result['augmented_noise']}"
        )

        print(
            f"Line width: "
            f"{result['line_width']}"
        )

        print(
            f"Text length: "
            f"{result['text_length']}"
        )

        print(
            f"Prediction length: "
            f"{result['prediction_length']}"
        )

        print(
            f"Prediction ratio: "
            f"{result['prediction_ratio']:.3f}"
        )

        print(
            f"Multiline: "
            f"{result['multiline']}"
        )

        print(
            f"Reference:\n"
            f"{result['reference']}"
        )

        print(
            f"\nPrediction:\n"
            f"{result['prediction']}"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    print()
    print(
        "Saving summary..."
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "EXPERIMENT 2 ERROR ANALYSIS\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "Configuration\n"
        )

        file.write(
            f"Image height: "
            f"{IMAGE_HEIGHT}\n"
        )

        file.write(
            f"Image width: "
            f"{IMAGE_WIDTH}\n"
        )

        file.write(
            f"CTC time steps: "
            f"{CTC_TIMESTEPS}\n\n"
        )

        file.write(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}\n"
        )

        file.write(
            f"Validation CER: "
            f"{checkpoint['validation_cer']:.6f}\n\n"
        )

        file.write(
            f"Test samples: "
            f"{total_samples}\n"
        )

        file.write(
            f"Mean sample CER: "
            f"{mean_cer:.6f}\n"
        )

        file.write(
            f"Mean sample CER (%): "
            f"{mean_cer * 100:.2f}%\n"
        )

        file.write(
            f"Exact matches: "
            f"{exact_matches}/"
            f"{total_samples}\n"
        )

        file.write(
            f"Exact-match rate: "
            f"{exact_match_rate * 100:.2f}%\n"
        )

        file.write(
            f"Severe prediction collapse: "
            f"{collapsed_predictions}/"
            f"{total_samples}\n"
        )

        file.write(
            f"Prediction collapse rate: "
            f"{collapsed_rate * 100:.2f}%\n"
        )

        # ----------------------------------------------------
        # Group writer
        # ----------------------------------------------------

        def write_group_analysis(
            title,
            analysis
        ):

            file.write(
                "\n"
                + "=" * 70
                + "\n"
            )

            file.write(
                title
                + "\n"
            )

            file.write(
                "=" * 70
                + "\n"
            )

            for item in analysis:

                file.write(
                    f"{item['group']} | "
                    f"samples="
                    f"{item['samples']} | "
                    f"mean_CER="
                    f"{item['mean_cer']:.6f} | "
                    f"{item['mean_cer'] * 100:.2f}%\n"
                )

        write_group_analysis(
            "CER BY NEATNESS",
            neatness_analysis
        )

        write_group_analysis(
            "CER BY NOISE",
            noise_analysis
        )

        write_group_analysis(
            "CER BY TEXT LENGTH",
            length_analysis
        )

        write_group_analysis(
            "CER BY MULTILINE",
            multiline_analysis
        )

        write_group_analysis(
            "CER BY LINE WIDTH",
            line_width_analysis
        )

        write_group_analysis(
            "CER BY SOURCE",
            source_analysis
        )

        # ----------------------------------------------------
        # Writer analysis
        # ----------------------------------------------------

        file.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        file.write(
            "WRITER VARIATION\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        for item in writer_analysis:

            file.write(
                f"Writer {item['writer_id']} | "
                f"samples={item['samples']} | "
                f"mean_CER="
                f"{item['mean_cer']:.6f}\n"
            )

        # ----------------------------------------------------
        # Worst samples
        # ----------------------------------------------------

        file.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        file.write(
            "10 WORST SAMPLES\n"
        )

        file.write(
            "=" * 70
            + "\n"
        )

        for index, result in enumerate(
            worst_results[:10],
            start=1
        ):

            file.write(
                f"\n#{index}\n"
            )

            file.write(
                f"CER: "
                f"{result['cer']:.6f}\n"
            )

            file.write(
                f"Writer: "
                f"{result['writer_id']}\n"
            )

            file.write(
                f"Neatness: "
                f"{result['neatness']}\n"
            )

            file.write(
                f"Noise: "
                f"{result['augmented_noise']}\n"
            )

            file.write(
                f"Line width: "
                f"{result['line_width']}\n"
            )

            file.write(
                f"Text length: "
                f"{result['text_length']}\n"
            )

            file.write(
                f"Prediction length: "
                f"{result['prediction_length']}\n"
            )

            file.write(
                f"Prediction ratio: "
                f"{result['prediction_ratio']:.3f}\n"
            )

            file.write(
                f"Multiline: "
                f"{result['multiline']}\n"
            )

            file.write(
                f"Reference:\n"
                f"{result['reference']}\n"
            )

            file.write(
                f"Prediction:\n"
                f"{result['prediction']}\n"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 2 ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Detailed results:\n"
        f"{DETAILED_RESULTS_PATH}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()