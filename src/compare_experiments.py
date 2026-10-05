import csv
import os

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
from model_experiment2 import CNNBiLSTMExperiment2
from evaluate import (
    greedy_decode,
    levenshtein_distance,
    calculate_cer,
    calculate_wer,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

COMMON_CTC_TIMESTEPS = 256
BATCH_SIZE = 4
NUM_WORKERS = 0

CHECKPOINTS = {
    "Baseline": {
        "path": "checkpoints/baseline_best.pt",
        "height": 128,
        "width": 1024,
    },
    "Experiment 1": {
        "path": "checkpoints/experiment1_best.pt",
        "height": 128,
        "width": 1536,
    },
    "Experiment 2": {
        "path": "checkpoints/experiment2_best.pt",
        "height": 192,
        "width": 1024,
    },
    "Experiment 3": {
        "path": "checkpoints/experiment3_best.pt",
        "height": 128,
        "width": 1024,
    },
}

RESULTS_DIR = "results"
CSV_PATH = os.path.join(
    RESULTS_DIR,
    "all_experiments_standardized.csv",
)
SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "all_experiments_standardized_summary.txt",
)


def macro_cer(
    references,
    hypotheses,
):
    sample_cers = []

    for reference, hypothesis in zip(
        references,
        hypotheses,
    ):
        if not reference:
            sample_cers.append(0.0)
        else:
            sample_cers.append(
                levenshtein_distance(
                    reference,
                    hypothesis,
                )
                / len(reference)
            )

    if not sample_cers:
        return 0.0

    return sum(sample_cers) / len(sample_cers)


def exact_match_rate(
    references,
    hypotheses,
):
    if not references:
        return 0.0

    matches = sum(
        reference == hypothesis
        for reference, hypothesis
        in zip(references, hypotheses)
    )

    return matches / len(references)


def create_model(
    name,
    vocabulary,
):
    if name == "Experiment 2":
        return CNNBiLSTMExperiment2(
            num_classes=vocabulary.num_classes,
            image_height=192,
        ).to(DEVICE)

    return CNNBiLSTM(
        num_classes=vocabulary.num_classes
    ).to(DEVICE)


def evaluate_model(
    name,
    config,
    vocabulary,
    common_indices,
):
    checkpoint_path = config["path"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    dataset = IndonesianHandwritingDataset(
        split="test",
        vocabulary=vocabulary,
        image_height=config["height"],
        image_width=config["width"],
        ctc_timesteps=COMMON_CTC_TIMESTEPS,
    )

    if dataset.indices != common_indices:
        raise RuntimeError(
            f"{name} does not use the same common-test "
            f"sample indices."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(
        name,
        vocabulary,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = nn.CTCLoss(
        blank=vocabulary.blank_id,
        zero_infinity=True,
    )

    model.eval()

    total_loss = 0.0
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            label_lengths = batch[
                "label_lengths"
            ].to(DEVICE)
            texts = batch["texts"]

            logits = model(images)

            log_probs = torch.log_softmax(
                logits,
                dim=2,
            )

            time_steps = logits.size(0)

            input_lengths = torch.full(
                (images.size(0),),
                time_steps,
                dtype=torch.long,
                device=DEVICE,
            )

            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                label_lengths,
            )

            total_loss += loss.item()

            predictions = greedy_decode(
                logits,
                vocabulary,
            )

            references.extend(texts)
            hypotheses.extend(predictions)

    return {
        "model": name,
        "epoch": checkpoint.get(
            "epoch",
            "",
        ),
        "input": (
            f"{config['height']} x "
            f"{config['width']}"
        ),
        "test_samples": len(dataset),
        "test_loss": (
            total_loss
            / len(loader)
        ),
        "corpus_cer": calculate_cer(
            references,
            hypotheses,
        ),
        "corpus_wer": calculate_wer(
            references,
            hypotheses,
        ),
        "macro_cer": macro_cer(
            references,
            hypotheses,
        ),
        "exact_match": exact_match_rate(
            references,
            hypotheses,
        ),
    }


def main():
    print("=" * 70)
    print("STANDARDIZED EXPERIMENT COMPARISON")
    print("=" * 70)

    print()
    print("Primary metrics:")
    print("  CER = corpus-level")
    print("  WER = corpus-level")
    print("  Exact match = sample-level")
    print("  Macro-CER = secondary diagnostic")

    raw_train_dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    vocabulary = CharacterVocabulary(
        raw_train_dataset
    )

    reference_dataset = IndonesianHandwritingDataset(
        split="test",
        vocabulary=vocabulary,
        image_height=128,
        image_width=1024,
        ctc_timesteps=COMMON_CTC_TIMESTEPS,
    )

    common_indices = list(
        reference_dataset.indices
    )

    print()
    print(
        f"Common test samples: "
        f"{len(common_indices)}"
    )

    results = []

    for name, config in CHECKPOINTS.items():
        print()
        print("=" * 70)
        print(
            f"Evaluating {name}"
        )
        print("=" * 70)

        result = evaluate_model(
            name,
            config,
            vocabulary,
            common_indices,
        )

        results.append(result)

        print(
            f"Corpus CER: "
            f"{result['corpus_cer'] * 100:.2f}%"
        )
        print(
            f"Corpus WER: "
            f"{result['corpus_wer'] * 100:.2f}%"
        )
        print(
            f"Macro-CER: "
            f"{result['macro_cer'] * 100:.2f}%"
        )
        print(
            f"Exact match: "
            f"{result['exact_match'] * 100:.2f}%"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "epoch",
                "input",
                "test_samples",
                "test_loss",
                "corpus_cer",
                "corpus_wer",
                "macro_cer",
                "exact_match",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    baseline = results[0]

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "STANDARDIZED EXPERIMENT COMPARISON\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "All models use the same common test samples "
            f"({len(common_indices)} samples), selected with "
            "the same CTC feasibility threshold of 256.\n"
        )

        file.write(
            "Primary comparison metrics are corpus-level CER "
            "and corpus-level WER. Macro-CER is retained "
            "as a secondary per-sample diagnostic.\n\n"
        )

        for result in results:
            file.write(
                f"{result['model']}\n"
            )

            file.write(
                f"  Input: {result['input']}\n"
            )

            file.write(
                f"  Checkpoint epoch: "
                f"{result['epoch']}\n"
            )

            file.write(
                f"  Test loss: "
                f"{result['test_loss']:.6f}\n"
            )

            file.write(
                f"  Corpus CER: "
                f"{result['corpus_cer']:.6f} "
                f"({result['corpus_cer'] * 100:.2f}%)\n"
            )

            file.write(
                f"  Corpus WER: "
                f"{result['corpus_wer']:.6f} "
                f"({result['corpus_wer'] * 100:.2f}%)\n"
            )

            file.write(
                f"  Macro-CER: "
                f"{result['macro_cer']:.6f} "
                f"({result['macro_cer'] * 100:.2f}%)\n"
            )

            file.write(
                f"  Exact match: "
                f"{result['exact_match']:.6f} "
                f"({result['exact_match'] * 100:.2f}%)\n\n"
            )

        file.write(
            "Changes relative to baseline using corpus metrics:\n"
        )

        for result in results[1:]:
            file.write(
                f"{result['model']}: "
                f"CER {(result['corpus_cer'] - baseline['corpus_cer']) * 100:+.2f} pp, "
                f"WER {(result['corpus_wer'] - baseline['corpus_wer']) * 100:+.2f} pp, "
                f"Exact match {(result['exact_match'] - baseline['exact_match']) * 100:+.2f} pp\n"
            )

    print()
    print("=" * 70)
    print("STANDARDIZED COMPARISON COMPLETE")
    print("=" * 70)

    print()
    print(
        f"CSV: {CSV_PATH}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
