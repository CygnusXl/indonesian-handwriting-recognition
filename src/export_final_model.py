"""
Export a trained checkpoint as the slim DEPLOYMENT checkpoint used by
the application.

A training checkpoint contains optimizer state (about 3x the size of the
weights). The deployment checkpoint keeps only what inference needs plus
the provenance required to trace the file back to a specific training
run:

    models/final_model.pt        weights + vocabulary + input config
    models/final_model.json      human-readable provenance / metrics

Usage (repository root):

    python src/export_final_model.py --model baseline

The exported file stores the model configuration (architecture, input
size, CTC time steps) so the app never has to guess it, and can be
loaded with torch.load(..., weights_only=True).

If results/analysis/<model>_analysis.json exists, the test metrics from
that real evaluation run are copied into the provenance file; otherwise
no metrics are written (nothing is ever invented).
"""

import argparse
import json
import os
from datetime import datetime, timezone

import torch

import htr_common as hc


def iam_test_metrics(summary_path, checkpoint_sha):
    """
    Pick the IAM numbers measured on EXACTLY this checkpoint from
    evaluate_iam.py's summary.json. Returns {} (with a warning) if the
    file has no entry for it, so nothing is ever attributed to the
    wrong model.
    """

    with open(summary_path, encoding="utf-8") as file:
        summary = json.load(file)

    for name, entry in summary["checkpoints"].items():
        if entry["sha256"] != checkpoint_sha:
            continue

        pads = entry["pads"]
        raw = pads.get("0.0") or next(iter(pads.values()))
        keys = ("samples", "corpus_cer", "corpus_wer", "exact_match")

        result = {
            "test_metrics": {key: raw[key] for key in keys},
            "test_metrics_label": "IAM test set, raw lines",
            "test_metrics_source": summary_path,
        }

        if "0.25" in pads:
            result["test_metrics_page_margin"] = {
                key: pads["0.25"][key] for key in keys
            }

        return result

    print(
        "WARNING: the IAM summary has no entry for this checkpoint; test "
        "metrics NOT copied."
    )

    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        choices=list(hc.MODEL_REGISTRY),
        help="registry entry (default: baseline, unless --checkpoint is "
        "given, then the checkpoint's own settings are used)",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--label",
        default=None,
        help="name shown in the app (default: from the checkpoint)",
    )
    parser.add_argument(
        "--iam-summary",
        default=None,
        help="results/iam_eval/summary.json from evaluate_iam.py; its "
        "numbers are copied only if they were measured on this exact "
        "checkpoint",
    )
    parser.add_argument(
        "--output", default=os.path.join("models", "final_model.pt")
    )
    parser.add_argument(
        "--analysis-dir", default=os.path.join("results", "analysis")
    )
    args = parser.parse_args()

    if args.model is None and args.checkpoint is None:
        args.model = "baseline"

    registry = hc.MODEL_REGISTRY.get(args.model) if args.model else None
    source = args.checkpoint or registry["checkpoint"]

    checkpoint = hc.read_checkpoint(source)
    vocabulary = hc.vocabulary_from_checkpoint(checkpoint)
    config = hc.infer_config(checkpoint, source)

    # Rebuild + load to prove the weights match the declared config.
    model = hc.build_model(
        config["architecture"],
        vocabulary.num_classes,
        config["image_height"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    source_sha = hc.file_sha256(source)

    deployment = {
        "model_state_dict": checkpoint["model_state_dict"],
        "vocabulary_char_to_id": checkpoint["vocabulary_char_to_id"],
        "vocabulary_id_to_char": {
            int(k): v for k, v in checkpoint["vocabulary_id_to_char"].items()
        },
        "num_classes": vocabulary.num_classes,
        "architecture": config["architecture"],
        "image_height": config["image_height"],
        "image_width": config["image_width"],
        "ctc_timesteps": config["ctc_timesteps"],
        "source_experiment": args.model or "custom",
        "source_checkpoint": os.path.basename(source),
        "source_checkpoint_sha256": source_sha,
        "source_epoch": checkpoint.get("epoch"),
        "source_validation_cer": checkpoint.get("validation_cer"),
    }

    # Training provenance, when the training script recorded it.
    training_fields = {
        key: checkpoint[key]
        for key in (
            "training_data",
            "training_data_label",
            "training_dataset_source",
            "augmentation",
            "train_lines",
            "validation_lines",
            "seed",
        )
        if key in checkpoint
    }
    deployment.update(training_fields)

    label = (
        args.label
        or (registry or {}).get("label")
        or checkpoint.get("training_data_label")
        or os.path.basename(source)
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(deployment, args.output)

    # Verify the exported file loads with the SAFE loader and that the
    # reloaded weights are identical.
    reloaded = torch.load(args.output, map_location="cpu", weights_only=True)

    for name, tensor in checkpoint["model_state_dict"].items():
        if not torch.equal(tensor.cpu(), reloaded["model_state_dict"][name]):
            raise RuntimeError(f"Export verification failed for {name}")

    provenance = {
        "exported_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "source_experiment": args.model or "custom",
        "label": label,
        **training_fields,
        "source_checkpoint": os.path.basename(source),
        "source_checkpoint_sha256": source_sha,
        "exported_file": os.path.basename(args.output),
        "exported_file_sha256": hc.file_sha256(args.output),
        "exported_file_bytes": os.path.getsize(args.output),
        "parameters": hc.count_parameters(model),
        "architecture": config["architecture"],
        "input_height": config["image_height"],
        "input_width": config["image_width"],
        "ctc_timesteps": config["ctc_timesteps"],
        "decoding": "CTC greedy",
        "num_classes": vocabulary.num_classes,
        "training_epoch_of_best_checkpoint": checkpoint.get("epoch"),
        "best_validation_cer": checkpoint.get("validation_cer"),
        "torch_version": torch.__version__,
        "test_metrics": None,
        "test_metrics_source": None,
    }

    analysis_path = os.path.join(
        args.analysis_dir, f"{args.model}_analysis.json"
    )

    if args.iam_summary:
        provenance.update(iam_test_metrics(args.iam_summary, source_sha))

    elif args.model and os.path.exists(analysis_path):
        with open(analysis_path, encoding="utf-8") as file:
            analysis = json.load(file)

        if analysis["meta"]["checkpoint_sha256"] == source_sha:
            provenance["test_metrics"] = {
                key: analysis["overall"][key]
                for key in (
                    "samples",
                    "corpus_cer",
                    "corpus_wer",
                    "exact_match",
                    "macro_cer",
                    "collapse_count",
                )
            }
            provenance["test_metrics_source"] = analysis_path
        else:
            print(
                "WARNING: analysis file was produced from a different "
                "checkpoint; test metrics NOT copied."
            )

    json_path = os.path.splitext(args.output)[0] + ".json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(provenance, file, indent=2)

    print(f"Exported:   {args.output}")
    print(f"Provenance: {json_path}")
    print(
        f"Size: {provenance['exported_file_bytes'] / 1e6:.1f} MB, "
        f"{provenance['parameters']:,} parameters"
    )
    print(f"Source SHA256: {source_sha}")


if __name__ == "__main__":
    main()
