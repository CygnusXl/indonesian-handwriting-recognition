"""Exporting an IAM-trained checkpoint: provenance, metrics, app wording."""

import json
import sys

import torch

import app_logic as logic
import export_final_model as export
import htr_common as hc
import iam_dataset as iam
import ui_components as ui


def make_checkpoint(path):
    vocabulary = iam.build_vocabulary(["abc def"])
    model = hc.build_model("cnn_bilstm", vocabulary.num_classes)
    torch.save(
        {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "validation_cer": 0.2,
            "vocabulary_char_to_id": vocabulary.char_to_id,
            "vocabulary_id_to_char": vocabulary.id_to_char,
            "num_classes": vocabulary.num_classes,
            "architecture": "cnn_bilstm",
            "image_height": 128,
            "image_width": 256,
            "ctc_timesteps": 64,
            "training_data": "iam",
            "training_data_label": "IAM handwriting database (English)",
            "augmentation": True,
            "train_lines": 10,
        },
        path,
    )


def run_export(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["export_final_model.py", *argv])
    export.main()


def write_summary(path, sha, cer=0.25):
    pad = {
        "samples": 7, "corpus_cer": cer, "corpus_wer": 0.5,
        "exact_match": 0.1,
    }
    path.write_text(json.dumps({
        "checkpoints": {
            "iam.pt": {"sha256": sha, "pads": {"0.0": pad, "0.25": pad}}
        }
    }))


def test_export_custom_checkpoint_records_training_data(tmp_path, monkeypatch):
    source = tmp_path / "iam.pt"
    make_checkpoint(source)
    sha = hc.file_sha256(str(source))
    summary = tmp_path / "summary.json"
    write_summary(summary, sha)
    out = tmp_path / "final.pt"

    run_export(
        monkeypatch, "--checkpoint", str(source), "--output", str(out),
        "--label", "IAM English", "--iam-summary", str(summary),
    )

    provenance = json.loads((tmp_path / "final.json").read_text())

    assert provenance["training_data"] == "iam"
    assert provenance["label"] == "IAM English"
    assert provenance["source_experiment"] == "custom"
    assert provenance["input_width"] == 256
    assert provenance["test_metrics"]["corpus_cer"] == 0.25
    assert provenance["test_metrics_label"].startswith("IAM test set")
    assert "test_metrics_page_margin" in provenance

    loaded = logic.load_app_model(str(out))
    assert loaded.provenance["training_data"] == "iam"
    assert loaded.config["image_width"] == 256


def test_metrics_for_a_different_checkpoint_are_not_copied(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "iam.pt"
    make_checkpoint(source)
    summary = tmp_path / "summary.json"
    write_summary(summary, "f" * 64)
    out = tmp_path / "final.pt"

    run_export(
        monkeypatch, "--checkpoint", str(source), "--output", str(out),
        "--iam-summary", str(summary),
    )

    provenance = json.loads((tmp_path / "final.json").read_text())

    assert provenance["test_metrics"] is None
    assert "NOT copied" in capsys.readouterr().out
    assert provenance["label"] == "IAM handwriting database (English)"


def test_app_wording_follows_training_data():
    default = ui.training_info({})
    iam_info = ui.training_info({"training_data": "iam"})

    assert "synthetic" in default["notice_title"].lower()
    assert "IAM" in iam_info["notice_title"]
    assert "synthetic" not in iam_info["about_bullet"].lower()

    assert "Trained on synthetic handwriting only" in ui.hero("m")
    assert "IAM" in ui.hero("m", provenance={"training_data": "iam"})
    assert "synthetic handwriting only" not in ui.hero(
        "m", provenance={"training_data": "iam"}
    )


def test_model_card_uses_given_metrics_label():
    card = ui.model_card(
        "IAM", 1, "128 x 1536 px",
        {"samples": 5, "corpus_cer": 0.1, "corpus_wer": 0.2,
         "exact_match": 0.3},
        "IAM test set, raw lines",
    )

    assert "IAM test set, raw lines (5 samples)" in card
    assert "Synthetic test set" not in card
