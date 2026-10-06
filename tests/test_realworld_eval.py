"""
Plumbing tests for the real-world evaluation scripts, using tiny
synthetic images and a RANDOM model. They verify that the scripts run
and aggregate correctly; the numbers they produce mean nothing.
"""

import csv
import os

import pytest
import torch
from PIL import Image, ImageDraw

import htr_common as hc
import evaluate_realworld as rw
import summarize_usability as su


@pytest.fixture()
def toy_setup(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()

    rows = []

    for i, (line_type, text) in enumerate(
        [("single-line", "saya suka kopi"),
         ("single-line", "ya"),
         ("multiline", "satu\\ndua")]
    ):
        image = Image.new("RGB", (800, 120 if line_type == "single-line" else 300), "white")
        ImageDraw.Draw(image).text((20, 20), text, fill="black")
        name = f"P0{i % 2 + 1}_{i}.png"
        image.save(photos / name)
        rows.append(
            {
                "image_file": name,
                "participant_id": f"P0{i % 2 + 1}",
                "reference_text": text,
                "line_type": line_type,
                "handwriting_condition": "average",
                "capture_condition": "scan",
            }
        )

    model = hc.build_model("cnn_bilstm", 70).eval()
    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\\")[:69]
    vocab = hc.CheckpointVocabulary({c: i for i, c in enumerate(pool)})
    config = {"architecture": "cnn_bilstm", "image_height": 128,
              "image_width": 1024, "ctc_timesteps": 256}

    return model, vocab, config, rows, str(photos)


def test_evaluate_images_and_summary(toy_setup, tmp_path):
    model, vocab, config, rows, photos = toy_setup

    results = rw.evaluate_images(model, vocab, config, rows, photos, "cpu")

    assert len(results) == len(rows) * 2          # auto-contrast off + on
    assert {r["auto_contrast"] for r in results} == {False, True}

    summary = rw.summary_rows(results)
    overall = summary[0][3]
    assert overall["samples"] == len(rows)

    rw.write_summary(str(tmp_path), summary)
    assert (tmp_path / "realworld_summary.md").exists()


def test_missing_image_is_reported(toy_setup):
    model, vocab, config, rows, photos = toy_setup
    rows[0]["image_file"] = "does_not_exist.png"

    with pytest.raises(FileNotFoundError):
        rw.evaluate_images(model, vocab, config, rows, photos, "cpu")


def test_out_of_vocabulary_characters_are_reported(toy_setup):
    model, vocab, config, rows, photos = toy_setup
    rows[0]["reference_text"] = "café ü"

    results = rw.evaluate_images(model, vocab, config, rows[:1], photos, "cpu")

    assert "é" in results[0]["out_of_vocabulary_chars"]


def test_sus_score_known_values():
    best = {f"sus_{i}": (5 if i % 2 else 1) for i in range(1, 11)}
    worst = {f"sus_{i}": (1 if i % 2 else 5) for i in range(1, 11)}
    neutral = {f"sus_{i}": 3 for i in range(1, 11)}

    assert su.sus_score(best) == 100.0
    assert su.sus_score(worst) == 0.0
    assert su.sus_score(neutral) == 50.0

    with pytest.raises(ValueError):
        su.sus_score({**neutral, "sus_3": 7})
