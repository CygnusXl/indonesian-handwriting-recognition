"""
Plumbing tests for the application logic and UI.

A RANDOMLY INITIALISED model is used: these tests verify that the
application code runs end to end (image in, text out, errors handled),
not that the model is accurate.
"""

import csv
import io
import os

import pytest
import torch
from PIL import Image

import htr_common as hc
import app_logic as logic
import evaluation_log as elog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def deployment_model(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    path = folder / "final_model.pt"

    model = hc.build_model("cnn_bilstm", 70)
    pool = list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\\"
    )[:69]
    char_to_id = {c: i for i, c in enumerate(pool)}

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary_char_to_id": char_to_id,
            "vocabulary_id_to_char": {i: c for c, i in char_to_id.items()},
            "num_classes": 70,
            "architecture": "cnn_bilstm",
            "image_height": 128,
            "image_width": 1024,
            "ctc_timesteps": 256,
        },
        path,
    )

    return str(path)


def test_load_missing_model_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        logic.load_app_model(str(tmp_path / "missing.pt"))

    assert "export_final_model.py" in str(error.value)


def test_end_to_end_recognition(deployment_model):
    loaded = logic.load_app_model(deployment_model)

    image = Image.new("RGB", (1200, 200), "white")
    result = logic.recognize_image(loaded, image)

    assert isinstance(result.text, str)
    assert result.canvas.size == (1024, 128)
    assert result.seconds >= 0


def test_corrupt_upload_is_reported_clearly():
    with pytest.raises(ValueError) as error:
        logic.open_user_image(io.BytesIO(b"this is not an image"))

    assert "could not be read" in str(error.value)


def test_warnings_for_tall_dark_and_flat_images():
    tall = Image.new("L", (400, 400), 255)
    assert any("several lines" in w for w in logic.image_warnings(tall))

    dark = Image.new("L", (1000, 100), 10)
    messages = " ".join(logic.image_warnings(dark))
    assert "dark" in messages and "low contrast" in messages

    ok = Image.new("L", (1000, 100), 255)
    ok.paste(0, (100, 40, 900, 60))
    assert logic.image_warnings(ok) == []


def test_evaluation_log_roundtrip(tmp_path):
    path = tmp_path / "sessions.csv"

    row = elog.build_row(
        "P01", "abc", "single-line", "neat", "scan", False,
        "saya suka kopi", "saya kopi", 4, 0.12, "f" * 64,
    )

    assert row["char_errors"] == 5
    assert row["wer"] == pytest.approx(1 / 3)

    elog.append_session(row, str(path))
    elog.append_session(row, str(path))

    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert list(rows[0].keys()) == elog.COLUMNS


def test_streamlit_ui_shows_error_when_model_missing(monkeypatch, tmp_path):
    testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("HTR_MODEL_PATH", str(tmp_path / "nope.pt"))

    app = testing.AppTest.from_file(
        os.path.join(ROOT, "app", "app.py"), default_timeout=60
    ).run()

    assert not app.exception
    assert any("model file is missing" in e.value for e in app.error)


def test_streamlit_ui_renders_with_model(monkeypatch, deployment_model):
    testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("HTR_MODEL_PATH", deployment_model)

    app = testing.AppTest.from_file(
        os.path.join(ROOT, "app", "app.py"), default_timeout=60
    ).run()

    assert not app.exception
    assert not app.error
    assert any("Upload an image" in m.value for m in app.markdown)


def test_streamlit_hosted_mode_renders(monkeypatch, deployment_model):
    testing = pytest.importorskip("streamlit.testing.v1")

    monkeypatch.setenv("HTR_MODEL_PATH", deployment_model)
    monkeypatch.setenv("HTR_HOSTED", "1")

    app = testing.AppTest.from_file(
        os.path.join(ROOT, "app", "app.py"), default_timeout=60
    ).run()

    assert not app.exception
    assert not app.error
