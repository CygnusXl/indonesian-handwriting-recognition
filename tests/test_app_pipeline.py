"""
App tests for the photograph pipeline (random-weight model: plumbing only).
The Streamlit file uploader cannot be driven by AppTest, so it is patched to
return a synthetic notebook-page photograph.
"""

import io
import os

import pytest
import torch

import app_logic as logic
import htr_common as hc
import synthetic_pages as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def model_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("m") / "final_model.pt"
    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\\")[:69]
    ids = {c: i for i, c in enumerate(pool)}

    torch.save({"model_state_dict": hc.build_model("cnn_bilstm", 70).state_dict(),
                "vocabulary_char_to_id": ids,
                "vocabulary_id_to_char": {i: c for c, i in ids.items()},
                "num_classes": 70, "architecture": "cnn_bilstm",
                "image_height": 128, "image_width": 1024, "ctc_timesteps": 256}, path)

    return str(path)


@pytest.fixture(scope="module")
def photo_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("p") / "page.jpg"
    sp.make_condition_page("shadow", 7, n_lines=4)[0].convert("RGB").save(path, quality=90)

    return str(path)


SCRIPT = """
import runpy
import streamlit as st

data = open({photo!r}, "rb").read()

class Upload:
    def getvalue(self):
        return data

st.file_uploader = lambda *a, **k: Upload()
runpy.run_path({app!r}, run_name="__main__")
"""


def make_app(photo):
    testing = pytest.importorskip("streamlit.testing.v1")
    code = SCRIPT.format(photo=photo, app=os.path.join(ROOT, "app", "app.py"))

    return testing.AppTest.from_string(code, default_timeout=180)


def test_recognize_photo_default_is_d_and_returns_lines(random_loaded):
    image = sp.make_condition_page("ruled", 3, n_lines=3)[0]
    result = logic.recognize_photo(random_loaded, image)

    assert logic.DEFAULT_PIPELINE == "D"
    assert result.spec.key == "D" and result.n_lines == 3
    assert logic.page_display_text(result).count("\n") == 2


def test_recognize_photo_degrades_instead_of_crashing(random_loaded, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("unexpected")

    monkeypatch.setattr(logic.pr, "recognize_page", broken)
    image = sp.make_condition_page("ruled", 3, n_lines=3)[0]
    result = logic.recognize_photo(random_loaded, image)

    assert result.fallback and result.n_lines == 1
    assert any("failed" in w and "single-line" in w for w in result.warnings)


def test_ui_shows_the_four_stages_and_heuristic_label(monkeypatch, model_file, photo_path):
    monkeypatch.setenv("HTR_MODEL_PATH", model_file)

    app = make_app(photo_path).run()

    assert not app.exception

    app.button[0].click().run()

    assert not app.exception
    assert [h.value for h in app.subheader] == [
        "1. Input image", "2. Preprocessed image",
        "3. Detected handwriting lines", "4. Recognised text"]
    assert any("several lines" in i.value for i in app.info)
    assert any("heuristic" in m.label.lower() for m in app.metric)
    assert [m.value for m in app.metric if m.label == "Lines"] == ["4"]
    assert any("What the model sees" in e.label for e in app.expander)


def test_ui_pipeline_a_still_works(monkeypatch, model_file, photo_path):
    monkeypatch.setenv("HTR_MODEL_PATH", model_file)

    app = make_app(photo_path).run()
    app.sidebar.selectbox[0].set_value("A").run()
    app.button[0].click().run()

    assert not app.exception
    assert any("Original single-line pipeline" in i.value for i in app.info)
    assert any("not much wider than it is tall" in w.value for w in app.warning)
