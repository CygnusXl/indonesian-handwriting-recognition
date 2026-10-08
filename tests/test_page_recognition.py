"""
Plumbing tests of the photograph pipeline with a RANDOMLY INITIALISED model.
They check structure, warnings, fallbacks and ordering; never accuracy.
"""

import numpy as np
import pytest
from PIL import Image

import document_pipeline as dp
import page_recognition as pr
import synthetic_pages as sp


@pytest.fixture(scope="module")
def ruled_page():
    return sp.make_condition_page("ruled", 3, n_lines=4)[0]


@pytest.fixture(scope="module")
def single_crop():
    page, truth = sp.render_page(31, n_lines=1, ruled=False)
    x0, y0, x1, y1 = truth["boxes"][0]

    return page.crop((x0 - 25, y0 - 25, x1 + 25, y1 + 25))


@pytest.fixture(scope="module")
def blank_page():
    return sp.render_page(41, n_lines=1, ruled=False)[0].crop((0, 1100, 1400, 1900))


def test_experiment_table_is_a_to_d():
    assert list(pr.EXPERIMENTS) == ["A", "B", "C", "D"]
    assert pr.DEFAULT_PIPELINE == "D"


def test_pipeline_a_is_the_original_path(random_loaded, ruled_page):
    import htr_common as hc

    result = pr.recognize_page(random_loaded, ruled_page, pr.EXPERIMENTS["A"])

    # identical to calling the original recognizer directly
    prepared = hc.prepare_user_image(ruled_page)
    canvas = hc.preprocess_image(prepared, 128, 1024)
    expected = hc.greedy_decode_with_confidence(
        random_loaded.model(canvas.unsqueeze(0)), random_loaded.vocabulary)[0]

    assert result.kind == "legacy"
    assert result.text == expected[0]
    assert result.n_lines == 1


def test_pipeline_b_sends_whole_page_as_one_input(random_loaded, ruled_page):
    result = pr.recognize_page(random_loaded, ruled_page, pr.EXPERIMENTS["B"])

    assert result.n_lines == 1 and result.n_chunks == 1
    assert result.kind == "whole_image"
    assert any("scaled" in w for w in result.warnings)


@pytest.mark.parametrize("key", ["C", "D"])
def test_segmenting_pipelines_recognise_each_line_separately(random_loaded, ruled_page, key):
    result = pr.recognize_page(random_loaded, ruled_page, pr.EXPERIMENTS[key])

    assert result.kind == "multi_line"
    assert result.n_lines == 4
    assert result.n_chunks >= 4
    assert [l.index for l in result.lines] == [0, 1, 2, 3]
    assert result.text == "\n".join(l.text for l in result.lines)
    assert [l.bbox[1] for l in result.lines] == sorted(l.bbox[1] for l in result.lines)

    for line in result.lines:
        for chunk in line.chunks:                      # exactly the model's input size
            assert chunk.canvas.size == (1024, 128)

    assert {"original", "normalized", "processed", "overlay"} <= set(result.images)
    assert result.timings["total_s"] > 0
    assert result.confidence is None or 0.0 <= result.confidence <= 1.0


def test_single_line_input_is_not_split_into_lines(random_loaded, single_crop):
    result = pr.recognize_page(random_loaded, single_crop, pr.EXPERIMENTS["D"])

    assert result.kind == "single_line"
    assert result.n_lines == 1


def test_blank_page_gives_clear_message_and_no_text(random_loaded, blank_page):
    result = pr.recognize_page(random_loaded, blank_page, pr.EXPERIMENTS["D"])

    assert result.kind in ("no_handwriting", "not_handwriting")
    assert result.text == "" and result.lines == []
    assert any("Recognition was skipped" in w for w in result.warnings)


def test_force_still_attempts_recognition_on_blank_page(random_loaded, blank_page):
    result = pr.recognize_page(random_loaded, blank_page, pr.EXPERIMENTS["D"], force=True)

    assert result.n_lines == 1                         # attempted on the whole image
    assert result.fallback
    assert any("anyway" in w for w in result.warnings)


def test_detection_failure_falls_back_to_single_line_pipeline(random_loaded, ruled_page, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(dp, "detect_lines", broken)

    result = pr.recognize_page(random_loaded, ruled_page, pr.EXPERIMENTS["D"])

    assert result.fallback
    assert result.n_lines == 1
    assert any("Line detection failed" in w and "falling back" in w for w in result.warnings)


def test_low_quality_image_warns_but_still_attempts(random_loaded):
    rng = np.random.default_rng(1)
    base = sp.render_page(5, n_lines=2, ruled=False)[0].resize((420, 300))
    faint = (np.array(base).astype(np.float32) * 0.1 + 215).astype(np.uint8)
    faint = np.clip(faint + rng.normal(0, 2, faint.shape), 0, 255).astype(np.uint8)

    result = pr.recognize_page(random_loaded, Image.fromarray(faint), pr.EXPERIMENTS["D"], force=True)

    assert result.warnings                              # explained, not silent
    assert result.n_lines >= 1                          # and still attempted


def test_manual_rotation_override(random_loaded):
    image = sp.make_condition_page("rotated_90", 6, n_lines=3)[0]
    result = pr.recognize_page(random_loaded, image, pr.EXPERIMENTS["D"], rotation=270)

    assert result.info["rotation_applied"] == 270
    assert result.n_lines == 3


def test_upright_page_keeps_rotation_zero(random_loaded, ruled_page):
    result = pr.recognize_page(random_loaded, ruled_page, pr.EXPERIMENTS["D"])

    assert result.info["rotation_applied"] == 0


def test_recognize_line_images_runs_original_pipeline(random_loaded, single_crop):
    outputs, seconds = pr.recognize_line_images(random_loaded, [single_crop, single_crop])

    assert len(outputs) == 2 and outputs[0] == outputs[1] and seconds > 0


def test_text_oneline_scores_line_breaks_as_spaces():
    result = pr.PageResult(spec=pr.EXPERIMENTS["D"], text="satu\ndua\\ntiga")

    assert result.text_oneline == "satu dua tiga"


def test_grayscale_rgba_and_palette_images_are_accepted(random_loaded):
    base = sp.render_page(9, n_lines=2, ruled=False)[0].resize((700, 500))

    for variant in (base.convert("L"), base.convert("RGBA"), base.convert("P")):
        result = pr.recognize_page(random_loaded, variant, pr.EXPERIMENTS["D"], force=True)

        assert result.timings["total_s"] > 0
