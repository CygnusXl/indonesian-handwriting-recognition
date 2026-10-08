"""
Tests of the document stages (orientation, preprocessing, ruling suppression,
line detection, chunking) on SYNTHETIC pages with known layout.

These verify geometry code only. They say nothing about real handwriting.
"""

import numpy as np
import pytest
from PIL import Image

import document_pipeline as dp
import page_recognition as pr
import synthetic_pages as sp

D = pr.EXPERIMENTS["D"]
C = pr.EXPERIMENTS["C"]


def detect(image, spec=D):
    stages = dp.preprocess_document(image, spec.preprocess)

    return stages, dp.detect_lines(stages.mask, spec.segmentation)


def test_multiline_ruled_page_finds_every_line_in_order():
    image, truth = sp.make_condition_page("ruled", 3, n_lines=5, )
    stages, detection = detect(image)

    assert detection.kind == "multi_line"
    assert len(detection.lines) == truth["n_lines"] == 5

    tops = [line.y0 for line in detection.lines]
    assert tops == sorted(tops)
    assert [line.index for line in detection.lines] == list(range(5))


@pytest.mark.parametrize("condition", ["shadow", "noisy_blur", "light_ink", "skewed", "perspective"])
def test_degraded_pages_still_give_correct_line_count(condition):
    image, truth = sp.make_condition_page(condition, 21, n_lines=4)
    _, detection = detect(image)

    assert len(detection.lines) == 4


@pytest.mark.parametrize("ruled", [False, True])
def test_single_line_crop_is_not_split(ruled):
    page, truth = sp.render_page(31, n_lines=1, ruled=ruled)
    x0, y0, x1, y1 = truth["boxes"][0]
    crop = page.crop((x0 - 25, y0 - 25, x1 + 25, y1 + 25))
    _, detection = detect(crop)

    assert detection.kind == "single_line"
    assert len(detection.lines) == 1


def test_blank_paper_and_noise_are_not_read_as_handwriting():
    page = sp.render_page(41, n_lines=1, ruled=False)[0].crop((0, 1100, 1400, 1900))
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (800, 600, 3), dtype=np.uint8))

    for image in (page, noise):
        _, detection = detect(image)

        assert detection.kind in ("no_handwriting", "not_handwriting")
        assert detection.lines == []
        assert detection.reason


def test_upright_page_is_not_rotated():
    image, _ = sp.make_condition_page("ruled", 5, n_lines=4)
    stages = dp.preprocess_document(image, D.preprocess)

    assert stages.info["rotation_applied"] == 0
    assert stages.info["rotation_candidates"] == [0]


def test_sideways_page_gets_a_quarter_turn_candidate():
    image, _ = sp.make_condition_page("rotated_90", 6, n_lines=4)
    stages = dp.preprocess_document(image, D.preprocess)

    assert 270 in stages.info["rotation_candidates"]

    upright = dp.preprocess_document(image, D.preprocess, rotation=270)
    assert len(dp.detect_lines(upright.mask, D.segmentation).lines) == 4


def test_explicit_rotation_is_applied_exactly():
    from dataclasses import replace

    config = replace(D.preprocess, perspective=False, deskew=False)
    image, _ = sp.make_condition_page("ruled", 7, n_lines=3)
    plain = dp.preprocess_document(image, config, rotation=0)
    turned = dp.preprocess_document(image, config, rotation=90)

    assert turned.info["rotation_applied"] == 90
    assert turned.geometry.shape == plain.geometry.shape[::-1]
    assert np.array_equal(turned.geometry, np.rot90(plain.geometry, 1))


def test_stages_are_exposed_and_consistent():
    image, _ = sp.make_condition_page("shadow", 8, n_lines=3)
    stages = dp.preprocess_document(image, D.preprocess)

    for name in ("original", "geometry", "normalized", "processed"):
        array = getattr(stages, name)
        assert array.dtype == np.uint8 and array.ndim == 2

    assert stages.processed.shape == stages.normalized.shape
    assert stages.mask.shape == stages.processed.shape
    assert not np.array_equal(stages.normalized, stages.processed)  # ruling removed in D


def test_configs_differ_only_where_intended():
    assert not pr.EXPERIMENTS["B"].preprocess.suppress_ruling
    assert not pr.EXPERIMENTS["C"].preprocess.suppress_ruling
    assert pr.EXPERIMENTS["D"].preprocess.suppress_ruling
    assert pr.EXPERIMENTS["A"].legacy and not pr.EXPERIMENTS["B"].segment
    assert pr.EXPERIMENTS["C"].segment and pr.EXPERIMENTS["D"].segment


def test_ruling_suppression_removes_rules_but_keeps_tall_strokes_and_dots():
    # white page, a faint blue rule across it, a bold 'i' (stem + dot) and a tall 'l'
    gray = np.full((140, 600), 245, np.uint8)
    gray[100:102, :] = 190                    # ruling
    gray[40:100, 200:206] = 20                # stem of i
    gray[30:36, 200:206] = 20                 # its dot (4 px gap above the stem)
    gray[30:108, 300:306] = 20                # tall l (crosses the rule)

    cleaned, removed = dp.suppress_ruling(gray)

    assert removed > 0
    assert cleaned[101, 50] > 235                       # rule gone where nothing crosses it
    assert cleaned[60, 203] < 60 and cleaned[60, 303] < 60   # strokes kept
    assert cleaned[101, 303] < 80                       # stroke kept where it crosses the rule
    assert cleaned[33, 203] < 60                        # the dot is still there
    assert cleaned[38, 203] > 200                       # and still separate from the stem


def test_suppression_does_not_erase_vertical_strokes_in_a_wide_crop():
    gray = np.full((90, 900), 245, np.uint8)
    for x in range(60, 800, 40):
        gray[15:80, x:x + 4] = 25                       # 65 px tall strokes in a 90 px crop

    cleaned, _ = dp.suppress_ruling(gray)

    assert (cleaned < 100).sum() >= 0.9 * (gray < 100).sum()


def test_chunks_fit_the_model_canvas_and_do_not_leak_into_neighbours():
    height, width = 60, 1500
    mask = np.zeros((height, width), bool)
    words = [(20, 180), (205, 400), (430, 600), (630, 900), (925, 1100), (1130, 1300)]

    for a, b in words:
        mask[10:50, a:b] = True

    chunks = dp.split_line_into_chunks(mask, max_aspect=8, ink_height=40, margin_x=30)

    assert len(chunks) >= 2
    assert all(b - a <= 8 * height for a, b in chunks)

    for (a, b) in chunks:
        inside = [w for w in words if a < w[1] and w[0] < b]
        for w in inside:                                 # every ink word is whole inside
            assert a <= w[0] and w[1] <= b

    for (a1, b1), (a2, b2) in zip(chunks, chunks[1:]):
        assert b1 <= a2 + 1                              # chunks do not overlap


def test_short_line_is_one_chunk():
    mask = np.zeros((60, 300), bool)
    mask[10:50, 20:280] = True

    assert dp.split_line_into_chunks(mask, 8, 40, 20) == [(0, 300)]
