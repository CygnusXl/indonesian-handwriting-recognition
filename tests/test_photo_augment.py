"""Photo-style augmentation: shapes, determinism, input untouched."""

import random

import numpy as np
from PIL import Image, ImageDraw

import photo_augment as pa


def line_image(text="hello world", height=64):
    image = Image.new("L", (400, height), 255)
    ImageDraw.Draw(image).text((20, 20), text, fill=0)
    return image


def test_augment_returns_grayscale_and_leaves_input_unchanged():
    image = line_image()
    before = np.asarray(image).copy()

    for seed in range(30):
        out = pa.augment_line(image, random.Random(seed))
        assert out.mode == "L"
        assert out.size[0] > 8 and out.size[1] > 8

    assert np.array_equal(before, np.asarray(image))


def test_augment_is_repeatable_with_same_seed():
    image = line_image()
    a = pa.augment_line(image, random.Random(5))
    b = pa.augment_line(image, random.Random(5))

    assert a.size == b.size
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_augment_varies_between_seeds():
    image = line_image()
    sizes = {pa.augment_line(image, random.Random(s)).size for s in range(20)}

    assert len(sizes) > 5


def test_margins_always_grow_image_by_fraction_of_height():
    image = line_image(height=100)
    out = pa.add_margins(image, random.Random(1), low=0.1, high=0.3)

    assert out.size[0] >= image.size[0] + 20
    assert out.size[1] >= image.size[1] + 20
    # Original ink is still there.
    assert np.asarray(out).min() == np.asarray(image).min()


def test_all_effects_off_leaves_image_identical():
    off = {k: 0.0 for k in pa.DEFAULT_PROBABILITIES}
    image = line_image()
    out = pa.augment_line(image, random.Random(0), off)

    assert out.size == image.size
    assert np.array_equal(np.asarray(out), np.asarray(image))


def test_pad_like_page_pipeline_is_deterministic_border():
    image = line_image(height=80)
    out = pa.pad_like_page_pipeline(image, 0.25)

    assert out.size == (image.size[0] + 40, 80 + 40)
    assert np.asarray(out)[:20].min() == 255  # top border is white
    assert np.array_equal(
        np.asarray(out)[20:100, 20:-20], np.asarray(image)
    )


def test_each_effect_runs_and_keeps_dimensions_sane():
    image = line_image()
    rng = random.Random(3)

    for effect in (
        pa.change_ink, pa.change_paper, pa.add_shading, pa.add_ruling,
        pa.add_blur, pa.add_noise, pa.add_jpeg,
    ):
        out = effect(image, rng)
        assert out.size == image.size
        assert out.mode == "L"
