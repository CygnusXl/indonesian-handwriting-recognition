"""
Photo-style augmentation for handwriting line images.

Why this exists
---------------
The earlier model was trained on clean synthetic line images. Two
failure modes were MEASURED on this project (see DOCUMENT_PIPELINE.md):

    * very different crop margins change the result (the training images
      have a left margin of about 0.15 x image height, the page pipeline
      crops with 0.25), and
    * the pipeline's own cleanup costs a little accuracy.

Augmentation exposes the model to the variation a phone photo adds
(margins, slight rotation and slant, uneven light, grey paper, noise,
blur, JPEG artefacts, faint ruling, thicker or thinner pen strokes) so
that the trained weights are not tied to one pristine rendering.

IMPORTANT: this is a *hypothesis* about what helps. Nothing here has
been shown to improve accuracy on real photos until it is measured
(src/evaluate_iam.py reports the held-out set; real photos need their
own ground truth). Only Pillow and NumPy are used, so it runs on CPU
and without OpenCV.

Everything takes a `random.Random` instance so tests and training runs
are repeatable.
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter


# Probability that each independent effect is applied to one image.
DEFAULT_PROBABILITIES = {
    "margins": 1.0,       # always vary the white border (see above)
    "geometry": 0.7,      # small rotation, slant, width change
    "ink": 0.4,           # thicker / thinner strokes
    "paper": 0.7,         # grey or tinted paper, lower contrast
    "shading": 0.5,       # uneven illumination across the line
    "ruling": 0.2,        # faint notebook lines
    "blur": 0.35,
    "noise": 0.4,
    "jpeg": 0.3,
}


def _chance(rng, name, probabilities):
    return rng.random() < probabilities.get(name, 0.0)


def add_margins(image, rng, low=0.03, high=0.30):
    """
    Pad with white on every side by a random fraction of the image
    height. Left/right margins vary independently because the page
    pipeline crops with a margin that is not the one used in training.
    """

    width, height = image.size
    fractions = [rng.uniform(low, high) for _ in range(4)]
    left, right, top, bottom = (int(round(f * height)) for f in fractions)

    canvas = Image.new(
        "L", (width + left + right, height + top + bottom), 255
    )
    canvas.paste(image, (left, top))

    return canvas


def add_geometry(image, rng):
    """Small rotation, forward/backward slant and horizontal stretch."""

    width, height = image.size

    # Horizontal stretch / squeeze.
    stretch = rng.uniform(0.85, 1.15)
    image = image.resize(
        (max(8, int(width * stretch)), height), Image.Resampling.BILINEAR
    )
    width = image.size[0]

    # Slant (shear along x).
    shear = rng.uniform(-0.25, 0.25)
    pad = int(abs(shear) * height) + 1
    wide = Image.new("L", (width + 2 * pad, height), 255)
    wide.paste(image, (pad, 0))
    wide = wide.transform(
        wide.size,
        Image.Transform.AFFINE,
        (1, shear, -shear * height / 2, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=255,
    )

    # Rotation.
    angle = rng.uniform(-2.5, 2.5)
    return wide.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=True,
        fillcolor=255,
    )


def change_ink(image, rng):
    """Make strokes slightly thicker or thinner."""

    if rng.random() < 0.5:
        return image.filter(ImageFilter.MinFilter(3))  # dark grows
    return image.filter(ImageFilter.MaxFilter(3))      # dark shrinks


def change_paper(image, rng):
    """
    Map white to a grey paper tone and black to a lifted ink tone
    (lower contrast), the way a dim or washed-out photo looks.
    """

    paper = rng.uniform(150, 255)
    ink = rng.uniform(0, 90)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = ink + array * (paper - ink)

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def add_shading(image, rng):
    """Brightness gradient across the line (uneven light / shadow)."""

    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape

    angle = rng.uniform(0, 2 * np.pi)
    strength = rng.uniform(0.1, 0.45)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = np.cos(angle) * (xs / max(width, 1) - 0.5) + np.sin(angle) * (
        ys / max(height, 1) - 0.5
    )
    ramp = ramp / (np.abs(ramp).max() + 1e-6)

    array = array * (1.0 - strength * (ramp * 0.5 + 0.5))

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def add_ruling(image, rng):
    """Faint horizontal notebook lines through the line image."""

    array = np.asarray(image, dtype=np.float32).copy()
    height, width = array.shape

    spacing = rng.uniform(0.35, 0.9) * height
    offset = rng.uniform(0, spacing)
    darkness = rng.uniform(0.05, 0.25)
    thickness = rng.choice([1, 1, 2])

    y = offset
    while y < height:
        row = int(y)
        array[row:row + thickness, :] *= 1.0 - darkness
        y += spacing

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def add_blur(image, rng):
    return image.filter(
        ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.4))
    )


def add_noise(image, rng):
    array = np.asarray(image, dtype=np.float32)
    sigma = rng.uniform(2.0, 12.0)
    generator = np.random.default_rng(rng.getrandbits(32))
    array = array + generator.normal(0.0, sigma, array.shape)

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def add_jpeg(image, rng):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(25, 80))
    buffer.seek(0)

    return Image.open(buffer).convert("L")


def augment_line(image, rng=None, probabilities=None):
    """
    Apply a random combination of photo-like effects to a grayscale
    PIL line image and return a new grayscale image. The input is not
    modified.

    The order mimics how a photo is formed: geometry of the writing
    first, then paper / lighting, then the camera (blur, noise, JPEG).
    """

    rng = rng or random
    probabilities = {**DEFAULT_PROBABILITIES, **(probabilities or {})}

    image = image.convert("L")

    if _chance(rng, "geometry", probabilities):
        image = add_geometry(image, rng)

    if _chance(rng, "ink", probabilities):
        image = change_ink(image, rng)

    if _chance(rng, "margins", probabilities):
        image = add_margins(image, rng)

    if _chance(rng, "paper", probabilities):
        image = change_paper(image, rng)

    if _chance(rng, "shading", probabilities):
        image = add_shading(image, rng)

    if _chance(rng, "ruling", probabilities):
        image = add_ruling(image, rng)

    if _chance(rng, "blur", probabilities):
        image = add_blur(image, rng)

    if _chance(rng, "noise", probabilities):
        image = add_noise(image, rng)

    if _chance(rng, "jpeg", probabilities):
        image = add_jpeg(image, rng)

    return image


def pad_like_page_pipeline(image, fraction=0.25):
    """
    Deterministic white border of `fraction` x height on every side.
    This is what the page pipeline's crop margin does (default 0.25),
    so evaluating with it measures the model under the conditions the
    app actually creates. It is NOT augmentation (nothing random).
    """

    image = image.convert("L")
    width, height = image.size
    margin = int(round(fraction * height))

    canvas = Image.new(
        "L", (width + 2 * margin, height + 2 * margin), 255
    )
    canvas.paste(image, (margin, margin))

    return canvas
