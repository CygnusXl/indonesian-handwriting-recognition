"""
Synthetic "photographed notebook page" generator.

PURPOSE: test and benchmark the DOCUMENT stages (orientation, perspective,
illumination, ruling suppression, line segmentation) where the ground truth
(number and position of lines) is known by construction.

WHAT IT IS NOT: real handwriting. Text is drawn with a font plus random
baseline wobble, so results on these pages say NOTHING about recognition
accuracy on real handwriting. They only test the geometry/segmentation code.

    python src/evaluate_segmentation.py         # uses this module
"""

import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SENTENCES = [
    "Meeting notes for the data class",
    "Remember to submit the report today",
    "Kita belajar jaringan saraf tiruan",
    "Tomorrow we visit the old library",
    "Harga buku itu sangat murah sekali",
    "Check the results before sending it",
    "Ibu pergi ke pasar setiap pagi hari",
    "Three apples and two green bananas",
    "Please call me after the lecture ends",
    "Hujan turun deras sejak tadi malam",
    "Draw a diagram of the whole system",
    "Saya suka membaca buku di kamar",
]

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

CONDITIONS = {
    "clean":       dict(ruled=False),
    "ruled":       dict(ruled=True),
    "shadow":      dict(ruled=True, shadow=True, gradient=True),
    "noisy_blur":  dict(ruled=True, noise=9.0, blur=1.2),
    "light_ink":   dict(ruled=True, ink=(110, 110, 150), gradient=True),
    "skewed":      dict(ruled=True, rotate=6.0),
    "perspective": dict(ruled=True, perspective=True, gradient=True),
    "rotated_90":  dict(ruled=True, quarter=1),
    "strikeout":   dict(ruled=True, strike=True),
}


def _font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default(size=size)


def _line_text(rng):
    words = rng.choice(SENTENCES).split()
    text = ""

    for word in words:
        if len(text) + len(word) + 1 > 36:
            break
        text = (text + " " + word).strip()

    return text


def _draw_line_layer(text, size, rng, ink):
    """Transparent layer holding one wobbly handwritten-looking line."""

    font = _font(size)
    width = int(font.getlength(text) + 80)
    height = int(size * 2.4)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    x = 20.0
    base = int(size * 0.7)
    drift = rng.uniform(-3, 3)

    for char in text:
        y = base + rng.uniform(-3, 3) + drift * (x / width)
        draw.text((x, y), char, font=font, fill=ink + (255,))
        x += font.getlength(char) * rng.uniform(0.95, 1.08)

    return layer.rotate(rng.uniform(-1.2, 1.2), expand=True, resample=Image.BICUBIC)


def render_page(
    seed,
    n_lines=None,
    width=1400,
    height=1900,
    ruled=True,
    shadow=False,
    gradient=False,
    noise=0.0,
    blur=0.0,
    ink=(25, 25, 70),
    rotate=0.0,
    perspective=False,
    quarter=0,
    strike=False,
    pitch=None,
    size_ratio=0.36,
):
    """
    Returns (PIL RGB image, truth) where truth =
        {"n_lines": int, "boxes": [(x0,y0,x1,y1)...] or None, "texts": [...]}
    `boxes` are in final image coordinates only when no geometric transform
    is applied (rotate == 0, no perspective, quarter == 0); otherwise None.
    """

    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    n_lines = n_lines or rng.randint(2, 7)
    pitch = pitch or rng.randint(125, 150)
    size = int(pitch * size_ratio)
    top = 170

    page = Image.new("RGB", (width, height), (247, 245, 238))
    draw = ImageDraw.Draw(page)

    if ruled:
        y = top
        while y < height - 60:
            draw.line([(0, y), (width, y)], fill=(150, 182, 222), width=2)
            y += pitch
        draw.line([(135, 0), (135, height)], fill=(205, 100, 100), width=2)

    boxes, texts = [], []

    for index in range(n_lines):
        text = _line_text(rng)
        layer = _draw_line_layer(text, size, rng, ink)
        x = 170 + rng.randint(0, 25)
        y = top + index * pitch - int(size * 0.95)
        page.paste(layer, (x, y), layer)

        alpha = np.array(layer.split()[-1])
        ys, xs = np.nonzero(alpha > 40)
        boxes.append((x + int(xs.min()), y + int(ys.min()),
                      x + int(xs.max()) + 1, y + int(ys.max()) + 1))
        texts.append(text)

        if strike:
            sy = (boxes[-1][1] + boxes[-1][3]) // 2
            if index % 2 == 0:
                draw.line([(boxes[-1][0] + 40, sy), (boxes[-1][0] + 260, sy)],
                          fill=ink, width=3)

    array = np.array(page).astype(np.float32)

    if gradient:
        xs = np.linspace(0.78, 1.05, width)[None, :]
        ys = np.linspace(1.0, 0.85, height)[:, None]
        array *= (xs * ys)[..., None]

    if shadow:
        yy, xx = np.mgrid[0:height, 0:width]
        band = np.exp(-(((xx - 0.2 * width) + 0.5 * (yy - 0.5 * height)) / 160.0) ** 2)
        array *= (1 - 0.45 * band)[..., None]

    if noise:
        array += nprng.normal(0, noise, array.shape)

    page = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

    if blur:
        page = page.filter(ImageFilter.GaussianBlur(blur))

    geometric = bool(rotate or perspective or quarter)

    if rotate:
        page = page.rotate(rotate, expand=True, resample=Image.BICUBIC,
                           fillcolor=(120, 112, 100))

    if perspective:
        w, h = page.size
        desk = np.full((h + 360, w + 440, 3), (62, 56, 50), dtype=np.uint8)
        dst = np.float32([
            [150 + rng.randint(0, 40), 90 + rng.randint(0, 40)],
            [w + 260 - rng.randint(0, 40), 40 + rng.randint(0, 30)],
            [w + 300 - rng.randint(0, 30), h + 250 - rng.randint(0, 30)],
            [110 + rng.randint(0, 30), h + 290 - rng.randint(0, 30)],
        ])
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(np.array(page), matrix,
                                     (desk.shape[1], desk.shape[0]))
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix,
                                   (desk.shape[1], desk.shape[0]))
        desk[mask > 0] = warped[mask > 0]
        page = Image.fromarray(desk)

    if quarter:
        page = Image.fromarray(np.ascontiguousarray(np.rot90(np.array(page), k=quarter)))

    truth = {
        "n_lines": n_lines,
        "boxes": None if geometric else boxes,
        "texts": texts,
    }

    return page, truth


def make_condition_page(condition, seed, **overrides):
    parameters = dict(CONDITIONS[condition])
    parameters.update(overrides)

    return render_page(seed, **parameters)
