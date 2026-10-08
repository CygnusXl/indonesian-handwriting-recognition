"""
Document pipeline: photograph of a notebook page -> clean page -> lines.

The trained recognizer (CNN-BiLSTM-CTC) is a SINGLE-LINE recognizer: it was
trained on tight images of one line of handwriting, scaled to 128 px high
(max 1024 px wide). Feeding it a whole photographed page squeezes the page
into that canvas and shrinks the handwriting to a few pixels, which is a
scale/layout mismatch, not something extra training epochs would fix.

This module therefore does the document work BEFORE the model:

    load_gray            EXIF orientation, transparency, grayscale
    find_page_quad       optional perspective correction (conservative)
    detect_text_axis     optional 90-degree orientation detection
    estimate_skew        small-angle deskew (projection-profile search)
    flatten_illumination shadows / uneven light (morphological background)
    stretch_contrast     percentile levels, SOFT (no hard binarisation)
    suppress_ruling      notebook ruling / margin lines (optional)
    ink_mask             binary mask used ONLY for line detection
    detect_lines         line segmentation + single/multi/no-handwriting
    crop_line / split_line_into_chunks   model-ready crops

Only OpenCV + numpy + PIL are used (no torch), so every stage is unit-testable
without a model. The image that finally goes to the model is always the
GRAY (not binarised) processed image; the binary mask never reaches the model.

Everything here is a heuristic. Thresholds are untuned on real photographs;
see README ("Known limitations") and src/evaluate_pipeline.py for how to
measure them on your own images.
"""

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PreprocessConfig:
    name: str = "improved"
    max_side: int = 2400            # larger photos are downscaled (INTER_AREA)
    perspective: bool = True        # only applied if a clear page quad is found
    auto_orient: bool = True        # detect 90-degree rotation
    try_upside_down: bool = False   # also test 180 degrees (needs the model)
    deskew: bool = True
    max_skew_deg: float = 12.0
    illumination: bool = True
    denoise: str = "bilateral"      # "none" | "bilateral" | "nlm"
    contrast: bool = True
    white_point: float = 0.88       # fraction of the ink->paper range mapped to white
    suppress_ruling: bool = False
    ruling_kernel_frac: float = 0.08

    def describe(self):
        parts = []
        if self.perspective:
            parts.append("perspective")
        if self.auto_orient:
            parts.append("orient")
        if self.deskew:
            parts.append("deskew")
        if self.illumination:
            parts.append("illumination")
        if self.denoise != "none":
            parts.append(f"denoise:{self.denoise}")
        if self.contrast:
            parts.append("contrast")
        if self.suppress_ruling:
            parts.append("ruling-suppression")
        return "+".join(parts) if parts else "none"


# The original pipeline (htr_common.prepare_user_image + preprocess_image)
# does none of this; it is represented by `LEGACY` for documentation only.
LEGACY = PreprocessConfig(
    name="original",
    perspective=False,
    auto_orient=False,
    deskew=False,
    illumination=False,
    denoise="none",
    contrast=False,
)


@dataclass(frozen=True)
class SegmentationConfig:
    smear_factor: float = 3.0           # horizontal smear = factor * typical height
    split_tall_factor: float = 1.7      # blobs taller than this x reference are split
    min_line_height_factor: float = 0.4
    margin: float = 0.25                # vertical margin around a line (x line height)
    margin_x: float = 0.5               # horizontal margin (x line height)
    isolate_lines: bool = False         # blank everything outside the line's own ink
    not_handwriting_ink_fraction: float = 0.30
    max_components: int = 5000


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def load_gray(image):
    """PIL image -> uint8 grayscale array (EXIF applied, transparency on white)."""

    image = ImageOps.exif_transpose(image)

    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)

    return np.array(image.convert("L"))


def limit_size(gray, max_side):
    height, width = gray.shape
    longest = max(height, width)

    if longest <= max_side:
        return gray

    scale = max_side / longest

    return cv2.resize(
        gray,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _smooth_1d(values, sigma):
    sigma = max(0.5, float(sigma))
    radius = int(3 * sigma) + 1
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    return np.convolve(values.astype(np.float64), kernel, mode="same")


def image_quality(gray):
    """
    Cheap input diagnostics on the ORIGINAL grayscale image. Thresholds used
    for warnings are heuristics and have not been validated.
    """

    height, width = gray.shape
    scale = 1000.0 / max(height, width)

    small = (
        cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))),
                   interpolation=cv2.INTER_AREA)
        if scale < 1.0 else gray
    )

    return {
        "width": width,
        "height": height,
        "mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        # paper level (median) minus the darkest 0.5% of pixels: robust to
        # pages that are mostly blank, where the plain std is always small
        "ink_contrast": float(np.median(gray) - np.percentile(gray, 0.5)),
        "sharpness": float(cv2.Laplacian(small, cv2.CV_64F).var()),
    }


# ============================================================
# ILLUMINATION, DENOISING, CONTRAST
# ============================================================

def estimate_background(gray):
    """
    Paper brightness field.

    Sparse ink (typical page photo): a large MEDIAN filter follows soft
    shadows well because ink covers only a small part of each window.
    Dense ink (tight line crops where strokes fill much of the window): the
    median would absorb ink into the background and erase it, so a large
    morphological CLOSING (which removes dark structures thinner than the
    kernel) is used instead.
    """

    height, width = gray.shape
    scale = min(1.0, 600.0 / max(height, width))

    small = (
        cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))),
                   interpolation=cv2.INTER_AREA)
        if scale < 1.0 else gray
    )

    # The kernel must be larger than the strokes.
    k = max(31, int(min(small.shape) / 6)) | 1
    closed = cv2.morphologyEx(
        small, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    closed = cv2.GaussianBlur(closed, (0, 0), k / 3.0)

    ink_coverage = float((small < 0.7 * closed).mean())

    if ink_coverage < 0.15 and min(small.shape) >= 2 * 31:
        km = max(31, int(0.12 * min(small.shape))) | 1
        background = cv2.GaussianBlur(cv2.medianBlur(small, km), (0, 0), km / 4.0)
    else:
        background = closed

    if scale < 1.0:
        background = cv2.resize(background, (width, height),
                                interpolation=cv2.INTER_LINEAR)

    return np.maximum(background, 1)


def flatten_illumination(gray):
    """Divide by the estimated paper brightness -> even background."""

    background = estimate_background(gray).astype(np.float32)
    flat = gray.astype(np.float32) / background * 255.0

    return np.clip(flat, 0, 255).astype(np.uint8)


def denoise(gray, method):
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=5)

    if method == "nlm":
        return cv2.fastNlMeansDenoising(gray, None, h=8,
                                        templateWindowSize=7,
                                        searchWindowSize=21)

    return gray


def stretch_contrast(gray, white_point=0.88):
    """
    Soft levels adjustment (NOT thresholding): the darkest ink maps to 0,
    anything lighter than `white_point` of the ink->paper range maps to 255
    (suppresses paper texture and faint ruling), with a linear ramp in
    between so anti-aliased stroke edges are preserved.

    Returns (image, applied). If the image has no usable dynamic range it is
    returned unchanged with applied=False.
    """

    paper = float(np.percentile(gray, 75))
    black = float(np.percentile(gray, 0.3))

    if paper - black < 20:
        return gray, False

    white = black + white_point * (paper - black)
    stretched = (gray.astype(np.float32) - black) / max(white - black, 1.0)

    return (np.clip(stretched, 0.0, 1.0) * 255.0).astype(np.uint8), True


# ============================================================
# NOTEBOOK RULING SUPPRESSION
# ============================================================

def suppress_ruling(gray, kernel_frac=0.08):
    """
    Remove long horizontal (ruled lines) and vertical (margin lines)
    structures while keeping handwriting strokes that cross them.

    Method: in the "ink is bright" domain, a morphological OPENING with a
    long 1-px line element keeps only structures that are continuous for
    >= kernel length. That estimate is the ruling's own intensity; because
    opening takes a minimum along the line, it does NOT include ink strokes
    crossing the ruling. Subtracting it therefore removes the ruling and
    leaves the stroke's excess darkness. A short vertical closing repairs
    the 1-3 px gaps left in strokes at the crossing.

    Side effect (documented limitation): genuinely long straight horizontal
    marks (underlines, strike-throughs) are removed as well.

    Returns (cleaned_gray, removed_fraction).
    """

    height, width = gray.shape
    dark = 255 - gray

    length_h = max(25, int(width * kernel_frac))

    # Vertical (margin-line) suppression only makes sense on page-like
    # images and for structures spanning a large part of the page height.
    # On a tight single-line crop, tall letters (l, k, b, h, d) are a large
    # fraction of the image height and would be erased.
    page_like = width <= 2.0 * height
    length_v = max(60, int(height * max(kernel_frac, 0.35)))

    # Thicken first so a line that drifts a few pixels across the window
    # (ruling not perfectly parallel to the deskewed text, ~1 degree) is
    # still one continuous structure for the 1-px opening below.
    thick_h = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    thick_v = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))

    horizontal = cv2.morphologyEx(
        thick_h, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (length_h, 1)),
    )
    if page_like:
        vertical = cv2.morphologyEx(
            thick_v, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, length_v)),
        )
    else:
        vertical = np.zeros_like(horizontal)

    structures = np.maximum(horizontal, vertical)
    subtracted = cv2.subtract(dark, structures)

    # Where a pixel is clearly darker than the ruling estimate it is a pen
    # stroke: keep it at full strength instead of lightening it by the
    # ruling's own darkness (that left grey patches where letters cross a
    # rule). Light ink (not that much darker than the rule) is still reduced.
    stroke = dark.astype(np.int16) > structures.astype(np.int16) + 70
    cleaned = np.where(stroke, dark, subtracted).astype(np.uint8)

    # Repair 1-3 px gaps that ruling removal can leave in light strokes, but
    # ONLY next to removed horizontal structure: a global closing would merge
    # the dots of i/j and accents into their letters.
    floor = float(np.median(horizontal))      # paper level, not a ruling
    zone = cv2.dilate((horizontal > floor + 12).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))) > 0
    closed = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    cleaned = np.where(zone, closed, cleaned).astype(np.uint8)

    removed_fraction = float((structures > floor + 12).mean())

    return 255 - cleaned, removed_fraction


# ============================================================
# INK MASK (for detection only)
# ============================================================

def ink_mask(gray):
    """
    Boolean mask of ink pixels on a flattened, contrast-stretched image.
    Used for orientation/skew/line detection ONLY; never given to the model.
    """

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    paper = float(np.percentile(blurred, 75))
    black = float(np.percentile(blurred, 0.5))

    if paper - black < 25:
        return np.zeros(gray.shape, dtype=bool)

    otsu, _ = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(np.clip(otsu, black + 0.25 * (paper - black),
                              black + 0.75 * (paper - black)))

    return blurred < threshold


def _quick_mask(gray, max_side=700):
    """Ink mask of a cheaply normalised downscaled copy (for estimation)."""

    small = limit_size(gray, max_side)
    flat = flatten_illumination(small)
    stretched, _ = stretch_contrast(flat)

    return ink_mask(stretched)


# ============================================================
# GEOMETRY: PERSPECTIVE, ORIENTATION, SKEW
# ============================================================

def _order_corners(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()

    return np.array([
        points[np.argmin(sums)],    # top-left
        points[np.argmin(diffs)],   # top-right
        points[np.argmax(sums)],    # bottom-right
        points[np.argmax(diffs)],   # bottom-left
    ], dtype=np.float32)


def _corner_angles(quad):
    angles = []

    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))

    return angles


def find_page_quad(gray):
    """
    Locate the paper as a convex quadrilateral (bright region against a
    darker surface). Deliberately conservative: returns None unless the
    result is clearly a page seen in perspective. A page that already fills
    the frame, or white paper on a white desk, returns None (no correction).
    """

    height, width = gray.shape
    scale = 500.0 / max(height, width)
    small = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else gray.copy()
    scale = small.shape[1] / width

    small = cv2.GaussianBlur(small, (5, 5), 0)
    _, bright = cv2.threshold(small, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(5, int(0.03 * max(small.shape))) | 1
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    frame_area = small.shape[0] * small.shape[1]

    if not (0.30 * frame_area <= area <= 0.97 * frame_area):
        return None

    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)

    if len(approx) != 4:
        return None

    quad = _order_corners(approx)

    if not all(55 <= angle <= 125 for angle in _corner_angles(quad)):
        return None

    # Page cut off by the frame: some quad EDGE runs along the image border
    # (corners merely touching the border is normal for a tightly framed,
    # rotated page).
    margin = 0.02 * max(small.shape)
    frame_h, frame_w = small.shape

    for i in range(4):
        samples = np.linspace(quad[i], quad[(i + 1) % 4], 12)
        along_border = np.mean([
            x < margin or y < margin or x > frame_w - margin or y > frame_h - margin
            for x, y in samples
        ])

        if along_border > 0.8:
            return None

    # already (almost) an upright rectangle: nothing to correct
    expected = np.array([[0, 0], [small.shape[1], 0],
                         [small.shape[1], small.shape[0]], [0, small.shape[0]]])
    if area > 0.9 * frame_area and np.abs(quad - expected).max() < 0.05 * max(small.shape):
        return None

    return quad / scale


def warp_to_quad(gray, quad, max_side):
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])

    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    longest = max(width, height)

    if longest > max_side:
        factor = max_side / longest
        width, height = int(width * factor), int(height * factor)

    target = np.array([[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]],
                      dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), target)

    warped = cv2.warpPerspective(gray, matrix, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    # drop a thin rim: the page edge / desk shadow is not handwriting
    inset_y, inset_x = int(0.015 * height), int(0.015 * width)

    return warped[inset_y:height - inset_y, inset_x:width - inset_x]


def detect_text_axis(gray):
    """
    Decide whether text runs horizontally or vertically (i.e. whether the
    page needs a 90-degree turn). Returns ("horizontal" | "vertical", ratio).

    Heuristic: for horizontal text the row ink profile fluctuates strongly
    (line / gap / line) while the column profile is comparatively flat.
    Wide strips (width/height > 2.5) are assumed to be single horizontal
    lines and are never rotated.
    """

    height, width = gray.shape

    if width / height > 2.5:
        return "horizontal", float("inf")

    mask = _quick_mask(gray)

    if mask.sum() < 50:
        return "horizontal", 1.0

    rows = mask.sum(axis=1).astype(np.float64)
    cols = mask.sum(axis=0).astype(np.float64)

    def coefficient_of_variation(profile):
        return profile.std() / (profile.mean() + 1e-9)

    ratio = coefficient_of_variation(rows) / (coefficient_of_variation(cols) + 1e-9)

    return ("vertical" if ratio < 0.8 else "horizontal"), float(ratio)


def candidate_rotations(gray, config):
    """
    Counter-clockwise rotations (degrees) worth trying. [0] when the page
    already looks upright. Several candidates mean the direction cannot be
    decided from ink statistics alone; the caller must pick one (the page
    recognizer uses the model's heuristic confidence for that).
    """

    if not config.auto_orient:
        return [0]

    axis, _ = detect_text_axis(gray)

    if axis == "vertical":
        return [90, 270]

    return [0, 180] if config.try_upside_down else [0]


def rotate_quarter(gray, degrees):
    return np.ascontiguousarray(np.rot90(gray, k=(degrees // 90) % 4))


def estimate_skew(gray, max_angle=12.0):
    """
    Small rotation (degrees, positive = text descends to the right) that
    makes the text rows sharpest, found by maximising the sharpness of the
    row projection profile. Returns (angle, confidence_ratio); the ratio
    compares the best score with the score at 0 degrees.
    """

    mask = _quick_mask(gray)
    ys, xs = np.nonzero(mask)

    if len(ys) < 80:
        return 0.0, 1.0

    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64)
    ys -= ys.mean()
    xs -= xs.mean()

    def score(angle_degrees):
        angle = np.radians(angle_degrees)
        rotated = ys * np.cos(angle) - xs * np.sin(angle)
        bins = np.round(rotated - rotated.min()).astype(np.int64)
        histogram = np.bincount(bins).astype(np.float64)

        return float(np.sum(np.diff(histogram) ** 2))

    coarse = np.arange(-max_angle, max_angle + 0.01, 0.5)
    coarse_scores = [score(a) for a in coarse]
    best = coarse[int(np.argmax(coarse_scores))]

    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    fine_scores = [score(a) for a in fine]
    best = float(fine[int(np.argmax(fine_scores))])

    baseline = score(0.0)
    ratio = max(fine_scores) / baseline if baseline > 0 else 1.0

    return best, float(ratio)


def rotate_small(gray, angle_degrees):
    """
    Rotate so text with positive skew (descending to the right) becomes
    level. The canvas is enlarged so nothing is cropped; the border
    replicates the paper.
    """

    height, width = gray.shape
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)

    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    return cv2.warpAffine(gray, matrix, (new_width, new_height),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# FULL PREPROCESSING
# ============================================================

@dataclass
class PageStages:
    original: np.ndarray      # grayscale after EXIF + size limit
    geometry: np.ndarray      # after perspective / rotation / deskew (raw intensities)
    normalized: np.ndarray    # + illumination + denoise + contrast
    processed: np.ndarray     # + ruling suppression (== normalized if off)
    mask: np.ndarray          # bool ink mask of `processed`
    info: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    config: PreprocessConfig = None


def preprocess_document(image, config=None, rotation=None):
    """
    Run the configurable preprocessing chain on a PIL image.

    rotation: None  -> detect (if config.auto_orient) and, when the
                       direction is ambiguous, use the first candidate
                       (page_recognition resolves ambiguity with the model);
              0/90/180/270 -> apply exactly this counter-clockwise turn.
    """

    config = config or PreprocessConfig()
    notes = []
    info = {}

    original = limit_size(load_gray(image), config.max_side)
    info["quality"] = image_quality(original)

    gray = original

    if config.perspective:
        quad = find_page_quad(gray)

        if quad is not None:
            gray = warp_to_quad(gray, quad, config.max_side)
            info["perspective_applied"] = True
            notes.append("Perspective corrected (page outline found).")
        else:
            info["perspective_applied"] = False
    else:
        info["perspective_applied"] = False

    if rotation is None:
        candidates = candidate_rotations(gray, config)
        rotation = candidates[0]
        info["rotation_candidates"] = candidates
    else:
        info["rotation_candidates"] = [rotation]

    info["rotation_applied"] = int(rotation)

    if rotation:
        gray = rotate_quarter(gray, rotation)
        notes.append(f"Rotated {rotation} degrees counter-clockwise.")

    skew = 0.0

    if config.deskew:
        skew, ratio = estimate_skew(gray, config.max_skew_deg)
        info["skew_confidence"] = ratio

        # leave level pages alone: tiny angles / no gain are noise
        if abs(skew) >= 0.8 and ratio > 1.1:
            gray = rotate_small(gray, skew)
            notes.append(f"Deskewed by {skew:+.1f} degrees.")
        else:
            skew = 0.0

    info["skew_degrees"] = skew
    geometry = gray

    normalized = geometry

    if config.illumination:
        normalized = flatten_illumination(normalized)

    normalized = denoise(normalized, config.denoise)

    if config.contrast:
        normalized, applied = stretch_contrast(normalized, config.white_point)
        info["contrast_applied"] = applied

        if not applied:
            notes.append("Image has almost no ink/paper contrast; "
                         "contrast stretch skipped.")

    processed = normalized
    info["ruling_removed_fraction"] = 0.0

    if config.suppress_ruling:
        processed, removed = suppress_ruling(normalized, config.ruling_kernel_frac)
        info["ruling_removed_fraction"] = removed

    mask = ink_mask(processed)
    info["ink_fraction"] = float(mask.mean())

    return PageStages(
        original=original,
        geometry=geometry,
        normalized=normalized,
        processed=processed,
        mask=mask,
        info=info,
        notes=notes,
        config=config,
    )


# ============================================================
# LINE DETECTION
# ============================================================

@dataclass
class LineBox:
    index: int
    x0: int
    y0: int
    x1: int           # exclusive
    y1: int           # exclusive
    mask: np.ndarray  # bool, shape (y1-y0, x1-x0): this line's ink pixels
    ink_pixels: int

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0


@dataclass
class LineDetection:
    kind: str                  # "single_line" | "multi_line" | "no_handwriting" | "not_handwriting"
    lines: list
    reason: str = ""
    info: dict = field(default_factory=dict)


def _estimate_pitch(clean, h_ref):
    """
    Line pitch (px) from the autocorrelation of the row profile, or None.
    Only trusted when the page is tall enough for at least 3 periods and the
    peak is strong; otherwise a single line's internal structure (x-height
    band / descenders) would be mistaken for several lines.
    """

    profile = _smooth_1d(clean.sum(axis=1), max(1.0, 0.1 * h_ref))
    profile = profile - profile.mean()

    if not np.any(profile):
        return None

    autocorrelation = np.correlate(profile, profile, mode="full")[len(profile) - 1:]
    autocorrelation = autocorrelation / (autocorrelation[0] + 1e-9)

    low = max(2, int(1.5 * h_ref))
    high = len(profile) // 3

    if high <= low + 2:
        return None

    lag = low + int(np.argmax(autocorrelation[low:high]))

    return lag if autocorrelation[lag] > 0.3 else None


def _split_rows(sub, ref_height):
    """
    Row ranges after cutting a too-tall blob at ink valleys. A cut is kept
    only if every resulting piece still contains a dense text band (its
    peak row-ink is at least half the blob's peak); otherwise the "piece"
    is just the sparse ascender/descender zone of one line and is merged
    back (row ink is normalised by each piece's own width, so a short line
    below a long one still counts as a text line). This keeps a single line from ever being sliced.
    """

    height = sub.shape[0]
    expected = int(round(height / ref_height))

    if expected < 2:
        return [(0, height)]

    profile = _smooth_1d(sub.sum(axis=1), max(1.0, 0.08 * ref_height))
    window = max(2, int(0.45 * ref_height))
    cuts = []

    for k in range(1, expected):
        center = int(round(k * height / expected))
        lo, hi = max(1, center - window), min(height - 1, center + window)

        if hi <= lo:
            continue

        cuts.append(lo + int(np.argmin(profile[lo:hi])))

    cuts = sorted(set(cuts))

    def strength(a, b):
        """Peak row-ink of a segment divided by its own horizontal extent, so a
        short line is not judged weak merely because a longer line sits next
        to it."""

        part = sub[a:b]
        cols = np.nonzero(part.any(axis=0))[0]
        extent = (cols[-1] - cols[0] + 1) if len(cols) else 1

        return float(profile[a:b].max()) / extent

    def weak_segment(boundaries):
        pairs = list(zip(boundaries[:-1], boundaries[1:]))
        best = max(strength(a, b) for a, b in pairs)

        for a, b in pairs:
            if b - a < 0.5 * ref_height or strength(a, b) < 0.5 * best:
                return (a, b)
        return None

    while cuts:
        boundaries = [0] + cuts + [height]
        weak = weak_segment(boundaries)

        if weak is None:
            break

        # drop the cut that bounds the weak segment on the side facing the
        # stronger neighbour
        a, b = weak
        candidates = [c for c in (a, b) if c in cuts]
        cuts.remove(candidates[0] if len(candidates) == 1 else
                    min(candidates, key=lambda c: profile[c]))

    boundaries = [0] + cuts + [height]

    return list(zip(boundaries[:-1], boundaries[1:]))


def detect_lines(mask, config=None):
    """
    Segment an ink mask into text lines (top-to-bottom reading order).

    Method: connected components -> remove specks / long straight remnants /
    page-sized blobs -> horizontally smear components into word/line blobs ->
    split blobs that are much taller than one line (at ink valleys) ->
    merge blobs that overlap vertically (same line) -> attach small marks
    (i-dots) to the nearest line, drop the rest -> sort by vertical position.
    """

    config = config or SegmentationConfig()
    height, width = mask.shape
    binary = mask.astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    info = {"components_raw": int(count - 1)}

    min_area = max(8, int(2e-6 * height * width))
    keep = np.zeros(count, dtype=bool)

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if area < min_area:
            continue
        if w * h > 0.5 * height * width and area / (w * h) > 0.5:
            continue                                  # shadow / dark border
        # thin sliver hugging the frame (shadow or desk edge after rotation)
        touches = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
        if touches and max(w / h, h / w) >= 8 and max(w, h) >= 0.25 * min(width, height):
            continue

        keep[i] = True

    # Straight remnants (ruling / margin-line fragments, strike marks): much
    # longer than a typical mark and extremely thin. Relative to the typical
    # mark height so real tall strokes (l, f, b) are not affected.
    if keep.any():
        typical = float(np.median(stats[keep, cv2.CC_STAT_HEIGHT]))

        for i in np.nonzero(keep)[0]:
            x, y, w, h, area = stats[i]
            long_thin_vertical = h >= 3.5 * typical and h >= 0.04 * height and h / w >= 15
            long_thin_horizontal = w >= 3.5 * typical and w >= 0.1 * width and w / h >= 15

            if long_thin_vertical or long_thin_horizontal:
                keep[i] = False

    kept = int(keep.sum())
    info["components_kept"] = kept
    ink_fraction = float(binary.mean())
    info["ink_fraction"] = ink_fraction

    if kept == 0:
        return LineDetection("no_handwriting", [],
                             "No ink-like marks were found on the page.", info)

    if ink_fraction > config.not_handwriting_ink_fraction:
        return LineDetection(
            "not_handwriting", [],
            f"Dark marks cover {ink_fraction * 100:.0f}% of the image; this "
            "does not look like handwriting on paper (check lighting/photo).",
            info)

    if kept > config.max_components:
        return LineDetection(
            "not_handwriting", [],
            f"{kept} separate marks were found; the image looks like texture "
            "or noise rather than handwriting.", info)

    clean = keep[labels]
    heights = stats[keep, cv2.CC_STAT_HEIGHT]
    h_ref = float(np.clip(np.median(heights), 6, max(8, height / 2)))
    info["typical_mark_height"] = h_ref

    kernel_w = max(3, int(config.smear_factor * h_ref))
    kernel_h = max(1, int(0.12 * h_ref))
    smeared = cv2.dilate(clean.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)))

    blob_count, blob_labels, blob_stats, _ = cv2.connectedComponentsWithStats(
        smeared, connectivity=8)

    blob_heights = []
    for j in range(1, blob_count):
        x, y, w, h, _a = blob_stats[j]
        sub = clean[y:y + h, x:x + w] & (blob_labels[y:y + h, x:x + w] == j)
        rows = np.nonzero(sub.any(axis=1))[0]
        if len(rows):
            blob_heights.append(rows[-1] - rows[0] + 1)

    median_blob = float(np.median(blob_heights)) if blob_heights else h_ref
    pitch = _estimate_pitch(clean, h_ref)
    info["line_pitch"] = pitch

    ref_height = median_blob
    if pitch and median_blob > 1.5 * pitch:
        ref_height = 0.85 * pitch              # blobs merged several lines
    ref_height = max(ref_height, h_ref)
    info["reference_line_height"] = ref_height

    # ---- pieces: (x0, y0, mask) with ink pixels only
    pieces = []

    for j in range(1, blob_count):
        x, y, w, h, _a = blob_stats[j]
        sub = clean[y:y + h, x:x + w] & (blob_labels[y:y + h, x:x + w] == j)

        if not sub.any():
            continue

        row_ids = np.nonzero(sub.any(axis=1))[0]
        sub = sub[row_ids[0]:row_ids[-1] + 1]
        y += row_ids[0]

        ranges = (_split_rows(sub, ref_height)
                  if sub.shape[0] > config.split_tall_factor * ref_height
                  else [(0, sub.shape[0])])

        for a, b in ranges:
            piece = sub[a:b]
            if not piece.any():
                continue
            cols = np.nonzero(piece.any(axis=0))[0]
            rows = np.nonzero(piece.any(axis=1))[0]
            piece = piece[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
            pieces.append((x + cols[0], y + a + rows[0], piece))

    if not pieces:
        return LineDetection("no_handwriting", [],
                             "No ink-like marks were found on the page.", info)

    # ---- merge pieces that share vertical extent (same line)
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    spans = [(p[1], p[1] + p[2].shape[0]) for p in pieces]

    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            overlap = min(spans[i][1], spans[j][1]) - max(spans[i][0], spans[j][0])
            smaller = min(spans[i][1] - spans[i][0], spans[j][1] - spans[j][0])

            if smaller > 0 and overlap / smaller > 0.5:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(len(pieces)):
        groups.setdefault(find(i), []).append(i)

    def group_box(members):
        x0 = min(pieces[i][0] for i in members)
        y0 = min(pieces[i][1] for i in members)
        x1 = max(pieces[i][0] + pieces[i][2].shape[1] for i in members)
        y1 = max(pieces[i][1] + pieces[i][2].shape[0] for i in members)
        return x0, y0, x1, y1

    boxes = {g: group_box(m) for g, m in groups.items()}
    group_heights = {g: b[3] - b[1] for g, b in boxes.items()}
    median_height = float(np.median(list(group_heights.values())))

    # ---- attach small marks (i-dots, commas) to a nearby line, drop the rest
    big = [g for g in groups if group_heights[g] >= config.min_line_height_factor * median_height]
    small = [g for g in groups if g not in big]

    if not big:
        big, small = list(groups), []

    for g in small:
        x0, y0, x1, y1 = boxes[g]
        best, best_gap = None, None

        for b in big:
            bx0, by0, bx1, by1 = boxes[b]
            gap = max(by0 - y1, y0 - by1, 0)
            inside = x0 < bx1 + h_ref and x1 > bx0 - h_ref

            if inside and gap <= 0.8 * ref_height and (best_gap is None or gap < best_gap):
                best, best_gap = b, gap

        if best is not None:
            groups[best].extend(groups[g])

    boxes = {g: group_box(groups[g]) for g in big}

    # ---- build LineBox objects in reading order
    ordered = sorted(big, key=lambda g: ((boxes[g][1] + boxes[g][3]) / 2, boxes[g][0]))
    lines = []

    for index, g in enumerate(ordered):
        x0, y0, x1, y1 = boxes[g]
        line_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)

        for i in groups[g]:
            px, py, piece = pieces[i]
            line_mask[py - y0:py - y0 + piece.shape[0],
                      px - x0:px - x0 + piece.shape[1]] |= piece

        lines.append(LineBox(index, int(x0), int(y0), int(x1), int(y1),
                             line_mask, int(line_mask.sum())))

    if len(lines) == 1 and lines[0].ink_pixels < 5 * min_area:
        return LineDetection("no_handwriting", [],
                             "Only tiny marks were found; no handwriting detected.", info)

    kind = "single_line" if len(lines) == 1 else "multi_line"

    return LineDetection(kind, lines, "", info)


# ============================================================
# LINE CROPS FOR THE MODEL
# ============================================================

def crop_line(gray, line, reference_height, config=None):
    """
    Cut one line out of the processed gray page, with margins, on a white
    canvas. The crop height is max(line height, reference_height) so lines
    without ascenders/descenders are not blown up relative to others.
    Returns (crop_uint8, (left, top)) where (left, top) is the crop origin in
    page coordinates.
    """

    config = config or SegmentationConfig()
    page_h, page_w = gray.shape

    base = max(line.height, reference_height)
    crop_h = max(8, int(round(base * (1 + 2 * config.margin))))
    margin_x = int(round(config.margin_x * base))

    center_y = (line.y0 + line.y1) / 2.0
    top = int(round(center_y - crop_h / 2.0))
    left = line.x0 - margin_x
    crop_w = line.width + 2 * margin_x

    canvas = np.full((crop_h, crop_w), 255, dtype=np.uint8)

    sy0, sy1 = max(0, top), min(page_h, top + crop_h)
    sx0, sx1 = max(0, left), min(page_w, left + crop_w)

    if sy1 > sy0 and sx1 > sx0:
        canvas[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = gray[sy0:sy1, sx0:sx1]

    if config.isolate_lines:
        own = np.zeros((crop_h, crop_w), dtype=np.uint8)
        oy, ox = line.y0 - top, line.x0 - left
        own[oy:oy + line.height, ox:ox + line.width] = line.mask.astype(np.uint8)
        radius = max(1, int(0.1 * line.height))
        own = cv2.dilate(own, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
        canvas = np.where(own > 0, canvas, 255).astype(np.uint8)

    return canvas, (left, top)


def line_mask_in_crop(line, crop_shape, origin):
    """The line's ink mask positioned inside its crop (for chunking)."""

    crop_h, crop_w = crop_shape
    left, top = origin
    placed = np.zeros((crop_h, crop_w), dtype=bool)
    oy, ox = line.y0 - top, line.x0 - left
    placed[oy:oy + line.height, ox:ox + line.width] = line.mask

    return placed


def split_line_into_chunks(crop_mask, max_aspect, ink_height, margin_x, min_fit=0.6):
    """
    A model canvas is at most `max_aspect` times wider than tall (8 for
    128x1024). A long handwritten line exceeds that and would be shrunk
    (making the text tiny). Split such a line at WORD GAPS into chunks that
    fit; each chunk is recognised separately and the texts are joined with
    spaces. A single word wider than the limit is kept whole (it will be
    shrunk).

    Returns a list of (x_start, x_end) in crop coordinates.
    """

    crop_h, crop_w = crop_mask.shape

    # A line only slightly wider than the canvas is shrunk to fit (as the
    # original pipeline does) rather than cut: cutting costs word context and
    # adds edge artefacts. Cut only when shrinking would go below `min_fit`.
    if crop_w <= max_aspect * crop_h / min_fit:
        return [(0, crop_w)]

    columns = crop_mask.any(axis=0)
    ink_columns = np.nonzero(columns)[0]

    if len(ink_columns) == 0:
        return [(0, crop_w)]

    gap_min = max(3, int(0.35 * ink_height))
    words, start, previous = [], ink_columns[0], ink_columns[0]

    for column in ink_columns[1:]:
        if column - previous > gap_min:
            words.append((int(start), int(previous) + 1))
            start = column
        previous = column

    words.append((int(start), int(previous) + 1))

    limit = max_aspect * crop_h - 2 * margin_x
    chunks, current_start, current_end = [], words[0][0], words[0][1]

    for word_start, word_end in words[1:]:
        if word_end - current_start <= limit:
            current_end = word_end
        else:
            chunks.append((current_start, current_end))
            current_start, current_end = word_start, word_end

    chunks.append((current_start, current_end))

    # Pad each chunk with blank margin, but never further than half-way into
    # the gap to the neighbouring chunk (otherwise a sliver of the next
    # word's first letter leaks into this chunk and is read as junk).
    padded = []

    for index, (a, b) in enumerate(chunks):
        left = margin_x if index == 0 else min(margin_x, (a - chunks[index - 1][1]) // 2)
        right = (margin_x if index == len(chunks) - 1
                 else min(margin_x, (chunks[index + 1][0] - b) // 2))

        padded.append((max(0, a - left), min(crop_w, b + right)))

    return padded


# ============================================================
# VISUALISATION
# ============================================================

def draw_line_overlay(gray, lines):
    """RGB uint8 image with numbered boxes around the detected lines."""

    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    palette = [(220, 50, 47), (38, 139, 210), (42, 161, 152),
               (181, 137, 0), (108, 113, 196), (203, 75, 22)]
    thickness = max(2, int(round(max(gray.shape) / 500)))
    scale = max(0.6, max(gray.shape) / 1200)

    for line in lines:
        color = palette[line.index % len(palette)]
        cv2.rectangle(canvas, (line.x0, line.y0), (line.x1, line.y1), color, thickness)
        cv2.putText(canvas, str(line.index + 1),
                    (line.x0, max(15, line.y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    max(1, thickness - 1), cv2.LINE_AA)

    return canvas


def with_config(config, **changes):
    return replace(config, **changes)
