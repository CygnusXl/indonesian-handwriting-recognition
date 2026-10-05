import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from datasets import load_dataset

from dataset import DATASET_NAME


# ============================================================
# CONFIGURATION
# ============================================================

SPLIT = "train"

# Test more samples before deciding whether the method works.
NUM_SAMPLES = 20

OUTPUT_DIR = "results/line_segmentation_v2"

# ------------------------------------------------------------
# Image threshold
# ------------------------------------------------------------
#
# Lower value = more conservative about what counts as ink.
#
# We use a grayscale threshold rather than aggressive
# morphology so neighboring lines are less likely to merge.
# ------------------------------------------------------------

BINARY_THRESHOLD = 200

# ------------------------------------------------------------
# Small connected components are removed as noise.
# ------------------------------------------------------------

MIN_COMPONENT_AREA = 8

# ------------------------------------------------------------
# Projection threshold.
#
# A row must contain at least this fraction of ink pixels
# before it is considered part of handwriting.
# ------------------------------------------------------------

MIN_INK_RATIO = 0.0015

# ------------------------------------------------------------
# Maximum number of empty rows allowed inside one line.
# Keep this small so separate lines remain separate.
# ------------------------------------------------------------

MAX_GAP = 3

# ------------------------------------------------------------
# Smooth the vertical projection over this many rows.
# Must be odd.
# ------------------------------------------------------------

SMOOTH_KERNEL = 5

# ------------------------------------------------------------
# Margin around detected line.
# ------------------------------------------------------------

LINE_MARGIN = 6


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TEXT HELPERS
# ============================================================

def is_multiline_text(text):
    """
    Determine whether the transcription contains multiple
    lines.

    The dataset may contain either an actual newline character
    or the literal characters "\\n".
    """

    return (
        "\n" in text
        or "\\n" in text
    )


def split_expected_lines(text):
    """
    Split the ground-truth transcription into its expected
    individual lines.
    """

    lines = re.split(
        r"\\n|\n",
        text
    )

    lines = [
        line
        for line in lines
        if line.strip()
    ]

    return lines


def expected_line_count(text):
    """
    Return the number of expected text lines.
    """

    return len(
        split_expected_lines(text)
    )


# ============================================================
# REMOVE SMALL COMPONENTS
# ============================================================

def remove_small_components(
    binary,
    min_area=MIN_COMPONENT_AREA
):
    """
    Remove tiny isolated components from a binary image.

    This reduces noise without using a large morphological
    closing operation that could connect separate lines.
    """

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        binary
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            cleaned[
                labels == label
            ] = 255

    return cleaned


# ============================================================
# LINE SEGMENTATION
# ============================================================

def segment_lines(
    image,
    binary_threshold=BINARY_THRESHOLD,
    min_ink_ratio=MIN_INK_RATIO,
    max_gap=MAX_GAP,
    smooth_kernel=SMOOTH_KERNEL,
    line_margin=LINE_MARGIN
):
    """
    Segment multiline handwriting using a vertical
    projection profile.

    Returns:

        crops:
            list of PIL Image objects

        boxes:
            list of (x1, y1, x2, y2)
    """

    # --------------------------------------------------------
    # Convert to grayscale.
    # --------------------------------------------------------

    gray = np.array(
        image.convert("L")
    )

    height, width = gray.shape

    # --------------------------------------------------------
    # Threshold dark handwriting.
    #
    # White background -> 0
    # Dark handwriting -> 255
    # --------------------------------------------------------

    binary = np.where(
        gray < binary_threshold,
        255,
        0
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Remove tiny isolated noise.
    # --------------------------------------------------------

    binary = remove_small_components(
        binary
    )

    # --------------------------------------------------------
    # Horizontal ink projection.
    #
    # For each y-coordinate, count the number of dark pixels.
    # --------------------------------------------------------

    projection = np.sum(
        binary > 0,
        axis=1
    ).astype(
        np.float32
    )

    # Normalize projection to [0,1].
    if projection.max() > 0:

        projection = (
            projection
            / projection.max()
        )

    # --------------------------------------------------------
    # Smooth projection.
    # --------------------------------------------------------

    if smooth_kernel % 2 == 0:

        smooth_kernel += 1

    smoothed = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (
            1,
            smooth_kernel
        ),
        0
    ).reshape(-1)

    # --------------------------------------------------------
    # Determine active rows.
    # --------------------------------------------------------

    active_rows = (
        smoothed >= min_ink_ratio
    )

    # --------------------------------------------------------
    # Find contiguous active regions.
    # --------------------------------------------------------

    raw_segments = []

    start = None

    for y, active in enumerate(
        active_rows
    ):

        if active:

            if start is None:

                start = y

        else:

            if start is not None:

                raw_segments.append(
                    (
                        start,
                        y - 1
                    )
                )

                start = None

    # Handle segment reaching image bottom.
    if start is not None:

        raw_segments.append(
            (
                start,
                height - 1
            )
        )

    # --------------------------------------------------------
    # Merge only very small gaps.
    #
    # This is intentionally much more conservative than the
    # previous version.
    # --------------------------------------------------------

    merged_segments = []

    for segment in raw_segments:

        if not merged_segments:

            merged_segments.append(
                [
                    segment[0],
                    segment[1]
                ]
            )

            continue

        previous_start = (
            merged_segments[-1][0]
        )

        previous_end = (
            merged_segments[-1][1]
        )

        current_start = (
            segment[0]
        )

        current_end = (
            segment[1]
        )

        gap = (
            current_start
            - previous_end
            - 1
        )

        if gap <= max_gap:

            merged_segments[-1][1] = (
                current_end
            )

        else:

            merged_segments.append(
                [
                    current_start,
                    current_end
                ]
            )

    # --------------------------------------------------------
    # Remove extremely thin detections.
    # --------------------------------------------------------

    filtered_segments = []

    for start_y, end_y in merged_segments:

        segment_height = (
            end_y
            - start_y
            + 1
        )

        if segment_height >= 5:

            filtered_segments.append(
                (
                    start_y,
                    end_y
                )
            )

    # --------------------------------------------------------
    # Create bounding boxes and crops.
    # --------------------------------------------------------

    boxes = []
    crops = []

    for start_y, end_y in filtered_segments:

        # Add vertical margin.
        y1 = max(
            0,
            start_y - line_margin
        )

        y2 = min(
            height,
            end_y + line_margin + 1
        )

        # ----------------------------------------------------
        # Determine horizontal ink bounds for this line.
        # ----------------------------------------------------

        line_binary = binary[
            y1:y2,
            :
        ]

        coordinates = np.where(
            line_binary > 0
        )

        if len(
            coordinates[0]
        ) == 0:

            continue

        x1 = max(
            0,
            int(
                coordinates[1].min()
            )
            - line_margin
        )

        x2 = min(
            width,
            int(
                coordinates[1].max()
            )
            + line_margin
            + 1
        )

        box = (
            x1,
            y1,
            x2,
            y2
        )

        boxes.append(
            box
        )

        crops.append(
            image.crop(box)
        )

    return (
        crops,
        boxes,
        projection,
        smoothed
    )


# ============================================================
# PREVIEW
# ============================================================

def create_preview(
    image,
    boxes,
    sample_number,
    expected_lines
):
    """
    Draw detected line boxes on the original image.
    """

    preview = image.convert(
        "RGB"
    ).copy()

    draw = ImageDraw.Draw(
        preview
    )

    # --------------------------------------------------------
    # Draw detected boxes.
    # --------------------------------------------------------

    for index, box in enumerate(
        boxes,
        start=1
    ):

        x1, y1, x2, y2 = box

        draw.rectangle(
            box,
            outline=(255, 0, 0),
            width=3
        )

        # Put line number above the box.
        label_y = max(
            0,
            y1 - 15
        )

        draw.text(
            (
                x1 + 4,
                label_y
            ),
            f"L{index}",
            fill=(255, 0, 0)
        )

    # --------------------------------------------------------
    # Add information banner.
    # --------------------------------------------------------

    banner_height = 35

    canvas = Image.new(
        "RGB",
        (
            preview.width,
            preview.height
            + banner_height
        ),
        "white"
    )

    canvas.paste(
        preview,
        (
            0,
            banner_height
        )
    )

    draw = ImageDraw.Draw(
        canvas
    )

    info = (
        f"Sample {sample_number} | "
        f"Expected: {expected_lines} | "
        f"Detected: {len(boxes)}"
    )

    draw.text(
        (
            10,
            8
        ),
        info,
        fill=(0, 0, 0)
    )

    return canvas


# ============================================================
# PROJECTION PREVIEW
# ============================================================

def save_projection_plot(
    projection,
    smoothed,
    sample_number
):
    """
    Save the vertical projection and smoothed projection
    as a simple PNG using OpenCV drawing.

    This helps us see why lines were or were not separated.
    """

    plot_height = 300
    plot_width = 1000

    canvas = np.ones(
        (
            plot_height,
            plot_width,
            3
        ),
        dtype=np.uint8
    ) * 255

    if len(projection) == 0:

        return

    maximum = max(
        1e-6,
        float(
            max(
                projection.max(),
                smoothed.max()
            )
        )
    )

    # --------------------------------------------------------
    # Draw raw projection.
    # --------------------------------------------------------

    raw_points = []

    for y, value in enumerate(
        projection
    ):

        x = int(
            value
            / maximum
            * (plot_width - 1)
        )

        py = int(
            y
            / max(1, len(projection) - 1)
            * (plot_height - 1)
        )

        raw_points.append(
            (
                x,
                py
            )
        )

    for i in range(
        1,
        len(raw_points)
    ):

        cv2.line(
            canvas,
            raw_points[i - 1],
            raw_points[i],
            (180, 180, 180),
            1
        )

    # --------------------------------------------------------
    # Draw smoothed projection.
    # --------------------------------------------------------

    smooth_points = []

    for y, value in enumerate(
        smoothed
    ):

        x = int(
            value
            / maximum
            * (plot_width - 1)
        )

        py = int(
            y
            / max(1, len(smoothed) - 1)
            * (plot_height - 1)
        )

        smooth_points.append(
            (
                x,
                py
            )
        )

    for i in range(
        1,
        len(smooth_points)
    ):

        cv2.line(
            canvas,
            smooth_points[i - 1],
            smooth_points[i],
            (0, 0, 255),
            2
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_number:02d}_projection.png"
    )

    cv2.imwrite(
        output_path,
        canvas
    )


# ============================================================
# SAVE CROPS
# ============================================================

def save_line_crops(
    crops,
    sample_number
):
    """
    Save every detected line crop.
    """

    sample_dir = os.path.join(
        OUTPUT_DIR,
        f"sample_{sample_number:02d}"
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )

    for index, crop in enumerate(
        crops,
        start=1
    ):

        path = os.path.join(
            sample_dir,
            f"line_{index:02d}.png"
        )

        crop.save(
            path
        )


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    previews
):
    """
    Create one large preview containing all tested samples.
    """

    if not previews:

        return

    columns = 2

    max_width = max(
        image.width
        for image in previews
    )

    max_height = max(
        image.height
        for image in previews
    )

    rows = (
        len(previews)
        + columns
        - 1
    ) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * max_width,
            rows * max_height
        ),
        "white"
    )

    for index, preview in enumerate(
        previews
    ):

        x = (
            index % columns
        ) * max_width

        y = (
            index // columns
        ) * max_height

        sheet.paste(
            preview,
            (
                x,
                y
            )
        )

    output_path = os.path.join(
        "results",
        "line_segmentation_v2_preview.png"
    )

    sheet.save(
        output_path
    )

    print()
    print(
        "Contact sheet saved to:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "MULTILINE HANDWRITING SEGMENTATION V2"
    )
    print("=" * 70)

    print()
    print(
        f"Dataset split: {SPLIT}"
    )

    print(
        f"Samples to test: "
        f"{NUM_SAMPLES}"
    )

    print(
        f"Binary threshold: "
        f"{BINARY_THRESHOLD}"
    )

    print(
        f"Minimum ink ratio: "
        f"{MIN_INK_RATIO}"
    )

    print(
        f"Maximum gap: "
        f"{MAX_GAP}"
    )

    print(
        f"Minimum component area: "
        f"{MIN_COMPONENT_AREA}"
    )

    # --------------------------------------------------------
    # Load dataset.
    # --------------------------------------------------------

    print()
    print(
        "Loading dataset..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT
    )

    # --------------------------------------------------------
    # Collect multiline Indonesian samples.
    # --------------------------------------------------------

    candidates = []

    for index, sample in enumerate(
        dataset
    ):

        if sample[
            "language"
        ] != "ind":

            continue

        text = sample[
            "text"
        ]

        if not is_multiline_text(
            text
        ):

            continue

        candidates.append(
            index
        )

        if len(candidates) >= NUM_SAMPLES:

            break

    print(
        f"Multiline Indonesian samples found: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Process.
    # --------------------------------------------------------

    previews = []

    correct_line_counts = 0
    total_absolute_error = 0

    print()
    print("=" * 70)
    print(
        "SEGMENTATION RESULTS"
    )
    print("=" * 70)

    for sample_number, dataset_index in enumerate(
        candidates,
        start=1
    ):

        sample = dataset[
            dataset_index
        ]

        image = sample[
            "image"
        ]

        text = sample[
            "text"
        ]

        expected = expected_line_count(
            text
        )

        # ----------------------------------------------------
        # Segment.
        # ----------------------------------------------------

        (
            crops,
            boxes,
            projection,
            smoothed
        ) = segment_lines(
            image
        )

        detected = len(
            boxes
        )

        # ----------------------------------------------------
        # Accuracy bookkeeping.
        # ----------------------------------------------------

        if detected == expected:

            correct_line_counts += 1

        total_absolute_error += abs(
            detected
            - expected
        )

        # ----------------------------------------------------
        # Print.
        # ----------------------------------------------------

        print()
        print(
            f"Sample {sample_number}"
        )

        print(
            f"Dataset index: "
            f"{dataset_index}"
        )

        print(
            f"Writer ID: "
            f"{sample['writer_id']}"
        )

        print(
            f"Expected lines: "
            f"{expected}"
        )

        print(
            f"Detected lines: "
            f"{detected}"
        )

        print(
            f"Text length: "
            f"{len(text)}"
        )

        print(
            f"Neatness: "
            f"{sample['neatness']}"
        )

        print(
            f"Noise: "
            f"{sample['augmented_noise']}"
        )

        print(
            f"Line width: "
            f"{sample['line_width']}"
        )

        if detected == expected:

            print(
                "Line count: PASS"
            )

        else:

            print(
                "Line count: REVIEW"
            )

        # ----------------------------------------------------
        # Save preview.
        # ----------------------------------------------------

        preview = create_preview(
            image=image,
            boxes=boxes,
            sample_number=sample_number,
            expected_lines=expected
        )

        preview_path = os.path.join(
            OUTPUT_DIR,
            f"sample_{sample_number:02d}_preview.png"
        )

        preview.save(
            preview_path
        )

        previews.append(
            preview
        )

        # ----------------------------------------------------
        # Save individual crops.
        # ----------------------------------------------------

        save_line_crops(
            crops,
            sample_number
        )

        # ----------------------------------------------------
        # Save projection.
        # ----------------------------------------------------

        save_projection_plot(
            projection,
            smoothed,
            sample_number
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    tested = len(
        candidates
    )

    if tested > 0:

        count_accuracy = (
            correct_line_counts
            / tested
        )

        mean_absolute_error = (
            total_absolute_error
            / tested
        )

    else:

        count_accuracy = 0.0
        mean_absolute_error = 0.0

    print()
    print("=" * 70)
    print(
        "SEGMENTATION SUMMARY"
    )
    print("=" * 70)

    print()
    print(
        f"Samples tested: "
        f"{tested}"
    )

    print(
        f"Exact line-count matches: "
        f"{correct_line_counts}/{tested}"
    )

    print(
        f"Line-count accuracy: "
        f"{count_accuracy * 100:.2f}%"
    )

    print(
        f"Mean absolute line-count error: "
        f"{mean_absolute_error:.2f}"
    )

    # --------------------------------------------------------
    # Contact sheet.
    # --------------------------------------------------------

    create_contact_sheet(
        previews
    )

    # --------------------------------------------------------
    # Complete.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SEGMENTATION V2 COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Main preview:"
    )

    print(
        "results/line_segmentation_v2_preview.png"
    )

    print()
    print(
        "Individual samples:"
    )

    print(
        f"{OUTPUT_DIR}/"
    )


if __name__ == "__main__":
    main()