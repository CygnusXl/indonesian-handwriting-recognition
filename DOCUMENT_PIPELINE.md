# Photograph pipeline (preprocessing -> line detection -> recognition)

## Why it exists

The trained model is a **single-line handwriting recognizer**. `htr_common.preprocess_image`
scales its input to 128 px high (at most 1024 px wide, aspect preserved, white canvas), the
form it saw in training. A photograph of a whole notebook page, squeezed to 128 px high,
leaves each handwritten line only a few pixels tall (observation from the geometry of the
code, e.g. a 1900 px-high page becomes 6.7% of its size). Typical failure: meaningless
output such as "SA." at ~42% heuristic confidence. This is a **layout/scale mismatch**
between the input and the training distribution; whether handwriting style and the
synthetic-only training also limit accuracy can only be answered with real photographs
(`evaluation/real_world/README.md`). The recognition model was **not changed or retrained**.

Measured geometry on one SYNTHETIC 1400x1900 page (4 lines, each about 50 px tall; not real
handwriting, geometry only): pipeline A scales the page by 0.067, so a line is about 3 px
tall inside the 128 px model input; pipeline D feeds each line at about 59-67% of the input
height, the scale the model was trained on (`ChunkResult.ink_height_fraction`, shown in
the app's "What the model sees"). The size of this gap on real photographs depends on how
large the writing is in the photo.

## Stages (`src/document_pipeline.py`, `src/page_recognition.py`)

| Stage | What is done | Configurable in |
|---|---|---|
| 1. Load | EXIF orientation, RGB/RGBA/palette -> grayscale, long side limited to 2400 px | `PreprocessConfig.max_side` |
| 2. Perspective | find the page outline (largest roughly-rectangular quadrilateral); warp it flat; skipped when no plausible quad is found or the quad is nearly the whole frame | `perspective` |
| 3. Orientation | upright pages are left alone. Text axis from row/column ink profiles; a sideways page gets a 90/270 candidate pair, resolved by the model's heuristic confidence (not validated). 180 degrees is only tried when asked | `auto_orient`, `try_upside_down`, manual choice in the app |
| 4. Deskew | projection-profile search within +-12 degrees; only applied when the angle is >= 0.8 degrees and the gain is clear | `deskew`, `max_skew_deg` |
| 5. Illumination | background estimated (median filter on sparse-ink pages, closing on dense crops) and divided out, so shadows and gradients flatten | `illumination` |
| 6. Denoise | edge-preserving bilateral filter | `denoise` |
| 7. Contrast | soft percentile levels (no hard threshold: strokes keep their grey values) | `contrast`, `white_point` |
| 8. Ruling suppression (D only) | long horizontal/vertical thin structures removed by a morphological opening estimate; strokes clearly darker than the rule are kept at full strength; vertical suppression only on page-like images | `suppress_ruling` |
| 9. Line detection | ink mask -> connected components -> horizontal smearing into blobs -> splitting of over-tall blobs at ink valleys -> merging/attaching small marks -> sorted top to bottom. Result is *single line*, *multiple lines*, *no handwriting* (blank) or *not handwriting* (ink everywhere / noise) | `SegmentationConfig` |
| 10. Crops | each line is cropped with a margin; a line wider than 8x its height is cut at word gaps into chunks that fit the 128x1024 canvas (a single over-long word is shrunk and flagged) | `margin`, `margin_x` |
| 11. Recognition | the unchanged CNN-BiLSTM-CTC model on every crop (batched; GPU if the model is on GPU, otherwise CPU) | - |
| 12. Join | chunks joined with spaces, lines joined with newlines in reading order | - |

Adaptive/binary thresholding is deliberately **not** applied to the recognizer input: the
mask is used only to find lines, the recognizer sees grey-level crops.

## Pipelines to compare

`A` original, `B` improved preprocessing only, `C` + line segmentation, `D` + ruling
suppression (default). See `evaluation/real_world/README.md` for how to run and read them.

## Fallbacks and messages

- Line detection raises an error -> warning + single-line pipeline on the whole image.
- Blank page / not handwriting -> clear message, recognition skipped unless "Recognize
  anyway" is ticked.
- Low resolution, low contrast, dark, blurry -> warning explaining the likely problem; the
  attempt is still made (thresholds are heuristics, unvalidated).
- Very small handwriting or an over-long word -> per-line warning.

## Recommended input

Flat, evenly lit, photographed from above with the page filling most of the frame and the
handwriting at least ~25 px tall in the image; dark ink on light paper; no fingers or
objects over the text. One page at a time. For best results on a single line, crop it
tightly (the app then does not split it).

## Known limitations (all UNVERIFIED on real handwriting unless stated)

- No real-handwriting measurement exists in this repository. Synthetic-page checks
  (`src/evaluate_segmentation.py`) test geometry only and are optimistic: the thresholds
  were developed on the same generator.
- The recognizer was trained on synthetic Indonesian handwriting; real handwriting, other
  languages (e.g. English photos), cursive and pen/ink variation may fail regardless of
  preprocessing.
- Ruling suppression removes long straight marks: underlines and strike-throughs vanish
  (crossed-out words are therefore *read*, not skipped), and a very light ink stroke can
  thin where it meets a rule.
- 90 vs 270 degree ambiguity is resolved with the model's heuristic confidence; 180-degree
  detection is not validated. Use the manual rotation selector when auto fails.
- Two-column pages, tables, drawings and margin notes are not understood; lines are found
  by geometry only, so a page with columns will interleave them.
- Perspective correction needs the paper to be clearly brighter than its surroundings (white paper on a white desk, or a page cut off by the frame, is left uncorrected).
- Heavily curved pages (open book, bent paper) are not dewarped.
- Heuristic confidence is not calibrated.
- Performance on a hosted CPU instance for large photos has not been measured.
