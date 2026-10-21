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
height (`ChunkResult.ink_height_fraction`, shown in the app's "What the model sees").

## Checks on dataset samples (real model, tiny n, NOT a benchmark)

`results/indonesian_samples.png` (committed) shows five OpenHand-Synth images. I cropped them
out of that figure (screenshot resolution; whether they were in the training split is unknown)
and ran the real `models/final_model.pt` on them. Observations, n = 1 multi-line image and
3 single-line images with a reference:

- Multi-line sample: pipeline A returns near-garbage for the whole page; pipelines C/D find the
  4 lines and read most of 3 of them (the first letter of line 1 and line 4 are wrong or lost).
  This is the failure mode the pipeline targets; it is one image.
- Single-line samples (CER, whole-image A vs D): A 25.9%, D 31.1%. Differences are within what
  3 images can show; D is **not** shown to be better than A on an already-cropped single line.
  For a tight single-line crop, pipeline A is a legitimate choice.
- Writing size matters a lot. Sweeping the vertical margin (so the writing fills 100% ... 20% of
  the 128 px input height) on the same 3 lines: CER mean 0.44 at 100%, 0.35 at 67% (default),
  0.34 at 50%, 0.44 at 33%, 0.65 at 25%, 0.76 at 20%. So text smaller than about a third of the
  input height is read badly, which is consistent with pipeline A's ~5% on a whole page; between
  50% and 67% there is no resolvable difference with this n, so the default was not retuned.
  What ink-height fraction the TRAINING data had was not measured (the dataset is not reachable
  from the build environment).
- Two bugs found this way and fixed (regression tests in `tests/test_document_pipeline.py`):
  tightly spaced lines where a short line sits under a longer one were merged into one box;
  lines only slightly wider than 8:1 were needlessly cut into word chunks (now shrunk to fit,
  like pipeline A, unless that would shrink them below 60%).
- A synthetic-only benchmark had not caught either bug, which is why its numbers are optimistic.

## Pipeline on the 208 synthetic test images (run by Peter, 2026-10-19; real model)

Command: `src/pipeline_on_test_set.py export`, then `src/evaluate_pipeline.py --experiments A C D`, then `summarize`. These are OpenHand-Synth **dataset images, not photographs**: this says nothing about real handwriting. One run, no confidence intervals for C/D (the A row reproduces the baseline's single-line/multiline numbers from `results/analysis/line_level_report.md`).

| Pipeline | Group | n | Corpus CER | Corpus WER | Exact match |
| --- | --- | ---: | ---: | ---: | ---: |
| A (original) | all | 208 | 68.3% | 87.3% | 6.2% |
| A | multiline | 71 | 90.3% | 99.6% | 0.0% |
| A | single | 137 | 17.1% | 58.1% | 9.5% |
| C (line splitting) | all | 208 | 28.8% | 75.9% | 1.4% |
| C | multiline | 71 | 31.6% | 77.6% | 0.0% |
| C | single | 137 | 22.2% | 71.9% | 2.2% |
| D (C + ruling suppression) | all | 208 | 28.8% | 75.8% | 1.4% |
| D | multiline | 71 | 31.6% | 77.6% | 0.0% |
| D | single | 137 | 22.2% | 71.6% | 2.2% |

**Observations.** (1) Line splitting cuts multiline CER from 90.3% to 31.6% on these images. (2) With the original crop margin (0.5) it made single-line images worse: 17.1% to 22.2% CER, exact match 9.5% to 2.2%. The margin fix below reduces this, and the final paired test shows the remaining single-line difference is not clearly more than noise. (3) C and D are nearly identical here, as expected: there is no ruling in these images. (4) Failures of the splitting itself are visible in the log: a few multiline images were read as 1-2 lines (e.g. `multiline_1792`, `4031`, `5858`) and one single line was split into two (`single_369`, CER 1.5).

**Observation on the single-line regression (6 dataset images only, n too small to conclude).** In the per-image log, C/D often drop or replace the first letter (`Dia` to `ia`, `Kereta` to `Jereta`, `1157-01-01` to `17657-01-0`) where A reads it. Reproduced on the six images in `app/examples`: pipeline B (same cleanup, whole image) reads those first letters correctly, so the cause is the crop, not the cleanup. The dataset's own images put the ink about 0.15 line heights from the left edge, while the C/D crop originally left 0.5 (`SegmentationConfig.margin_x`). Corpus CER on those 6 lines by margin: 0.50 -> 12.0%, 0.35 -> 12.6%, 0.25 -> 10.9%, 0.15 -> 10.9%, 0.10 -> 11.4%, 0.05 -> 12.6% (A: 10.3%).

**Test of the margin hypothesis on all 208 images (Peter's machine, same model, A unchanged at 17.1% single / 90.3% multiline).** Corpus CER, pipeline D:

| Horizontal crop margin | Single-line | Multiline | All |
| --- | ---: | ---: | ---: |
| 0.50 (old default) | 22.2% | 31.6% | 28.8% |
| 0.25 | 20.0% | 29.6% | 26.7% |
| 0.15 | 20.8% | 30.3% | 27.4% |

(Exact match D: 1.4% at 0.50, 6.2% at 0.25, 5.8% at 0.15; WER 75.8%, 68.9%, 72.3%.) **Supported, but only in part:** a smaller margin helps both groups and recovers about 40% of the single-line gap to A (5.1 to 2.9 points). A gap of about 3 points remains, cause not yet identified. **Caveats:** the margin was chosen among three values on the same 208 images it is reported on, so the 0.25 figures are slightly optimistic; the images are synthetic; one run each. The default is now `margin_x = 0.25`; the "0.50" rows above are the earlier default, kept for traceability.

**Where the remaining single-line gap comes from (Peter's run, `--experiments A B C --margin-x 0.25`).** Single-line corpus CER: A 17.1%, B 18.9%, C 20.0%. So about 1.8 points come with the cleanup (A to B) and 1.1 with the cropping (B to C). Multiline: A 90.3%, B 90.1%, C 29.6%. **Inference, not tested:** these dataset images are already clean and evenly lit, so the cleanup has nothing to fix and can only alter strokes; its value is meant for real photos, which are unmeasured. **Paired test (Peter's run, `summarize --results results_b`, bootstrap over images, default margin 0.25).** Change in corpus CER, second minus first, 95% CI:

| From | To | Group | n | Change | 95% CI | Within noise? |
| --- | --- | --- | ---: | ---: | --- | --- |
| A | B | single | 137 | +1.7 pt | +0.4 to +3.1 | no |
| A | B | multiline | 71 | -0.2 pt | -0.6 to +0.2 | yes |
| A | B | all | 208 | +0.4 pt | -0.1 to +0.9 | yes |
| A | C | single | 137 | +2.8 pt | -0.0 to +6.5 | yes |
| A | C | multiline | 71 | -60.6 pt | -65.5 to -55.1 | no |
| A | C | all | 208 | -41.5 pt | -46.8 to -35.5 | no |
| B | C | single | 137 | +1.1 pt | -1.8 to +4.6 | yes |
| B | C | multiline | 71 | -60.4 pt | -65.4 to -54.8 | no |
| B | C | all | 208 | -41.9 pt | -47.2 to -35.9 | no |

**Conclusions the data supports (synthetic dataset images, one run, one model).** (1) Line splitting is a large, clear gain on multiline images (about -60 points CER). (2) On single-line images the cleanup alone costs a small but detectable +1.7 points; the full pipeline C is +2.8 points with an interval that includes 0, so it is not shown to be worse or better than A there. (3) The single-line interval is wide because a few images fail badly (for example `single_369`, wrongly split into two lines). **Not supported by any data yet:** any claim about real photographs, where cleanup and ruling suppression are meant to help. Further tuning on these synthetic images is not worthwhile; the next evidence has to come from real photos.

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
| 10. Crops | each line is cropped with a margin; a line needing more than ~1.7x shrink (wider than ~13x its height) is cut at word gaps into chunks that fit the 128x1024 canvas, a slightly-too-wide line is shrunk to fit like in pipeline A (a single over-long word is shrunk and flagged) | `margin`, `margin_x` |
| 11. Recognition | the unchanged CNN-BiLSTM-CTC model on every crop (batched; GPU if the model is on GPU, otherwise CPU) | - |
| 12. Join | chunks joined with spaces, lines joined with newlines in reading order | - |

Adaptive/binary thresholding is deliberately **not** applied to the recognizer input: the
mask is used only to find lines, the recognizer sees grey-level crops.

## Paragraphs

After the lines are found, `assign_paragraphs` groups them into paragraphs from the **page layout only** (no recognition): a new paragraph starts after a skipped line (line-to-line distance more than 1.6 x the median distance) or at a first line indented more than 1.0 x the median line height to the right of the left margin. The result text keeps a line break between lines and a blank line between paragraphs; scoring ignores both (whitespace-normalised).

- **Checked (synthetic pages only, optimistic):** `src/evaluate_segmentation.py` has paragraph layouts (skipped line, indent only, both, none) under clean, ruled, shadow, noisy and skewed conditions: 30/30 pages correct for each layout and for both pipelines C and D. The thresholds were developed on the same generator.
- **False positives on dataset images (one image only so far):** the one multiline dataset sample I have here is not split into paragraphs. To check all 71 multiline test images, re-run `evaluate_pipeline.py` on `evaluation/synthetic_test` and `pipeline_on_test_set.py summarize`; it now prints how many images were given more than one paragraph (the dataset's multiline images have none).
- **Not checked on real handwriting.** Limits: needs at least 3 lines (gap rule) or 4 (indent rule); one-line paragraphs with normal spacing, centred titles, poems and multi-column pages are not recognised.

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
