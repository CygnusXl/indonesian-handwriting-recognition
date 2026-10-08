# Real-world photograph evaluation

This folder is where YOU put photographs of handwritten notebook pages (and
single lines) and run the pipeline comparison on them. The repository does not
contain any real photographs and no results from them: **every number about
real handwriting still has to be produced by running the command below.**

Raw photos, hand crops and outputs are git-ignored (they may show people's
handwriting); only this README and the template are tracked.

## 1. Add your images

```
evaluation/real_world/
    page1.jpg                 any .png .jpg .jpeg .webp .bmp .tif
    page1.txt                 (optional) what page1.jpg really says
    fox.jpg
    ground_truth.csv          (optional) image_file,reference_text[,any other columns]
    manual_lines/             (optional, for experiment E)
        page1/01.png 02.png   one hand-cropped image per handwritten line, top to bottom
```

Ground truth is optional per image. Images without it are still run and
recorded, but get **no** CER/WER (the cells are empty, not zero). See
`ground_truth_template.csv`. Write the reference exactly as written on the
paper (including crossed-out words only if you want them counted). Line breaks
may be real line breaks or the two characters `\n`; for scoring they count as
spaces.

## 2. Run

```bash
python src/evaluate_pipeline.py                         # GPU if available, else CPU
python src/evaluate_pipeline.py --device cpu
python src/evaluate_pipeline.py --experiments A C D     # a subset
python src/evaluate_pipeline.py --images some/other/folder --model models/final_model.pt
```

Needs `models/final_model.pt` (`python src/export_final_model.py --model baseline`),
or pass `--model`.

## 3. Experiments

| Key | What happens to the photo |
|---|---|
| A | Original pipeline: grayscale, scale the WHOLE image to 128 px high, one model call. |
| B | Improved preprocessing (orientation, perspective, deskew, illumination flattening, denoise, contrast), then the whole page is still ONE model input. |
| C | B + line detection: every detected line is cropped and recognised separately (long lines are cut at word gaps to fit the 128x1024 input). |
| D | C + notebook-ruling suppression (the default in the app). |
| E | Diagnostic: your hand-cropped lines through the original pipeline (A), joined with spaces. Isolates the recognizer from automatic preprocessing and segmentation. |

## 4. Outputs (`evaluation/real_world/results/`)

- `per_image.csv`: one row per image and experiment: filename, experiment,
  preprocessing description, input kind (single line / multiple lines / ...),
  detected number of lines, number of model inputs, predicted text, heuristic
  confidence, preprocessing / detection / inference / total seconds, and, only
  with ground truth: reference, CER, WER, case-insensitive CER, exact match,
  whether the detected line count equals the true one, characters the model can
  never output.
- `summary.csv`, `summary.json`, `report.md`: per experiment corpus CER/WER
  (primary: total errors / total reference length), macro CER (secondary), exact
  match, line-count accuracy, mean time. `report.md` states the number of images
  behind every figure and warns when fewer than 20 have ground truth.
- `debug/<image>/<experiment>/`: processed page, line overlay, every crop and the
  exact 128x1024 model input. **Look at these first** when a result is bad.

## 5. How to read the results

Do not read these as conclusions; they are the questions the experiments answer.

| Observation | Points to |
|---|---|
| B much better than A | whole-page preprocessing (lighting/contrast/orientation) |
| C much better than B | line segmentation / giving the model line-sized input |
| D better than C | ruling suppression helps (check `debug` crops for damaged letters) |
| E much better than C/D | automatic preprocessing or segmentation is still the bottleneck: inspect `debug` crops |
| E also poor, and the debug crops look clean | the recognizer does not generalise to this handwriting: training-data mismatch (synthetic training data; English photos vs an Indonesian-trained model). Only THEN is more data or a different model justified. |

Because the model was trained on synthetic Indonesian handwriting, English text
photos ("The quick brown fox...") are also a vocabulary/language mismatch:
character shapes are shared, but there is no language model and the training
text was Indonesian.

## 6. Caveats

- Heuristic confidence = mean probability of the emitted characters. It is not
  calibrated and can be high on wrong text.
- With a handful of images the percentages are descriptive only.
- Experiment E needs you to crop lines by hand; that is a manual step.
- Timing includes preprocessing; the first model call is a warm-up and is not
  billed to the first image.
