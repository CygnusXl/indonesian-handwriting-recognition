# Training on English handwriting (IAM)

**Status: code written and unit-tested on CPU with fake data. Never run on
the real IAM data. There is no result yet.** Everything below that says
"expected" or "hypothesis" is untested.

## Why this change

The deployed model was trained only on synthetic Indonesian handwriting.
On one real cursive photo (a friend's "Proklamasi", n = 1) it could not
read the word even after two pipeline bugs were fixed and the line was
cropped by hand the way the training images look
(`DOCUMENT_PIPELINE.md`, first real photo case study).

**Hypothesis:** the training data, not the architecture, is the main
limit. IAM contains real pen-on-paper writing from 657 writers. This is
unproven until the new model is measured.

**What this will not do.** IAM is one collection method (forms, one kind
of paper, scanner). "Any photo, any handwriting" is not a claim this
project can make. Real photos, other writing styles and other languages
can still fail. Only measured numbers on held-out data count.

## Project scope change: tell your lecturer

The project title and README say *Indonesian* handwriting. A model trained
on IAM reads **English**. Indonesian text uses the same Latin letters, so
it may be partly readable, but that is untested and IAM has no Indonesian
vocabulary or language statistics. Before you submit, check with your
lecturer whether the title and research question still match, and say so
in the report. The old Indonesian model and all its results stay in the
repository for comparison.

## Licence and citation

The official IAM page says the database "may be used for non-commercial
research purpose only", and registration is required for the official
download. The Hugging Face mirror's own terms are **unverified**; read
them on its page. A public Streamlit demo that serves a model trained on
IAM is a use you should check against those terms (the weights are
derived from the data). Cite:

> U. Marti and H. Bunke. The IAM-database: an English sentence database
> for offline handwriting recognition. *International Journal on Document
> Analysis and Recognition*, 5:39-46, 2002.

## What was added

| File | Purpose |
|---|---|
| `src/check_iam.py` | Loads the data, prints columns, sizes, vocabulary, saves 5 sample images. Run first. |
| `src/iam_dataset.py` | Loader (`Teklia/IAM-line` from Hugging Face; **unverified** columns/splits, auto-detected). |
| `src/photo_augment.py` | Photo-style augmentation (margins, slant, rotation, grey paper, shading, ruling, blur, noise, JPEG). |
| `src/train_iam.py` | Training (same model, AdamW + CTC, cosine learning-rate decay, input 128 x 1536). |
| `src/evaluate_iam.py` | IAM test CER / WER / exact match with 95 % bootstrap intervals, paired differences between checkpoints. |
| `src/export_final_model.py` | Now accepts `--checkpoint`, `--label`, `--iam-summary`; the app wording follows the recorded `training_data`. |

Design choices (all untuned, none shown to help yet): width 1536 px / 384
CTC steps because IAM lines are long; trained from scratch with a new
character set built from the training lines; learning rate 3e-4 with a
cosine decay; the best epoch is chosen on validation CER; margins are
augmented because a margin sensitivity was measured earlier.

## Commands (your PC, in `indonesian-handwriting-recognition/`)

```bash
git pull
source .venv/bin/activate

# 1. Can the data be loaded? (prints a report; send it to Claude if it fails)
python3 src/check_iam.py

# 2. Ten-second check that training runs on your GPU
python3 src/train_iam.py --limit 200 --epochs 1 --workers 0 \
    --output checkpoints/iam_smoke.pt --history results/iam_smoke_history.csv

# 3. Full run. Watch the first epoch's time, then multiply by --epochs.
python3 src/train_iam.py

# 4. Evaluate: new model vs the old Indonesian model, on IAM test lines
python3 src/evaluate_iam.py \
    --checkpoint checkpoints/iam_best.pt \
    --checkpoint models/final_model.pt
```

If step 3 runs out of GPU memory, add `--batch-size 8`. The 8 GB limit is
why; batch 16 at 128 x 1536 is untested on your card.

Training can be interrupted; the best checkpoint so far is on disk and
`results/iam_history.csv` is rewritten every epoch.

## Reading the results honestly

* `results/iam_eval/summary.md` has CER and WER with intervals for two
  border settings: `0` (raw IAM lines) and `0.25` (the white margin the
  page pipeline's crop adds in the app). Report both.
* The old model's CER on IAM is a real number: letters it was never
  trained on always count as errors.
* IAM test lines are clean scans. A good IAM score does **not** mean good
  phone-photo accuracy. The decisive test is your own photos with ground
  truth (`evaluation/PROTOCOL.md`, 20-30 photos, with people's
  permission):

```bash
python3 src/evaluate_pipeline.py --images evaluation/photos \
    --ground-truth evaluation/photos/ground_truth.csv \
    --model checkpoints/iam_best.pt --experiments A C D --output results_iam_photos
```

  Run it again with `--model models/final_model.pt` and compare. Do not
  claim an improvement from one photo.

## Putting the new model on the Streamlit app

Only after steps 3-4 show it is better on your own photos (or you decide
to ship it anyway and say so):

```bash
python3 src/export_final_model.py --checkpoint checkpoints/iam_best.pt \
    --label "IAM English model" --iam-summary results/iam_eval/summary.json
git checkout -b iam-model
git add models/final_model.pt models/final_model.json
git commit -m "Deploy IAM-trained model"
git push -u origin iam-model      # open a PR; merging to main updates the app
```

The page text (training-data notice, model card, "what to keep in mind")
changes automatically from the `training_data` field in the exported file.
The example pictures in the app are still the Indonesian synthetic test
images.

## Known unknowns

* Whether `Teklia/IAM-line` loads without a login, and its real column
  and split names (`check_iam.py` answers this).
* Training time, GPU memory at batch 16, and whether 40 epochs is right.
* Every accuracy figure, on IAM and on photos.
* Whether augmentation helps at all (to find out, train once with
  `--no-augment --output checkpoints/iam_noaug.pt --history
  results/iam_noaug_history.csv` and compare with `evaluate_iam.py`).
