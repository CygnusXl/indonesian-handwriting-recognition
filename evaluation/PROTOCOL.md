# Real-world evaluation protocol

**Purpose.** The model was trained and tested only on synthetic handwriting
(OpenHand-Synth). This protocol measures what actually happens with real
handwriting and real photos, and how usable the application is. It produces
the evidence for report section 8 (Real-World Evaluation).

**Rules.** Every number in the report must come from files in
`evaluation/` created by following this protocol. Nothing is simulated.
If fewer participants than planned take part, report the true number.

## Part A - Controlled photo test set (measures recognition under real conditions)

**Who.** At least 5 writers (team members + friends/classmates). State in
the report who they were (e.g. "3 team members, 4 classmates"); team
members are NOT independent users and the report must say so.

**What each writer produces** (see `prompts.md`):
- 8 single-line prompts + 2 multiline prompts, handwritten with their own pen
  on their own paper (do not ask for "neat" writing; record what they write).

**Capture conditions** (record one label per image):
- `scan` - flatbed scanner or scanner app;
- `good photo` - phone photo, even daylight, paper flat;
- `dim or shadowed photo` - indoor/dim light or hand shadow;
- `blurry photo` - slight motion blur / poor focus (do this deliberately for
  a subset, e.g. 2 images per writer).

**Handwriting label** (`neat`, `average`, `messy`): assigned by the writer
BEFORE seeing any model output. Subjective; say so in the report.

**Cropping.** Single-line prompts: crop tightly to the line (this is the
supported use). Multiline prompts: photograph the whole block, uncropped,
because that is exactly the weakness under test.

**Files.** Put images in `evaluation/photos/` named `P01_s01.jpg`,
`P01_m01.jpg`... and list every image in `evaluation/ground_truth.csv`
(template: `ground_truth_template.csv`; line breaks written as `\n`).

**Run.**

```bash
python src/evaluate_realworld.py
```

Outputs `evaluation/results/realworld_per_image.csv` and
`realworld_summary.md` with corpus CER / WER / exact match by line type,
handwriting label, capture condition and participant, with and without the
auto-contrast aid.

**Report.** Show the table with `n` next to every percentage; compare to the
synthetic-test numbers (same metrics); show 3-5 failure examples (image +
reference + prediction); note characters outside the vocabulary
(`out_of_vocabulary_chars` column).

## Part B - Usability session (measures practical usefulness)

**Who.** At least 5 participants who write notes by hand (students). Each
session is 10-15 minutes. Collect consent to record answers; do not store
images unless the participant agrees. No names in the data: use `P01`...

**Tasks** (participant uses the Streamlit app on a prepared laptop):
1. Open the app and upload a prepared single-line handwriting photo.
2. Press *Recognize text* and read the result aloud.
3. Upload one of their OWN handwriting photos and recognise it.
4. Use *Practical evaluation* to enter the true text and save the result.

Note whether each task was completed without help (`tasks_completed`).
Observe and write down (verbatim) where they hesitated or made mistakes.

**Questionnaire.** After the tasks give `questionnaire.md`; enter answers in
`evaluation/usability_responses.csv` (template provided).

**Run.**

```bash
python src/summarize_usability.py
```

## Part C - If participants are not available

Do Part A with team members only and say so explicitly; skip Part B or run it
with whoever is available. State that results are not generalisable. Never
substitute invented responses.

## What the report must say about this evaluation

- number of writers/participants and who they were;
- that the sample is small and descriptive;
- that the training data were synthetic, so the gap between synthetic and
  real results (if any) is a finding, not an error in the evaluation;
- known biases: team members know the system; writers chose their own pen and
  paper; labels are self-assigned.

## Data hygiene

- `evaluation/photos/` may contain personal handwriting: keep it out of the
  public repository unless every writer agreed (it is listed in `.gitignore`).
- `evaluation/results/*.csv` contain only text and metrics and may be
  committed as the evidence trail.
