# Experiments

Experiments are controlled changes from the documented baseline.

## Baseline

- Input: 128 x 1024
- CNN-BiLSTM
- CTC timesteps: 256
- Effective batch size: 4
- Learning rate: 0.0003
- Weight decay: 0.0001
- Epochs: 10

Common-test macro-CER: **41.28%**

## Experiment 1 — Increased horizontal resolution

Question: Does increasing horizontal image resolution and CTC sequence capacity improve recognition of long and multiline handwriting?

Changed:
- Image width: 1024 -> 1536
- CTC timesteps: 256 -> 384

Result on the same 208 test samples: **47.18% macro-CER**, which is **+5.89 percentage points** versus baseline.

Conclusion: increased horizontal resolution did not improve recognition.

## Experiment 2 — Increased vertical resolution

Question: Does increasing vertical image resolution preserve more visual information from multiline handwriting?

Changed:
- Image height: 128 -> 192

Unchanged:
- Image width: 1024
- CTC timesteps: 256
- CNN-BiLSTM sequence design
- Learning rate, weight decay, effective batch size, and epochs

Training result: best validation CER **82.06%** at epoch 10.

Common-test result:
- Macro-CER: **67.84%**
- WER: **99.00%**
- Exact match: **0.00%**

Compared with baseline:
- CER: **+26.56 percentage points**
- WER: **+25.17 percentage points**
- Exact match: **-6.25 percentage points**

Error analysis found **93.10% mean sample CER on multiline samples** and **93.87% on very long samples**. Increasing vertical resolution therefore did not solve the dominant long/multiline failure and also degraded shorter and single-line samples.

See `experiment_2.md` for the full Experiment 2 record.

## Experiment 3 — Planned

Multiline-balanced training will keep the baseline architecture and image representation unchanged while increasing training exposure to multiline samples. Validation and test distributions will remain unchanged.

The purpose is to test whether training sample distribution contributes to the severe multiline-recognition weakness.