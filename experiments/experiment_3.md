# Experiment 3 — Multiline-Balanced Training

## Research question

Does increasing training exposure to multiline handwriting improve recognition without changing the baseline architecture or image representation?

## Hypothesis

The baseline error analysis identified multiline and very long samples as the dominant failure groups. Experiment 3 therefore changes the training sample distribution instead of changing the model architecture.

The training loader targets approximately 50% single-line and 50% multiline exposure per epoch.

## Controlled variables

The following remain the same as the baseline:

- CNN-BiLSTM architecture
- Input size: 128 x 1024
- CTC sequence length: 256
- Batch size: 4
- Learning rate: 0.0003
- Weight decay: 0.0001
- Epochs: 10
- CTC loss
- Validation split
- Test split

## Experimental variable

Only the training sample distribution changes.

A weighted random sampler samples with replacement so that multiline samples receive more exposure when they are under-represented in the natural training distribution.

The validation and test sets are not rebalanced.

## Evaluation

The experiment should answer:

1. Does balanced multiline exposure improve overall recognition?
2. Does it reduce the performance gap between single-line and multiline samples?
3. Is there a trade-off where multiline performance improves but ordinary single-line performance decreases?

A result does not need to beat the baseline to be scientifically useful; a measurable trade-off is also an experimental finding.

## Reproducibility

- Random seed: 42
- Same architecture and image representation as baseline
- Same core training hyperparameters as baseline
- Same CTC filtering
- Best checkpoint selected by validation CER

## Run

From the repository root:

\`\`\`bash
python src/train_experiment3.py
\`\`\`

Expected generated files:

- \`checkpoints/experiment3_best.pt\`
- \`results/experiment3_history.csv\`

Do not add final test metrics until the actual experiment has been run and evaluated.
