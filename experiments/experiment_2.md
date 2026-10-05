# Experiment 2 — Increased Vertical Resolution

## Research question

> Does increasing vertical image resolution preserve more information about multiline handwriting and improve recognition?

## Baseline configuration

- Image: 128 x 1024
- CTC timesteps: 256
- CNN-BiLSTM

## Experiment configuration

- Image: 192 x 1024
- CTC timesteps: 256
- CNN-BiLSTM architecture adapted to the increased vertical feature dimension
- Physical batch size: 2
- Gradient accumulation: 2
- Effective batch size: 4
- Learning rate: 0.0003
- Weight decay: 0.0001
- Epochs: 10

## Training result

Best validation CER: 82.06% at epoch 10.

Training loss decreased from 3.6999 at epoch 1 to 2.2844 at epoch 10. Validation loss decreased from 3.1318 to 2.1372. Validation CER decreased from 97.69% to 82.06%.

## Common test-set result

All models were evaluated on the same 208 CTC-feasible Indonesian test samples.

| Metric | Baseline | Experiment 2 | Change |
| --- | ---: | ---: | ---: |
| Test loss | 1.4498 | 2.1445 | +0.6947 |
| Macro-CER | 41.28% | 67.84% | +26.56 pp |
| WER | 73.82% | 99.00% | +25.17 pp |
| Exact match | 6.25% | 0.00% | -6.25 pp |

## Error analysis

Experiment 2 produced severe prediction collapse on 71/208 samples (34.13%).

Mean sample CER by condition:

- Single-line: 54.75%
- Multiline: 93.10%
- Short: 57.49%
- Medium: 55.43%
- Long: 84.62%
- Very long: 93.87%

Noise showed a comparatively small difference: 67.09% mean sample CER without augmented noise versus 68.63% with augmented noise.

The worst examples were dominated by multiline and long references, with predictions collapsing to only a few characters.

## Interpretation

Increasing vertical resolution did not solve the multiline problem. It also substantially degraded shorter and single-line inputs relative to the baseline. The experiment therefore does not support increased vertical resolution as an effective improvement for the current CNN-BiLSTM configuration.

The results motivate Experiment 3, which changes the training sample distribution rather than changing image dimensions.