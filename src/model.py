import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 70          # 69 characters + CTC blank
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 2


# ============================================================
# CNN + BiLSTM Model
# ============================================================

class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM handwriting recognition model.

    Input:
        [batch, 1, 128, 1024]

    Output:
        [time_steps, batch, num_classes]

    The CNN reduces image height while preserving more
    horizontal information for the CTC sequence.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
    ):
        super().__init__()

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------
        #
        # Pooling:
        #
        # 128 x 1024
        #    ↓ (2,2)
        #  64 x 512
        #
        #    ↓ (2,2)
        #  32 x 256
        #
        #    ↓ (2,2)
        #  16 x 128
        #
        #    ↓ (2,1)
        #   8 x 128
        #
        # Width is preserved in the final pooling layer.
        #

        self.cnn = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Block 2
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Block 3
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Block 4
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Reduce height but preserve width
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # ----------------------------------------------------
        # CNN output
        # ----------------------------------------------------
        #
        # Input:
        #   [batch, 1, 128, 1024]
        #
        # Output:
        #   [batch, 256, 8, 256]
        #
        # We treat the width (256) as the time dimension.
        #

        cnn_height = 8
        cnn_channels = 256

        lstm_input_size = cnn_channels * cnn_height

        # ----------------------------------------------------
        # Bidirectional LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=False,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )

        # ----------------------------------------------------
        # Classification layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            lstm_hidden_size * 2,
            num_classes
        )

    def forward(self, x):
        """
        Forward pass.

        Input:
            [batch, 1, 128, 1024]

        Output:
            [time_steps, batch, num_classes]
        """

        # CNN
        x = self.cnn(x)

        # Expected:
        # [batch, 256, 8, 256]

        batch_size = x.size(0)
        channels = x.size(1)
        height = x.size(2)
        width = x.size(3)

        # Move width into the sequence dimension:
        #
        # [batch, channels, height, width]
        #       ↓
        # [batch, width, channels, height]

        x = x.permute(0, 3, 1, 2)

        # Combine CNN channels and height:
        #
        # [batch, width, channels * height]
        #
        x = x.contiguous().view(
            batch_size,
            width,
            channels * height
        )

        # LSTM expects:
        #
        # [sequence_length, batch, features]

        x = x.permute(1, 0, 2)

        # BiLSTM
        x, _ = self.lstm(x)

        # Classifier
        #
        # [time, batch, 512]
        #       ↓
        # [time, batch, 70]

        x = self.classifier(x)

        return x


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CNN-BiLSTM MODEL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CNNBiLSTM()

    model = model.to(device)

    # Count parameters
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Number of parameters: {parameter_count:,}")

    # --------------------------------------------------------
    # Fake input
    # --------------------------------------------------------

    batch_size = 4

    test_input = torch.randn(
        batch_size,
        1,
        128,
        1024,
        device=device
    )

    print(f"\nInput shape: {tuple(test_input.shape)}")

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(test_input)

    print(f"Output shape: {tuple(output.shape)}")

    # Expected:
    #
    # 256 time steps
    # 4 batch
    # 70 classes

    expected_shape = (256, 4, 70)

    print(f"Expected shape: {expected_shape}")

    if tuple(output.shape) == expected_shape:
        print("\nShape test: PASS")
    else:
        print("\nShape test: FAIL")

    # --------------------------------------------------------
    # Probability test
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=2
    )

    probability_sum = probabilities[0, 0].sum().item()

    print(
        f"\nProbability sum for first timestep: "
        f"{probability_sum:.6f}"
    )

    if abs(probability_sum - 1.0) < 1e-5:
        print("Probability test: PASS")
    else:
        print("Probability test: FAIL")

    # --------------------------------------------------------
    # CNN shape test
    # --------------------------------------------------------

    with torch.no_grad():

        cnn_output = model.cnn(test_input)

    print(
        f"\nCNN output shape: "
        f"{tuple(cnn_output.shape)}"
    )

    expected_cnn_shape = (4, 256, 8, 256)

    print(
        f"Expected CNN shape: "
        f"{expected_cnn_shape}"
    )

    if tuple(cnn_output.shape) == expected_cnn_shape:
        print("CNN shape test: PASS")
    else:
        print("CNN shape test: FAIL")

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL TEST COMPLETE")
    print("=" * 70)