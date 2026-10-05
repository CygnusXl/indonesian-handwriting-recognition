import torch
import torch.nn as nn


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 70

LSTM_HIDDEN_SIZE = 256

LSTM_LAYERS = 2

IMAGE_HEIGHT = 192

IMAGE_WIDTH = 1024


# ============================================================
# CNN + BiLSTM
# ============================================================

class CNNBiLSTMExperiment2(nn.Module):
    """
    CNN-BiLSTM model for Experiment 2.

    Experiment 2 changes the input height from:

        128 -> 192

    while keeping the width at:

        1024

    This preserves more vertical information in the CNN
    feature representation.

    Input:

        [batch, 1, 192, 1024]

    CNN output:

        [batch, 256, 12, 256]

    Sequence:

        256 CTC time steps
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        image_height=IMAGE_HEIGHT,
    ):
        super().__init__()

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

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
            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            # Block 2
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),

            # Block 3
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),

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
            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),
        )

        # ----------------------------------------------------
        # Calculate CNN output dimensions
        # ----------------------------------------------------

        # Four vertical pooling operations divide height by 16:
        #
        # 192 -> 96 -> 48 -> 24 -> 12
        #
        self.cnn_height = (
            image_height // 16
        )

        self.cnn_channels = 256

        # Each horizontal position contains all CNN
        # channels and vertical features.
        self.lstm_input_size = (
            self.cnn_channels
            * self.cnn_height
        )

        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=False,
            bidirectional=True,
            dropout=(
                0.2
                if lstm_layers > 1
                else 0.0
            ),
        )

        # ----------------------------------------------------
        # Classification layer
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            lstm_hidden_size * 2,
            num_classes
        )

    def forward(self, x):

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        x = self.cnn(x)

        # Shape:
        #
        # [batch, channels, height, width]
        #

        batch_size = x.size(0)

        channels = x.size(1)

        height = x.size(2)

        width = x.size(3)

        # ----------------------------------------------------
        # Move width into sequence dimension
        # ----------------------------------------------------

        x = x.permute(
            0,
            3,
            1,
            2
        )

        # Shape:
        #
        # [batch, width, channels, height]
        #

        x = x.contiguous().view(
            batch_size,
            width,
            channels * height
        )

        # ----------------------------------------------------
        # LSTM expects:
        #
        # [sequence, batch, features]
        # ----------------------------------------------------

        x = x.permute(
            1,
            0,
            2
        )

        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        x, _ = self.lstm(
            x
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        x = self.classifier(
            x
        )

        return x


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "EXPERIMENT 2 MODEL TEST"
    )
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CNNBiLSTMExperiment2()

    model = model.to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Test input
    # --------------------------------------------------------

    test_input = torch.randn(
        2,
        1,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        device=device
    )

    print()
    print(
        f"Input shape: "
        f"{tuple(test_input.shape)}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            test_input
        )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    # Expected:
    #
    # 256 time steps
    # 2 samples
    # 70 classes
    #

    expected_shape = (
        256,
        2,
        70
    )

    print(
        f"Expected shape: "
        f"{expected_shape}"
    )

    if tuple(output.shape) == expected_shape:

        print(
            "Shape test: PASS"
        )

    else:

        print(
            "Shape test: FAIL"
        )

    # --------------------------------------------------------
    # CNN test
    # --------------------------------------------------------

    with torch.no_grad():

        cnn_output = model.cnn(
            test_input
        )

    print()
    print(
        f"CNN output shape: "
        f"{tuple(cnn_output.shape)}"
    )

    expected_cnn_shape = (
        2,
        256,
        12,
        256
    )

    print(
        f"Expected CNN shape: "
        f"{expected_cnn_shape}"
    )

    if (
        tuple(cnn_output.shape)
        == expected_cnn_shape
    ):

        print(
            "CNN shape test: PASS"
        )

    else:

        print(
            "CNN shape test: FAIL"
        )

    print()
    print("=" * 70)
    print(
        "EXPERIMENT 2 MODEL TEST COMPLETE"
    )
    print("=" * 70)