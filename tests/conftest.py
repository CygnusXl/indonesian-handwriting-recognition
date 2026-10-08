import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "app"))


import types  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def random_loaded():
    """
    A RANDOMLY INITIALISED model with the real architecture and vocabulary
    size. For plumbing tests only: its predictions are meaningless.
    """

    import htr_common as hc

    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\\")[:69]
    vocabulary = hc.CheckpointVocabulary({c: i for i, c in enumerate(pool)})
    model = hc.build_model("cnn_bilstm", 70).eval()

    return types.SimpleNamespace(
        model=model, vocabulary=vocabulary,
        config={"image_height": 128, "image_width": 1024},
        sha256="0" * 64,
    )
