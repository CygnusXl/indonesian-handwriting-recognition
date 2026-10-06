"""
Regression / plumbing tests for src/htr_common.py.

Run from the repository root:

    python -m pytest tests -q

These tests need NO dataset, NO GPU and NO trained checkpoint. Tests that
build a model use RANDOMLY INITIALISED weights purely to check shapes and
code paths; they never produce or imply any performance result.
"""

import numpy as np
import pytest
import torch
from PIL import Image

import htr_common as hc

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1024


# ------------------------------------------------------------
# FROZEN reference copy of the preprocessing function exactly as it
# existed in src/dataset.py when every reported model was trained.
# DO NOT EDIT: it exists to prove the shared implementation is
# numerically identical, so reported results stay valid.
# ------------------------------------------------------------

def original_preprocess_image(
    image,
    target_height=IMAGE_HEIGHT,
    target_width=IMAGE_WIDTH
):
    """
    Convert an image into a fixed-size grayscale tensor.

    The aspect ratio of the original image is preserved.

    Output:

        [1, target_height, target_width]
    """

    # --------------------------------------------------------
    # Convert to grayscale.
    # --------------------------------------------------------

    image = image.convert("L")

    original_width, original_height = (
        image.size
    )

    # --------------------------------------------------------
    # Resize according to target height.
    # --------------------------------------------------------

    scale = (
        target_height
        / original_height
    )

    new_width = int(
        original_width * scale
    )

    # --------------------------------------------------------
    # Prevent width from exceeding target.
    # --------------------------------------------------------

    if new_width > target_width:

        new_width = target_width

        scale = (
            target_width
            / original_width
        )

        new_height = int(
            original_height * scale
        )

    else:

        new_height = target_height

    # --------------------------------------------------------
    # Resize.
    # --------------------------------------------------------

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Convert to NumPy.
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize [0,255] -> [0,1].
    image_array = (
        image_array / 255.0
    )

    # --------------------------------------------------------
    # Create white padded canvas.
    # --------------------------------------------------------

    padded = np.ones(
        (
            target_height,
            target_width
        ),
        dtype=np.float32
    )

    # Center vertically.
    y_offset = (
        target_height - new_height
    ) // 2

    padded[
        y_offset:y_offset + new_height,
        :new_width
    ] = image_array

    # --------------------------------------------------------
    # Convert to tensor.
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        padded
    )

    # [H,W] -> [1,H,W]
    tensor = tensor.unsqueeze(0)

    return tensor



def _random_image(width, height, seed):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.mark.parametrize(
    "size",
    [(300, 60), (1000, 100), (2400, 200), (800, 700), (1200, 1200), (64, 64)],
)
@pytest.mark.parametrize("target", [(128, 1024), (128, 1536), (192, 1024)])
def test_preprocess_matches_original(size, target):
    image = _random_image(size[0], size[1], seed=size[0] + size[1])
    height, width = target

    expected = original_preprocess_image(image, height, width)
    actual = hc.preprocess_image(image, height, width)

    assert actual.shape == (1, height, width)
    assert torch.equal(expected, actual)


def test_dataset_module_uses_shared_preprocessing():
    # dataset.py imports the `datasets` package; skip if unavailable.
    pytest.importorskip("datasets")
    import dataset

    assert dataset.preprocess_image is hc.preprocess_image


# ------------------------------------------------------------
# Edit distance and metrics
# ------------------------------------------------------------

def test_edit_distance_known_values():
    assert hc.edit_distance("kitten", "sitting") == 3
    assert hc.edit_distance("", "abc") == 3
    assert hc.edit_distance("abc", "") == 3
    assert hc.edit_distance("abc", "abc") == 0
    assert hc.edit_distance(["a", "b"], ["a", "c", "b"]) == 1


def test_corpus_vs_macro_cer_are_different_aggregations():
    refs = ["aaaaaaaaaa", "bb"]       # 10 and 2 characters
    hyps = ["aaaaaaaaaa", "xx"]       # 0 errors and 2 errors

    assert hc.corpus_cer(refs, hyps) == pytest.approx(2 / 12)
    assert hc.macro_cer(refs, hyps) == pytest.approx((0.0 + 1.0) / 2)
    assert hc.exact_match_rate(refs, hyps) == 0.5


def test_corpus_metrics_match_jiwer():
    jiwer = pytest.importorskip("jiwer")

    refs = ["di mana tempat pemberhentian busnya", "saya suka kopi", "ya"]
    hyps = ["di mana tempat pemberhentian busnya.", "saya kopi", "yaa"]

    assert hc.corpus_wer(refs, hyps) == pytest.approx(
        jiwer.wer(refs, hyps)
    )
    assert hc.corpus_cer(refs, hyps) == pytest.approx(
        jiwer.cer(refs, hyps)
    )


def test_wer_does_not_split_on_literal_backslash_n():
    # Documented behaviour: "\\n" is not whitespace.
    reference = "satu\\ndua tiga"
    assert len(reference.split()) == 2


# ------------------------------------------------------------
# Subgroup definitions
# ------------------------------------------------------------

def test_is_multiline_accepts_both_encodings():
    assert hc.is_multiline("a\nb")          # real newline
    assert hc.is_multiline("a\\nb")        # literal backslash + n
    assert not hc.is_multiline("a b")


def test_length_category_boundaries():
    assert hc.length_category(30) == "Short (1-30)"
    assert hc.length_category(31) == "Medium (31-60)"
    assert hc.length_category(60) == "Medium (31-60)"
    assert hc.length_category(61) == "Long (61-100)"
    assert hc.length_category(100) == "Long (61-100)"
    assert hc.length_category(101) == "Very long (101+)"


def test_is_collapsed_definition():
    assert hc.is_collapsed(100, 24)          # 24% of 100
    assert not hc.is_collapsed(100, 25)      # exactly 25% -> not collapsed
    assert not hc.is_collapsed(19, 0)        # reference too short
    assert hc.is_collapsed(20, 4)


# ------------------------------------------------------------
# CTC greedy decoding
# ------------------------------------------------------------

def _vocab():
    return hc.CheckpointVocabulary({"a": 0, "b": 1, " ": 2})


def _logits_for(sequence, num_classes):
    logits = torch.full((len(sequence), 1, num_classes), -10.0)
    for t, token in enumerate(sequence):
        logits[t, 0, token] = 10.0
    return logits


def test_greedy_decode_collapses_repeats_and_removes_blanks():
    vocab = _vocab()
    blank = vocab.blank_id
    a, b = 0, 1

    # a a blank a b b blank -> "aab"
    # (a a collapse to one 'a'; blank separates the next 'a')
    logits = _logits_for([a, a, blank, a, b, b, blank], vocab.num_classes)

    assert hc.greedy_decode(logits, vocab) == ["aab"]

    text, confidence = hc.greedy_decode_with_confidence(logits, vocab)[0]
    assert text == "aab"
    assert 0.99 < confidence <= 1.0


def test_greedy_decode_all_blank_is_empty_string_with_no_confidence():
    vocab = _vocab()
    logits = _logits_for([vocab.blank_id] * 5, vocab.num_classes)

    assert hc.greedy_decode(logits, vocab) == [""]
    assert hc.greedy_decode_with_confidence(logits, vocab)[0] == ("", None)


# ------------------------------------------------------------
# User-image preparation
# ------------------------------------------------------------

def test_prepare_user_image_flattens_transparency_to_white():
    transparent = Image.new("RGBA", (50, 20), (0, 0, 0, 0))
    prepared = hc.prepare_user_image(transparent)

    assert prepared.mode == "L"
    assert np.array(prepared).min() == 255


def test_prepare_user_image_auto_contrast_changes_low_contrast_image():
    rng = np.random.default_rng(0)
    low_contrast = rng.integers(100, 140, size=(40, 200), dtype=np.uint8)
    image = Image.fromarray(low_contrast, mode="L")

    plain = np.array(hc.prepare_user_image(image))
    stretched = np.array(hc.prepare_user_image(image, auto_contrast=True))

    assert plain.max() - plain.min() < stretched.max() - stretched.min()


# ------------------------------------------------------------
# Model construction and loading (random weights, plumbing only)
# ------------------------------------------------------------

def test_registry_models_have_documented_sizes_and_shapes():
    baseline = hc.build_model("cnn_bilstm", 70)
    exp2 = hc.build_model("cnn_bilstm_exp2", 70, image_height=192)

    assert hc.count_parameters(baseline) == 7_296_710
    assert hc.count_parameters(exp2) == 9_393_862

    baseline.eval()
    exp2.eval()

    with torch.no_grad():
        assert baseline(torch.zeros(1, 1, 128, 1024)).shape == (256, 1, 70)
        assert exp2(torch.zeros(1, 1, 192, 1024)).shape == (256, 1, 70)


def _write_random_checkpoint(path, deployment):
    model = hc.build_model("cnn_bilstm", 70)

    # Exactly 69 distinct characters (+ CTC blank = 70 classes).
    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\\")
    characters = pool[:69]
    char_to_id = {c: i for i, c in enumerate(characters)}
    id_to_char = {i: c for c, i in char_to_id.items()}

    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "validation_cer": 1.0,
        "vocabulary_char_to_id": char_to_id,
        "vocabulary_id_to_char": id_to_char,
        "num_classes": 70,
    }

    if deployment:
        checkpoint.update(
            architecture="cnn_bilstm",
            image_height=128,
            image_width=1024,
            ctc_timesteps=256,
        )

    torch.save(checkpoint, path)


@pytest.mark.parametrize("deployment", [True, False])
def test_load_model_roundtrip_and_recognize(tmp_path, deployment):
    # Legacy (training-style) checkpoints are recognised by file name.
    name = "deploy.pt" if deployment else "baseline_best.pt"
    path = tmp_path / name
    _write_random_checkpoint(path, deployment)

    model, vocab, config, _ = hc.load_model(str(path))

    assert config["image_height"] == 128
    assert config["image_width"] == 1024
    assert vocab.num_classes == 70

    image = Image.new("RGB", (900, 150), "white")
    text, confidence = hc.recognize(model, vocab, config, image)

    assert isinstance(text, str)
    assert confidence is None or 0.0 <= confidence <= 1.0


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hc.load_model(str(tmp_path / "nope.pt"))
