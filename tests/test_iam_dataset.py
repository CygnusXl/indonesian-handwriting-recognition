"""IAM loader logic on FAKE in-memory data (the real IAM is not used)."""

import random

import pytest
from PIL import Image, ImageDraw

import htr_common as hc
import iam_dataset as iam


def make_pairs(texts):
    pairs = []
    for text in texts:
        image = Image.new("L", (300, 50), 255)
        ImageDraw.Draw(image).text((10, 15), text, fill=0)
        pairs.append((image, text))
    return pairs


def test_detect_columns_finds_common_names():
    assert iam.detect_columns(["image", "text"]) == ("image", "text")
    assert iam.detect_columns(["Img", "transcription", "writer"]) == (
        "Img", "transcription",
    )


def test_detect_columns_error_lists_real_columns():
    with pytest.raises(iam.IamUnavailableError) as error:
        iam.detect_columns(["foo", "bar"])

    assert "foo" in str(error.value) and "bar" in str(error.value)


def test_required_timesteps_counts_repeats():
    assert iam.required_timesteps("abc") == 3
    assert iam.required_timesteps("aab") == 4
    assert iam.required_timesteps("") == 0


def test_normalize_text_collapses_whitespace_only():
    assert iam.normalize_text("  a   b\t c ") == "a b c"
    assert iam.normalize_text("It's \"x\"") == "It's \"x\""


def test_vocabulary_is_sorted_and_has_blank_last():
    vocabulary = iam.build_vocabulary(["ba", "ca "])

    assert list(vocabulary.char_to_id) == [" ", "a", "b", "c"]
    assert vocabulary.blank_id == 4
    assert vocabulary.num_classes == 5


def test_dataset_drops_and_counts_bad_lines():
    vocabulary = iam.build_vocabulary(["abc def"])
    pairs = make_pairs(["abc", "   ", "abcabcabcabc", "abz"])
    dataset = iam.IamLineDataset.from_pairs(
        pairs, vocabulary, image_width=128, ctc_timesteps=8
    )

    assert len(dataset) == 1
    assert dataset.dropped == {
        "empty": 1, "too_long_for_ctc": 1, "unknown_char": 1,
    }


def test_item_shapes_and_label_round_trip():
    vocabulary = iam.build_vocabulary(["hello world"])
    dataset = iam.IamLineDataset.from_pairs(
        make_pairs(["hello world"]), vocabulary,
        image_width=512, ctc_timesteps=128,
    )
    item = dataset[0]

    assert item["image"].shape == (1, 128, 512)
    assert vocabulary.decode(item["label"].tolist()) == "hello world"
    assert item["text"] == "hello world"


def test_augmented_items_differ_but_labels_do_not():
    vocabulary = iam.build_vocabulary(["hello"])
    dataset = iam.IamLineDataset.from_pairs(
        make_pairs(["hello"]), vocabulary, image_width=512,
        ctc_timesteps=128, augment=True, rng=random.Random(1),
    )

    a, b = dataset[0], dataset[0]

    assert not (a["image"] == b["image"]).all()
    assert a["label"].tolist() == b["label"].tolist()


def test_pad_fraction_changes_image_not_label():
    vocabulary = iam.build_vocabulary(["hello"])
    pairs = make_pairs(["hello"])
    plain = iam.IamLineDataset.from_pairs(
        pairs, vocabulary, image_width=512, ctc_timesteps=128
    )[0]
    padded = iam.IamLineDataset.from_pairs(
        pairs, vocabulary, image_width=512, ctc_timesteps=128,
        pad_fraction=0.25,
    )[0]

    assert not (plain["image"] == padded["image"]).all()
    assert plain["label"].tolist() == padded["label"].tolist()


class FakeHF:
    """Mimics the parts of a Hugging Face dataset the loader touches."""

    def __init__(self, columns, pairs):
        self.column_names = columns
        self._pairs = pairs

    def __getitem__(self, key):
        if isinstance(key, str):
            return [t for _, t in self._pairs] if key == "text" else []
        return {"image": self._pairs[key][0], "text": self._pairs[key][1]}


def test_from_huggingface_is_lazy_and_reads_images():
    vocabulary = iam.build_vocabulary(["hi there"])
    hf = FakeHF(["image", "text"], make_pairs(["hi there", "hi"]))
    dataset = iam.IamLineDataset.from_huggingface(
        hf, vocabulary, image_width=256, ctc_timesteps=64
    )

    assert len(dataset) == 2
    assert dataset[1]["text"] == "hi"


def test_load_split_reports_actionable_error(monkeypatch):
    import datasets

    def fail(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(datasets, "load_dataset", fail)

    with pytest.raises(iam.IamUnavailableError) as error:
        iam.load_split("train", "Nope/Nope")

    message = str(error.value)
    assert "no network" in message and "fki.tic.heia-fr.ch" in message


def test_vocabulary_works_with_checkpoint_vocabulary_type():
    vocabulary = iam.build_vocabulary(["ab"])
    assert isinstance(vocabulary, hc.CheckpointVocabulary)


def test_check_iam_reports_and_saves_samples(tmp_path, monkeypatch, capsys):
    import check_iam

    pairs = make_pairs(["hello world", "abc", "hello again", "xyz",
                        "hello", "fine"])

    class Split(FakeHF):
        def __len__(self):
            return len(self._pairs)

    monkeypatch.setattr(
        iam, "load_split", lambda split, name=None: Split(["image", "text"], pairs)
    )

    check_iam.main(["--output-dir", str(tmp_path)])

    out = capsys.readouterr().out
    assert "[train] 6 lines" in out and "[test]" in out
    assert (tmp_path / "train_0.png").exists()
