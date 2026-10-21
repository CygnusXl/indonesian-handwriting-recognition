"""IAM evaluation on fake data with a RANDOM model (plumbing + statistics)."""

import numpy as np
import torch
from PIL import Image, ImageDraw

import evaluate_iam as ev
import htr_common as hc
import iam_dataset as iam


def pairs_for(texts):
    out = []
    for text in texts:
        image = Image.new("L", (240, 40), 255)
        ImageDraw.Draw(image).text((8, 10), text, fill=0)
        out.append((image, text))
    return out


def test_line_statistics_and_rates_match_hand_computation():
    stats = ev.line_statistics(["abc", "ab cd"], ["abd", "ab cd"])
    cer, wer, exact = ev._rates(stats)

    assert cer == 1 / 8          # one substitution over 8 characters
    assert wer == 1 / 3          # "abd" vs "abc" over 3 reference words
    assert exact == 0.5


def test_bootstrap_interval_contains_point_and_is_repeatable():
    refs = ["hello world"] * 40
    hyps = ["hello world"] * 30 + ["hallo world"] * 10
    stats = ev.line_statistics(refs, hyps)
    point = ev._rates(stats)[0]

    a = ev.bootstrap_intervals(stats, draws=500, seed=1)
    b = ev.bootstrap_intervals(stats, draws=500, seed=1)

    assert np.array_equal(a, b)
    assert a[0][0] <= point <= a[1][0]


def test_paired_difference_sign_and_zero_for_identical():
    refs = ["hello world"] * 30
    good = ev.line_statistics(refs, refs)
    bad = ev.line_statistics(refs, ["hxllo world"] * 30)

    point, (lo, hi) = ev.paired_cer_difference(good, bad, draws=300)
    assert point > 0 and lo > 0

    point, (lo, hi) = ev.paired_cer_difference(good, good, draws=300)
    assert point == 0 and lo == 0 and hi == 0


def test_evaluate_checkpoint_end_to_end_with_random_model(tmp_path):
    texts = ["abc abc", "cab", "bca ab"]
    vocabulary = iam.build_vocabulary(texts)
    model = hc.build_model("cnn_bilstm", vocabulary.num_classes)

    path = str(tmp_path / "random.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary_char_to_id": vocabulary.char_to_id,
            "vocabulary_id_to_char": vocabulary.id_to_char,
            "num_classes": vocabulary.num_classes,
            "architecture": "cnn_bilstm",
            "image_height": 128,
            "image_width": 256,
            "ctc_timesteps": 64,
            "training_data": "iam",
        },
        path,
    )

    pairs = pairs_for(texts)
    results = ev.evaluate_checkpoint(
        path, pairs, [0.0, 0.25], "cpu", batch_size=2, draws=50
    )

    assert set(results) == {0.0, 0.25}
    assert len(results[0.0]["hypotheses"]) == 3
    assert results[0.0]["training_data"] == "iam"

    text = ev.summary_markdown(
        [(path, results), (path, results)], [0.0, 0.25], 3, 50
    )
    assert "Border 0.25" in text and "Paired CER difference" in text

    ev.write_line_csv(str(tmp_path / "x.csv"), results[0.0])
    assert (tmp_path / "x.csv").read_text().startswith("reference,")
