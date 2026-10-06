"""
Tests for src/analyze_model.py (pure functions only; no dataset/GPU).

The regression tests use the REAL per-sample predictions that were
committed in results/ by the original baseline / Experiment 2 error
analyses. They prove that the unified analysis code reproduces the
numbers that were previously reported from those predictions.
"""

import csv
import json
import os

import pytest

import analyze_model as am

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_legacy(name):
    path = os.path.join(ROOT, "results", f"{name}_error_analysis.csv")

    if not os.path.exists(path):
        pytest.skip(f"{path} not present")

    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    records = [
        am.make_record(
            {
                "dataset_index": int(r["dataset_index"]),
                "writer_id": r["writer_id"],
                "source": r["source"],
                "neatness": int(r["neatness"]),
                "line_width": int(r["line_width"]),
                "augmented_noise": r["augmented_noise"] == "True",
            },
            r["reference"],
            r["prediction"],
        )
        for r in rows
    ]

    return rows, records


def test_baseline_reproduces_reported_numbers():
    rows, records = _load_legacy("baseline")
    overall = am.summarize(records)["overall"]

    assert overall["samples"] == 208
    assert overall["corpus_cer"] * 100 == pytest.approx(68.79, abs=0.005)
    assert overall["corpus_wer"] * 100 == pytest.approx(85.08, abs=0.005)
    assert overall["macro_cer"] * 100 == pytest.approx(41.28, abs=0.005)
    assert overall["exact_match"] * 100 == pytest.approx(6.25, abs=0.005)
    assert overall["collapse_count"] == 60

    # Per-sample values and flags identical to the legacy file.
    for row, record in zip(rows, records):
        assert float(row["cer"]) == pytest.approx(record["cer"], abs=1e-9)
        assert (row["collapsed_prediction"] == "True") == record[
            "collapsed_prediction"
        ]
        assert row["length_category"] == record["length_category"]


def test_baseline_subgroups_reproduce_reported_numbers():
    _, records = _load_legacy("baseline")
    subgroups = am.summarize(records)["subgroups"]

    by_multiline = {
        row["group"]: row for row in subgroups["multiline"]["rows"]
    }
    assert by_multiline[False]["samples"] == 137
    assert by_multiline[True]["samples"] == 71
    assert by_multiline[False]["macro_cer"] * 100 == pytest.approx(
        17.08, abs=0.005
    )
    assert by_multiline[True]["macro_cer"] * 100 == pytest.approx(
        87.99, abs=0.005
    )

    by_length = {
        row["group"]: row for row in subgroups["length_category"]["rows"]
    }
    expected = {
        "Short (1-30)": (84, 17.72),
        "Medium (31-60)": (57, 23.86),
        "Long (61-100)": (18, 68.78),
        "Very long (101+)": (49, 91.85),
    }
    for name, (count, macro) in expected.items():
        assert by_length[name]["samples"] == count
        assert by_length[name]["macro_cer"] * 100 == pytest.approx(
            macro, abs=0.005
        )


def test_experiment2_reproduces_reported_numbers():
    _, records = _load_legacy("experiment2")
    overall = am.summarize(records)["overall"]

    assert overall["samples"] == 208
    assert overall["corpus_cer"] * 100 == pytest.approx(81.70, abs=0.005)
    assert overall["corpus_wer"] * 100 == pytest.approx(99.07, abs=0.005)
    assert overall["macro_cer"] * 100 == pytest.approx(67.84, abs=0.005)
    assert overall["exact_match"] == 0.0
    assert overall["collapse_count"] == 71


def test_group_counts_are_consistent():
    _, records = _load_legacy("baseline")
    summary = am.summarize(records)

    for field, block in summary["subgroups"].items():
        assert sum(row["samples"] for row in block["rows"]) == len(records)


def test_make_record_toy_example():
    raw = {
        "dataset_index": 7,
        "writer_id": "w1",
        "source": "tatoeba",
        "neatness": 85,
        "line_width": 2,
        "augmented_noise": False,
    }

    record = am.make_record(raw, "saya suka kopi", "saya kopi")

    assert record["char_errors"] == 5
    assert record["cer"] == pytest.approx(5 / 14)
    assert record["word_errors"] == 1
    assert record["reference_words"] == 3
    assert record["neatness_category"] == "Neat (81-90)"
    assert not record["multiline"]
    assert not record["exact_match"]


def test_save_outputs_writes_all_files(tmp_path):
    _, records = _load_legacy("baseline")

    meta = {
        "label": "Baseline",
        "checkpoint": "checkpoints/baseline_best.pt",
        "checkpoint_sha256": "test",
        "checkpoint_epoch": 10,
        "checkpoint_validation_cer": 0.6956,
        "parameters": 7296710,
        "architecture": "cnn_bilstm",
        "image_height": 128,
        "image_width": 1024,
        "ctc_timesteps": 256,
        "device": "cpu",
        "batch_size": 2,
        "mean_ms_per_image": 1.0,
    }

    summary, paths = am.save_outputs(str(tmp_path), "baseline", meta, records)

    for path in paths.values():
        assert os.path.exists(path)

    with open(paths["json"], encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["overall"]["samples"] == 208
    assert "multiline_x_length" in payload["subgroups"]

    text = open(paths["txt"], encoding="utf-8").read()
    assert "Corpus CER:  68.79%" in text
