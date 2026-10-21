import csv
import os

import numpy as np

import line_level_report as llr


def _write(directory, key, rows):
    fields = ["dataset_index", "writer_id", "multiline", "text_length", "char_errors",
              "reference_words", "word_errors", "exact_match", "collapsed_prediction"]

    with open(os.path.join(directory, f"{key}_error_analysis.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows(errors):
    out = []

    for i, e in enumerate(errors):
        out.append({"dataset_index": i, "writer_id": f"w{i // 2}", "multiline": str(i % 2 == 1),
                    "text_length": 10, "char_errors": e, "reference_words": 2,
                    "word_errors": 1 if e else 0, "exact_match": str(e == 0),
                    "collapsed_prediction": "False"})

    return out


def test_point_metrics_and_groups(tmp_path):
    _write(tmp_path, "baseline", _rows([0, 10, 2, 10, 0, 10, 2, 10]))
    models = {"baseline": llr.load_model(str(tmp_path), "baseline")}
    data = models["baseline"]

    single = llr.point_metrics(data, ~data["multiline"])
    multi = llr.point_metrics(data, data["multiline"])

    assert single["n"] == 4 and multi["n"] == 4
    assert np.isclose(single["corpus_cer"], 4 / 40)
    assert np.isclose(multi["corpus_cer"], 1.0)


def test_report_files_and_paired_difference(tmp_path):
    _write(tmp_path, "baseline", _rows([0, 10, 2, 10, 0, 10, 2, 10]))
    _write(tmp_path, "experiment1", _rows([5, 10, 7, 10, 5, 10, 7, 10]))

    llr.main(["--directory", str(tmp_path), "--models", "baseline", "experiment1", "--resamples", "300"])

    assert (tmp_path / "line_level_report.md").exists()

    with open(tmp_path / "line_level_report.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    row = next(r for r in rows if r["model"] == "experiment1" and r["group"] == "single-line" and r["unit"] == "sample")

    assert float(row["diff_cer_vs_baseline"]) > 0.1
    assert float(row["diff_lo"]) > 0 and float(row["corpus_cer_lo"]) <= float(row["corpus_cer"]) <= float(row["corpus_cer_hi"])


def test_mismatched_samples_rejected(tmp_path):
    import pytest

    _write(tmp_path, "baseline", _rows([0, 1, 2, 3]))
    shifted = _rows([0, 1, 2, 3])

    for r in shifted:
        r["dataset_index"] += 100

    _write(tmp_path, "experiment1", shifted)

    with pytest.raises(SystemExit):
        llr.main(["--directory", str(tmp_path), "--models", "baseline", "experiment1", "--resamples", "10"])
