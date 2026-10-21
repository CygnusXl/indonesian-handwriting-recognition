"""
Tests of the real-world evaluation script. A random-weight model is used,
so only the bookkeeping is verified: files found, ground truth matched,
metrics computed ONLY when a reference exists, nothing invented.
"""

import csv
import json
import os

import pytest

import evaluate_pipeline as ev
import synthetic_pages as sp


def test_reference_normalisation_and_line_count():
    assert ev.normalise_reference("a\nb  c\\nd") == "a b c d"
    assert ev.count_reference_lines("one\ntwo") == 2
    assert ev.count_reference_lines("one\\ntwo\\nthree") == 3
    assert ev.count_reference_lines("single") == 1


def test_score_is_empty_without_ground_truth():
    metrics, pair = ev.score(None, "anything")

    assert pair is None
    assert metrics["cer"] == "" and metrics["wer"] == "" and metrics["char_errors"] == ""


def test_score_values_and_case_insensitive_variant():
    metrics, _ = ev.score("Hello world", "hello world")

    assert metrics["char_errors"] == 1 and metrics["reference_chars"] == 11
    assert metrics["cer"] == pytest.approx(1 / 11)
    assert metrics["cer_ignoring_case"] == 0
    assert metrics["wer"] == pytest.approx(1 / 2)
    assert metrics["exact_match"] is False

    perfect, _ = ev.score("a\nb", "a b")
    assert perfect["cer"] == 0 and perfect["exact_match"] is True and perfect["true_lines"] == 2


def test_out_of_vocabulary_characters_are_reported(random_loaded):
    metrics, _ = ev.score("ab@#", "ab", random_loaded.vocabulary)

    assert metrics["out_of_vocabulary_chars"] == "#@"


@pytest.fixture()
def folder(tmp_path):
    page, truth = sp.make_condition_page("ruled", 11, n_lines=3)
    page.convert("RGB").save(tmp_path / "page1.jpg")

    other, truth2 = sp.make_condition_page("shadow", 12, n_lines=2)
    other.convert("RGB").save(tmp_path / "page2.png")

    single, truth3 = sp.render_page(5, n_lines=1, ruled=False)
    single.crop((120, 100, 1000, 330)).save(tmp_path / "single.png")
    (tmp_path / "single.txt").write_text(truth3["texts"][0], encoding="utf-8")

    with open(tmp_path / "ground_truth.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_file", "reference_text", "capture_condition"])
        writer.writerow(["page1.jpg", "\n".join(truth["texts"]), "synthetic"])
        # page2.png deliberately has NO ground truth

    (tmp_path / "manual_lines" / "page1").mkdir(parents=True)

    for index, box in enumerate(truth["boxes"], 1):
        page.crop((box[0] - 10, box[1] - 8, box[2] + 10, box[3] + 8)).save(
            tmp_path / "manual_lines" / "page1" / f"{index:02d}.png")

    return tmp_path


def test_ground_truth_loading_csv_and_sidecar(folder):
    truth, extras = ev.load_ground_truth(str(folder))

    assert set(truth) == {"page1.jpg", "single.png"}
    assert extras["page1.jpg"]["capture_condition"] == "synthetic"


def test_find_images_ignores_non_images(folder):
    assert ev.find_images(str(folder)) == ["page1.jpg", "page2.png", "single.png"]


def test_end_to_end_run_writes_honest_outputs(folder, random_loaded, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    truth, extras = ev.load_ground_truth(str(folder))

    rows = ev.evaluate_folder(random_loaded, str(folder), ["A", "B", "C", "D", "E"],
                              truth, extras, str(out), log=lambda *_: None)

    by = {(r["filename"], r["experiment"]): r for r in rows}

    # A-D for every image; E only where manual crops exist
    assert len([r for r in rows if r["experiment"] == "E"]) == 1
    assert ("page1.jpg", "E") in by and ("page2.png", "E") not in by
    assert len(rows) == 3 * 4 + 1

    # metrics exist only where ground truth exists
    assert by[("page2.png", "D")]["has_ground_truth"] is False
    assert by[("page2.png", "D")]["cer"] == "" and by[("page2.png", "D")]["wer"] == ""
    assert by[("page1.jpg", "D")]["cer"] != "" and by[("single.png", "D")]["cer"] != ""

    # required per-image fields
    for key in ("filename", "detected_lines", "preprocessing", "predicted_text",
                "inference_s", "heuristic_confidence"):
        assert key in by[("page1.jpg", "D")]

    assert by[("page1.jpg", "D")]["detected_lines"] == 3
    assert by[("page1.jpg", "D")]["line_count_correct"] is True
    assert by[("page2.png", "D")]["line_count_correct"] == ""     # unknown, not guessed
    assert by[("single.png", "D")]["input_kind"] == "single_line"

    assert os.path.isdir(out / "debug" / "page1" / "D")
    assert any(n.endswith("_model_input.png") for n in os.listdir(out / "debug" / "page1" / "D"))

    summary = ev.summarise(rows, ["A", "B", "C", "D", "E"])
    d = [s for s in summary if s["experiment"] == "D"][0]

    assert d["images"] == 3 and d["images_with_ground_truth"] == 2
    assert d["corpus_cer"] != ""

    no_truth = ev.summarise([r for r in rows if not r["has_ground_truth"]], ["D"])
    assert no_truth[0]["corpus_cer"] == "" and no_truth[0]["macro_cer"] == ""


def test_main_cli_writes_all_files(folder, tmp_path, monkeypatch):
    import torch
    import htr_common as hc

    pool = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\\")[:69]
    ids = {c: i for i, c in enumerate(pool)}
    path = tmp_path / "final_model.pt"
    torch.save({"model_state_dict": hc.build_model("cnn_bilstm", 70).state_dict(),
                "vocabulary_char_to_id": ids,
                "vocabulary_id_to_char": {i: c for c, i in ids.items()},
                "num_classes": 70, "architecture": "cnn_bilstm",
                "image_height": 128, "image_width": 1024, "ctc_timesteps": 256}, path)

    out = tmp_path / "out"
    rows, summary = ev.main(["--images", str(folder), "--model", str(path), "--device", "cpu",
                             "--output", str(out), "--experiments", "A", "D",
                             "--no-debug-images"])

    for name in ("per_image.csv", "summary.csv", "summary.json", "report.md"):
        assert (out / name).exists()

    with open(out / "per_image.csv", encoding="utf-8") as file:
        read = list(csv.DictReader(file))

    assert len(read) == 6 and list(read[0].keys())[:2] == ["filename", "experiment"]
    assert json.load(open(out / "summary.json"))["images"] == 3

    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Confidence = model heuristic" in report and "descriptive" in report


def test_main_exits_cleanly_without_images(tmp_path):
    with pytest.raises(SystemExit) as error:
        ev.main(["--images", str(tmp_path), "--model", "unused.pt"])

    assert "No images" in str(error.value)


def test_margin_override_changes_only_segmenting_experiments():
    import page_recognition as pr

    saved = dict(pr.EXPERIMENTS)

    try:
        ev.override_margin_x(0.2)

        assert pr.EXPERIMENTS["C"].segmentation.margin_x == 0.2
        assert pr.EXPERIMENTS["D"].segmentation.margin_x == 0.2
        assert pr.EXPERIMENTS["A"] == saved["A"] and pr.EXPERIMENTS["B"] == saved["B"]
    finally:
        pr.EXPERIMENTS.clear()
        pr.EXPERIMENTS.update(saved)
