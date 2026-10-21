import pipeline_on_test_set as p


def _row(exp, group, ce, cn, we, wn, em):
    return {"experiment": exp, "gt_group": group, "has_ground_truth": "True", "char_errors": ce,
            "reference_chars": cn, "word_errors": we, "reference_words": wn, "exact_match": em}


def test_summarize_rows_groups_and_corpus_weighting():
    rows = [_row("D", "single", 1, 10, 1, 2, "False"), _row("D", "multiline", 30, 30, 6, 6, "False"),
            _row("D", "single", 0, 10, 0, 2, "True"),
            {"experiment": "D", "gt_group": "single", "has_ground_truth": "False"}]
    out = p.summarize_rows(rows)

    assert out[("D", "single")]["n"] == 2 and abs(out[("D", "single")]["corpus_cer"] - 0.05) < 1e-9
    assert out[("D", "single")]["exact_match"] == 0.5
    assert abs(out[("D", "all")]["corpus_cer"] - 31 / 50) < 1e-9


def _pi(exp, name, ce, cn, group="single"):
    return {"experiment": exp, "filename": name, "gt_group": group, "has_ground_truth": "True",
            "char_errors": ce, "reference_chars": cn}


def test_paired_differences_detect_a_consistent_change_and_flag_noise():
    rows = []

    for i in range(40):
        rows += [_pi("A", f"f{i}", 1, 10), _pi("B", f"f{i}", 3, 10)]          # B always worse by 2/10
        rows += [_pi("C", f"f{i}", 1 + (i % 2) * 2 - 1, 10)]                  # C vs A: +-1 alternating

    out = {(d["first"], d["second"], d["group"]): d for d in p.paired_differences(rows, resamples=2000)}

    ab = out[("A", "B", "single")]
    assert abs(ab["diff"] - 0.2) < 1e-9 and ab["lo"] > 0.19          # consistent: interval excludes 0

    ac = out[("A", "C", "single")]
    assert abs(ac["diff"]) < 1e-9 and ac["lo"] < 0 < ac["hi"]        # no net change: within noise
    assert "yes" in p.paired_table([ac]) and "no" in p.paired_table([ab])


def test_paired_differences_only_use_images_run_in_both_pipelines():
    rows = [_pi("A", "x", 1, 10), _pi("A", "y", 1, 10), _pi("B", "x", 2, 10)]

    assert p.paired_differences(rows, resamples=50) == []                  # one common image: too few


def test_paragraph_rates_count_images_with_more_than_one_paragraph():
    rows = [{"experiment": "D", "gt_group": "multiline", "detected_paragraphs": v} for v in ("1", "1", "2", "1")]
    rows.append({"experiment": "A", "gt_group": "multiline", "detected_paragraphs": ""})   # A: no detection

    counts = p.paragraph_rates(rows)

    assert counts == {("D", "multiline"): (1, 4)}
    assert "1 (25%)" in p.paragraph_table(counts)
