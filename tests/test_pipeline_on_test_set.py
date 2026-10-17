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
