"""HTML building blocks of the web UI: escaping, structure, no model needed."""

import ui_components as ui


def test_recognised_text_is_escaped():
    html = ui.paper_text([(0, '<script>alert("x")</script> & co', 0.5)])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&amp; co" in html


def test_paper_has_one_row_per_line_with_heuristic_confidence_tooltip():
    html = ui.paper_text([(0, "satu", 0.9), (1, "", None), (2, "tiga", 0.123)])

    assert html.count("<li>") == 3
    assert "(nothing read)" in html
    assert "heuristic, not a calibrated probability" in html
    assert "12%" in html and "width:12%" in html


def test_confidence_is_clamped_to_a_valid_percentage():
    html = ui.paper_text([(0, "a", 1.7), (1, "b", -0.3)])

    assert "width:100%" in html and "width:0%" in html


def test_empty_and_banner_states_render_text_safely():
    assert "&lt;b&gt;" in ui.status_banner("warn", "<b>bad</b>")
    assert "banner-stop" in ui.status_banner("stop", "x")
    assert "Upload an image" in ui.empty_state()
    assert "Trained on synthetic handwriting only" in ui.hero("m")


def test_model_card_shows_measured_numbers_only_when_given():
    plain = ui.model_card("Baseline", 7296710, "128 x 1024 px")
    scored = ui.model_card("Baseline", 7296710, "128 x 1024 px",
                           {"samples": 208, "corpus_cer": 0.6879,
                            "corpus_wer": 0.8508, "exact_match": 0.0625})

    assert "Synthetic test set" not in plain and "7,296,710" in plain
    assert "208 samples" in scored and "CER 68.8%" in scored


def test_stylesheet_loads_and_keeps_streamlit_icons():
    css = ui.load_css()

    assert css.startswith("<style>") and "Material Symbols" in css


def test_dark_stylesheet_loads_and_overrides_tokens():
    css = ui.load_dark_css()

    assert css.startswith("<style>") and "--page:" in css and "--ink:" in css


def test_example_label_truncates_and_ground_truth_is_escaped():
    assert ui.example_label(0, "Short") == 'Sample 1: "Short"'
    assert ui.example_label(2, "x" * 60).endswith('..."') and len(ui.example_label(2, "x" * 60)) < 60
    assert "&lt;b&gt;" in ui.ground_truth("<b>")
    assert "<b>" not in ui.ground_truth("<b>")


def test_paper_marks_the_start_of_a_new_paragraph_only():
    html_out = ui.paper_text([(0, "a", None, 0), (1, "b", None, 0), (2, "c", None, 1)])

    assert html_out.count('class="para-start"') == 1
    assert html_out.index('class="para-start"') > html_out.index(">b<")
    assert 'class="para-start"' not in ui.paper_text([(0, "a", None), (1, "b", None)])
