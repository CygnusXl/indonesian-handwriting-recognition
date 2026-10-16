"""
HTML building blocks for the web UI (app/app.py).

Kept free of Streamlit so they can be unit-tested. Every piece of text that
comes from the model or the user is HTML-escaped here; nothing else in the
app injects unescaped model output into the page.
"""

import html
import os

STYLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")


def load_css():
    with open(STYLE_PATH, encoding="utf-8") as file:
        return f"<style>{file.read()}</style>"


def hero(model_label, parameters=None):
    chips = ["Reads one line at a time", "Finds the lines in page photos", "Runs on CPU"]
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)

    return f"""
<section class="hero" aria-labelledby="hero-title">
  <div class="hero-margin" aria-hidden="true"></div>
  <div class="hero-body">
    <p class="hero-course">Deep Learning final project &middot; BINUS University</p>
    <h1 id="hero-title">Photograph handwriting.<br>Get the text.</h1>
    <p class="hero-sub">Upload a photo of a notebook page or a single line. The app straightens
    and cleans the image, finds each line of writing, and reads the lines one by one with
    a CNN&ndash;BiLSTM&ndash;CTC model.</p>
    <div class="chips">{chip_html}</div>
  </div>
</section>
<aside class="notice" role="note">
  <strong>Trained on synthetic handwriting only.</strong>
  Accuracy on real handwriting photos has not been measured yet, so treat the output
  as a draft and check it against the page.
</aside>
"""


def empty_state():
    tips = [
        ("Lay the page flat", "Photograph from straight above so lines look straight."),
        ("Use even light", "Avoid shadows across the writing and glare on the paper."),
        ("Fill the frame", "Writing should be at least about 25 px tall in the photo."),
        ("Dark ink on light paper", "Pencil and pale pens are harder to read."),
    ]
    items = "".join(
        f'<li><strong>{html.escape(t)}</strong><span>{html.escape(d)}</span></li>'
        for t, d in tips
    )

    return f"""
<div class="empty" role="region" aria-label="Photo tips">
  <h2>Upload an image of handwriting to begin.</h2>
  <p>One line or a whole notebook page, as PNG, JPG or WEBP (up to 10&nbsp;MB).</p>
  <ul class="tips">{items}</ul>
</div>
"""


def step_header(number, title, caption=""):
    cap = f'<p class="step-cap">{html.escape(caption)}</p>' if caption else ""

    return (f'<div class="step"><span class="step-no" aria-hidden="true">{number}</span>'
            f'<div><h3>{html.escape(title)}</h3>{cap}</div></div>')


def status_banner(kind, message):
    """kind: info | warn | stop"""

    icons = {"info": "i", "warn": "!", "stop": "&times;"}

    return (f'<div class="banner banner-{kind}" role="status">'
            f'<span class="banner-icon" aria-hidden="true">{icons.get(kind, "i")}</span>'
            f'<span>{html.escape(message)}</span></div>')


def paper_text(lines):
    """
    The recognised text on ruled paper: one row per detected line with its
    line number in the margin and the model's HEURISTIC confidence.

    lines: iterable of (index, text, confidence_or_None)
    """

    rows = []

    for index, text, confidence in lines:
        shown = html.escape(text) if text else '<span class="blank">(nothing read)</span>'

        if confidence is None:
            meter = ""
        else:
            percent = max(0, min(100, int(round(confidence * 100))))
            meter = (f'<span class="meter" title="Model confidence (heuristic, not a '
                     f'calibrated probability): {percent}%"><span class="meter-track"><span class="meter-bar" '
                     f'style="width:{percent}%"></span></span><span class="meter-n">{percent}%</span></span>')

        rows.append(f'<li><span class="ln" aria-label="Line {index + 1}">{index + 1}</span>'
                    f'<span class="txt">{shown}</span>{meter}</li>')

    return f'<ol class="paper" aria-label="Recognised text">{"".join(rows)}</ol>'


def paper_empty(message):
    return f'<div class="paper paper-empty"><p>{html.escape(message)}</p></div>'


def model_card(label, parameters, input_size, metrics=None):
    rows = [("Model", label), ("Input", input_size)]

    if parameters:
        rows.append(("Parameters", f"{parameters:,}"))

    rows.append(("Decoding", "CTC greedy"))

    if metrics:
        rows.append((f"Synthetic test set ({metrics['samples']} samples)",
                     f"CER {metrics['corpus_cer'] * 100:.1f}% &middot; "
                     f"WER {metrics['corpus_wer'] * 100:.1f}% &middot; "
                     f"exact {metrics['exact_match'] * 100:.1f}%"))

    cells = "".join(
        f'<div class="{"wide" if "&middot;" in str(v) else ""}"><dt>{html.escape(k)}</dt>'
        f'<dd>{v if "&middot;" in str(v) else html.escape(str(v))}</dd></div>'
        for k, v in rows
    )

    return f'<dl class="card">{cells}</dl>'
