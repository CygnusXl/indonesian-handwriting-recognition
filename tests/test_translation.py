"""Translation module and the app's button, with a stand-in translator
(the real service is not called)."""

import os

import pytest

import app_logic as logic
import translation as tr
import ui_components as ui


def fake(text):
    return f"ID[{text}]"


def test_paragraph_breaks_kept_and_line_breaks_joined():
    out = tr.translate_text("Hello\nworld.\n\nSecond one", fake)

    assert out == "ID[Hello world.]\n\nID[Second one]"


def test_empty_and_no_letter_text_make_no_request():
    calls = []

    def spy(text):
        calls.append(text)
        return text

    assert tr.translate_text("   \n ", spy) == ""
    assert tr.translate_text("", spy) == ""
    assert tr.translate_text("123 - 45", spy) == "123 - 45"
    assert calls == []


def test_long_paragraph_is_chunked_under_the_limit():
    sentence = "This is a sentence. "
    text = sentence * 600
    pieces = tr.chunk_text(text.strip(), limit=100)

    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces).split() == text.split()

    out = tr.translate_text(text, fake)
    assert out.count("ID[") == len(tr.chunk_text(text.strip()))


def test_unbreakable_text_is_hard_cut():
    pieces = tr.chunk_text("x" * 250, limit=100)

    assert [len(p) for p in pieces] == [100, 100, 50]


def test_translator_failure_becomes_translation_error():
    def broken(text):
        raise ConnectionError("down")

    with pytest.raises(tr.TranslationError) as error:
        tr.translate_text("Hello", broken)

    assert "ConnectionError" in str(error.value)


def test_non_string_result_is_an_error():
    with pytest.raises(tr.TranslationError):
        tr.translate_text("Hello", lambda t: None)


def test_missing_package_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(tr, "_OFFLINE", {})
    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(tr.TranslationError) as error:
        tr.translate_text("Hello")

    assert "not installed" in str(error.value)


def test_model_load_failure_message(monkeypatch):
    import transformers

    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(tr, "_OFFLINE", {})
    monkeypatch.setattr(transformers.MarianTokenizer, "from_pretrained", boom)

    with pytest.raises(tr.TranslationError) as error:
        tr.translate_text("Hello")

    assert "300 MB" in str(error.value) and "OSError" in str(error.value)


def test_model_loads_lazily_and_only_when_letters_present(monkeypatch):
    loads = []
    monkeypatch.setattr(
        tr, "load_offline_translator", lambda: loads.append(1) or fake
    )

    assert tr.translate_text("123 !!") == "123 !!"
    assert loads == []

    assert tr.translate_text("Hello") == "ID[Hello]"
    assert tr.translate_text("a\n\nb") == "ID[a]\n\nID[b]"
    assert len(loads) == 2  # once per call, never when nothing needs it


@pytest.mark.skipif(
    os.environ.get("HTR_TEST_TRANSLATION") != "1",
    reason="downloads ~300 MB; run with HTR_TEST_TRANSLATION=1",
)
def test_real_offline_model_translates():
    out = tr.translate_text("Good morning, my friend.")

    assert "pagi" in out.lower()


def test_app_logic_wrapper_never_raises(monkeypatch):
    monkeypatch.setattr(
        tr, "translate_text",
        lambda text: (_ for _ in ()).throw(tr.TranslationError("busy")),
    )

    assert logic.translate_to_indonesian("Hi") == ("", "busy")

    monkeypatch.setattr(tr, "translate_text", lambda text: "Halo")
    assert logic.translate_to_indonesian("Hi") == ("Halo", None)


def test_translation_block_escapes_html_and_keeps_breaks():
    block = ui.translation_block("<b>a</b>\n\nb")

    assert "&lt;b&gt;" in block and block.count("<br>") == 2


def test_hint_only_for_the_indonesian_trained_model():
    assert ui.training_info({})["translation_hint"]
    assert ui.training_info({"training_data": "iam"})["translation_hint"] == ""
