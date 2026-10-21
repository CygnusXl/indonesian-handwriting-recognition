"""
English -> Indonesian translation of the recognised text, OFFLINE.

The translator is the open MarianMT model Helsinki-NLP/opus-mt-en-id
(about 73 million parameters, run with the `transformers` library). It
runs inside the app: the text is never sent to a translation service or
an API. The only network use is the one-time download of the model files
(about 300 MB) from the Hugging Face Hub on first use, after which they
are cached on disk.

Checked on the development workspace (CPU): the model loaded in about
25 s, translated four short test sentences into fluent Indonesian in
0.3-1.1 s each, and the process used about 1.1 GB of memory in total
(about 0.8 GB of that is Python + PyTorch + transformers before the
model loads). That is a single informal run, not a benchmark, and it is
NOT checked on Streamlit Community Cloud, whose free tier may not have
enough memory for it (see DEPLOY.md).

Recognition mistakes are translated too: the app lets the user correct
the English text first. Translation quality on real notes has not been
measured.
"""

import re

MODEL_NAME = "Helsinki-NLP/opus-mt-en-id"
MAX_CHARS = 500  # per piece; the model handles at most 512 tokens


class TranslationError(RuntimeError):
    """Translation failed; the message is safe to show to the user."""


def split_paragraphs(text):
    """
    Paragraphs are separated by a blank line. Line breaks inside a
    paragraph are visual (where the handwriting wrapped), so they become
    spaces: translating a sentence cut in the middle gives worse output.
    """

    paragraphs = []

    for block in re.split(r"\n\s*\n", text.strip()):
        joined = " ".join(block.split())

        if joined:
            paragraphs.append(joined)

    return paragraphs


def chunk_text(text, limit=MAX_CHARS):
    """Split one paragraph into pieces of at most `limit` characters,
    preferably at sentence ends, otherwise at spaces."""

    pieces = []

    while len(text) > limit:
        cut = max(
            (text.rfind(mark, 0, limit) for mark in (". ", "! ", "? ")),
            default=-1,
        )
        cut = cut + 1 if cut > 0 else text.rfind(" ", 0, limit)

        if cut <= 0:
            cut = limit

        pieces.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        pieces.append(text)

    return pieces


_OFFLINE = {}


def load_offline_translator(name=MODEL_NAME):
    """
    Load (once per process) the offline model and return a callable
    str -> str. Raises TranslationError with a user-readable message if
    the packages or the model files are not available.
    """

    if name in _OFFLINE:
        return _OFFLINE[name]

    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as error:
        raise TranslationError(
            "Translation is not available: the 'transformers' and "
            "'sentencepiece' packages are not installed."
        ) from error

    try:
        tokenizer = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name).eval()
    except Exception as error:
        raise TranslationError(
            "The translation model could not be loaded. The first use "
            "downloads about 300 MB, so it needs an internet connection "
            f"once ({type(error).__name__})."
        ) from error

    def translate(piece):
        batch = tokenizer(
            [piece], return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output = model.generate(**batch, num_beams=4, max_length=512)

        return tokenizer.decode(output[0], skip_special_tokens=True)

    _OFFLINE[name] = translate

    return translate


def translate_text(text, translator=None):
    """
    Translate `text`, keeping paragraph breaks. `translator` is any
    callable str -> str (tests pass a stand-in); by default the offline
    model is used. Returns "" for empty text, and returns text with no
    letters (numbers, punctuation) unchanged without running the model.
    Raises TranslationError with a user-readable message on failure.
    """

    paragraphs = split_paragraphs(text or "")

    if not paragraphs:
        return ""

    translate = translator
    output = []

    for paragraph in paragraphs:
        if not any(character.isalpha() for character in paragraph):
            output.append(paragraph)
            continue

        if translate is None:
            translate = load_offline_translator()

        translated = []

        for piece in chunk_text(paragraph):
            try:
                result = translate(piece)
            except TranslationError:
                raise
            except Exception as error:
                raise TranslationError(
                    "The translation failed "
                    f"({type(error).__name__}). Try a shorter text."
                ) from error

            if not isinstance(result, str):
                raise TranslationError("The translator returned nothing.")

            translated.append(result.strip())

        output.append(" ".join(translated))

    return "\n\n".join(output)
