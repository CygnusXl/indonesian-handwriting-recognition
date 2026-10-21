"""
English -> Indonesian translation of the recognised text.

STATUS: the splitting / error handling is unit-tested with a stand-in
translator. The live service (Google Translate through the
`deep-translator` package) has NOT been exercised from the development
workspace (its requests were refused there), so whether it works from
your machine and from Streamlit Community Cloud is UNVERIFIED until you
click the button once. It is an unofficial, free endpoint: it can be
rate-limited or change without notice, which the app reports as a plain
error instead of crashing.

PRIVACY: the text is sent to Google for translation. The photo itself is
never sent. The app shows a note saying so next to the button.

Recognition mistakes are translated too: the app lets the user correct
the English text first.
"""

import re

MAX_CHARS = 4500  # the free endpoint accepts up to 5000 characters


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


def _default_translator(source, target):
    try:
        from deep_translator import GoogleTranslator
    except ImportError as error:
        raise TranslationError(
            "Translation is not available: the 'deep-translator' package "
            "is not installed."
        ) from error

    return GoogleTranslator(source=source, target=target).translate


def translate_text(text, translator=None, source="en", target="id"):
    """
    Translate `text`, keeping paragraph breaks. `translator` is any
    callable str -> str (tests pass a stand-in); by default Google
    Translate is used. Returns "" for empty text, and returns text with
    no letters (numbers, punctuation) unchanged without a request.
    Raises TranslationError with a user-readable message on failure.
    """

    paragraphs = split_paragraphs(text or "")

    if not paragraphs:
        return ""

    translate = translator or _default_translator(source, target)
    output = []

    for paragraph in paragraphs:
        if not any(character.isalpha() for character in paragraph):
            output.append(paragraph)
            continue

        translated = []

        for piece in chunk_text(paragraph):
            try:
                result = translate(piece)
            except TranslationError:
                raise
            except Exception as error:
                raise TranslationError(
                    "The translation service could not be reached or "
                    "refused the request (it may be busy). Try again in "
                    f"a moment. Details: {type(error).__name__}"
                ) from error

            if not isinstance(result, str):
                raise TranslationError(
                    "The translation service returned nothing for this text."
                )

            translated.append(result.strip())

        output.append(" ".join(translated))

    return "\n\n".join(output)
