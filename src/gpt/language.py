"""Language identification utilities."""

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException


# Make language detection reproducible.
DetectorFactory.seed = 0


MIN_TEXT_LENGTH = 50


def detect_language(
    text: str,
) -> tuple[str | None, float]:
    """
    Detect the most likely language of a text.

    Returns:
        A tuple containing the ISO 639-1 language code and confidence.
        Returns (None, 0.0) when the text is too short or cannot be
        reliably processed.
    """

    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return None, 0.0

    try:
        predictions = detect_langs(text)

    except LangDetectException:
        return None, 0.0

    if not predictions:
        return None, 0.0

    prediction = predictions[0]

    return prediction.lang, float(prediction.prob)