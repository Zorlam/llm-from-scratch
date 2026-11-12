from gpt.language import detect_language


def test_detects_english():
    text = (
        "Artificial intelligence is transforming the world. "
        "Researchers are developing increasingly capable systems. "
        "These systems can understand language, process information, "
        "and assist people with complex tasks."
    )

    language, confidence = detect_language(text)

    assert language == "en"
    assert 0 <= confidence <= 1


def test_detects_french():
    text = (
        "L'intelligence artificielle transforme le monde. "
        "Les chercheurs développent des systèmes de plus en plus "
        "performants. Ces systèmes peuvent comprendre le langage, "
        "traiter des informations et aider les personnes."
    )

    language, confidence = detect_language(text)

    assert language == "fr"
    assert 0 <= confidence <= 1


def test_empty_text_returns_unknown():
    language, confidence = detect_language("")

    assert language is None
    assert confidence == 0.0


def test_very_short_text_returns_unknown():
    language, confidence = detect_language("Hello world")

    assert language is None
    assert confidence == 0.0


def test_whitespace_only_text_returns_unknown():
    language, confidence = detect_language("     \n\t ")

    assert language is None
    assert confidence == 0.0


def test_detection_is_deterministic():
    text = (
        "Machine learning allows computers to identify patterns "
        "and make predictions from data. It is widely used in "
        "science, medicine, business, and engineering."
    )

    first_result = detect_language(text)
    second_result = detect_language(text)

    assert first_result == second_result