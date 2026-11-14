"""Tests for translation backends. Fast tests mock the model; the real model
test is marked ``slow`` (run with: pytest -m slow)."""

import pytest

from src.encoding.translation import (
    CachedTranslator,
    NLLBBackend,
    Translator,
    build_translator,
    split_into_segments,
)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def _rebuild(pairs):
    return "".join(seg + sep for seg, sep in pairs)


def test_split_sentences_on_one_line():
    pairs = split_into_segments("First one. Second one! Third?")
    assert [s for s, _ in pairs] == ["First one.", "Second one!", "Third?"]
    assert _rebuild(pairs) == "First one. Second one! Third?"


def test_split_preserves_line_breaks_and_blank_lines():
    text = "Line one. Still one.\n\nLine three."
    pairs = split_into_segments(text)
    assert _rebuild(pairs) == text


def test_split_keeps_initialisms_together():
    pairs = split_into_segments("Write about a U.S. president. Then stop.")
    assert [s for s, _ in pairs if s] == ["Write about a U.S. president.", "Then stop."]


def test_split_handles_devanagari_danda():
    pairs = split_into_segments("यह पहला है। यह दूसरा है।")
    assert len([s for s, _ in pairs if s]) == 2


# --------------------------------------------------------------------------- #
# NLLB backend (model mocked)
# --------------------------------------------------------------------------- #
class MockedNLLB(NLLBBackend):
    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def _load(self):  # no model download
        pass

    def _generate(self, sentences, src, tgt):
        self.calls.append(list(sentences))
        return [f"<{tgt}:{s}>" for s in sentences]


def test_language_code_mapping():
    assert NLLBBackend.to_nllb_code("hi") == "hin_Deva"
    assert NLLBBackend.to_nllb_code("hin_Deva") == "hin_Deva"
    with pytest.raises(ValueError):
        NLLBBackend.to_nllb_code("auto")
    with pytest.raises(ValueError):
        NLLBBackend.to_nllb_code("xx")


def test_nllb_batch_translates_sentences_and_rebuilds_structure():
    be = MockedNLLB()
    out = be.translate_batch(["A. B.", "C\n\nD."], "en", "hi")
    assert out == ["<hin_Deva:A.> <hin_Deva:B.>", "<hin_Deva:C>\n\n<hin_Deva:D.>"]
    assert be.calls == [["A.", "B.", "C", "D."]]  # one flattened batch


def test_nllb_same_language_is_identity():
    be = MockedNLLB()
    assert be.translate("hello", "en", "en") == "hello"
    assert be.calls == []


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class CountingTranslator(Translator):
    name = "counting"

    def __init__(self):
        self.n = 0

    def translate(self, text, source, target):
        self.n += 1
        return text.upper()


def test_cache_avoids_repeat_calls_and_persists(tmp_path):
    path = tmp_path / "cache.json"
    be = CountingTranslator()
    c = CachedTranslator(be, path)
    assert c.translate_batch(["a", "b", "a"], "en", "hi") == ["A", "B", "A"]
    assert be.n == 2
    assert c.translate("b", "en", "hi") == "B"
    assert be.n == 2

    c2 = CachedTranslator(CountingTranslator(), path)  # reload from disk
    assert c2.is_cached("a", "en", "hi")
    assert c2.translate("a", "en", "hi") == "A"
    assert c2.backend.n == 0


def test_cache_key_depends_on_direction(tmp_path):
    c = CachedTranslator(CountingTranslator(), tmp_path / "c.json")
    c.translate("x", "en", "hi")
    assert not c.is_cached("x", "hi", "en")


def test_build_translator_unknown_backend():
    with pytest.raises(ValueError):
        build_translator("babelfish")


# --------------------------------------------------------------------------- #
# Real model (slow, downloads ~2.5GB on first run)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_nllb_translates_benign_sentence_to_devanagari():
    out = NLLBBackend().translate("The weather is nice today.", "en", "hi")
    assert any("ऀ" <= ch <= "ॿ" for ch in out)
