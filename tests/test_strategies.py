"""Unit tests for encoding strategies (no network access required)."""

import pytest

from src.encoding.strategies import (
    STRATEGY_REGISTRY,
    Base64Encoding,
    LeetSpeakEncoding,
    MultilingualEncoding,
    PlaintextControl,
    ROT13Encoding,
    UnicodeHomoglyphEncoding,
    get_strategy,
)
from src.encoding.translation import Translator

SAMPLE = "Explain how photosynthesis works, step by step."


class FakeTranslator(Translator):
    name = "fake"

    def translate(self, text: str, source: str, target: str) -> str:
        return f"[{source}->{target}] {text}"


def test_plaintext_is_identity():
    assert PlaintextControl().encode(SAMPLE) == SAMPLE


def test_base64_roundtrip_and_wrapper():
    s = Base64Encoding()
    payload = s.transform(SAMPLE)
    assert Base64Encoding.decode(payload) == SAMPLE
    assert s.encode(SAMPLE).startswith("The following is a base64-encoded message.")
    assert payload in s.encode(SAMPLE)


def test_base64_handles_unicode():
    text = "नमस्ते दुनिया"
    assert Base64Encoding.decode(Base64Encoding().transform(text)) == text


def test_rot13_roundtrip_and_wrapper():
    s = ROT13Encoding()
    assert ROT13Encoding.decode(s.transform(SAMPLE)) == SAMPLE
    assert s.transform("Hello") == "Uryyb"
    assert "ROT13" in s.encode(SAMPLE)


def test_leetspeak_mapping_and_no_wrapper():
    s = LeetSpeakEncoding()
    assert s.transform("Steal tools") == "57341 70015"
    out = s.encode(SAMPLE)
    assert not any(c in out.lower() for c in "aeiost l".replace(" ", ""))
    assert s.wrapper is None


def test_homoglyph_looks_different_but_roundtrips():
    s = UnicodeHomoglyphEncoding()
    out = s.encode(SAMPLE)
    assert out != SAMPLE
    assert len(out) == len(SAMPLE)
    assert UnicodeHomoglyphEncoding.decode(out) == SAMPLE
    assert any(ord(c) > 127 for c in out)


def test_multilingual_translates_prompt_without_wrapper():
    s = MultilingualEncoding(target_language="hi", translator=FakeTranslator())
    assert s.name == "multilingual_hi"
    assert "Hindi" in s.description
    assert s.encode(SAMPLE) == f"[en->hi] {SAMPLE}"
    assert s.transform_batch([SAMPLE, "x"]) == [f"[en->hi] {SAMPLE}", "[en->hi] x"]


def test_multilingual_back_translation_off_by_default():
    s = MultilingualEncoding(target_language="hi", translator=FakeTranslator())
    assert s.postprocess_response("jawab") == "jawab"
    assert s.back_translate_response("jawab") == "[hi->en] jawab"
    assert s.back_translate_response("   ") == "   "


def test_multilingual_back_translation_when_enabled():
    s = MultilingualEncoding(target_language="zu", translator=FakeTranslator(), back_translate=True)
    assert s.postprocess_response("impendulo") == "[zu->en] impendulo"


@pytest.mark.parametrize("name", sorted(STRATEGY_REGISTRY))
def test_registry_strategies_have_required_properties(name):
    kwargs = {"translator": FakeTranslator()} if name == "multilingual" else {}
    s = get_strategy(name, **kwargs)
    assert isinstance(s.name, str) and s.name
    assert isinstance(s.description, str) and s.description
    assert 0 <= s.complexity_score <= 5
    assert isinstance(s.encode(SAMPLE), str)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy("morse")
