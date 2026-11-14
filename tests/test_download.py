"""Tests for the dataset preparation filters in scripts/download_datasets.py."""

import base64
import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "download_datasets", Path(__file__).resolve().parents[1] / "scripts" / "download_datasets.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

PLAIN = "Explain how photosynthesis works and why it is important for the planet."


def test_plain_english_is_not_flagged():
    assert dd.detect_obfuscation(PLAIN) is None
    assert dd.detect_obfuscation("Write a tutorial on how to make a cake") is None


def test_base64_prompt_is_flagged():
    assert dd.detect_obfuscation(base64.b64encode(PLAIN.encode()).decode()) == "base64"


def test_reversed_prompt_is_flagged():
    text = "This is a long sentence and it is written for the test of what we do with the text."
    assert dd.detect_obfuscation(text[::-1]) == "reversed"


def test_exclusion_reason_for_long_and_encoded_pairs():
    ok = pd.Series({"adversarial_prompt": PLAIN, "benign_prompt": PLAIN})
    long = pd.Series({"adversarial_prompt": "word " * 2000, "benign_prompt": PLAIN})
    enc = pd.Series({"adversarial_prompt": PLAIN, "benign_prompt": base64.b64encode(PLAIN.encode()).decode()})
    assert dd.fortress_exclusion_reason(ok) is None
    assert "too long" in dd.fortress_exclusion_reason(long)
    assert "base64" in dd.fortress_exclusion_reason(enc)
