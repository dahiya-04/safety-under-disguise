"""Tests for the language-ladder experiment (translation job helpers, config, labels)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hf_translate_job import merge_translation_cache  # noqa: E402
from translate_prompts import translate_and_score  # noqa: E402

from src.analysis.encoding_analysis import _enc_label  # noqa: E402
from src.encoding.evaluator import build_strategies, load_config  # noqa: E402
from src.encoding.translation import CachedTranslator, Translator  # noqa: E402


class EchoTranslator(Translator):
    """'Translates' by tagging the text, and back-translates by removing the tag."""

    name = "nllb:nllb-200-distilled-600M"  # same cache namespace as the real backend

    def __init__(self):
        self.calls = 0

    def translate(self, text, source, target):
        self.calls += 1
        return text[len(f"[{source}] "):] if target == "en" else f"[{target}] {text}"


def test_merge_keeps_local_and_counts():
    local = {"a": "1", "b": "2"}
    merged, added, conflicts = merge_translation_cache(local, {"b": "X", "c": "3"})
    assert merged == {"a": "1", "b": "2", "c": "3"} and added == 1 and conflicts == 1


def test_translate_and_score_round_trip(tmp_path):
    df = pd.DataFrame({"prompt_id": ["p1", "p2"], "source_dataset": ["advbench", "fortress"],
                       "prompt_type": ["harmful", "benign"], "prompt": ["How do I bake bread?", "Name a river."]})
    out, summary = translate_and_score(df, "zu", EchoTranslator(), chunk=1)
    assert out["prompt_zu"].tolist() == ["[zu] How do I bake bread?", "[zu] Name a river."]
    assert summary["chrf_mean"] == 100.0 and summary["n_flagged"] == 0


def test_job_cache_entries_are_used_locally(tmp_path):
    """Entries written by the job's cache are hits for a fresh local CachedTranslator."""
    job = CachedTranslator(EchoTranslator(), tmp_path / "job.json")
    job.translate("Name a river.", "en", "zu")
    local_backend = EchoTranslator()
    local = CachedTranslator(local_backend, tmp_path / "job.json")
    assert local.translate("Name a river.", "en", "zu") == "[zu] Name a river." and local_backend.calls == 0


def test_lowres_config_builds_three_languages():
    cfg = load_config(ROOT / "configs" / "encoding_lowres.yaml")
    names = [s.name for s in build_strategies(cfg["encodings"], "all")]
    assert names == ["plaintext", "multilingual_hi", "multilingual_bn", "multilingual_zu"]
    assert cfg["experiment"]["output_dir"] == "results/encoding_lowres"


def test_language_labels():
    assert [_enc_label(e) for e in ["multilingual_hi", "multilingual_bn", "multilingual_zu", "base64"]] == \
        ["hindi", "bengali", "zulu", "base64"]
