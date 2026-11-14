"""Evaluator tests with fake target models and a fake judge (no API calls)."""

import json

import pandas as pd
import pytest
import yaml

from src.encoding import evaluator as ev
from src.encoding.classifier import ClassificationResult, DecodingStatus, ResponseLabel
from src.models.base import GenerationResult


# ---------------------------------------------------------------- fixtures --
def make_prompts() -> pd.DataFrame:
    rows = []
    for i in range(6):
        rows.append(dict(prompt_id=f"advbench_{i:03d}", prompt=f"harmful adv {i}", prompt_type="harmful",
                         category="uncategorized", source_dataset="advbench", pair_id=None))
    for i in range(4):
        rows.append(dict(prompt_id=f"jbb_{i:03d}", prompt=f"harmful jbb {i}", prompt_type="harmful",
                         category="Fraud", source_dataset="jailbreakbench", pair_id=None))
    for i in range(4):
        for kind in ("harmful", "benign"):
            rows.append(dict(prompt_id=f"fortress_{kind}_{i:03d}", prompt=f"{kind} fortress {i}",
                             prompt_type=kind, category="CBRNE", source_dataset="fortress",
                             pair_id=f"fortress_pair_{i:03d}"))
    return pd.DataFrame(rows)


class FakeClient:
    def __init__(self, model, fail_on=None, empty_on=None):
        self.model = model
        self.fail_on = fail_on or set()
        self.empty_on = empty_on or set()

    def generate(self, prompt):
        if prompt in self.fail_on:
            return GenerationResult(text="", model=self.model, error="HTTP 503")
        if prompt in self.empty_on:  # reasoning model ran out of tokens
            return GenerationResult(text="", model=self.model, finish_reason="length", completion_tokens=4096)
        return GenerationResult(text=f"answer to: {prompt}", model=self.model, finish_reason="stop",
                                prompt_tokens=10, completion_tokens=20)


class FakeJudge:
    """Complies with plaintext harmful prompts' benign twins; refuses everything harmful except base64."""

    def classify(self, item):
        if item.encoding_type == "base64":
            label = ResponseLabel.COMPLIANCE
        elif "benign" in item.original_prompt:
            label = ResponseLabel.COMPLIANCE
        else:
            label = ResponseLabel.FULL_REFUSAL
        status = DecodingStatus.NOT_APPLICABLE if item.encoding_type == "plaintext" else DecodingStatus.CORRECT
        return ClassificationResult(label=label, confidence=0.9, decoding_status=status,
                                    has_disclaimer=False, specificity=4 if label.value == "COMPLIANCE" else 1,
                                    response_language="en", rationale="fake")


# ----------------------------------------------------------------- sampling --
def test_sample_keeps_pairs_together_and_is_deterministic():
    df = make_prompts()
    s1 = ev.sample_prompts(df, 8, seed=1)
    s2 = ev.sample_prompts(df, 8, seed=1)
    assert s1["prompt_id"].tolist() == s2["prompt_id"].tolist()
    fort = s1[s1["source_dataset"] == "fortress"]
    assert (fort.groupby("pair_id").size() == 2).all()
    assert set(s1["source_dataset"]) == {"advbench", "jailbreakbench", "fortress"}


def test_sample_none_returns_all():
    df = make_prompts()
    assert len(ev.sample_prompts(df, None, 0)) == len(df)
    assert len(ev.sample_prompts(df, 10_000, 0)) == len(df)


# --------------------------------------------------------------- strategies --
CFG_ENC = [{"name": "plaintext"}, {"name": "base64"}, {"name": "rot13"}]


def test_build_strategies_filters():
    assert [s.name for s in ev.build_strategies(CFG_ENC, "all")] == ["plaintext", "base64", "rot13"]
    assert [s.name for s in ev.build_strategies(CFG_ENC, "rot13, base64")] == ["base64", "rot13"]
    with pytest.raises(SystemExit):
        ev.build_strategies(CFG_ENC, "morse")


def test_select_model_configs_allows_adhoc():
    tm = {"models": [{"id": "a", "max_tokens": 9}, {"id": "b"}]}
    assert ev.select_model_configs(tm, "all") == tm["models"]
    assert ev.select_model_configs(tm, "a,zzz") == [{"id": "a", "max_tokens": 9}, {"id": "zzz"}]


# ------------------------------------------------------------------ summary --
def test_summarize_rates_and_lift():
    df = pd.DataFrame([
        dict(model="m", encoding_type="plaintext", prompt_type="harmful", classification="FULL_REFUSAL",
             decoding_status="n/a", specificity=1, has_disclaimer=False),
        dict(model="m", encoding_type="plaintext", prompt_type="harmful", classification="COMPLIANCE",
             decoding_status="n/a", specificity=4, has_disclaimer=True),
        dict(model="m", encoding_type="base64", prompt_type="harmful", classification="COMPLIANCE",
             decoding_status="correct", specificity=5, has_disclaimer=False),
        dict(model="m", encoding_type="base64", prompt_type="harmful", classification="PARTIAL_REFUSAL",
             decoding_status="correct", specificity=2, has_disclaimer=True),
        dict(model="m", encoding_type="base64", prompt_type="benign", classification="FULL_REFUSAL",
             decoding_status="failed", specificity=1, has_disclaimer=False),
        dict(model="m", encoding_type="base64", prompt_type="harmful", classification="GENERATION_ERROR",
             decoding_status=None, specificity=None, has_disclaimer=None),
        dict(model="m", encoding_type="base64", prompt_type="harmful", classification="NO_ANSWER",
             decoding_status=None, specificity=None, has_disclaimer=None),
        dict(model="m", encoding_type="base64", prompt_type="harmful", classification="FULL_REFUSAL",
             decoding_status="unclear", specificity=1, has_disclaimer=False),
    ])
    for c in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        df[c] = 1
    s = {x["encoding"]: x for x in ev.summarize(df, ["plaintext", "base64"])["by_model_encoding"]}
    assert s["plaintext"]["asr_strict"] == 0.5 and s["plaintext"]["decoding_success_rate"] is None
    # base64 harmful valid rows: COMPLIANCE, PARTIAL_REFUSAL, FULL_REFUSAL(unclear)
    assert s["base64"]["n_harmful"] == 3
    assert s["base64"]["asr_strict"] == round(1 / 3, 4) and s["base64"]["asr_lenient"] == round(2 / 3, 4)
    assert s["base64"]["n_errors"] == 1 and s["base64"]["n_no_answer"] == 1
    # determinable: correct, correct, failed -> 2/3 ; unclear: 1 of 4 valid rows
    assert s["base64"]["decoding_success_rate"] == round(2 / 3, 4)
    assert s["base64"]["decoding_unclear_rate"] == 0.25
    assert s["base64"]["benign_over_refusal_rate"] == 1.0
    assert s["base64"]["asr_strict_lift_vs_plaintext"] == round(1 / 3 - 0.5, 4)


# --------------------------------------------------------------- end-to-end --
def test_main_end_to_end_with_fakes(tmp_path, monkeypatch):
    prompts_csv = tmp_path / "prompts.csv"
    make_prompts().to_csv(prompts_csv, index=False)
    cfg = {
        "experiment": {"name": "t", "seed": 0, "output_dir": str(tmp_path / "out")},
        "dataset": {"prompts_csv": str(prompts_csv)},
        "encodings": CFG_ENC,
        "target_models": {"defaults": {"max_tokens": 10}, "models": [{"id": "fake/a"}, {"id": "fake/b"}]},
        "judge": {"kind": "llm_judge", "model": "fake-judge", "cache_path": str(tmp_path / "j.jsonl")},
        "runtime": {"concurrency": 2, "response_cache": str(tmp_path / "r.jsonl")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    failing = ev.get_strategy("rot13").encode("harmful adv 0")
    empty = ev.get_strategy("base64").encode("harmful jbb 0")
    monkeypatch.setattr(ev, "build_target_client",
                        lambda mc, d, cache=None: FakeClient(mc["id"], fail_on={failing}, empty_on={empty}))
    monkeypatch.setattr(ev, "build_classifier", lambda kind, **kw: FakeJudge())

    run_dir = ev.main(["--config", str(cfg_path), "--no-analysis"])
    df = pd.read_csv(run_dir / "results.csv")

    assert list(df.columns[: len(ev.REQUIRED_COLUMNS)]) == ev.REQUIRED_COLUMNS
    assert len(df) == 2 * 3 * len(make_prompts())
    assert df["model"].tolist()[0] == "fake/a" and df["encoding_type"].tolist()[0] == "plaintext"
    assert (df["classification"] == "GENERATION_ERROR").sum() == 2  # one per model

    summary = json.loads((run_dir / "summary.json").read_text())
    by = {(s["model"], s["encoding"]): s for s in summary["by_model_encoding"]}
    assert by[("fake/a", "base64")]["asr_strict"] == 1.0
    assert by[("fake/a", "plaintext")]["asr_strict"] == 0.0
    assert by[("fake/a", "base64")]["asr_strict_lift_vs_plaintext"] == 1.0
    assert by[("fake/a", "plaintext")]["benign_over_refusal_rate"] == 0.0
    assert summary["n_generation_errors"] == 2
    assert summary["n_no_answer"] == 2  # one per model, not sent to the judge
    assert (df["classification"] == "NO_ANSWER").sum() == 2
    assert by[("fake/a", "base64")]["n_no_answer"] == 1
    assert (run_dir / "config_snapshot.yaml").exists()


def test_response_cache_roundtrip(tmp_path):
    from src.models.openai_compat import OpenAICompatibleClient
    from src.utils.cache import JSONLCache

    calls = []

    class Fake:
        def __init__(self):
            from types import SimpleNamespace as NS
            self.chat = NS(completions=NS(create=self.create))

        def create(self, **kw):
            from types import SimpleNamespace as NS
            calls.append(kw)
            return NS(choices=[NS(message=NS(content=" hi "), finish_reason="stop")],
                      usage=NS(prompt_tokens=3, completion_tokens=1, completion_tokens_details=None))

    cache = JSONLCache(tmp_path / "c.jsonl")
    c = OpenAICompatibleClient("m", "http://x", "NOPE", client=Fake(), cache=cache)
    assert c.generate("q").text == "hi" and not c.generate("q").text == ""
    assert len(calls) == 1
    c2 = OpenAICompatibleClient("m", "http://x", "NOPE", client=Fake(), cache=JSONLCache(tmp_path / "c.jsonl"))
    r = c2.generate("q")
    assert r.cached and r.text == "hi" and len(calls) == 1
