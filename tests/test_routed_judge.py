"""Tests for RoutedJudge (two endpoints, randomised assignment, daily caps)."""

import json
from types import SimpleNamespace

from src.encoding.classifier import ClassificationInput, RoutedJudge, build_classifier

GOOD = json.dumps({"label": "FULL_REFUSAL", "decoding_status": "correct", "has_disclaimer": False,
                   "specificity": 1, "response_language": "en", "confidence": 0.9, "rationale": "r"})


class FakeClient:
    def __init__(self, fail_with=None):
        self.calls = 0
        self.fail_with = fail_with
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=GOOD), finish_reason="stop")],
                               usage=None)


def make(tmp_path, cap=None, primary_fail=None, share=0.5):
    a, b = FakeClient(primary_fail), FakeClient()
    judge = RoutedJudge(
        judges=[dict(name="goedel", model="judge-a", base_url="x", api_key_env="X", client=a,
                     cache_path=tmp_path / "a.jsonl", daily_call_cap=cap, concurrency=2, max_retries=1),
                dict(name="hf", model="judge-b", base_url="y", api_key_env="Y", client=b,
                     cache_path=tmp_path / "b.jsonl", daily_call_cap=None, concurrency=2)],
        primary_share=share, budget_path=tmp_path / "budget.json")
    return judge, a, b


def item(i):
    return ClassificationInput(original_prompt=f"request {i}", response=f"I can't help with {i}",
                               encoding_type="base64", encoded_prompt=f"enc {i}")


def test_assignment_is_deterministic_and_balanced(tmp_path):
    judge, _, _ = make(tmp_path)
    picks = [judge.assign(item(i)) for i in range(2000)]
    assert picks == [judge.assign(item(i)) for i in range(2000)]  # stable
    share = picks.count(0) / len(picks)
    assert 0.46 < share < 0.54


def test_routes_to_assigned_judge(tmp_path):
    judge, a, b = make(tmp_path)
    results = [judge.classify(item(i)) for i in range(200)]
    expected_a = sum(judge.assign(item(i)) == 0 for i in range(200))
    assert a.calls == expected_a and b.calls == 200 - expected_a
    assert {r.judge_model for r in results} == {"judge-a", "judge-b"}


def test_daily_cap_overflows_to_secondary_and_persists(tmp_path):
    judge, a, b = make(tmp_path, cap=5, share=1.0)  # everything prefers the primary
    for i in range(20):
        judge.classify(item(i))
    assert a.calls == 5 and b.calls == 15
    assert judge.usage_report()["goedel"]["calls_today"] == 5

    judge2, a2, b2 = make(tmp_path, cap=5, share=1.0)  # new process, same day: cap already used
    judge2.classify(item(999))
    assert a2.calls == 0 and b2.calls == 1


def test_rate_limit_falls_back_and_disables_primary(tmp_path, monkeypatch):
    monkeypatch.setattr("src.encoding.classifier.time.sleep", lambda s: None)
    err = type("RateLimitError", (Exception,), {})("429 Too Many Requests")
    judge, a, b = make(tmp_path, primary_fail=err, share=1.0)
    r = judge.classify(item(1))
    assert r.label is not None and r.judge_model == "judge-b"
    judge.classify(item(2))
    assert a.calls == 1  # primary not retried after being rate-limited
    assert judge.usage_report()["goedel"]["rate_limited"] is True


def test_cached_judgement_from_either_judge_is_reused(tmp_path):
    judge, a, b = make(tmp_path)
    for i in range(30):
        judge.classify(item(i))
    calls = (a.calls, b.calls)
    judge2, a2, b2 = make(tmp_path)  # fresh instance reads both caches
    for i in range(30):
        judge2.classify(item(i))
    assert (a2.calls, b2.calls) == (0, 0) and calls[0] + calls[1] == 30


def test_build_classifier_routed(tmp_path):
    judge = build_classifier(
        "routed", budget_path=tmp_path / "b.json", defaults={"max_tokens": 64},
        judges=[dict(name="p", model="m1", base_url="x", api_key_env="X", client=FakeClient(), cache_path=None),
                dict(name="s", model="m2", base_url="y", api_key_env="Y", client=FakeClient(), cache_path=None)])
    assert isinstance(judge, RoutedJudge) and judge.members[0].judge.max_tokens == 64
