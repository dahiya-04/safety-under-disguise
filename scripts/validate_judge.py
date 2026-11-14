"""Sanity-check the LLM judge on hand-written cases with known labels.

Runs every case in tests/fixtures/judge_cases.json through the real judge
(no cache, so each run is a fresh measurement) and reports agreement with the
expected label, decoding status, disclaimer, language and specificity.
Use --repeats N to measure judge consistency across identical calls.

Usage
-----
    python scripts/validate_judge.py
    python scripts/validate_judge.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoding.classifier import ClassificationInput, LLMJudgeClassifier  # noqa: E402
from src.encoding.strategies import get_strategy  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "judge_cases.json"
OUT_DIR = ROOT / "results" / "judge_validation"


def encoded_prompt_for(case: dict) -> str:
    if case.get("encoded_prompt"):
        return case["encoded_prompt"]
    return get_strategy(case["encoding_type"]).encode(case["original_prompt"])


def check(case: dict, r) -> dict[str, bool]:
    checks = {
        "label": r.label is not None and r.label.value == case["expected_label"],
        "decoding_status": r.decoding_status is not None
        and r.decoding_status.value == case["expected_decoding_status"],
    }
    if "expected_has_disclaimer" in case:
        checks["disclaimer"] = r.has_disclaimer == case["expected_has_disclaimer"]
    if "expected_language" in case:
        checks["language"] = r.response_language == case["expected_language"]
    if "expected_max_specificity" in case:
        checks["specificity"] = r.specificity is not None and r.specificity <= case["expected_max_specificity"]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=1, help="judge each case N times (consistency check)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    judge = LLMJudgeClassifier(cache_path=None)
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        item = ClassificationInput(
            original_prompt=case["original_prompt"], response=case["response"],
            encoding_type=case["encoding_type"], encoded_prompt=encoded_prompt_for(case),
        )
        for rep in range(args.repeats):
            r = judge.classify(item)
            checks = check(case, r)
            rows.append({
                "id": case["id"], "repeat": rep, "expected": case["expected_label"],
                "got": r.label.value if r.label else f"ERROR({r.error})",
                "decoding": r.decoding_status.value if r.decoding_status else None,
                "disclaimer": r.has_disclaimer, "specificity": r.specificity,
                "lang": r.response_language, "conf": r.confidence,
                "pass": all(checks.values()),
                "failed_checks": ",".join(k for k, ok in checks.items() if not ok),
                "rationale": r.rationale,
            })
            mark = "PASS" if rows[-1]["pass"] else "FAIL"
            print(f"[{mark}] {case['id']:<32} expected={case['expected_label']:<17} "
                  f"got={rows[-1]['got']:<17} dec={rows[-1]['decoding']} spec={r.specificity} "
                  f"conf={r.confidence:.2f} {rows[-1]['failed_checks']}")

    df = pd.DataFrame(rows)
    label_acc = (df["expected"] == df["got"]).mean()
    print(f"\nLabel accuracy: {label_acc:.0%}  |  all-checks pass rate: {df['pass'].mean():.0%}  "
          f"({len(cases)} cases x {args.repeats} repeats)")
    if args.repeats > 1:
        consistent = df.groupby("id")["got"].agg(lambda s: Counter(s).most_common(1)[0][1] == len(s))
        print(f"Consistency (same label on every repeat): {consistent.mean():.0%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "judge_validation.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved -> {out.relative_to(ROOT)}  (see 'rationale' column for failures)")


if __name__ == "__main__":
    main()
