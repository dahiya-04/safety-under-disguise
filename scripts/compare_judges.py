"""Compare two judge endpoints on identical inputs (e.g. Goedel vs Hugging Face).

Both judges grade the same items with the same rubric (no cache, fresh calls).
Items = the hand-written validation cases + a stratified sample of real model
responses from earlier runs. Reports agreement, Cohen's kappa, disagreements
and per-call token usage (for quota planning).

Usage
-----
    python scripts/compare_judges.py                       # 5 fixtures + 40 real responses
    python scripts/compare_judges.py --n-real 60
    python scripts/compare_judges.py --runs results/encoding/*_full_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.encoding.classifier import ClassificationInput, LLMJudgeClassifier  # noqa: E402
from src.encoding.strategies import get_strategy  # noqa: E402

JUDGES = {
    "goedel": dict(model="deepseek/deepseek-v4.1-flash",
                   base_url="https://api.goedelmachines.com/v1", api_key_env="GOEDEL_API_KEY"),
    "hf": dict(model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
               base_url="https://router.huggingface.co/v1", api_key_env="HF_TOKEN"),
}
FIXTURES = ROOT / "tests" / "fixtures" / "judge_cases.json"
OUT_DIR = ROOT / "results" / "judge_comparison"
VALID_LABELS = {"FULL_REFUSAL", "PARTIAL_REFUSAL", "COMPLIANCE", "EVASION", "DECODING_FAILURE"}


def cohens_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def load_items(run_globs: list[str], n_real: int, seed: int) -> list[dict]:
    items = []
    for case in json.loads(FIXTURES.read_text(encoding="utf-8")):
        encoded = case.get("encoded_prompt") or get_strategy(case["encoding_type"]).encode(case["original_prompt"])
        items.append(dict(item_id=f"fixture:{case['id']}", source="fixture", original_prompt=case["original_prompt"],
                          response=case["response"], encoding_type=case["encoding_type"], encoded_prompt=encoded,
                          expected=case["expected_label"]))

    paths = [p for g in run_globs for p in glob.glob(str(ROOT / g) if not Path(g).is_absolute() else g)]
    frames = [pd.read_csv(Path(p) / "results.csv") for p in paths if (Path(p) / "results.csv").exists()]
    if frames and n_real > 0:
        df = pd.concat(frames)
        df = df[df["classification"].isin(VALID_LABELS)]
        df = df.drop_duplicates(subset=["model", "encoding_type", "prompt_id"])
        # Stratify by the original judge's label so rare labels are represented.
        per_label = max(1, n_real // max(df["classification"].nunique(), 1))
        sample = (df.groupby("classification", group_keys=False)
                  .apply(lambda g: g.sample(n=min(len(g), per_label), random_state=seed)))
        if len(sample) < n_real:
            rest = df.drop(sample.index, errors="ignore")
            sample = pd.concat([sample, rest.sample(n=min(len(rest), n_real - len(sample)), random_state=seed)])
        for _, r in sample.head(n_real).iterrows():
            items.append(dict(item_id=f"{r['model'].split('/')[-1]}|{r['encoding_type']}|{r['prompt_id']}",
                              source="real", original_prompt=r["original_prompt"], response=r["model_response"],
                              encoding_type=r["encoding_type"], encoded_prompt=r["encoded_prompt"], expected=None))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=["results/encoding/*_full_v1", "results/encoding_lowres/*_full"])
    parser.add_argument("--n-real", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=3, help="per judge (Goedel allows 4 in flight)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(ROOT / ".env")

    items = load_items(args.runs, args.n_real, args.seed)
    print(f"Comparing judges on {len(items)} items "
          f"({sum(i['source'] == 'fixture' for i in items)} fixtures + {sum(i['source'] == 'real' for i in items)} real)")

    results: dict[str, list] = {}
    for name, cfg in JUDGES.items():
        judge = LLMJudgeClassifier(cache_path=None, **cfg)
        inputs = [ClassificationInput(original_prompt=i["original_prompt"], response=i["response"],
                                      encoding_type=i["encoding_type"], encoded_prompt=i["encoded_prompt"])
                  for i in items]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results[name] = list(tqdm(pool.map(judge.classify, inputs), total=len(inputs), desc=name))

    rows = []
    for i, item in enumerate(items):
        row = {"item_id": item["item_id"], "source": item["source"], "encoding": item["encoding_type"],
               "expected": item["expected"]}
        for name in JUDGES:
            r = results[name][i]
            row.update({
                f"{name}_label": r.label.value if r.label else f"ERROR:{r.error}",
                f"{name}_decoding": r.decoding_status.value if r.decoding_status else None,
                f"{name}_specificity": r.specificity, f"{name}_conf": r.confidence,
                f"{name}_in_tok": r.judge_prompt_tokens, f"{name}_cached_tok": r.judge_cached_tokens,
                f"{name}_out_tok": r.judge_completion_tokens, f"{name}_rationale": r.rationale,
            })
        rows.append(row)
    df = pd.DataFrame(rows)

    ok = df[~df["goedel_label"].str.startswith("ERROR") & ~df["hf_label"].str.startswith("ERROR")]
    a, b = ok["goedel_label"].tolist(), ok["hf_label"].tolist()
    print(f"\n=== Agreement ({len(ok)} items both judged; errors: goedel="
          f"{df['goedel_label'].str.startswith('ERROR').sum()}, hf={df['hf_label'].str.startswith('ERROR').sum()}) ===")
    print(f"Label agreement:     {sum(x == y for x, y in zip(a, b)) / max(len(a), 1):.0%}   "
          f"Cohen's kappa = {cohens_kappa(a, b):.2f}")
    dec_ok = ok.dropna(subset=["goedel_decoding", "hf_decoding"])
    print(f"Decoding agreement:  {(dec_ok['goedel_decoding'] == dec_ok['hf_decoding']).mean():.0%}")
    spec = ok.dropna(subset=["goedel_specificity", "hf_specificity"])
    print(f"Specificity within ±1: {((spec['goedel_specificity'] - spec['hf_specificity']).abs() <= 1).mean():.0%}")
    fx = ok[ok["source"] == "fixture"]
    for name in JUDGES:
        print(f"Fixture accuracy [{name}]: {(fx[f'{name}_label'] == fx['expected']).mean():.0%}")
    print("\nConfusion (rows = goedel, cols = hf):")
    print(pd.crosstab(ok["goedel_label"], ok["hf_label"]).to_string())

    dis = ok[ok["goedel_label"] != ok["hf_label"]]
    if len(dis):
        print(f"\nDisagreements ({len(dis)}):")
        for _, r in dis.iterrows():
            print(f"  - {r['item_id']}: goedel={r['goedel_label']} hf={r['hf_label']}")

    print("\n=== Token usage per call (mean) ===")
    for name in JUDGES:
        print(f"  {name:<7} input={df[f'{name}_in_tok'].mean():.0f}  cached={df[f'{name}_cached_tok'].mean():.0f}  "
              f"output={df[f'{name}_out_tok'].mean():.0f}")
    g_in = (df["goedel_in_tok"] - df["goedel_cached_tok"].fillna(0)).mean()
    g_out = df["goedel_out_tok"].mean()
    if g_in > 0 and g_out > 0:
        print(f"  -> Goedel daily capacity ~ {min(1_800_000 / g_in, 450_000 / g_out):.0f} judge calls/day "
              f"(limits: 1.8M new input, 450k output)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "judge_comparison.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSaved -> {out.relative_to(ROOT)}  (see *_rationale columns for disagreements)")


if __name__ == "__main__":
    main()
