"""Experiment 3: build the direct-request prompt set (no FORTRESS).

    200 harmful = all 100 JailbreakBench behaviours
                + 100 AdvBench prompts (those used in Experiments 1-2, topped up with new
                  ones after near-duplicate removal). 18 JBB behaviours come from
                  AdvBench, so AdvBench prompts overlapping JBB are dropped.
    100 benign  = the JailbreakBench benign twins, matched to the harmful set by Index

Prompts already run in Experiments 1-2 keep their exact text, so their responses
and judgements come from the shared cache at no cost.

Reads data/raw/*.csv (built by scripts/download_datasets.py) and
data/processed/encoding_prompts.csv; writes experiment_3/data/direct_prompts.csv
and experiment_3/data/direct_prompts_meta.json.

    python experiment_3/build_prompts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from download_datasets import _tokens, dedupe_near_duplicates  # noqa: E402

RAW = ROOT / "data" / "raw"
EXP1_PROMPTS = ROOT / "data" / "processed" / "encoding_prompts.csv"
OUT_DIR = Path(__file__).resolve().parent / "data"
SEED = 42
N_ADVBENCH = 100


def build(advbench: pd.DataFrame, jbb_harmful: pd.DataFrame, jbb_benign: pd.DataFrame,
          exp1: pd.DataFrame, n_advbench: int = N_ADVBENCH, seed: int = SEED) -> pd.DataFrame:
    rows: list[dict] = []

    # 18 JBB behaviours were taken from AdvBench. JBB is kept whole, so AdvBench prompts that
    # (nearly) duplicate any JBB goal are dropped (token Jaccard >= 0.7, same rule as dedup).
    jbb_tokens = [_tokens(g) for g in jbb_harmful["Goal"]]

    def overlaps_jbb(text: str) -> bool:
        t = _tokens(text)
        return any(len(t & j) / max(len(t | j), 1) >= 0.7 for j in jbb_tokens)

    # AdvBench: keep the Exp. 1 prompts (same ids) unless they overlap JBB, then top up
    # from the de-duplicated pool.
    exp1_adv = exp1[exp1["source_dataset"] == "advbench"].sort_values("prompt_id")
    exp1_adv = exp1_adv[~exp1_adv["prompt"].map(overlaps_jbb)]
    for _, r in exp1_adv.iterrows():
        rows.append(dict(prompt_id=r["prompt_id"], prompt=r["prompt"], prompt_type="harmful",
                         category="uncategorized", source_dataset="advbench", pair_id=None,
                         in_exp1=True))
    pool = dedupe_near_duplicates(advbench["prompt"])
    pool = pool[~pool.isin(set(exp1["prompt"])) & ~pool.map(overlaps_jbb)]
    extra = pool.sample(n=n_advbench - len(exp1_adv), random_state=seed)
    for i, prompt in enumerate(extra, start=50):  # new ids after Exp. 1's advbench_000-049
        rows.append(dict(prompt_id=f"advbench_{i:03d}", prompt=prompt, prompt_type="harmful",
                         category="uncategorized", source_dataset="advbench", pair_id=None, in_exp1=False))

    # JailbreakBench: all 100 behaviours + their benign twins (matched by Index).
    exp1_jbb = set(exp1.loc[exp1["source_dataset"] == "jailbreakbench", "prompt"])
    benign = jbb_benign.set_index("Index")
    for _, h in jbb_harmful.sort_values("Index").iterrows():
        idx = int(h["Index"])
        pair = f"jbb_pair_{idx:03d}"
        rows.append(dict(prompt_id=f"jbb_harmful_{idx:03d}", prompt=h["Goal"], prompt_type="harmful",
                         category=h["Category"], source_dataset="jailbreakbench", pair_id=pair,
                         in_exp1=h["Goal"] in exp1_jbb))
        b = benign.loc[idx]
        rows.append(dict(prompt_id=f"jbb_benign_{idx:03d}", prompt=b["Goal"], prompt_type="benign",
                         category=b["Category"], source_dataset="jailbreakbench", pair_id=pair, in_exp1=False))

    df = pd.DataFrame(rows)
    assert df["prompt_id"].is_unique and df["prompt"].is_unique
    return df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = build(pd.read_csv(RAW / "advbench.csv"), pd.read_csv(RAW / "jbb_harmful.csv"),
               pd.read_csv(RAW / "jbb_benign.csv"), pd.read_csv(EXP1_PROMPTS))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "direct_prompts.csv", index=False, encoding="utf-8")
    counts = df.groupby(["source_dataset", "prompt_type"]).size()
    meta = {"seed": SEED, "n_prompts": len(df),
            "counts": {f"{s}/{t}": int(n) for (s, t), n in counts.items()},
            "reused_from_exp1": int(df["in_exp1"].sum())}
    (OUT_DIR / "direct_prompts_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(counts.to_string())
    print(f"\n{len(df)} prompts ({meta['reused_from_exp1']} already run in Exp. 1) -> experiment_3/data/direct_prompts.csv")


if __name__ == "__main__":
    main()
