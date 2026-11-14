"""Download raw evaluation datasets and build a fixed, reproducible prompt sample.

Outputs
-------
data/raw/advbench.csv            full AdvBench (520 harmful behaviours)
data/raw/jbb_harmful.csv         JailbreakBench harmful behaviours (100)
data/raw/jbb_benign.csv          JailbreakBench benign behaviours (100)
data/raw/fortress.csv            FORTRESS public set (500 adversarial/benign pairs)
data/processed/encoding_prompts.csv   the frozen sample used by the encoding module
data/processed/sample_metadata.json   seed, counts and provenance for the sample

Usage
-----
    python scripts/download_datasets.py
    python scripts/download_datasets.py --advbench 50 --jbb 30 --fortress 50 --seed 42
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ADVBENCH_HF = "walledai/AdvBench"
ADVBENCH_GITHUB = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
JBB_HF = "JailbreakBench/JBB-Behaviors"
FORTRESS_HF = "ScaleAI/fortress_public"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_advbench(token: str | None) -> tuple[pd.DataFrame, str]:
    """Try the gated HF mirror first, fall back to the authors' original CSV."""
    try:
        df = load_dataset(ADVBENCH_HF, split="train", token=token).to_pandas()
        source = f"hf:{ADVBENCH_HF}"
    except Exception as exc:  # gated / no access yet
        print(f"  ! HF AdvBench unavailable ({type(exc).__name__}); using GitHub original")
        df = pd.read_csv(ADVBENCH_GITHUB).rename(columns={"goal": "prompt"})
        source = ADVBENCH_GITHUB
    return df[["prompt", "target"]], source


def download_jbb() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = load_dataset(JBB_HF, "behaviors")
    return ds["harmful"].to_pandas(), ds["benign"].to_pandas()


def download_fortress(token: str | None) -> pd.DataFrame:
    return load_dataset(FORTRESS_HF, split="train", token=token).to_pandas()


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def dedupe_near_duplicates(prompts: pd.Series, threshold: float = 0.7) -> pd.Series:
    """Greedy near-duplicate removal by token Jaccard similarity.

    AdvBench contains many paraphrases of the same behaviour (e.g. a dozen
    bomb-making variants); keeping them inflates per-topic weight.
    """
    kept_idx, kept_tokens = [], []
    for idx, text in prompts.items():
        toks = _tokens(text)
        if all(len(toks & k) / max(len(toks | k), 1) < threshold for k in kept_tokens):
            kept_idx.append(idx)
            kept_tokens.append(toks)
    return prompts.loc[kept_idx]


_COMMON_WORDS = re.compile(r"\b(the|and|to|of|is|you|that|how|for|with|what|in|a)\b", re.IGNORECASE)
MAX_PROMPT_CHARS = 5000


def detect_obfuscation(text: str) -> str | None:
    """Return 'base64' / 'reversed' if a prompt is ALREADY obfuscated, else None.

    Some FORTRESS prompts ship pre-encoded. Encoding those again would make the
    'plaintext' control not plain and turn e.g. base64 into double-base64.
    """
    s = text.strip()
    # Base64: almost no spaces (plain English has many) and decodes to readable text.
    if len(s) >= 40 and s.count(" ") <= len(s) / 100:
        compact = re.sub(r"\s+", "", s)
        if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", compact):
            try:
                decoded = base64.b64decode(compact, validate=True).decode("utf-8")
                if sum(c.isprintable() or c.isspace() for c in decoded) / max(len(decoded), 1) > 0.9:
                    return "base64"
            except (ValueError, UnicodeDecodeError):
                pass
    # Reversed: common English words appear far more often when the text is reversed.
    fwd, rev = len(_COMMON_WORDS.findall(s)), len(_COMMON_WORDS.findall(s[::-1]))
    if rev > 3 and rev > 2 * fwd:
        return "reversed"
    return None


def fortress_exclusion_reason(row: pd.Series) -> str | None:
    """Why a FORTRESS pair is unusable for the encoding study (None = usable)."""
    for col in ("adversarial_prompt", "benign_prompt"):
        text = str(row[col])
        kind = detect_obfuscation(text)
        if kind:
            return f"{col} already {kind}-encoded"
        if len(text) > MAX_PROMPT_CHARS:
            return f"{col} too long ({len(text)} chars > {MAX_PROMPT_CHARS})"
    return None


def stratified_sample(df: pd.DataFrame, by: str, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample of exactly n rows (largest-remainder rounding)."""
    counts = df[by].value_counts()
    quotas = counts / counts.sum() * n
    alloc = quotas.astype(int)
    remainder = n - alloc.sum()
    for key in (quotas - alloc).sort_values(ascending=False).index[:remainder]:
        alloc[key] += 1
    parts = [
        df[df[by] == key].sample(n=int(k), random_state=seed)
        for key, k in alloc.items()
        if k > 0
    ]
    return pd.concat(parts).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Build the unified sample
# --------------------------------------------------------------------------- #
def build_sample(
    advbench: pd.DataFrame,
    jbb_harmful: pd.DataFrame,
    fortress: pd.DataFrame,
    n_adv: int,
    n_jbb: int,
    n_fortress: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (unified sample, excluded FORTRESS pairs with a 'reason' column)."""
    rows: list[dict] = []

    # AdvBench: dedupe, then random sample (no native categories).
    deduped = dedupe_near_duplicates(advbench["prompt"])
    adv = deduped.sample(n=n_adv, random_state=seed).reset_index(drop=True)
    for i, prompt in enumerate(adv):
        rows.append(dict(
            prompt_id=f"advbench_{i:03d}", prompt=prompt, prompt_type="harmful",
            category="uncategorized", source_dataset="advbench", pair_id=None,
        ))

    # JailbreakBench: stratified over its 10 policy categories.
    jbb = stratified_sample(jbb_harmful, by="Category", n=n_jbb, seed=seed)
    for i, r in jbb.iterrows():
        rows.append(dict(
            prompt_id=f"jbb_{i:03d}", prompt=r["Goal"], prompt_type="harmful",
            category=r["Category"], source_dataset="jailbreakbench", pair_id=None,
        ))

    # FORTRESS: stratified by risk domain; keep BOTH the adversarial prompt and
    # its benign twin so we can measure over-refusal / decoding ability.
    # Pairs that are already obfuscated or extremely long are replaced by another
    # pair from the same risk domain. Kept pairs retain their original ids;
    # replacements get new ids (>= n_fortress) so old results are never mislabelled.
    fort = stratified_sample(fortress, by="risk_domain", n=n_fortress, seed=seed)
    reasons = fort.apply(fortress_exclusion_reason, axis=1)
    excluded = fort[reasons.notna()].assign(reason=reasons[reasons.notna()])
    pool = fortress[~fortress["ID"].isin(fort["ID"])]
    pool = pool[pool.apply(fortress_exclusion_reason, axis=1).isna()]
    replacements = []
    for _, bad in excluded.iterrows():
        candidates = pool[pool["risk_domain"] == bad["risk_domain"]]
        pick = candidates.sample(n=1, random_state=seed)
        replacements.append(pick)
        pool = pool.drop(pick.index)
    kept = fort[reasons.isna()]
    replaced = pd.concat(replacements).reset_index(drop=True) if replacements else fort.iloc[0:0]
    replaced.index = range(n_fortress, n_fortress + len(replaced))
    fort = pd.concat([kept, replaced])
    for i, r in fort.iterrows():
        pair = f"fortress_pair_{i:03d}"
        for kind, col in (("harmful", "adversarial_prompt"), ("benign", "benign_prompt")):
            rows.append(dict(
                prompt_id=f"fortress_{kind}_{i:03d}", prompt=r[col], prompt_type=kind,
                category=r["risk_domain"], source_dataset="fortress", pair_id=pair,
            ))

    return pd.DataFrame(rows), excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--advbench", type=int, default=50)
    parser.add_argument("--jbb", type=int, default=30)
    parser.add_argument("--fortress", type=int, default=50, help="number of FORTRESS pairs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading AdvBench ...")
    advbench, adv_source = download_advbench(token)
    advbench.to_csv(RAW_DIR / "advbench.csv", index=False)

    print("Downloading JailbreakBench ...")
    jbb_harmful, jbb_benign = download_jbb()
    jbb_harmful.to_csv(RAW_DIR / "jbb_harmful.csv", index=False)
    jbb_benign.to_csv(RAW_DIR / "jbb_benign.csv", index=False)

    print("Downloading FORTRESS ...")
    fortress = download_fortress(token)
    fortress.to_csv(RAW_DIR / "fortress.csv", index=False)

    print("Building sample ...")
    sample, excluded = build_sample(advbench, jbb_harmful, fortress,
                                    args.advbench, args.jbb, args.fortress, args.seed)
    out_csv = PROCESSED_DIR / "encoding_prompts.csv"
    sample.to_csv(out_csv, index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "sources": {"advbench": adv_source, "jailbreakbench": f"hf:{JBB_HF}", "fortress": f"hf:{FORTRESS_HF}"},
        "raw_sizes": {
            "advbench": len(advbench),
            "advbench_after_dedup": len(dedupe_near_duplicates(advbench["prompt"])),
            "jbb_harmful": len(jbb_harmful),
            "fortress": len(fortress),
        },
        "sample_counts": sample.groupby(["source_dataset", "prompt_type"]).size()
            .rename("n").reset_index().to_dict(orient="records"),
        "fortress_exclusions": [
            {"fortress_id": str(r["ID"]), "risk_domain": r["risk_domain"], "reason": r["reason"]}
            for _, r in excluded.iterrows()
        ],
        "notes": [
            "AdvBench has no category labels -> 'uncategorized'.",
            f"FORTRESS pairs whose prompt is already obfuscated (base64/reversed) or longer than "
            f"{MAX_PROMPT_CHARS} chars are replaced by a pair from the same risk domain; "
            "replacement pairs get ids >= the requested pair count.",
            "FORTRESS adversarial prompts already contain jailbreak framing; "
            "encoding them is a stacked attack. Benign twins share pair_id.",
        ],
    }
    (PROCESSED_DIR / "sample_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved {len(sample)} prompts -> {out_csv.relative_to(ROOT)}")
    print(sample.groupby(["source_dataset", "prompt_type"]).size().to_string())


if __name__ == "__main__":
    main()
