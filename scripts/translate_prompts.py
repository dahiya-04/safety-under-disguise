"""Pre-translate the encoding prompt sample and measure translation quality.

1. Translates every prompt EN -> target language (cached on disk, so the
   evaluator never has to load the translation model again).
2. Round-trips each translation back to English and scores it against the
   original with chrF (sacrebleu). Low scores flag translations that may have
   lost meaning and should be checked by hand.

Outputs
-------
data/cache/translations.json                     translation cache (both directions)
data/processed/translation_quality_<lang>.csv    per-prompt translation + round-trip + chrF
data/processed/translation_quality_<lang>.json   summary statistics

Usage
-----
    python scripts/translate_prompts.py                 # Hindi, NLLB-600M
    python scripts/translate_prompts.py --lang zu --flag-below 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import CHRF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src.encoding.translation import build_translator  # noqa: E402

PROMPTS_CSV = ROOT / "data" / "processed" / "encoding_prompts.csv"
CACHE_PATH = ROOT / "data" / "cache" / "translations.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default="hi", help="ISO-639-1 target language (default: hi)")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=10, help="prompts per progress step / cache save")
    parser.add_argument("--flag-below", type=float, default=50.0, help="chrF threshold for manual review")
    args = parser.parse_args()

    df = pd.read_csv(PROMPTS_CSV)
    translator = build_translator("nllb", CACHE_PATH, model_name=args.model, batch_size=args.batch_size)
    prompts = df["prompt"].tolist()

    t0 = time.time()
    forward: list[str] = []
    backward: list[str] = []
    for i in tqdm(range(0, len(prompts), args.chunk), desc=f"en->{args.lang}->en", unit="chunk"):
        chunk = prompts[i : i + args.chunk]
        fwd = translator.translate_batch(chunk, source="en", target=args.lang)
        forward.extend(fwd)
        backward.extend(translator.translate_batch(fwd, source=args.lang, target="en"))
    elapsed = time.time() - t0

    chrf = CHRF()
    df[f"prompt_{args.lang}"] = forward
    df["roundtrip_en"] = backward
    df["chrf"] = [round(chrf.sentence_score(b, [o]).score, 1) for b, o in zip(backward, prompts)]
    df["needs_review"] = df["chrf"] < args.flag_below

    out_dir = ROOT / "data" / "processed"
    cols = ["prompt_id", "source_dataset", "prompt_type", "prompt", f"prompt_{args.lang}",
            "roundtrip_en", "chrf", "needs_review"]
    df[cols].to_csv(out_dir / f"translation_quality_{args.lang}.csv", index=False, encoding="utf-8")

    by_source = df.groupby(["source_dataset", "prompt_type"])["chrf"].agg(["mean", "median", "min", "count"]).round(1)
    summary = {
        "target_language": args.lang,
        "model": args.model,
        "n_prompts": len(df),
        "elapsed_seconds": round(elapsed, 1),
        "chrf_mean": round(df["chrf"].mean(), 1),
        "chrf_median": round(df["chrf"].median(), 1),
        "flag_threshold": args.flag_below,
        "n_flagged": int(df["needs_review"].sum()),
        "flagged_ids": df.loc[df["needs_review"], "prompt_id"].tolist(),
        "by_source": by_source.reset_index().to_dict(orient="records"),
    }
    (out_dir / f"translation_quality_{args.lang}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTranslated {len(df)} prompts in {elapsed / 60:.1f} min")
    print(f"chrF mean={summary['chrf_mean']}  median={summary['chrf_median']}  "
          f"flagged (<{args.flag_below}) = {summary['n_flagged']}")
    print(by_source.to_string())


if __name__ == "__main__":
    main()
