"""Pre-translate the encoding prompt sample and measure translation quality.

1. Translates every prompt EN -> target language (cached on disk, so the
   evaluator never has to load the translation model again).
2. Round-trips each translation back to English and scores it against the
   original with chrF (sacrebleu). Low scores flag translations that may have
   lost meaning and should be checked by hand.

Runs locally (CPU, ~70 min per language) or on a Hugging Face GPU job via
scripts/hf_translate_job.py (same code, same model, same decoding settings).

Outputs
-------
data/cache/translations.json                     translation cache (both directions)
data/processed/translation_quality_<lang>.csv    per-prompt translation + round-trip + chrF
data/processed/translation_quality_<lang>.json   summary statistics

Usage
-----
    python scripts/translate_prompts.py                 # Hindi, NLLB-600M
    python scripts/translate_prompts.py --langs bn,zu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src.encoding.translation import Translator, build_translator  # noqa: E402

PROMPTS_CSV = ROOT / "data" / "processed" / "encoding_prompts.csv"
CACHE_PATH = ROOT / "data" / "cache" / "translations.json"
OUT_DIR = ROOT / "data" / "processed"
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


def translate_and_score(df: pd.DataFrame, lang: str, translator: Translator, chunk: int = 10,
                        flag_below: float = 50.0, model: str = DEFAULT_MODEL) -> tuple[pd.DataFrame, dict]:
    """Translate ``df['prompt']`` to ``lang`` and back; score the round trip with chrF."""
    from sacrebleu.metrics import CHRF

    prompts = df["prompt"].tolist()
    t0 = time.time()
    forward: list[str] = []
    backward: list[str] = []
    for i in tqdm(range(0, len(prompts), chunk), desc=f"en->{lang}->en", unit="chunk"):
        part = prompts[i: i + chunk]
        fwd = translator.translate_batch(part, source="en", target=lang)
        forward.extend(fwd)
        backward.extend(translator.translate_batch(fwd, source=lang, target="en"))
    elapsed = time.time() - t0

    chrf = CHRF()
    out = df.copy()
    out[f"prompt_{lang}"] = forward
    out["roundtrip_en"] = backward
    out["chrf"] = [round(chrf.sentence_score(b, [o]).score, 1) for b, o in zip(backward, prompts)]
    out["needs_review"] = out["chrf"] < flag_below
    cols = ["prompt_id", "source_dataset", "prompt_type", "prompt", f"prompt_{lang}", "roundtrip_en", "chrf",
            "needs_review"]
    out = out[[c for c in cols if c in out.columns]]

    group_cols = [c for c in ["source_dataset", "prompt_type"] if c in out.columns]
    by_source = out.groupby(group_cols)["chrf"].agg(["mean", "median", "min", "count"]).round(1)
    summary = {
        "target_language": lang,
        "model": model,
        "n_prompts": len(out),
        "elapsed_seconds": round(elapsed, 1),
        "chrf_mean": round(float(out["chrf"].mean()), 1),
        "chrf_median": round(float(out["chrf"].median()), 1),
        "flag_threshold": flag_below,
        "n_flagged": int(out["needs_review"].sum()),
        "flagged_ids": out.loc[out["needs_review"], "prompt_id"].tolist(),
        "by_source": by_source.reset_index().to_dict(orient="records"),
    }
    return out, summary


def write_outputs(out: pd.DataFrame, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lang = summary["target_language"]
    out.to_csv(out_dir / f"translation_quality_{lang}.csv", index=False, encoding="utf-8")
    (out_dir / f"translation_quality_{lang}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(summary: dict) -> None:
    print(f"\n[{summary['target_language']}] {summary['n_prompts']} prompts in {summary['elapsed_seconds'] / 60:.1f} min"
          f" | chrF mean={summary['chrf_mean']} median={summary['chrf_median']}"
          f" | flagged (<{summary['flag_threshold']}) = {summary['n_flagged']}")
    print(pd.DataFrame(summary["by_source"]).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", "--lang", default="hi", help="comma-separated ISO-639-1 codes (default: hi)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=10, help="prompts per progress step / cache save")
    parser.add_argument("--flag-below", type=float, default=50.0, help="chrF threshold for manual review")
    parser.add_argument("--prompts-csv", type=Path, default=PROMPTS_CSV)
    parser.add_argument("--cache-path", type=Path, default=CACHE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df = pd.read_csv(args.prompts_csv)
    translator = build_translator("nllb", args.cache_path, model_name=args.model, batch_size=args.batch_size,
                                  device=args.device)
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        out, summary = translate_and_score(df, lang, translator, args.chunk, args.flag_below, args.model)
        write_outputs(out, summary, args.out_dir)
        print_summary(summary)


if __name__ == "__main__":
    main()
