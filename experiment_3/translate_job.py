"""Experiment 3: translate the direct-request prompts on a Hugging Face GPU job.

Forward only (English -> each language); no back-translation this time.
Same model and decoding settings as Experiments 1-2 (NLLB-200-distilled-600M,
beam 4, batch 8), so cache entries are interchangeable with earlier ones.
Translations that already exist locally (Hindi/Bengali/Zulu for prompts run
before) are kept as they are when merging.

    submit : uploads experiment_3/data/direct_prompts.csv to the PRIVATE HF dataset
             repo (<you>/safety-under-disguise-work) and starts the GPU job
    worker : (inside the job) translates and uploads to outputs/exp3/
    status : job state + progress
    fetch  : merges translations into data/cache/translations.json and writes
             experiment_3/data/translations_<lang>.csv (readable side-by-side)

    python experiment_3/translate_job.py submit
    python experiment_3/translate_job.py status
    python experiment_3/translate_job.py fetch
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hf_translate_job import (  # noqa: E402
    CACHE_PATH, DEFAULT_MODEL, GITHUB_REPO, PIP, default_repo, get_api, merge_translation_cache,
    print_job_status, sha256_file,
)

EXP_DIR = Path(__file__).resolve().parent
PROMPTS_CSV = EXP_DIR / "data" / "direct_prompts.csv"
STATE_FILE = EXP_DIR / "data" / "translate_job.json"
LANGS = "hi,fr,bn,ta,zu"
TAG = "exp3"
REMOTE_INPUT = "inputs/exp3_direct_prompts.csv"


def cmd_submit(args) -> None:
    api, token = get_api()
    repo = args.data_repo or default_repo(api)
    if not PROMPTS_CSV.exists():
        raise SystemExit("Run experiment_3/build_prompts.py first.")
    sha = sha256_file(PROMPTS_CSV)
    worker = (f"python experiment_3/translate_job.py worker --langs {args.langs} --data-repo {repo} "
              f"--sha {sha} --model {args.model} --batch-size {args.batch_size}")
    command = ["bash", "-c",
               f"pip install -q {PIP} && git clone --depth 1 --branch {args.branch} {GITHUB_REPO} /app && "
               f"cd /app && nvidia-smi --query-gpu=name,memory.total --format=csv && {worker}"]
    if args.dry_run:
        print("DRY RUN - would upload the prompts and submit:\n ", command[2])
        return

    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(PROMPTS_CSV), path_in_repo=REMOTE_INPUT, repo_id=repo,
                    repo_type="dataset", commit_message="experiment 3 prompts")
    print(f"Uploaded prompts to private dataset repo: {repo}  (sha256 {sha[:12]})")
    job = api.run_job(image="python:3.12", command=command, secrets={"HF_TOKEN": token}, flavor=args.flavor,
                      env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, timeout=args.timeout,
                      labels={"name": "exp3-translate"})
    STATE_FILE.write_text(json.dumps({"job_id": job.id, "url": job.url, "data_repo": repo, "langs": args.langs,
                                      "sha": sha}, indent=2), encoding="utf-8")
    print(f"Job submitted: {job.id}  ({args.flavor})\n  {job.url}\n"
          f"Check:  python experiment_3/translate_job.py status\n"
          f"Then:   python experiment_3/translate_job.py fetch")


def cmd_worker(args) -> None:
    """Runs inside the HF job."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm import tqdm

    from src.encoding.translation import build_translator

    api = HfApi(token=os.environ["HF_TOKEN"])
    csv_path = Path(hf_hub_download(args.data_repo, REMOTE_INPUT, repo_type="dataset"))
    if sha256_file(csv_path) != args.sha:
        raise SystemExit("prompts CSV hash mismatch - refusing to translate a different sample")
    df = pd.read_csv(csv_path)
    out_dir = Path("/tmp/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    translator = build_translator("nllb", out_dir / "translations.json", model_name=args.model,
                                  batch_size=args.batch_size, device="cuda")
    meta = {"model": args.model, "batch_size": args.batch_size, "direction": "en->xx only", "langs": {}}
    prompts = df["prompt"].tolist()
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        t0, out = time.time(), []
        for i in tqdm(range(0, len(prompts), 20), desc=f"en->{lang}", unit="chunk"):
            out.extend(translator.translate_batch(prompts[i:i + 20], source="en", target=lang))
        df.assign(translation=out)[["prompt_id", "prompt_type", "source_dataset", "prompt", "translation"]].to_csv(
            out_dir / f"translations_{lang}.csv", index=False, encoding="utf-8")
        meta["langs"][lang] = {"n": len(out), "seconds": round(time.time() - t0, 1)}
        (out_dir / "job_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        api.upload_folder(folder_path=str(out_dir), path_in_repo=f"outputs/{TAG}", repo_id=args.data_repo,
                          repo_type="dataset", commit_message=f"exp3 translations {lang}")
        print(f"[{lang}] {len(out)} prompts in {meta['langs'][lang]['seconds'] / 60:.1f} min | e.g. {out[0][:80]}")
        print(f"Uploaded results for {lang}")
    print("DONE")


def _state() -> dict:
    if not STATE_FILE.exists():
        raise SystemExit("No submitted job recorded - run `submit` first.")
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def cmd_status(args) -> None:
    api, _ = get_api()
    print_job_status(api, args.job_id or _state()["job_id"], args.lines)


def cmd_fetch(args) -> None:
    from huggingface_hub import snapshot_download

    api, token = get_api()
    repo = args.data_repo or (_state()["data_repo"] if STATE_FILE.exists() else default_repo(api))
    local_dir = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"outputs/{TAG}/*"],
                                       token=token)) / "outputs" / TAG
    if not (local_dir / "translations.json").exists():
        raise SystemExit(f"No outputs/{TAG} in {repo} yet - is the job finished? (status)")

    remote = json.loads((local_dir / "translations.json").read_text(encoding="utf-8"))
    local = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    merged, added, conflicts = merge_translation_cache(local, remote)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(CACHE_PATH)
    print(f"Translation cache: +{added} new entries, {conflicts} already existed (kept the earlier version)")
    for f in sorted(local_dir.glob("translations_*.csv")):
        shutil.copy(f, EXP_DIR / "data" / f.name)
        print(f"  {f.name} -> experiment_3/data/")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--langs", default=LANGS)
    s.add_argument("--flavor", default="t4-small")
    s.add_argument("--timeout", default="1h")
    s.add_argument("--branch", default="main")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--data-repo", default=None)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_submit)
    w = sub.add_parser("worker")
    w.add_argument("--langs", required=True)
    w.add_argument("--data-repo", required=True)
    w.add_argument("--sha", required=True)
    w.add_argument("--model", default=DEFAULT_MODEL)
    w.add_argument("--batch-size", type=int, default=8)
    w.set_defaults(func=cmd_worker)
    st = sub.add_parser("status")
    st.add_argument("--job-id", default=None)
    st.add_argument("--lines", type=int, default=25)
    st.set_defaults(func=cmd_status)
    f = sub.add_parser("fetch")
    f.add_argument("--data-repo", default=None)
    f.set_defaults(func=cmd_fetch)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.func(args)


if __name__ == "__main__":
    main()
