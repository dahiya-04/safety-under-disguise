"""Translate the prompt sample on a Hugging Face GPU job (NLLB, same settings as local).

Why: NLLB on CPU takes ~70 min per language; on a T4 GPU it takes a few minutes.
The job runs exactly the same code (scripts/translate_prompts.py) with the same
model and decoding settings, so its cache entries are interchangeable with
locally produced ones.

Data flow (nothing goes to GitHub):
    submit  : uploads data/processed/encoding_prompts.csv to a PRIVATE HF dataset
              repo (<you>/safety-under-disguise-work), then starts the GPU job.
    worker  : (runs inside the job) clones this GitHub repo, downloads the prompts,
              translates EN -> each language -> EN, uploads results to the same
              private repo under outputs/<tag>/.
    status  : shows job state and the last log lines.
    fetch   : downloads outputs/<tag>/, merges the translations into
              data/cache/translations.json and copies the quality reports to
              data/processed/.

Usage
-----
    python scripts/hf_translate_job.py submit --langs bn,zu
    python scripts/hf_translate_job.py status
    python scripts/hf_translate_job.py fetch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PROMPTS_CSV = ROOT / "data" / "processed" / "encoding_prompts.csv"
CACHE_PATH = ROOT / "data" / "cache" / "translations.json"
PROCESSED = ROOT / "data" / "processed"
STATE_FILE = ROOT / "data" / "cache" / "hf_translate_job.json"
GITHUB_REPO = "https://github.com/dahiya-04/safety-under-disguise.git"
REPO_SUFFIX = "safety-under-disguise-work"
DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
PIP = ("torch==2.15.0 transformers==5.23.0 sentencepiece sacrebleu==2.6.0 pandas==3.1.1 "
       "huggingface_hub==2.4.0 tqdm")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_api():
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (.env)")
    return HfApi(token=token), token


def default_repo(api) -> str:
    return f"{api.whoami()['name']}/{REPO_SUFFIX}"


def merge_translation_cache(local: dict[str, str], remote: dict[str, str]) -> tuple[dict[str, str], int, int]:
    """Add remote entries missing locally. Existing local entries always win.

    Returns (merged, n_added, n_conflicts) where a conflict is a key present in
    both with a different translation (kept as the local value).
    """
    merged = dict(local)
    added = conflicts = 0
    for key, value in remote.items():
        if key not in merged:
            merged[key] = value
            added += 1
        elif merged[key] != value:
            conflicts += 1
    return merged, added, conflicts


# --------------------------------------------------------------------------- #
def cmd_submit(args) -> None:
    api, token = get_api()
    repo = args.data_repo or default_repo(api)
    if not PROMPTS_CSV.exists():
        raise SystemExit(f"{PROMPTS_CSV} missing - run scripts/download_datasets.py first")
    sha = sha256_file(PROMPTS_CSV)

    if not args.dry_run:
        try:
            api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=str(PROMPTS_CSV), path_in_repo="inputs/encoding_prompts.csv",
                            repo_id=repo, repo_type="dataset", commit_message="prompts for translation job")
        except Exception as e:  # most often: fine-grained token without repo write access
            raise SystemExit(f"Could not create/upload to private repo {repo}: {e}\n"
                             "Give HF_TOKEN 'Write access to contents/settings of all repos under your "
                             "personal namespace' at https://huggingface.co/settings/tokens") from None
        print(f"Uploaded prompts to private dataset repo: {repo}  (sha256 {sha[:12]})")

    worker = (f"python scripts/hf_translate_job.py worker --langs {args.langs} --data-repo {repo} "
              f"--tag {args.tag} --sha {sha} --model {args.model} --batch-size {args.batch_size}")
    command = ["bash", "-c",
               f"pip install -q {PIP} && "
               f"git clone --depth 1 --branch {args.branch} {GITHUB_REPO} /app && cd /app && "
               f"nvidia-smi --query-gpu=name,memory.total --format=csv && {worker}"]
    if args.dry_run:
        print("DRY RUN - would submit:\n ", " ".join(command[2:]))
        return
    job = api.run_job(image="python:3.12", command=command, secrets={"HF_TOKEN": token}, flavor=args.flavor,
                      timeout=args.timeout, labels={"name": f"translate-{args.langs.replace(',', '-')}"})
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"job_id": job.id, "url": job.url, "data_repo": repo, "tag": args.tag,
                                      "langs": args.langs, "sha": sha}, indent=2), encoding="utf-8")
    print(f"Job submitted: {job.id}  ({args.flavor})\n  {job.url}\n"
          f"Check:  python scripts/hf_translate_job.py status\n"
          f"Then:   python scripts/hf_translate_job.py fetch")


def cmd_worker(args) -> None:
    """Runs inside the HF job."""
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    from src.encoding.translation import build_translator
    from translate_prompts import print_summary, translate_and_score, write_outputs

    api = HfApi(token=os.environ["HF_TOKEN"])
    csv_path = Path(hf_hub_download(args.data_repo, "inputs/encoding_prompts.csv", repo_type="dataset"))
    if args.sha and sha256_file(csv_path) != args.sha:
        raise SystemExit("prompts CSV hash mismatch - refusing to translate a different sample")

    out_dir = Path("/tmp/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    translator = build_translator("nllb", out_dir / "translations.json", model_name=args.model,
                                  batch_size=args.batch_size, device="cuda")
    meta = {"model": args.model, "batch_size": args.batch_size, "langs": {}, "started": time.time()}
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        out, summary = translate_and_score(df, lang, translator, chunk=20, model=args.model)
        write_outputs(out, summary, out_dir)
        print_summary(summary)
        meta["langs"][lang] = {k: summary[k] for k in ("chrf_mean", "chrf_median", "n_flagged", "elapsed_seconds")}
        (out_dir / "job_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        api.upload_folder(folder_path=str(out_dir), path_in_repo=f"outputs/{args.tag}", repo_id=args.data_repo,
                          repo_type="dataset", commit_message=f"translations {lang}")
        print(f"Uploaded results for {lang}")
    print("DONE")


def _state() -> dict:
    if not STATE_FILE.exists():
        raise SystemExit("No submitted job recorded - run `submit` first (or pass --job-id / --data-repo).")
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def cmd_status(args) -> None:
    api, _ = get_api()
    job_id = args.job_id or _state()["job_id"]
    info = api.inspect_job(job_id=job_id)
    print(f"{job_id}: {info.status.stage}  {info.status.message or ''}\n  {info.url}")
    logs = list(api.fetch_job_logs(job_id=job_id))
    print("--- last log lines ---")
    print("\n".join(str(line) for line in logs[-args.lines:]))


def cmd_fetch(args) -> None:
    from huggingface_hub import snapshot_download

    api, token = get_api()
    state = _state() if STATE_FILE.exists() else {}
    repo = args.data_repo or state.get("data_repo") or default_repo(api)
    tag = args.tag or state.get("tag", "lowres")
    local_dir = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"outputs/{tag}/*"],
                                       token=token)) / "outputs" / tag
    if not local_dir.exists():
        raise SystemExit(f"No outputs/{tag} in {repo} yet - is the job finished? (status)")

    remote = json.loads((local_dir / "translations.json").read_text(encoding="utf-8"))
    local = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    merged, added, conflicts = merge_translation_cache(local, remote)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(CACHE_PATH)
    print(f"Translation cache: +{added} entries ({conflicts} conflicts kept local) -> {CACHE_PATH.relative_to(ROOT)}")

    for f in sorted(local_dir.glob("translation_quality_*")):
        shutil.copy(f, PROCESSED / f.name)
    meta_path = local_dir / "job_meta.json"
    if meta_path.exists():
        for lang, m in json.loads(meta_path.read_text(encoding="utf-8"))["langs"].items():
            print(f"  [{lang}] chrF mean={m['chrf_mean']} median={m['chrf_median']} flagged={m['n_flagged']} "
                  f"({m['elapsed_seconds'] / 60:.1f} min on GPU)")
    print(f"Quality reports -> {PROCESSED.relative_to(ROOT)}/translation_quality_<lang>.csv|json")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="upload prompts + start the GPU job")
    s.add_argument("--langs", default="bn,zu")
    s.add_argument("--flavor", default="t4-small")
    s.add_argument("--timeout", default="1h")
    s.add_argument("--branch", default="main")
    s.add_argument("--tag", default="lowres")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--data-repo", default=None)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_submit)

    w = sub.add_parser("worker", help="(inside the job) translate + upload")
    w.add_argument("--langs", required=True)
    w.add_argument("--data-repo", required=True)
    w.add_argument("--tag", default="lowres")
    w.add_argument("--sha", default=None)
    w.add_argument("--model", default=DEFAULT_MODEL)
    w.add_argument("--batch-size", type=int, default=32)
    w.set_defaults(func=cmd_worker)

    st = sub.add_parser("status", help="job state + last log lines")
    st.add_argument("--job-id", default=None)
    st.add_argument("--lines", type=int, default=25)
    st.set_defaults(func=cmd_status)

    f = sub.add_parser("fetch", help="download results and merge into the local cache")
    f.add_argument("--data-repo", default=None)
    f.add_argument("--tag", default=None)
    f.set_defaults(func=cmd_fetch)

    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.func(args)


if __name__ == "__main__":
    main()
