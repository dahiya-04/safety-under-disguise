# Experiment 3: Direct requests only

**Question:** with no role-play tricks (no FORTRESS), does the *language alone* change whether a model refuses? And does it follow the rule from Experiment 2: **a language breaks safety when the model can read it but its safety training does not cover it?**

## Design

| | |
|---|---|
| Harmful prompts | 200 = all 100 JailbreakBench behaviours + 100 AdvBench (near-duplicates and JBB overlaps removed) |
| Benign prompts | 100 JailbreakBench benign twins, each matched to a harmful behaviour |
| Conditions | English, Hindi, French, Bengali, Tamil, Zulu |
| Models | Llama-3.1-8B, Llama-3.3-70B, Qwen2.5-72B |
| Responses | 300 × 6 × 3 = 5,400 (79 prompts re-used from Experiment 1 via the cache) |
| Translation | NLLB-600M on a Hugging Face GPU job, **forward only** (no back-translation), plus documented fixes (below) |

## Translation fixes

A full manual review of all 300 French translations (`data/translation_review_fr.csv`) found:

| Issue | Prompts | Fix | Languages |
|---|---|---|---|
| Our sentence splitter cut prompts at "U.S." | 6 | Splitter fixed; re-translated with the same NLLB model | all 5 |
| NLLB lost the meaning of jargon/idioms (insider trading, shell companies, keylogger, blackmail…) | 11 (7 harmful, 4 benign) | Corrected by hand (NLLB is deterministic, so re-running repeats the error) | French |

Every change (41 in total) is logged in `data/translation_corrections.csv`. The analysis reports French **with and without** the 11 hand-corrected prompts, because only French got this manual review. The other languages may contain similar jargon errors.

## Predictions (locked before running)

| Language | Llama 3.1 supports it? | Model can read it? | Predicted ASR for Llama-8B |
|---|---|---|---|
| English | ✅ | ✅ | low (baseline) |
| Hindi | ✅ | ✅ | low, close to English |
| French | ✅ | ✅ | low, close to English |
| Bengali | ❌ | ✅ | **high** |
| Tamil | ❌ | ✅ (to check) | **high** |
| Zulu | ❌ | ❌ mostly | low (safe by incompetence) |

| # | Hypothesis |
|---|---|
| H7 | ASR rises most in languages a model reads well but is not officially supported in (Bengali, Tamil) |
| H8 | Officially supported languages (Hindi, French) behave like English |
| H9 | The language gap shrinks with model size (Llama-70B showed no Bengali effect in Experiment 2) |
| H10 | Over-refusal of benign prompts rises in languages a model reads poorly |

If French or Hindi turns out as unsafe as Bengali, the "official support" explanation is wrong. That's what makes this a real test.

## Folder

```
experiment_3/
├── build_prompts.py     builds data/direct_prompts.csv
├── translate_job.py     GPU translation job (submit / status / fetch)
├── fix_translations.py  applies the documented fixes -> data/translations.json
├── config.yaml          experiment config
├── data/                prompts + translations (not on GitHub)
└── results/             runs + analysis
```

## Steps

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\python.exe experiment_3\build_prompts.py                       # done
.venv\Scripts\python.exe experiment_3\translate_job.py submit                # GPU translation
.venv\Scripts\python.exe experiment_3\translate_job.py status
.venv\Scripts\python.exe experiment_3\translate_job.py fetch
.venv\Scripts\python.exe experiment_3\fix_translations.py                    # done
.venv\Scripts\python.exe -m src.encoding.evaluator --config experiment_3\config.yaml --sample-size 30 --run-name pilot
.venv\Scripts\python.exe -m src.encoding.evaluator --config experiment_3\config.yaml --run-name full
```
