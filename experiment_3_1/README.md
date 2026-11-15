# Experiment 3.1: Second-judge check (Indian languages)

**Question:** Experiment 3 found that in Bengali and Tamil the model **stops refusing** (high raw ASR) but what it writes is **vague** (low specificity). Both findings come from one judge, DeepSeek-V4.1-Flash, reading Bengali and Tamil, which had never been validated. **Does an independent, much larger judge agree?**

## Design

| | |
|---|---|
| Responses | From the Experiment 3 full run: **harmful** prompts in **Hindi, Bengali, Tamil** only |
| Sample | 100 per language = 300. Half are DeepSeek "complied / partial" (the contested ones), half are the rest; balanced across the 3 target models; seed 42 |
| Second judge | **Kimi-K2-Instruct** (Moonshot AI, ~1T-parameter MoE): a different family from DeepSeek and from every target model |
| Rubric | Identical: [`JUDGE_PROMPT.txt`](../JUDGE_PROMPT.txt) |
| Cost | ~300 calls, roughly $0.50 |

**Why Kimi-K2:** we tested Gemma-3-27B, Kimi-K2, gpt-oss-120b and GLM-4.5 on a correct and a garbled Tamil answer. Kimi-K2 graded both correctly, is the largest, and is fast. Sarvam-M (India-specialised) was the first choice, but its only provider isn't enabled on our account.

## What we compare

1. **Label agreement and Cohen's κ** per language
2. **"Complied?" agreement**: is the refusal really broken?
3. **Specificity agreement**: are the compliances really vague?
4. **Raw vs substantive ASR by each judge**, re-weighted from the stratified sample to the full Experiment 3 distribution

**If Kimi-K2 agrees with DeepSeek**, the Experiment 3 conclusion ("refusal breaks, but little usable harm leaks") holds with two independent judges. **If it disagrees**, a native speaker reviews `results/disagreements_kimi_k2.csv`.

## Run

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\python.exe experiment_3_1\judge_check.py --dry-run   # sample + cost, no API calls
.venv\Scripts\python.exe experiment_3_1\judge_check.py             # ~5-10 min, resumable
```

Outputs go to `results/`: `report.md`, `summary.json`, `judge_check.csv`, `disagreements_kimi_k2.csv`. The last two contain model responses, so they stay local.
